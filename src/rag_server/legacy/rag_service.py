import os
import yaml
import sys
import logging
import httpx
from typing import Any, Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ollama import Client as OllamaClient
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

# --- Configuration des Chemins ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")
from src.utils.custom_embedding import CustomEmbedder

# --- Configuration Environnement ---
MILVUS_URI = os.getenv("MILVUS_URI")
COLL = os.getenv("MILVUS_COLLECTION")
OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
RERANK_URL = os.getenv("RERANK_URL")
RERANK_HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=5.0)
HF_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_service")

current_dir = os.path.dirname(os.path.abspath(__file__))
PROMPTS_PATH = os.path.join(current_dir, "prompt.yaml")

# =========================
# Modèles de données API
# =========================
class ChatMessage(BaseModel):
    role: str  # "user" ou "assistant"
    content: str

class AskRequest(BaseModel):
    query: str
    chat_history: List[ChatMessage] = []
    top_k_final: int = 5
    top_k_recall: int = 60

class ContextItem(BaseModel):
    id: Any
    text: str
    source: Optional[str] = None
    page_no: Optional[int] = None
    section_title: Optional[str] = None
    rerank_score: Optional[float] = None

class AskResponse(BaseModel):
    answer: str
    rewritten_query: str
    contexts: List[ContextItem]

# =========================
# Chargeur de Prompts YAML
# =========================
class YamlPromptLoader:
    def __init__(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)

    def get_template(self, name: str) -> ChatPromptTemplate:
        data = self.prompts.get(name)
        if not data:
            raise ValueError(f"Prompt template '{name}' non trouvé")
        messages = []
        if data.get("system") and data["system"].strip():
            sys_msg = data["system"].replace("/no_think", "").strip()
            messages.append(("system", sys_msg))
        if data.get("human"):
            messages.append(("human", data["human"]))
        return ChatPromptTemplate.from_messages(messages)

# =========================
# Core Service Class
# =========================
class RagService:
    def __init__(self):
        self.milvus = MilvusClient(MILVUS_URI)
        self.prompt_loader = YamlPromptLoader(PROMPTS_PATH)
        self.embedder = CustomEmbedder(HF_EMBEDDING_MODEL) 
        
        # Initialisation du LLM principal (Température 0)
        self.llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.0)

        # --- Chaînes LCEL NVIDIA Style ---
        # 1. Reformulation & Décomposition
        self.rewriter_chain = self.prompt_loader.get_template("query_rewriter_prompt") | self.llm | StrOutputParser()
        self.decomposition_chain = self.prompt_loader.get_template("query_decomposition_multiquery_prompt") | self.llm | StrOutputParser()

        # 2. RAG & Génération
        self.rag_chain = self.prompt_loader.get_template("rag_template") | self.llm | StrOutputParser()

        # 3. Réflexion & Garde-fous
        self.relevance_chain = self.prompt_loader.get_template("reflection_relevance_check_prompt") | self.llm | StrOutputParser()
        self.groundedness_chain = self.prompt_loader.get_template("reflection_groundedness_check_prompt") | self.llm | StrOutputParser()
        self.regeneration_chain = self.prompt_loader.get_template("reflection_response_regeneration_prompt") | self.llm | StrOutputParser()

    def get_embeddings(self, text: str) -> List[float]:
        return self.embedder(text)

    def hybrid_retrieval(self, query: str, top_k: int) -> List[Dict]:
        qvec = self.get_embeddings(query)
        dense_req = AnnSearchRequest(data=[qvec], anns_field="vector", param={"metric_type": "COSINE", "params": {"ef": 100}}, limit=top_k)
        sparse_req = AnnSearchRequest(data=[query], anns_field="sparse", param={"metric_type": "BM25"}, limit=top_k)
        res = self.milvus.hybrid_search(COLL, [sparse_req, dense_req], RRFRanker(k=60), limit=top_k, output_fields=["id", "text", "source", "page_no", "section_title"])
        return res[0] if res else []

    def remote_rerank(self, query: str, hits: List[Any], top_k: int, threshold: float = 0.35) -> List[Dict]:
        if not hits: return []
        passages, meta = [], []
        for h in hits:
            ent = h.get('entity')
            text, title = ent.get("text", ""), ent.get("section_title", "")
            passages.append(f"{title}\n\n{text}" if title else text)
            meta.append({"id": ent.get("id"), "text": text, "source": ent.get("source"), "page_no": ent.get("page_no"), "section_title": title})
        
        try:
            with httpx.Client(timeout=RERANK_HTTP_TIMEOUT) as client:
                r = client.post(RERANK_URL, json={"query": query, "passages": passages})
                r.raise_for_status()
                scores = r.json()["scores"]

            logger.info(f"DEBUG SCORES : {scores}") # Regarde les logs console
            
            for m, s in zip(meta, scores): 
                m["rerank_score"] = float(s)
            
            # FILTRAGE SUR SCORES BRUTS :
            # -2 à +10 : Très pertinent
            # -5 à -2  : Moyennement pertinent (utile pour le contexte)
            # < -7     : Hors sujet
            threshold = -8.0 
            filtered = [m for m in meta if m["rerank_score"] >= threshold]
            
            filtered.sort(key=lambda x: x["rerank_score"], reverse=True)
            return filtered[:top_k]
        except Exception as e:
            logger.warning(f"Rerank failed: {e}")
            return meta[:top_k]

    def format_context(self, contexts: List[Dict]) -> str:
        if not contexts: return ""
        return "\n\n".join([f"DOC {i+1}: {c['text']}" for i, c in enumerate(contexts)])

    async def run_pipeline(self, req: AskRequest):
        # 1. Historique & Reformulation
        last_messages = req.chat_history[-6:]
        history_str = "\n".join([f"{'User' if m.role=='user' else 'Assistant'}: {m.content}" for m in last_messages]) or "Pas d'historique."
        
        #rewritten_query = self.rewriter_chain.invoke({"chat_history": history_str, "input": req.query}).strip().split('\n')[0]
        # Dans run_pipeline, remplace le split actuel par :
        rewritten_query = self.rewriter_chain.invoke({"chat_history": history_str, "input": req.query})
        # Nettoyage : on enlève les préfixes courants que le LLM ajoute parfois
        rewritten_query = rewritten_query.replace("Requête reformulée :", "").replace("Question Réécrite :", "").strip().split('\n')[0]

        # 2. Multi-Query (Décomposition) pour augmenter le Recall
        sub_queries_raw = self.decomposition_chain.invoke({"question": rewritten_query})
        sub_queries = [line.strip() for line in sub_queries_raw.split('\n') if line.strip() and any(c.isdigit() for c in line[:2])]
        if not sub_queries: sub_queries = [rewritten_query]

        # 3. Retrieval Hybride Global
        all_hits = []
        for sq in sub_queries:
            all_hits.extend(self.hybrid_retrieval(sq, top_k=req.top_k_recall))
        
        # Déduplication par ID
        seen_ids, unique_hits = set(), []
        for h in all_hits:
            if h['id'] not in seen_ids:
                unique_hits.append(h)
                seen_ids.add(h['id'])

        # 4. Reranking avec Seuil de Confiance
        reranked_contexts = self.remote_rerank(rewritten_query, unique_hits, top_k=req.top_k_final)

        # 5. Reflection: Relevance Check (Contexte suffisant ?)
        context_str = self.format_context(reranked_contexts)
        if reranked_contexts:
            rel_score = self.relevance_chain.invoke({"query": rewritten_query, "context": context_str})
            if "0" in rel_score:
                logger.info("⚠️ Relevance 0 : Contexte jugé inutile.")
                return {"answer": "Désolé, les documents disponibles ne permettent pas de répondre à votre question.", 
                        "rewritten_query": rewritten_query, "contexts": []}

        # 6. Génération RAG
        initial_answer = self.rag_chain.invoke({
            "context": context_str or "Aucun contexte pertinent.",
            "chat_history": history_str,
            "question": req.query
        })

        # 7. Reflection: Groundedness Check (Vérification Hallucination)
        ground_score = self.groundedness_chain.invoke({"context": context_str, "response": initial_answer})
        
        final_answer = initial_answer
        if "0" in ground_score:
            logger.warning("🚨 Hallucination détectée. Régénération en cours...")
            final_answer = self.regeneration_chain.invoke({"context": context_str, "query": rewritten_query})

        return {
            "answer": final_answer,
            "rewritten_query": rewritten_query,
            "contexts": reranked_contexts
        }

# =========================
# FastAPI App
# =========================
app = FastAPI()
service: RagService = None

@app.on_event("startup")
def startup():
    global service
    service = RagService()

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        result = await service.run_pipeline(req)
        return AskResponse(
            answer=result["answer"],
            rewritten_query=result["rewritten_query"],
            contexts=[ContextItem(**c) for c in result["contexts"]]
        )
    except Exception as e:
        logger.error(f"Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# import os
# import yaml
# import sys
# import logging
# import httpx
# from typing import Any, Dict, List, Optional
# from pathlib import Path
# from dotenv import load_dotenv

# # =========================================================
# # 2. IMPORTS DES MODULES (APRÈS AJOUT DU PATH)
# # =========================================================
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from ollama import Client as OllamaClient
# from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

# from pathlib import Path
# ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# if str(ROOT_DIR) not in sys.path:
#     sys.path.append(str(ROOT_DIR))

# load_dotenv(ROOT_DIR / ".env")
# from src.utils.custom_embedding import CustomEmbedder


# # =========================
# # 3. CONFIGURATION VIA .ENV
# # =========================
# MILVUS_URI = os.getenv("MILVUS_URI") #, "http://localhost:19530")
# COLL = os.getenv("MILVUS_COLLECTION") #, "rag_minist_int_hybrid_custom_embedding_infloat")
# OLLAMA_URL = os.getenv("OLLAMA_URL") #, "http://localhost:11434")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") #, "llama3.2:3b")

# RERANK_URL = os.getenv("RERANK_URL") #, "http://localhost:8001/rerank")
# RERANK_HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=5.0)
# HF_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME") #, "intfloat/multilingual-e5-base")
# #PROMPTS_PATH = "prompt.yaml"

# # Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("rag_service")

# current_dir = os.path.dirname(os.path.abspath(__file__))
# PROMPTS_PATH = os.path.join(current_dir, "prompt.yaml")

# logger.info(f"Chargement des prompts depuis : {PROMPTS_PATH}")

# # =========================
# # Modèles de données API
# # =========================
# class ChatMessage(BaseModel):
#     role: str  # "user" ou "assistant"
#     content: str

# class AskRequest(BaseModel):
#     query: str
#     chat_history: List[ChatMessage] = []
#     top_k_final: int = 5
#     top_k_recall: int = 60

# class ContextItem(BaseModel):
#     id: Any
#     text: str
#     source: Optional[str] = None
#     page_no: Optional[int] = None
#     section_title: Optional[str] = None
#     rerank_score: Optional[float] = None

# class AskResponse(BaseModel):
#     answer: str
#     rewritten_query: str
#     contexts: List[ContextItem]

# # =========================
# # Chargeur de Prompts YAML
# # =========================
# class YamlPromptLoader:
#     def __init__(self, path: str):
#         with open(path, 'r', encoding='utf-8') as f:
#             self.prompts = yaml.safe_load(f)

#     def get_template(self, name: str) -> ChatPromptTemplate:
#         data = self.prompts.get(name)
#         if not data:
#             raise ValueError(f"Prompt template '{name}' non trouvé")
        
#         messages = []
#         if data.get("system") and data["system"].strip():
#             # Nettoyage des balises de contrôle LLM si présentes
#             sys_msg = data["system"].replace("/no_think", "").strip()
#             messages.append(("system", sys_msg))
        
#         if data.get("human"):
#             messages.append(("human", data["human"]))
        
#         return ChatPromptTemplate.from_messages(messages)

# # =========================
# # Core Service Class
# # =========================
# class RagService:
#     def __init__(self):
#         self.milvus = MilvusClient(MILVUS_URI)
#         self.ollama_client = OllamaClient(host=OLLAMA_URL)
#         self.prompt_loader = YamlPromptLoader(PROMPTS_PATH)
#         self.embedder = CustomEmbedder(HF_EMBEDDING_MODEL) 
        
#         # Initialisation du LLM (Température 0 pour la précision)
#         self.llm = ChatOllama(
#             model=OLLAMA_MODEL,
#             base_url=OLLAMA_URL,
#             temperature=0.0,
#         )

#         # Chaîne LCEL 1: Reformulation (Rewriting)
#         self.rewriter_chain = (
#             self.prompt_loader.get_template("query_rewriter_prompt")
#             | self.llm
#             | StrOutputParser()
#         )

#         # Chaîne LCEL 2: Génération Finale (RAG)
#         self.rag_chain = (
#             self.prompt_loader.get_template("rag_template")
#             | self.llm
#             | StrOutputParser()
#         )

#     def get_embeddings(self, text: str) -> List[float]:
#         #res = self.ollama_client.embeddings(model=EMB_MODEL, prompt=text)
#         #return res["embedding"]
#         return self.embedder(text)

#     def hybrid_retrieval(self, query: str, top_k: int) -> List[Dict]:
#         qvec = self.get_embeddings(query)
#         dense_req = AnnSearchRequest(
#             data=[qvec], anns_field="vector",
#             param={"metric_type": "COSINE", "params": {"ef": 100}},
#             limit=top_k
#         )
#         sparse_req = AnnSearchRequest(
#             data=[query], anns_field="sparse",
#             param={"metric_type": "BM25"},
#             limit=top_k
#         )
#         res = self.milvus.hybrid_search(
#             COLL, [sparse_req, dense_req],
#             RRFRanker(k=60), limit=top_k,
#             output_fields=["id", "text", "source", "page_no", "section_title"]
#         )
#         return res[0] if res else []

#     def remote_rerank(self, query: str, hits: List[Any], top_k: int) -> List[Dict]:
#         if not hits: return []
#         passages, meta = [], []
#         for h in hits:
#             ent = h.get('entity')
#             text, title = ent.get("text", ""), ent.get("section_title", "")
#             passages.append(f"{title}\n\n{text}" if title else text)
#             meta.append({
#                 "id": ent.get("id"), "text": text, "source": ent.get("source"),
#                 "page_no": ent.get("page_no"), "section_title": title
#             })
#         try:
#             with httpx.Client(timeout=RERANK_HTTP_TIMEOUT) as client:
#                 r = client.post(RERANK_URL, json={"query": query, "passages": passages})
#                 r.raise_for_status()
#                 scores = r.json()["scores"]
#             for m, s in zip(meta, scores): m["rerank_score"] = float(s)
#             meta.sort(key=lambda x: x["rerank_score"], reverse=True)
#         except Exception as e:
#             logger.warning(f"Rerank failed: {e}")
#         return meta[:top_k]

#     def format_context(self, contexts: List[Dict]) -> str:
#         if not contexts: return "Aucun document pertinent trouvé."
#         return "\n\n".join([f"DOC {i+1}: {c['text']}" for i, c in enumerate(contexts)])

#     # async def run_pipeline(self, req: AskRequest):
#     #     # 1. Limitation de l'historique aux 5 DERNIERS MESSAGES
#     #     last_messages = req.chat_history[-5:]
        
#     #     history_str = ""
#     #     for m in last_messages:
#     #         role = "Utilisateur" if m.role == "user" else "Assistant"
#     #         history_str += f"{role}: {m.content}\n"
        
#     #     if not history_str:
#     #         history_str = "(Aucun échange précédent)"

#     #     # 2. Reformulation (Rewriting)
#     #     # On passe history_str au rewriter pour qu'il gère les pronoms (il, ce projet, etc.)
#     #     rewritten_query = self.rewriter_chain.invoke({
#     #         "chat_history": history_str,
#     #         "input": req.query
#     #     })
#     #     # Nettoyage pour ne garder que la première ligne reformulée
#     #     rewritten_query = rewritten_query.strip().split('\n')[0]

#     #     # 3. Retrieval + Reranking (avec la query reformulée)
#     #     hits = self.hybrid_retrieval(rewritten_query, top_k=req.top_k_recall)
#     #     reranked_contexts = self.remote_rerank(rewritten_query, hits, top_k=req.top_k_final)

#     #     # 4. Génération de la réponse Finale (RAG)
#     #     context_str = self.format_context(reranked_contexts)
        
#     #     # Passage de l'historique réduit à la chaîne RAG pour la mémoire
#     #     answer = self.rag_chain.invoke({
#     #         "context": context_str,
#     #         "chat_history": history_str,
#     #         "question": req.query
#     #     })

#     #     return {
#     #         "answer": answer,
#     #         "rewritten_query": rewritten_query,
#     #         "contexts": reranked_contexts
#     #     }

#     async def run_pipeline(self, req: AskRequest):
#         """Pipeline RAG complet avec Router, Rewriter et Reranker distant."""
        
#         # 1. Préparation de l'historique structuré (fenêtre coulissante de 6 messages)
#         # On utilise des préfixes clairs pour que le LLM distingue les rôles
#         last_messages = req.chat_history[-6:]
#         history_str = ""
#         for m in last_messages:
#             prefix = "👤 Utilisateur" if m.role == "user" else "🤖 Assistant"
#             history_str += f"{prefix}: {m.content}\n"
        
#         if not history_str:
#             history_str = "Début de la conversation."

#         # 2. Router : Est-ce une question technique ou une simple politesse ?
#         # Si la requête est très courte (ex: "Bonjour", "Merci"), on peut bypasser le RAG
#         is_greeting = any(word in req.query.lower() for word in ["bonjour", "salut", "merci", "ça va"])
        
#         if is_greeting and len(req.query) < 15:
#             answer = self.llm.invoke(f"Réponds poliment à : {req.query}").content
#             return {
#                 "answer": answer,
#                 "rewritten_query": req.query,
#                 "contexts": []
#             }

#         # 3. Reformulation (Rewriting)
#         # On force le rewriter à produire une requête de recherche optimisée
#         try:
#             rewritten_query = self.rewriter_chain.invoke({
#                 "chat_history": history_str,
#                 "input": req.query
#             })
#             # Nettoyage : on prend la première ligne et on enlève les guillemets éventuels
#             rewritten_query = rewritten_query.strip().split('\n')[0].replace('"', '')
#         except Exception as e:
#             logger.warning(f"Rewriting failed, using original query: {e}")
#             rewritten_query = req.query

#         # 4. Retrieval Hybride (Milvus : Dense + Sparse)
#         # On récupère top_k_recall (ex: 60) pour donner du choix au Reranker
#         hits = self.hybrid_retrieval(rewritten_query, top_k=req.top_k_recall)

#         # 5. Reranking Distant (BGE Reranker)
#         # C'est ici que la Context Precision va s'améliorer
#         reranked_contexts = self.remote_rerank(
#             query=rewritten_query, 
#             hits=hits, 
#             top_k=req.top_k_final
#         )

#         # 6. Génération de la réponse finale avec le contexte filtré
#         context_str = self.format_context(reranked_contexts)
        
#         # On passe la question ORIGINALE au RAG pour garder l'intention de l'utilisateur
#         # Mais on utilise le CONTEXTE trouvé grâce à la requête REFORMULÉE
#         answer = self.rag_chain.invoke({
#             "context": context_str,
#             "chat_history": history_str,
#             "question": req.query
#         })

#         # On retourne tout pour l'évaluation RAGAS
#         return {
#             "answer": answer,
#             "rewritten_query": rewritten_query,
#             "contexts": reranked_contexts
#         }

    

# # =========================
# # FastAPI App
# # =========================
# app = FastAPI()
# service: RagService = None

# @app.on_event("startup")
# def startup():
#     global service
#     service = RagService()

# @app.post("/ask", response_model=AskResponse)
# async def ask(req: AskRequest):
#     try:
#         result = await service.run_pipeline(req)
#         return AskResponse(
#             answer=result["answer"],
#             rewritten_query=result["rewritten_query"],
#             contexts=[ContextItem(**c) for c in result["contexts"]]
#         )
#     except Exception as e:
#         logger.error(f"Pipeline Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8004)