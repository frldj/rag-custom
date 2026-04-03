# import os
# import yaml
# import sys
# import logging
# import httpx
# import uuid
# import time
# from typing import Any, Dict, List, Optional
# from pathlib import Path
# from dotenv import load_dotenv

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# # Fix pour macOS OpenMP
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # =========================================================
# # 1. CONFIGURATION DES CHEMINS
# # =========================================================
# current_file = Path(__file__).resolve()
# ROOT_DIR = current_file.parent.parent.parent
# load_dotenv(ROOT_DIR / ".env")

# if str(ROOT_DIR) not in sys.path:
#     sys.path.insert(0, str(ROOT_DIR))

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("rag_service")

# # =========================================================
# # 2. INTERFACE LANGFUSE API (AVEC MÉTRIQUES)
# # =========================================================
# LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
# LF_PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY")
# LF_SECRET = os.getenv("LANGFUSE_SECRET_KEY")

# def log_to_langfuse(name: str, input_data: Any, output_data: Any, rel_score: int, metrics: dict):
#     """Envoie une trace et un score de pertinence à Langfuse"""
#     if not LF_PUBLIC or not LF_SECRET: return
    
#     trace_id = str(uuid.uuid4())
#     url = f"{LANGFUSE_HOST}/api/public/ingestion"
#     ts = time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())
    
#     payload = {
#         "batch": [
#             {
#                 "id": str(uuid.uuid4()),
#                 "type": "trace-create",
#                 "timestamp": ts,
#                 "body": {
#                     "id": trace_id,
#                     "name": name,
#                     "input": input_data,
#                     "output": output_data,
#                     "metadata": metrics
#                 }
#             },
#             {
#                 "id": str(uuid.uuid4()),
#                 "type": "score-create",
#                 "timestamp": ts,
#                 "body": {
#                     "traceId": trace_id,
#                     "name": "context_relevance",
#                     "value": float(rel_score),
#                     "comment": "Auto-eval via LLM reflection"
#                 }
#             }
#         ]
#     }
    
#     try:
#         httpx.post(url, json=payload, auth=(LF_PUBLIC, LF_SECRET), timeout=2.0)
#     except Exception as e:
#         logger.warning(f"Langfuse Logging Error: {e}")

# # =========================================================
# # 3. IMPORTS COMPOSANTS RAG
# # =========================================================
# try:
#     from src.utils.custom_embedding import CustomEmbedder
#     from langchain_ollama import ChatOllama
#     from langchain_core.prompts import ChatPromptTemplate
#     from langchain_core.output_parsers import StrOutputParser
#     from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
# except ImportError as e:
#     logger.error(f"Erreur d'import des modules internes : {e}")
#     raise

# # =========================================================
# # 4. MODÈLES DE DONNÉES
# # =========================================================
# class ChatMessage(BaseModel):
#     role: str
#     content: str

# class AskRequest(BaseModel):
#     query: str
#     chat_history: List[ChatMessage] = []
#     top_k_final: int = 5
#     top_k_recall: int = 60

# class ContextItem(BaseModel):
#     id: Any
#     text: str
#     rerank_score: Optional[float] = None

# class AskResponse(BaseModel):
#     answer: str
#     rewritten_query: str
#     contexts: List[ContextItem]
#     run_id: str 

# # =========================================================
# # 5. LOGIQUE RAG
# # =========================================================
# class RagService:
#     def __init__(self):
#         self.milvus = MilvusClient(os.getenv("MILVUS_URI"))
#         self.embedder = CustomEmbedder(os.getenv("EMBEDDING_MODEL_NAME")) 
#         self.model_name = os.getenv("OLLAMA_MODEL")
#         self.llm = ChatOllama(
#             model=self.model_name, 
#             base_url=os.getenv("OLLAMA_URL"), 
#             temperature=0.0
#         )
        
#         from pathlib import Path
#         prompt_path = Path(__file__).parent / "prompt.yaml"
#         with open(prompt_path, 'r', encoding='utf-8') as f:
#             self.prompts = yaml.safe_load(f)

#         self.rewriter_chain = self.get_template("query_rewriter_prompt") | self.llm | StrOutputParser()
#         self.rag_chain = self.get_template("rag_template") | self.llm | StrOutputParser()
#         self.relevance_eval = self.get_template("reflection_relevance_check_prompt") | self.llm | StrOutputParser()

#     def get_template(self, name: str) -> ChatPromptTemplate:
#         data = self.prompts.get(name)
#         msgs = [("system", data["system"].replace("/no_think", "").strip()), ("human", data["human"])]
#         return ChatPromptTemplate.from_messages(msgs)

#     def hybrid_retrieval(self, query: str, top_k: int) -> List[Dict]:
#         qvec = self.embedder(query)
#         dense_req = AnnSearchRequest(data=[qvec], anns_field="vector", param={"metric_type": "COSINE"}, limit=top_k)
#         sparse_req = AnnSearchRequest(data=[query], anns_field="sparse", param={"metric_type": "BM25"}, limit=top_k)
#         res = self.milvus.hybrid_search(os.getenv("MILVUS_COLLECTION"), [sparse_req, dense_req], RRFRanker(k=60), limit=top_k, output_fields=["id", "text"])
#         return res[0] if res else []

#     def remote_rerank(self, query: str, hits: List[Any], top_k: int) -> List[Dict]:
#         if not hits: return []
#         passages = [h.get('entity', h).get("text") for h in hits]
#         meta = [{"id": h.get('entity', h).get("id"), "text": h.get('entity', h).get("text")} for h in hits]
#         try:
#             r = httpx.post(os.getenv("RERANK_URL"), json={"query": query, "passages": passages}, timeout=60.0)
#             scores = r.json()["scores"]
#             for m, s in zip(meta, scores): m["rerank_score"] = float(s)
#             return sorted([m for m in meta if m["rerank_score"] >= -8.0], key=lambda x: x["rerank_score"], reverse=True)[:top_k]
#         except Exception: return meta[:top_k]

#     async def run_pipeline(self, req: AskRequest):
#         t_start = time.time()
#         history_str = "\n".join([f"{m.role}: {m.content}" for m in req.chat_history[-5:]]) or "Aucun historique."

#         # 1. Reformulation (Latence T1)
#         t1 = time.time()
#         rewritten = self.rewriter_chain.invoke({"chat_history": history_str, "input": req.query}).strip().split('\n')[0]
#         lat_rewrite = round(time.time() - t1, 3)

#         # 2. Retrieval & Rerank (Latence T2)
#         t2 = time.time()
#         raw_hits = self.hybrid_retrieval(rewritten, top_k=req.top_k_recall)
#         contexts = self.remote_rerank(rewritten, raw_hits, top_k=req.top_k_final)
#         lat_retrieval = round(time.time() - t2, 3)
        
#         ctx_str = "\n\n".join([f"DOC {i+1}: {c['text']}" for i, c in enumerate(contexts)])

#         # 3. Évaluation Pertinence (Latence T3)
#         t3 = time.time()
#         rel_score = 0
#         if contexts:
#             rel_raw = self.relevance_eval.invoke({"query": rewritten, "context": ctx_str})
#             rel_score = int(''.join(filter(str.isdigit, rel_raw)) or "0")
#         lat_eval = round(time.time() - t3, 3)

#         # 4. Génération (Latence T4)
#         t4 = time.time()
#         if rel_score == 0 and contexts:
#             answer = "Désolé, les documents trouvés ne me permettent pas de répondre avec certitude."
#         else:
#             answer = self.rag_chain.invoke({
#                 "context": ctx_str or "Aucun contexte trouvé.",
#                 "chat_history": history_str,
#                 "question": req.query
#             })
#         lat_gen = round(time.time() - t4, 3)

#         total_latency = round(time.time() - t_start, 3)

#         # 5. LOG VERS LANGFUSE (MÉTRIQUES)
#         metrics = {
#             "model": self.model_name,
#             "total_latency": total_latency,
#             "lat_rewrite": lat_rewrite,
#             "lat_retrieval": lat_retrieval,
#             "lat_eval": lat_eval,
#             "lat_gen": lat_gen,
#             "docs_retrieved": len(contexts),
#             "query_length": len(req.query)
#         }
        
#         log_to_langfuse(
#             name="RAG_Full_Chain",
#             input_data={"query": req.query, "rewritten": rewritten},
#             output_data={"answer": answer},
#             rel_score=rel_score,
#             metrics=metrics
#         )

#         return {
#             "answer": answer, 
#             "rewritten_query": rewritten, 
#             "contexts": contexts,
#             "run_id": str(uuid.uuid4())
#         }

# # =========================================================
# # 6. FASTAPI APP
# # =========================================================
# app = FastAPI(title="RAG Service avec Métriques Langfuse")
# service = None

# @app.on_event("startup")
# def startup():
#     global service
#     service = RagService()

# @app.post("/ask", response_model=AskResponse)
# async def ask(req: AskRequest):
#     try:
#         return await service.run_pipeline(req)
#     except Exception as e:
#         logger.error(f"Pipeline Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8004)

import os
import yaml
import sys
import logging
import httpx
import uuid
import time
from typing import Any, Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

# --- Config ---
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
load_dotenv(ROOT_DIR / ".env")
if str(ROOT_DIR) not in sys.path: sys.path.insert(0, str(ROOT_DIR))

from src.utils.custom_embedding import CustomEmbedder
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_service")

# =========================================================
# 1. LOGIQUE DE LOGGING (MÉTHODE GLOBALE QUI MARCHE)
# =========================================================
def log_to_langfuse(name: str, input_data: Any, output_data: Any, rel_score: int, metrics: dict):
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    
    if not public_key or not secret_key: return
    
    trace_id = str(uuid.uuid4())
    ts = time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())
    
    payload = {
        "batch": [
            {
                "id": str(uuid.uuid4()),
                "type": "trace-create",
                "timestamp": ts,
                "body": {
                    "id": trace_id,
                    "name": name,
                    "input": input_data,
                    "output": output_data,
                    "metadata": metrics
                }
            },
            {
                "id": str(uuid.uuid4()),
                "type": "score-create",
                "timestamp": ts,
                "body": {
                    "traceId": trace_id,
                    "name": "context_relevance",
                    "value": float(rel_score)
                }
            }
        ]
    }
    try:
        httpx.post(f"{host}/api/public/ingestion", json=payload, 
                   auth=(public_key, secret_key), timeout=2.0)
    except Exception as e:
        logger.warning(f"Langfuse Error: {e}")

# =========================================================
# 2. CORE SERVICE RAG
# =========================================================
class RagService:
    def __init__(self):
        self.milvus = MilvusClient(os.getenv("MILVUS_URI"))
        self.embedder = CustomEmbedder(os.getenv("EMBEDDING_MODEL_NAME"))
        self.model_name = os.getenv("OLLAMA_MODEL")
        
        self.llm = ChatOllama(
            model=self.model_name, 
            base_url=os.getenv("OLLAMA_URL"), 
            temperature=0.0
        )

        with open(Path(__file__).parent / "prompt.yaml", 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)

        # Initialisation de TOUTES tes chaînes
        self.rewriter_chain = self._build_chain("query_rewriter_prompt")
        self.decomp_chain = self._build_chain("query_decomposition_multiquery_prompt")
        self.rag_chain = self._build_chain("rag_template")
        self.relevance_chain = self._build_chain("reflection_relevance_check_prompt")
        self.grounded_chain = self._build_chain("reflection_groundedness_check_prompt")
        self.regen_chain = self._build_chain("reflection_response_regeneration_prompt")

    def _build_chain(self, prompt_name: str):
        data = self.prompts.get(prompt_name)
        msgs = [("system", data["system"].replace("/no_think", "").strip()), ("human", data["human"])]
        return ChatPromptTemplate.from_messages(msgs) | self.llm | StrOutputParser()

    def hybrid_retrieval(self, query: str, top_k: int):
        qvec = self.embedder(query)
        dense_req = AnnSearchRequest([qvec], "vector", {"metric_type": "COSINE"}, limit=top_k)
        sparse_req = AnnSearchRequest([query], "sparse", {"metric_type": "BM25"}, limit=top_k)
        res = self.milvus.hybrid_search(os.getenv("MILVUS_COLLECTION"), [sparse_req, dense_req], RRFRanker(k=60), limit=top_k, output_fields=["id", "text"])
        return res[0] if res else []

    def remote_rerank(self, query: str, hits: List[Any], top_k: int):
        if not hits: return []
        passages = [h['entity'].get("text", "") for h in hits]
        try:
            r = httpx.post(os.getenv("RERANK_URL"), json={"query": query, "passages": passages}, timeout=60.0)
            scores = r.json()["scores"]
            for i, h in enumerate(hits): h['entity']["rerank_score"] = scores[i]
            hits.sort(key=lambda x: x['entity']["rerank_score"], reverse=True)
            return [h['entity'] for h in hits if h['entity']["rerank_score"] > -8.0][:top_k]
        except: return [h['entity'] for h in hits][:top_k]

    async def run_pipeline(self, req: Any):
        t_start = time.time()
        history_str = "\n".join([f"{m.role}: {m.content}" for m in req.chat_history[-5:]]) or "Aucun historique."

        # 1. Reformulation
        rewritten = self.rewriter_chain.invoke({"chat_history": history_str, "input": req.query}).strip().split('\n')[0]

        # 2. Décomposition
        sub_queries_raw = self.decomp_chain.invoke({"question": rewritten})
        sub_queries = [line.strip() for line in sub_queries_raw.split('\n') if line.strip()][:3]
        if not sub_queries: sub_queries = [rewritten]

        # 3. Retrieval & Rerank
        all_hits = []
        for sq in sub_queries:
            all_hits.extend(self.hybrid_retrieval(sq, req.top_k_recall))
        
        unique_hits = {h['id']: h for h in all_hits}.values()
        contexts = self.remote_rerank(rewritten, list(unique_hits), req.top_k_final)
        ctx_str = "\n\n".join([f"DOC {i}: {c['text']}" for i, c in enumerate(contexts)])

        # 4. Relevance Check
        rel_score = 0
        if contexts:
            rel_raw = self.relevance_chain.invoke({"query": rewritten, "context": ctx_str})
            rel_score = int(''.join(filter(str.isdigit, rel_raw)) or "0")

        # 5. Génération & Groundedness
        if rel_score == 0:
            answer = "Désolé, les informations sont insuffisantes."
        else:
            answer = self.rag_chain.invoke({"context": ctx_str, "chat_history": history_str, "question": req.query})
            
            ground_raw = self.grounded_chain.invoke({"context": ctx_str, "response": answer})
            if "0" in ground_raw:
                answer = self.regen_chain.invoke({"context": ctx_str, "query": rewritten})

        total_latency = round(time.time() - t_start, 3)

        # 6. LOG GLOBAL (Même logique que ton script qui marchait)
        metrics = {
            "model": self.model_name,
            "total_latency": total_latency,
            "docs_retrieved": len(contexts),
            "sub_queries": len(sub_queries)
        }
        
        log_to_langfuse(
            name="RAG_Full_Chain",
            input_data={"query": req.query, "rewritten": rewritten},
            output_data={"answer": answer},
            rel_score=rel_score,
            metrics=metrics
        )

        return {
            "answer": answer,
            "rewritten_query": rewritten,
            "contexts": contexts,
            "run_id": str(uuid.uuid4())
        }

# =========================================================
# 3. FASTAPI APP
# =========================================================
class ChatMessage(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    query: str
    chat_history: List[ChatMessage] = []
    top_k_final: int = 5
    top_k_recall: int = 60

app = FastAPI()
service = None

@app.on_event("startup")
def startup():
    global service
    service = RagService()

@app.post("/ask")
async def ask(req: AskRequest):
    return await service.run_pipeline(req)