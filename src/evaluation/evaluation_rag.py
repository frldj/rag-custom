# import os
# import sys
# import yaml
# import logging
# import asyncio
# import pandas as pd
# import numpy as np
# from typing import Any, Dict, List, Optional
# from pathlib import Path
# from dotenv import load_dotenv

# # Libs RAG & Evaluation
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.embeddings import Embeddings
# from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
# from FlagEmbedding import FlagReranker
# from datasets import Dataset
# from ragas import evaluate, RunConfig
# from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
# from ragas.llms import LangchainLLMWrapper



# # --- CORRECTION DU CHEMIN (PATH) ---
# # On récupère le chemin absolu du dossier où se trouve le script (evaluation/)
# CURRENT_DIR = Path(__file__).resolve().parent

# # On remonte jusqu'à la racine du projet (ministere_interieur/)
# # Si ton script est dans src/evaluation, il faut remonter de 2 niveaux :
# PROJECT_ROOT = CURRENT_DIR.parent.parent

# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))
#     print(f"✅ Root ajouté au sys.path : {PROJECT_ROOT}")

# # Maintenant l'import fonctionnera
# try:
#     from src.utils.custom_embedding import CustomEmbedder
# except ModuleNotFoundError:
#     print("❌ Erreur : Le module 'src' est toujours introuvable.")
#     print(f"DEBUG : sys.path est {sys.path}")
#     sys.exit(1)

# # --- CORRECTION DES IMPORTS RAGAS (v0.2+) ---
# from ragas.metrics import (
#     faithfulness, 
#     answer_relevancy, 
#     context_recall, 
#     context_precision
# )
# # Note : Si les warnings persistent, Ragas v0.2+ demande :
# # from ragas.metrics import faithfulness

# # --- 1. CONFIGURATION DES CHEMINS & ENV ---


# load_dotenv(PROJECT_ROOT / ".env")

# # Imports personnalisés (après sys.path.append)
# from src.utils.custom_embedding import CustomEmbedder

# # Variables d'env
# MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
# COLL = os.getenv("MILVUS_COLLECTION")
# OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# OLLAMA_MODEL = "qwen2.5:7b" 
# RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
# HF_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger("rag_eval")

# # --- 2. WRAPPER RAGAS ---
# class RagasEmbeddingsWrapper(Embeddings):
#     def __init__(self, custom_embedder):
#         self.custom_embedder = custom_embedder
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return [self.custom_embedder(t) for t in texts]
#     def embed_query(self, text: str) -> List[float]:
#         return self.custom_embedder(text)

# # --- 3. CLASSE RAGSERVICE ---
# class RagService:
#     def __init__(self):
#         logger.info("🚀 Initialisation du Pipeline Local (Agentic RAG)...")
#         self.milvus = MilvusClient(MILVUS_URI)
#         self.embedder = CustomEmbedder(HF_EMBEDDING_MODEL)
#         self.reranker = FlagReranker(RERANK_MODEL_NAME, use_fp16=False)
#         self.llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.0)

#         self.prompts_path = PROJECT_ROOT / "src" / "rag_server" / "prompt.yaml"
#         with open(self.prompts_path, 'r', encoding='utf-8') as f:
#             self.prompts_data = yaml.safe_load(f)

#         self.rewriter_chain = self._setup_chain("query_rewriter_prompt")
#         self.decomposition_chain = self._setup_chain("query_decomposition_multiquery_prompt")
#         self.rag_chain = self._setup_chain("rag_template")
#         self.relevance_chain = self._setup_chain("reflection_relevance_check_prompt")
#         self.groundedness_chain = self._setup_chain("reflection_groundedness_check_prompt")
#         self.regeneration_chain = self._setup_chain("reflection_response_regeneration_prompt")

#     def _setup_chain(self, template_name: str):
#         data = self.prompts_data.get(template_name)
#         messages = []
#         if data.get("system"):
#             messages.append(("system", data["system"].replace("/no_think", "").strip()))
#         human_text = data.get("human") or data.get("user") or data.get("question")
#         messages.append(("human", human_text))
#         return ChatPromptTemplate.from_messages(messages) | self.llm | StrOutputParser()

#     def hybrid_retrieval(self, query: str, top_k: int) -> List[Dict]:
#         qvec = self.embedder(query)
#         dense_req = AnnSearchRequest(data=[qvec], anns_field="vector", param={"metric_type": "COSINE", "params": {"ef": 100}}, limit=top_k)
#         sparse_req = AnnSearchRequest(data=[query], anns_field="sparse", param={"metric_type": "BM25"}, limit=top_k)
#         res = self.milvus.hybrid_search(COLL, [sparse_req, dense_req], RRFRanker(k=60), limit=top_k, output_fields=["id", "text", "source", "page_no", "section_title"])
#         return res[0] if res else []

#     def local_rerank(self, query: str, hits: List[Any], top_k: int) -> List[Dict]:
#         if not hits: return []
#         passages, meta = [], []
#         for h in hits:
#             ent = h.get('entity') if isinstance(h, dict) else h.entity
#             text, title = ent.get("text", ""), ent.get("section_title", "")
#             passages.append(f"{title}\n\n{text}" if title else text)
#             meta.append({"id": ent.get("id"), "text": text, "source": ent.get("source"), "page_no": ent.get("page_no"), "section_title": title})
        
#         scores = self.reranker.compute_score([[query, p[:2000]] for p in passages])
#         for m, s in zip(meta, scores): m["rerank_score"] = float(s)
#         filtered = [m for m in meta if m["rerank_score"] >= -8.0]
#         filtered.sort(key=lambda x: x["rerank_score"], reverse=True)
#         return filtered[:top_k]

#     async def run_pipeline(self, query: str, chat_history: List[Dict] = None):
#         history = chat_history or []
#         h_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]]) or "(Vide)"
        
#         # 1. Reformulation
#         rewritten = self.rewriter_chain.invoke({"chat_history": h_str, "input": query})
#         rewritten = rewritten.replace("Requête reformulée :", "").strip().split('\n')[0]

#         # 2. Multi-Query
#         sub_queries_raw = self.decomposition_chain.invoke({"question": rewritten})
#         sub_queries = [l.strip() for l in sub_queries_raw.split('\n') if l.strip() and any(c.isdigit() for c in l[:2])]
#         if not sub_queries: sub_queries = [rewritten]

#         # 3. Retrieval
#         all_hits = []
#         for sq in sub_queries: all_hits.extend(self.hybrid_retrieval(sq, top_k=20))
#         seen_ids, unique_hits = set(), []
#         for h in all_hits:
#             hid = h['id'] if isinstance(h, dict) else h.id
#             if hid not in seen_ids: unique_hits.append(h); seen_ids.add(hid)

#         # 4. Rerank
#         contexts = self.local_rerank(rewritten, unique_hits, top_k=5)
#         ctx_text = "\n\n".join([f"DOC {i+1}: {c['text']}" for i, c in enumerate(contexts)])

#         # 5. Reflection Relevance
#         if contexts:
#             rel_score = self.relevance_chain.invoke({"query": rewritten, "context": ctx_text})
#             if "0" in rel_score: return {"answer": "Non trouvé.", "rewritten_query": rewritten, "contexts": []}

#         # 6. Génération
#         answer = self.rag_chain.invoke({"context": ctx_text, "chat_history": h_str, "question": query})

#         # 7. Groundedness
#         ground_score = self.groundedness_chain.invoke({"context": ctx_text, "response": answer})
#         if "0" in ground_score:
#             answer = self.regeneration_chain.invoke({"context": ctx_text, "query": rewritten})

#         return {"answer": answer, "rewritten_query": rewritten, "contexts": contexts}

# # --- 4. LOGIQUE D'EVALUATION ---
# def check_source_match(expected_source, final_contexts):
#     if not expected_source: return None
#     expected_clean = str(expected_source).strip().lower()
#     for rank, ctx in enumerate(final_contexts, start=1):
#         current_src = str(ctx.get("source", "")).strip().lower()
#         if expected_clean in current_src or current_src in expected_clean:
#             return rank
#     return None

# async def main():
#     # Initialisation
#     service = RagService()
#     testset_path = PROJECT_ROOT / "src" / "evaluation" / "data" / "testset.jsonl"
    
#     if not testset_path.exists():
#         print(f"❌ Fichier non trouvé : {testset_path}")
#         return

#     df_test = pd.read_json(testset_path, lines=True)
#     print(f"📊 Chargé {len(df_test)} questions pour évaluation.")

#     # --- Phase 1: Retrieval Eval ---
#     results = []
#     print("🏃‍♂️ Lancement de la génération des réponses...")
#     for idx, row in df_test.iterrows():
#         question = row["question"]
#         ref_source = row.get("metadata", {}).get("source") or row.get("reference_context", "")
        
#         output = await service.run_pipeline(query=str(question))
#         found_at_rank = check_source_match(ref_source, output["contexts"])
        
#         results.append({
#             "question": question,
#             "answer": output["answer"],
#             "contexts": [c["text"] for c in output["contexts"]],
#             "ground_truth": row.get("reference_answer", ""),
#             "found_rank": found_at_rank,
#             "hit": found_at_rank is not None
#         })
#         print(f"✅ [{idx+1}/{len(df_test)}] Traité")

#     results_df = pd.DataFrame(results)

#     # Métriques Retrieval
#     hit_rate = results_df["hit"].mean()
#     mrr = np.mean([1.0/r if pd.notnull(r) else 0 for r in results_df["found_rank"]])
    
#     print(f"\n🎯 HIT RATE: {hit_rate:.2%}")
#     print(f"🔍 MRR: {mrr:.3f}")

#     # --- Phase 2: Ragas Eval ---
#     print("\n⚖️ Lancement de l'évaluation Ragas...")
#     llm_juge = ChatOllama(model="llama3.1:8b", base_url=OLLAMA_URL, temperature=0)
#     ragas_llm = LangchainLLMWrapper(llm_juge)
#     ragas_emb = RagasEmbeddingsWrapper(service.embedder)

#     ds = Dataset.from_pandas(results_df[["question", "answer", "contexts", "ground_truth"]])
    
#     ragas_result = evaluate(
#         ds,
#         metrics=[faithfulness, context_recall, context_precision],
#         llm=ragas_llm,
#         embeddings=ragas_emb,
#         run_config=RunConfig(timeout=240, max_workers=1)
#     )

#     print("\n📊 SYNTHÈSE RAGAS :")
#     print(ragas_result)
    
#     # Sauvegarde finale
#     results_df.to_csv("evaluation_finale.csv", index=False)
#     print("💾 Tout est sauvegardé dans 'evaluation_finale.csv'")

# if __name__ == "__main__":
#     asyncio.run(main())


# # import os
# # import sys
# # import yaml
# # import logging
# # import asyncio
# # import pandas as pd
# # import numpy as np
# # from typing import Any, Dict, List, Optional
# # from pathlib import Path
# # from dotenv import load_dotenv

# # # Libs RAG & Evaluation
# # import mlflow
# # import mlflow.langchain
# # from langchain_ollama import ChatOllama
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_core.output_parsers import StrOutputParser
# # from langchain_core.embeddings import Embeddings
# # from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
# # from FlagEmbedding import FlagReranker
# # from datasets import Dataset

# # # Imports Ragas (Version stable 0.1 / 0.2)
# # from ragas import evaluate, RunConfig
# # from ragas.metrics import (
# #     faithfulness,
# #     answer_relevancy,
# #     context_recall,
# #     context_precision
# # )
# # from ragas.llms import LangchainLLMWrapper

# # # --- 1. GESTION DU PYTHON PATH ---
# # # Récupère le chemin du script actuel (src/evaluation/)
# # CURRENT_DIR = Path(__file__).resolve().parent
# # # Remonte à la racine du projet (ministere_interieur/)
# # PROJECT_ROOT = CURRENT_DIR.parent.parent

# # if str(PROJECT_ROOT) not in sys.path:
# #     sys.path.insert(0, str(PROJECT_ROOT))
# #     print(f"✅ Racine du projet ajoutée au sys.path : {PROJECT_ROOT}")

# # # --- 2. CONFIGURATION & ENV ---
# # load_dotenv(PROJECT_ROOT / ".env")

# # try:
# #     from src.utils.custom_embedding import CustomEmbedder
# # except ModuleNotFoundError:
# #     print(f"❌ Erreur : Module 'src' non trouvé. Vérifiez que vous lancez le script depuis {PROJECT_ROOT}")
# #     sys.exit(1)

# # MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
# # COLL = os.getenv("MILVUS_COLLECTION")
# # OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# # OLLAMA_MODEL = "qwen2.5:7b" 
# # RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
# # HF_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")

# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# # logger = logging.getLogger("rag_eval")

# # # --- 3. WRAPPERS & SERVICES ---

# # class RagasEmbeddingsWrapper(Embeddings):
# #     def __init__(self, custom_embedder):
# #         self.custom_embedder = custom_embedder
# #     def embed_documents(self, texts: List[str]) -> List[List[float]]:
# #         return [self.custom_embedder(t) for t in texts]
# #     def embed_query(self, text: str) -> List[float]:
# #         return self.custom_embedder(text)

# # class RagService:
# #     def __init__(self):
# #         logger.info("🚀 Initialisation du Pipeline Local (Agentic RAG)...")
# #         self.milvus = MilvusClient(MILVUS_URI)
# #         self.embedder = CustomEmbedder(HF_EMBEDDING_MODEL)
# #         self.reranker = FlagReranker(RERANK_MODEL_NAME, use_fp16=False)
# #         self.llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.0)

# #         self.prompts_path = PROJECT_ROOT / "src" / "rag_server" / "prompt.yaml"
# #         with open(self.prompts_path, 'r', encoding='utf-8') as f:
# #             self.prompts_data = yaml.safe_load(f)

# #         self.rewriter_chain = self._setup_chain("query_rewriter_prompt")
# #         self.decomposition_chain = self._setup_chain("query_decomposition_multiquery_prompt")
# #         self.rag_chain = self._setup_chain("rag_template")
# #         self.relevance_chain = self._setup_chain("reflection_relevance_check_prompt")
# #         self.groundedness_chain = self._setup_chain("reflection_groundedness_check_prompt")
# #         self.regeneration_chain = self._setup_chain("reflection_response_regeneration_prompt")

# #     def _setup_chain(self, template_name: str):
# #         data = self.prompts_data.get(template_name)
# #         if not data: raise ValueError(f"Template {template_name} absent")
# #         messages = []
# #         if data.get("system"):
# #             messages.append(("system", data["system"].replace("/no_think", "").strip()))
# #         human_text = data.get("human") or data.get("user") or data.get("question")
# #         messages.append(("human", human_text))
# #         return ChatPromptTemplate.from_messages(messages) | self.llm | StrOutputParser()

# #     def hybrid_retrieval(self, query: str, top_k: int) -> List[Dict]:
# #         qvec = self.embedder(query)
# #         dense_req = AnnSearchRequest(data=[qvec], anns_field="vector", param={"metric_type": "COSINE", "params": {"ef": 100}}, limit=top_k)
# #         sparse_req = AnnSearchRequest(data=[query], anns_field="sparse", param={"metric_type": "BM25"}, limit=top_k)
# #         res = self.milvus.hybrid_search(COLL, [sparse_req, dense_req], RRFRanker(k=60), limit=top_k, output_fields=["id", "text", "source", "page_no", "section_title"])
# #         return res[0] if res else []

# #     def local_rerank(self, query: str, hits: List[Any], top_k: int) -> List[Dict]:
# #         if not hits: return []
# #         passages, meta = [], []
# #         for h in hits:
# #             ent = h.get('entity') if isinstance(h, dict) else h.entity
# #             text, title = ent.get("text", ""), ent.get("section_title", "")
# #             passages.append(f"{title}\n\n{text}" if title else text)
# #             meta.append({"id": ent.get("id"), "text": text, "source": ent.get("source"), "page_no": ent.get("page_no"), "section_title": title})
        
# #         scores = self.reranker.compute_score([[query, p[:2000]] for p in passages])
# #         for m, s in zip(meta, scores): m["rerank_score"] = float(s)
# #         filtered = [m for m in meta if m["rerank_score"] >= -8.0]
# #         filtered.sort(key=lambda x: x["rerank_score"], reverse=True)
# #         return filtered[:top_k]

# #     async def run_pipeline(self, query: str, chat_history: List[Dict] = None):
# #         history = chat_history or []
# #         h_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]]) or "(Vide)"
# #         rewritten = self.rewriter_chain.invoke({"chat_history": h_str, "input": query})
# #         rewritten = rewritten.strip().split('\n')[0]

# #         sub_queries_raw = self.decomposition_chain.invoke({"question": rewritten})
# #         sub_queries = [l.strip() for l in sub_queries_raw.split('\n') if l.strip() and any(c.isdigit() for c in l[:2])]
# #         if not sub_queries: sub_queries = [rewritten]

# #         all_hits = []
# #         for sq in sub_queries: all_hits.extend(self.hybrid_retrieval(sq, top_k=20))
# #         seen_ids, unique_hits = set(), []
# #         for h in all_hits:
# #             hid = h['id'] if isinstance(h, dict) else h.id
# #             if hid not in seen_ids: unique_hits.append(h); seen_ids.add(hid)

# #         contexts = self.local_rerank(rewritten, unique_hits, top_k=5)
# #         ctx_text = "\n\n".join([f"DOC {i+1}: {c['text']}" for i, c in enumerate(contexts)])

# #         if contexts:
# #             rel_score = self.relevance_chain.invoke({"query": rewritten, "context": ctx_text})
# #             if "0" in rel_score: return {"answer": "Information non trouvée.", "rewritten_query": rewritten, "contexts": []}

# #         answer = self.rag_chain.invoke({"context": ctx_text, "chat_history": h_str, "question": query})
# #         ground_score = self.groundedness_chain.invoke({"context": ctx_text, "response": answer})
# #         if "0" in ground_score:
# #             answer = self.regeneration_chain.invoke({"context": ctx_text, "query": rewritten})

# #         return {"answer": answer, "rewritten_query": rewritten, "contexts": contexts}

# # # --- 4. FONCTION MATCHING ---
# # def check_source_match(expected_source, final_contexts):
# #     if not expected_source: return None
# #     expected_clean = str(expected_source).strip().lower()
# #     for rank, ctx in enumerate(final_contexts, start=1):
# #         current_src = str(ctx.get("source", "")).strip().lower()
# #         if expected_clean in current_src or current_src in expected_clean:
# #             return rank
# #     return None

# # # --- 5. MAIN EVALUATION ---

# # async def main():
# #     # Setup MLflow
# #     mlflow.set_tracking_uri("file:./mlruns")
# #     mlflow.set_experiment("RAG_Ministere_Evaluation")

# #     service = RagService()
# #     testset_path = PROJECT_ROOT / "src" / "evaluation" / "data" / "testset_filtered.jsonl"
    
# #     if not testset_path.exists():
# #         logger.error(f"Fichier testset non trouvé à {testset_path}")
# #         return

# #     df_test = pd.read_json(testset_path, lines=True)
    
# #     with mlflow.start_run(run_name=f"Eval_{OLLAMA_MODEL}"):
# #         # Logging Params
# #         mlflow.log_params({
# #             "model": OLLAMA_MODEL,
# #             "embedder": HF_EMBEDDING_MODEL,
# #             "reranker": RERANK_MODEL_NAME,
# #             "agentic_features": "Multi-query, Reflection, Groundedness"
# #         })

# #         results = []
# #         logger.info(f"🏃 Lancement sur {len(df_test)} questions...")
        
# #         for idx, row in df_test.iterrows():
# #             q = row["question"]
# #             ref_src = row.get("metadata", {}).get("source") or row.get("reference_context", "")
            
# #             try:
# #                 out = await service.run_pipeline(query=q)
# #                 rank = check_source_match(ref_src, out["contexts"])
                
# #                 results.append({
# #                     "question": q,
# #                     "answer": out["answer"],
# #                     "contexts": [c["text"] for c in out["contexts"]],
# #                     "ground_truth": row.get("reference_answer", ""),
# #                     "found_rank": rank,
# #                     "hit": rank is not None
# #                 })
# #             except Exception as e:
# #                 logger.error(f"Erreur question {idx}: {e}")

# #         results_df = pd.DataFrame(results)
        
# #         # Métriques Retrieval
# #         hit_rate = results_df["hit"].mean()
# #         mrr = np.mean([1.0/r if pd.notnull(r) else 0 for r in results_df["found_rank"]])
        
# #         mlflow.log_metric("hit_rate", hit_rate)
# #         mlflow.log_metric("mrr", mrr)

# #         # --- Ragas Eval ---
# #         logger.info("⚖️ Calcul des scores Ragas...")
# #         llm_juge = ChatOllama(model="llama3.1:8b", base_url=OLLAMA_URL, temperature=0)
# #         ragas_llm = LangchainLLMWrapper(llm_juge)
# #         ragas_emb = RagasEmbeddingsWrapper(service.embedder)
        
# #         ds = Dataset.from_pandas(results_df[["question", "answer", "contexts", "ground_truth"]])
# #         ragas_result = evaluate(
# #             ds,
# #             metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
# #             llm=ragas_llm,
# #             embeddings=ragas_emb,
# #             run_config=RunConfig(timeout=240, max_workers=1)
# #         )

# #         for metric, score in ragas_result.items():
# #             mlflow.log_metric(f"ragas_{metric}", score)

# #         # Artefacts
# #         output_csv = "evaluation_results.csv"
# #         results_df.to_csv(output_csv, index=False)
# #         mlflow.log_artifact(output_csv)
# #         mlflow.log_artifact(str(service.prompts_path))

# #         print(f"\n✅ Évaluation terminée.\nHit Rate: {hit_rate:.2%}\nMRR: {mrr:.3f}")
# #         print(f"Consultez MLflow avec: mlflow ui")

# # if __name__ == "__main__":
# #     asyncio.run(main())


import os
import sys
import yaml
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv


CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")


from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from datasets import Dataset

# Ragas imports
from ragas import evaluate, RunConfig
from ragas.metrics import (
    Faithfulness, 
    AnswerRelevancy, 
    ContextRecall, 
    ContextPrecision
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langfuse import Langfuse


class RagasCompatibleEmbedder:
    def __init__(self, custom_embedder_instance):
        self.instance = custom_embedder_instance
    def embed_query(self, text):
        return self.instance(text)
    def embed_documents(self, texts):
        return [self.instance(t) for t in texts]


class RagService:
    def __init__(self):
        from src.utils.custom_embedding import CustomEmbedder
        self.milvus = MilvusClient(os.getenv("MILVUS_URI", "http://localhost:19530"))
        self.embedder = CustomEmbedder(os.getenv("EMBEDDING_MODEL_NAME"))
        self.llm = ChatOllama(
            model="qwen2.5:7b", 
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"), 
            temperature=0.0
        )
        prompt_path = ROOT_DIR / "src" / "rag_server" / "prompt.yaml"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)

    async def run(self, query: str):
        qvec = self.embedder(query)
        res = self.milvus.hybrid_search(
            os.getenv("MILVUS_COLLECTION"),
            [AnnSearchRequest(data=[qvec], anns_field="vector", param={}, limit=10),
             AnnSearchRequest(data=[query], anns_field="sparse", param={}, limit=10)],
            RRFRanker(), limit=5, output_fields=["text", "source"]
        )
        contexts = [{"text": h['entity']['text'], "source": h['entity']['source']} for h in res[0]]
        ctx_text = "\n".join([c['text'] for c in contexts])
        prompt_val = self.prompts["rag_template"]["human"]
        prompt = ChatPromptTemplate.from_messages([("human", prompt_val)])
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": ctx_text, "question": query, "chat_history": ""})
        return {"answer": answer, "contexts": contexts}


async def main():
    service = RagService()
    testset_path = ROOT_DIR / "src" / "evaluation" / "data" / "testset_filtered.jsonl"
    
    if not testset_path.exists():
        print(f"Testset introuvable : {testset_path}")
        return

    df_test = pd.read_json(testset_path, lines=True)
    dataset_name = "Giskard_Expert_V2_20260331_1727"
    
    langfuse = Langfuse()
    results_for_ragas = []
    stats = {"hits": [], "mrr": []}

    print(f"Évaluation lancée sur {len(df_test)} questions...")

    for idx, row in df_test.iterrows():
        question = row["question"]
        ref_src = row.get("metadata", {}).get("source") or row.get("source", "")
        ground_truth = row.get("reference_answer", "")

        try:
            out = await service.run(question)
            rank = next((i for i, c in enumerate(out["contexts"], 1) if str(ref_src).lower() in str(c['source']).lower()), 0)
            stats["hits"].append(1 if rank > 0 else 0)
            stats["mrr"].append(1.0/rank if rank > 0 else 0)

            results_for_ragas.append({
                "user_input": question,
                "response": out["answer"],
                "retrieved_contexts": [c["text"] for c in out["contexts"]],
                "reference": ground_truth
            })

            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                input=question,
                expected_output=ground_truth,
                metadata={"hit": 1 if rank > 0 else 0}
            )
            print(f"✅ {idx+1}/{len(df_test)} | Rank: {rank}")

        except Exception as e:
            print(f"Erreur item {idx}: {e}")

  
    print("\nCalcul des scores Ragas...")
    
    # Juge LLM (Llama 3.2 3B) et Embedder TEI
    eval_llm = LangchainLLMWrapper(ChatOllama(model="llama3.2:3b", base_url=os.getenv("OLLAMA_URL"), format="json", temperature=0))
    eval_emb = LangchainEmbeddingsWrapper(RagasCompatibleEmbedder(service.embedder))
    
    ds = Dataset.from_pandas(pd.DataFrame(results_for_ragas))
    
    ragas_result = evaluate(
        dataset=ds,
        metrics=[ 
            AnswerRelevancy(), 
            ContextPrecision(), 
            ContextRecall()
        ],
        llm=eval_llm,
        embeddings=eval_emb,
        column_map={
            "question": "user_input", 
            "answer": "response", 
            "contexts": "retrieved_contexts", 
            "ground_truth": "reference"
        },
        run_config=RunConfig(max_workers=1, timeout=600)
    )


    print("\n" + "="*50)
    print("📊 RÉSULTATS FINAUX")
    print("-" * 50)
    print(f"Hit Rate (Top-5) : {np.mean(stats['hits']):.2%}")
    print(f"MRR              : {np.mean(stats['mrr']):.3f}")
    print("-" * 50)


    scores_dict = {}
    
    # Cas 1 : ragas_result est un dictionnaire
    if isinstance(ragas_result, dict):
        scores_dict = ragas_result
    # Cas 2 : ragas_result est un objet Result avec un attribut .scores (Dictionnaire)
    elif hasattr(ragas_result, "scores") and isinstance(ragas_result.scores, dict):
        scores_dict = ragas_result.scores
    # Cas 3 : ragas_result est un objet Result (itérable ou dataframe-like)
    else:
        try:
            scores_dict = ragas_result.to_pandas().mean(numeric_only=True).to_dict()
        except:
            print("Impossible de parser automatiquement les scores Ragas.")

    for metric, val in scores_dict.items():
        # Ragas retourne parfois une liste de scores par question, on prend la moyenne
        mean_score = np.nanmean(val) if isinstance(val, list) else val
        print(f"{metric:18}: {mean_score:.3f}")
        
        # Envoi à Langfuse
        try:
            langfuse.score(name=f"avg_{metric}", value=float(mean_score))
        except Exception as lf_e:
            pass 

    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())