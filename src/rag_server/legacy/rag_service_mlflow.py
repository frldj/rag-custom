import os
import yaml
import sys
import logging
import httpx
import time
from typing import Any, Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# MLflow Tracking & New Tracing feature
import mlflow
from mlflow.entities import SpanType

# =========================================================
# 1. ENVIRONNEMENT & CONFIGURATION
# =========================================================
current_dir = Path(__file__).resolve().parent 
ROOT_DIR = current_dir.parent.parent         
load_dotenv(ROOT_DIR / ".env")

# Ajout au path pour les imports custom
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.custom_embedding import CustomEmbedder
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_service")

# CONFIG MLFLOW
DB_PATH = current_dir / "mlflow.db"
TRACKING_URI = f"sqlite:///{DB_PATH.absolute()}"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("RAG_Production_Observability")

# Active le tracing automatique pour LangChain
mlflow.langchain.autolog(log_traces=True)

# =========================================================
# 2. MODÈLES DE DONNÉES
# =========================================================
class ChatMessage(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    query: str
    chat_history: List[ChatMessage] = []
    top_k_final: int = 5
    top_k_recall: int = 60

class ContextItem(BaseModel):
    id: Any
    text: str
    rerank_score: Optional[float] = None

class AskResponse(BaseModel):
    answer: str
    rewritten_query: str
    contexts: List[ContextItem]
    run_id: str  # Pour pouvoir faire du feedback plus tard

# =========================================================
# 3. LOGIQUE DE SERVICE
# =========================================================
class YamlPromptLoader:
    def __init__(self, path: Path):
        with open(path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)

    def get_template(self, name: str) -> ChatPromptTemplate:
        data = self.prompts.get(name)
        msgs = [
            ("system", data["system"].replace("/no_think", "").strip()), 
            ("human", data["human"])
        ]
        return ChatPromptTemplate.from_messages(msgs)

class RagService:
    def __init__(self):
        self.milvus = MilvusClient(os.getenv("MILVUS_URI"))
        self.embedder = CustomEmbedder(os.getenv("EMBEDDING_MODEL_NAME")) 
        self.llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"), base_url=os.getenv("OLLAMA_URL"), temperature=0.0)
        
        # Loader de prompts
        self.loader = YamlPromptLoader(current_dir / "prompt.yaml")
        
        # Initialisation des chaînes
        self.rewriter_chain = self.loader.get_template("query_rewriter_prompt") | self.llm | StrOutputParser()
        self.decomposition_chain = self.loader.get_template("query_decomposition_multiquery_prompt") | self.llm | StrOutputParser()
        self.rag_chain = self.loader.get_template("rag_template") | self.llm | StrOutputParser()
        
        self.relevance_eval = self.loader.get_template("reflection_relevance_check_prompt") | self.llm | StrOutputParser()
        self.groundedness_eval = self.loader.get_template("reflection_groundedness_check_prompt") | self.llm | StrOutputParser()
        self.regeneration_chain = self.loader.get_template("reflection_response_regeneration_prompt") | self.llm | StrOutputParser()

    def hybrid_retrieval(self, query: str, top_k: int) -> List[Dict]:
        qvec = self.embedder(query)
        dense_req = AnnSearchRequest(data=[qvec], anns_field="vector", param={"metric_type": "COSINE"}, limit=top_k)
        sparse_req = AnnSearchRequest(data=[query], anns_field="sparse", param={"metric_type": "BM25"}, limit=top_k)
        res = self.milvus.hybrid_search(os.getenv("MILVUS_COLLECTION"), [sparse_req, dense_req], RRFRanker(k=60), limit=top_k, output_fields=["id", "text"])
        return res[0] if res else []

    def remote_rerank(self, query: str, hits: List[Any], top_k: int) -> List[Dict]:
        if not hits: return []
        passages = [h.get('entity', h).get("text") for h in hits]
        meta = [{"id": h.get('entity', h).get("id"), "text": h.get('entity', h).get("text")} for h in hits]
        
        try:
            r = httpx.post(os.getenv("RERANK_URL"), json={"query": query, "passages": passages}, timeout=60.0)
            scores = r.json()["scores"]
            for m, s in zip(meta, scores): m["rerank_score"] = float(s)
            return sorted([m for m in meta if m["rerank_score"] >= -8.0], key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        except Exception as e:
            logger.warning(f"Rerank failed: {e}")
            return meta[:top_k]

    async def run_pipeline(self, req: AskRequest):
        # On démarre un Run MLflow global pour la requête
        with mlflow.start_run(run_name=f"RAG_Session_{int(time.time())}") as run:
            run_id = run.info.run_id
            
            # --- 0. LOGGING CONFIG (Comme Langfuse) ---
            mlflow.log_dict(self.loader.prompts, "used_prompts.yaml")
            start_time = time.time()
            history_str = "\n".join([f"{m.role}: {m.content}" for m in req.chat_history[-5:]]) or "Aucun historique."

            # --- 1. REFORMULATION (Sub-run) ---
            with mlflow.start_run(run_name="Rewriting", nested=True):
                rewritten = self.rewriter_chain.invoke({"chat_history": history_str, "input": req.query})
                rewritten = rewritten.strip().split('\n')[0]
                mlflow.log_param("original_query", req.query)
                mlflow.log_param("rewritten_query", rewritten)

            # --- 2. MULTI-QUERY (Sub-run) ---
            with mlflow.start_run(run_name="Decomposition", nested=True):
                sub_queries_raw = self.decomposition_chain.invoke({"question": rewritten})
                sub_queries = [line.strip() for line in sub_queries_raw.split('\n') if line.strip() and any(c.isdigit() for c in line[:2])]
                if not sub_queries: sub_queries = [rewritten]
                mlflow.log_metric("nb_sub_queries", len(sub_queries))

            # --- 3. RETRIEVAL & RERANK ---
            retrieval_start = time.time()
            all_hits = []
            for sq in sub_queries:
                all_hits.extend(self.hybrid_retrieval(sq, top_k=req.top_k_recall))
            
            # Déduplication
            seen_ids, unique_hits = set(), []
            for h in all_hits:
                h_id = h.get('id') or h.get('entity', {}).get('id')
                if h_id not in seen_ids:
                    unique_hits.append(h); seen_ids.add(h_id)

            contexts = self.remote_rerank(rewritten, unique_hits, top_k=req.top_k_final)
            mlflow.log_metric("retrieval_latency", time.time() - retrieval_start)
            mlflow.log_metric("nb_contexts_final", len(contexts))

            # --- 4. EVALUATION RELEVANCE ---
            ctx_str = "\n\n".join([f"DOC {i+1}: {c['text']}" for i, c in enumerate(contexts)])
            rel_score = 0
            if contexts:
                with mlflow.start_run(run_name="Eval_Relevance", nested=True):
                    rel_score_raw = self.relevance_eval.invoke({"query": rewritten, "context": ctx_str})
                    rel_score = int(''.join(filter(str.isdigit, rel_score_raw)) or "0")
                    mlflow.log_metric("relevance_score", rel_score)

            if rel_score == 0 and contexts:
                mlflow.set_tag("outcome", "rejected_no_relevance")
                return {"answer": "Désolé, les documents ne permettent pas de répondre.", "rewritten_query": rewritten, "contexts": [], "run_id": run_id}

            # --- 5. GENERATION ---
            gen_start = time.time()
            answer = self.rag_chain.invoke({
                "context": ctx_str or "Aucun contexte trouvé.",
                "chat_history": history_str,
                "question": req.query
            })
            mlflow.log_metric("generation_latency", time.time() - gen_start)

            # --- 6. EVALUATION GROUNDEDNESS (Hallucination) ---
            with mlflow.start_run(run_name="Eval_Groundedness", nested=True):
                ground_score_raw = self.groundedness_eval.invoke({"context": ctx_str, "response": answer})
                ground_score = int(''.join(filter(str.isdigit, ground_score_raw)) or "0")
                mlflow.log_metric("groundedness_score", ground_score)

            # --- 7. AUTO-CORRECTION ---
            if ground_score == 0 and contexts:
                with mlflow.start_run(run_name="Regeneration", nested=True):
                    answer = self.regeneration_chain.invoke({"context": ctx_str, "query": rewritten})
                    mlflow.set_tag("correction_performed", "true")

            # Finalisation des stats
            mlflow.log_metric("total_latency", time.time() - start_time)
            mlflow.set_tag("status", "completed")
            
            return {
                "answer": answer, 
                "rewritten_query": rewritten, 
                "contexts": contexts,
                "run_id": run_id
            }

# =========================================================
# 4. FASTAPI APP
# =========================================================
app = FastAPI()
service = None

@app.on_event("startup")
def startup():
    global service
    service = RagService()

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        result = await service.run_pipeline(req)
        return AskResponse(**result)
    except Exception as e:
        logger.error(f"Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)