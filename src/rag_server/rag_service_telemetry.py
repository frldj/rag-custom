import os, re, yaml, sys, logging, httpx, time, asyncio
from typing import Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from nemoguardrails import RailsConfig, LLMRails
import redis
from langfuse import Langfuse

# --- Imports Observabilité ---
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from src.rag_server.observability import setup_observability # Import corrigé selon ton fichier

# --- Config ---
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
load_dotenv(ROOT_DIR / ".env")
if str(ROOT_DIR) not in sys.path: sys.path.insert(0, str(ROOT_DIR))

from src.utils.custom_embedding import CustomEmbedder
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_service")

# --- Initialisation Observabilité ---
rag_metrics = setup_observability()
tracer = trace.get_tracer(__name__)

class CircuitBreaker:
    def __init__(self, name: str, failure_threshold=3, recovery_timeout=30):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            if self.state != "OPEN":
                logger.error(f"CIRCUIT BREAKER OPEN [{self.name}]: Service DOWN.")
                self.state = "OPEN"

    def record_success(self):
        if self.state != "CLOSED":
            logger.info(f"✅ CIRCUIT BREAKER CLOSED [{self.name}]: Rétabli.")
        self.failures = 0
        self.state = "CLOSED"

    def can_proceed(self) -> bool:
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF-OPEN"
                return True
            return False
        return True

class ChatMessage(BaseModel): 
    role: str
    content: str

class AskRequest(BaseModel):
    query: str
    chat_history: List[ChatMessage] = []
    top_k_final: int = 5
    top_k_recall: int = 60

class RagService:
    def __init__(self, metrics: dict):
        self.rag_metrics = metrics # On injecte les métriques
        self.milvus_breaker = CircuitBreaker("Milvus", 3, 60)
        self.rerank_breaker = CircuitBreaker("Reranker", 3, 30)
        self.milvus = MilvusClient(uri=os.getenv("MILVUS_URI"))
        self.embedder = CustomEmbedder(os.getenv("EMBEDDING_MODEL_NAME"))
        self.model_name = os.getenv("OLLAMA_MODEL")
        self.semaphore = asyncio.Semaphore(2)
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
        
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=os.getenv("OLLAMA_URL"),
            temperature=0.0,
            num_predict=int(os.getenv("LLM_MAX_TOKENS", 2048)),
            num_ctx=4096
        )

        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT")),
            db=0,
            decode_responses=True
        )

        pub_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        sec_key = os.getenv("LANGFUSE_SECRET_KEY")
        langfuse_host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        if pub_key and sec_key:
            self.langfuse = Langfuse(public_key=pub_key, secret_key=sec_key, host=langfuse_host)
            try:
                self.langfuse.auth_check()
                logger.info(f"Langfuse connecté : {langfuse_host}")
            except Exception as e:
                logger.error(f"Langfuse auth échoué ({langfuse_host}) : {e} — traces désactivées")
                self.langfuse = None
        else:
            self.langfuse = None
            logger.warning("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY non définis — traces désactivées")

        config = RailsConfig.from_path(str(Path(__file__).parent / "config"))
        gliner_endpoint = os.getenv("GLINER_SERVER_ENDPOINT")
        if gliner_endpoint and config.rails.config.gliner:
            config.rails.config.gliner.server_endpoint = gliner_endpoint
        self.rails = LLMRails(config)

        with open(Path(__file__).parent / "prompt.yaml", 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)

        self.rag_chain = self._build_chain("rag_template")
        self.grounded_chain = self._build_chain("reflection_groundedness_check_prompt")
        self.regen_chain = self._build_chain("reflection_response_regeneration_prompt")

    def _build_chain(self, prompt_name: str):
        data = self.prompts.get(prompt_name)
        system_content = data["system"].replace("/no_think", "").strip()
        msgs = [("system", system_content), ("human", data["human"])]
        return ChatPromptTemplate.from_messages(msgs) | self.llm | StrOutputParser()
    
    async def anonymize_text(self, text: str) -> str:
        if not text:
            return ""
        try:
            result, status = await self.rails.runtime.action_dispatcher.execute_action(
                action_name="gliner_mask_pii",
                params={"text": text, "config": self.rails.config, "source": "output"}
            )
            if status == "success" and result is not None:
                if isinstance(result, dict):
                    return result.get("tagged_text", text)
                return str(result)
            return text
        except Exception as e:
            logger.error(f"Erreur anonymisation GLiNER : {e}")
            return text
    
    async def log_to_langfuse(self, name: str, input_data: Any, output_data: Any, rel_score: int, metrics: dict):
        if not self.langfuse:
            return
        try:
            q = input_data.get("query", "") if isinstance(input_data, dict) else str(input_data)
            a = output_data.get("answer", "") if isinstance(output_data, dict) else str(output_data)

            safe_query, safe_answer = await asyncio.gather(
                self.anonymize_text(q),
                self.anonymize_text(a)
            )
            clean_metrics = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in metrics.items()}

            trace = self.langfuse.trace(
                name=name,
                input={"query": safe_query},
                output={"answer": safe_answer},
                metadata=clean_metrics,
            )
            trace.score(
                name="groundedness",
                value=float(rel_score or 0),
            )
            await asyncio.to_thread(self.langfuse.flush)
            logger.info(f"Trace Langfuse OK: {name} (trace_id={trace.id})")

        except Exception as e:
            logger.error(f"Erreur log_to_langfuse : {type(e).__name__} - {e}")
    
    async def run_pipeline_stream(self, req: AskRequest):
        self.rag_metrics["api_requests"].add(1, {"status": "started"})
        t_start = time.time()

        # 1. Cache Sémantique
        normalized_query = req.query.strip().lower()
        try:
            cached_answer = self.redis_client.get(f"rag_cache:{normalized_query}")
            if cached_answer:
                logger.info(f"Cache Hit")
                self.rag_metrics["cache_hits"].add(1)
                yield cached_answer
                return
        except Exception: pass

        # 2. Retrieval avec Tracing & Metrics
        contexts = []
        if not self.milvus_breaker.can_proceed():
            yield "ERREUR : Milvus indisponible."
            return

        with tracer.start_as_current_span("vector_db_retrieval"):
            t_ret_start = time.time()
            try:
                contexts = await self.hybrid_retrieval_batch([req.query], req.top_k_recall)
                self.milvus_breaker.record_success()
                self.rag_metrics["retrieval_time"].record((time.time() - t_ret_start) * 1000)
            except Exception:
                self.milvus_breaker.record_failure()
                self.rag_metrics["circuit_breaker_open"].add(1, {"service": "milvus"})
                yield "Erreur Milvus."
                return

        # 3. Rerank avec Tracing & Metrics
        if contexts and self.rerank_breaker.can_proceed():
            with tracer.start_as_current_span("context_reranking"):
                t_rer_start = time.time()
                try:
                    contexts = await self.remote_rerank(req.query, contexts, req.top_k_final)
                    self.rerank_breaker.record_success()
                    self.rag_metrics["reranker_time"].record((time.time() - t_rer_start) * 1000)
                except Exception:
                    self.rerank_breaker.record_failure()

        if not contexts:
            yield "Aucune information trouvée."
            return
        
        # 4. Préparation Contexte
        ctx_str = "\n".join([f"SOURCE {i+1}: {c.get('text', '')}" for i, c in enumerate(contexts)])
        history_str = "\n".join([f"{m.role}: {m.content}" for m in req.chat_history[-3:]]) or "Aucun."

        # 5. GÉNÉRATION & GROUNDEDNESS
        final_answer = "" # Initialisation en string, pas en list
        is_grounded = False
        
        # Estimation tokens entrée (basée sur ton calcul)
        in_tokens = len(ctx_str + history_str + req.query) // 4
        
        with tracer.start_as_current_span("llm_generation"):
            t_llm_start = time.time()
            async with self.semaphore:
                # Appel Rag Chain
                raw_response = await self.rag_chain.ainvoke({
                    "context": ctx_str, 
                    "chat_history": history_str, 
                    "question": req.query
                })
                
                # Télémétrie
                self.rag_metrics["llm_ttft"].record((time.time() - t_llm_start) * 1000)
                self.rag_metrics["rag_ttft"].record((time.time() - t_start) * 1000)

                # Groundedness Check
                grounded_raw = await self.grounded_chain.ainvoke({"context": ctx_str, "response": raw_response})
                score = "".join(filter(str.isdigit, grounded_raw))[:1]
                
                if score == "0":
                    logger.warning("Hallucination détectée. Régénération...")
                    self.rag_metrics["hallucinations"].add(1, {"model": self.model_name})
                    final_answer = await self.regen_chain.ainvoke({"context": ctx_str, "query": req.query})
                else:
                    final_answer = raw_response
                    is_grounded = True

                # Metrics Tokens (Correction des calculs)
                out_tokens = len(final_answer) // 4
                
                # Utilisation de tes compteurs spécifiques
                self.rag_metrics["input_tokens"].add(in_tokens, {"status": "200"})
                self.rag_metrics["output_tokens"].add(out_tokens, {"status": "200"})
                self.rag_metrics["total_tokens"].add(in_tokens + out_tokens, {"status": "200"})
                
                if "token_usage_distribution" in self.rag_metrics:
                    self.rag_metrics["token_usage_distribution"].record(in_tokens + out_tokens)

        # 6. Streaming final — re.split préserve les espaces/newlines sans jamais couper un mot
        if final_answer:
            for token in re.split(r'(\s+)', final_answer):
                if token:
                    yield token
                    await asyncio.sleep(0.01)

        # 7. Post-process (Cache & Log Langfuse)
        latency = round(time.time() - t_start, 3)

        # On prépare un dictionnaire de métriques détaillé pour Langfuse
        pipeline_metadata = {
            "latency": latency,
            "model": self.model_name,
            "nb_contexts_retrieved": len(contexts),
            "is_grounded": is_grounded,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "milvus_status": self.milvus_breaker.state,
            "rerank_status": self.rerank_breaker.state
        }

        if final_answer:
            try: self.redis_client.setex(f"rag_cache:{normalized_query}", 86400, final_answer)
            except: pass
        
        asyncio.create_task(self.log_to_langfuse(
            name="RAG_Stream_Instrumented", 
            input_data={"query": req.query}, 
            output_data={"answer": final_answer}, 
            rel_score=1 if is_grounded else 0, 
            metrics=pipeline_metadata 
        ))

    async def hybrid_retrieval_batch(self, queries: List[str], top_k: int):
        qvecs = await self.embedder.embed_documents(queries)
        search_requests = []
        for i, qvec in enumerate(qvecs):
            search_requests.append(AnnSearchRequest([qvec], "vector", {"metric_type": "COSINE"}, limit=top_k))
            search_requests.append(AnnSearchRequest([queries[i]], "sparse", {"metric_type": "BM25"}, limit=top_k))
        
        res = await asyncio.to_thread(
            self.milvus.hybrid_search, os.getenv("MILVUS_COLLECTION"),
            search_requests, RRFRanker(k=30), limit=top_k, 
            output_fields=["text", "doc_date", "doc_summary"]
        )
        unique_hits = {}
        for sublist in res:
            for hit in sublist:
                doc_id = hit.get('id')
                if doc_id not in unique_hits:
                    entity = hit.get('entity', {})
                    entity['id'] = doc_id
                    unique_hits[doc_id] = entity
        return list(unique_hits.values())

    async def remote_rerank(self, query: str, hits: List[Any], top_k: int):
        if not hits: return []

        rerank_url = "http://127.0.0.1:8282/rerank"
        query_text = str(query).strip()
        passages = [str(h.get("text", ""))[:2000].strip() for h in hits]
        
        # --- Reprise de ta logique de batching ---
        batch_size = 16
        t_rer_start = time.time() # Pour la métrique globale

        try:
            # On utilise le client HTTP du service (déjà configuré avec timeouts)
            for i in range(0, len(passages), batch_size):
                batch = passages[i : i + batch_size]
                logger.info(f"📡 Envoi batch rerank {i//batch_size + 1}")
                
                r = await self.http_client.post(
                    rerank_url, 
                    json={"query": query_text, "texts": batch}
                )
                
                if r.status_code == 200:
                    batch_results = r.json() 
                    for res in batch_results:
                        # On recalcule l'index global par rapport au batch
                        idx = i + res["index"]
                        if idx < len(hits):
                            hits[idx]["rerank_score"] = res.get("score")
                else:
                    logger.error(f"Rerank Error {r.status_code} sur batch {i}: {r.text}")

            # Une fois tous les batches finis, on trie
            hits.sort(key=lambda x: x.get("rerank_score", -100.0), reverse=True)
            
            # --- Enregistrement de la métrique SEULEMENT si succès ---
            duration_ms = (time.time() - t_rer_start) * 1000
            self.rag_metrics["reranker_time"].record(duration_ms)
            
            return hits[:top_k]
                
        except Exception as e:
            logger.error(f"Exception Rerank : {e}")
            self.rerank_breaker.record_failure() # On utilise le breaker en cas de crash
            return hits[:top_k]



# --- FastAPI App ---
app = FastAPI()
service = None

# Instrumentation auto FastAPI pour les traces HTTP
FastAPIInstrumentor.instrument_app(app)

@app.on_event("startup")
async def startup():
    global service
    # On initialise l'observabilité et on injecte dans le service
    metrics = setup_observability()
    service = RagService(metrics=metrics)

@app.on_event("shutdown")
async def shutdown():
    if service:
        await service.http_client.aclose()
        await service.embedder.client.aclose()
        service.redis_client.close()
        if service.langfuse:
            service.langfuse.flush()

@app.post("/ask")
async def ask(req: AskRequest):
    return StreamingResponse(service.run_pipeline_stream(req), media_type="text/event-stream")