import os, yaml, sys, logging, httpx, uuid, time, asyncio
from typing import Any, List
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from nemoguardrails import RailsConfig, LLMRails
import redis
import time
#from langfuse import Langfuse

# --- Config ---
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
load_dotenv(ROOT_DIR / ".env")
if str(ROOT_DIR) not in sys.path: sys.path.insert(0, str(ROOT_DIR))

from src.utils.custom_embedding import CustomEmbedder
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_service")


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
                logger.error(f"CIRCUIT BREAKER OPEN [{self.name}]: Service considéré comme DOWN.")
                self.state = "OPEN"

    def record_success(self):
        if self.state != "CLOSED":
            logger.info(f"✅ CIRCUIT BREAKER CLOSED [{self.name}]: Service rétabli.")
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
    def __init__(self):
        self.milvus_breaker = CircuitBreaker("Milvus", failure_threshold=3, recovery_timeout=60)
        self.rerank_breaker = CircuitBreaker("Reranker", failure_threshold=3, recovery_timeout=30)
        self.milvus = MilvusClient(uri=os.getenv("MILVUS_URI"))
        self.embedder = CustomEmbedder(os.getenv("EMBEDDING_MODEL_NAME"))
        self.model_name = os.getenv("OLLAMA_MODEL")
        self.semaphore = asyncio.Semaphore(2) # Max 2 appels LLM simultanés sur Mac
        
        
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
        
        self.llm = ChatOllama(
            model=self.model_name, 
            base_url=os.getenv("OLLAMA_URL"), 
            temperature=0.0,
            num_predict=512, # Évite les réponses interminables
            num_ctx=4096     # Plus rapide à charger en mémoire
        )

        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT")),
            db=0,
            decode_responses=True # Important pour récupérer des strings et non des bytes
        )

        config = RailsConfig.from_path(str(Path(__file__).parent / "config"))
        self.rails = LLMRails(config)

        # self.langfuse = Langfuse(
        #     public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        #     secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        #     host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        # )

        with open(Path(__file__).parent / "prompt.yaml", 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)

        # Chaînes
        self.rewriter_chain = self._build_chain("query_rewriter_prompt")
        self.decomp_chain = self._build_chain("query_decomposition_multiquery_prompt")
        self.rag_chain = self._build_chain("rag_template")
        self.relevance_chain = self._build_chain("reflection_relevance_check_prompt")
        self.grounded_chain = self._build_chain("reflection_groundedness_check_prompt")
        self.regen_chain = self._build_chain("reflection_response_regeneration_prompt")

    def _build_chain(self, prompt_name: str):
        data = self.prompts.get(prompt_name)
        system_content = data["system"].replace("/no_think", "").strip()
        msgs = [("system", system_content), ("human", data["human"])]
        return ChatPromptTemplate.from_messages(msgs) | self.llm | StrOutputParser()
    
    async def _stream_final_text(self, text: str):
        """Simule le streaming pour un texte déjà généré et validé."""
        for chunk in text.split(' '):
            yield chunk + ' '
            await asyncio.sleep(0.02) # Petit délai pour l'expérience utilisateur
    
    # async def anonymize_text(self, text: str) -> str:
    #     """Masque les PII en utilisant le serveur GLiNER via NeMo Guardrails."""
    #     if not text:
    #         return ""
    #     try:
    #         # On force l'exécution du rail de masquage
    #         result = await self.rails.generate_async(prompt=text)
    #         return result.content
    #     except Exception as e:
    #         logger.error(f"Erreur anonymisation : {e}")
    #         return "[ANONYMIZATION_ERROR]"

    # async def anonymize_text(self, text: str) -> str:
    #     """Masque les PII en appelant DIRECTEMENT l'action sans passer par LLMRails.generate."""
    #     if not text:
    #         return ""
    #     try:
    #         # On passe par le runtime pour exécuter l'action Python pure
    #         # C'est la méthode la plus rapide (pas de LLM, pas de latence)
    #         result = await self.rails.runtime.action_dispatcher.execute_action(
    #             action_name="gliner_mask_pii",
    #             params={"text": text}
    #         )
            
    #         # Si le dispatcher renvoie une string, on la prend.
    #         # Sinon on essaye d'extraire la valeur de retour.
    #         if isinstance(result, str):
    #             return result
            
    #         # Parfois NeMo renvoie un objet ActionResult
    #         if hasattr(result, 'return_value') and result.return_value:
    #             return str(result.return_value)

    #         return str(result)

    #     except Exception as e:
    #         logger.error(f"❌ Erreur anonymisation directe : {e}")
    #         # En cas de fail, on renvoie une version sécurisée (tronquée) 
    #         # pour ne pas stopper le processus de log
    #         return f"[ANON_FAIL] {text[:30]}..."

    async def anonymize_text(self, text: str) -> str:
        """Masque les PII en fournissant les arguments requis par l'action interne NeMo."""
        if not text:
            return ""
        try:
            # L'action gliner_mask_pii de NeMo attend : text, source et config
            # On récupère la config depuis l'instance rails
            result = await self.rails.runtime.action_dispatcher.execute_action(
                action_name="gliner_mask_pii",
                params={
                    "text": text,
                    "source": "input", # Obligatoire pour gliner_mask_pii
                    "config": self.rails.config.rails.config.gliner # On injecte la config GLiNER
                }
            )
            
            # Gestion du retour (souvent une string après traitement)
            if isinstance(result, str):
                return result
            
            if hasattr(result, 'return_value') and result.return_value:
                return str(result.return_value)

            return str(result)

        except Exception as e:
            logger.error(f"❌ Erreur anonymisation (Arguments manquants réglés) : {e}")
            # Fallback de sécurité : on renvoie le texte original (ou tronqué) 
            # pour ne pas bloquer l'envoi vers Langfuse
            return text


    async def log_to_langfuse(self, name: str, input_data: Any, output_data: Any, rel_score: int, metrics: dict):
        host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        pub_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        sec_key = os.getenv("LANGFUSE_SECRET_KEY")
        if not pub_key or not sec_key: return

        # --- ÉTAPE RGPD CRITIQUE : Anonymisation avant envoi ---
        # On extrait les textes bruts
        raw_query = input_data.get("query", "") if isinstance(input_data, dict) else str(input_data)
        raw_answer = output_data.get("answer", "") if isinstance(output_data, dict) else str(output_data)

        # On lance l'anonymisation GLiNER en parallèle pour l'input et l'output
        safe_query, safe_answer = await asyncio.gather(
            self.anonymize_text(raw_query),
            self.anonymize_text(raw_answer)
        )

        trace_id = str(uuid.uuid4())
        payload = {
            "batch": [
                {
                    "id": str(uuid.uuid4()), 
                    "type": "trace-create", 
                    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime()),
                    "body": {
                        "id": trace_id, 
                        "name": name, 
                        "input": {"query": safe_query}, # Donnée masquée
                        "output": {"answer": safe_answer}, # Donnée masquée
                        "metadata": metrics
                    }
                },
                {
                    "id": str(uuid.uuid4()), 
                    "type": "score-create", 
                    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime()),
                    "body": {"traceId": trace_id, "name": "context_relevance", "value": float(rel_score)}
                }
            ]
        }
        
        try:
            # On utilise le client HTTP existant
            await self.http_client.post(f"{host}/api/public/ingestion", json=payload, auth=(pub_key, sec_key), timeout=5.0)
            logger.info(f"✨ Trace anonymisée envoyée à Langfuse : {name}")
        except Exception as e:
            logger.warning(f"Langfuse Error: {e}")



    async def hybrid_retrieval_batch(self, queries: List[str], top_k: int):
        qvecs = await self.embedder.embed_documents(queries)
        
        search_requests = []
        for i, qvec in enumerate(qvecs):
            search_requests.append(AnnSearchRequest([qvec], "vector", {"metric_type": "COSINE"}, limit=top_k))
            search_requests.append(AnnSearchRequest([queries[i]], "sparse", {"metric_type": "BM25"}, limit=top_k))

        res = await asyncio.to_thread(
            self.milvus.hybrid_search,
            os.getenv("MILVUS_COLLECTION"),
            search_requests,
            RRFRanker(k=30),
            limit=top_k,
            output_fields=["text", "doc_date", "doc_summary"]
        )
        
        unique_hits = {}
        for sublist in res:
            for hit in sublist:
                # Milvus renvoie l'ID à la racine du hit, les champs dans 'entity'
                doc_id = hit.get('id') 
                if doc_id not in unique_hits:
                    # On fusionne l'ID et les champs pour que contexts[i].get('text') fonctionne
                    entity = hit.get('entity', {})
                    entity['id'] = doc_id 
                    unique_hits[doc_id] = entity
        return list(unique_hits.values())
    


    async def remote_rerank(self, query: str, hits: List[Any], top_k: int):
        if not hits: return []

        rerank_url = "http://127.0.0.1:8282/rerank"
        query_text = str(query).strip()
        passages = [str(h.get("text", ""))[:2000].strip() for h in hits]
        
        batch_size = 16
        
        # On utilise un client temporaire sans pooling pour éviter les erreurs réseau vides
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            try:
                for i in range(0, len(passages), batch_size):
                    batch = passages[i : i + batch_size]
                    logger.info(f"📡 Envoi batch rerank {i//batch_size + 1}")
                    
                    try:
                        r = await client.post(
                            rerank_url, 
                            json={"query": query_text, "texts": batch},
                            headers={"Connection": "close"} # On demande de fermer après le batch
                        )
                        
                        if r.status_code != 200:
                            logger.error(f"TEI Error {r.status_code}: {r.text}")
                            continue

                        batch_results = r.json() 
                        for res in batch_results:
                            idx = i + res["index"]
                            if idx < len(hits):
                                hits[idx]["rerank_score"] = res.get("score")

                    except Exception as inner_e:
                        logger.error(f"Erreur sur batch {i}: {type(inner_e).__name__} - {inner_e}")

                hits.sort(key=lambda x: x.get("rerank_score", -100.0), reverse=True)
                return hits[:top_k]
                
            except Exception as e:
                logger.error(f"Exception globale Rerank : {e}")
                return hits[:top_k]

    async def run_pipeline_stream(self, req: AskRequest):
        t_start = time.time()

        # 1. Cache sémantique/exact
        normalized_query = req.query.strip().lower()
        cache_key = f"rag_cache:{normalized_query}"
        try:
            cached_answer = self.redis_client.get(cache_key)
            if cached_answer:
                logger.info(f"🚀 Cache Hit : {normalized_query}")
                yield cached_answer
                return
        except Exception as e:
            logger.warning(f"Redis Error: {e}")

        # 2. Retrieval direct (On saute la reformulation pour gagner 1-2s de latence)
        if not self.milvus_breaker.can_proceed():
            yield "ERREUR : Service indisponible."
            return

        try:
            # On utilise directement req.query pour le retrieval
            unique_hits = await self.hybrid_retrieval_batch([req.query], req.top_k_recall)
            self.milvus_breaker.record_success()
        except Exception as e:
            self.milvus_breaker.record_failure()
            yield "Erreur lors de la recherche."
            return

        # Rerank si disponible
        contexts = unique_hits 
        if self.rerank_breaker.can_proceed():
            try:
                contexts = await self.remote_rerank(req.query, unique_hits, req.top_k_final)
                self.rerank_breaker.record_success()
            except Exception:
                self.rerank_breaker.record_failure()

        if not contexts:
            yield "Aucune information trouvée."
            return
        
        # 3. Construction du contexte optimisé (Fiches d'indexation uniquement)
        ctx_str = "\n".join([
            f"SOURCE {i+1} [Résumé: {c.get('doc_summary', '')}]: {c.get('text', '')}" 
            for i, c in enumerate(contexts)
        ])

        history_str = "\n".join([f"{m.role}: {m.content}" for m in req.chat_history[-3:]]) or "Aucun."


        # 4. GÉNÉRATION + UNIQUE CHECK (GROUNDEDNESS)
        final_answer = ""
        is_grounded = False
        
        # Une seule tentative de génération + vérification pour minimiser la latence
        async with self.semaphore:
            # Génération de la réponse
            raw_response = await self.rag_chain.ainvoke({
                "context": ctx_str, 
                "chat_history": history_str, 
                "question": req.query
            })

            # UNIQUE CHECK : Groundedness (Hallucination)
            # On utilise {response} et {context} comme défini dans vos prompts
            grounded_raw = await self.grounded_chain.ainvoke({
                "context": ctx_str, 
                "response": raw_response 
            })
            
            # Extraction du score (0, 1 ou 2)
            score = "".join(filter(str.isdigit, grounded_raw))[:1]
            
            if score == "0":
                logger.warning("Hallucination détectée. Tentative de régénération unique...")
                final_answer = await self.regen_chain.ainvoke({
                    "context": ctx_str, 
                    "query": req.query
                })
            else:
                final_answer = raw_response
                is_grounded = True

        # 5. STREAMING FINAL
        async for chunk in self._stream_final_text(final_answer):
            yield chunk

        # 6. LOGS & CACHE
        latency = round(time.time() - t_start, 3)
        if final_answer and "HORS CONTEXTE" not in final_answer.upper():
            try:
                self.redis_client.setex(cache_key, 86400, final_answer)
            except: pass

        # Log asynchrone sécurisé
        async def safe_log():
            try:
                await self.log_to_langfuse(
                    name="RAG_Fast_Reflective", 
                    input_data={"query": req.query}, 
                    output_data={"answer": final_answer}, 
                    rel_score=1 if is_grounded else 0, 
                    metrics={"latency": latency, "grounded_score": score}
                )
            except Exception as e:
                logger.error(f"Langfuse Log Error: {e}")

        asyncio.create_task(safe_log())

    async def _finalize(self, start_time, q, rewritten, answer, ctx, rel_score, sub_qs):
        latency = round(time.time() - start_time, 3)
        await self.log_to_langfuse("RAG_Final_Optimized", {"query": q, "rewritten": rewritten}, {"answer": answer}, rel_score, {"latency": latency, "docs": len(ctx), "sub_queries": len(sub_qs)})
        return {"answer": answer, "rewritten_query": rewritten, "contexts": ctx, "latency": latency, "run_id": str(uuid.uuid4())}



app = FastAPI()
service = None

@app.on_event("startup")
async def startup():
    global service
    service = RagService()

# @app.on_event("shutdown")
# async def shutdown():
#     if service:
#         await service.http_client.aclose()

@app.on_event("shutdown")
async def shutdown():
    if service:
        # On ferme le client HTTP du service RAG
        await service.http_client.aclose()
        # On ferme AUSSI le client HTTP de l'embedder
        await service.embedder.client.aclose()
        logger.info("Clients HTTP fermés proprement.")

# @app.post("/ask")
# async def ask(req: AskRequest):
#     return await service.run_pipeline(req)

from fastapi.responses import StreamingResponse
import json

@app.post("/ask")
async def ask(req: AskRequest):
    # On renvoie un générateur asynchrone
    return StreamingResponse(
        service.run_pipeline_stream(req), 
        media_type="text/event-stream"
    )