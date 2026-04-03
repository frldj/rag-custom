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

# --- Config ---
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
load_dotenv(ROOT_DIR / ".env")
if str(ROOT_DIR) not in sys.path: sys.path.insert(0, str(ROOT_DIR))

from src.utils.custom_embedding import CustomEmbedder
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_service")

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
        self.milvus = MilvusClient(os.getenv("MILVUS_URI"))
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

    async def log_to_langfuse(self, name: str, input_data: Any, output_data: Any, rel_score: int, metrics: dict):
        host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        pub_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        sec_key = os.getenv("LANGFUSE_SECRET_KEY")
        if not pub_key or not sec_key: return

        trace_id = str(uuid.uuid4())
        payload = {
            "batch": [
                {"id": str(uuid.uuid4()), "type": "trace-create", "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime()),
                 "body": {"id": trace_id, "name": name, "input": input_data, "output": output_data, "metadata": metrics}},
                {"id": str(uuid.uuid4()), "type": "score-create", "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime()),
                 "body": {"traceId": trace_id, "name": "context_relevance", "value": float(rel_score)}}
            ]
        }
        try:
            await self.http_client.post(f"{host}/api/public/ingestion", json=payload, auth=(pub_key, sec_key), timeout=2.0)
        except Exception as e:
            logger.warning(f"Langfuse Error: {e}")

    # def hybrid_retrieval(self, query: str, top_k: int):
    #     qvec = self.embedder(query)
    #     dense_req = AnnSearchRequest([qvec], "vector", {"metric_type": "COSINE"}, limit=top_k)
    #     sparse_req = AnnSearchRequest([query], "sparse", {"metric_type": "BM25"}, limit=top_k)
    #     res = self.milvus.hybrid_search(os.getenv("MILVUS_COLLECTION"), [sparse_req, dense_req], RRFRanker(k=60), limit=top_k, output_fields=["id", "text", "doc_date", "doc_summary"])
    #     return res[0] if res else []

    # async def hybrid_retrieval_batch(self, queries: List[str], top_k: int):
    #     # 1. On vectorise tout d'un coup (Batch Embedding)
    #     # Vérifie que ton CustomEmbedder supporte le passage d'une liste
    #     #qvecs = await asyncio.to_thread(self.embedder.embed_documents, queries)
    #     qvecs = await self.embedder.embed_documents(queries)
        
    #     search_requests = []
    #     for i, qvec in enumerate(qvecs):
    #         # On prépare les requêtes denses et sparses pour CHAQUE sous-question
    #         search_requests.append(AnnSearchRequest([qvec], "vector", {"metric_type": "COSINE"}, limit=top_k))
    #         search_requests.append(AnnSearchRequest([queries[i]], "sparse", {"metric_type": "BM25"}, limit=top_k))

    #     # 2. Un seul appel réseau vers Milvus pour N recherches
    #     res = await asyncio.to_thread(
    #         self.milvus.hybrid_search,
    #         os.getenv("MILVUS_COLLECTION"),
    #         search_requests,
    #         RRFRanker(k=60),
    #         limit=top_k,
    #         output_fields=["id", "text", "doc_date", "doc_summary"]
    #     )
    #     # On aplatit les résultats et on déduplique par ID
    #     all_hits = [hit for sublist in res for hit in sublist]
    #     return {h['id']: h for h in all_hits}.values()

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
        
        # On prépare des passages enrichis pour le Reranker
        passages = []
        for h in hits:
            meta = h['entity'].get("metadata", {})
            text = h['entity'].get("text", "")
            summary = meta.get("document_summary", "")
            # On combine Résumé + Début du texte pour donner du contexte au Reranker
            combined = f"{summary[:200]}... {text}" 
            passages.append(combined)
            
        try:
            r = await self.http_client.post(os.getenv("RERANK_URL"), json={"query": query, "passages": passages})
            scores = r.json()["scores"]
            for i, h in enumerate(hits): h['entity']["rerank_score"] = scores[i]
            hits.sort(key=lambda x: x['entity']["rerank_score"], reverse=True)
            return [h['entity'] for h in hits if h['entity']["rerank_score"] > -8.0][:top_k]
        except Exception as e:
            logger.error(f"Rerank Error: {e}")
            return [h['entity'] for h in hits][:top_k]

    # async def run_pipeline(self, req: Any):
    #     t_start = time.time()
    #     history_str = "\n".join([f"{m.role}: {m.content}" for m in req.chat_history[-5:]]) or "Aucun historique."

    #     1 & 2. Reformulation & Décomposition
    #     rewritten = (await asyncio.to_thread(self.rewriter_chain.ainvoke, {"chat_history": history_str, "input": req.query})).strip().split('\n')[0]
    #     sub_queries_raw = await asyncio.to_thread(self.decomp_chain.ainvoke, {"question": rewritten})
    #     sub_queries = [line.strip() for line in sub_queries_raw.split('\n') if line.strip() and not line.startswith('-')][:3] or [rewritten]

    #     3. Retrieval Parallèle
    #     retrieval_tasks = [asyncio.to_thread(self.hybrid_retrieval, sq, req.top_k_recall) for sq in sub_queries]
    #     retrieval_results = await asyncio.gather(*retrieval_tasks)
    #     all_hits = [hit for sublist in retrieval_results for hit in sublist]
    #     unique_hits = {h['id']: h for h in all_hits}.values()

    #     4. Reranking
    #     contexts = await self.remote_rerank(rewritten, list(unique_hits), req.top_k_final)

    #     if not contexts:
    #         return await self._finalize(t_start, req.query, rewritten, "Désolé, aucune information trouvée.", [], 0, sub_queries)

    #     Re-construction enrichie du contexte
    #     ctx_blocks = []
    #     for i, c in enumerate(contexts):
    #         summary = c.get("doc_summary") or "Résumé non disponible"
    #         doc_date = c.get("doc_date") or "Date inconnue"
    #         content = c.get("text") or "Contenu manquant"

    #         block = (
    #             f"--- SOURCE {i+1} [Date: {doc_date}] ---\n"
    #             f"CONTEXTE GÉNÉRAL : {summary}\n"
    #             f"EXTRAIT SPÉCIFIQUE :\n{content}\n"
    #             f"--------------------------------"
    #         )
    #         ctx_blocks.append(block)

    #     ctx_str = "\n\n".join(ctx_blocks)


    #     5 & 6. PARALLÉLISATION : Relevance Check + Génération
    #     rel_task = asyncio.to_thread(self.relevance_chain.invoke, {"query": rewritten, "context": ctx_str})
    #     gen_task = asyncio.to_thread(self.rag_chain.invoke, {"context": ctx_str, "chat_history": history_str, "question": req.query})
        
    #     rel_raw, answer = await asyncio.gather(rel_task, gen_task)
    #     rel_score = int(''.join(filter(str.isdigit, rel_raw)) or "0")

    #     if rel_score == 0:
    #         answer = "Désolé, les informations extraites de ma base ne sont pas suffisantes pour répondre précisément."
    #     else:
    #         7. Groundedness check (Vérification finale)
    #         ground_raw = await asyncio.to_thread(self.grounded_chain.invoke, {"context": ctx_str, "response": answer})
    #         if "0" in ground_raw:
    #             answer = await asyncio.to_thread(self.regen_chain.invoke, {"context": ctx_str, "query": rewritten})

    #     final_contexts_for_api = []
    #     for c in contexts:
    #         final_contexts_for_api.append({
    #             "id": c.get("id"),
    #             "text": c.get("text"),
    #             "rerank_score": c.get("rerank_score"),
    #             "doc_date": c.get("doc_date", "Inconnue"),
    #             "doc_summary": c.get("doc_summary", "Non disponible")
    #         })

    #     return await self._finalize(t_start, req.query, rewritten, answer, final_contexts_for_api, rel_score, sub_queries)

    # async def run_pipeline(self, req: Any):
    #     t_start = time.time()
    #     history_str = "\n".join([f"{m.role}: {m.content}" for m in req.chat_history[-5:]]) or "Aucun historique."

    #     # --- ÉTAPE 1 & 2 : Reformulation & Décomposition ---
    #     # Utilise directement await sur .ainvoke(), pas besoin de to_thread ici !
    #     rewritten_raw = await self.rewriter_chain.ainvoke({"chat_history": history_str, "input": req.query})
    #     rewritten = rewritten_raw.strip().split('\n')[0]
        
    #     # sub_queries_raw = await self.decomp_chain.ainvoke({"question": rewritten})
    #     # sub_queries = [line.strip() for line in sub_queries_raw.split('\n') if line.strip() and not line.startswith('-')][:3] or [rewritten]

    #     sub_queries = [rewritten]

    #     # --- ÉTAPE 3 : Retrieval BATCH (L'optimisation Milvus) ---
    #     # Ici, on n'utilise plus gather() sur 3 fonctions, mais UN SEUL appel qui contient tout.
    #     unique_hits = await self.hybrid_retrieval_batch(sub_queries, req.top_k_recall)

    #     # --- ÉTAPE 4 : Reranking ---
    #     contexts = await self.remote_rerank(rewritten, list(unique_hits), req.top_k_final)

    #     if not contexts:
    #         return await self._finalize(t_start, req.query, rewritten, "Désolé, aucune source trouvée.", [], 0, sub_queries)

    #     # Reconstruction du contexte (ta logique reste la même)
    #     ctx_str = "\n\n".join([
    #         f"--- SOURCE {i+1} [Date: {c.get('doc_date', 'Inconnue')}] ---\n"
    #         f"CONTEXTE GÉNÉRAL : {c.get('doc_summary', '')}\n"
    #         f"EXTRAIT : {c.get('text', '')}"
    #         for i, c in enumerate(contexts)
    #     ])

    #     # --- ÉTAPE 5 & 6 : PARALLÉLISATION LLM avec SÉMAPHORE ---
    #     # On utilise le sémaphore pour protéger ton Mac
    #     async with self.semaphore:
    #         rel_task = self.relevance_chain.ainvoke({"query": rewritten, "context": ctx_str})
    #         gen_task = self.rag_chain.ainvoke({"context": ctx_str, "chat_history": history_str, "question": req.query})
    #         rel_raw, answer = await asyncio.gather(rel_task, gen_task)

    #     # --- ÉTAPE 7 : Validation Score & Groundedness ---
    #     rel_score = int(''.join(filter(str.isdigit, rel_raw)) or "0")
        
    #     if rel_score == 0:
    #         answer = "Désolé, les informations ne sont pas suffisantes."
    #     else:
    #         ground_raw = await self.grounded_chain.ainvoke({"context": ctx_str, "response": answer})
    #         if "0" in ground_raw:
    #             answer = await self.regen_chain.ainvoke({"context": ctx_str, "query": rewritten})

    #     return await self._finalize(t_start, req.query, rewritten, answer, contexts, rel_score, sub_queries)

    # async def run_pipeline(self, req: Any):
    #     t_start = time.time()
    #     history_str = "\n".join([f"{m.role}: {m.content}" for m in req.chat_history[-5:]]) or "Aucun historique."

    #     # --- ÉTAPE 1 : Reformulation (1 seul appel Ollama) ---
    #     rewritten_raw = await self.rewriter_chain.ainvoke({"chat_history": history_str, "input": req.query})
    #     rewritten = rewritten_raw.strip().split('\n')[0]
    #     sub_queries = [rewritten]

    #     # --- ÉTAPE 2 : Retrieval Milvus ---
    #     unique_hits = await self.hybrid_retrieval_batch(sub_queries, req.top_k_recall)

    #     # --- ÉTAPE 3 : Reranking (CORRECTION ICI) ---
    #     # On ne lance PAS remote_rerank si l'API est éteinte. 
    #     # On récupère directement les entités de Milvus.
    #     hits_list = list(unique_hits)
    #     contexts = [h['entity'] for h in hits_list][:req.top_k_final]

    #     if not contexts:
    #         return await self._finalize(t_start, req.query, rewritten, "Désolé, aucune source trouvée.", [], 0, sub_queries)

    #     # Reconstruction du texte de contexte
    #     ctx_str = "\n\n".join([
    #         f"--- SOURCE {i+1} [Date: {c.get('doc_date', 'Inconnue')}] ---\n"
    #         f"RÉSUMÉ : {c.get('doc_summary', '')}\n"
    #         f"TEXTE : {c.get('text', '')}"
    #         for i, c in enumerate(contexts)
    #     ])

    #     # --- ÉTAPE 4 : Génération (On simplifie pour tester la vitesse) ---
    #     # Pour l'instant, on ignore la "relevance_chain" et la "grounded_chain" 
    #     # car chaque appel rajoute 5 à 10 secondes sur un Mac.
    #     async with self.semaphore:
    #         answer = await self.rag_chain.ainvoke({
    #             "context": ctx_str, 
    #             "chat_history": history_str, 
    #             "question": req.query
    #         })
        
    #     rel_score = 1 # Valeur par défaut pour le test

    #     return await self._finalize(t_start, req.query, rewritten, answer, contexts, rel_score, sub_queries)

    async def run_pipeline_stream(self, req: AskRequest):
        t_start = time.time()
        history_str = "\n".join([f"{m.role}: {m.content}" for m in req.chat_history[-5:]]) or "Aucun historique."

        # 1. Étape rapide : Reformulation
        rewritten_raw = await self.rewriter_chain.ainvoke({"chat_history": history_str, "input": req.query})
        rewritten = rewritten_raw.strip().split('\n')[0]

        # 2. Étape rapide : Milvus
        unique_hits = await self.hybrid_retrieval_batch([rewritten], req.top_k_recall)
        contexts = [h for h in unique_hits][:req.top_k_final]

        if not contexts:
            yield "Désolé, je n'ai trouvé aucune information."
            return

        ctx_str = "\n\n".join([
            f"SOURCE {i+1}: {c.get('text', '')}" for i, c in enumerate(contexts)
        ])

        # 3. Étape longue : Génération en STREAMING
        # On utilise .astream() au lieu de .ainvoke()
        async with self.semaphore:
            async for chunk in self.rag_chain.astream({
                "context": ctx_str, 
                "chat_history": history_str, 
                "question": req.query
            }):
                # On envoie chaque morceau de texte immédiatement
                yield chunk 

        # 4. Log final (en arrière-plan pour ne pas bloquer)
        latency = round(time.time() - t_start, 3)
        asyncio.create_task(self.log_to_langfuse(
            "RAG_Stream", {"query": req.query}, {"answer": "streaming_done"}, 1, {"latency": latency}
        ))

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