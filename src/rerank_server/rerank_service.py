# # #export KMP_DUPLICATE_LIB_OK=TRUE
# # #export OMP_NUM_THREADS=1
# # #export MKL_NUM_THREADS=1


# # import os
# # import time
# # import math
# # from typing import List

# # from fastapi import FastAPI, HTTPException
# # from pydantic import BaseModel, Field
# # from FlagEmbedding import FlagReranker

# # from dotenv import load_dotenv

# # load_dotenv()



# # # -------------------------
# # # CONFIG
# # # -------------------------
# # RERANK_MODEL = os.getenv("RERANK_MODEL") #, "BAAI/bge-reranker-base")

# # # Sur CPU/Mac, fp16 peut être inutile : mettre false si soucis/perf
# # USE_FP16 = os.getenv("USE_FP16", "false").lower() == "true"

# # # Garde-fous
# # MAX_PASSAGES = int(os.getenv("MAX_PASSAGES", "256"))
# # MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "2000"))

# # app = FastAPI(title="BGE Rerank HTTP Service")


# # reranker = FlagReranker(RERANK_MODEL, use_fp16=USE_FP16)

# # class RerankRequest(BaseModel):
# #     query: str = Field(..., min_length=1)
# #     passages: List[str] = Field(..., min_items=1)

# # class RerankResponse(BaseModel):
# #     scores: List[float]
# #     model: str
# #     took_ms: int

# # def _truncate(s: str, n: int) -> str:
# #     return (s or "")[:n]

# # def sigmoid(x):
# #     return 1 / (1 + math.exp(-x))

# # @app.get("/health")
# # def health():
# #     return {"ok": True, "model": RERANK_MODEL, "use_fp16": USE_FP16}

# # @app.post("/rerank", response_model=RerankResponse)
# # def rerank(req: RerankRequest):
# #     if len(req.passages) > MAX_PASSAGES:
# #         raise HTTPException(status_code=400, detail=f"Too many passages (>{MAX_PASSAGES})")

# #     query = req.query.strip()
# #     passages = [_truncate(p.strip(), MAX_TEXT_CHARS) for p in req.passages]

# #     pairs = [[query, p] for p in passages]

# #     t0 = time.time()
# #     scores = reranker.compute_score(pairs)  # batch scoring
# #     took_ms = int((time.time() - t0) * 1000)

# #     return RerankResponse(
# #         scores=[float(s) for s in scores],
# #         model=RERANK_MODEL,
# #         took_ms=took_ms,
# #     )


# # export KMP_DUPLICATE_LIB_OK=TRUE
# # export OMP_NUM_THREADS=1
# # export MKL_NUM_THREADS=1

# import os
# import time
# import math
# import logging
# from typing import List
# import asyncio
# from concurrent.futures import ThreadPoolExecutor

# # Créer un pool dédié au CPU-bound task
# executor = ThreadPoolExecutor(max_workers=1)

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from FlagEmbedding import FlagReranker
# from dotenv import load_dotenv

# load_dotenv()

# # --- LOGGING ---
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("reranker")

# # --- CONFIG ---
# RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
# USE_FP16 = os.getenv("USE_FP16", "false").lower() == "true"
# MAX_PASSAGES = int(os.getenv("MAX_PASSAGES", "256"))
# MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "2000"))

# app = FastAPI(title="BGE Rerank HTTP Service")

# # Initialisation
# logger.info(f"Chargement du modèle : {RERANK_MODEL}")
# reranker = FlagReranker(RERANK_MODEL, use_fp16=USE_FP16)

# logger.info("Démarrage du warmup...")
# reranker.compute_score(["warmup", "query"]) 
# logger.info("Modèle prêt et chaud !")

# class RerankRequest(BaseModel):
#     query: str = Field(..., min_length=1)
#     passages: List[str] = Field(..., min_items=1)

# class RerankResponse(BaseModel):
#     scores: List[float]
#     model: str
#     took_ms: int

# def _truncate(s: str, n: int) -> str:
#     return (s or "")[:n]

# def sigmoid(x: float) -> float:
#     """Transforme les logits BGE en probabilité [0, 1]"""
#     try:
#         return 1 / (1 + math.exp(-x))
#     except OverflowError:
#         return 1.0 if x > 0 else 0.0

# @app.get("/health")
# def health():
#     return {"ok": True, "model": RERANK_MODEL}

# # @app.post("/rerank", response_model=RerankResponse)
# # def rerank(req: RerankRequest):
# #     if len(req.passages) > MAX_PASSAGES:
# #         raise HTTPException(status_code=400, detail=f"Too many passages (>{MAX_PASSAGES})")

# #     query = req.query.strip()
# #     passages = [_truncate(p.strip(), MAX_TEXT_CHARS) for p in req.passages]
# #     pairs = [[query, p] for p in passages]

# #     t0 = time.time()
# #     try:
# #         # Calcul des scores bruts (logits)
# #         scores_raw = reranker.compute_score(pairs)
        
# #         if isinstance(scores_raw, (float, int)):
# #             scores_raw = [scores_raw]
        
# #         # On renvoie les scores SANS transformation
# #         final_scores = [float(s) for s in scores_raw]
        
# #     except Exception as e:
# #         logger.error(f"Erreur Reranking: {e}")
# #         raise HTTPException(status_code=500, detail=str(e))

# #     took_ms = int((time.time() - t0) * 1000)
# #     return RerankResponse(
# #         scores=final_scores,
# #         model=RERANK_MODEL,
# #         took_ms=took_ms,
# #     )

# @app.post("/rerank", response_model=RerankResponse)
# async def rerank(req: RerankRequest):
#     loop = asyncio.get_event_loop()
    
#     # Préparation des données
#     query = req.query.strip()
#     passages = [_truncate(p.strip(), MAX_TEXT_CHARS) for p in req.passages]
#     pairs = [[query, p] for p in passages]

#     t0 = time.time()
#     try:
#         # Exécution dans le pool pour ne pas bloquer l'event loop asynchrone
#         scores_raw = await loop.run_in_executor(executor, reranker.compute_score, pairs)
        
#         if isinstance(scores_raw, (float, int)):
#             scores_raw = [scores_raw]
        
#         final_scores = [float(s) for s in scores_raw]
        
#     except Exception as e:
#         logger.error(f"Erreur Reranking: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

#     took_ms = int((time.time() - t0) * 1000)
#     return RerankResponse(scores=final_scores, model=RERANK_MODEL, took_ms=took_ms)

import os
import time
import math
import logging
import asyncio
import torch
from typing import List
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from FlagEmbedding import FlagReranker
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reranker")

# Optimisation M2 : Détection du GPU Metal (Tahoe supporte MPS partiellement, c'est plus rapide)
device = "mps" if torch.backends.mps.is_available() else "cpu"
RERANK_MODEL = os.getenv("RERANK_MODEL") 
USE_FP16 = True if device == "mps" else False # FP16 parfait pour le M2

executor = ThreadPoolExecutor(max_workers=1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="BGE Rerank M2 Optimized")


logger.info(f"Chargement du modèle sur {device.upper()} : {RERANK_MODEL}")
# On force le chargement sur le bon device
reranker = FlagReranker(RERANK_MODEL, use_fp16=USE_FP16)

logger.info("Warmup du modèle...")
reranker.compute_score(["warmup", "query"]) 
logger.info("Modèle prêt !")

class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    passages: List[str] = Field(..., min_length=1)

class RerankResponse(BaseModel):
    scores: List[float]
    model: str
    took_ms: int

@app.get("/health")
def health():
    return {"ok": True, "model": RERANK_MODEL}

@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    loop = asyncio.get_event_loop()
    
    query = req.query.strip()
    # On limite à 2000 caractères par passage pour protéger la RAM du Mac Air
    passages = [p[:2000].strip() for p in req.passages]
    pairs = [[query, p] for p in passages]

    t0 = time.time()
    try:
        # Exécution sans bloquer l'event loop
        scores_raw = await loop.run_in_executor(executor, reranker.compute_score, pairs)
        
        if isinstance(scores_raw, (float, int)):
            scores_raw = [scores_raw]
        
        final_scores = [float(s) for s in scores_raw]
        
    except Exception as e:
        logger.error(f"Erreur Reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    took_ms = int((time.time() - t0) * 1000)
    return RerankResponse(scores=final_scores, model=RERANK_MODEL, took_ms=took_ms)

