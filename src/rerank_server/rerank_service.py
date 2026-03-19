#export KMP_DUPLICATE_LIB_OK=TRUE
#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1


import os
import time
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from FlagEmbedding import FlagReranker

from dotenv import load_dotenv

load_dotenv()



# -------------------------
# CONFIG
# -------------------------
RERANK_MODEL = os.getenv("RERANK_MODEL") #, "BAAI/bge-reranker-base")

# Sur CPU/Mac, fp16 peut être inutile : mettre false si soucis/perf
USE_FP16 = os.getenv("USE_FP16", "false").lower() == "true"

# Garde-fous
MAX_PASSAGES = int(os.getenv("MAX_PASSAGES", "256"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "2000"))

app = FastAPI(title="BGE Rerank HTTP Service")


reranker = FlagReranker(RERANK_MODEL, use_fp16=USE_FP16)

class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    passages: List[str] = Field(..., min_items=1)

class RerankResponse(BaseModel):
    scores: List[float]
    model: str
    took_ms: int

def _truncate(s: str, n: int) -> str:
    return (s or "")[:n]

@app.get("/health")
def health():
    return {"ok": True, "model": RERANK_MODEL, "use_fp16": USE_FP16}

@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    if len(req.passages) > MAX_PASSAGES:
        raise HTTPException(status_code=400, detail=f"Too many passages (>{MAX_PASSAGES})")

    query = req.query.strip()
    passages = [_truncate(p.strip(), MAX_TEXT_CHARS) for p in req.passages]

    pairs = [[query, p] for p in passages]

    t0 = time.time()
    scores = reranker.compute_score(pairs)  # batch scoring
    took_ms = int((time.time() - t0) * 1000)

    return RerankResponse(
        scores=[float(s) for s in scores],
        model=RERANK_MODEL,
        took_ms=took_ms,
    )

