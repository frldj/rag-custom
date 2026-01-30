# rag_service.py
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ollama import Client as OllamaClient
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker


# =========================
# Config
# =========================
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLL = os.getenv("MILVUS_COLLECTION", "rag_minist_int_hybrid")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
EMB_MODEL = os.getenv("OLLAMA_EMB_MODEL", "qwen3-embedding:0.6b")

# Rerank remote 
RERANK_URL = os.getenv("RERANK_URL", "http://localhost:8001/rerank")
RERANK_HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=5.0)

# Search / rerank params 
DEFAULT_TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))
DEFAULT_TOP_K_RECALL = int(os.getenv("TOP_K_RECALL", "60"))  
EF_SEARCH = int(os.getenv("EF_SEARCH", "100"))


RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "2000"))
BUILD_CONTEXT_MAX_CHARS = int(os.getenv("BUILD_CONTEXT_MAX_CHARS", "6000"))

# Ollama generate
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "400"))
OLLAMA_TIMEOUT = httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=5.0)


# =========================
# App + state
# =========================
app = FastAPI(title="RAG Service (Milvus Hybrid + Remote Rerank + Ollama)")
milvus: Optional[MilvusClient] = None
ollama_client: Optional[OllamaClient] = None


# =========================
# API models
# =========================
class AskRequest(BaseModel):
    query: str
    top_k_final: int = DEFAULT_TOP_K_FINAL
    top_k_recall: int = DEFAULT_TOP_K_RECALL
    rerank_max_chars: int = RERANK_MAX_CHARS
    build_context_max_chars: int = BUILD_CONTEXT_MAX_CHARS


class ContextItem(BaseModel):
    id: Any
    text: str
    source: Optional[str] = None
    page_no: Optional[int] = None
    section_title: Optional[str] = None
    milvus_score: Optional[float] = None
    rerank_score: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    contexts: List[ContextItem]
    context_prompt: Optional[str] = None


# =========================
# Startup
# =========================
@app.on_event("startup")
def startup():
    global milvus, ollama_client

    milvus = MilvusClient(MILVUS_URI)
    if not milvus.has_collection(COLL):
        raise RuntimeError(f"Collection Milvus inexistante: {COLL}")
    milvus.load_collection(COLL)

    ollama_client = OllamaClient(host=OLLAMA_URL)


# =========================
# Helpers 
# =========================
def emb_text(text: str) -> List[float]:
    assert ollama_client is not None
    return ollama_client.embeddings(model=EMB_MODEL, prompt=text)["embedding"]


def _get_entity_getter(h):
    ent = getattr(h, "entity", None)
    if ent is None:
        return lambda k, d=None: d
    if hasattr(ent, "get"):
        return ent.get
    return lambda k, d=None: getattr(ent, k, d)


def _format_text_for_rerank(section_title: str, text: str) -> str:
    section_title = (section_title or "").strip()
    text = (text or "").strip()
    if section_title:
        return f"{section_title}\n\n{text}"
    return text


def rerank_remote(query: str, passages: List[str]) -> List[float]:
    payload = {"query": query, "passages": passages}
    with httpx.Client(timeout=RERANK_HTTP_TIMEOUT) as client:
        r = client.post(RERANK_URL, json=payload)
        r.raise_for_status()
        data = r.json()
        # attendu: {"scores":[...]}
        return data["scores"]


def rerank_hits_with_bge_remote(
    query: str,
    hits: List[Any],
    max_chars: int,
) -> List[Dict[str, Any]]:
    meta: List[Dict[str, Any]] = []
    passages: List[str] = []

    for h in hits:
        get = _get_entity_getter(h)

        text = get("text") or ""
        section_title = get("section_title") or ""
        passage = _format_text_for_rerank(section_title, text)[:max_chars]

        passages.append(passage)
        meta.append({
            "id": get("id"),
            "page_no": get("page_no"),
            "section_title": section_title,
            "source": get("source"),
            "milvus_score": getattr(h, "distance", None) or getattr(h, "score", None),
            "text": text,
        })

    scores = rerank_remote(query, passages)

    for m, s in zip(meta, scores):
        m["rerank_score"] = float(s)

    meta.sort(key=lambda x: x["rerank_score"], reverse=True)
    return meta


def build_context(reranked: List[Dict[str, Any]], max_chars: int) -> str:
    chunks = []
    total = 0
    for i, r in enumerate(reranked, start=1):
        src = r.get("source") or "unknown"
        page = r.get("page_no")
        title = (r.get("section_title") or "").strip()
        txt = (r.get("text") or "").strip()

        block = f"[{i}] source={src} page={page} title={title}\n{txt}\n"
        if total + len(block) > max_chars:
            break

        chunks.append(block)
        total += len(block)

    return "\n".join(chunks)


# =========================
# Milvus hybrid
# =========================
def hybrid_search(query: str, top_k: int, ef: int) -> List[Any]:
    assert milvus is not None

    qvec = emb_text(query)

    dense_req = AnnSearchRequest(
        data=[qvec],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"ef": ef}},
        limit=top_k,
        expr="",
    )

    sparse_req = AnnSearchRequest(
        data=[query],
        anns_field="sparse",
        param={"metric_type": "BM25"},
        limit=top_k,
        expr="",
    )

    res = milvus.hybrid_search(
        COLL,
        [sparse_req, dense_req],
        RRFRanker(k=60),
        limit=top_k,
        output_fields=["id", "text", "source", "page_no", "section_title"],
        consistency_level="Strong",
    )
    return res


def hybrid_search_with_rerank(
    query: str,
    top_k_final: int,
    top_k_recall: int,
    ef: int,
    rerank_max_chars: int,
) -> List[Dict[str, Any]]:
    res = hybrid_search(query, top_k=top_k_recall, ef=ef)
    hits = res[0]
    reranked = rerank_hits_with_bge_remote(query, hits, max_chars=rerank_max_chars)
    return reranked[:top_k_final]


# =========================
# Ollama generate
# =========================
def ollama_generate(question: str, context: str) -> str:
    prompt = f"""Tu es un assistant RAG. Réponds uniquement à partir du CONTEXTE.
Si l'information n'est pas dans le contexte, dis "Je ne sais pas d'après les sources fournies".

CONTEXTE:
{context}

QUESTION:
{question}

RÉPONSE (en français, concise, avec éventuellement des références [1], [2]...):
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
        },
    }

    with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
        r = client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        r.raise_for_status()
        return (r.json().get("response") or "").strip()


def answer(query: str, top_k_final: int, top_k_recall: int, rerank_max_chars: int, build_context_max_chars: int):
    reranked = hybrid_search_with_rerank(
        query=query,
        top_k_final=top_k_final,
        top_k_recall=top_k_recall,
        ef=EF_SEARCH,
        rerank_max_chars=rerank_max_chars,
    )
    context_prompt = build_context(reranked, max_chars=build_context_max_chars)
    ans = ollama_generate(query, context_prompt)
    return {"answer": ans, "contexts": reranked, "context_prompt": context_prompt}


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "collection": COLL,
        "ollama_model": OLLAMA_MODEL,
        "emb_model": EMB_MODEL,
        "rerank_url": RERANK_URL,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        out = answer(
            query=req.query,
            top_k_final=req.top_k_final,
            top_k_recall=req.top_k_recall,
            rerank_max_chars=req.rerank_max_chars,
            build_context_max_chars=req.build_context_max_chars,
        )
        contexts = [ContextItem(**c) for c in out["contexts"]]
        return AskResponse(answer=out["answer"], contexts=contexts, context_prompt=out["context_prompt"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_service:app", host="0.0.0.0", port=8004, reload=False)
