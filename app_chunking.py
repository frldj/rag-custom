## To do :
# - filtre pour ne pas prendre la table des matières en chunks
# - mettre un path pour mettre en json les chunks si nombreux avant de lancer ingestion dans milvus
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel, Field

import re
import httpx


# -------------------------
# Path setup
# -------------------------
PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT / "utils"))

# -------------------------
# tiktoken (tokens)
# -------------------------
try:
    import tiktoken
except Exception:
    tiktoken = None

# -------------------------
# Docling extractor factory
# -------------------------
from utils.chunking.factory import ExtractConfig, UniversalExtractorFactory
from utils.chunking.factory import make_ollama_image_summarizer, make_ollama_text_summarizer

# -------------------------
# Chunkers
# -------------------------
from utils.chunking.chunks_pdfs_words import (
    chunk_dispatch,
    ChunkingConfig as PdfChunkingConfig,
)
from utils.chunking.chunks_ppt import chunk_pptx_from_res, ChunkingConfig as PptChunkingConfig  # noqa: F401

# -------------------------
# Env
# -------------------------
load_dotenv()

DOCLING_MODELS_PATH = os.getenv("DOCLING_MODELS_PATH")
if not DOCLING_MODELS_PATH:
    raise RuntimeError(
        "DOCLING_MODELS_PATH manquant. "
        "Ajoute-le dans .env ou exporte-le avant de lancer uvicorn."
    )
VDB_SERVICE_URL = os.getenv("VDB_SERVICE_URL", "http://localhost:8003")
DEFAULT_COLLECTION = os.getenv("VDB_DEFAULT_COLLECTION", "rag_minist_int_hybrid_v2")
MAX_TEXT = 60000
BATCH_SIZE = 256


app = FastAPI(title="Chunking API", version="0.1.0")


# -------------------------
# Réponses (Pydantic)
# -------------------------
class ChunkOut(BaseModel):
    id: str
    source: str
    text: str
    page_no: Optional[int] = None
    section_title: Optional[str] = None
    section_path: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class ChunkStatsOut(BaseModel):
    n_chunks: int
    encoding: str
    token_counts: List[int]
    char_counts: List[int]
    min_tokens: int
    max_tokens: int
    mean_tokens: float
    median_tokens: int
    p95_tokens: int
    histogram: Dict[str, Any]  # {"bin_edges": [...], "counts": [...]}

class IngestChunksOut(BaseModel):
    ok: bool
    collection: str
    mode: str
    inserted: int
    n_chunks: int
    chunks: List[ChunkOut]



# -------------------------
# Helpers
# -------------------------
def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower().strip(".")


def _get_encoding(name: str):
    if not tiktoken:
        return None
    return tiktoken.get_encoding(name)


def _percentile(sorted_vals: List[int], p: float) -> int:
    # p in [0, 100]
    if not sorted_vals:
        return 0
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return int(round(sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])))


def _histogram(values: List[int], bins: int = 30) -> Dict[str, Any]:
    if not values:
        return {"bin_edges": [0, 1], "counts": [0]}

    vmin, vmax = min(values), max(values)
    if vmin == vmax:
        return {"bin_edges": [vmin, vmax + 1], "counts": [len(values)]}

    width = (vmax - vmin) / bins
    edges = [vmin + i * width for i in range(bins + 1)]
    counts = [0] * bins

    for v in values:
        idx = int((v - vmin) / width)
        if idx == bins:
            idx -= 1
        counts[idx] += 1

    edges_int = [int(round(e)) for e in edges]
    for i in range(1, len(edges_int)):
        if edges_int[i] <= edges_int[i - 1]:
            edges_int[i] = edges_int[i - 1] + 1

    return {"bin_edges": edges_int, "counts": counts}


def _extract_with_docling(
    filepath: str,
    artifacts_path: Optional[str],
    image_model: str,
    text_model: str,
):
    final_path = artifacts_path or DOCLING_MODELS_PATH
    cfg = ExtractConfig(artifacts_path=final_path) if final_path else ExtractConfig()

    image_summarizer = make_ollama_image_summarizer(image_model)
    text_summarizer = make_ollama_text_summarizer(text_model)

    factory = UniversalExtractorFactory(
        config=cfg,
        image_summarizer=image_summarizer,
        text_summarizer=text_summarizer,
    )
    return factory.extract(filepath)


def _to_chunk_out_list(chunks: List[Any]) -> List[ChunkOut]:
    out: List[ChunkOut] = []
    for c in chunks:
        out.append(
            ChunkOut(
                id=getattr(c, "id", ""),
                source=getattr(c, "source", "unknown"),
                text=getattr(c, "text", "") or "",
                page_no=getattr(c, "page_no", None),
                section_title=getattr(c, "section_title", None),
                section_path=getattr(c, "section_path", None),
                meta=getattr(c, "meta", {}) or {},
            )
        )
    return out


def _chunk_by_extension(
    res: Any,
    filename: str,
    *,
    pdf_strategy: str = "structure",
    cfg_pdf: Optional[PdfChunkingConfig] = None,
):
    e = _ext(filename)

    if e in {"pdf", "docx", "doc"}:
        if pdf_strategy not in {"structure", "semantic"}:
            raise HTTPException(
                status_code=400,
                detail="pdf_strategy must be 'structure' or 'semantic'",
            )
        return chunk_dispatch(res, cfg=cfg_pdf, strategy=pdf_strategy)

    if e in {"pptx", "ppt"}:
        return chunk_pptx_from_res(res)

    raise HTTPException(status_code=400, detail=f"Extension non supportée: .{e}")


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chunks", response_model=List[ChunkOut])
async def create_chunks(
    file: UploadFile = File(...),
    artifacts_path: Optional[str] = Query(
        default=None,
        description="Chemin cache modèles Docling (ex: /Users/.../.cache/docling/models)",
    ),
    image_model: str = Query(default="qwen2.5vl:3b"),
    text_model: str = Query(default="llama3.2:3b"),
    # ---- PDF chunking strategy ----
    pdf_strategy: str = Query(default="structure", description="structure | semantic"),
    semantic_scope: str = Query(default="section", description="section | page"),
    semantic_embeddings_model: str = Query(default="qwen3-embedding:0.6b"),
    semantic_ollama_base_url: str = Query(default="http://localhost:11434"),
    semantic_breakpoint_threshold_type: str = Query(default="percentile"),
    chunk_size_tokens: int = Query(default=1000, ge=100, le=8000),
    min_chunk_chars: int = Query(default=200, ge=0, le=5000),
):
    suffix = os.path.splitext(file.filename)[1] or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        res = _extract_with_docling(
            filepath=tmp_path,
            artifacts_path=artifacts_path,
            image_model=image_model,
            text_model=text_model,
        )

        cfg_pdf = PdfChunkingConfig(
            chunk_size_tokens=chunk_size_tokens,
            min_chunk_chars=min_chunk_chars,
            semantic_scope=semantic_scope,
            semantic_embeddings_model=semantic_embeddings_model,
            semantic_ollama_base_url=semantic_ollama_base_url,
            semantic_breakpoint_threshold_type=semantic_breakpoint_threshold_type,
        )

        chunks = _chunk_by_extension(
            res,
            file.filename,
            pdf_strategy=pdf_strategy,
            cfg_pdf=cfg_pdf,
        )
        return _to_chunk_out_list(chunks)

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post("/chunks/stats", response_model=ChunkStatsOut)
async def chunks_stats(
    file: UploadFile = File(...),
    artifacts_path: Optional[str] = Query(default=None),
    image_model: str = Query(default="qwen2.5vl:3b"),
    text_model: str = Query(default="llama3.2:3b"),
    encoding: str = Query(
        default="o200k_base",
        description="Encoding tiktoken (ex: o200k_base, cl100k_base)",
    ),
    bins: int = Query(default=30, ge=1, le=200),
    # ---- PDF chunking strategy ----
    pdf_strategy: str = Query(default="structure", description="structure | semantic"),
    semantic_scope: str = Query(default="section", description="section | page"),
    semantic_embeddings_model: str = Query(default="qwen3-embedding:0.6b"),
    semantic_ollama_base_url: str = Query(default="http://localhost:11434"),
    semantic_breakpoint_threshold_type: str = Query(default="percentile"),
    chunk_size_tokens: int = Query(default=1000, ge=100, le=8000),
    min_chunk_chars: int = Query(default=200, ge=0, le=5000),
):
    suffix = os.path.splitext(file.filename)[1] or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        res = _extract_with_docling(
            filepath=tmp_path,
            artifacts_path=artifacts_path,
            image_model=image_model,
            text_model=text_model,
        )

        cfg_pdf = PdfChunkingConfig(
            chunk_size_tokens=chunk_size_tokens,
            min_chunk_chars=min_chunk_chars,
            semantic_scope=semantic_scope,
            semantic_embeddings_model=semantic_embeddings_model,
            semantic_ollama_base_url=semantic_ollama_base_url,
            semantic_breakpoint_threshold_type=semantic_breakpoint_threshold_type,
        )

        chunks = _chunk_by_extension(
            res,
            file.filename,
            pdf_strategy=pdf_strategy,
            cfg_pdf=cfg_pdf,
        )

        texts = [getattr(c, "text", "") or "" for c in chunks]
        texts = [t for t in texts if t.strip()]

        char_counts = [len(t) for t in texts]

        enc = _get_encoding(encoding)
        if enc:
            token_counts = [len(enc.encode(t)) for t in texts]
        else:
            token_counts = [max(1, len(t) // 4) for t in texts]

        if not token_counts:
            raise HTTPException(status_code=400, detail="Aucun chunk texte exploitable.")

        sorted_tokens = sorted(token_counts)
        n = len(sorted_tokens)
        mean_tokens = sum(sorted_tokens) / n
        median_tokens = sorted_tokens[n // 2]
        p95 = _percentile(sorted_tokens, 95)

        hist = _histogram(token_counts, bins=bins)

        return ChunkStatsOut(
            n_chunks=n,
            encoding=encoding,
            token_counts=token_counts,
            char_counts=char_counts,
            min_tokens=min(sorted_tokens),
            max_tokens=max(sorted_tokens),
            mean_tokens=float(mean_tokens),
            median_tokens=int(median_tokens),
            p95_tokens=int(p95),
            histogram=hist,
        )

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# @app.post("/chunks/ingest", response_model=IngestChunksOut)
# async def chunk_and_ingest(
#     file: UploadFile = File(...),

#     # chunking params
#     artifacts_path: Optional[str] = Query(default=None),
#     image_model: str = Query(default="qwen2.5vl:3b"),
#     text_model: str = Query(default="llama3.2:3b"),
#     pdf_strategy: str = Query(default="structure"),
#     semantic_scope: str = Query(default="section"),
#     semantic_embeddings_model: str = Query(default="qwen3-embedding:0.6b"),
#     semantic_ollama_base_url: str = Query(default="http://localhost:11434"),
#     semantic_breakpoint_threshold_type: str = Query(default="percentile"),
#     chunk_size_tokens: int = Query(default=1000, ge=100, le=8000),
#     min_chunk_chars: int = Query(default=200, ge=0, le=5000),

#     # ingestion params
#     collection: str = Query(default=DEFAULT_COLLECTION),
#     mode: str = Query(default="upsert", description='insert | upsert'),
#     batch_size: int = Query(default=256, ge=1, le=2000),

#     # defaults type/source (comme ton notebook)
#     default_source: str = Query(default="doc"),
#     default_chunk_type: str = Query(default="doc_chunk"),

#     drop_toc: bool = Query(default=True, description="Filtrer la table des matières (si tu implémentes le filtre)"),
# ):
#     suffix = os.path.splitext(file.filename)[1] or ""
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         tmp_path = tmp.name
#         content = await file.read()
#         tmp.write(content)

#     try:
#         # 1) Extract + chunk (pareil que /chunks)
#         res = _extract_with_docling(
#             filepath=tmp_path,
#             artifacts_path=artifacts_path,
#             image_model=image_model,
#             text_model=text_model,
#         )

#         cfg_pdf = PdfChunkingConfig(
#             chunk_size_tokens=chunk_size_tokens,
#             min_chunk_chars=min_chunk_chars,
#             semantic_scope=semantic_scope,
#             semantic_embeddings_model=semantic_embeddings_model,
#             semantic_ollama_base_url=semantic_ollama_base_url,
#             semantic_breakpoint_threshold_type=semantic_breakpoint_threshold_type,
#         )

#         chunks_raw = _chunk_by_extension(
#             res,
#             file.filename,
#             pdf_strategy=pdf_strategy,
#             cfg_pdf=cfg_pdf,
#         )
#         chunks = _to_chunk_out_list(chunks_raw)
#         n_chunks = len(chunks)

#         # (optionnel) filtre TOC — pas implémenté ici
#         # chunks = _filter_chunks(chunks, drop_toc=drop_toc)

#         # 2) Construire les items à envoyer au VDB (logique notebook)
#         items = []
#         for c in chunks:
#             txt = (c.text or "")[:MAX_TEXT]  # si tu veux garder ton MAX_TEXT notebook, sinon enlève
#             if not txt.strip():
#                 continue

#             meta = dict(c.meta or {})
#             # notebook: meta.setdefault("type", obj.get("chunk_type", chunk_type))
#             # ici on set default si absent
#             chunk_type = (meta.get("type") or default_chunk_type).strip()
#             meta.setdefault("type", chunk_type)

#             items.append({
#                 "id": c.id,  # l'id existe déjà côté chunking
#                 "text": txt[:65535],
#                 "source": (c.source or default_source),
#                 "section_path": c.section_path or "",
#                 "section_title": c.section_title or "",
#                 "page_no": int(c.page_no) if c.page_no is not None else -1,
#                 "chunk_type": chunk_type,   # ✅ top-level
#                 "meta": meta,
#             })

#         # 3) Appeler le VDB /upsert
#         payload = {
#             "collection": collection,
#             "items": items,
#             "batch_size": batch_size,
#             "mode": mode,
#         }

#         async with httpx.AsyncClient(timeout=300.0) as client:
#             r = await client.post(f"{VDB_SERVICE_URL}/upsert", json=payload)
#             if r.status_code >= 400:
#                 raise HTTPException(status_code=500, detail={"vdb_error": r.text})
#             data = r.json()

#         inserted = int(data.get("count", 0))

#         # 4) Retourne chunks + stats ingestion
#         return IngestChunksOut(
#             ok=True,
#             collection=collection,
#             mode=mode,
#             inserted=inserted,
#             n_chunks=n_chunks,
#             chunks=chunks,
#         )

#     finally:
#         try:
#             os.remove(tmp_path)
#         except Exception:
#             pass


