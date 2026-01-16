from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel

import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT / "utils"))

# --- tiktoken (tokens) ---
try:
    import tiktoken
except Exception:
    tiktoken = None


# avec markdown intégré sans images et avec balise
from utils.chunking.factory import ExtractConfig, UniversalExtractorFactory
from utils.chunking.factory import make_ollama_image_summarizer, make_ollama_text_summarizer

# Tes fonctions chunking (tu peux les mettre dans chunking_pdf.py / chunking_pptx.py)
from utils.chunking.chunks_pdfs_words import chunk_from_extract_result, ChunkingConfig as PdfChunkingConfig
from utils.chunking.chunks_ppt import chunk_pptx_from_res, ChunkingConfig as PptChunkingConfig

from dotenv import load_dotenv
load_dotenv()

DOCLING_MODELS_PATH = os.getenv("DOCLING_MODELS_PATH")
if not DOCLING_MODELS_PATH:
    raise RuntimeError("DOCLING_MODELS_PATH manquant. à rajouter dans .env ou export-le avant de lancer uvicorn.")


# Ici, je suppose que tu as déjà ces fonctions en scope:
# - chunk_from_extract_result(res, cfg)
# - chunk_pptx_from_res(res, cfg)


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
    meta: Dict[str, Any] = {}


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
    # interpolation linéaire
    return int(round(sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])))


def _histogram(values: List[int], bins: int = 30) -> Dict[str, Any]:
    if not values:
        return {"bin_edges": [0, 1], "counts": [0]}

    vmin, vmax = min(values), max(values)
    if vmin == vmax:
        # un seul bin
        return {"bin_edges": [vmin, vmax + 1], "counts": [len(values)]}

    # binning uniforme
    width = (vmax - vmin) / bins
    edges = [vmin + i * width for i in range(bins + 1)]
    counts = [0] * bins

    for v in values:
        idx = int((v - vmin) / width)
        if idx == bins:  # vmax pile
            idx -= 1
        counts[idx] += 1

    # edges en int pour lisibilité
    edges_int = [int(round(e)) for e in edges]
    # évite les doublons dus à l’arrondi
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
    # Config extract
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


def _chunk_by_extension(res: Any, filename: str):
    e = _ext(filename)
    if e in {"pdf", "docx", "doc"}:
        # PDF/DOCX
        return chunk_from_extract_result(res)
    if e in {"pptx", "ppt"}:
        # PPTX
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
):
    # sauvegarde temporaire
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
        # IMPORTANT: on utilise l'extension originale (file.filename), pas tmp_path
        chunks = _chunk_by_extension(res, file.filename)
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
    encoding: str = Query(default="o200k_base", description="Encoding tiktoken (ex: o200k_base, cl100k_base)"),
    bins: int = Query(default=30, ge=1, le=200),
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
        chunks = _chunk_by_extension(res, file.filename)

        texts = [getattr(c, "text", "") or "" for c in chunks]
        texts = [t for t in texts if t.strip()]

        char_counts = [len(t) for t in texts]

        enc = _get_encoding(encoding)
        if enc:
            token_counts = [len(enc.encode(t)) for t in texts]
        else:
            # fallback simple si tiktoken absent
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
