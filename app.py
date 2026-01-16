from __future__ import annotations

import os
import sys
import json
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel, Field

# -------------------------
# PATH / imports projet
# -------------------------
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

# Chunking
from utils.chunking.chunks_pdfs_words import (
    chunk_from_extract_result,
    ChunkingConfig as PdfChunkingConfig,
)
from utils.chunking.chunks_ppt import (
    chunk_pptx_from_res,
    ChunkingConfig as PptChunkingConfig,
)

app = FastAPI(title="Chunking API", version="0.2.0")

# -------------------------
# Stockage local des chunks
# -------------------------
CHUNKS_DIR = Path("data/chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Modèles (Pydantic)
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


class CreateChunksOut(BaseModel):
    job_id: str
    n_chunks: int
    storage_path: str
    # Si tu veux éviter de renvoyer tous les chunks (payload lourd),
    # passe return_chunks=false dans l'endpoint.
    chunks: Optional[List[ChunkOut]] = None


# -------------------------
# Helpers
# -------------------------
def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower().strip(".")


def _get_encoding(name: str):
    if not tiktoken:
        return None
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        return None


def _percentile(sorted_vals: List[int], p: float) -> int:
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


def _make_job_id(filename: str, content: bytes) -> str:
    h = hashlib.sha256()
    h.update(filename.encode("utf-8", errors="ignore"))
    h.update(content)
    return h.hexdigest()[:16]


def _save_chunks(job_id: str, chunks_out: List[ChunkOut]) -> Path:
    job_dir = CHUNKS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_path = job_dir / "chunks.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in chunks_out], f, ensure_ascii=False, indent=2)
    return out_path


def _load_chunks(job_id: str) -> List[Dict[str, Any]]:
    path = CHUNKS_DIR / job_id / "chunks.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="job_id inconnu (chunks.json introuvable)")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_with_docling(
    filepath: str,
    artifacts_path: Optional[str],
    image_model: str,
    text_model: str,
):
    if artifacts_path:
        cfg = ExtractConfig(artifacts_path=artifacts_path)
    else:
        cfg = ExtractConfig()

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
        # Tu peux passer un cfg custom ici si besoin
        # cfg = PdfChunkingConfig(chunk_size_tokens=1000, min_chunk_chars=200)
        # return chunk_from_extract_result(res, cfg=cfg)
        return chunk_from_extract_result(res)
    if e in {"pptx", "ppt"}:
        # idem cfg custom
        # cfg = PptChunkingConfig(min_chunk_chars=100)
        # return chunk_pptx_from_res(res, cfg=cfg)
        return chunk_pptx_from_res(res)
    raise HTTPException(status_code=400, detail=f"Extension non supportée: .{e}")


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chunks", response_model=CreateChunksOut)
async def create_chunks(
    file: UploadFile = File(...),
    artifacts_path: Optional[str] = Query(
        default=None,
        description="Chemin cache modèles Docling (ex: /Users/.../.cache/docling/models)",
    ),
    image_model: str = Query(default="qwen2.5vl:3b"),
    text_model: str = Query(default="llama3.2:3b"),
    return_chunks: bool = Query(default=True, description="Si false, ne renvoie pas la liste complète des chunks (payload léger)"),
):
    suffix = os.path.splitext(file.filename)[1] or ""
    content = await file.read()
    job_id = _make_job_id(file.filename, content)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(content)

    try:
        res = _extract_with_docling(
            filepath=tmp_path,
            artifacts_path=artifacts_path,
            image_model=image_model,
            text_model=text_model,
        )

        chunks = _chunk_by_extension(res, file.filename)
        chunks_out = _to_chunk_out_list(chunks)

        storage_path = _save_chunks(job_id, chunks_out)

        return CreateChunksOut(
            job_id=job_id,
            n_chunks=len(chunks_out),
            storage_path=str(storage_path),
            chunks=chunks_out if return_chunks else None,
        )
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.get("/chunks/{job_id}/stats", response_model=ChunkStatsOut)
def chunks_stats_by_job(
    job_id: str,
    encoding: str = Query(default="o200k_base", description="Encoding tiktoken (ex: o200k_base, cl100k_base)"),
    bins: int = Query(default=30, ge=1, le=200),
):
    rows = _load_chunks(job_id)
    texts = [r.get("text", "") for r in rows if (r.get("text") or "").strip()]
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
