#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any

import httpx


# -----------------------
# Defaults (override via CLI)
# -----------------------
DEFAULT_CHUNKING_API_URL = os.getenv("CHUNKING_API_URL", "http://localhost:8002")
DEFAULT_VDB_SERVICE_URL = os.getenv("VDB_SERVICE_URL", "http://localhost:8003")
DEFAULT_COLLECTION = os.getenv("VDB_COLLECTION", "rag_minist_int_hybrid")

MAX_TEXT = 60000


def make_chunk_id(prefix: str = "chunk") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def get_chunks_from_pdf(
    pdf_path: Path,
    *,
    chunking_api_url: str,
    pdf_strategy: str,
    chunk_size_tokens: int,
    min_chunk_chars: int,
    artifacts_path: str | None = None,
    image_model: str | None = None,
    text_model: str | None = None,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "pdf_strategy": pdf_strategy,
        "chunk_size_tokens": chunk_size_tokens,
        "min_chunk_chars": min_chunk_chars,
    }
    if artifacts_path:
        params["artifacts_path"] = artifacts_path
    if image_model:
        params["image_model"] = image_model
    if text_model:
        params["text_model"] = text_model

    with pdf_path.open("rb") as f, httpx.Client(timeout=600.0) as client:
        r = client.post(
            f"{chunking_api_url}/chunks",
            params=params,
            files={"file": (pdf_path.name, f, "application/pdf")},
        )
        r.raise_for_status()
        chunks = r.json()

    return chunks


def ingest_chunks(
    chunks: List[Dict[str, Any]],
    *,
    vdb_service_url: str,
    collection: str,
    default_source: str,
    default_chunk_type: str,
    batch_size: int,
    mode: str,
) -> Dict[str, Any]:
    payload_items: List[Dict[str, Any]] = []

    for chunk in chunks:
        text = (chunk.get("text") or "")[:MAX_TEXT]
        if not text.strip():
            continue

        meta = dict(chunk.get("meta") or {})
        chunk_type = (chunk.get("chunk_type") or "").strip() or default_chunk_type
        meta.setdefault("type", chunk_type)

        payload_items.append({
            "id": chunk.get("id") or make_chunk_id(),
            "text": text[:65535],
            "source": chunk.get("source") or default_source,
            "section_path": chunk.get("section_path", "") or "",
            "section_title": chunk.get("section_title", "") or "",
            "page_no": int(chunk.get("page_no", -1)) if chunk.get("page_no") is not None else -1,
            "chunk_type": chunk_type,
            "meta": meta,
        })

    if not payload_items:
        return {"ok": True, "count": 0, "collection": collection, "mode": mode}

    body = {
        "collection": collection,
        "items": payload_items,
        "batch_size": batch_size,
        "mode": mode,
    }

    with httpx.Client(timeout=600.0) as client:
        r = client.post(f"{vdb_service_url}/upsert", json=body)
        r.raise_for_status()
        return r.json()


def iter_pdfs(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted([p for p in path.rglob("*.pdf") if p.is_file()])
    raise FileNotFoundError(f"Chemin introuvable: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk + ingest un PDF (ou dossier de PDF) via /chunks puis /upsert."
    )
    parser.add_argument("path", help="Chemin vers un PDF ou un dossier contenant des PDFs")
    parser.add_argument("--chunking-url", default=DEFAULT_CHUNKING_API_URL, help="URL API chunking (ex: http://localhost:8000)")
    parser.add_argument("--vdb-url", default=DEFAULT_VDB_SERVICE_URL, help="URL API VDB (ex: http://localhost:8003)")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Nom de la collection Milvus")

    parser.add_argument("--mode", choices=["insert", "upsert"], default="insert", help="insert (plus rapide) ou upsert")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size pour /upsert")

    parser.add_argument("--pdf-strategy", choices=["structure", "semantic"], default="structure", help="Stratégie de chunking PDF")
    parser.add_argument("--chunk-size-tokens", type=int, default=1000, help="Taille des chunks (tokens)")
    parser.add_argument("--min-chunk-chars", type=int, default=200, help="Taille minimale d'un chunk (chars)")

    parser.add_argument("--default-chunk-type", default="doc_chunk", help="chunk_type par défaut")
    parser.add_argument("--source", default=None, help="source forcée (sinon = nom du fichier)")

    # optionnels si tu veux les passer depuis la CLI
    parser.add_argument("--artifacts-path", default=None, help="artifacts_path Docling (optionnel)")
    parser.add_argument("--image-model", default=None, help="Modèle Ollama image (optionnel)")
    parser.add_argument("--text-model", default=None, help="Modèle Ollama texte (optionnel)")

    args = parser.parse_args()

    path = Path(args.path).expanduser().resolve()
    pdfs = iter_pdfs(path)
    if not pdfs:
        print("⚠️ Aucun PDF trouvé.")
        return 1

    total_chunks = 0
    total_inserted = 0

    for pdf in pdfs:
        src = args.source or pdf.name
        print(f"\n📄 {pdf}  (source={src})")

        # 1) chunking
        chunks = get_chunks_from_pdf(
            pdf,
            chunking_api_url=args.chunking_url,
            pdf_strategy=args.pdf_strategy,
            chunk_size_tokens=args.chunk_size_tokens,
            min_chunk_chars=args.min_chunk_chars,
            artifacts_path=args.artifacts_path,
            image_model=args.image_model,
            text_model=args.text_model,
        )
        print(f"✅ Chunking OK | n_chunks={len(chunks)}")
        total_chunks += len(chunks)

        # 2) ingestion
        res = ingest_chunks(
            chunks,
            vdb_service_url=args.vdb_url,
            collection=args.collection,
            default_source=src,
            default_chunk_type=args.default_chunk_type,
            batch_size=args.batch_size,
            mode=args.mode,
        )
        inserted = int(res.get("count", 0))
        total_inserted += inserted
        print(f"✅ Ingestion OK | inserted={inserted} | mode={args.mode}")

    print("\n--- Résumé ---")
    print(f"PDFs traités: {len(pdfs)}")
    print(f"Chunks total: {total_chunks}")
    print(f"Chunks insérés: {total_inserted}")
    print(f"Collection: {args.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
