#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

import httpx


DEFAULT_CHUNKING_API_URL = os.getenv("CHUNKING_API_URL", "http://localhost:8002")
DEFAULT_VDB_SERVICE_URL = os.getenv("VDB_SERVICE_URL", "http://localhost:8003")
DEFAULT_COLLECTION = os.getenv("VDB_COLLECTION", "finetuning")

# On garde volontairement bas pour éviter de faire exploser Ollama/Milvus sur des gros chunks
MAX_TEXT_DEFAULT = 8000


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def make_doc_id(pdf_path: Path, root: Path) -> str:
    rel = str(pdf_path.relative_to(root)).replace("\\", "/")
    return f"doc_{sha1(rel)[:16]}"


def make_chunk_id(doc_id: str, page_no: int, chunk_index: int) -> str:
    return f"{doc_id}_p{page_no}_c{chunk_index}"


def batched(seq: List[Dict[str, Any]], n: int) -> Iterable[List[Dict[str, Any]]]:
    if n <= 0:
        raise ValueError("http_batch_size must be > 0")
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def restart_milvus(cmd: str) -> None:
    print(f"🧯 Restart Milvus: {cmd}")
    # check=False: on ne casse pas le run si la commande renvoie un code != 0
    subprocess.run(cmd, shell=True, check=False)


def wait_vdb_ready(
    vdb_url: str,
    *,
    timeout_s: int = 180,
    sleep_s: float = 2.0,
    health_timeout_s: float = 5.0,
    auto_restart: bool = False,
    restart_cmd: str = "docker compose up -d standalone",
) -> None:
    """
    Attend que /health réponde et que milvus_ok==True.
    Si /health timeout et auto_restart=True, tente 1 restart Milvus puis continue d'attendre.
    """
    t0 = time.time()
    restarted = False
    last_err: Optional[str] = None

    while True:
        try:
            r = httpx.get(f"{vdb_url}/health", timeout=health_timeout_s)
            if r.status_code == 200:
                j = r.json()
                if j.get("milvus_ok") is True:
                    return
                last_err = f"milvus_ok={j.get('milvus_ok')} milvus_err={j.get('milvus_err')}"
            else:
                last_err = f"health status_code={r.status_code}"

        except httpx.ReadTimeout as e:
            last_err = f"ReadTimeout({e})"
            if auto_restart and not restarted:
                restarted = True
                restart_milvus(restart_cmd)
                time.sleep(5.0)

        except Exception as e:
            last_err = repr(e)

        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"VDB pas prêt après {timeout_s}s. Dernière erreur: {last_err}")

        time.sleep(sleep_s)


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
        return r.json()


def build_payload_items(
    chunks: List[Dict[str, Any]],
    *,
    doc_id: str,
    file_path: str,
    source: str,
    default_chunk_type: str,
    max_text: int,
) -> List[Dict[str, Any]]:
    payload_items: List[Dict[str, Any]] = []
    chunk_index = 0

    for chunk in chunks:
        text = (chunk.get("text") or "")[:max_text]
        if not text.strip():
            continue

        page_no = int(chunk.get("page_no", -1)) if chunk.get("page_no") is not None else -1
        chunk_type = (chunk.get("chunk_type") or "").strip() or default_chunk_type

        meta = dict(chunk.get("meta") or {})
        meta.setdefault("type", chunk_type)

        # training-friendly metadata
        meta["doc_id"] = doc_id
        meta["paper_id"] = doc_id
        meta["file_path"] = file_path
        meta["source_file"] = source
        meta["chunk_index"] = chunk_index
        meta["page_no"] = page_no

        cid = chunk.get("id") or make_chunk_id(doc_id, page_no, chunk_index)

        payload_items.append(
            {
                "id": cid,
                "text": text[:65535],
                "source": source,
                "section_path": chunk.get("section_path", "") or "",
                "section_title": chunk.get("section_title", "") or "",
                "page_no": page_no,
                "chunk_type": chunk_type,
                "meta": meta,
            }
        )

        chunk_index += 1

    return payload_items


def post_with_retry(
    *,
    url: str,
    json_body: Dict[str, Any],
    vdb_url_for_health: str,
    attempts: int,
    base_sleep_s: float,
    health_timeout_s: float,
    health_wait_s: int,
    auto_restart_milvus: bool,
    milvus_restart_cmd: str,
    client: httpx.Client,
) -> Dict[str, Any]:
    """
    POST robuste:
      - tente le POST
      - si timeout/5xx/erreur réseau => attend /health (avec possible restart Milvus) puis retry avec backoff
    """
    last_exc: Optional[Exception] = None

    for i in range(1, attempts + 1):
        try:
            r = client.post(url, json=json_body)
            r.raise_for_status()
            return r.json()

        except Exception as e:
            last_exc = e
            print(f"⚠️ POST failed (attempt {i}/{attempts}): {repr(e)}")
            sleep_s = base_sleep_s * i
            print(f"   -> retry in {sleep_s:.1f}s (and waiting for /health)")

            try:
                wait_vdb_ready(
                    vdb_url_for_health,
                    timeout_s=health_wait_s,
                    sleep_s=2.0,
                    health_timeout_s=health_timeout_s,
                    auto_restart=auto_restart_milvus,
                    restart_cmd=milvus_restart_cmd,
                )
                print("✅ VDB ready")
            except Exception as w:
                print(f"⚠️ Health wait failed: {repr(w)}")

            time.sleep(sleep_s)

    raise RuntimeError(f"POST failed after {attempts} attempts. Last error: {repr(last_exc)}") from last_exc


def ingest_chunks_insert_fast(
    chunks: List[Dict[str, Any]],
    *,
    vdb_service_url: str,
    collection: str,
    source: str,
    doc_id: str,
    file_path: str,
    default_chunk_type: str,
    batch_size: int,
    http_batch_size: int,
    max_text: int,
    flush: bool,
    retry_attempts: int,
    retry_base_sleep_s: float,
    health_timeout_s: float,
    health_wait_s: int,
    auto_restart_milvus: bool,
    milvus_restart_cmd: str,
) -> Dict[str, Any]:
    payload_items = build_payload_items(
        chunks,
        doc_id=doc_id,
        file_path=file_path,
        source=source,
        default_chunk_type=default_chunk_type,
        max_text=max_text,
    )

    if not payload_items:
        return {"ok": True, "count": 0, "collection": collection, "mode": "insert_fast"}

    parts = list(batched(payload_items, http_batch_size))
    total = 0

    print(f"⏳ Ingestion: {len(payload_items)} chunks -> {len(parts)} requêtes /insert_fast (http_batch_size={http_batch_size})")

    with httpx.Client(timeout=600.0) as client:
        for i, part in enumerate(parts, start=1):
            is_last = (i == len(parts))
            body: Dict[str, Any] = {
                "collection": collection,
                "items": part,
                "batch_size": batch_size,
                "max_text": max_text,
                "default_source": source,
                "default_chunk_type": default_chunk_type,
                # flush uniquement à la toute fin du PDF
                "flush": (flush and is_last),
            }

            print(f"   -> POST /insert_fast {i}/{len(parts)} | items={len(part)} | flush={body['flush']}")
            out = post_with_retry(
                url=f"{vdb_service_url}/insert_fast",
                json_body=body,
                vdb_url_for_health=vdb_service_url,
                attempts=retry_attempts,
                base_sleep_s=retry_base_sleep_s,
                health_timeout_s=health_timeout_s,
                health_wait_s=health_wait_s,
                auto_restart_milvus=auto_restart_milvus,
                milvus_restart_cmd=milvus_restart_cmd,
                client=client,
            )
            total += int(out.get("count", 0))

    return {"ok": True, "count": total, "collection": collection, "mode": "insert_fast"}


def iter_pdfs(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() == ".pdf":
        return [path]
    if path.is_dir():
        return sorted([p for p in path.rglob("*.pdf") if p.is_file()])
    raise FileNotFoundError(f"Chemin introuvable: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk + ingest un dossier de PDFs via /chunks puis /insert_fast, robuste (retry + health + auto-restart Milvus)."
    )
    parser.add_argument("path", help="Chemin vers un PDF ou un dossier contenant des PDFs (ex: ./pdfs)")
    parser.add_argument("--chunking-url", default=DEFAULT_CHUNKING_API_URL)
    parser.add_argument("--vdb-url", default=DEFAULT_VDB_SERVICE_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)

    parser.add_argument("--pdf-strategy", choices=["structure", "semantic"], default="structure")
    parser.add_argument("--chunk-size-tokens", type=int, default=300)
    parser.add_argument("--min-chunk-chars", type=int, default=200)

    parser.add_argument("--default-chunk-type", default="doc_chunk")
    parser.add_argument("--source", default=None, help="Force source (sinon = nom du fichier)")

    parser.add_argument("--artifacts-path", default=None)
    parser.add_argument("--image-model", default=None)
    parser.add_argument("--text-model", default=None)

    parser.add_argument("--batch-size", type=int, default=32, help="Batch interne côté VDB (Milvus insert)")
    parser.add_argument("--http-batch-size", type=int, default=16, help="Nb de chunks envoyés par requête /insert_fast")
    parser.add_argument("--max-text", type=int, default=MAX_TEXT_DEFAULT)

    parser.add_argument("--sleep-between-pdfs", type=float, default=2.0, help="Sleep (s) entre PDFs")
    parser.add_argument("--flush", action="store_true", help="Flush Milvus à la fin de chaque PDF (plus stable, plus lent)")

    # retry / health
    parser.add_argument("--retry-attempts", type=int, default=5)
    parser.add_argument("--retry-base-sleep", type=float, default=2.0)
    parser.add_argument("--health-timeout", type=float, default=5.0)
    parser.add_argument("--health-wait", type=int, default=180)

    # auto restart
    parser.add_argument(
        "--auto-restart-milvus",
        action="store_true",
        help="Si /health timeout ou VDB pas prêt, lance 'docker compose up -d standalone' puis retente.",
    )
    parser.add_argument(
        "--milvus-restart-cmd",
        default="docker compose up -d standalone",
        help="Commande shell pour redémarrer Milvus (standalone).",
    )

    args = parser.parse_args()

    root = Path(args.path).expanduser().resolve()
    pdfs = iter_pdfs(root)
    if not pdfs:
        print("⚠️ Aucun PDF trouvé.")
        return 1

    root_dir = root if root.is_dir() else root.parent

    print("Checking VDB health...")
    wait_vdb_ready(
        args.vdb_url,
        timeout_s=args.health_wait,
        sleep_s=2.0,
        health_timeout_s=args.health_timeout,
        auto_restart=args.auto_restart_milvus,
        restart_cmd=args.milvus_restart_cmd,
    )
    print("✅ VDB ready")

    total_chunks = 0
    total_inserted = 0

    for idx, pdf in enumerate(pdfs, start=1):
        src = args.source or pdf.name
        doc_id = make_doc_id(pdf, root_dir)
        rel_path = str(pdf.relative_to(root_dir)).replace("\\", "/")

        print(f"\n[{idx}/{len(pdfs)}] 📄 {rel_path}")
        print(f"   source={src}")
        print(f"   doc_id={doc_id}")

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

        if args.sleep_between_pdfs and args.sleep_between_pdfs > 0:
            time.sleep(args.sleep_between_pdfs)

        res = ingest_chunks_insert_fast(
            chunks,
            vdb_service_url=args.vdb_url,
            collection=args.collection,
            source=src,
            doc_id=doc_id,
            file_path=rel_path,
            default_chunk_type=args.default_chunk_type,
            batch_size=args.batch_size,
            http_batch_size=args.http_batch_size,
            max_text=args.max_text,
            flush=args.flush,
            retry_attempts=args.retry_attempts,
            retry_base_sleep_s=args.retry_base_sleep,
            health_timeout_s=args.health_timeout,
            health_wait_s=args.health_wait,
            auto_restart_milvus=args.auto_restart_milvus,
            milvus_restart_cmd=args.milvus_restart_cmd,
        )

        inserted = int(res.get("count", 0))
        total_inserted += inserted
        print(f"✅ Ingestion OK | inserted={inserted} | endpoint=insert_fast")

    print("\n--- Résumé ---")
    print(f"PDFs traités: {len(pdfs)}")
    print(f"Chunks total: {total_chunks}")
    print(f"Chunks insérés: {total_inserted}")
    print(f"Collection: {args.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
