#!/usr/bin/env python3
# python ingestor.py ./pdfs --strategy hybrid --max-chars 1200
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from dotenv import load_dotenv
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

# -----------------------
# Configuration par défaut
# -----------------------
DEFAULT_CHUNKING_API_URL = os.getenv("CHUNKING_API_URL") #, "http://localhost:8002")
DEFAULT_VDB_SERVICE_URL = os.getenv("VDB_SERVICE_URL") #, "http://localhost:8003")
DEFAULT_COLLECTION = os.getenv("VDB_COLLECTION") #, "rag_minist_int_hybrid_custom_embedding_infloat_v2")
DEFAULT_STATE_DB = os.getenv("INGEST_STATE_DB", "./ingestion_state.db")

MIME_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ingestor")

# -----------------------
# Gestion de l'état (SQLite)
# -----------------------
def init_state_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingested_files (
            file_path TEXT PRIMARY KEY,
            md5 TEXT NOT NULL,
            collection TEXT NOT NULL,
            status TEXT NOT NULL,
            inserted_count INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def is_already_ingested(conn: sqlite3.Connection, file_path: str, md5: str, collection: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM ingested_files WHERE file_path = ? AND md5 = ? AND collection = ? AND status = 'success'",
        (file_path, md5, collection),
    )
    return cur.fetchone() is not None

def save_state(conn: sqlite3.Connection, file_path: str, md5: str, collection: str, status: str, count: int):
    conn.execute(
        """INSERT INTO ingested_files (file_path, md5, collection, status, inserted_count, updated_at)
           VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
           ON CONFLICT(file_path) DO UPDATE SET md5=excluded.md5, status=excluded.status, inserted_count=excluded.inserted_count, updated_at=CURRENT_TIMESTAMP""",
        (file_path, md5, collection, status, count)
    )
    conn.commit()

# -----------------------
# Appels API Chunker (FastAPI)
# -----------------------
def get_chunks_from_file(
    file_path: Path, 
    chunking_api_url: str, 
    max_chars: int, 
    min_chars: int,
    strategy: str 
) -> List[Dict[str, Any]]:
    """Appelle l'API FastAPI pour découper le fichier avec les paramètres choisis."""
    params = {
        "max_chars": max_chars,
        "min_chars": min_chars,
        "strategy": strategy 
    }
    mime_type = MIME_TYPES.get(file_path.suffix.lower(), "application/octet-stream")
    
    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f, mime_type)}
        with httpx.Client(timeout=600.0) as client:
            response = client.post(f"{chunking_api_url}/chunks", params=params, files=files)
            response.raise_for_status()
            data = response.json()
            return data.get("chunks", [])

# -----------------------
# Appels API VDB (Upsert)
# -----------------------
# def ingest_chunks_to_vdb(
#     chunks: List[Dict[str, Any]], 
#     vdb_url: str, 
#     collection: str, 
#     source_name: str,
#     batch_size: int,
#     mode: str
# ) -> int:
#     payload_items = []
#     for chunk in chunks:
#         payload_items.append({
#             "id": chunk.get("id") or str(uuid.uuid4()),
#             "text": chunk.get("text", ""),
#             "source": source_name,
#             "page_no": chunk.get("page_no", -1),
#             "meta": chunk.get("meta", {}),
#             "chunk_type": chunk.get("meta", {}).get("strategy", "text")
#         })

#     if not payload_items:
#         return 0

#     body = {
#         "collection": collection,
#         "items": payload_items,
#         "batch_size": batch_size,
#         "mode": mode
#     }

#     with httpx.Client(timeout=600.0) as client:
#         response = client.post(f"{vdb_url}/upsert", json=body)
#         response.raise_for_status()
#         return int(response.json().get("count", 0))

def ingest_chunks_to_vdb(
    chunks: List[Dict[str, Any]], 
    vdb_url: str, 
    collection: str, 
    source_name: str,
    batch_size: int,
    mode: str
) -> int:
    payload_items = []
    for chunk in chunks:
        payload_items.append({
            "id": chunk.get("id") or str(uuid.uuid4()),
            "text": chunk.get("text", ""),
            "source": source_name,
            "page_no": chunk.get("page_no", -1),
            "meta": chunk.get("meta", {}),
            "chunk_type": chunk.get("meta", {}).get("strategy", "text")
        })

    if not payload_items:
        return 0

    total_inserted = 0
    
    # ---- découpe en lots HTTP ---
    for i in range(0, len(payload_items), batch_size):
        batch = payload_items[i : i + batch_size]
        
        body = {
            "collection": collection,
            "items": batch,
            "batch_size": batch_size,
            "mode": mode
        }

        try:
            with httpx.Client(timeout=600.0) as client:
                response = client.post(f"{vdb_url}/upsert", json=body)
                response.raise_for_status()
                total_inserted += int(response.json().get("count", 0))
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 413:
                logger.error(f"  [413] Le batch de taille {len(batch)} est trop lourd pour le serveur.")
            raise e

    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="Ingestion globale : Fichiers -> Chunker -> VDB")
    parser.add_argument("path", help="Fichier ou dossier à traiter")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--strategy", choices=["hybrid", "recursive"], default="hybrid", help="Stratégie de découpage")
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument("--min-chars", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--mode", choices=["insert", "upsert"], default="upsert")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--force", action="store_true", help="Forcer l'ingestion même si déjà fait")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    state_conn = init_state_db(DEFAULT_STATE_DB)
    
    extensions = set(MIME_TYPES.keys())
    if root.is_file():
        files = [root] if root.suffix.lower() in extensions else []
    else:
        pattern = "**/*" if args.recursive else "*"
        files = [p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in extensions]

    logger.info(f"Trouvé {len(files)} fichiers. Stratégie : {args.strategy}")

    for file_path in sorted(files):
        md5 = file_md5(file_path)
        rel_path = str(file_path.relative_to(root.parent))

        if not args.force and is_already_ingested(state_conn, str(file_path), md5, args.collection):
            logger.info(f"Ignoré (déjà ingéré) : {file_path.name}")
            continue

        try:
            logger.info(f"Traitement : {file_path.name}")
            
            # 1. Chunking avec passage de la stratégie
            chunks = get_chunks_from_file(
                file_path, 
                DEFAULT_CHUNKING_API_URL, 
                args.max_chars, 
                args.min_chars,
                args.strategy 
            )
            logger.info(f"  -> {len(chunks)} chunks générés via {args.strategy}.")

            # 2. Ingestion VDB
            inserted = ingest_chunks_to_vdb(
                chunks, 
                DEFAULT_VDB_SERVICE_URL, 
                args.collection, 
                rel_path,
                args.batch_size,
                args.mode
            )
            
            save_state(state_conn, str(file_path), md5, args.collection, "success", inserted)
            logger.info(f"  -> {inserted} vecteurs insérés dans {args.collection}.")

        except Exception as e:
            logger.error(f"Erreur sur {file_path.name} : {e}")
            save_state(state_conn, str(file_path), md5, args.collection, "error", 0)

    state_conn.close()
    logger.info("Ingestion terminée.")

if __name__ == "__main__":
    main()