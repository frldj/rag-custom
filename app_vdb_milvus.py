# app_vdb_milvus.py
# Run:
#   uvicorn app_vdb_milvus:app --host 0.0.0.0 --port 8003
#
# Deps:
#   pip install fastapi uvicorn pymilvus pydantic ollama

from __future__ import annotations

import hashlib
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)
from pymilvus.exceptions import MilvusException

from ollama import Client as OllamaClient

JsonDict = Dict[str, Any]

# =========================
# Embeddings (Ollama)
# =========================
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_EMB_MODEL = "qwen3-embedding:0.6b"
MAX_EMB_CHARS = 8000

# Milvus
MILVUS_URI = "http://localhost:19530"

ollama_client = OllamaClient(host=OLLAMA_HOST)


def emb_text(text: str) -> List[float]:
    text = (text or "")[:MAX_EMB_CHARS]
    out = ollama_client.embeddings(model=OLLAMA_EMB_MODEL, prompt=text)
    vec = out.get("embedding")
    if not isinstance(vec, list) or not vec:
        raise RuntimeError("Embedding Ollama invalide (liste vide / mauvais format).")
    return vec


def get_embed_dim() -> int:
    return len(emb_text("ping"))


EMBED_DIM = get_embed_dim()

# =========================
# Helpers
# =========================
def safe_primary_id(raw: str, *, max_len: int = 512) -> str:
    """
    Milvus VARCHAR PK est limité à max_len.
    Si l'id dépasse, on le transforme en id stable via hash.
    """
    s = (raw or "").strip()
    if not s:
        return ""
    if len(s) <= max_len:
        return s
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    keep = max_len - 1 - len(h)
    return (s[:keep] + "_" + h)[:max_len]


# =========================
# Milvus Adapter (Hybrid + schema v2 strict)
# =========================
@dataclass
class MilvusHybridConfig:
    uri: str = MILVUS_URI
    collection: str = "rag_minist_int_hybrid"
    embed_dim: int = EMBED_DIM

    # lifecycle
    drop_if_exists: bool = False
    consistency_level: str = "Strong"

    # BM25 analyzer
    analyzer_type: str = "english"  # mets "french" si supporté chez toi

    # HNSW params
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    metric_dense: str = "COSINE"

    # RRF
    rrf_k: int = 60

    # load robustness
    load_retries: int = 10
    load_sleep_s: float = 1.0


class MilvusHybridVDB:
    """
    Collection HYBRIDE avec schéma FIXE "v2":
      id, vector, text, source, section_path, section_title, page_no, chunk_type, sparse
    + Function BM25 (text->sparse) + index HNSW + sparse inverted.

    - ensure_collection(): create schema + indexes + load
    - insert(): insert "classique" (envoie au service des dicts sans vector)
    - insert_fast(): insert "comme notebook" (calcule vector + insert + flush)
    - upsert(): delete(ids) + insert()
    - search_dense(), search_hybrid()
    """

    def __init__(
        self,
        cfg: MilvusHybridConfig,
        embed_fn: Callable[[str], List[float]],
        client: Optional[MilvusClient] = None,
    ):
        self.cfg = cfg
        self.embed_fn = embed_fn
        self.client = client or MilvusClient(uri=cfg.uri)

    # ---------- Connection / load helpers ----------
    def _reconnect(self) -> None:
        self.client = MilvusClient(uri=self.cfg.uri)

    def _load_with_retry(self, collection_name: str) -> None:
        last_err: Exception | None = None

        for _ in range(self.cfg.load_retries):
            try:
                self.client.load_collection(collection_name)
                return
            except MilvusException as e:
                last_err = e
                msg = str(e)

                if "service resource insufficient" in msg or "currentStreamingNode=0" in msg:
                    time.sleep(self.cfg.load_sleep_s)
                    self._reconnect()
                    continue

                raise

        raise RuntimeError(
            f"Impossible de load '{collection_name}' après {self.cfg.load_retries} tentatives: {last_err}"
        ) from last_err

    # ---------- Collection lifecycle ----------
    def ensure_collection(self) -> None:
        name = self.cfg.collection

        if self.client.has_collection(name):
            if self.cfg.drop_if_exists:
                self.client.drop_collection(name)
            else:
                self._load_with_retry(name)
                return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)

        # PK + dense
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.cfg.embed_dim)

        # texte indexable (BM25)
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": self.cfg.analyzer_type},
            enable_match=True,
        )

        # champs v2
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="section_path", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="section_title", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="page_no", datatype=DataType.INT64)
        schema.add_field(field_name="chunk_type", datatype=DataType.VARCHAR, max_length=64)

        # sparse output du BM25
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # Function BM25 : text -> sparse
        schema.add_function(
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names="sparse",
            )
        )

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="dense_index",
            index_type="HNSW",
            metric_type=self.cfg.metric_dense,
            params={"M": self.cfg.hnsw_m, "efConstruction": self.cfg.hnsw_ef_construction},
        )
        index_params.add_index(
            field_name="sparse",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
            consistency_level=self.cfg.consistency_level,
        )

        self._load_with_retry(name)

    def load(self) -> None:
        if not self.client.has_collection(self.cfg.collection):
            raise RuntimeError(
                f"Collection '{self.cfg.collection}' n'existe pas. Appelle /collections/create ou ensure_collection()."
            )
        self._load_with_retry(self.cfg.collection)

    # ---------- Ingestion helpers ----------
    def _to_record(self, obj: JsonDict, *, max_text: int = 65535) -> JsonDict:
        """
        Version "service": l'objet ne contient PAS vector => on calcule embed.
        """
        rid = safe_primary_id(obj.get("id", ""))

        text = (obj.get("text") or "")[:max_text]
        if not text.strip():
            raise RuntimeError("Chunk vide: 'text' est vide après trim.")

        vec = self.embed_fn(text)
        if len(vec) != self.cfg.embed_dim:
            raise RuntimeError(f"embed_dim mismatch: attendu {self.cfg.embed_dim}, reçu {len(vec)}.")

        meta = obj.get("meta") or {}

        chunk_type = (obj.get("chunk_type") or "").strip()
        if not chunk_type:
            chunk_type = (meta.get("type") or "").strip()

        return {
            "id": rid,
            "vector": vec,
            "text": text,
            "source": obj.get("source", "") or "",
            "section_path": obj.get("section_path", "") or "",
            "section_title": obj.get("section_title", "") or "",
            "page_no": int(obj.get("page_no", -1)) if obj.get("page_no") is not None else -1,
            "chunk_type": chunk_type,
        }

    # ---------- Ingestion (service style) ----------
    def insert(
        self,
        items: Sequence[JsonDict],
        *,
        batch_size: int = 256,
        max_text: int = 65535,
        flush: bool = True,
    ) -> int:
        if not items:
            return 0

        self.ensure_collection()
        coll = self.cfg.collection

        batch: List[JsonDict] = []
        total = 0

        for obj in items:
            rec = self._to_record(obj, max_text=max_text)
            if not rec["id"]:
                continue
            batch.append(rec)

            if len(batch) >= batch_size:
                self.client.insert(collection_name=coll, data=batch)
                total += len(batch)
                batch.clear()

        if batch:
            self.client.insert(collection_name=coll, data=batch)
            total += len(batch)

        if flush:
            self.client.flush(collection_name=coll)

        return total

    def upsert(
        self,
        items: Sequence[JsonDict],
        *,
        batch_size: int = 256,
        max_text: int = 65535,
        delete_batch: int = 2000,
        flush: bool = True,
    ) -> int:
        """
        Upsert: delete(ids) puis insert()
        """
        if not items:
            return 0

        self.ensure_collection()
        coll = self.cfg.collection

        ids = [safe_primary_id(obj.get("id", "")) for obj in items if obj.get("id")]
        ids = [i for i in ids if i]

        for i in range(0, len(ids), delete_batch):
            sub = ids[i : i + delete_batch]
            self.client.delete(collection_name=coll, ids=sub)

        return self.insert(items, batch_size=batch_size, max_text=max_text, flush=flush)

    # ---------- Ingestion FAST (notebook style) ----------
    def insert_fast(
        self,
        items: Sequence[JsonDict],
        *,
        batch_size: int = 256,
        max_text: int = 60000,
        default_source: str = "",
        default_chunk_type: str = "",
        flush: bool = True,
    ) -> int:
        """
        Insert FAST "comme notebook":
        - text[:max_text]
        - embed_fn(text)
        - insert record complet (avec vector)
        - flush en fin
        - pas de delete (donc pas un upsert)
        """
        if not items:
            return 0

        self.ensure_collection()
        coll = self.cfg.collection

        batch: List[JsonDict] = []
        total = 0

        for obj in items:
            text = (obj.get("text") or "")[:max_text]
            if not text.strip():
                continue

            rid = safe_primary_id(obj.get("id", ""))
            if not rid:
                continue

            vec = self.embed_fn(text)
            if len(vec) != self.cfg.embed_dim:
                raise RuntimeError(f"embed_dim mismatch: attendu {self.cfg.embed_dim}, reçu {len(vec)}.")

            chunk_type = (obj.get("chunk_type") or "").strip() or default_chunk_type

            batch.append({
                "id": rid,
                "vector": vec,
                "text": text,  # déclenche BM25->sparse via Function
                "source": obj.get("source", default_source) or default_source,
                "section_path": obj.get("section_path", "") or "",
                "section_title": obj.get("section_title", "") or "",
                "page_no": int(obj.get("page_no", -1)) if obj.get("page_no") is not None else -1,
                "chunk_type": chunk_type,
            })

            if len(batch) >= batch_size:
                self.client.insert(collection_name=coll, data=batch)
                total += len(batch)
                batch.clear()

        if batch:
            self.client.insert(collection_name=coll, data=batch)
            total += len(batch)

        if flush:
            self.client.flush(collection_name=coll)

        return total

    # ---------- Retrieval ----------
    def search_dense(
        self,
        query: str,
        *,
        top_k: int = 5,
        ef: int = 100,
        output_fields: Optional[List[str]] = None,
        expr: str = "",
    ):
        self.load()
        qvec = self.embed_fn(query)

        output_fields = output_fields or [
            "id",
            "text",
            "source",
            "page_no",
            "section_title",
            "chunk_type",
            "section_path",
        ]

        return self.client.search(
            collection_name=self.cfg.collection,
            data=[qvec],
            anns_field="vector",
            limit=top_k,
            output_fields=output_fields,
            filter=expr,
            search_params={"metric_type": self.cfg.metric_dense, "params": {"ef": ef}},
            consistency_level=self.cfg.consistency_level,
        )

    def search_hybrid(
        self,
        query: str,
        *,
        top_k: int = 5,
        ef: int = 100,
        rrf_k: Optional[int] = None,
        output_fields: Optional[List[str]] = None,
        expr: str = "",
    ):
        self.load()
        rrf_k = self.cfg.rrf_k if rrf_k is None else rrf_k

        output_fields = output_fields or [
            "id",
            "text",
            "source",
            "page_no",
            "section_title",
            "chunk_type",
            "section_path",
        ]

        qvec = self.embed_fn(query)

        dense_req = AnnSearchRequest(
            data=[qvec],
            anns_field="vector",
            param={"metric_type": self.cfg.metric_dense, "params": {"ef": ef}},
            limit=top_k,
            expr=expr,
        )

        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=top_k,
            expr=expr,
        )

        return self.client.hybrid_search(
            self.cfg.collection,
            [sparse_req, dense_req],
            RRFRanker(k=rrf_k),
            limit=top_k,
            output_fields=output_fields,
            consistency_level=self.cfg.consistency_level,
        )


# =========================
# FastAPI Models
# =========================
class CreateCollectionRequest(BaseModel):
    collection: str
    drop_if_exists: bool = False
    analyzer_type: str = "english"
    consistency_level: str = "Strong"


class ChunkItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    source: Optional[str] = ""
    section_path: Optional[str] = ""
    section_title: Optional[str] = ""
    page_no: Optional[int] = -1
    chunk_type: Optional[str] = ""
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection: str
    items: List[ChunkItem]
    batch_size: int = 256
    mode: str = "upsert"  # "insert" ou "upsert"
    flush: bool = True


class InsertFastRequest(BaseModel):
    """
    Request permissive pour insert_fast (plus proche notebook).
    """
    collection: str
    items: List[Dict[str, Any]]
    batch_size: int = 256
    max_text: int = 60000
    default_source: str = ""
    default_chunk_type: str = ""
    flush: bool = True


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection: str
    query: str
    top_k: int = 5
    ef: int = 100
    expr: str = ""


class HybridSearchRequest(SearchRequest):
    rrf_k: Optional[int] = None


# =========================
# FastAPI App
# =========================
app = FastAPI(title="Milvus VDB Service (Ollama embeddings)", version="2.1.0")


def make_vdb(
    collection: str,
    *,
    analyzer_type: str = "english",
    drop_if_exists: bool = False,
    consistency_level: str = "Strong",
) -> MilvusHybridVDB:
    cfg = MilvusHybridConfig(
        uri=MILVUS_URI,
        collection=collection,
        embed_dim=EMBED_DIM,
        analyzer_type=analyzer_type,
        drop_if_exists=drop_if_exists,
        consistency_level=consistency_level,
    )
    return MilvusHybridVDB(cfg=cfg, embed_fn=emb_text)


@app.get("/health")
def health():
    milvus_ok = False
    milvus_err = None
    try:
        c = MilvusClient(uri=MILVUS_URI)
        _ = c.list_collections()
        milvus_ok = True
    except Exception as e:
        milvus_err = repr(e)

    return {
        "status": "ok",
        "milvus_uri": MILVUS_URI,
        "milvus_ok": milvus_ok,
        "milvus_err": milvus_err,
        "ollama_host": OLLAMA_HOST,
        "embedding_model": OLLAMA_EMB_MODEL,
        "embed_dim": EMBED_DIM,
    }


@app.post("/collections/create")
def create_collection(req: CreateCollectionRequest):
    try:
        vdb = make_vdb(
            req.collection,
            analyzer_type=req.analyzer_type,
            drop_if_exists=req.drop_if_exists,
            consistency_level=req.consistency_level,
        )
        vdb.ensure_collection()
        return {"ok": True, "collection": req.collection, "embed_dim": EMBED_DIM}
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/collections")
def list_collections():
    try:
        client = MilvusClient(uri=MILVUS_URI)
        cols = client.list_collections()
        return {"collections": cols}
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/upsert")
def upsert(req: UpsertRequest):
    try:
        if not req.items:
            raise HTTPException(status_code=400, detail="items is empty")

        vdb = make_vdb(req.collection)
        payload = [i.model_dump() for i in req.items]

        if req.mode == "insert":
            n = vdb.insert(payload, batch_size=req.batch_size, flush=req.flush)
        elif req.mode == "upsert":
            n = vdb.upsert(payload, batch_size=req.batch_size, flush=req.flush)
        else:
            raise HTTPException(status_code=400, detail='mode must be "insert" or "upsert"')

        return {"ok": True, "count": n, "collection": req.collection, "mode": req.mode}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/insert_fast")
def insert_fast(req: InsertFastRequest):
    """
    Insert "comme notebook" :
    - calc embeddings
    - insert record complet (avec vector)
    - flush
    """
    try:
        if not req.items:
            raise HTTPException(status_code=400, detail="items is empty")

        vdb = make_vdb(req.collection)

        n = vdb.insert_fast(
            req.items,
            batch_size=req.batch_size,
            max_text=req.max_text,
            default_source=req.default_source,
            default_chunk_type=req.default_chunk_type,
            flush=req.flush,
        )

        return {"ok": True, "count": n, "collection": req.collection, "mode": "insert_fast"}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/search/dense")
def search_dense(req: SearchRequest):
    try:
        vdb = make_vdb(req.collection)
        res = vdb.search_dense(query=req.query, top_k=req.top_k, ef=req.ef, expr=req.expr)
        return {"results": res}
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/search/hybrid")
def search_hybrid(req: HybridSearchRequest):
    try:
        vdb = make_vdb(req.collection)
        res = vdb.search_hybrid(
            query=req.query,
            top_k=req.top_k,
            ef=req.ef,
            rrf_k=req.rrf_k,
            expr=req.expr,
        )
        return {"results": res}
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())
