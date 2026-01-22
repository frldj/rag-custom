# to do : 
# - pouvoir créer BDD milvus
# - voir les BDD milvus disponible
# - requêter BDD milvus
# - ajouter de nouvelles données

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)
from pymilvus.exceptions import MilvusException

JsonDict = Dict[str, Any]


@dataclass
class MilvusHybridConfig:
    uri: str = "http://localhost:19530"
    collection: str = "rag_minist_int_hybrid"
    embed_dim: int = 1024

    # schema/index
    drop_if_exists: bool = False
    consistency_level: str = "Strong"

    # BM25 analyzer
    analyzer_type: str = "english"  # "french" si dispo/utile côté Milvus

    # HNSW params
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    metric_dense: str = "COSINE"  # ou "IP" si tu normalises

    # RRF
    rrf_k: int = 60

    # load robustness
    load_retries: int = 10
    load_sleep_s: float = 1.0


class MilvusHybridVDB:
    """
    Adapter Milvus (dense + BM25 sparse) :
    - ensure_collection(): create schema + indexes + load
    - insert(): batch insert
    - search_dense(), search_hybrid(): retrieval
    """

    def __init__(
        self,
        cfg: MilvusHybridConfig,
        embed_fn: Callable[[str], List[float]],
        client: Optional[MilvusClient] = None,
    ):
        self.cfg = cfg
        self.embed_fn = embed_fn
        self.client = client or MilvusClient(cfg.uri)

    # ---------- Connection / load helpers ----------
    def _reconnect(self) -> None:
        # Recrée un client "propre" (comme ton notebook de reconnexion)
        self.client = MilvusClient(self.cfg.uri)

    def _load_with_retry(self, collection_name: str) -> None:
        last_err: Exception | None = None

        for _ in range(self.cfg.load_retries):
            try:
                self.client.load_collection(collection_name)
                return
            except MilvusException as e:
                last_err = e
                msg = str(e)

                # Cas typique: Milvus joignable mais pas prêt (services/nodes pas encore OK)
                if "service resource insufficient" in msg or "currentStreamingNode=0" in msg:
                    time.sleep(self.cfg.load_sleep_s)
                    self._reconnect()
                    continue

                # Autre erreur: on remonte
                raise

        raise RuntimeError(
            f"Impossible de load '{collection_name}' après {self.cfg.load_retries} tentatives: {last_err}"
        ) from last_err

    # ---------- Collection lifecycle ----------
    def ensure_collection(self) -> None:
        name = self.cfg.collection

        # IMPORTANT: utiliser self.client partout (car _reconnect() peut remplacer self.client)
        if self.client.has_collection(name):
            if self.cfg.drop_if_exists:
                self.client.drop_collection(name)
            else:
                self._load_with_retry(name)
                return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)

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

        # Évite replica_number=1 (peut ajouter des contraintes côté cluster)
        self._load_with_retry(name)

    def load(self) -> None:
        if not self.client.has_collection(self.cfg.collection):
            raise RuntimeError(f"Collection '{self.cfg.collection}' n'existe pas. Appelle ensure_collection().")
        self._load_with_retry(self.cfg.collection)

    # ---------- Ingestion ----------
    def _to_record(self, obj: JsonDict, *, max_text: int = 65535) -> JsonDict:
        text = (obj.get("text") or "")[:max_text]
        vec = self.embed_fn(text)

        return {
            "id": obj["id"],
            "vector": vec,
            "text": text,
            # champs dynamiques
            "source": obj.get("source", ""),
            "section_path": obj.get("section_path", ""),
            "section_title": obj.get("section_title", ""),
            "page_no": int(obj.get("page_no", -1)) if obj.get("page_no") is not None else -1,
            "chunk_type": (obj.get("meta") or {}).get("type", ""),
        }

    def insert(
        self,
        items: Sequence[JsonDict],
        *,
        batch_size: int = 256,
        max_text: int = 65535,
    ) -> int:
        self.ensure_collection()
        coll = self.cfg.collection

        batch: List[JsonDict] = []
        total = 0

        for obj in items:
            batch.append(self._to_record(obj, max_text=max_text))
            if len(batch) >= batch_size:
                self.client.insert(coll, batch)
                total += len(batch)
                batch.clear()

        if batch:
            self.client.insert(coll, batch)
            total += len(batch)

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
        output_fields = output_fields or ["id", "text", "source", "page_no", "section_title"]

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
        output_fields = output_fields or ["id", "text", "source", "page_no", "section_title"]

        qvec = self.embed_fn(query)

        dense_req = AnnSearchRequest(
            data=[qvec],
            anns_field="vector",
            param={"metric_type": self.cfg.metric_dense, "params": {"ef": ef}},
            limit=top_k,
            expr=expr,
        )

        sparse_req = AnnSearchRequest(
            data=[query],  # texte brut pour BM25
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
