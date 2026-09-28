"""检索器：dense / sparse / hybrid 三档 + 元数据 pre-filter + rerank。

设计（plan.md）：三档可切换以支撑消融实验。
    dense  : 仅稠密向量 search
    sparse : 仅 BM25 稀疏 search（传自然语言，Milvus 内部分词）
    hybrid : hybrid_search + RRFRanker 融合
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from app.config import get_settings
from app.indexing.embedder import Embedder
from app.indexing.milvus_store import make_client
from app.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

OUTPUT_FIELDS = [
    "content", "doc_id", "doc_title", "doc_type", "product_category",
    "chunk_type", "source_path",
]


@dataclass
class RetrievalConfig:
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    use_rerank: bool = True
    rerank_top_n: int = 5
    use_query_rewrite: bool = True
    product_category: str | None = None
    doc_type: str | None = None
    min_score_threshold: float = 0.0

    @classmethod
    def from_settings(cls) -> RetrievalConfig:
        s = get_settings()
        return cls(
            mode=s.retrieval_mode,
            dense_top_k=s.dense_top_k,
            sparse_top_k=s.sparse_top_k,
            rrf_k=s.rrf_k,
            use_rerank=s.use_rerank,
            rerank_top_n=s.rerank_top_n,
            use_query_rewrite=s.use_query_rewrite,
            min_score_threshold=s.min_score_threshold,
        )


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    doc_id: int
    doc_title: str
    doc_type: str
    product_category: str
    chunk_type: str
    source_path: str
    score: float | None = None


@dataclass
class RetrievalResult:
    query: str
    rewritten_query: str | None
    chunks: list[RetrievedChunk] = field(default_factory=list)
    mode: str = "hybrid"


def _build_filter(cfg: RetrievalConfig) -> str:
    exprs = []
    if cfg.product_category:
        exprs.append(f'product_category == "{cfg.product_category}"')
    if cfg.doc_type:
        exprs.append(f'doc_type == "{cfg.doc_type}"')
    return " and ".join(exprs)


def _row_to_chunk(row: dict) -> RetrievedChunk:
    entity = row.get("entity", row)
    return RetrievedChunk(
        chunk_id=str(row.get("id") or entity.get("id") or ""),
        content=entity.get("content", ""),
        doc_id=int(entity.get("doc_id", 0)),
        doc_title=entity.get("doc_title", ""),
        doc_type=entity.get("doc_type", ""),
        product_category=entity.get("product_category", ""),
        chunk_type=entity.get("chunk_type", ""),
        source_path=entity.get("source_path", ""),
        score=float(row["distance"]) if row.get("distance") is not None else None,
    )


class Retriever:
    def __init__(self, client: MilvusClient | None = None, embedder: Embedder | None = None):
        self._client = client or make_client()
        self._embedder = embedder or Embedder()
        self._reranker = Reranker()
        self._settings = get_settings()

    # ---- 单路 ----
    def _dense_search(self, query: str, cfg: RetrievalConfig, expr: str) -> list[RetrievedChunk]:
        vec = self._embedder.embed_one(query)
        res = self._client.search(
            collection_name=self._settings.milvus_collection,
            data=[vec],
            anns_field="dense",
            search_params={"metric_type": "COSINE"},
            limit=cfg.dense_top_k,
            filter=expr or None,
            output_fields=OUTPUT_FIELDS,
        )
        return [_row_to_chunk(r) for r in (res[0] if res else [])]

    def _sparse_search(self, query: str, cfg: RetrievalConfig, expr: str) -> list[RetrievedChunk]:
        res = self._client.search(
            collection_name=self._settings.milvus_collection,
            data=[query],  # 传自然语言，Milvus 内部分词并做 BM25
            anns_field="sparse",
            search_params={"metric_type": "BM25"},
            limit=cfg.sparse_top_k,
            filter=expr or None,
            output_fields=OUTPUT_FIELDS,
        )
        return [_row_to_chunk(r) for r in (res[0] if res else [])]

    def _hybrid_search(self, query: str, cfg: RetrievalConfig, expr: str) -> list[RetrievedChunk]:
        vec = self._embedder.embed_one(query)
        dense_req = AnnSearchRequest(
            data=[vec], anns_field="dense",
            param={"metric_type": "COSINE"}, limit=cfg.dense_top_k, expr=expr or None,
        )
        sparse_req = AnnSearchRequest(
            data=[query], anns_field="sparse",
            param={"metric_type": "BM25"}, limit=cfg.sparse_top_k, expr=expr or None,
        )
        res = self._client.hybrid_search(
            collection_name=self._settings.milvus_collection,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=cfg.rrf_k),
            limit=cfg.dense_top_k,
            output_fields=OUTPUT_FIELDS,
        )
        return [_row_to_chunk(r) for r in (res[0] if res else [])]

    def retrieve(self, query: str, cfg: RetrievalConfig | None = None) -> RetrievalResult:
        cfg = cfg or RetrievalConfig.from_settings()
        expr = _build_filter(cfg)

        if cfg.mode == "dense":
            candidates = self._dense_search(query, cfg, expr)
        elif cfg.mode == "sparse":
            candidates = self._sparse_search(query, cfg, expr)
        else:
            candidates = self._hybrid_search(query, cfg, expr)

        # rerank
        if cfg.use_rerank and candidates:
            pairs = self._reranker.rerank(
                query, [c.content for c in candidates], cfg.rerank_top_n
            )
            ranked = []
            for idx, score in pairs:
                if 0 <= idx < len(candidates):
                    c = candidates[idx]
                    c.score = score
                    ranked.append(c)
            if ranked:
                candidates = ranked
            else:
                candidates = candidates[: cfg.rerank_top_n]
        else:
            candidates = candidates[: cfg.rerank_top_n]

        # 空结果/低分兜底
        if cfg.min_score_threshold > 0:
            candidates = [
                c for c in candidates if c.score is not None and c.score >= cfg.min_score_threshold
            ]

        return RetrievalResult(query=query, rewritten_query=None, chunks=candidates, mode=cfg.mode)
