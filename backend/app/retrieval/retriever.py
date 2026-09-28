"""检索器：dense / sparse / hybrid 三档 + 元数据 pre-filter + rerank。

设计（plan.md）：三档可切换以支撑消融实验。
    dense  : 仅稠密向量 search
    sparse : 仅 BM25 稀疏 search（传自然语言，Milvus 内部分词）
    hybrid : hybrid_search + RRFRanker 融合
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
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
    # 意图识别相关（软注入）
    use_intent: bool = True
    intent: str | None = None          # 由意图分类器写入；None 表示未识别/未启用
    diversify_by_doc: bool = False     # compare 意图：同一文档最多保留 1 个 chunk
    max_per_doc: int = 1
    # policy 意图：doc_type 过滤取「并集」而非排他过滤，
    # 保证召回不会低于无过滤 baseline（防止误判导致漏召，见 ADR-0007）。
    union_unfiltered: bool = False

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
            use_intent=s.use_intent,
            min_score_threshold=s.min_score_threshold,
        )

    def clone(self) -> RetrievalConfig:
        """浅拷贝，供消融/覆盖参数时避免污染默认值。"""
        return replace(self)

    def describe(self) -> str:
        """供评估表格/日志展示的简短标识。"""
        flags = [self.mode]
        if self.diversify_by_doc:
            flags.append("diversify")
        return "+".join(flags)


def diversify_by_doc(chunks: list[RetrievedChunk], max_per_doc: int = 1) -> list[RetrievedChunk]:
    """按文档去重：同一 doc_id 最多保留 max_per_doc 个 chunk，保持原顺序。

    用于 compare 意图——确保对比对象的多个文档都能进入上下文，
    而不是被同一文档的多个高相似 chunk 占满。
    """
    counts: dict[int, int] = {}
    out: list[RetrievedChunk] = []
    for c in chunks:
        n = counts.get(c.doc_id, 0)
        if n < max_per_doc:
            out.append(c)
            counts[c.doc_id] = n + 1
    return out


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
    intent: str | None = None


def _build_filter(cfg: RetrievalConfig) -> str:
    exprs = []
    if cfg.product_category:
        exprs.append(f'product_category == "{cfg.product_category}"')
    if cfg.doc_type:
        exprs.append(f'doc_type == "{cfg.doc_type}"')
    return " and ".join(exprs)


def _merge_dedupe(primary: list[RetrievedChunk], secondary: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """按 (优先序) 合并两组结果，按 chunk_id 去重（先出现者胜）。"""
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for c in [*primary, *secondary]:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        out.append(c)
    return out


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

        # policy 并集：把「不加 doc_type 过滤」的全量结果并入候选池，
        # 保证召回不低于 baseline（误判时正确文档仍有机会被 rerank 救回）。
        if cfg.union_unfiltered and cfg.doc_type:
            base_expr = _build_filter(replace(cfg, doc_type=None, union_unfiltered=False))
            if cfg.mode == "dense":
                extra = self._dense_search(query, cfg, base_expr)
            elif cfg.mode == "sparse":
                extra = self._sparse_search(query, cfg, base_expr)
            else:
                extra = self._hybrid_search(query, cfg, base_expr)
            candidates = _merge_dedupe(extra, candidates)  # 过滤结果优先

        # 文档去重（compare 意图）：在 rerank 之前做，
        # 保证候选池里有多个文档，rerank 才有跨文档的选择空间。
        if cfg.diversify_by_doc:
            candidates = diversify_by_doc(candidates, cfg.max_per_doc)

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

        return RetrievalResult(
            query=query,
            rewritten_query=None,
            chunks=candidates,
            mode=cfg.mode,
            intent=cfg.intent,
        )
