"""父子块的切分、数据库生命周期与检索上下文契约（无需外部服务）。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import pytest
from app.chunking.chunker import ChunkingConfig, chunk_parent_child_ir
from app.db.models import Base, Chunk, ChunkParent, Document, DocumentStatus, ParentChunk
from app.generation.generator import build_context, parse_citations
from app.parsing.parser import IRBlock
from app.retrieval import parent_context
from app.retrieval.retriever import RetrievalConfig, RetrievedChunk, Retriever
from app.services import ingest_service
from sqlalchemy import BigInteger, Integer, MetaData, create_engine, event, func, select
from sqlalchemy.orm import Session


def text_blocks():
    return [
        IRBlock(type="text", level=1, content="退换货政策"),
        IRBlock(type="text", level=0, content="七天退货需保留包装。" * 50),
        IRBlock(type="text", level=0, content="运费由责任方承担。" * 30),
        IRBlock(type="text", level=0, content="例外：定制品。"),
    ]


def test_children_stay_within_parent_and_preserve_short_facts():
    hierarchy = chunk_parent_child_ir(text_blocks(), ChunkingConfig(
        chunk_size=60, chunk_overlap=10, parent_chunk_size=180,
    ))
    assert len(hierarchy.children) > len(hierarchy.parents) > 1
    # 短标题/尾句合并时允许略超父块目标，避免孤立标题或丢失事实。
    assert all(len(p.content) <= 200 for p in hierarchy.parents)
    assert all(len(c.content) <= 60 for c in hierarchy.children)
    assert all(c.content in hierarchy.parents[c.parent_index].content for c in hierarchy.children)
    assert "例外：定制品。" in "".join(c.content for c in hierarchy.children)
    assert chunk_parent_child_ir([]).children == []


def test_table_has_one_complete_parent_and_child():
    table = IRBlock(type="table", level=0, content=[["项目", "金额"]] + [[str(i), "100"] for i in range(50)])
    hierarchy = chunk_parent_child_ir([table], ChunkingConfig(chunk_size=30, parent_chunk_size=90))
    assert len(hierarchy.parents) == len(hierarchy.children) == 1
    assert hierarchy.children[0].content == hierarchy.parents[0].content
    assert "| 49 | 100 |" in hierarchy.children[0].content
    assert hierarchy.children[0].chunk_type == "table"


def test_flat_ingestion_is_available_and_invalid_sizes_rejected():
    flat = chunk_parent_child_ir(text_blocks(), ChunkingConfig(parent_chunk_size=None))
    assert flat.parents == []
    assert all(c.parent_index is None for c in flat.children)
    with pytest.raises(ValueError, match="parent_chunk_size"):
        ChunkingConfig(chunk_size=200, parent_chunk_size=100)


@pytest.fixture
def database(monkeypatch):
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    # SQLite 自增需要 INTEGER；只转换测试 schema，不修改生产模型。
    metadata = MetaData()
    for table in Base.metadata.sorted_tables:
        copied = table.to_metadata(metadata)
        for column in copied.columns:
            if isinstance(column.type, BigInteger):
                column.type = Integer()
    metadata.create_all(engine)

    @contextmanager
    def scope():
        with Session(engine, expire_on_commit=False) as db, db.begin():
            yield db

    monkeypatch.setattr(ingest_service, "session_scope", scope)
    monkeypatch.setattr(parent_context, "session_scope", scope)
    yield scope
    engine.dispose()


@pytest.fixture
def ingested(database, monkeypatch):
    embedded = []

    class FakeEmbedder:
        def embed(self, texts):
            embedded.extend(texts)
            return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(ingest_service, "Embedder", FakeEmbedder)
    monkeypatch.setattr(ingest_service, "make_client", object)
    monkeypatch.setattr(ingest_service, "ensure_collection", lambda *_: None)
    monkeypatch.setattr(ingest_service, "delete_by_doc", lambda *_, **__: None)
    monkeypatch.setattr(ingest_service, "insert_rows", lambda *_, **__: None)
    monkeypatch.setattr(ingest_service, "parse_file", lambda _: text_blocks())
    with database() as db:
        db.add(Document(id=1, title="退换货", doc_type="policy", product_category="phone", source_path="test.md"))
    cfg = ChunkingConfig(chunk_size=60, chunk_overlap=0, parent_chunk_size=180)
    ingest_service.process_document(1, cfg)
    with database() as db:
        assert db.get(Document, 1).status == DocumentStatus.done
        children = db.scalars(select(Chunk).order_by(Chunk.chunk_index)).all()
        assert embedded == [c.content for c in children]
    return cfg


def candidates(database):
    with database() as db:
        children = db.scalars(select(Chunk).order_by(Chunk.chunk_index)).all()
    return [RetrievedChunk(
        chunk_id=c.milvus_id, content=c.content, doc_id=1, doc_title="退换货",
        doc_type="policy", product_category="phone", chunk_type=c.chunk_type,
        source_path="test.md", score=1 - i * 0.01,
    ) for i, c in enumerate(children)]


def counts(database):
    with database() as db:
        return [db.scalar(select(func.count()).select_from(model)) for model in (Chunk, ParentChunk, ChunkParent)]


def test_reprocessing_and_document_deletion_clear_all_levels(database, ingested):
    before = counts(database)
    assert before[0] == before[2] > before[1] > 1
    ingest_service.process_document(1, ingested)
    assert counts(database) == before
    with database() as db:
        assert db.get(Document, 1).chunk_count == before[0]
        db.delete(db.get(Document, 1))
    assert counts(database) == [0, 0, 0]


def test_failed_retry_leaves_no_stale_parents(database, ingested, monkeypatch):
    def fail(_path):
        raise ValueError("无法解析")

    monkeypatch.setattr(ingest_service, "parse_file", fail)
    ingest_service.process_document(1, ingested)
    assert counts(database) == [0, 0, 0]
    with database() as db:
        document = db.get(Document, 1)
        assert document.status == DocumentStatus.failed
        assert document.chunk_count == 0


def test_expansion_dedupes_preserves_scores_and_citations(database, ingested):
    children = candidates(database)
    contexts = parent_context.expand_parent_context(children)
    assert len(contexts) == counts(database)[1]
    assert contexts[0].score == children[0].score
    assert len(contexts[0].matched_child_ids) > 1
    assert children[0].parent_id is None  # 不污染输入
    assert contexts[0].content in build_context(contexts)
    citation = parse_citations("退货要求见[1]。", contexts)[0]
    assert citation.chunk_id == contexts[0].parent_id
    assert citation.content == contexts[0].content
    legacy = replace(children[0], chunk_id="legacy")
    assert parent_context.expand_parent_context([legacy]) == [legacy]
    # 防止错误映射把另一篇文档的父块注入。
    mismatched = replace(children[0], doc_id=2)
    assert parent_context.expand_parent_context([mismatched]) == [mismatched]


@pytest.mark.parametrize("use_rerank", [False, True])
def test_retrieval_limits_after_parent_dedupe(database, ingested, monkeypatch, use_rerank):
    children = candidates(database)
    retriever = Retriever(client=object(), embedder=object())
    monkeypatch.setattr(retriever, "_hybrid_search", lambda *_: children)
    rerank_sizes = []

    def rerank(_query, texts, top_n):
        rerank_sizes.append(top_n)
        assert texts == [c.content for c in children]  # 重排序使用子块
        return [(i, 0.99 - i * 0.01) for i in range(top_n)]

    monkeypatch.setattr(retriever._reranker, "rerank", rerank)
    cfg = RetrievalConfig(use_rerank=use_rerank, rerank_top_n=2, use_parent_retrieval=True)
    contexts = retriever.retrieve("退货", cfg).chunks
    assert len(contexts) == 2
    assert len({c.parent_id for c in contexts}) == 2
    if use_rerank:
        assert rerank_sizes == [len(children)]
    cfg.use_parent_retrieval = False
    flat = retriever.retrieve("退货", cfg).chunks
    assert [c.chunk_id for c in flat] == [c.chunk_id for c in children[:2]]
    assert all(c.parent_id is None for c in flat)


def test_score_threshold_applies_before_parent_expansion(database, ingested, monkeypatch):
    children = candidates(database)
    retriever = Retriever(client=object(), embedder=object())
    monkeypatch.setattr(retriever, "_hybrid_search", lambda *_: children)
    contexts = retriever.retrieve("退货", RetrievalConfig(
        use_rerank=False, min_score_threshold=0.995, use_parent_retrieval=True,
    )).chunks
    assert len(contexts) == 1
    assert contexts[0].matched_child_ids == [children[0].chunk_id]
