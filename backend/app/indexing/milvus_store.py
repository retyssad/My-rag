"""Milvus 索引层：collection schema + 写入 + 删除。

collection `chunks`（单 collection，靠元数据 pre-filter，见 plan.md / ADR-0001）：
- id            VARCHAR 主键 = f"{doc_id}_{chunk_index}"
- dense         FLOAT_VECTOR(dim)，HNSW / COSINE
- content       VARCHAR，enable_analyzer=True（jieba），经 FunctionType.BM25 生成 sparse
- sparse        SPARSE_FLOAT_VECTOR，SPARSE_INVERTED_INDEX / BM25
- 标量字段      doc_id / doc_type / product_category / chunk_type / source_path（INVERTED）
"""

from __future__ import annotations

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from app.config import get_settings


def make_client() -> MilvusClient:
    s = get_settings()
    return MilvusClient(uri=s.milvus_uri)


def ensure_collection(client: MilvusClient, recreate: bool = False) -> None:
    s = get_settings()
    name = s.milvus_collection
    if client.has_collection(name):
        if not recreate:
            return
        client.drop_collection(name)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=s.embedding_dim)
    schema.add_field(
        "content",
        DataType.VARCHAR,
        max_length=8000,
        enable_analyzer=True,
        # 必须显式指定中文分词，否则默认分析器对中文无效，BM25 形同虚设
        analyzer_params={"type": "chinese"},
    )
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("doc_id", DataType.INT64)
    schema.add_field("doc_title", DataType.VARCHAR, max_length=255)
    schema.add_field("doc_type", DataType.VARCHAR, max_length=32)
    schema.add_field("product_category", DataType.VARCHAR, max_length=32)
    schema.add_field("chunk_type", DataType.VARCHAR, max_length=16)
    schema.add_field("source_path", DataType.VARCHAR, max_length=512)

    schema.add_function(
        Function(
            name="bm25_fn",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=["sparse"],
            params={},
        )
    )

    index_params = client.prepare_index_params()
    index_params.add_index("dense", index_type="HNSW", metric_type="COSINE",
                           params={"M": 16, "efConstruction": 200})
    index_params.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
                           params={"bm25_k1": 1.2, "bm25_b": 0.75})
    for scalar in ("doc_id", "doc_type", "product_category", "chunk_type"):
        index_params.add_index(scalar, index_type="INVERTED")

    client.create_collection(name, schema=schema, index_params=index_params)


def build_row(
    *, doc_id: int, chunk_index: int, content: str, dense: list[float],
    doc_title: str, doc_type: str, product_category: str,
    chunk_type: str, source_path: str,
) -> dict:
    return {
        "id": f"{doc_id}_{chunk_index}",
        "dense": dense,
        "content": content,
        "doc_id": doc_id,
        "doc_title": doc_title,
        "doc_type": doc_type,
        "product_category": product_category,
        "chunk_type": chunk_type,
        "source_path": source_path,
    }


def insert_rows(client: MilvusClient, rows: list[dict], flush: bool = True) -> None:
    if not rows:
        return
    s = get_settings()
    client.insert(s.milvus_collection, rows)
    if flush:
        # flush 会等待 segment 构建，成本高；批量入库时应置 False，最后统一 flush 一次
        client.flush(s.milvus_collection)


def delete_by_doc(client: MilvusClient, doc_id: int, flush: bool = True) -> None:
    """按 doc_id 删除该文档全部向量（重试/幂等重建用）。"""
    s = get_settings()
    client.delete(s.milvus_collection, filter=f"doc_id == {doc_id}")
    if flush:
        client.flush(s.milvus_collection)


def flush_collection(client: MilvusClient) -> None:
    """统一 flush：批量写入后调用一次，使数据可被检索。"""
    s = get_settings()
    client.flush(s.milvus_collection)


__all__ = [
    "AnnSearchRequest",
    "RRFRanker",
    "build_row",
    "delete_by_doc",
    "ensure_collection",
    "insert_rows",
    "make_client",
]
