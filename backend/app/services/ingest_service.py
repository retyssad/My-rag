"""入库服务：解析 → 切分 → embedding → 双写（MySQL 为准、Milvus 为索引）。

流程与状态机（见 plan.md）：
    pending → processing → done / failed

幂等（ADR-0002）：入库/重试前先按 doc_id 清掉 MySQL 旧 chunk 与 Milvus 旧向量。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy import delete, select

from app.chunking.chunker import ChunkingConfig, chunk_ir
from app.db.models import Chunk, Document, DocumentStatus
from app.db.session import session_scope
from app.indexing.embedder import Embedder
from app.indexing.milvus_store import (
    build_row,
    delete_by_doc,
    ensure_collection,
    insert_rows,
    make_client,
)
from app.parsing.parser import parse_file

logger = logging.getLogger(__name__)


def create_document(
    *, title: str, doc_type: str, product_category: str, source_path: str
) -> int:
    """建文档记录（pending），返回 doc_id。"""
    with session_scope() as db:
        doc = Document(
            title=title,
            doc_type=doc_type,
            product_category=product_category,
            source_path=source_path,
            status=DocumentStatus.pending,
        )
        db.add(doc)
        db.flush()
        return doc.id


def process_document(
    doc_id: int,
    chunk_config: ChunkingConfig | None = None,
    flush: bool = True,
) -> None:
    """完整入库流程。异常时置 failed 并记录 error_msg。

    flush=False 用于批量入库：不逐篇等待 Milvus flush（单次可达数秒），
    由调用方在批量结束后调用 flush_collection() 一次。
    """
    embedder = Embedder()
    client = make_client()
    ensure_collection(client)

    try:
        with session_scope() as db:
            doc = db.get(Document, doc_id)
            if doc is None:
                logger.warning("doc %s 不存在，跳过", doc_id)
                return
            doc.status = DocumentStatus.processing
            doc.error_msg = None
            source_path = doc.source_path
            title = doc.title
            doc_type = doc.doc_type
            product_category = doc.product_category
            # 幂等：清掉旧的 MySQL chunk
            db.execute(delete(Chunk).where(Chunk.doc_id == doc_id))

        # 幂等：清掉旧向量（放在 MySQL 事务外，避免长事务）
        delete_by_doc(client, doc_id, flush=flush)

        path = Path(source_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / source_path

        blocks = parse_file(path)
        drafts = chunk_ir(blocks, chunk_config)
        if not drafts:
            raise ValueError("解析后没有产生任何 chunk")

        vectors = embedder.embed([d.content for d in drafts])

        rows = []
        chunk_records = []
        for idx, (d, vec) in enumerate(zip(drafts, vectors, strict=True)):
            milvus_id = f"{doc_id}_{idx}"
            meta = {
                "heading_path": d.heading_path,
                "chunk_type": d.chunk_type,
            }
            rows.append(
                build_row(
                    doc_id=doc_id,
                    chunk_index=idx,
                    content=d.content,
                    dense=vec,
                    doc_title=title,
                    doc_type=doc_type,
                    product_category=product_category,
                    chunk_type=d.chunk_type,
                    source_path=source_path,
                )
            )
            chunk_records.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_index=idx,
                    content=d.content,
                    token_count=d.token_count,
                    chunk_type=d.chunk_type,
                    milvus_id=milvus_id,
                    metadata_json=json.dumps(meta, ensure_ascii=False),
                )
            )

        # 写 Milvus（索引）
        insert_rows(client, rows, flush=flush)

        # 写 MySQL（事实来源）+ 置 done
        with session_scope() as db:
            db.add_all(chunk_records)
            doc = db.get(Document, doc_id)
            doc.status = DocumentStatus.done
            doc.chunk_count = len(chunk_records)
            doc.error_msg = None

        logger.info("doc %s 入库完成：%d chunks", doc_id, len(chunk_records))

    except Exception as e:  # noqa: BLE001
        logger.exception("doc %s 入库失败", doc_id)
        with session_scope() as db:
            doc = db.get(Document, doc_id)
            if doc is not None:
                doc.status = DocumentStatus.failed
                doc.error_msg = f"{type(e).__name__}: {e}"[:2000]


def reconcile_stuck_documents() -> int:
    """启动对账：把卡在 processing 的文档重置为 failed（进程曾崩溃）。"""
    with session_scope() as db:
        stuck = db.scalars(
            select(Document).where(Document.status == DocumentStatus.processing)
        ).all()
        for doc in stuck:
            doc.status = DocumentStatus.failed
            doc.error_msg = "服务重启，处理被中断；请重试"
        return len(stuck)
