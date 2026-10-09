"""批量回查父块，按最优子块的排名去重；旧文档继续使用子块。"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import ChunkParent, ParentChunk
from app.db.session import session_scope

if TYPE_CHECKING:
    from app.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


def expand_parent_context(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not chunks:
        return []
    try:
        with session_scope() as db:
            rows = db.execute(
                select(ChunkParent.child_id, ParentChunk)
                .join(ParentChunk, ChunkParent.parent_id == ParentChunk.id)
                .where(ChunkParent.child_id.in_([c.chunk_id for c in chunks]))
            ).all()
            parents = {child_id: parent for child_id, parent in rows}
    except SQLAlchemyError:
        logger.warning("父块回查失败，降级为子块上下文", exc_info=True)
        return chunks

    out: list[RetrievedChunk] = []
    seen: dict[str, RetrievedChunk] = {}
    for child in chunks:
        parent = parents.get(child.chunk_id)
        if parent is None or parent.doc_id != child.doc_id:
            out.append(child)
            continue
        if parent.id in seen:
            seen[parent.id].matched_child_ids.append(child.chunk_id)
            continue
        context = replace(
            child, chunk_id=parent.id, content=parent.content, chunk_type=parent.chunk_type,
            parent_id=parent.id, matched_child_ids=[child.chunk_id],
        )
        seen[parent.id] = context
        out.append(context)
    return out
