"""问答 API：SSE 流式（plan.md 事件协议）。

事件顺序：retrieval → delta* → citations → done（或 error）。
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import ChatRequest, ConversationOut, MessageOut
from app.db.models import Conversation, Message
from app.db.session import get_db, session_scope
from app.generation.generator import (
    NO_RESULT_ANSWER,
    LLMClient,
    build_messages,
    parse_citations,
)
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.retriever import RetrievalConfig, Retriever

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

HISTORY_TURNS = 5  # 注入 prompt 的最近轮数
REWRITE_HISTORY_TURNS = 5  # 喂给改写的最近轮数


def _get_history(db, conversation_id: int, turns: int) -> list[dict]:
    rows = db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id.desc())
        .limit(turns * 2)
    ).all()
    rows = list(reversed(rows))
    return [
        {"role": m.role, "content": m.content}
        for m in rows
        if m.role in ("user", "assistant")
    ]


@router.post("/chat")
async def chat(req: ChatRequest):
    """SSE 流式问答。

    注意：本函数为 async（SSE），内部对同步的检索/DB 调用需谨慎；
    这里检索是短耗时操作，直接调用（见 ADR-0004 的取舍说明）。
    """
    cfg = RetrievalConfig.from_settings()
    if req.mode:
        cfg.mode = req.mode  # type: ignore[assignment]
    if req.use_rerank is not None:
        cfg.use_rerank = req.use_rerank
    if req.use_query_rewrite is not None:
        cfg.use_query_rewrite = req.use_query_rewrite
    if req.product_category:
        cfg.product_category = req.product_category
    if req.doc_type:
        cfg.doc_type = req.doc_type
    if req.dense_top_k:
        cfg.dense_top_k = req.dense_top_k
    if req.sparse_top_k:
        cfg.sparse_top_k = req.sparse_top_k
    if req.rerank_top_n:
        cfg.rerank_top_n = req.rerank_top_n

    # 会话准备（同步 DB 操作，简短）
    conversation_id = req.conversation_id
    history: list[dict] = []
    with session_scope() as db:
        if conversation_id is None:
            conv = Conversation(title=req.query[:30])
            db.add(conv)
            db.flush()
            conversation_id = conv.id
        else:
            conv = db.get(Conversation, conversation_id)
            if conv is None:
                raise HTTPException(404, "会话不存在")
            history = _get_history(db, conversation_id, HISTORY_TURNS)

    async def event_source():
        full_answer = ""
        citations_payload: list[dict] = []
        retrieved_payload: list[dict] = []
        try:
            # 1) query 改写（多轮）
            rewritten = req.query
            if cfg.use_query_rewrite and history:
                rewritten = QueryRewriter().rewrite(req.query, history)

            # 2) 检索
            result = Retriever().retrieve(rewritten, cfg)
            retrieved_payload = [
                {
                    "chunk_id": c.chunk_id,
                    "doc_title": c.doc_title,
                    "doc_type": c.doc_type,
                    "chunk_type": c.chunk_type,
                    "content": c.content,
                    "score": c.score,
                }
                for c in result.chunks
            ]
            yield {
                "event": "retrieval",
                "data": json.dumps(
                    {
                        "conversation_id": conversation_id,
                        "query": req.query,
                        "query_rewritten": rewritten if rewritten != req.query else None,
                        "mode": cfg.mode,
                        "chunks": retrieved_payload,
                    },
                    ensure_ascii=False,
                ),
            }

            # 3) 空结果兜底：不调 LLM
            if not result.chunks:
                full_answer = NO_RESULT_ANSWER
                yield {"event": "delta", "data": json.dumps({"text": full_answer}, ensure_ascii=False)}
            else:
                messages = build_messages(req.query, result.chunks, history)
                for token in LLMClient().stream(messages):
                    full_answer += token
                    yield {"event": "delta", "data": json.dumps({"text": token}, ensure_ascii=False)}

            # 4) 引用
            if result.chunks:
                cites = parse_citations(full_answer, result.chunks)
                citations_payload = [c.__dict__ for c in cites]
            yield {
                "event": "citations",
                "data": json.dumps({"citations": citations_payload}, ensure_ascii=False),
            }

            # 5) 落库
            with session_scope() as db:
                db.add(Message(conversation_id=conversation_id, role="user", content=req.query))
                db.add(
                    Message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=full_answer,
                        citations_json=json.dumps(citations_payload, ensure_ascii=False),
                    )
                )

            yield {
                "event": "done",
                "data": json.dumps({"conversation_id": conversation_id}, ensure_ascii=False),
            }
        except Exception as e:  # noqa: BLE001
            logger.exception("chat 失败")
            yield {
                "event": "error",
                "data": json.dumps({"message": f"{type(e).__name__}: {e}"}, ensure_ascii=False),
            }

    return EventSourceResponse(event_source())


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(db=Depends(get_db)) -> list[ConversationOut]:
    rows = db.scalars(select(Conversation).order_by(Conversation.id.desc()).limit(50)).all()
    return [ConversationOut.model_validate(r) for r in rows]


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
def list_messages(conversation_id: int, db=Depends(get_db)) -> list[MessageOut]:
    conv = db.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(404, "会话不存在")
    rows = db.scalars(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
    ).all()
    out = []
    for m in rows:
        out.append(
            MessageOut(
                id=m.id,
                role=m.role,
                content=m.content,
                citations=json.loads(m.citations_json or "[]"),
                created_at=m.created_at,
            )
        )
    return out
