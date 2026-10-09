"""API 请求/响应模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: int
    title: str
    doc_type: str
    product_category: str
    source_path: str
    status: str
    chunk_count: int
    error_msg: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentListOut(BaseModel):
    total: int
    items: list[DocumentOut]


class UploadOut(BaseModel):
    doc_id: int
    status: str
    message: str = "文档已接收，正在后台处理"


class ChunkOut(BaseModel):
    id: int
    chunk_index: int
    content: str
    chunk_type: str
    token_count: int
    milvus_id: str

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    conversation_id: int | None = None
    # 可选覆盖检索配置（消融/演示用）
    mode: str | None = Field(default=None, description="dense | sparse | hybrid")
    use_rerank: bool | None = None
    use_query_rewrite: bool | None = None
    use_intent: bool | None = None
    use_parent_retrieval: bool | None = None
    product_category: str | None = None
    doc_type: str | None = None
    dense_top_k: int | None = Field(default=None, ge=1, le=100)
    sparse_top_k: int | None = Field(default=None, ge=1, le=100)
    rerank_top_n: int | None = Field(default=None, ge=1, le=50)


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    doc_id: int
    doc_title: str
    doc_type: str
    product_category: str
    content: str
    score: float | None = None


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    doc_title: str
    doc_type: str
    chunk_type: str
    content: str
    score: float | None = None
    parent_id: str | None = None
    matched_child_ids: list[str] = Field(default_factory=list)


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    citations: list[CitationOut] = []
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: datetime

    model_config = {"from_attributes": True}
