"""文档管理 API：上传、列表、详情、删除、重试。

写操作需要 X-Admin-Token（plan.md：极简鉴权）。
端点用 def（同步，见 ADR-0004），FastAPI 会自动丢线程池。
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from sqlalchemy import func, select

from app.api.schemas import ChunkOut, DocumentListOut, DocumentOut, UploadOut
from app.config import get_settings
from app.db.models import Chunk, Document, DocumentStatus
from app.db.session import get_db, session_scope
from app.services.ingest_service import create_document, process_document

router = APIRouter(prefix="/api/documents", tags=["documents"])

UPLOAD_DIR = Path(__file__).resolve().parents[3] / "data" / "uploads"
ALLOWED_SUFFIX = {".pdf", ".docx", ".html", ".htm", ".md"}
ALLOWED_DOC_TYPES = {"product_detail", "comparison", "faq", "policy"}
ALLOWED_CATEGORIES = {"phone", "laptop", "earbuds", "all"}


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    s = get_settings()
    if x_admin_token != s.admin_token:
        raise HTTPException(status_code=401, detail="无效或缺失 X-Admin-Token")


@router.post("/upload", response_model=UploadOut, dependencies=[Depends(require_admin)])
def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: str = Form(...),
    product_category: str = Form(...),
    title: str | None = Form(default=None),
    db=Depends(get_db),
) -> UploadOut:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIX:
        raise HTTPException(400, f"不支持的格式 {suffix}，允许：{sorted(ALLOWED_SUFFIX)}")
    if doc_type not in ALLOWED_DOC_TYPES:
        raise HTTPException(400, f"doc_type 必须是 {sorted(ALLOWED_DOC_TYPES)}")
    if product_category not in ALLOWED_CATEGORIES:
        raise HTTPException(400, f"product_category 必须是 {sorted(ALLOWED_CATEGORIES)}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}{suffix}"
    dest = UPLOAD_DIR / stored_name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    rel_path = str(dest.relative_to(Path(__file__).resolve().parents[3])).replace("\\", "/")
    doc_id = create_document(
        title=title or (file.filename or stored_name),
        doc_type=doc_type,
        product_category=product_category,
        source_path=rel_path,
    )
    background.add_task(process_document, doc_id)
    return UploadOut(doc_id=doc_id, status=DocumentStatus.pending.value)


@router.get("", response_model=DocumentListOut)
def list_documents(
    status: str | None = None,
    product_category: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db=Depends(get_db),
) -> DocumentListOut:
    stmt = select(Document)
    count_stmt = select(func.count()).select_from(Document)
    if status:
        stmt = stmt.where(Document.status == status)
        count_stmt = count_stmt.where(Document.status == status)
    if product_category:
        stmt = stmt.where(Document.product_category == product_category)
        count_stmt = count_stmt.where(Document.product_category == product_category)
    total = db.scalar(count_stmt) or 0
    rows = db.scalars(
        stmt.order_by(Document.id.desc()).limit(limit).offset(offset)
    ).all()
    return DocumentListOut(
        total=total, items=[DocumentOut.model_validate(r) for r in rows]
    )


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int, db=Depends(get_db)) -> DocumentOut:
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "文档不存在")
    return DocumentOut.model_validate(doc)


@router.get("/{doc_id}/chunks", response_model=list[ChunkOut])
def list_chunks(doc_id: int, db=Depends(get_db)) -> list[ChunkOut]:
    if db.get(Document, doc_id) is None:
        raise HTTPException(404, "文档不存在")
    rows = db.scalars(
        select(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.chunk_index)
    ).all()
    return [ChunkOut.model_validate(r) for r in rows]


@router.post("/{doc_id}/retry", response_model=UploadOut, dependencies=[Depends(require_admin)])
def retry_document(
    doc_id: int, background: BackgroundTasks, db=Depends(get_db)
) -> UploadOut:
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "文档不存在")
    if doc.status == DocumentStatus.processing:
        raise HTTPException(409, "文档正在处理中")
    doc.status = DocumentStatus.pending
    doc.error_msg = None
    db.commit()
    background.add_task(process_document, doc_id)
    return UploadOut(doc_id=doc_id, status=DocumentStatus.pending.value, message="已重新入队")


@router.delete("/{doc_id}", dependencies=[Depends(require_admin)])
def delete_document(doc_id: int, db=Depends(get_db)) -> dict:
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "文档不存在")
    # 先删向量（索引），再删 MySQL（事实来源）
    from app.indexing.milvus_store import delete_by_doc, make_client

    try:
        delete_by_doc(make_client(), doc_id)
    except Exception as e:  # noqa: BLE001
        # Milvus 删除失败不应阻断 MySQL 删除，但要如实报告
        raise HTTPException(500, f"删除向量失败：{type(e).__name__}: {e}") from e
    with session_scope() as s:
        obj = s.get(Document, doc_id)
        if obj is not None:
            s.delete(obj)
    return {"deleted": doc_id}
