"""FastAPI 应用装配。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, documents
from app.config import get_settings
from app.db.models import Base
from app.db.session import engine
from app.indexing.milvus_store import ensure_collection, make_client
from app.services.ingest_service import reconcile_stuck_documents

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 建表（作品集用 create_all；正式化可换 Alembic 迁移）
    Base.metadata.create_all(engine)
    try:
        ensure_collection(make_client())
    except Exception as e:  # noqa: BLE001
        logger.warning("Milvus collection 初始化失败（可能 Milvus 未就绪）：%s", e)
    try:
        n = reconcile_stuck_documents()
        if n:
            logger.warning("启动对账：%d 篇卡在 processing 的文档已置为 failed", n)
    except Exception as e:  # noqa: BLE001
        logger.warning("启动对账失败：%s", e)
    yield


app = FastAPI(title="企业电商客服知识库问答系统", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 作品集；生产应限定来源
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
