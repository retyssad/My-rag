"""应用配置：从环境变量 / .env 读取。

所有外部依赖的地址、模型名、检索默认参数集中在此，便于替换与测试。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- 百炼（DashScope）---
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "qwen3.7-text-embedding"
    embedding_dim: int = 1024
    rerank_model: str = "qwen3-rerank"
    chat_model: str = "qwen-plus"
    rewrite_model: str = "qwen-turbo"

    # --- MySQL ---
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "rag"
    mysql_password: str = "rag_pass"
    mysql_db: str = "rag"

    # --- Milvus ---
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_collection: str = "chunks"

    # --- 应用 ---
    admin_token: str = "change-me-admin-token"
    log_level: str = "INFO"

    # --- 父子块入库（字符数）；父块仅存 MySQL，子块写入 Milvus ---
    chunk_size: int = Field(default=500, ge=1)
    chunk_overlap: int = Field(default=80, ge=0)
    parent_chunk_size: int = Field(default=1500, ge=1)

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> Settings:
        if self.parent_chunk_size < self.chunk_size:
            raise ValueError("PARENT_CHUNK_SIZE 必须不小于 CHUNK_SIZE")
        return self

    # --- 检索默认配置（可被 /api/chat 请求覆盖）---
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    use_rerank: bool = True
    rerank_top_n: int = 5
    use_query_rewrite: bool = True
    use_intent: bool = True
    min_score_threshold: float = 0.0
    use_parent_retrieval: bool = True

    # --- 可选：LangSmith ---
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "rag-customer-service"

    @property
    def mysql_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}?charset=utf8mb4"
        )

    @property
    def milvus_uri(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
