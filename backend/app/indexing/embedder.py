"""Embedding 客户端：封装百炼（OpenAI 兼容）。

抽象成类便于替换为本地模型（plan 中「模型层可插拔」）。
批量调用 + 指数退避重试（tenacity）。
"""

from __future__ import annotations

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

# 百炼 text-embedding 单次批量上限（保守取 10，见控制台以实际为准）
BATCH_SIZE = 10


class Embedder:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, dim: int | None = None) -> None:
        s = get_settings()
        self._client = OpenAI(
            api_key=api_key or s.dashscope_api_key,
            base_url=base_url or s.dashscope_base_url,
        )
        self.model = model or s.embedding_model
        self.dim = dim or s.embedding_dim

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        # 按 index 排序，确保与输入顺序一致
        data = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in data]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            out.extend(self._embed_batch(texts[i : i + BATCH_SIZE]))
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
