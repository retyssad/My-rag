"""Rerank 客户端：调用百炼 qwen3-rerank（非 OpenAI 兼容命名空间，走原生 endpoint）。

失败时**降级**为返回原顺序（不阻断问答），并记日志。
"""

from __future__ import annotations

import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self) -> None:
        s = get_settings()
        self.api_key = s.dashscope_api_key
        self.model = s.rerank_model
        self.url = s.dashscope_base_url.replace(
            "/compatible-mode/v1",
            "/api/v1/services/rerank/text-rerank/text-rerank",
        )

    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """返回 [(原索引, 相关性分数)]，已按分数降序、截断到 top_n。"""
        if not documents:
            return []
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "input": {"query": query, "documents": documents},
                        "parameters": {"top_n": min(top_n, len(documents))},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            results = data.get("output", {}).get("results", [])
            out = [(int(r["index"]), float(r["relevance_score"])) for r in results]
            out.sort(key=lambda x: x[1], reverse=True)
            return out[:top_n]
        except Exception as e:  # noqa: BLE001
            logger.warning("Rerank 失败，降级为原顺序：%s", e)
            return [(i, 0.0) for i in range(min(top_n, len(documents)))]
