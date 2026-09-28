"""Query 改写：多轮对话中补全指代与省略（plan.md Q3-A）。

只补全、不扩写；只在有历史的多轮时触发；用廉价模型（rewrite_model）。
"""

from __future__ import annotations

import logging

from app.config import get_settings
from app.generation.generator import LLMClient

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """你是一个查询改写助手。下面是一个多轮对话，最后一行是用户的最新问题。
请把最新问题改写成一个**可以独立检索**的完整问题：补全其中的指代（如"它""这个""那款"）
和省略的主语/宾语。只输出改写后的问题本身，不要解释、不要加引号。

对话历史：
{history}

最新问题：{query}
改写后的问题："""


class QueryRewriter:
    def __init__(self) -> None:
        s = get_settings()
        self._llm = LLMClient(model=s.rewrite_model)

    def rewrite(self, query: str, history: list[dict]) -> str:
        """history: [{"role": "user"/"assistant", "content": ...}]，正序。"""
        if not history:
            return query
        hist_text = "\n".join(
            f"{'用户' if h['role'] == 'user' else '客服'}：{h['content']}" for h in history[-6:]
        )
        try:
            out = self._llm.complete(
                [{"role": "user", "content": REWRITE_PROMPT.format(history=hist_text, query=query)}],
                max_tokens=128,
            )
            out = out.strip().strip("“”\"'")
            return out or query
        except Exception as e:  # noqa: BLE001
            logger.warning("Query 改写失败，使用原问题：%s", e)
            return query
