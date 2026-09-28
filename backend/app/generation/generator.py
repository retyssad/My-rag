"""生成层：prompt 组装、LLM 流式调用、引用解析。

约定（plan.md）：
- System prompt 约束：只依据给定资料回答，用 [1][2] 标来源，资料不足时明说"未提及"。
- 引用只回传被答案实际引用的 chunk；若模型没标，兜底回传全部候选。
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from openai import OpenAI

from app.config import get_settings

SYSTEM_PROMPT = """你是企业电商客服助手。请严格依据下面提供的「参考资料」回答用户问题。

要求：
1. 只使用参考资料中的信息，不要编造、不要依据你自己的知识补充。
2. 引用来源时，在对应句子末尾用 [1]、[2] 这样的编号标注（编号对应参考资料序号）。
3. 若参考资料不足以回答，请明确说明「资料中未提及」，不要猜测。
4. 回答简洁、口语化，适合直接读给客服使用。"""

_CONTEXT_TEMPLATE = """参考资料：
{context}

用户问题：{query}"""


@dataclass
class Citation:
    index: int
    chunk_id: str
    doc_id: int
    doc_title: str
    doc_type: str
    product_category: str
    content: str
    score: float | None = None


def build_context(chunks: list) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] 《{c.doc_title}》（{c.product_category}/{c.doc_type}）\n{c.content}")
    return "\n\n".join(parts)


def build_messages(query: str, chunks: list, history: list[dict] | None = None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append(
        {"role": "user", "content": _CONTEXT_TEMPLATE.format(context=build_context(chunks), query=query)}
    )
    return messages


class LLMClient:
    def __init__(self, model: str | None = None) -> None:
        s = get_settings()
        self._client = OpenAI(api_key=s.dashscope_api_key, base_url=s.dashscope_base_url)
        self.model = model or s.chat_model

    def stream(self, messages: list[dict]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model, messages=messages, stream=True, temperature=0.2
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def complete(self, messages: list[dict], max_tokens: int = 512) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.2, max_tokens=max_tokens
        )
        return resp.choices[0].message.content or ""


_CITE_RE = re.compile(r"\[(\d+)\]")


def parse_citations(answer: str, chunks: list) -> list[Citation]:
    """从答案里提取 [n] 标记；没有标记则兜底回传全部候选。"""
    used: list[int] = []
    for m in _CITE_RE.finditer(answer):
        idx = int(m.group(1))
        if 1 <= idx <= len(chunks) and idx not in used:
            used.append(idx)

    if not used:
        used = list(range(1, len(chunks) + 1))

    out: list[Citation] = []
    for idx in used:
        c = chunks[idx - 1]
        out.append(
            Citation(
                index=idx,
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                doc_title=c.doc_title,
                doc_type=c.doc_type,
                product_category=c.product_category,
                content=c.content,
                score=c.score,
            )
        )
    return out


NO_RESULT_ANSWER = "抱歉，未能在知识库中找到与该问题相关的资料。"
