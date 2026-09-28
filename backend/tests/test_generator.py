"""生成层纯函数测试：引用解析与 prompt 组装。"""

from __future__ import annotations

from app.generation.generator import build_context, build_messages, parse_citations
from app.retrieval.retriever import RetrievedChunk


def _chunk(cid: str, title: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, content=f"{title} 的内容", doc_id=1, doc_title=title,
        doc_type="faq", product_category="phone", chunk_type="text",
        source_path=f"data/raw/phone/{cid}.md", score=0.9,
    )


def test_parse_citations_extracts_marked_indices():
    chunks = [_chunk("a", "A"), _chunk("b", "B"), _chunk("c", "C")]
    cites = parse_citations("答案是 B[2]，另见[3]。", chunks)
    assert [c.index for c in cites] == [2, 3]
    assert cites[0].doc_title == "B"


def test_parse_citations_dedupes_repeated_marks():
    chunks = [_chunk("a", "A")]
    cites = parse_citations("见[1]，又如[1]。", chunks)
    assert len(cites) == 1


def test_parse_citations_falls_back_to_all_when_no_marker():
    chunks = [_chunk("a", "A"), _chunk("b", "B")]
    cites = parse_citations("没有任何标记的答案。", chunks)
    assert [c.index for c in cites] == [1, 2]


def test_parse_citations_ignores_out_of_range_indices():
    chunks = [_chunk("a", "A")]
    cites = parse_citations("引用[9]不存在。", chunks)
    # [9] 越界被忽略，无有效标记 → 兜底返回全部候选
    assert [c.index for c in cites] == [1]


def test_build_context_numbers_chunks_from_one():
    chunks = [_chunk("a", "A"), _chunk("b", "B")]
    ctx = build_context(chunks)
    assert "[1]" in ctx and "[2]" in ctx
    assert "《A》" in ctx


def test_build_messages_includes_system_and_history():
    chunks = [_chunk("a", "A")]
    history = [{"role": "user", "content": "上一问"}, {"role": "assistant", "content": "上一答"}]
    msgs = build_messages("新问题", chunks, history)
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "上一问"
    assert "新问题" in msgs[-1]["content"]
    assert "参考资料" in msgs[-1]["content"]
