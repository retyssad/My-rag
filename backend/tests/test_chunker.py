"""切分与表格旁路的测试。"""

from __future__ import annotations

from app.chunking.chunker import ChunkingConfig, chunk_ir, table_to_markdown
from app.parsing.parser import IRBlock


def test_table_to_markdown_renders_header_separator():
    md = table_to_markdown([["参数", "值"], ["电池", "5000mAh"]])
    assert "| 参数 | 值 |" in md
    assert "| --- | --- |" in md
    assert "| 电池 | 5000mAh |" in md


def test_table_to_markdown_pads_ragged_rows():
    md = table_to_markdown([["a", "b", "c"], ["1"]])
    lines = md.split("\n")
    assert lines[0].count("|") == 4  # 3 列
    assert lines[2].count("|") == 4  # 补齐


def test_table_is_kept_whole_not_split():
    """表格必须整表一块，不被递归切分。"""
    big_table = IRBlock(type="table", content=[["列1", "列2"]] + [[f"行{i}", "x" * 30] for i in range(40)], level=0)
    drafts = chunk_ir([big_table], ChunkingConfig(chunk_size=50))
    table_chunks = [d for d in drafts if d.chunk_type == "table"]
    assert len(table_chunks) == 1, "表格应作为单个 chunk"
    # 表头与最后一行都应存在（未被截断）
    assert "列1" in table_chunks[0].content
    assert "行39" in table_chunks[0].content


def test_text_split_respects_chunk_size_threshold():
    blocks = [
        IRBlock(type="text", content=f"这是第{i}段内容，" + "填充文字" * 10, level=0)
        for i in range(20)
    ]
    drafts = chunk_ir(blocks, ChunkingConfig(chunk_size=200, chunk_overlap=0))
    assert len(drafts) > 1, "长文本应被切成多块"
    assert all(d.chunk_type == "text" for d in drafts)


def test_heading_path_is_chunk_start_heading_and_content_keeps_inline_headings():
    """契约：heading_path 记录 chunk 起始处的标题栈；各节标题以 `## 标题` 内联进正文。

    正文跨节累积（保证 chunk_size 生效），故单个 chunk 可能跨多节；标题信息不丢失，
    因为它以 Markdown 标题形式内联在 content 中。
    """
    blocks = [
        IRBlock(type="text", content="商品详情", level=1),
        IRBlock(type="text", content="核心参数说明，" + "细节" * 200, level=2),
    ]
    drafts = chunk_ir(blocks, ChunkingConfig(chunk_size=100, chunk_overlap=0))
    assert len(drafts) > 1, "长内容应切成多块"
    assert all(d.heading_path == "商品详情" for d in drafts)
    joined = "".join(d.content for d in drafts)
    assert "核心参数说明" in joined, "节标题应内联在 chunk 正文中，不丢失"


def test_empty_input_yields_no_chunks():
    assert chunk_ir([]) == []
