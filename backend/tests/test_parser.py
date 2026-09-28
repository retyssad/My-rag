"""解析层测试：Markdown/HTML 的 IR 结构与表格识别。"""

from __future__ import annotations

from app.parsing.parser import parse_html, parse_md


def test_parse_md_extracts_headings_and_paragraphs():
    blocks = parse_md("# 标题一\n\n正文内容。\n\n## 标题二\n\n更多正文。")
    levels = [(b["type"], b["level"]) for b in blocks]
    assert ("text", 1) in levels
    assert ("text", 2) in levels
    assert any(b["type"] == "text" and b["level"] == 0 for b in blocks)


def test_parse_md_extracts_table_as_single_block():
    md = "| 参数 | 值 |\n| --- | --- |\n| 电池 | 5000mAh |\n"
    blocks = parse_md(md)
    tables = [b for b in blocks if b["type"] == "table"]
    assert len(tables) == 1
    assert tables[0]["content"] == [["参数", "值"], ["电池", "5000mAh"]]


def test_parse_md_skips_table_separator_row():
    md = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    blocks = parse_md(md)
    rows = blocks[0]["content"]
    assert rows == [["a", "b"], ["1", "2"]]


def test_parse_html_extracts_table():
    html = "<html><body><h1>标题</h1><p>段落</p><table><tr><th>参数</th><th>值</th></tr><tr><td>重量</td><td>1.2kg</td></tr></table></body></html>"
    blocks = parse_html(html)
    tables = [b for b in blocks if b["type"] == "table"]
    assert len(tables) >= 1
    assert ["参数", "值"] in tables[0]["content"]


def test_parse_html_ignores_script_style():
    html = "<html><body><script>var x=1</script><style>.a{}</style><p>正文</p></body></html>"
    blocks = parse_html(html)
    text = " ".join(str(b["content"]) for b in blocks)
    assert "var x" not in text
    assert "正文" in text
