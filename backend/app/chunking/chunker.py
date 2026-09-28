"""切分层：IR → chunks。

规则（见 plan.md / ADR-0005）：
- 表格旁路：整表作为一个 chunk，Markdown 表格格式存正文（不参与递归切分）。
- 文本：按标题分段，段内用 LangChain 递归切分（字符数口径），并带上标题路径。
- 每个 chunk 记录 chunk_type（text/table）与来源标题。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.parsing.parser import IRBlock

# 中文递归分隔符优先级
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


@dataclass
class ChunkDraft:
    content: str
    chunk_type: str  # "text" | "table"
    heading_path: str = ""
    token_count: int = 0


@dataclass
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 80
    min_chunk_len: int = 20
    separators: list[str] = field(default_factory=lambda: list(SEPARATORS))


def table_to_markdown(rows: list[list[str]]) -> str:
    """把二维表格转为 Markdown 表格；列数不齐时补齐。"""
    if not rows:
        return ""
    n_cols = max(len(r) for r in rows)
    norm = [list(r) + [""] * (n_cols - len(r)) for r in rows]
    header = norm[0]
    body = norm[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * n_cols) + " |"]
    for r in body:
        lines.append("| " + " | ".join(c.replace("\n", " ") for c in r) + " |")
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """粗略估计：中文按字符数近似，英文按 4 字符 1 token。"""
    return max(1, len(text) // 2)


def chunk_ir(blocks: list[IRBlock], config: ChunkingConfig | None = None) -> list[ChunkDraft]:
    cfg = config or ChunkingConfig()
    # 防御：overlap 必须小于 chunk_size，否则 LangChain 直接报错
    overlap = min(cfg.chunk_overlap, max(0, cfg.chunk_size - 1))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=overlap,
        separators=cfg.separators,
        keep_separator=True,
        length_function=len,
        is_separator_regex=False,
    )

    drafts: list[ChunkDraft] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    buffer: list[str] = []
    buf_heading: str = ""          # chunk 起始处的标题路径（见下方说明）
    buf_len: int = 0

    # 设计说明：正文跨"节"累积到 chunk_size 再切，以保证 chunk_size 参数真正影响切分
    # （语料单节很短，若逐节切分则 chunk_size 失效）。代价是一个 chunk 可能跨多节，
    # 因此 heading_path 记录的是「chunk 起始处」的标题栈；各节标题已以 `## 标题` 内联进
    # chunk 正文，检索与生成时仍可见正确标题。

    def current_heading() -> str:
        return " > ".join(t for _, t in heading_stack)

    def flush_text() -> None:
        nonlocal buffer, buf_heading, buf_len
        if not buffer:
            return
        joined = "\n".join(buffer).strip()
        heading = buf_heading
        buffer, buf_len, buf_heading = [], 0, ""
        if len(joined) < cfg.min_chunk_len:
            return
        # 切分后，过短的片段「并入下一片」而不是丢弃——否则短标题/短事实句会丢失
        pieces = [p.strip() for p in splitter.split_text(joined) if p.strip()]
        merged: list[str] = []
        pending = ""
        for piece in pieces:
            pending = f"{pending}\n{piece}" if pending else piece
            if len(pending) >= cfg.min_chunk_len:
                merged.append(pending)
                pending = ""
        if pending:  # 末尾残留：并入上一片，避免丢弃
            if merged:
                merged[-1] = f"{merged[-1]}\n{pending}"
            else:
                merged.append(pending)

        for piece in merged:
            drafts.append(
                ChunkDraft(
                    content=piece,
                    chunk_type="text",
                    heading_path=heading,
                    token_count=_estimate_tokens(piece),
                )
            )

    for block in blocks:
        if block["type"] == "table":
            # 表格旁路：先落盘已有正文，再整表成块（不参与递归切分）
            flush_text()
            md = table_to_markdown(block["content"])  # type: ignore[arg-type]
            if md:
                drafts.append(
                    ChunkDraft(
                        content=md,
                        chunk_type="table",
                        heading_path=current_heading(),
                        token_count=_estimate_tokens(md),
                    )
                )
            continue

        level = block["level"]
        text = str(block["content"]).strip()
        if level > 0:
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
            # 标题不单独切：若已积累较多内容，先落盘，避免 chunk 跨度过大
            if buf_len >= cfg.chunk_size:
                flush_text()
            if not buffer:
                buf_heading = current_heading()
            buffer.append(f"{'#' * level} {text}")
            buf_len += len(text) + level + 1
        else:
            if not buffer:
                buf_heading = current_heading()
            buffer.append(text)
            buf_len += len(text) + 1

        # 累积到 chunk_size 即切，使 chunk_size 参数真正生效
        if buf_len >= cfg.chunk_size:
            flush_text()

    flush_text()
    return drafts
