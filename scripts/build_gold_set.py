"""构造 gold set（评估集）。

标注粒度（ADR-0005）：(query, 分层类型, relevant=[{doc_id, source_path, snippet}])
- snippet 是文档中**真实存在**的连续文本，评估时按文本匹配判命中，不依赖 chunk id，
  因此可自由做切分策略消融。

分层类型：
- simple : 单跳事实型，query 与原文用词接近
- hard   : 同义/口语化改写，考验语义检索（dense 更有优势）
- filter : 需要元数据过滤（类目/文档类型）才能稳定命中
- cross  : 跨文档对比型，正确结果需要多篇文档同时命中
- coref  : 带指代/省略，需要结合上下文改写（配 context 字段）

输出：data/eval/gold.jsonl（提交进仓库以支持复现）
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_corpus import (  # noqa: E402
    COMPARISON_PAIRS,
    FAQ_TEMPLATES,
    PRODUCTS,
)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "eval" / "gold.jsonl"


def load_manifest() -> list[dict]:
    return json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))


def find(manifest: list[dict], *, title: str | None = None, category: str | None = None,
         doc_type: str | None = None) -> dict | None:
    for m in manifest:
        if title and m["title"] != title:
            continue
        if category and m["product_category"] != category:
            continue
        if doc_type and m["doc_type"] != doc_type:
            continue
        return m
    return None


def entry(query: str, qtype: str, relevant: list[dict], context: list[dict] | None = None,
          note: str = "", intent: str = "product_qa") -> dict:
    e = {
        "query": query,
        "type": qtype,
        "intent": intent,          # 期望的意图标签（用于分类准确率评估）
        "relevant": relevant,
    }
    if context:
        e["context"] = context
    if note:
        e["note"] = note
    return e


def rel(doc: dict, snippet: str) -> dict:
    return {"doc_id": doc["doc_id"], "source_path": doc["source_path"], "snippet": snippet}


def _file_contains(path: Path, snippet: str) -> bool:
    """跨格式判断源文件是否包含某片段（用于把 snippet 绑定到真实文档）。"""
    try:
        if path.suffix == ".md":
            return snippet in path.read_text(encoding="utf-8")
        if path.suffix in (".html", ".htm"):
            from bs4 import BeautifulSoup
            return snippet in BeautifulSoup(path.read_text(encoding="utf-8"), "lxml").get_text(" ", strip=True)
        if path.suffix == ".docx":
            import docx
            d = docx.Document(str(path))
            text = "\n".join(p.text for p in d.paragraphs)
            for t in d.tables:
                for r in t.rows:
                    text += "\n" + " ".join(c.text for c in r.cells)
            return snippet in text
        if path.suffix == ".pdf":
            import pymupdf
            d = pymupdf.open(str(path))
            try:
                return any(snippet in pg.get_text() for pg in d)
            finally:
                d.close()
    except Exception:  # noqa: BLE001
        return False
    return False


def main() -> None:
    man = load_manifest()
    gold: list[dict] = []

    # ---------------- 1) 商品详情：simple 事实型（处理器/屏幕/电池/售价） ----------------
    for _category, products in PRODUCTS.items():
        for brand, model, price, chip, screen, _battery, _extra in products:
            doc = find(man, title=f"{brand} {model} 商品详情")
            if not doc:
                continue
            gold.append(entry(f"{brand} {model} 的处理器是什么？", "simple", [rel(doc, chip)], intent="product_qa"))
            gold.append(entry(f"{brand} {model} 多少钱？", "simple", [rel(doc, f"{price} 元")], intent="product_qa"))
            gold.append(entry(f"{brand} {model} 的屏幕规格？", "simple", [rel(doc, screen)], intent="product_qa"))

    # ---------------- 2) 商品详情：hard 口语化/同义 ----------------
    hard_map = {
        "phone": [
            ("想买个打游戏不卡的手机，哪个处理器强一点", "骁龙8 Gen3"),
            ("有没有电池大一点的手机", "5400mAh"),
            ("充电最快的手机是哪款", "120W"),
        ],
        "laptop": [
            ("想买台轻的笔记本带着出差", "1.19kg"),
            ("哪款笔记本显卡最好", "RTX 4080"),
            ("做视频剪辑用哪个笔记本合适", "Game 16"),
        ],
        "earbuds": [
            ("降噪最好的耳机是哪款", "50dB"),
            ("戴着跑步不容易掉的耳机", "IP55"),
            ("耳机能撑一整天吗", "45小时"),
        ],
    }
    for category, items in hard_map.items():
        for q, snippet in items:
            # 找到包含该 snippet 的商品详情文档（直接在源文件里搜）
            doc = None
            for m in man:
                if m["product_category"] != category or m["doc_type"] != "product_detail":
                    continue
                p = ROOT / m["source_path"]
                if _file_contains(p, snippet):
                    doc = m
                    break
            if doc:
                gold.append(entry(q, "hard", [rel(doc, snippet)], intent="product_qa"))

    # ---------------- 3) FAQ ----------------
    for category, faqs in FAQ_TEMPLATES.items():
        for _idx, (q, a) in enumerate(faqs):
            doc = None
            for m in man:
                if m["product_category"] == category and m["doc_type"] == "faq" and m["title"].endswith(q):
                    doc = m
                    break
            if doc:
                # 用解答的前半段作为 snippet（保证连续存在）
                snippet = a[:12]
                gold.append(entry(q, "simple", [rel(doc, snippet)], intent="product_qa"))

    # ---------------- 4) 对比类：cross ----------------
    # 说明：COMPARISON_PAIRS 里有两类语义——
    #   (a) "A 与 B 怎么选"  → 真·对比（intent=compare）
    #   (b) "xx 选购指南/怎么挑" → 开放推荐（intent=product_qa），
    #       它不是"A vs B"，归为对比会与意图定义冲突（实测模型判 product_qa 更合理）。
    def _is_pairwise(t: str) -> bool:
        # 形如「X30 Pro 与 X30 怎么选」：含具体型号 + "与/和/区别/差异/对比"
        return bool(
            re.search(r"[A-Za-z0-9]+\s*(?:与|和|vs|VS)\s*[A-Za-z0-9]+", t)
            or re.search(r"(区别|差异|对比)", t)
        )

    for _category, pairs in COMPARISON_PAIRS.items():
        for title, _subjects in pairs:
            doc = find(man, title=title)
            if doc and _file_contains(ROOT / doc["source_path"], "核心差异"):
                q_intent = "compare" if _is_pairwise(title) else "product_qa"
                gold.append(entry(title + "？", "cross", [rel(doc, "核心差异")], intent=q_intent))
    # 跨文档对比（需要两篇同时命中）
    cross_pairs = [
        ("对比一下 X30 Pro 和 X30 的处理器差异", "phone"),
        ("Y9 Ultra 和 Y9 的电池哪个大", "phone"),
        ("Book 14 Pro 和 Book 14 有什么区别", "laptop"),
    ]
    for q, category in cross_pairs:
        rels = [
            rel(m, m["title"].replace(" 商品详情", ""))
            for m in man
            if m["product_category"] == category and m["doc_type"] == "product_detail"
        ][:2]
        if rels:
            gold.append(entry(q, "cross", rels[:2], intent="compare"))

    # ---------------- 5) filter：需要元数据过滤 ----------------
    filter_cases = [
        ("耳机的退换货政策是什么？", "earbuds", "policy", "七天无理由退货"),
        ("笔记本的保修期是多久？", "laptop", "policy", "保修 12 个月"),
        ("手机的配送运费怎么算？", "phone", "policy", "满 99 元包邮"),
        ("耳机坏了去哪里修？", "earbuds", "policy", "联系客服报修"),
        ("笔记本支持七天无理由吗？", "laptop", "policy", "七天无理由退货"),
        ("手机的发票能开专票吗？", "phone", "policy", "增值税专用发票"),
    ]
    for q, category, doc_type, snippet in filter_cases:
        # 政策文档：找该 (类目, 类型) 下含 snippet 的文档
        doc = None
        for m in man:
            if m["product_category"] == category and m["doc_type"] == doc_type:
                if _file_contains(ROOT / m["source_path"], snippet):
                    doc = m
                    break
        if doc:
            gold.append(entry(q, "filter", [rel(doc, snippet)], intent="policy"))

    # ---------------- 6) coref：带指代，需上下文改写 ----------------
    coref_cases = [
        # (本轮问题, 上下文, 类目, 期望 snippet)
        ("它支持快充吗？",
         [{"role": "user", "content": "星耀 X30 Pro 怎么样？"},
          {"role": "assistant", "content": "星耀 X30 Pro 搭载骁龙8 Gen3，屏幕 6.78英寸 2K 120Hz。"}],
         "phone", "120W"),
        ("那款耳机的降噪多少？",
         [{"role": "user", "content": "极光 Sonic 2 Pro 值得买吗？"},
          {"role": "assistant", "content": "极光 Sonic 2 Pro 主打主动降噪。"}],
         "earbuds", "50dB"),
        ("这个笔记本多重？",
         [{"role": "user", "content": "云雀 Air 13 适合出差吗？"},
          {"role": "assistant", "content": "云雀 Air 13 是轻薄本。"}],
         "laptop", "1.19kg"),
    ]
    for q, ctx, category, snippet in coref_cases:
        # 找到含 snippet 的商品详情文档
        doc = None
        for m in man:
            if m["product_category"] != category or m["doc_type"] != "product_detail":
                continue
            if _file_contains(ROOT / m["source_path"], snippet):
                doc = m
                break
        if doc:
            gold.append(entry(q, "coref", [rel(doc, snippet)], context=ctx, intent="product_qa"))

    # ---------------- 7) chitchat：寒暄/无关，不检索 ----------------
    chitchat_cases = [
        "你好",
        "你是谁？",
        "在吗",
        "谢谢你的帮助",
    ]
    for q in chitchat_cases:
        gold.append(entry(q, "chitchat", [], intent="chitchat"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    from collections import Counter

    print(f"写入 {len(gold)} 条 -> {OUT}")
    print("分层分布:", dict(Counter(g["type"] for g in gold)))


if __name__ == "__main__":
    main()
