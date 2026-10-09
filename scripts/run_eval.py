"""检索评估：recall@k / MRR@k / nDCG@k + 消融表。

判定（ADR-0005）：按**文本匹配**判命中——检索返回的 chunk 若包含 gold 的 snippet，
即视为命中该相关文档。不依赖 chunk id，因此切分策略可变。

指标手写（plan Q4）。

用法：
    uv run python scripts/run_eval.py                       # 跑默认消融组
    uv run python scripts/run_eval.py --configs hybrid,hybrid_rerank
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "backend")

from app.retrieval.retriever import RetrievalConfig, Retriever  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "eval" / "gold.jsonl"

# 消融配置：名字 -> RetrievalConfig 覆盖项
ABLATIONS: dict[str, dict] = {
    "dense": {"mode": "dense", "use_rerank": False, "use_intent": False, "use_parent_retrieval": False},
    "sparse": {"mode": "sparse", "use_rerank": False, "use_intent": False, "use_parent_retrieval": False},
    "hybrid": {"mode": "hybrid", "use_rerank": False, "use_intent": False, "use_parent_retrieval": False},
    "hybrid_rerank": {"mode": "hybrid", "use_rerank": True, "use_intent": False, "use_parent_retrieval": False},
    "hybrid_rerank_intent": {"mode": "hybrid", "use_rerank": True, "use_intent": True, "use_parent_retrieval": False},
    "hybrid_rerank_parent": {"mode": "hybrid", "use_rerank": True, "use_intent": False, "use_parent_retrieval": True},
}


def load_gold() -> list[dict]:
    return [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_hit(chunk_content: str, snippet: str) -> bool:
    """文本匹配：归一化后判断包含（去掉空白差异）。"""
    a = "".join(chunk_content.split())
    b = "".join(snippet.split())
    return b in a


def evaluate_one(retriever: Retriever, item: dict, cfg: RetrievalConfig, k: int = 10) -> dict:
    """返回该 query 的各指标所需原始量。

    匹配键用 source_path（稳定），不用 DB 自增 id（重入库会变）。

    注意：chitchat 类样本没有 relevant 文档（预期不检索），
    这里标记 evaluated=False，不计入检索指标。
    """
    query = item["query"]
    run_cfg = cfg.clone()

    # 软注入：意图识别（仅当该档启用）
    if run_cfg.use_intent:
        from app.retrieval.intent import IntentClassifier, apply_intent_to_config

        ir = IntentClassifier().classify(query)
        run_cfg.intent = ir.intent
        short = apply_intent_to_config(ir, run_cfg)
        if short:  # chitchat：不检索
            return {
                "query": query,
                "type": item["type"],
                "evaluated": False,
                "intent": ir.intent,
                "n_relevant": 0,
                "n_hit": 0,
                "first_hit_rank": None,
                "hits_flags": [],
                "chunks": [],
            }

    if not item.get("relevant"):
        # 非 chitchat 却无标注：不计入
        return {
            "query": query, "type": item["type"], "evaluated": False,
            "intent": run_cfg.intent, "n_relevant": 0, "n_hit": 0,
            "first_hit_rank": None, "hits_flags": [], "chunks": [],
        }

    result = retriever.retrieve(query, run_cfg)
    chunks = result.chunks[:k]

    # 每个相关文档（按 source_path 标识）及其目标片段
    relevant_docs = {r["source_path"]: r["snippet"] for r in item["relevant"]}
    hit_docs: set[str] = set()
    first_hit_rank = None
    hits_flags: list[int] = []

    for rank, c in enumerate(chunks, start=1):
        matched = False
        if c.source_path in relevant_docs and is_hit(c.content, relevant_docs[c.source_path]):
            if c.source_path not in hit_docs:  # 同一文档只计一次，避免重复加分
                matched = True
                if first_hit_rank is None:
                    first_hit_rank = rank
            hit_docs.add(c.source_path)
        hits_flags.append(1 if matched else 0)

    total_rel = len(relevant_docs)
    return {
        "query": query,
        "type": item["type"],
        "n_relevant": total_rel,
        "n_hit": len(hit_docs),
        "first_hit_rank": first_hit_rank,
        "hits_flags": hits_flags,
        "chunks": [(c.source_path, c.chunk_id) for c in chunks],
    }


def metrics(rows: list[dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    # 只统计有检索预期的样本（剔除 chitchat 等无标注项）
    rows = [r for r in rows if r.get("evaluated", True) and r["n_relevant"] > 0]
    n = len(rows)
    out = {}

    # recall@k：命中的相关文档数 / 相关文档总数，按 query 平均
    for k in ks:
        total = 0.0
        for r in rows:
            hits_k = sum(r["hits_flags"][:k])
            # 同一文档可能被多个 chunk 命中，最多计到 n_relevant
            hits_k = min(hits_k, r["n_relevant"])
            total += hits_k / r["n_relevant"] if r["n_relevant"] else 0.0
        out[f"recall@{k}"] = round(total / n, 4) if n else 0.0

    # MRR@10：第一个命中排名的倒数
    mrr = 0.0
    for r in rows:
        if r["first_hit_rank"] is not None and r["first_hit_rank"] <= 10:
            mrr += 1.0 / r["first_hit_rank"]
    out["mrr@10"] = round(mrr / n, 4) if n else 0.0

    # nDCG@10：二值相关性
    ndcg = 0.0
    for r in rows:
        dcg = 0.0
        for i, flag in enumerate(r["hits_flags"][:10]):
            if flag:
                dcg += 1.0 / math.log2(i + 2)
        # ideal：前 min(n_relevant, 10) 个位置全命中
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(r["n_relevant"], 10)))
        ndcg += (dcg / ideal) if ideal > 0 else 0.0
    out["ndcg@10"] = round(ndcg / n, 4) if n else 0.0

    return out


def by_type(rows: list[dict]) -> dict[str, dict]:
    types = sorted({r["type"] for r in rows})
    return {t: metrics([r for r in rows if r["type"] == t]) for t in types}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="dense,sparse,hybrid,hybrid_rerank")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试）")
    ap.add_argument("--out", default="data/eval/report.json")
    args = ap.parse_args()

    gold = load_gold()
    if args.limit:
        gold = gold[: args.limit]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    retriever = Retriever()
    report: dict = {"n_queries": len(gold), "configs": {}}

    for name in configs:
        overrides = ABLATIONS.get(name, {})
        cfg = RetrievalConfig.from_settings()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        print(f"\n=== 配置 {name} ({overrides}) ===")
        t0 = time.time()
        rows = []
        for i, item in enumerate(gold, 1):
            rows.append(evaluate_one(retriever, item, cfg))
            if i % 20 == 0:
                print(f"  {i}/{len(gold)} ...")
        m = metrics(rows)
        m["by_type"] = by_type(rows)
        m["elapsed_sec"] = round(time.time() - t0, 1)
        report["configs"][name] = m
        print(f"  recall@1={m['recall@1']} recall@5={m['recall@5']} "
              f"mrr@10={m['mrr@10']} ndcg@10={m['ndcg@10']} ({m['elapsed_sec']}s)")

    # 打印对比表
    print("\n" + "=" * 78)
    header = f"{'配置':<16}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'R@10':>8}{'MRR@10':>9}{'nDCG@10':>9}"
    print(header)
    print("-" * 78)
    for name in configs:
        m = report["configs"][name]
        print(f"{name:<16}{m['recall@1']:>8}{m['recall@3']:>8}{m['recall@5']:>8}"
              f"{m['recall@10']:>8}{m['mrr@10']:>9}{m['ndcg@10']:>9}")
    print("=" * 78)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详细结果写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
