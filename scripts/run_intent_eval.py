"""意图分类准确率评估。

用 gold set 里的 intent 标签作为真值，跑分类器并统计：
- 整体准确率
- 每类的 precision / recall / F1（macro）
- 混淆矩阵

用法：
    uv run python scripts/run_intent_eval.py
    uv run python scripts/run_intent_eval.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "backend")

from app.retrieval.intent import INTENTS, IntentClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "eval" / "gold.jsonl"


def load_gold() -> list[dict]:
    return [
        json.loads(line)
        for line in GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/eval/intent_report.json")
    args = ap.parse_args()

    gold = load_gold()
    if args.limit:
        gold = gold[: args.limit]

    clf = IntentClassifier()
    rows = []
    t0 = time.time()
    for i, item in enumerate(gold, 1):
        truth = item.get("intent", "product_qa")
        # 多轮样本：分类器看到的是改写后的问题更合理，但这里评估分类器本身，
        # 用原始 query（与线上首轮一致）。coref 样本单独用 context 末轮。
        query = item["query"]
        pred = clf.classify(query)
        rows.append(
            {
                "query": query,
                "truth": truth,
                "pred": pred.intent,
                "source": pred.source,
                "correct": pred.intent == truth,
            }
        )
        if i % 25 == 0:
            print(f"  {i}/{len(gold)} ...")

    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n if n else 0.0

    # 混淆矩阵 + 每类 P/R/F1
    conf: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        conf[r["truth"]][r["pred"]] += 1

    per_class = {}
    for label in INTENTS:
        tp = conf[label][label]
        fp = sum(conf[t][label] for t in INTENTS if t != label)
        fn = sum(v for p, v in conf[label].items() if p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = sum(conf[label].values())
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(INTENTS)

    # 误判明细
    errors = [r for r in rows if not r["correct"]]

    report = {
        "n": n,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion": {t: dict(conf[t]) for t in INTENTS},
        "errors": errors,
        "elapsed_sec": round(time.time() - t0, 1),
        "fallback_count": sum(1 for r in rows if r["source"] != "llm"),
    }

    print("\n" + "=" * 62)
    print(f"意图分类准确率: {acc:.4f}  (macro-F1 {macro_f1:.4f}, n={n})")
    print("-" * 62)
    print(f"{'类别':<12}{'P':>8}{'R':>8}{'F1':>8}{'support':>9}")
    for label in INTENTS:
        v = per_class[label]
        print(f"{label:<12}{v['precision']:>8}{v['recall']:>8}{v['f1']:>8}{v['support']:>9}")
    print("=" * 62)
    if errors:
        print(f"\n误判明细（{len(errors)} 条）：")
        for r in errors[:15]:
            print(f"  {r['truth']:>10} -> {r['pred']:<10} | {r['query'][:38]}")
    print(f"\n兜底触发次数: {report['fallback_count']}（含规则快路）")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"详细结果写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
