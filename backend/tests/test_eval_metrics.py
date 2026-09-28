"""评估指标与命中判定的测试。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from run_eval import is_hit, metrics  # noqa: E402


def _row(hits_flags, first_hit_rank, n_relevant=1, qtype="simple"):
    return {
        "query": "q",
        "type": qtype,
        "n_relevant": n_relevant,
        "n_hit": sum(hits_flags),
        "first_hit_rank": first_hit_rank,
        "hits_flags": hits_flags,
        "chunks": [],
    }


def test_is_hit_ignores_whitespace():
    assert is_hit("支持 120W 快充", "120W")
    assert is_hit("支持120W快充", "120W")
    # 空白差异被归一化，因此 "120W 快充" 也视为命中 "支持120W快充"
    assert is_hit("支持120W快充", "120W 快充")
    # 真正不同的事实不应命中
    assert not is_hit("支持67W快充", "120W")


def test_recall_at_k_perfect():
    rows = [_row([1, 0, 0], 1), _row([0, 1, 0], 2)]
    m = metrics(rows, ks=(1, 3))
    assert m["recall@1"] == 0.5      # 第一条 query 命中，第二条没有
    assert m["recall@3"] == 1.0


def test_mrr_first_hit_rank():
    rows = [_row([1, 0], 1), _row([0, 1], 2)]
    m = metrics(rows, ks=(2,))
    assert m["mrr@10"] == round((1.0 + 0.5) / 2, 4)


def test_ndcg_never_exceeds_one():
    # 二值相关性下 nDCG 必须 <= 1
    rows = [_row([1, 0, 0, 0], 1), _row([1, 1, 0, 0], 1, n_relevant=2)]
    m = metrics(rows, ks=(4,))
    assert m["ndcg@10"] <= 1.0


def test_ndcg_perfect_ranking_is_one():
    rows = [_row([1, 0, 0], 1, n_relevant=1)]
    m = metrics(rows, ks=(3,))
    assert abs(m["ndcg@10"] - 1.0) < 1e-9


def test_metrics_handles_zero_queries_gracefully():
    m = metrics([], ks=(1,))
    assert m["recall@1"] == 0.0
    assert m["mrr@10"] == 0.0
