"""意图识别测试：规则快路、JSON 解析、软注入、文档去重。

不调真实 LLM：分类器的 _parse 与规则快路是纯逻辑，直接测；
apply_intent_to_config 与 diversify_by_doc 是纯函数。
"""

from __future__ import annotations

from app.retrieval.intent import IntentClassifier, apply_intent_to_config
from app.retrieval.retriever import RetrievalConfig, RetrievedChunk, diversify_by_doc


def _chunk(chunk_id: str, doc_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, content=f"c{chunk_id}", doc_id=doc_id, doc_title=f"d{doc_id}",
        doc_type="faq", product_category="phone", chunk_type="text",
        source_path=f"p/{doc_id}.md", score=1.0,
    )


# ---------------- 规则快路 ----------------

def test_chitchat_rule_fastpath():
    c = IntentClassifier()
    r = c.classify("你好")
    assert r.intent == "chitchat"
    assert r.source == "override"  # 未调用 LLM


def test_chitchat_rule_variants():
    c = IntentClassifier()
    for q in ["您好", "hi", "Hello", "在吗", "谢谢", "再见"]:
        assert c.classify(q).intent == "chitchat", q


def test_non_chitchat_not_fastpathed():
    """含商品词的问题不应被规则误判为寒暄。"""
    from app.retrieval.intent import _CHITCHAT_RE
    assert _CHITCHAT_RE.match("你好吗，我想问下手机保修") is None
    assert _CHITCHAT_RE.match("退货怎么弄") is None


# ---------------- JSON 解析 ----------------

def test_parse_valid_json():
    c = IntentClassifier()
    assert c._parse('{"intent": "policy"}') == "policy"


def test_parse_json_with_surrounding_text():
    c = IntentClassifier()
    assert c._parse('好的，结果是 {"intent": "compare"} 。') == "compare"


def test_parse_rejects_unknown_intent():
    c = IntentClassifier()
    assert c._parse('{"intent": "unknown_label"}') is None


def test_parse_rejects_garbage():
    c = IntentClassifier()
    assert c._parse("我觉得这是个政策问题") is None
    assert c._parse("{not json}") is None


# ---------------- 软注入 ----------------

def test_policy_injects_doc_type_filter():
    cfg = RetrievalConfig()
    from app.retrieval.intent import IntentResult
    short = apply_intent_to_config(IntentResult("policy", "llm"), cfg)
    assert short is False
    assert cfg.doc_type == "policy"


def test_policy_uses_union_not_exclusive_filter():
    """policy 必须取并集，防止误判导致正确文档被过滤掉（ADR-0007）。"""
    cfg = RetrievalConfig()
    from app.retrieval.intent import IntentResult
    apply_intent_to_config(IntentResult("policy", "llm"), cfg)
    assert cfg.union_unfiltered is True


def test_compare_widens_recall_and_enables_diversify():
    cfg = RetrievalConfig(dense_top_k=20, sparse_top_k=20, rerank_top_n=5)
    from app.retrieval.intent import IntentResult
    short = apply_intent_to_config(IntentResult("compare", "llm"), cfg)
    assert short is False
    assert cfg.dense_top_k >= 40
    assert cfg.sparse_top_k >= 40
    assert cfg.rerank_top_n >= 8
    assert cfg.diversify_by_doc is True


def test_compare_does_not_shrink_larger_configs():
    """已是更大的 top_k 时不应被改小。"""
    cfg = RetrievalConfig(dense_top_k=100, sparse_top_k=100, rerank_top_n=20)
    from app.retrieval.intent import IntentResult
    apply_intent_to_config(IntentResult("compare", "llm"), cfg)
    assert cfg.dense_top_k == 100
    assert cfg.rerank_top_n == 20


def test_chitchat_short_circuits():
    cfg = RetrievalConfig()
    from app.retrieval.intent import IntentResult
    short = apply_intent_to_config(IntentResult("chitchat", "override"), cfg)
    assert short is True
    assert cfg.doc_type is None  # 不注入过滤


def test_product_qa_changes_nothing():
    cfg = RetrievalConfig()
    before = (cfg.dense_top_k, cfg.sparse_top_k, cfg.doc_type, cfg.diversify_by_doc)
    from app.retrieval.intent import IntentResult
    short = apply_intent_to_config(IntentResult("product_qa", "llm"), cfg)
    assert short is False
    assert (cfg.dense_top_k, cfg.sparse_top_k, cfg.doc_type, cfg.diversify_by_doc) == before


# ---------------- 文档去重 ----------------

def test_diversify_keeps_one_per_doc():
    chunks = [_chunk("a", 1), _chunk("b", 1), _chunk("c", 2), _chunk("d", 2), _chunk("e", 3)]
    out = diversify_by_doc(chunks, max_per_doc=1)
    assert [c.doc_id for c in out] == [1, 2, 3]


def test_diversify_preserves_order_and_max():
    chunks = [_chunk("a", 1), _chunk("b", 2), _chunk("c", 1), _chunk("d", 3)]
    out = diversify_by_doc(chunks, max_per_doc=2)
    assert [c.chunk_id for c in out] == ["a", "b", "c", "d"]  # 各 doc 均未超 2


def test_diversify_empty():
    assert diversify_by_doc([]) == []


# ---------------- config clone ----------------

def test_config_clone_is_independent():
    cfg = RetrievalConfig()
    c2 = cfg.clone()
    c2.doc_type = "policy"
    c2.dense_top_k = 99
    assert cfg.doc_type is None
    assert cfg.dense_top_k == 20
