"""意图识别：判断用户问题属于哪一类，用于**软注入**检索参数。

设计（见 docs/adr/0007）：
- 只调整检索参数，永远回退到 hybrid 检索（软注入），识别错也不崩。
- 唯一例外：chitchat 短路（不检索直接答），因为"不检索"本身是正确的。
- 分类失败/超时/JSON 解析失败 → 兜底 product_qa（最安全的默认）。

意图类别：
- product_qa : 询问商品参数/功能/价格（默认）
- policy     : 询问退换货/保修/配送/发票
- compare    : 要求对比两个及以上商品
- chitchat   : 寒暄、与商品无关
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.config import get_settings
from app.generation.generator import LLMClient

logger = logging.getLogger(__name__)

INTENTS = ("product_qa", "policy", "compare", "chitchat")
DEFAULT_INTENT = "product_qa"

INTENT_PROMPT = """判断用户问题属于下列哪一类，只输出 JSON，不要解释。

类别：
- product_qa: 询问某商品/产品的参数、功能、价格、规格（如"X30 的处理器是什么""支持快充吗"）
- policy: 询问退换货、保修、维修、配送、运费、发票等售后政策（如"怎么退货""保修多久"）
- compare: 明确要求对比两个或多个商品（如"A 和 B 哪个好""对比 X 和 Y"）
- chitchat: 寒暄、与商品和售后无关（如"你好""你是谁"）

用户问题：{query}

只输出形如 {{"intent": "product_qa"}} 的 JSON。"""


@dataclass
class IntentResult:
    intent: str
    source: str  # "llm" | "fallback" | "override"

    @property
    def is_default(self) -> bool:
        return self.intent == DEFAULT_INTENT


# 快速规则：明显寒暄直接判定，省一次 LLM 调用
_CHITCHAT_RE = re.compile(
    r"^\s*(你好|您好|hi|hello|hey|在吗|你是谁|谢谢|多谢|再见|拜拜)[\s!！。.~]*$",
    re.IGNORECASE,
)


class IntentClassifier:
    def __init__(self) -> None:
        s = get_settings()
        # 与 query 改写一样用廉价模型
        self._llm = LLMClient(model=s.rewrite_model)

    def _parse(self, text: str) -> str | None:
        """从模型输出里稳健地抽出 intent。"""
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        intent = str(data.get("intent", "")).strip()
        return intent if intent in INTENTS else None

    def classify(self, query: str) -> IntentResult:
        q = query.strip()
        if not q:
            return IntentResult(DEFAULT_INTENT, "fallback")

        # 规则快路：寒暄
        if _CHITCHAT_RE.match(q):
            return IntentResult("chitchat", "override")

        try:
            out = self._llm.complete(
                [{"role": "user", "content": INTENT_PROMPT.format(query=q)}],
                max_tokens=32,
            )
            intent = self._parse(out)
            if intent:
                return IntentResult(intent, "llm")
            logger.warning("意图识别输出无法解析，兜底 %s：%r", DEFAULT_INTENT, out[:120])
        except Exception as e:  # noqa: BLE001
            logger.warning("意图识别调用失败，兜底 %s：%s", DEFAULT_INTENT, e)

        return IntentResult(DEFAULT_INTENT, "fallback")


def apply_intent_to_config(result: IntentResult, cfg) -> bool:
    """把意图**软注入**为检索参数（就地修改 cfg）。

    返回 True 表示命中短路意图（chitchat，调用方应跳过检索）。
    """
    if result.intent == "policy":
        # 政策问题：倾向政策文档，但**不排他**——取「过滤 + 全量」的并集，
        # 避免误判（如"屏幕碎了保修吗"被当成政策）导致正确文档被过滤掉。
        cfg.doc_type = "policy"
        cfg.union_unfiltered = True
    elif result.intent == "compare":
        # 对比问题：扩大召回，让两篇商品都进来（去重在检索器里做）
        cfg.dense_top_k = max(cfg.dense_top_k, 40)
        cfg.sparse_top_k = max(cfg.sparse_top_k, 40)
        cfg.rerank_top_n = max(cfg.rerank_top_n, 8)
        cfg.diversify_by_doc = True
    elif result.intent == "chitchat":
        return True
    # product_qa：保持默认，不改任何参数
    return False
