# LangChain 只用于切分与模型客户端，检索链路自研

## 背景

技术栈声明使用 LangChain 实现 RAG 全链路。LangChain 提供了开箱即用的 RAG 抽象（`RetrievalQA`、`MultiQueryRetriever`、`ConversationalRetrievalChain` 等），使用它们可以用极少的代码得到一条完整链路。

## 决定

LangChain 只用于两件事：**文本切分器**（`TextSplitter` 系列）与**模型客户端接入**（Embedding / Chat 模型）。以下部分全部自研：

- 检索链路的编排与可配置开关（单路 / 混合对照）——注意融合本身由 Milvus `hybrid_search` + `RRFRanker` 承担，见 ADR-0001
- Rerank 调用与截断
- Query 改写（指代消解）
- 引用来源的提取、编号与绑定
- SSE 流式协议实现
- 向量库访问直接走 `pymilvus`

不使用 `RetrievalQA` / `ConversationalRetrievalChain` 等高层 Chain。

## 理由

1. **能力错配**：本项目的核心卖点——混合检索、RRF 融合、rerank、分环节消融——恰恰是 LangChain 高层抽象最弱的区域。用高层 Chain 反而要不断绕过框架的默认行为。
2. **可解释性即评分点**：项目定位是求职作品集，面试必问"RRF 怎么实现的""引用编号怎么绑定的"。答"框架内置"会直接减分；自研链路则每个环节都能展开讲。
3. **消融实验的前提**：要在同一套 gold set 上对比 `dense-only / sparse-only / hybrid / +rerank / +改写`，检索链路必须是**可配置、可拆装**的。高层 Chain 把步骤耦合在一起，无法干净地单独开关。（融合由库内 RRFRanker 完成，但「走单路还是走 hybrid_search」这一开关仍在应用层，消融得以成立。）
4. **依赖风险**：LangChain 版本迭代快、高层 API 变动频繁（DashScope 集成即已出现弃用）。只依赖其稳定底层（splitter、模型客户端）可显著降低返工风险。

## 后果

- 需要自行实现 RRF、rerank 调用、引用解析等约数百行代码——这是有意为之的成本。
- 技术栈"基于 LangChain"的表述依然成立（切分与模型接入确实使用），但需在 README 中明确说明边界，避免夸大。
- 模型客户端被封装为 `Embedder` / `LLMClient` 抽象，使"外部 API / 本地模型"可插拔。
