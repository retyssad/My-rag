# 企业电商客服知识库问答系统

面向企业客服业务场景的 RAG 问答系统：基于私有业务文档检索，回答用户关于商品与售后的问题，并返回引用来源。
定位：求职作品集，核心卖点是**可量化的检索增益**（recall@k / MRR）。

## Language

**知识库 (Knowledge Base)**：
系统据以回答问题的业务文档集合，是检索的唯一事实来源。
_Avoid_: 语料、数据集、资料库

**文档 (Document)**：
一次导入的源文件，是入库处理的单位，带有处理状态与文档类型。一份文档会被切分为多个 Chunk。
_Avoid_: 文件、资料、article

**Chunk**：
从一份 Document 切分得到的可检索文本段，携带元数据，是向量检索与关键词检索的最小单位，也是引用来源的粒度。
_Avoid_: 片段、block、paragraph、分片

**引用来源 (Citation)**：
答案所依据的 Chunk 集合，随答案一并返回给用户。
_Avoid_: 参考、reference、source

**混合检索 (Hybrid Retrieval)**：
同时对同一 query 执行稠密向量检索与稀疏关键词检索，再用融合算法合并两路结果。
_Avoid_: 多路召回、联合检索

**稀疏检索 (Sparse Retrieval)**：
基于关键词词频统计（BM25）的检索路径，与稠密向量的语义检索相区别。
_Avoid_: 关键词检索、全文检索

**融合 (Fusion)**：
把稠密与稀疏两路检索结果合并为单一排序的算法步骤，当前实现为 RRF。
_Avoid_: 合并、rerank

**重排序 (Rerank)**：
对融合后的候选集用交叉编码模型重新打分的步骤，独立于融合，发生在融合之后。
_Avoid_: 精排、二次排序

**查询改写 (Query Rewrite)**：
多轮对话中，结合历史补全 query 的指代与省略，使其可独立检索的预处理步骤。
_Avoid_: 问题重写、query 扩展

**意图识别 (Intent Recognition)**：
在检索前判断问题所属类别（商品咨询 / 售后政策 / 商品对比 / 闲聊），用于调整检索参数或决定是否检索。
_Avoid_: 意图分类、query 分类

**软注入 (Soft Injection)**：
意图识别只调整检索参数（如注入过滤、扩大召回），始终回退到统一检索链路的做法；与之相对的是按意图走不同链路的硬路由。
_Avoid_: 参数覆盖、动态配置

**文档处理状态 (Document Status)**：
一份 Document 在入库流程中的位置：pending → processing → done / failed。
_Avoid_: 任务状态、job state

**文档类型 (Doc Type)**：
Document 的业务分类，用于检索时的元数据过滤与评估分组。
_Avoid_: category、类型标签

**商品类目 (Product Category)**：
Document 所归属的商品品类，用于检索时的元数据过滤与评估分组。
_Avoid_: 品类、category
