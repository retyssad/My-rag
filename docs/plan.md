# 实施 Plan — 企业电商客服知识库问答系统（RAG）

> 状态：已确认。相关决策见 `docs/adr/0001~0006`，术语见 `CONTEXT.md`。

## 1. 目标与验收

用 75 篇自建电子产品语料，搭一条模型层可插拔的混合检索 RAG 问答服务，并**用消融表量化各环节增益**。

可演示能力：
- 提问 → 实时显示检索命中的文档 → 流式吐字 → 引用可折叠查看原文
- 上传文档 → 状态 `pending→processing→done` 实时刷新 → 失败可重试
- **消融表**：同一 gold set 下 `dense-only / sparse-only / hybrid / +rerank / +query改写` 的 recall@1/3/5/10、MRR@10、nDCG@10

## 2. 技术栈

| 层 | 选型 |
|---|---|
| 语言/包管理 | Python 3.11 + `uv` |
| Web | FastAPI + Pydantic v2 + `pydantic-settings` |
| SSE | `sse-starlette`（后端）· `@microsoft/fetch-event-source`（前端） |
| 模型 | 阿里云百炼（OpenAI 兼容 `base_url`） |
| Embedding | `qwen3.7-text-embedding`（1024 维，以控制台实际型号为准） |
| Rerank | `qwen3-rerank` |
| 对话 LLM | `qwen-plus`（改写用更廉价型号） |
| 接入方式 | `langchain-openai` 指向兼容 endpoint（**不用** `langchain-community` 的 DashScope 集成） |
| 向量库 | Milvus **v2.6.24** standalone（HNSW/COSINE 稠密 + SPARSE_INVERTED_INDEX/BM25 稀疏） |
| 关键词检索 | **Milvus 内置 BM25**（内置 Jieba 分析器），融合用库内 `RRFRanker`（见 ADR-0001） |
| 关系库 | MySQL 8 + SQLAlchemy 2.0 + Alembic |
| 编排 | LangChain 仅用于 splitter + 模型客户端；检索链路自研（ADR-0003） |
| 前端 | Vue 3 + Vite + TypeScript + Element Plus，两页面 |
| 测试/质量 | `pytest`（纯函数）+ `ruff` |
| 部署 | 开发期中间件 compose；收尾 Dockerfile + prod compose |

## 3. 目录结构

```
rag项目2/
├── CONTEXT.md
├── docs/{plan.md, adr/0001~0006}
├── docker/{docker-compose.yml, docker-compose.prod.yml}
├── backend/
│   ├── app/
│   │   ├── main.py / config.py          # 装配 + Settings(含 RetrievalConfig)
│   │   ├── db/                          # engine/models/session/alembic
│   │   ├── parsing/                     # pdf/docx/html/md → IR
│   │   ├── chunking/                    # IR → chunks（表格旁路）
│   │   ├── indexing/                    # Embedder 抽象 + Milvus schema 写入
│   │   ├── retrieval/                   # dense / sparse / hybrid / rerank / 编排
│   │   ├── generation/                  # prompt 组装 + LLMClient + 引用解析
│   │   ├── api/                         # documents.py / chat.py / schemas.py
│   │   └── services/ingest_service.py   # 后台入库任务
│   └── tests/
├── scripts/{generate_corpus.py, build_gold_set.py, run_eval.py}
├── frontend/                            # ChatView / UploadView
├── data/{raw/**, eval/gold.jsonl}
└── .env.example / .gitignore
```

## 4. 数据模型

**MySQL（事实来源）**
```sql
documents(id, title, doc_type, product_category, source_path,
          status ENUM(pending,processing,done,failed), chunk_count,
          error_msg, created_at, updated_at)
chunks(id, doc_id, chunk_index, content TEXT, token_count,
       chunk_type, milvus_id, metadata_json)
parent_chunks(id VARCHAR(64), doc_id, chunk_index, content TEXT,
              chunk_type, heading_path, token_count)
chunk_parents(child_id VARCHAR(64) PK/FK -> chunks.milvus_id,
              parent_id FK -> parent_chunks.id)
conversations(id, title, created_at)
messages(id, conversation_id, role, content, citations_json, created_at)
```

**Milvus collection `chunks`**：`id`(PK=`{doc_id}_{chunk_index}`)、`dense_vector`(FLOAT_VECTOR 1024)、`content`(VARCHAR，`enable_analyzer=True`，经 `FunctionType.BM25` 生成 sparse)、`sparse_vector`(SPARSE_FLOAT_VECTOR)、标量 `doc_id/doc_type/product_category/chunk_type/source_path`（INVERTED）。稠密索引 HNSW/COSINE，稀疏索引 SPARSE_INVERTED_INDEX/BM25。

父子块（ADR-0008）：父 Chunk 默认 1500 字、无重叠，子 Chunk 默认 500 字、重叠 80 字；只索引子块，父块和映射存 MySQL。表格各生成一个完整父块与子块。`chunk_count` 统计子块数。

## 5. 检索链路

```
query ─(多轮先改写)─┬─ dense search (top20) ─┐
                    ├─ sparse search(top20) ─┤  ← 可单路/混合切换
                    └─ 元数据 pre-filter      │
                                              ↓
                        hybrid_search + RRFRanker(k=60) → top10
                                              ↓
                              子块 qwen3-rerank
                                              ↓
                       阈值过滤 → 批量回查父块 → 去重 top5
                       （无合格结果 → 固定话术，不调 LLM）
                                              ↓
             prompt(编号[1..5]) → LLM 流式 → 解析[n] → citations
```

父块回查由 `use_parent_retrieval` 控制；缺少父子映射的旧文档沿用原 Chunk。关闭回查时直接返回子块。引用与 Prompt 使用相同的上下文正文及 ID，详见 ADR-0008。

`RetrievalConfig`：
```python
dense_top_k=20  sparse_top_k=20  rrf_k=60
mode: Literal["dense","sparse","hybrid"]=  "hybrid"
use_rerank=True  rerank_top_n=5  use_query_rewrite=True
product_category=None  doc_type=None
```
同时作为服务端默认值 + `/api/chat` 可选覆盖参数（消融与现场演示的支点）。

## 6. 里程碑

| 阶段 | 交付物 | 验收 |
|---|---|---|
| **P0** 骨架 | git init + `.gitignore` + compose(etcd/minio/milvus v2.6.24/mysql) + Settings + 冒烟脚本 | 一条命令起中间件，脚本打印 Milvus/MySQL 版本与模型连通性 |
| **P1** 语料 | `generate_corpus.py` → 75 篇（3 类目×25，PDF/DOCX/HTML/MD，含表格/多级标题） | 四格式文件均可打开，含参数表与标题层级 |
| **P2** 入库 | 解析→IR→切分(表格旁路)→embedding→MySQL+Milvus 双写 + 文档 CRUD | 上传一篇→走到 done；MySQL chunk 数与 Milvus 行数一致；稀疏字段可被检索 |
| **P3** 检索问答 | dense/sparse/hybrid 三路 + rerank + `/api/chat` SSE | curl 能看到 retrieval→delta→citations 事件流 |
| **P4** 评估 | gold.jsonl + `run_eval.py` + 消融表 | 产出对比表（与 P3 交错，早跑早调） |
| **P5** 增强 | query 改写（指代消解）+ 会话记忆（近 5 轮） | 多轮"它支持快充吗"改写后能命中 |
| **P6** 前端 | ChatView + UploadView | 两页面可用，引用可折叠，上传状态轮询 |
| **P7** 交付 | Dockerfile + prod compose + README + 截图 | `docker compose -f docker-compose.prod.yml up` 全栈起来 |

关键路径：P1 先于 P2；P4 尽早介入，反向暴露 P2/P3 缺陷；前端后置。

## 7. 风险与待确认

| 风险/项 | 应对 |
|---|---|
| PDF 表格抽取不准（最易翻车） | PyMuPDF 主 + pdfplumber 兜底；P0 先验证 |
| 百炼型号名/维度以控制台为准 | `Embedder` 封装，型号与维度走 Settings；P0 冒烟即验证 |
| Milvus `hybrid_search` + BM25 函数 API 细节 | P0 用最小 collection 验证 `FunctionType.BM25` + `RRFRanker` 可用 |
| LangChain 接入路径（`langchain-openai` 指向 DashScope 兼容口）为推断 | P0 冒烟测试实测（ADR-0003 备注） |
| gold set 质量 | 生成+规则筛，用户终审抽查；分层 simple/coref/cross/filter |
| 范围蔓延 | 管理端/权限/多租户不做；写操作仅 `X-Admin-Token` |

## 8. 不做的事（明确边界）

- 不做答案质量的自动评估（仅 10~20 条人工抽检）
- 不做用户体系、多租户、权限矩阵
- 不做摘要压缩/长期记忆（仅近 5 轮）
- 不接 LangSmith（自建结构化日志；预留开关）
