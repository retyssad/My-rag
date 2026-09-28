# 企业电商客服知识库问答系统（RAG）

面向企业客服场景的 RAG 问答系统：从多格式业务文档构建知识库，用**混合检索 + 重排序**回答问题并返回引用来源，支持多轮对话与流式输出。核心卖点是**可量化的检索增益**——用同一套评估集对比各检索环节的贡献。

## 功能

- **多格式文档入库**：PDF / DOCX / HTML / Markdown → 统一中间表示 → 中文递归切分（**表格整表旁路**，不被切碎）
- **混合检索**：稠密向量（HNSW/COSINE）+ 稀疏 BM25（Milvus 内置中文分析器），RRF 融合，可选重排序
- **元数据过滤**：按商品类目 / 文档类型做 pre-filter
- **多轮对话**：query 改写（补全指代与省略）+ 最近 5 轮记忆
- **SSE 流式问答**：先推检索结果，再逐 token 出答案，最后返回被引用的来源
- **检索消融评估**：`dense / sparse / hybrid / +rerank` 四档对比，手写 recall@k / MRR / nDCG

## 技术栈

| 层 | 选型 |
|---|---|
| 语言 | Python 3.11 + uv |
| Web | FastAPI + Pydantic v2 + sse-starlette |
| 模型 | 阿里云百炼（OpenAI 兼容）：`qwen3.7-text-embedding`(1024d) / `qwen3-rerank` / `qwen-plus` |
| 向量库 | Milvus v2.6.24（standalone：etcd + MinIO + Milvus） |
| 关系库 | MySQL 8 + SQLAlchemy 2.0 |
| 编排 | LangChain 仅用于切分器与模型客户端，检索链路自研 |
| 前端 | Vue 3 + Vite + TypeScript + Element Plus |

## 架构

```
                    ┌─────────────── 入库 ───────────────┐
上传文档 → 解析(PDF/DOCX/HTML/MD → IR) → 切分(表格旁路) → Embedding → ┬→ MySQL(documents/chunks，事实来源)
                                                                      └→ Milvus(dense + sparse，可重建索引)

                    ┌─────────────── 问答 ───────────────┐
提问 → (多轮则 query 改写) → ┬→ dense 检索 top20 ─┐
                            └→ sparse 检索 top20 ─┴→ RRF 融合 → rerank top5 → Prompt → LLM 流式 → 答案 + 引用
```

数据流与设计取舍见 `docs/adr/`；术语表见 `CONTEXT.md`。

## 快速开始（本地开发）

```bash
# 1) 配置
cp .env.example .env        # 填入 DASHSCOPE_API_KEY

# 2) 起中间件（Milvus + MySQL）。端口刻意避开宿主机已有服务
docker compose -f docker/docker-compose.yml up -d
# MySQL 13306 · Milvus 29530 · MinIO 19000/19001

# 3) 安装依赖
uv sync

# 4) 冒烟测试（验证 MySQL / Milvus 内置 BM25 / 百炼三模型连通）
uv run python scripts/smoke.py

# 5) 生成语料并入库（76 篇四格式）
uv run python scripts/generate_corpus.py
uv run python scripts/ingest_corpus.py

# 6) 启动后端 + 前端
uv run uvicorn app.main:app --reload --app-dir backend   # http://127.0.0.1:8000/docs
cd frontend && npm install && npm run dev                # http://127.0.0.1:5173
```

## Docker 一键部署

```bash
cp .env.example .env        # 填入 DASHSCOPE_API_KEY
docker compose -f docker/docker-compose.prod.yml up -d --build
# 打开 http://localhost:8088
```

## 检索消融评估

评估集 `data/eval/gold.jsonl` 共 **132 条**，按难度分层：`simple`（事实型）/ `hard`（口语化）/ `cross`（跨文档对比）/ `filter`（需元数据过滤）/ `coref`（指代，配上下文）。
标注采用**文本片段**而非 chunk id（见 ADR-0005），因此更换切分策略后评估集无需重标。

```bash
uv run python scripts/build_gold_set.py   # 构建 gold set
uv run python scripts/run_eval.py         # 跑四档消融，输出表格
```

### 结果（132 queries）

| 配置 | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| dense only | 0.841 | 0.909 | 0.939 | 0.939 | 0.882 | 0.896 |
| sparse only (BM25) | 0.750 | 0.898 | 0.902 | 0.902 | 0.817 | 0.838 |
| hybrid (RRF) | 0.773 | 0.883 | 0.894 | 0.894 | 0.832 | 0.847 |
| **hybrid + rerank** | **0.875** | **0.913** | **0.943** | **0.943** | **0.905** | **0.913** |

**结论与解读**（不做粉饰）：

1. **重排序是增益最大的环节**：hybrid+rerank 相比 hybrid 提升 R@1 +10.2pt、R@5 +4.9pt。这是本项目最明确的信号。
2. **纯 dense 在本语料上就很强**，hybrid（无权重 RRF）反而不如 dense（R@5 0.894 vs 0.939）。原因是 gold set 中简单题占比高，dense 已接近饱和，融合的边际收益被天花板效应吃掉。
3. **难例上差距才显现**：`hard` 层（口语化，如"充电最快的手机"）与 `coref` 层（指代）中，dense 单路 R@1 为 0，rerank 后才提升——说明 rerank 与 query 改写主要救的是难例。
4. sparse 单路（0.750）弱于 dense，但在含精确型号/参数的查询上更稳，是混合检索的互补项。

> 已知局限：语料偏薄（76 篇 / 142 chunk），单节正文短，导致**切分策略消融（chunk_size）缺乏区分度**；`simple` 类问题占比偏高稀释了融合增益。这两点是后续可改进项，不影响上述四项核心结论。

## 项目结构

```
backend/app/
  config.py            # Settings（含检索默认参数）
  db/                  # SQLAlchemy 模型与会话（MySQL 为事实来源）
  parsing/parser.py    # PDF/DOCX/HTML/MD → 统一 IR（表格单独成块）
  chunking/chunker.py  # IR → chunks（表格旁路 + 中文递归切分）
  indexing/            # Embedder 抽象 + Milvus schema（dense + 内置 BM25 稀疏）
  retrieval/           # 三档检索 + rerank + query 改写
  generation/          # prompt 组装 + 引用解析 + LLM 流式
  api/                 # documents CRUD + chat SSE
  services/            # 入库服务（状态机 + 幂等）
scripts/               # 语料生成 / 批量入库 / gold set / 评估 / 冒烟
frontend/src/views/    # ChatView（检索过程+引用折叠）/ UploadView（状态轮询）
docs/adr/              # 架构决策记录
```

## 设计决策（ADR）

- [0001](docs/adr/0001-milvus-builtin-bm25.md) 稀疏检索用 Milvus 内置 BM25，融合用库内 RRFRanker
- [0002](docs/adr/0002-mysql-source-of-truth.md) MySQL 为事实来源，Milvus 为可重建索引
- [0003](docs/adr/0003-langchain-light-usage.md) LangChain 只用于切分与模型客户端，检索链路自研
- [0004](docs/adr/0004-sync-data-layer.md) 数据访问层同步，仅 LLM 流式异步
- [0005](docs/adr/0005-gold-set-snippet-granularity.md) gold set 标注到文本片段而非 chunk id
- [0006](docs/adr/0006-milvus-version-pin.md) Milvus 版本锁定 2.6 线

## 测试

```bash
uv run pytest        # 纯函数测试：切分/表格旁路/解析/引用解析/评估指标
uv run ruff check backend scripts
```
