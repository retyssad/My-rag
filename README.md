# 企业电商客服知识库问答系统（RAG）

面向企业客服场景的 RAG 问答系统：从多格式业务文档构建知识库，用**混合检索 + 重排序**回答问题并返回引用来源，支持多轮对话与流式输出。核心卖点是**可量化的检索增益**——用同一套评估集对比各检索环节的贡献。

## 功能

- **多格式文档入库**：PDF / DOCX / HTML / Markdown → 统一中间表示 → 中文递归切分（**表格整表旁路**，不被切碎）
- **父子块**：子块负责检索与重排序（默认 500 字 / 重叠 80 字），回查 MySQL 中的父块（默认 1500 字）作为生成上下文；按父块去重，表格保持完整
- **混合检索**：稠密向量（HNSW/COSINE）+ 稀疏 BM25（Milvus 内置中文分析器），RRF 融合，可选重排序
- **意图识别（软注入）**：识别商品咨询 / 售后政策 / 商品对比 / 闲聊，只调整检索参数而非切换链路；闲聊短路不检索
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
上传文档 → 解析(PDF/DOCX/HTML/MD → IR) → 父子块切分(表格旁路) → ┬→ MySQL(父块/子块/映射，事实来源)
                                                            └→ 子块 Embedding → Milvus(dense + sparse)

                    ┌─────────────── 问答 ───────────────┐
提问 → (多轮则 query 改写) → ┬→ dense 检索 top20 ─┐
                            └→ sparse 检索 top20 ─┴→ RRF 融合 → 子块 rerank → 回查父块/去重 top5 → Prompt → LLM 流式 → 答案 + 引用
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

### 父子块启用与旧知识库升级

新上传文档默认生成父子块。父块和映射通过新增 `parent_chunks` / `chunk_parents` 表保存，启动时自动建表；已有 `chunks` 表和 Milvus collection 无需修改。`Document.chunk_count` 仍统计可检索子块数。

旧文档没有父子映射时继续返回原 Chunk；通过上传页的「重试」可重新处理单篇文档，批量升级仓库中的知识库文档使用：

```bash
uv run python scripts/ingest_corpus.py --reprocess
```

该命令保留文档 ID，重新解析、切分并调用 embedding 更新索引。上传文档不在 manifest 中，需单篇重试。

父块大小由 `PARENT_CHUNK_SIZE=1500` 配置，子块由 `CHUNK_SIZE=500` / `CHUNK_OVERLAP=80` 配置（字符数口径）。短标题或末尾事实句合并时父块可以略超目标，整表不受大小限制。子块仅在所属父块内切分。

`USE_PARENT_RETRIEVAL=false` 或 `/api/chat` 的 `use_parent_retrieval=false` 可关闭回查；前端「扩展上下文」开关支持即时对比。启用时分数来自最优命中子块，生成和引用使用父块正文与父块 ID；SSE 检索结果附带 `parent_id` 和 `matched_child_ids`。

```bash
uv run python scripts/run_eval.py --configs hybrid_rerank,hybrid_rerank_parent --out data/eval/parent_report.json
```

这两档在同一子块索引上对比关闭/开启父块回查。父块档的片段命中衡量扩展上下文的覆盖率；需结合上下文长度判断收益，不能据此直接断言子块排序精度提高。以下历史结果来自原单层切分，父子块尚未做在线指标评估。

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
| **hybrid + rerank** | 0.875 | 0.913 | 0.943 | 0.943 | 0.905 | 0.913 |
| **hybrid + rerank + intent** | 0.875 | 0.913 | 0.943 | **0.947** | 0.905 | **0.915** |

**结论与解读**（不做粉饰）：

1. **重排序是增益最大的环节**：hybrid+rerank 相比 hybrid 提升 R@1 +10.2pt、R@5 +4.9pt。这是本项目最明确的信号。
2. **纯 dense 在本语料上就很强**，hybrid（无权重 RRF）反而不如 dense（R@5 0.894 vs 0.939）。原因是 gold set 中简单题占比高，dense 已接近饱和，融合的边际收益被天花板效应吃掉。
3. **难例上差距才显现**：`hard` 层（口语化，如"充电最快的手机"）与 `coref` 层（指代）中，dense 单路 R@1 为 0，rerank 后才提升——说明 rerank 与 query 改写主要救的是难例。
4. sparse 单路（0.750）弱于 dense，但在含精确型号/参数的查询上更稳，是混合检索的互补项。
5. **意图识别的价值是"零风险增益"，而非大幅提升**：R@5 持平、R@10 与 nDCG 小幅提升。它把 `policy` 收窄到政策文档（并集，不排他）、让 `compare` 扩大召回、让 `chitchat` 短路省一次检索。**它的主要收益在架构完整性与延迟，而非检索指标**——这一点如实标注，不夸大。

### 意图分类准确率

用 gold set 的 `intent` 标签评估（`uv run python scripts/run_intent_eval.py`）：

| 指标 | 值 |
|---|---|
| 准确率 | **0.971** |
| macro-F1 | 0.932 |
| product_qa / policy / compare / chitchat F1 | 0.983 / 0.857 / 0.889 / 1.000 |

剩余误判集中在语义边界（如"屏幕碎了保修吗"介于商品问题与售后政策之间），未继续调优以免对测试集过拟合。

> 已知局限：语料偏薄（76 篇 / 142 chunk），单节正文短，导致**切分策略消融（chunk_size）缺乏区分度**；`simple` 类问题占比偏高稀释了融合增益；`hard` 层（口语化最高级查询，如"降噪最好的耳机"）dense 单路全错，是 RAG 的 top-k 固有难点。以上为后续可改进项，不影响消融表的核心结论。

## 项目结构

```
backend/app/
  config.py            # Settings（含检索默认参数）
  db/                  # SQLAlchemy 模型与会话（MySQL 为事实来源）
  parsing/parser.py    # PDF/DOCX/HTML/MD → 统一 IR（表格单独成块）
  chunking/chunker.py  # IR → 父子块（表格旁路 + 中文递归切分）
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
- [0007](docs/adr/0007-intent-soft-injection.md) 意图识别采用软注入而非硬路由（含 policy 并集的实测教训）
- [0008](docs/adr/0008-parent-child-chunks.md) 子块检索与重排序，父块回查供生成，旧文档兼容

## 测试

```bash
uv run pytest        # 纯函数测试：切分/表格旁路/解析/引用解析/评估指标
uv run ruff check backend scripts
```
