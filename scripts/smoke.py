"""P0 冒烟测试：验证外部依赖连通性与关键 API 可用性。

运行：uv run python scripts/smoke.py
逐项打印 [OK]/[FAIL]，不因单项失败而中断。
"""

from __future__ import annotations

import sys
import traceback

sys.path.insert(0, "backend")

from app.config import get_settings  # noqa: E402

settings = get_settings()
results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    flag = "OK  " if ok else "FAIL"
    print(f"[{flag}] {name}" + (f" :: {detail}" if detail else ""))


# ---------- 1. MySQL ----------
def check_mysql() -> None:
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(settings.mysql_url, pool_pre_ping=True)
        with engine.connect() as conn:
            version = conn.execute(text("SELECT VERSION()")).scalar()
        record("MySQL 连通", True, f"version={version}")
    except Exception as e:  # noqa: BLE001
        record("MySQL 连通", False, f"{type(e).__name__}: {e}")


# ---------- 2. Milvus：连接 / BM25 函数 / hybrid_search + RRF ----------
def check_milvus() -> None:
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=settings.milvus_uri)
        record("Milvus 连接", True, f"uri={settings.milvus_uri}")
    except Exception as e:  # noqa: BLE001
        record("Milvus 连接", False, f"{type(e).__name__}: {e}")
        return

    # 2a. 尝试用 BM25 Function + 稀疏向量建表，验证 2.6 内置全文检索可用
    coll = "smoke_bm25"
    try:
        from pymilvus import DataType, Function, FunctionType

        if client.has_collection(coll):
            client.drop_collection(coll)

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("content", DataType.VARCHAR, max_length=4096, enable_analyzer=True)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

        schema.add_function(
            Function(
                name="bm25_fn",
                function_type=FunctionType.BM25,
                input_field_names=["content"],
                output_field_names=["sparse"],
                params={},
            )
        )

        index_params = client.prepare_index_params()
        index_params.add_index("dense", index_type="HNSW", metric_type="COSINE")
        index_params.add_index(
            "sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25"
        )

        client.create_collection(coll, schema=schema, index_params=index_params)
        record("Milvus 内置 BM25（FunctionType.BM25）", True, "collection 创建成功")
    except Exception as e:  # noqa: BLE001
        record("Milvus 内置 BM25（FunctionType.BM25）", False, f"{type(e).__name__}: {e}")
        return

    # 2b. 插入几条假数据，验证 hybrid_search + RRFRanker 能跑
    try:
        import random

        rows = []
        for i, txt in enumerate(
            ["手机支持快充", "耳机续航时间", "笔记本散热设计", "退换货政策说明"]
        ):
            rows.append(
                {
                    "id": f"t{i}",
                    "content": txt,
                    "dense": [random.random() for _ in range(settings.embedding_dim)],
                }
            )
        client.insert(coll, rows)
        client.flush(coll)  # 必须 flush 后才能被检索到

        from pymilvus import AnnSearchRequest, RRFRanker

        dense_req = AnnSearchRequest(
            data=[[random.random() for _ in range(settings.embedding_dim)]],
            anns_field="dense",
            param={"metric_type": "COSINE"},
            limit=3,
        )
        sparse_req = AnnSearchRequest(
            data=["快充"],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=3,
        )
        res = client.hybrid_search(
            collection_name=coll,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=settings.rrf_k),
            limit=3,
            output_fields=["content"],
        )
        n = len(res[0]) if res else 0
        record("hybrid_search + RRFRanker", True, f"返回 {n} 条")
    except Exception as e:  # noqa: BLE001
        record("hybrid_search + RRFRanker", False, f"{type(e).__name__}: {e}")
    finally:
        try:
            client.drop_collection(coll)
        except Exception:  # noqa: BLE001
            pass


# ---------- 3. 百炼：Embedding / Chat / Rerank ----------
def check_dashscope() -> None:
    if not settings.dashscope_api_key or settings.dashscope_api_key.startswith("sk-xxxx"):
        record("百炼 API Key", False, "未配置（.env 中 DASHSCOPE_API_KEY 为空或占位符）")
        return

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url
        )
        emb = client.embeddings.create(
            model=settings.embedding_model, input=["这是一条连通性测试"]
        )
        dim = len(emb.data[0].embedding)
        ok = dim == settings.embedding_dim
        record(
            "百炼 Embedding",
            True,
            f"model={settings.embedding_model} dim={dim} (配置={settings.embedding_dim}"
            + ("" if ok else " ⚠️不一致")
            + ")",
        )
    except Exception as e:  # noqa: BLE001
        record("百炼 Embedding", False, f"{type(e).__name__}: {e}")

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url
        )
        resp = client.chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "user", "content": "回复两个字：连通"}],
            max_tokens=10,
        )
        record("百炼 Chat", True, f"model={settings.chat_model} -> {resp.choices[0].message.content!r}")
    except Exception as e:  # noqa: BLE001
        record("百炼 Chat", False, f"{type(e).__name__}: {e}")

    # Rerank 不在 OpenAI 兼容命名空间下，用 DashScope 原生 endpoint
    try:
        import httpx

        url = settings.dashscope_base_url.replace(
            "/compatible-mode/v1", "/api/v1/services/rerank/text-rerank/text-rerank"
        )
        r = httpx.post(
            url,
            headers={"Authorization": f"Bearer {settings.dashscope_api_key}"},
            json={
                "model": settings.rerank_model,
                "input": {"query": "快充", "documents": ["支持快充", "续航很长"]},
                "parameters": {"top_n": 2},
            },
            timeout=30,
        )
        if r.status_code == 200:
            data = r.json()
            n = len(data.get("output", {}).get("results", []))
            record("百炼 Rerank", True, f"model={settings.rerank_model} 返回 {n} 条")
        else:
            record("百炼 Rerank", False, f"HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:  # noqa: BLE001
        record("百炼 Rerank", False, f"{type(e).__name__}: {e}")


def main() -> int:
    print(f"=== 冒烟测试 base_url={settings.dashscope_base_url} ===\n")
    check_mysql()
    check_milvus()
    check_dashscope()

    print("\n=== 汇总 ===")
    failed = [n for n, ok, _ in results if not ok]
    for name, ok, _detail in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    if failed:
        print(f"\n失败 {len(failed)} 项：{', '.join(failed)}")
        return 1
    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
