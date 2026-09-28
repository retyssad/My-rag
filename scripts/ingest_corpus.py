"""批量入库语料（P2 验收 / P4 评估前置）。

用法：
    uv run python scripts/ingest_corpus.py          # 全量入库（跳过已 done）
    uv run python scripts/ingest_corpus.py --reset  # 清空 MySQL+Milvus 后重建
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "backend")

from app.db.models import Base, Document, DocumentStatus  # noqa: E402
from app.db.session import engine, session_scope  # noqa: E402
from app.indexing.milvus_store import (  # noqa: E402
    ensure_collection,
    flush_collection,
    make_client,
)
from app.services.ingest_service import create_document, process_document  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="清空后重建")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 篇（调试用）")
    args = ap.parse_args()

    Base.metadata.create_all(engine)
    client = make_client()
    if args.reset:
        from sqlalchemy import delete

        with session_scope() as db:
            db.execute(delete(Document))
        if client.has_collection("chunks"):
            client.drop_collection("chunks")
        print("已清空 MySQL documents 与 Milvus collection")
    ensure_collection(client)

    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))
    if args.limit:
        manifest = manifest[: args.limit]

    # 跳过已 done 的（按 source_path 判重）
    with session_scope() as db:
        from sqlalchemy import select

        done_paths = {
            r[0]
            for r in db.execute(
                select(Document.source_path).where(Document.status == DocumentStatus.done)
            ).all()
        }

    todo = [m for m in manifest if m["source_path"] not in done_paths]
    print(f"待入库 {len(todo)} / {len(manifest)} 篇")

    ok, fail = 0, 0
    t0 = time.time()
    for i, m in enumerate(todo, 1):
        doc_id = create_document(
            title=m["title"],
            doc_type=m["doc_type"],
            product_category=m["product_category"],
            source_path=m["source_path"],
        )
        # 批量模式：不逐篇 flush，最后统一 flush（否则每篇多等数秒）
        process_document(doc_id, flush=False)
        with session_scope() as db:
            doc = db.get(Document, doc_id)
            status = doc.status.value
            nchunks = doc.chunk_count
            err = doc.error_msg
        if status == "done":
            ok += 1
            print(f"  [{i}/{len(todo)}] ✓ {m['title'][:30]} ({nchunks} chunks)")
        else:
            fail += 1
            print(f"  [{i}/{len(todo)}] ✗ {m['title'][:30]} :: {err}")

    # 统一 flush 一次，使全部数据可被检索
    print("统一 flush 中 ...")
    flush_collection(client)
    print("flush 完成")

    dt = time.time() - t0
    print(f"\n完成：成功 {ok}，失败 {fail}，耗时 {dt:.1f}s")

    with session_scope() as db:
        from sqlalchemy import func, select

        total_chunks = db.scalar(select(func.count()).select_from(Document.__table__))
    milvus_rows = client.query("chunks", filter="", output_fields=["count(*)"])[0]["count(*)"]
    print(f"文档数 {total_chunks} | Milvus 行数 {milvus_rows}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
