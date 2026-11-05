# -*- coding: utf-8 -*-
"""bench_pg.py —— PostgreSQL+pgvector 检索延迟基准（评审错误#3 的实测数字）。

场景：N=20,000 节点（256 维向量，HNSW 余弦索引）+ M=200,000 边。
测量：
  1. HNSW 向量 top-50 召回延迟
  2. 递归 CTE 双向 2 跳图游走延迟（5 种子）
  3. 端到端 retrieve（向量+图+关键词+最近+常驻 五通道，InMemory 管线层）
输出 p50/p95/max（毫秒）。结果写入 stdout，供 docs/04 引用。

用法：PYTHONPATH=src .venv/Scripts/python.exe scripts/bench_pg.py
DSN：env DEUSMITH_PG_DSN_TEST 或默认 postgresql://postgres@127.0.0.1:55433/postgres
"""
from __future__ import annotations

import os
import statistics
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deusmith.embeddings import HashEmbedding  # noqa: E402
from deusmith.models import EdgeType, MemoryEdge  # noqa: E402
from deusmith.store import PostgresStore, _vec_to_pg  # noqa: E402

N_NODES = 20_000
N_EDGES = 200_000
RUNS = 20


def main() -> int:
    import psycopg

    dsn = os.environ.get("DEUSMITH_PG_DSN_TEST",
                         "postgresql://postgres@127.0.0.1:55433/postgres")
    db = f"deusmith_bench_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(dsn, autocommit=True) as admin:
        with admin.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{db}"')
    store = PostgresStore(dsn=dsn.rsplit("/", 1)[0] + "/" + db, dim=256)
    emb = HashEmbedding(dim=256)
    scope = "bench/u1"

    t0 = time.perf_counter()
    store.ensure_schema(256)
    # ---- 批量灌节点（20k，256 维向量）
    topics = ["项目周会记录", "用户偏好设置", "系统架构评审", "旅行计划讨论",
              "健康饮食记录", "技术沙龙笔记", "读书心得摘要", "健身训练日志"]
    with store.conn.cursor() as cur:
        args = []
        for i in range(N_NODES):
            vec = _vec_to_pg(emb.embed(f"{topics[i % len(topics)]}#{i}"))
            args.append((f"bn{i}", "event", f"{topics[i % len(topics)]} 第{i}条",
                         scope, "middle", vec, emb.model_name,
                         f"h{i}"))
        cur.executemany(
            "INSERT INTO memory_node (id, type, content, scope, tier, content_vector, "
            "embedding_model, content_hash) VALUES (%s,%s,%s,%s,%s,%s::vector,%s,%s)",
            args)
    t_nodes = time.perf_counter() - t0

    # ---- 批量灌边（200k 唯一，co_occur + semantic_similar 混合）
    t1 = time.perf_counter()
    with store.conn.cursor() as cur:
        seen_pairs: set = set()
        eargs = []
        i = 0
        while len(eargs) < N_EDGES and i < N_EDGES * 20:
            src = f"bn{i % N_NODES}"
            dst = f"bn{(i * 7919 + (i // N_NODES) * 977 + 13) % N_NODES}"
            i += 1
            if src == dst:
                continue
            etype = "co_occur" if i % 2 else "semantic_similar"
            if (src, dst, etype) in seen_pairs:
                continue
            seen_pairs.add((src, dst, etype))
            eargs.append((f"be{len(eargs)}", src, dst, etype,
                          0.3 + (i % 7) * 0.1, scope))
        cur.executemany(
            "INSERT INTO memory_edge (edge_id, src_id, dst_id, type, weight, scope) "
            "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING", eargs)
    t_edges = time.perf_counter() - t1

    with store.conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM memory_edge")
        m_real = cur.fetchone()["count"]
        cur.execute("ANALYZE memory_node")
        cur.execute("ANALYZE memory_edge")

    qvec = emb.embed("项目周会记录")
    seeds = [f"bn{i}" for i in (1, 500, 5000, 10000, 19999)]

    def bench(fn):
        for _ in range(3):        # 预热（规划器/缓冲池）
            fn()
        lat = []
        for _ in range(RUNS):
            t = time.perf_counter()
            fn()
            lat.append((time.perf_counter() - t) * 1000)
        lat.sort()
        return lat[len(lat) // 2], lat[int(len(lat) * 0.95)], lat[-1]

    v_p50, v_p95, v_max = bench(
        lambda: store.vector_search(scope, qvec, topk=50))
    g_p50, g_p95, g_max = bench(
        lambda: store.neighbors(scope, seeds, hops=2, max_expand=100))
    gu_p50, gu_p95, gu_max = bench(
        lambda: store.neighbors(scope, seeds, hops=2))
    e_p50, e_p95, e_max = bench(
        lambda: (store.vector_search(scope, qvec, topk=50),
                 store.neighbors(scope, seeds, hops=2, max_expand=100),
                 store.keyword_search(scope, "周会", limit=10),
                 store.recent_active(scope, 20),
                 store.resident_nodes(scope)))

    print(f"规模：{N_NODES} 节点 / {m_real} 边（HNSW 256 维）")
    print(f"建库：schema+节点 {t_nodes:.1f}s，边 {t_edges:.1f}s")
    print(f"HNSW 向量 top-50 召回：p50={v_p50:.1f}ms p95={v_p95:.1f}ms max={v_max:.1f}ms")
    print(f"有界 2 跳游走（5 种子，100/跳，生产路径）：p50={g_p50:.1f}ms p95={g_p95:.1f}ms max={g_max:.1f}ms")
    print(f"无界 2 跳游走（对照，稠密图会子图爆炸）：p50={gu_p50:.1f}ms p95={gu_p95:.1f}ms max={gu_max:.1f}ms")
    print(f"五通道全量（向量+图+关键词+最近+常驻，有界）：p50={e_p50:.1f}ms p95={e_p95:.1f}ms max={e_max:.1f}ms")

    store.close()
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
