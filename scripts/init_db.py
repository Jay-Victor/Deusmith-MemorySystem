# -*- coding: utf-8 -*-
"""init_db.py —— 初始化业务库：建 schema（按配置的 embedding 维度渲染 vector 列）。

用法：
  PYTHONPATH=src .venv/Scripts/python.exe scripts/init_db.py
DSN 优先级：env DEUSMITH_PG_DSN > config.json(storage.dsn) > 默认本机 55433。
维度来源：config.json(embedding) 的 provider 配置（默认 hash-256）。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deusmith.embeddings import get_provider  # noqa: E402
from deusmith.store import DEFAULT_DSN, PostgresStore  # noqa: E402


def main() -> int:
    cfg = {}
    cfg_path = ROOT / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    dsn = os.environ.get("DEUSMITH_PG_DSN") \
        or (cfg.get("storage") or {}).get("dsn") or DEFAULT_DSN
    emb = get_provider(cfg)
    print(f"连接 {dsn} · embedding={emb.model_name} (dim={emb.dim})")
    store = PostgresStore(dsn=dsn, dim=emb.dim)
    try:
        store.ensure_schema(emb.dim)
        counts = store.counts("init/check")
        print(f"schema 就绪：{counts['nodes_total']} 节点 / {counts['edges_total']} 边（空库为 0）")
        print("提示：docs/02-schema.sql 与 src/deusmith/schema.sql 的一致性由 check_sql.py 保证。")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
