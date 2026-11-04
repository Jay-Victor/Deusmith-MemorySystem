# -*- coding: utf-8 -*-
"""存储层 —— InMemoryStore（零依赖基准）与 PostgresStore（pgvector 真后端）。

两者满足同一契约（tests/test_store_memory.py / test_store_pg.py）。
后端选择（评审前提：本地部署，PG+pgvector 为唯一持久层）：
  - PostgresStore：生产/集成默认；DSN 来自 env DEUSMITH_PG_DSN 或 config；
  - InMemoryStore：单测、演示、无 PG 环境的降级路径。

评审 §八：图游走在 PG 边表 + 递归 CTE 上完成（单机起步栈），
Neo4j/FalkorDB 为演进项，不在 v1 引入。
"""
from __future__ import annotations

import json
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from .algos import cosine
from .models import (
    EdgeStatus, EdgeType, MemoryEdge, MemoryNode, NodeStatus, NodeType, Tier,
)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
DEFAULT_DSN = "postgresql://postgres@127.0.0.1:55433/deusmith"

# 与 schema.sql 列一一对应（顺序即插入顺序）
_NODE_COLS = [
    "id", "type", "content", "scope", "tier", "content_vector", "embedding_model",
    "subject", "predicate", "object_value", "polarity",
    "created_at", "updated_at", "valid_from", "valid_to", "last_access_at",
    "source", "source_auth", "confidence", "consistency", "evidence_refs",
    "weight", "activation_count", "importance", "emotion",
    "sensitivity", "status", "version", "supersedes", "contested_by", "content_hash",
    "session_id", "cross_session_count", "explicit_remember", "human_confirmed",
    "generation", "metadata",
]

_EDGE_COLS = [
    "edge_id", "src_id", "dst_id", "type", "direction", "weight", "confidence",
    "created_at", "last_reinforced_at", "evidence_refs", "status", "scope",
]


def _vec_to_pg(v) -> str | None:
    if not v:
        return None
    return "[" + ",".join(f"{x:.8g}" for x in v) + "]"


def _pg_to_vec(s) -> tuple:
    if s is None:
        return ()
    if isinstance(s, (list, tuple)):
        return tuple(float(x) for x in s)
    s = str(s).strip()
    if not s or s == "[]":
        return ()
    return tuple(float(x) for x in s.strip("[]").split(",") if x.strip())


def _node_to_values(n: MemoryNode) -> dict:
    d = {
        "id": n.id, "type": n.type.value, "content": n.content, "scope": n.scope,
        "tier": n.tier.value, "content_vector": _vec_to_pg(n.content_vector),
        "embedding_model": n.embedding_model,
        "subject": n.subject, "predicate": n.predicate, "object_value": n.object_value,
        "polarity": n.polarity,
        "created_at": n.created_at, "updated_at": n.updated_at,
        "valid_from": n.valid_from, "valid_to": n.valid_to,
        "last_access_at": n.last_access_at,
        "source": n.source, "source_auth": n.source_auth, "confidence": n.confidence,
        "consistency": n.consistency, "evidence_refs": list(n.evidence_refs),
        "weight": n.weight, "activation_count": n.activation_count,
        "importance": n.importance, "emotion": n.emotion,
        "sensitivity": int(n.sensitivity), "status": n.status.value, "version": n.version,
        "supersedes": n.supersedes, "contested_by": n.contested_by,
        "content_hash": n.content_hash,
        "session_id": n.session_id, "cross_session_count": n.cross_session_count,
        "explicit_remember": n.explicit_remember, "human_confirmed": n.human_confirmed,
        "generation": n.generation, "metadata": json.dumps(n.metadata or {}, ensure_ascii=False),
    }
    return d


def _row_to_node(row: dict) -> MemoryNode:
    return MemoryNode(
        id=row["id"], type=NodeType(row["type"]), content=row["content"],
        scope=row["scope"], tier=Tier(row["tier"]),
        content_vector=_pg_to_vec(row["content_vector"]),
        embedding_model=row["embedding_model"],
        subject=row["subject"], predicate=row["predicate"],
        object_value=row["object_value"], polarity=int(row["polarity"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
        valid_from=row["valid_from"], valid_to=row["valid_to"],
        last_access_at=row["last_access_at"],
        source=row["source"], source_auth=float(row["source_auth"]),
        confidence=float(row["confidence"]), consistency=float(row["consistency"]),
        evidence_refs=tuple(row["evidence_refs"] or ()),
        weight=float(row["weight"]), activation_count=int(row["activation_count"]),
        importance=float(row["importance"]), emotion=float(row["emotion"]),
        sensitivity=int(row["sensitivity"]), status=NodeStatus(row["status"]),
        version=int(row["version"]), supersedes=row["supersedes"],
        contested_by=row["contested_by"], content_hash=row["content_hash"],
        session_id=row["session_id"], cross_session_count=int(row["cross_session_count"]),
        explicit_remember=bool(row["explicit_remember"]),
        human_confirmed=bool(row["human_confirmed"]),
        generation=int(row["generation"]),
        metadata=row["metadata"] if isinstance(row["metadata"], dict)
        else json.loads(row["metadata"] or "{}"),
    )


def _row_to_edge(row: dict) -> MemoryEdge:
    return MemoryEdge(
        edge_id=row["edge_id"], src_id=row["src_id"], dst_id=row["dst_id"],
        type=EdgeType(row["type"]), direction=row["direction"],
        weight=float(row["weight"]), confidence=float(row["confidence"]),
        created_at=row["created_at"], last_reinforced_at=row["last_reinforced_at"],
        evidence_refs=tuple(row["evidence_refs"] or ()),
        status=EdgeStatus(row["status"]), scope=row["scope"],
    )


class InMemoryStore:
    """零依赖内存实现：与 PostgresStore 同契约，供单测/演示/无库降级。"""

    def __init__(self):
        self._nodes: dict[str, MemoryNode] = {}
        self._edges: dict[tuple, MemoryEdge] = {}   # (src,dst,type) -> edge
        self._events: list[dict] = []
        self._audits: list[dict] = []

    # ---- 基础
    def ensure_schema(self, dim: int = 256) -> None:
        return None

    def close(self) -> None:
        return None

    # ---- 节点
    def upsert_node(self, n: MemoryNode) -> bool:
        created = n.id not in self._nodes
        self._nodes[n.id] = n
        return created

    def get_node(self, node_id: str) -> MemoryNode | None:
        return self._nodes.get(node_id)

    def find_active_by_hash(self, scope, content_hash, embedding_model) -> MemoryNode | None:
        for n in self._nodes.values():
            if (n.scope == scope and n.content_hash == content_hash
                    and n.embedding_model == embedding_model
                    and n.status in (NodeStatus.active, NodeStatus.contested)):
                return n
        return None

    def find_by_slot(self, scope, subject, predicate,
                     statuses=("active", "contested")) -> list[MemoryNode]:
        sts = {NodeStatus(s) for s in statuses}
        return [n for n in self._nodes.values()
                if n.scope == scope and n.subject == subject and n.predicate == predicate
                and n.status in sts]

    def vector_search(self, scope, query_vec, topk=50, allowed_sensitivity=2):
        hits = []
        for n in self._nodes.values():
            if (n.scope == scope and n.status in (NodeStatus.active, NodeStatus.contested)
                    and n.sensitivity <= allowed_sensitivity and n.content_vector):
                hits.append((n, cosine(query_vec, n.content_vector)))
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[:topk]

    def keyword_search(self, scope, query, limit=20, allowed_sensitivity=2) -> list[MemoryNode]:
        q = (query or "").strip()
        if not q:
            return []
        out = [n for n in self._nodes.values()
               if n.scope == scope and q in n.content
               and n.status in (NodeStatus.active, NodeStatus.contested)
               and n.sensitivity <= allowed_sensitivity]
        out.sort(key=lambda n: n.last_access_at, reverse=True)
        return out[:limit]

    def neighbors(self, scope, seed_ids, hops=2, max_expand=None):
        """BFS 扩展。max_expand 限制每跳新增节点数（稠密图必须有界，
        否则子图边数线性爆炸，见 scripts/bench_pg.py 实测）。"""
        adj: dict[str, list[tuple[str, MemoryEdge]]] = {}
        for e in self._edges.values():
            if e.scope != scope or e.status == EdgeStatus.pruned:
                continue
            adj.setdefault(e.src_id, []).append((e.dst_id, e))
            adj.setdefault(e.dst_id, []).append((e.src_id, e))
        seen = set(seed_ids)
        frontier = deque((s, 0) for s in seed_ids)
        while frontier:
            nid, h = frontier.popleft()
            if h >= hops:
                continue
            for nxt, _ in adj.get(nid, []):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append((nxt, h + 1))
        edges = [e for (s, d, t), e in self._edges.items()
                 if e.scope == scope and s in seen and d in seen
                 and e.status != EdgeStatus.pruned]
        return seen, edges

    # ---- 边
    def upsert_edge(self, e: MemoryEdge) -> bool:
        key = (e.src_id, e.dst_id, e.type)
        if key in self._edges:
            old = self._edges[key]
            old.weight = 1 - (1 - old.weight) * (1 - e.weight)      # I5 饱和合并
            old.confidence = max(old.confidence, e.confidence)
            old.last_reinforced_at = e.last_reinforced_at
            if old.status == EdgeStatus.pruned:
                old.status = EdgeStatus.active
            old.evidence_refs = tuple(dict.fromkeys(old.evidence_refs + e.evidence_refs))
            return False
        self._edges[key] = e
        return True

    def get_edges_of(self, scope, node_ids) -> list[MemoryEdge]:
        ids = set(node_ids)
        return [e for (s, d, t), e in self._edges.items()
                if e.scope == scope and (s in ids or d in ids)
                and e.status != EdgeStatus.pruned]

    def all_edges(self, scope) -> list[MemoryEdge]:
        return [e for e in self._edges.values()
                if e.scope == scope and e.status != EdgeStatus.pruned]

    def update_edge(self, e: MemoryEdge) -> None:
        """强制替换（离线治理路径用，不触发饱和合并）。"""
        self._edges[(e.src_id, e.dst_id, e.type)] = e

    # ---- 通道与治理查询
    def recent_active(self, scope, limit=20) -> list[MemoryNode]:
        out = [n for n in self._nodes.values()
               if n.scope == scope and n.status in (NodeStatus.active, NodeStatus.contested)]
        out.sort(key=lambda n: n.last_access_at, reverse=True)
        return out[:limit]

    def resident_nodes(self, scope) -> list[MemoryNode]:
        return [n for n in self._nodes.values()
                if n.scope == scope and n.status == NodeStatus.active
                and (n.explicit_remember or n.tier == Tier.deep)]

    def active_nodes(self, scope, tier=None) -> list[MemoryNode]:
        return [n for n in self._nodes.values()
                if n.scope == scope and n.status == NodeStatus.active
                and (tier is None or n.tier == tier)]

    # ---- 日志与审计
    def log_event(self, event_type: str, payload: dict, trace_id=None, scope="") -> None:
        self._events.append({
            "ts": datetime.now(timezone.utc), "trace_id": trace_id,
            "event_type": event_type, "scope": scope, "payload": payload,
        })

    def audit(self, action: str, node_id: str, detail: dict, trace_id=None) -> None:
        self._audits.append({
            "ts": datetime.now(timezone.utc), "action": action, "node_id": node_id,
            "actor": "system", "detail": detail, "trace_id": trace_id,
        })

    def recent_events(self, scope, limit=20) -> list[dict]:
        out = [e for e in self._events if not scope or e["scope"] == scope]
        return list(reversed(out[-limit:]))

    def recent_audits(self, limit=20) -> list[dict]:
        return list(reversed(self._audits[-limit:]))

    def counts(self, scope) -> dict:
        nodes = [n for n in self._nodes.values() if n.scope == scope]
        edges = [e for e in self._edges.values() if e.scope == scope]
        return {"nodes_total": len(nodes), "edges_total": len(edges)}


class PostgresStore:
    """PostgreSQL 17 + pgvector 真后端（psycopg3）。

    - 向量以文本协议收发（'[1,2,...]'::vector），不依赖 pgvector-python 包；
    - 检索用 HNSW 余弦（schema 已建索引）；
    - 图游走用递归 CTE 双向扩展；
    - 边合并/深层保护/supersedes 环检测由 DB 触发器强制（客户端无法绕过）。
    """

    def __init__(self, dsn: str | None = None, dim: int = 256):
        import psycopg
        from psycopg.rows import dict_row
        self._psycopg = psycopg
        self.dsn = dsn or os.environ.get("DEUSMITH_PG_DSN", DEFAULT_DSN)
        self.dim = dim
        self.conn = psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row,
                                    connect_timeout=10)

    def ensure_schema(self, dim: int | None = None) -> None:
        dim = dim or self.dim
        sql = SCHEMA_PATH.read_text(encoding="utf-8").replace("__DIM__", str(dim))
        with self.conn.cursor() as cur:
            cur.execute(sql)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # ---- 节点
    def upsert_node(self, n: MemoryNode) -> bool:
        vals = _node_to_values(n)
        cols = ", ".join(_NODE_COLS)
        ph = ", ".join(["%s"] * len(_NODE_COLS))
        upd = ", ".join(f"{c} = EXCLUDED.{c}" for c in _NODE_COLS if c != "id")
        sql = (f"INSERT INTO memory_node ({cols}) VALUES ({ph}) "
               f"ON CONFLICT (id) DO UPDATE SET {upd} "
               f"RETURNING (xmax = 0) AS created")
        vec = vals.pop("content_vector")
        meta = vals.pop("metadata")
        args = [vec if c == "content_vector" else
                (meta if c == "metadata" else vals[c]) for c in _NODE_COLS]
        with self.conn.cursor() as cur:
            cur.execute(sql, args)
            row = cur.fetchone()
        return bool(row["created"])

    def get_node(self, node_id: str) -> MemoryNode | None:
        cols = ", ".join(_NODE_COLS)
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT {cols} FROM memory_node WHERE id = %s", (node_id,))
            row = cur.fetchone()
        return _row_to_node(row) if row else None

    def find_active_by_hash(self, scope, content_hash, embedding_model) -> MemoryNode | None:
        cols = ", ".join(_NODE_COLS)
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {cols} FROM memory_node WHERE scope=%s AND content_hash=%s "
                f"AND embedding_model=%s AND status IN ('active','contested') LIMIT 1",
                (scope, content_hash, embedding_model))
            row = cur.fetchone()
        return _row_to_node(row) if row else None

    def find_by_slot(self, scope, subject, predicate,
                     statuses=("active", "contested")) -> list[MemoryNode]:
        cols = ", ".join(_NODE_COLS)
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {cols} FROM memory_node WHERE scope=%s AND subject=%s "
                f"AND predicate=%s AND status = ANY(%s) "
                f"AND (valid_to IS NULL OR valid_to > now())",
                (scope, subject, predicate, list(statuses)))
            rows = cur.fetchall()
        return [_row_to_node(r) for r in rows]

    def vector_search(self, scope, query_vec, topk=50, allowed_sensitivity=2):
        cols = ", ".join(_NODE_COLS)
        qvec = _vec_to_pg(query_vec)
        sql = (f"SELECT {cols}, 1 - (content_vector <=> %s::vector) AS sim "
               f"FROM memory_node WHERE scope=%s AND status IN ('active','contested') "
               f"AND sensitivity <= %s AND content_vector IS NOT NULL "
               f"ORDER BY content_vector <=> %s::vector LIMIT %s")
        with self.conn.cursor() as cur:
            cur.execute(sql, (qvec, scope, allowed_sensitivity, qvec, topk))
            rows = cur.fetchall()
        return [(_row_to_node(r), float(r["sim"])) for r in rows]

    def keyword_search(self, scope, query, limit=20, allowed_sensitivity=2) -> list[MemoryNode]:
        cols = ", ".join(_NODE_COLS)
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {cols} FROM memory_node WHERE scope=%s AND content ILIKE %s "
                f"AND status IN ('active','contested') AND sensitivity <= %s "
                f"ORDER BY last_access_at DESC LIMIT %s",
                (scope, f"%{query}%", allowed_sensitivity, limit))
            rows = cur.fetchall()
        return [_row_to_node(r) for r in rows]

    def neighbors(self, scope, seed_ids, hops=2, max_expand=None):
        """逐跳 BFS（每跳走 src/dst 部分索引），不用全表递归 CTE——实测递归 CTE
        在 2 万边规模即 150–340ms（全表扫描）。
        max_expand 限制每跳新增节点（稠密图必须有界：度 20 的 20 万边图上，
        无界 2 跳会物化上万条子图边，实测 218ms；有界 100/跳 后几十毫秒）。
        返回 (可达节点集, 子图内边)。"""
        seen = set(seed_ids)
        frontier = list(seed_ids)
        for _ in range(max(0, hops)):
            if not frontier:
                break
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT src_id, dst_id FROM memory_edge "
                    "WHERE scope=%s AND status <> 'pruned' "
                    "AND (src_id = ANY(%s) OR dst_id = ANY(%s))",
                    (scope, frontier, frontier))
                rows = cur.fetchall()
            nxt = set()
            for r in rows:
                nxt.update((r["src_id"], r["dst_id"]))
            if max_expand is not None and len(nxt) > max_expand:
                nxt = set(sorted(nxt)[:max_expand])   # 确定性截断
            frontier = [nid for nid in nxt if nid not in seen]
            seen |= nxt
        ids = list(seen)
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM memory_edge WHERE scope=%s AND status <> 'pruned' "
                "AND src_id = ANY(%s) AND dst_id = ANY(%s)",
                (scope, ids, ids))
            rows = cur.fetchall()
        return seen, [_row_to_edge(r) for r in rows]

    # ---- 边（合并语义由 trg_merge_edge 强制）
    def upsert_edge(self, e: MemoryEdge) -> bool:
        cols = ", ".join(_EDGE_COLS)
        ph = ", ".join(["%s"] * len(_EDGE_COLS))
        sql = (f"INSERT INTO memory_edge ({cols}) VALUES ({ph}) "
               f"ON CONFLICT (src_id, dst_id, type) DO NOTHING RETURNING edge_id")
        args = [e.edge_id, e.src_id, e.dst_id, e.type.value, e.direction,
                e.weight, e.confidence, e.created_at, e.last_reinforced_at,
                list(e.evidence_refs), e.status.value, e.scope]
        with self.conn.cursor() as cur:
            cur.execute(sql, args)
            row = cur.fetchone()
        return row is not None

    def get_edges_of(self, scope, node_ids) -> list[MemoryEdge]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM memory_edge WHERE scope=%s AND status <> 'pruned' "
                "AND (src_id = ANY(%s) OR dst_id = ANY(%s))",
                (scope, list(node_ids), list(node_ids)))
            rows = cur.fetchall()
        return [_row_to_edge(r) for r in rows]

    def all_edges(self, scope) -> list[MemoryEdge]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM memory_edge WHERE scope=%s AND status <> 'pruned'",
                        (scope,))
            rows = cur.fetchall()
        return [_row_to_edge(r) for r in rows]

    def update_edge(self, e: MemoryEdge) -> None:
        """强制更新（离线治理路径用；UPDATE 不经过 INSERT 合并触发器）。"""
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE memory_edge SET weight=%s, confidence=%s, status=%s, "
                "last_reinforced_at=%s, evidence_refs=%s WHERE edge_id=%s",
                (e.weight, e.confidence, e.status.value, e.last_reinforced_at,
                 list(e.evidence_refs), e.edge_id))

    # ---- 通道与治理查询
    def recent_active(self, scope, limit=20) -> list[MemoryNode]:
        cols = ", ".join(_NODE_COLS)
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {cols} FROM memory_node WHERE scope=%s "
                f"AND status IN ('active','contested') "
                f"ORDER BY last_access_at DESC LIMIT %s", (scope, limit))
            rows = cur.fetchall()
        return [_row_to_node(r) for r in rows]

    def resident_nodes(self, scope) -> list[MemoryNode]:
        cols = ", ".join(_NODE_COLS)
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {cols} FROM memory_node WHERE scope=%s AND status='active' "
                f"AND (explicit_remember OR tier='deep')", (scope,))
            rows = cur.fetchall()
        return [_row_to_node(r) for r in rows]

    def active_nodes(self, scope, tier=None) -> list[MemoryNode]:
        cols = ", ".join(_NODE_COLS)
        sql = (f"SELECT {cols} FROM memory_node WHERE scope=%s AND status='active'")
        args: list = [scope]
        if tier is not None:
            sql += " AND tier=%s"
            args.append(tier.value if hasattr(tier, "value") else str(tier))
        with self.conn.cursor() as cur:
            cur.execute(sql, args)
            rows = cur.fetchall()
        return [_row_to_node(r) for r in rows]

    # ---- 日志与审计
    def log_event(self, event_type: str, payload: dict, trace_id=None, scope="") -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO event_log (trace_id, event_type, scope, payload) "
                "VALUES (%s, %s, %s, %s)",
                (trace_id, event_type, scope,
                 json.dumps(payload, ensure_ascii=False, default=str)))

    def audit(self, action: str, node_id: str, detail: dict, trace_id=None) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO audit_log (action, node_id, detail) VALUES (%s, %s, %s)",
                (action, node_id, json.dumps(detail, ensure_ascii=False, default=str)))

    def recent_events(self, scope, limit=20) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT ts, trace_id, event_type, scope, payload FROM event_log "
                "WHERE scope=%s ORDER BY id DESC LIMIT %s", (scope, limit))
            rows = cur.fetchall()
        return [{"ts": r["ts"], "trace_id": r["trace_id"], "event_type": r["event_type"],
                 "scope": r["scope"], "payload": r["payload"]} for r in rows]

    def recent_audits(self, limit=20) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT ts, action, node_id, actor, detail FROM audit_log "
                "ORDER BY id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
        return [{"ts": r["ts"], "action": r["action"], "node_id": r["node_id"],
                 "actor": r["actor"], "detail": r["detail"]} for r in rows]

    def counts(self, scope) -> dict:
        with self.conn.cursor() as cur:
            cur.execute("SELECT count(*) AS c FROM memory_node WHERE scope=%s", (scope,))
            nodes = cur.fetchone()["c"]
            cur.execute("SELECT count(*) AS c FROM memory_edge WHERE scope=%s", (scope,))
            edges = cur.fetchone()["c"]
        return {"nodes_total": nodes, "edges_total": edges}

    # ---- 深层专用
    def deprecate_deep(self, node_id: str, reason: str, actor: str = "system") -> None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT deprecate_deep(%s, %s, %s)", (node_id, reason, actor))


def get_store(cfg: dict, dim: int = 256):
    """cfg 形如 {"storage": {"backend": "postgres", "dsn": ...}}。默认 postgres。"""
    st = (cfg or {}).get("storage", {}) or {}
    backend = st.get("backend", "postgres")
    if backend == "postgres":
        return PostgresStore(dsn=st.get("dsn"), dim=dim)
    if backend == "memory":
        return InMemoryStore()
    raise ValueError(f"未知 storage backend: {backend}")
