# -*- coding: utf-8 -*-
"""PostgresStore 集成测试 —— 真实 PostgreSQL 17 + pgvector 0.8.x。

自动发现：优先 env DEUSMITH_PG_DSN_TEST；否则探测 127.0.0.1:55433。
连不上则整模块 skip（无库环境仍可跑其余单测）。
每个测试运行建独立数据库，每个用例用独立 scope，互不干扰。
"""
import os
import unittest
import uuid

PSYCOPG = None
try:
    import psycopg  # noqa
    PSYCOPG = psycopg
except ImportError:
    pass

_DSN_CANDIDATES = []
if os.environ.get("DEUSMITH_PG_DSN_TEST"):
    _DSN_CANDIDATES.append(os.environ["DEUSMITH_PG_DSN_TEST"])
_DSN_CANDIDATES.append("postgresql://postgres@127.0.0.1:55433/postgres")

_ADMIN_DSN = None
if PSYCOPG:
    for dsn in _DSN_CANDIDATES:
        try:
            with PSYCOPG.connect(dsn, connect_timeout=3, autocommit=True):
                _ADMIN_DSN = dsn
                break
        except Exception:
            continue

DIM = 8


@unittest.skipUnless(PSYCOPG and _ADMIN_DSN, "PostgreSQL 不可达，跳过集成测试")
class TestPostgresStore(unittest.TestCase):
    store = None
    test_db = None

    @classmethod
    def setUpClass(cls):
        from deusmith.store import PostgresStore
        cls.test_db = f"deusmith_pgtest_{uuid.uuid4().hex[:8]}"
        with PSYCOPG.connect(_ADMIN_DSN, autocommit=True) as c:
            with c.cursor() as cur:
                cur.execute(f'CREATE DATABASE "{cls.test_db}"')
        dsn = _ADMIN_DSN.rsplit("/", 1)[0] + "/" + cls.test_db
        cls.store = PostgresStore(dsn=dsn, dim=DIM)
        cls.store.ensure_schema(DIM)

    @classmethod
    def tearDownClass(cls):
        if cls.store:
            cls.store.close()
        with PSYCOPG.connect(_ADMIN_DSN, autocommit=True) as c:
            c.execute(f'DROP DATABASE IF EXISTS "{cls.test_db}" WITH (FORCE)')

    def setUp(self):
        self.scope = f"t/{uuid.uuid4().hex[:8]}"

    def _mk(self, **over):
        from deusmith.models import NodeType, Tier, MemoryNode
        kw = dict(type=NodeType.fact, content=f"集成测试记忆 {uuid.uuid4().hex[:6]}",
                  scope=self.scope, tier=Tier.middle, embedding_model="hash-8",
                  content_vector=(0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0))
        kw.update(over)
        return MemoryNode(**kw)

    # ------------------------------------------------------------ 基础读写
    def test_node_roundtrip_with_vector(self):
        n = self._mk(id="pgA")
        self.assertTrue(self.store.upsert_node(n))
        got = self.store.get_node("pgA")
        self.assertEqual(got.content, n.content)
        self.assertEqual(got.scope, self.scope)
        self.assertEqual(len(got.content_vector), DIM)
        for a, b in zip(got.content_vector, n.content_vector):
            self.assertAlmostEqual(a, b, places=6)
        self.assertFalse(self.store.upsert_node(self._mk(id="pgA")))  # 第二次=更新

    def test_vector_search_ordering_and_sensitivity(self):
        near = self._mk(id="pgN", content_vector=(1.0, 0, 0, 0, 0, 0, 0, 0))
        far = self._mk(id="pgF", content_vector=(0, 0, 0, 0, 0, 0, 0, 1.0))
        secret = self._mk(id="pgS", content_vector=(1.0, 0, 0, 0, 0, 0, 0, 0),
                          sensitivity=3)
        for n in (near, far, secret):
            self.store.upsert_node(n)
        hits = self.store.vector_search(self.scope, (1.0, 0, 0, 0, 0, 0, 0, 0), topk=10)
        ids = [n.id for n, _ in hits]
        self.assertEqual(ids[0], "pgN")
        self.assertNotIn("pgS", ids)                       # 默认门控 2：restricted 不可见
        hits2 = self.store.vector_search(self.scope, (1.0, 0, 0, 0, 0, 0, 0, 0), topk=10,
                                         allowed_sensitivity=3)
        self.assertIn("pgS", [n.id for n, _ in hits2])
        sim_map = {n.id: s for n, s in hits}
        self.assertGreater(sim_map["pgN"], 0.99)
        self.assertLess(sim_map["pgF"], 0.01)

    def test_scope_isolation(self):
        other = self.scope + "b"
        self.store.upsert_node(self._mk(id="pgI1", scope=self.scope))
        self.store.upsert_node(self._mk(id="pgI2", scope=other))
        hits = self.store.vector_search(self.scope, (1.0, 0, 0, 0, 0, 0, 0, 0), topk=10)
        self.assertEqual([n.id for n, _ in hits], ["pgI1"])

    def test_neighbors_recursive_cte(self):
        for nid in ("pgW1", "pgW2", "pgW3", "pgW4"):
            self.store.upsert_node(self._mk(id=nid))
        from deusmith.models import EdgeType, MemoryEdge
        for s, d in (("pgW1", "pgW2"), ("pgW2", "pgW3"), ("pgW3", "pgW4")):
            self.store.upsert_edge(MemoryEdge(src_id=s, dst_id=d,
                                              type=EdgeType.semantic_similar,
                                              scope=self.scope))
        ids, edges = self.store.neighbors(self.scope, ["pgW1"], hops=2)
        self.assertEqual(ids, {"pgW1", "pgW2", "pgW3"})
        self.assertEqual(len(edges), 2)

    # ------------------------------------------------------------ 约束与触发器
    def test_check_constraints_reject_bad_values(self):
        from deusmith.models import NodeType, Tier, MemoryNode
        bad_weight = dict(type=NodeType.fact, content="x", scope=self.scope,
                          tier=Tier.middle, embedding_model="m", weight=1.5)
        with self.assertRaises(Exception):
            self.store.upsert_node(MemoryNode(**bad_weight))
        bad_scope = dict(type=NodeType.fact, content="x", scope="",
                         tier=Tier.middle, embedding_model="m")
        with self.assertRaises(Exception):
            self.store.upsert_node(MemoryNode(**bad_scope))
        bad_pol = dict(type=NodeType.fact, content="x", scope=self.scope,
                       tier=Tier.middle, embedding_model="m", polarity=2)
        with self.assertRaises(Exception):
            self.store.upsert_node(MemoryNode(**bad_pol))

    def test_edge_self_loop_rejected(self):
        from deusmith.models import EdgeType, MemoryEdge
        with self.assertRaises(Exception):
            self.store.upsert_edge(MemoryEdge(src_id="pgE1", dst_id="pgE1",
                                              type=EdgeType.co_occur, scope=self.scope))

    def test_edge_merge_trigger_saturation(self):
        from deusmith.models import EdgeType, MemoryEdge
        self.store.upsert_node(self._mk(id="pgM1"))
        self.store.upsert_node(self._mk(id="pgM2"))
        self.assertTrue(self.store.upsert_edge(MemoryEdge(
            src_id="pgM1", dst_id="pgM2", type=EdgeType.co_occur,
            weight=0.5, scope=self.scope)))
        self.assertFalse(self.store.upsert_edge(MemoryEdge(
            src_id="pgM1", dst_id="pgM2", type=EdgeType.co_occur,
            weight=0.2, scope=self.scope)))
        edges = self.store.get_edges_of(self.scope, ["pgM1"])
        self.assertEqual(len(edges), 1)
        self.assertAlmostEqual(edges[0].weight, 0.6, places=6)   # 1-(1-.5)(1-.2)

    def test_deep_node_cannot_be_deleted_or_content_updated(self):
        self.store.upsert_node(self._mk(id="pgD1", tier="deep", human_confirmed=True))
        with self.store.cursor() as cur:
            with self.assertRaises(Exception):   # I2：DELETE 拦截
                cur.execute("DELETE FROM memory_node WHERE id='pgD1'")
            with self.assertRaises(Exception):   # I2：内容 UPDATE 拦截
                cur.execute("UPDATE memory_node SET content='篡改' WHERE id='pgD1'")

    def test_deprecate_deep_creates_version_and_audit(self):
        self.store.upsert_node(self._mk(id="pgD2", tier="deep", human_confirmed=True))
        self.store.deprecate_deep("pgD2", "集成测试废止")
        node = self.store.get_node("pgD2")
        self.assertEqual(node.status.value, "deprecated")
        with self.store.cursor() as cur:
            cur.execute("SELECT count(*) AS c FROM memory_version WHERE node_id='pgD2'")
            self.assertEqual(cur.fetchone()["c"], 1)
            cur.execute("SELECT count(*) AS c FROM audit_log "
                        "WHERE node_id='pgD2' AND action='deprecate_deep'")
            self.assertEqual(cur.fetchone()["c"], 1)

    def test_deep_identity_policy_requires_human_confirmation(self):
        from deusmith.models import MemoryNode, NodeType, Tier
        with self.assertRaises(Exception):
            self.store.upsert_node(MemoryNode(
                type=NodeType.identity_policy, content="身份", scope=self.scope,
                tier=Tier.deep, embedding_model="m", human_confirmed=False))

    def test_supersedes_cycle_rejected(self):
        """pgC1 ← pgC2（supersedes），再把 pgC1 回指 pgC2 → 成环被拒。"""
        from deusmith.models import MemoryNode
        self.store.upsert_node(self._mk(id="pgC1"))
        self.store.upsert_node(MemoryNode(
            type="fact", content="版本2", scope=self.scope, tier="middle",
            embedding_model="m", id="pgC2", supersedes="pgC1"))
        with self.store.cursor() as cur:
            with self.assertRaises(Exception):
                cur.execute("UPDATE memory_node SET supersedes='pgC2' WHERE id='pgC1'")

    def test_partial_unique_active_hash(self):
        n1 = self._mk(id="pgU1", content="幂等测试")
        self.store.upsert_node(n1)
        with self.assertRaises(Exception):   # I8：active 同 hash 再插入 → 唯一冲突
            self.store.upsert_node(self._mk(id="pgU2", content="幂等测试"))
        # 归档后允许再写
        from deusmith.models import NodeStatus
        n1.status = NodeStatus.archived
        self.store.upsert_node(n1)
        self.store.upsert_node(self._mk(id="pgU3", content="幂等测试"))  # 不再冲突
        self.assertEqual(self.store.get_node("pgU3").id, "pgU3")

    def test_hnsw_index_exists(self):
        with self.store.cursor() as cur:
            cur.execute("SELECT indexname FROM pg_indexes WHERE tablename='memory_node'")
            names = {r["indexname"] for r in cur.fetchall()}
        self.assertIn("idx_node_vector", names)

    # ------------------------------------------------------------ 阶段二新增：池化 / 会话 / 证据 / 时序 / 追踪
    def test_pool_concurrent_reads(self):
        """连接池：8 线程并发检索不崩、结果一致（评审 #4）。"""
        import threading
        self.assertIsNotNone(self.store._pool)
        errs = []

        def worker():
            try:
                for _ in range(10):
                    hits = self.store.vector_search(
                        self.scope, (1.0, 0, 0, 0, 0, 0, 0, 0), topk=5)
                    assert isinstance(hits, list)
            except Exception as e:  # noqa: BLE001
                errs.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errs, [])

    def test_session_roundtrip_pg(self):
        from deusmith.llm import RuleBasedLLM
        from deusmith.params import Params
        from deusmith.shallow import ShallowSession
        s = ShallowSession(self.scope, "pg-sess-1", Params(), RuleBasedLLM())
        s.add_round("user", "你好")
        s.add_round("user", "我喜欢摄影。")
        s.rolling_summary = "会话摘要：偏好摄影"
        self.store.save_session(s)
        raw = self.store.load_session_raw(self.scope, "pg-sess-1")
        self.assertEqual(len(raw["rounds"]), 2)
        self.assertEqual(raw["rounds"][0][1], "你好")
        self.assertEqual(raw["summary"], "会话摘要：偏好摄影")
        self.assertIsNone(self.store.load_session_raw("other", "pg-sess-1"))  # I1

    def test_record_evidence_pg(self):
        self.store.upsert_node(self._mk(id="pgEv1"))
        self.store.record_evidence("pgEv1", "ev_abc123", "chat", "h123",
                                   "file:///e/abc.txt")
        records = self.store.list_evidence("pgEv1")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["evidence_id"], "ev_abc123")

    def test_slot_history_pg(self):
        from datetime import timedelta
        from deusmith.models import utcnow
        now = utcnow()
        old = self._mk(id="pgT1", subject="user", predicate="住在",
                       object_value="上海", status="deprecated",
                       valid_from=now - timedelta(days=60),
                       valid_to=now - timedelta(days=5))
        new = self._mk(id="pgT2", subject="user", predicate="住在",
                       object_value="北京", valid_from=now)
        self.store.upsert_node(old)
        self.store.upsert_node(new)
        hist = self.store.slot_history(self.scope, "user", "住在")
        self.assertEqual([n.id for n in hist], ["pgT1", "pgT2"])
        self.assertIsNotNone(hist[0].valid_to)
        self.assertIsNone(hist[1].valid_to)

    def test_trace_events_pg(self):
        self.store.log_event("retrieval", {"x": 1}, trace_id="tr_pg",
                             scope=self.scope)
        self.store.log_event("inject", {"tokens": 5}, trace_id="tr_pg",
                             scope=self.scope)
        evs = self.store.trace_events("tr_pg")
        self.assertEqual([e["event_type"] for e in evs], ["retrieval", "inject"])
        self.assertEqual(evs[0]["payload"]["x"], 1)


if __name__ == "__main__":
    unittest.main()
