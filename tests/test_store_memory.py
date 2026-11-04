# -*- coding: utf-8 -*-
"""存储层契约测试 —— InMemoryStore（无 PG 环境的单测基准）。

PostgresStore 必须满足同一契约（tests/test_store_pg.py 在真实 PG 上复验）。
契约要点：scope 隔离（I1）、幂等 hash（I8）、边饱和合并（I5）、敏感过滤（I10）。
"""
import unittest
from datetime import datetime, timedelta, timezone

from deusmith.models import (
    EdgeType, MemoryEdge, MemoryNode, NodeStatus, NodeType, Tier,
)


def mk(scope="t0/u1", **over):
    kw = dict(type=NodeType.fact, content="示例", scope=scope,
              tier=Tier.middle, embedding_model="hash-256")
    kw.update(over)
    return MemoryNode(**kw)


class TestInMemoryStore(unittest.TestCase):
    def setUp(self):
        from deusmith.store import InMemoryStore
        self.store = InMemoryStore()
        self.store.ensure_schema()

    def test_node_roundtrip_with_vector(self):
        n = mk(id="mAAA", content_vector=(0.1, 0.2, 0.3))
        self.assertTrue(self.store.upsert_node(n))
        got = self.store.get_node("mAAA")
        self.assertEqual(got.id, "mAAA")
        self.assertEqual(got.content_vector, (0.1, 0.2, 0.3))
        self.assertEqual(got.scope, "t0/u1")
        self.assertFalse(self.store.upsert_node(n))  # 第二次是更新

    def test_missing_node_returns_none(self):
        self.assertIsNone(self.store.get_node("nope"))

    def test_find_active_by_hash(self):
        n = mk(id="mH1")
        self.store.upsert_node(n)
        found = self.store.find_active_by_hash("t0/u1", n.content_hash, "hash-256")
        self.assertIsNotNone(found)
        self.assertEqual(found.id, "mH1")
        self.assertIsNone(self.store.find_active_by_hash("other", n.content_hash, "hash-256"))

    def test_find_by_slot_sees_active_and_contested(self):
        a = mk(id="mS1", subject="u1", predicate="喜欢", object_value="摄影")
        self.store.upsert_node(a)
        hits = self.store.find_by_slot("t0/u1", "u1", "喜欢")
        self.assertEqual([h.id for h in hits], ["mS1"])
        a.status = NodeStatus.contested
        self.store.upsert_node(a)
        hits = self.store.find_by_slot("t0/u1", "u1", "喜欢")
        self.assertEqual(len(hits), 1)  # contested 仍可见（冲突检测需要）
        a.status = NodeStatus.archived
        self.store.upsert_node(a)
        self.assertEqual(self.store.find_by_slot("t0/u1", "u1", "喜欢"), [])

    def test_vector_search_ranks_and_filters_sensitivity(self):
        va = mk(id="mV1", content_vector=(1.0, 0.0), sensitivity=1)
        vb = mk(id="mV2", content_vector=(0.9, 0.1), sensitivity=2)
        vc = mk(id="mV3", content_vector=(0.0, 1.0), sensitivity=3)  # restricted
        for n in (va, vb, vc):
            self.store.upsert_node(n)
        hits = self.store.vector_search("t0/u1", (1.0, 0.0), topk=10)
        ids = [n.id for n, _ in hits]
        self.assertEqual(ids[0], "mV1")
        self.assertIn("mV2", ids)
        self.assertNotIn("mV3", ids)                       # I10：超敏过滤
        hits2 = self.store.vector_search("t0/u1", (1.0, 0.0), topk=10, allowed_sensitivity=3)
        self.assertIn("mV3", [n.id for n, _ in hits2])     # 提高允许级别后可见

    def test_vector_search_scope_isolation(self):
        self.store.upsert_node(mk(id="mA", scope="t0/u1", content_vector=(1.0, 0.0)))
        self.store.upsert_node(mk(id="mB", scope="t0/u2", content_vector=(1.0, 0.0)))
        hits = self.store.vector_search("t0/u1", (1.0, 0.0), topk=10)
        self.assertEqual([n.id for n, _ in hits], ["mA"])  # I1：跨 scope 不可见

    def test_vector_search_skips_archived(self):
        n = mk(id="mAR", content_vector=(1.0, 0.0), status=NodeStatus.archived)
        self.store.upsert_node(n)
        self.assertEqual(self.store.vector_search("t0/u1", (1.0, 0.0), topk=10), [])

    def test_keyword_search(self):
        self.store.upsert_node(mk(id="mK1", content="用户喜欢把会议安排在早上"))
        self.store.upsert_node(mk(id="mK2", content="股价今天大跌"))
        hits = self.store.keyword_search("t0/u1", "会议", limit=5)
        self.assertEqual([h.id for h in hits], ["mK1"])

    def test_neighbors_two_hops(self):
        for nid in ("mA", "mB", "mC", "mD"):
            self.store.upsert_node(mk(id=nid))
        self.store.upsert_edge(MemoryEdge(src_id="mA", dst_id="mB", type=EdgeType.semantic_similar,
                                          scope="t0/u1"))
        self.store.upsert_edge(MemoryEdge(src_id="mB", dst_id="mC", type=EdgeType.semantic_similar,
                                          scope="t0/u1"))
        self.store.upsert_edge(MemoryEdge(src_id="mC", dst_id="mD", type=EdgeType.semantic_similar,
                                          scope="t0/u1"))
        ids, edges = self.store.neighbors("t0/u1", ["mA"], hops=2)
        self.assertEqual(ids, {"mA", "mB", "mC"})          # 2 跳不含 mD
        self.assertEqual(len(edges), 2)
        ids1, _ = self.store.neighbors("t0/u1", ["mA"], hops=1)
        self.assertEqual(ids1, {"mA", "mB"})

    def test_edge_upsert_saturation_merge(self):
        self.store.upsert_node(mk(id="mX"))
        self.store.upsert_node(mk(id="mY"))
        e1 = MemoryEdge(src_id="mX", dst_id="mY", type=EdgeType.co_occur, weight=0.5,
                        scope="t0/u1")
        self.assertTrue(self.store.upsert_edge(e1))
        e2 = MemoryEdge(src_id="mX", dst_id="mY", type=EdgeType.co_occur, weight=0.2,
                        scope="t0/u1")
        self.assertFalse(self.store.upsert_edge(e2))       # 合并非新建
        edges = self.store.get_edges_of("t0/u1", ["mX"])
        self.assertEqual(len(edges), 1)
        self.assertAlmostEqual(edges[0].weight, 1 - (1 - 0.5) * (1 - 0.2), places=9)  # 0.6

    def test_recent_active_ordering(self):
        now = datetime.now(timezone.utc)
        self.store.upsert_node(mk(id="mR1", last_access_at=now - timedelta(days=5)))
        self.store.upsert_node(mk(id="mR2", last_access_at=now))
        rec = self.store.recent_active("t0/u1", limit=10)
        self.assertEqual([n.id for n in rec], ["mR2", "mR1"])

    def test_resident_nodes(self):
        self.store.upsert_node(mk(id="mP1", explicit_remember=True))
        self.store.upsert_node(mk(id="mP2", tier=Tier.deep, human_confirmed=True,
                                  type=NodeType.identity_policy))
        self.store.upsert_node(mk(id="mP3"))
        res = self.store.resident_nodes("t0/u1")
        self.assertEqual({n.id for n in res}, {"mP1", "mP2"})

    def test_log_event_and_audit(self):
        self.store.log_event("write", {"node_id": "mQ"}, trace_id="tr1", scope="t0/u1")
        self.store.audit("deprecate_deep", "mQ", {"reason": "test"})
        events = self.store.recent_events("t0/u1", limit=10)
        self.assertEqual(events[0]["event_type"], "write")
        audits = self.store.recent_audits(limit=10)
        self.assertEqual(audits[0]["action"], "deprecate_deep")

    def test_counts(self):
        self.store.upsert_node(mk(id="mC1"))
        self.store.upsert_node(mk(id="mC2", status=NodeStatus.archived))
        c = self.store.counts("t0/u1")
        self.assertEqual(c["nodes_total"], 2)

    def test_active_nodes_tier_filter(self):
        self.store.upsert_node(mk(id="mT1", tier=Tier.middle))
        self.store.upsert_node(mk(id="mT2", tier=Tier.shallow))
        mids = [n.id for n in self.store.active_nodes("t0/u1", tier=Tier.middle)]
        self.assertEqual(mids, ["mT1"])

    def test_update_node_fields_persisted(self):
        n = mk(id="mU1", weight=0.5)
        self.store.upsert_node(n)
        n.weight = 0.9
        self.store.upsert_node(n)
        self.assertEqual(self.store.get_node("mU1").weight, 0.9)


if __name__ == "__main__":
    unittest.main()
