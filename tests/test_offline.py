# -*- coding: utf-8 -*-
"""离线治理单测 —— §五.8 离线路径：衰减重算 → 压缩合并 → 层级流转 → 边剪枝/环路 → 一致性校验。"""
import unittest
from datetime import datetime, timedelta, timezone

from deusmith.embeddings import HashEmbedding
from deusmith.llm import RuleBasedLLM
from deusmith.models import (
    EdgeStatus, EdgeType, MemoryEdge, MemoryNode, NodeStatus, NodeType, Tier,
)
from deusmith.params import Params
from deusmith.store import InMemoryStore

SCOPE = "t0/u1"
NOW = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)
OLD = NOW - timedelta(days=90)


def mk_node(**over):
    kw = dict(type=NodeType.event, content="离线治理测试", scope=SCOPE,
              tier=Tier.middle, embedding_model="hash-64")
    kw.update(over)
    return MemoryNode(**kw)


class TestOfflineGovernor(unittest.TestCase):
    def setUp(self):
        from deusmith.offline import OfflineGovernor
        self.store = InMemoryStore()
        self.embedder = HashEmbedding(dim=64)
        self.gov = OfflineGovernor(self.store, self.embedder, RuleBasedLLM(),
                                   params=Params())

    def test_edge_decay_applied(self):
        a, b = mk_node(id="oa"), mk_node(id="ob")
        self.store.upsert_node(a)
        self.store.upsert_node(b)
        e = MemoryEdge(src_id="oa", dst_id="ob", type=EdgeType.co_occur,
                       weight=0.5, scope=SCOPE, last_reinforced_at=OLD)
        self.store.upsert_edge(e)
        self.gov.run(SCOPE, now=NOW)
        got = self.store.get_edges_of(SCOPE, ["oa"])[0]
        import math
        self.assertAlmostEqual(got.weight, 0.5 * math.exp(-0.01 * 90), places=6)
        self.assertNotEqual(got.status, EdgeStatus.pruned)   # 未达剪枝条件

    def test_edge_pruned_when_all_three_conditions(self):
        a, b = mk_node(id="pa"), mk_node(id="pb")
        self.store.upsert_node(a)
        self.store.upsert_node(b)
        e = MemoryEdge(src_id="pa", dst_id="pb", type=EdgeType.co_occur,
                       weight=0.05, confidence=0.2, scope=SCOPE,
                       last_reinforced_at=NOW - timedelta(days=70))
        self.store.upsert_edge(e)
        report = self.gov.run(SCOPE, now=NOW)
        self.assertEqual(report["edges_pruned"], 1)
        self.assertEqual(e.status, EdgeStatus.pruned)       # InMemory 持对象引用

    def test_cycle_resolved_by_weakest_edge_removal(self):
        for nid in ("ca", "cb", "cc", "cd"):
            self.store.upsert_node(mk_node(id=nid))
        edges = []
        for s, d, w in (("ca", "cb", 0.5), ("cb", "cc", 0.5), ("cc", "ca", 0.2)):
            e = MemoryEdge(src_id=s, dst_id=d, type=EdgeType.derives_from,
                           weight=w, scope=SCOPE)
            self.store.upsert_edge(e)
            edges.append(e)
        tail = MemoryEdge(src_id="cd", dst_id="ca", type=EdgeType.derives_from,
                          scope=SCOPE)
        self.store.upsert_edge(tail)
        report = self.gov.run(SCOPE, now=NOW)
        self.assertEqual(report["cycles_resolved"], 1)
        weakest = [e for e in edges if (e.src_id, e.dst_id) == ("cc", "ca")][0]
        self.assertEqual(weakest.status, EdgeStatus.pruned)   # 最弱边被剪
        for e in edges:
            if (e.src_id, e.dst_id) != ("cc", "ca"):
                self.assertNotEqual(e.status, EdgeStatus.pruned)
        self.assertNotEqual(tail.status, EdgeStatus.pruned)

    def test_compress_merge_creates_summary_and_archives_originals(self):
        vec = self.embedder.embed("项目周会记录")
        ids = []
        for i, content in enumerate(("周一开了项目周会", "本周项目周会延期", "项目周会纪要归档")):
            n = mk_node(id=f"sm{i}", content=content, content_vector=vec,
                        type=NodeType.event)
            self.store.upsert_node(n)
            ids.append(n.id)
        report = self.gov.run(SCOPE, now=NOW)
        self.assertGreaterEqual(report["summaries_created"], 1)
        summaries = [n for n in self.store.active_nodes(SCOPE)
                     if n.type == NodeType.summary]
        self.assertEqual(len(summaries), 1)
        sm = summaries[0]
        self.assertEqual(sm.generation, 1)
        for mid in ids:
            member = self.store.get_node(mid)
            self.assertEqual(member.status, NodeStatus.archived)   # 原文归档可回查
        edges = self.store.get_edges_of(SCOPE, [sm.id])
        so = [e for e in edges if e.type == EdgeType.summary_of]
        self.assertEqual(len(so), 3)                               # summary_of 边

    def test_high_generation_exempt_from_clustering(self):
        vec = self.embedder.embed("顶层摘要")
        for i in range(2):
            self.store.upsert_node(mk_node(id=f"hg{i}", content=f"顶层摘要{i}",
                                           content_vector=vec,
                                           type=NodeType.summary, generation=2))
        report = self.gov.run(SCOPE, now=NOW)
        self.assertEqual(report["summaries_created"], 0)           # 级联压缩被拦

    def test_tier_transitions_applied(self):
        # 陈旧低权重 → 归档
        self.store.upsert_node(mk_node(id="tt1", weight=0.1,
                                       last_access_at=NOW - timedelta(days=120),
                                       importance=0.2, confidence=0.3))
        # 达标 → 晋升深层
        self.store.upsert_node(mk_node(id="tt2", confidence=0.92, source_auth=0.85,
                                       cross_session_count=6, human_confirmed=True,
                                       content="稳定的核心事实"))
        self.gov.run(SCOPE, now=NOW)
        self.assertEqual(self.store.get_node("tt1").status, NodeStatus.archived)
        self.assertEqual(self.store.get_node("tt2").tier, Tier.deep)
        audits = self.store.recent_audits(limit=20)
        self.assertTrue(any(a["action"] == "promote_deep" for a in audits))

    def test_consistency_check_flags_unevidenced_deep(self):
        self.store.upsert_node(mk_node(id="dp1", tier=Tier.deep, human_confirmed=True,
                                       evidence_refs=()))
        self.gov.run(SCOPE, now=NOW)
        audits = self.store.recent_audits(limit=20)
        self.assertTrue(any(a["action"] == "consistency_warning" and
                            a["node_id"] == "dp1" for a in audits))

    def test_report_has_all_stage_keys(self):
        report = self.gov.run(SCOPE, now=NOW)
        for key in ("edges_decayed", "edges_pruned", "cycles_resolved",
                    "summaries_created", "promoted_deep", "archived",
                    "consistency_warnings"):
            self.assertIn(key, report)


if __name__ == "__main__":
    unittest.main()
