# -*- coding: utf-8 -*-
"""权重防垄断单测 —— 设计隐忧落地：Gini 监控 + 节点权重惰性慢衰减。

失效模式（评审遗留）：检索命中 → weight 只升不降 → 长期运行后集体饱和到 1.0，
失去区分度；少数节点垄断榜单。
分工原则：recency 快时钟（τ≈7–20 天）管"新鲜度"；λ_w 慢时钟（半衰期约一年）
管"强化历史强度"。floor 以下不动，交给归档逻辑。
"""
import unittest
from datetime import datetime, timedelta, timezone

from deusmith.models import MemoryNode, NodeType, Tier
from deusmith.params import Params

NOW = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)
P = Params()


def mk(**over):
    kw = dict(type=NodeType.fact, content="权重测试", scope="t0/u1",
              tier=Tier.middle, embedding_model="hash-64")
    kw.update(over)
    return MemoryNode(**kw)


class TestGini(unittest.TestCase):
    def test_perfect_equality_is_zero(self):
        from deusmith.algos import gini
        self.assertEqual(gini([0.5] * 10), 0.0)

    def test_max_inequality(self):
        from deusmith.algos import gini
        self.assertAlmostEqual(gini([0.0] * 19 + [1.0]), 0.95, places=6)

    def test_empty_single_allzero(self):
        from deusmith.algos import gini
        self.assertEqual(gini([]), 0.0)
        self.assertEqual(gini([0.7]), 0.0)
        self.assertEqual(gini([0.0, 0.0]), 0.0)

    def test_scale_invariance(self):
        """Gini 对比例缩放不变（对加性平移不不变——标准性质，勿写错断言）。"""
        from deusmith.algos import gini
        self.assertAlmostEqual(gini([0.2, 0.4, 0.6, 0.8]),
                               gini([0.6, 1.2, 1.8, 2.4]), places=9)


class TestNodeWeightDecay(unittest.TestCase):
    def test_recently_accessed_unchanged(self):
        from deusmith.algos import decay_node_weight
        n = mk(weight=0.9, last_access_at=NOW)
        self.assertAlmostEqual(decay_node_weight(n, NOW, P), 0.9, places=9)

    def test_idle_weight_decays_slowly(self):
        from deusmith.algos import decay_node_weight
        n = mk(weight=0.9, last_access_at=NOW - timedelta(days=365))
        w = decay_node_weight(n, NOW, P)
        self.assertLess(w, 0.5)     # 0.9·exp(-0.73)≈0.44
        self.assertGreater(w, 0.3)

    def test_floor_kept_for_very_old(self):
        from deusmith.algos import decay_node_weight
        n = mk(weight=0.9, last_access_at=NOW - timedelta(days=3650))
        self.assertAlmostEqual(decay_node_weight(n, NOW, P),
                               P.reinforce.node_weight_floor, places=6)

    def test_equilibrium_under_weekly_retrieval(self):
        """每周被检索一次的节点：增益与衰减平衡，维持高位（防垄断不误伤常用记忆）。"""
        from deusmith.algos import decay_node_weight, reinforce
        n = mk(weight=0.5)
        t = NOW
        for _ in range(300):
            reinforce(n, 0.5, t)
            t = t + timedelta(days=7)
            n.weight = decay_node_weight(n, t, P)
        self.assertGreater(n.weight, 0.75)
        self.assertLess(n.weight, 1.0)


class TestOfflineRenormalize(unittest.TestCase):
    def _gov(self, store):
        from deusmith.embeddings import HashEmbedding
        from deusmith.llm import RuleBasedLLM
        from deusmith.offline import OfflineGovernor
        return OfflineGovernor(store, HashEmbedding(dim=32), RuleBasedLLM(), P)

    def test_governance_reduces_monopoly(self):
        from deusmith.algos import gini
        from deusmith.store import InMemoryStore
        store = InMemoryStore()
        store.upsert_node(mk(id="hot", weight=0.99))
        for i in range(10):
            store.upsert_node(mk(id=f"cold{i}", weight=0.3))
        g0 = gini([n.weight for n in store.active_nodes("t0/u1")])
        report = self._gov(store).run("t0/u1", now=NOW + timedelta(days=400))
        self.assertIn("gini_weight", report)
        self.assertLess(report["gini_weight"], g0)
        # 热门节点闲置 400 天后明显回落，但仍高于 floor
        self.assertGreater(store.get_node("hot").weight,
                           P.reinforce.node_weight_floor)
        self.assertLess(store.get_node("hot").weight, 0.6)

    def test_below_floor_untouched(self):
        from deusmith.store import InMemoryStore
        store = InMemoryStore()
        store.upsert_node(mk(id="low", weight=0.1,
                             last_access_at=NOW - timedelta(days=800)))
        self._gov(store).run("t0/u1", now=NOW)
        self.assertEqual(store.get_node("low").weight, 0.1)  # 不抬升，归档逻辑管

    def test_monopoly_warning_fires(self):
        from deusmith.store import InMemoryStore
        store = InMemoryStore()
        store.upsert_node(mk(id="rich", weight=1.0))
        for i in range(19):
            store.upsert_node(mk(id=f"poor{i}", weight=0.0))
        report = self._gov(store).run("t0/u1", now=NOW)
        self.assertGreater(report["gini_weight"], 0.8)
        self.assertTrue(report["weight_monopoly_warning"])
        audits = store.recent_audits(limit=20)
        self.assertTrue(any(a["action"] == "weight_monopoly_warning" for a in audits))


if __name__ == "__main__":
    unittest.main()
