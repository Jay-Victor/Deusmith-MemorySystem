# -*- coding: utf-8 -*-
"""双时间推理单测 —— supersede 写 valid_from/valid_to + 历史查询 + 过期降权（评审第三梯队 #7）。"""
import unittest
from datetime import datetime, timedelta, timezone

from deusmith.models import MemoryNode, NodeType, Tier
from deusmith.params import Params

NOW = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)
P = Params()


def make_mem():
    from deusmith.embeddings import HashEmbedding
    from deusmith.llm import RuleBasedLLM
    from deusmith.pipeline import DeusmithMemory
    from deusmith.store import InMemoryStore
    return DeusmithMemory(store=InMemoryStore(), embedder=HashEmbedding(dim=32),
                          llm=RuleBasedLLM(), scope="t0/u1")


class TestSupersedeValidity(unittest.TestCase):
    def test_supersede_writes_validity_windows(self):
        from deusmith.models import utcnow
        m = make_mem()
        first = m.write(m.session("s1"), "我现在住在上海。", source_auth=0.4)
        second = m.write(m.session("s2"), "我现在住在北京。", source_auth=0.95)
        old, new = first[0], second[0]
        now = utcnow()
        self.assertIsNotNone(old.valid_to)       # 旧版本失效时间
        self.assertLessEqual(old.valid_to, now)  # 已过期
        self.assertIsNone(new.valid_to)          # 新版本当前有效
        self.assertIsNotNone(new.valid_from)
        self.assertLessEqual(new.valid_from, now)

    def test_contested_no_validity(self):
        """势均力敌的双版本不写时间窗（都还是候选）。"""
        m = make_mem()
        m.write(m.session("s1"), "我喜欢摄影。")
        second = m.write(m.session("s2"), "我不再喜欢摄影了。")
        self.assertIsNone(second[0].valid_to)
        self.assertIsNone(second[0].valid_from)


class TestSlotHistory(unittest.TestCase):
    def test_history_walks_versions_in_time_order(self):
        m = make_mem()
        m.write(m.session("s1"), "我现在住在上海。", source_auth=0.4)
        m.write(m.session("s2"), "我现在住在北京。", source_auth=0.95)
        hist = m.slot_history("user", "住在")
        self.assertEqual(len(hist), 2)
        self.assertIn("上海", hist[0]["content"])   # 时间序：先上海
        self.assertIn("北京", hist[1]["content"])
        self.assertIsNotNone(hist[0]["valid_to"])   # 历史
        self.assertIsNone(hist[1]["valid_to"])      # 当前

    def test_current_retrieval_prefers_new_version(self):
        m = make_mem()
        m.write(m.session("s1"), "我现在住在上海。", source_auth=0.4)
        m.write(m.session("s2"), "我现在住在北京。", source_auth=0.95)
        res = m.retrieve(m.session("s3"), "用户现在住在哪个城市")
        tops = [n.content for n, _, _ in res.selected[:3]]
        self.assertTrue(any("北京" in t for t in tops))
        self.assertFalse(any("上海" in t for t in tops))


class TestHistoricalPenalty(unittest.TestCase):
    def test_expired_fact_scored_lower(self):
        from deusmith.algos import retrieval_score
        base = dict(type=NodeType.fact, content="x", scope="t0/u1",
                    tier=Tier.middle, embedding_model="h", weight=0.6,
                    confidence=0.9, importance=0.8,
                    last_access_at=NOW - timedelta(days=7))
        current = MemoryNode(**base)
        expired = MemoryNode(**base, valid_to=NOW - timedelta(days=1))
        self.assertLess(retrieval_score(0.8, expired, NOW, params=P),
                        retrieval_score(0.8, current, NOW, params=P))


if __name__ == "__main__":
    unittest.main()
