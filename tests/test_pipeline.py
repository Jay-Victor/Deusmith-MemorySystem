# -*- coding: utf-8 -*-
"""管线层单测 —— 写入/读取/注入三路径（§五.8 调用顺序）+ 浅层会话。

使用 InMemoryStore + HashEmbedding + RuleBasedLLM（全离线）。
覆盖评审三条最贵错误的端到端防线：
  #1 冲突不静默覆盖（否定句进 contested / 仲裁 / 深层红线）
  #2 绝对刻度（retrieval_score 由 rrf_fuse 供 rel）
  #3 预算约束与降级阶梯（I7）
"""
import unittest

from deusmith.embeddings import HashEmbedding
from deusmith.llm import RuleBasedLLM
from deusmith.models import EdgeType, MemoryEdge, NodeStatus, NodeType, Tier
from deusmith.params import Params
from deusmith.store import InMemoryStore

SCOPE = "t0/u1"


def make_memory(**over):
    from deusmith.pipeline import DeusmithMemory
    kw = dict(store=InMemoryStore(), embedder=HashEmbedding(dim=64),
              llm=RuleBasedLLM(), params=Params(), scope=SCOPE)
    kw.update(over)
    return DeusmithMemory(**kw)


class TestWritePath(unittest.TestCase):
    def test_write_extracts_and_stores_nodes(self):
        m = make_memory()
        s = m.session("s1")
        nodes = m.write(s, "我喜欢摄影。")
        self.assertTrue(nodes)
        self.assertEqual(nodes[0].type.value, "preference")
        self.assertEqual(nodes[0].tier.value, "middle")   # 显式偏好 importance≥0.7 直接晋升
        self.assertTrue(nodes[0].content_vector)

    def test_duplicate_write_reinforces_not_duplicates(self):
        m = make_memory()
        s = m.session("s1")
        m.write(s, "我喜欢摄影。")
        m.write(s, "我喜欢摄影。")
        from deusmith.models import NodeType
        nodes = [n for n in m.store.active_nodes(SCOPE, tier=Tier.middle)]
        self.assertEqual(len(nodes), 1)                    # I8：同 hash 只有一个节点
        self.assertGreaterEqual(nodes[0].activation_count, 1)

    def test_cross_session_repeat_increments(self):
        m = make_memory()
        m.write(m.session("s1"), "我喜欢摄影。")
        m.write(m.session("s2"), "我喜欢摄影。")
        nodes = [n for n in m.store.active_nodes(SCOPE, tier=Tier.middle)]
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].cross_session_count, 1)  # 跨会话重复 +1

    def test_negation_marks_contested_no_silent_overwrite(self):
        """错误#1 核心场景：『不再喜欢』不得静默顶掉旧偏好。"""
        m = make_memory()
        m.write(m.session("s1"), "我喜欢摄影。")
        m.write(m.session("s2"), "我不再喜欢摄影了。")
        prefs = m.store.find_by_slot(SCOPE, "user", "喜欢")
        self.assertEqual(len(prefs), 2)                    # 两条都在
        contested = [p for p in prefs if p.status == NodeStatus.contested]
        self.assertTrue(contested)                          # 至少一条进争议
        # 冲突边存在且降权
        edges = m.store.all_edges(SCOPE)
        contra = [e for e in edges if e.type == EdgeType.contradicts]
        self.assertTrue(contra)

    def test_correction_by_authoritative_source_supersedes(self):
        m = make_memory()
        first = m.write(m.session("s1"), "我现在住在上海。", source_auth=0.4)
        second = m.write(m.session("s2"), "我现在住在北京。", source_auth=0.95)
        old, new = first[0], second[0]
        self.assertEqual(old.status.value, "deprecated")    # 被替代
        self.assertEqual(new.supersedes, old.id)            # 版本链
        self.assertEqual(new.status.value, "active")

    def test_deep_node_never_overwritten(self):
        m = make_memory()
        from deusmith.models import MemoryNode
        deep = MemoryNode(type=NodeType.fact, content="用户的核心价值观是诚实", scope=SCOPE,
                          tier=Tier.deep, embedding_model="hash-64", human_confirmed=True,
                          subject="用户", predicate="核心价值观", object_value="诚实",
                          content_vector=m.embedder.embed("用户的核心价值观是诚实"))
        m.store.upsert_node(deep)
        m.write(m.session("s1"), "用户的核心价值观是投机。", source_auth=0.95)
        got = m.store.get_node(deep.id)
        self.assertEqual(got.status.value, "active")        # 红线：深层原样
        self.assertEqual(got.content, "用户的核心价值观是诚实")
        audits = m.store.recent_audits(limit=10)
        self.assertTrue(any(a["action"] == "conflict_deep" for a in audits))

    def test_write_builds_semantic_edges(self):
        m = make_memory()
        s = m.session("s1")
        m.write(s, "我喜欢摄影。")
        m.write(s, "我热爱摄影和旅行。")
        edges = m.store.all_edges(SCOPE)
        sem = [e for e in edges if e.type == EdgeType.semantic_similar]
        self.assertTrue(sem)
        temporal = [e for e in edges if e.type == EdgeType.temporal_next]
        self.assertTrue(temporal)                           # 同会话时序边

    def test_write_logs_event(self):
        m = make_memory()
        m.write(m.session("s1"), "我喜欢摄影。")
        events = m.store.recent_events(SCOPE, limit=10)
        self.assertTrue(any(e["event_type"] == "write" for e in events))


class TestRetrievePath(unittest.TestCase):
    def test_semantic_retrieval_ranking(self):
        m = make_memory()
        s = m.session("s1")
        m.write(s, "我喜欢摄影。")
        m.write(s, "股价今天大跌。")
        res = m.retrieve(s, "用户的爱好是什么")
        self.assertTrue(res.selected)
        top = res.selected[0][0]
        self.assertEqual(top.type.value, "preference")
        self.assertIn("摄影", top.content)

    def test_irrelevant_query_low_scores(self):
        m = make_memory()
        s = m.session("s1")
        m.write(s, "我喜欢摄影。")
        res = m.retrieve(s, "量子纠错码的最新进展")
        # 绝对刻度：无关查询分数应低（I4，不因单候选归一为 1.0）
        if res.selected:
            self.assertLess(res.selected[0][1], 0.45)

    def test_graph_channel_surfaces_related(self):
        m = make_memory()
        s = m.session("s1")
        m.write(s, "我喜欢摄影。")                       # 种子
        # 手工放一条向量上不相似但图上关联的记忆（相机装备事件）
        from deusmith.models import MemoryNode
        gear = MemoryNode(type=NodeType.event, content="昨天买了三脚架和滤镜", scope=SCOPE,
                          tier=Tier.middle, embedding_model="hash-64",
                          content_vector=m.embedder.embed("昨天买了三脚架和滤镜"))
        m.store.upsert_node(gear)
        photo = [n for n in m.store.active_nodes(SCOPE) if "摄影" in n.content][0]
        m.store.upsert_edge(MemoryEdge(src_id=photo.id, dst_id=gear.id,
                                       type=EdgeType.co_occur, scope=SCOPE, weight=0.9))
        res = m.retrieve(s, "用户的爱好", topk=5)
        ids = [n.id for n, _, _ in res.selected]
        self.assertIn(gear.id, ids)                        # 图游走召出向量不相似的关联记忆

    def test_retrieval_reinforces_hits(self):
        m = make_memory()
        s = m.session("s1")
        nodes = m.write(s, "我喜欢摄影。")
        before = nodes[0].activation_count
        m.retrieve(s, "用户的爱好")
        got = m.store.get_node(nodes[0].id)
        self.assertGreater(got.activation_count, before)   # 命中即强化

    def test_retrieval_logs_trace(self):
        m = make_memory()
        s = m.session("s1")
        m.write(s, "我喜欢摄影。")
        m.retrieve(s, "用户的爱好", trace_id="tr_001")
        events = m.store.recent_events(SCOPE, limit=10)
        tr = [e for e in events if e["event_type"] == "retrieval"]
        self.assertTrue(tr)
        self.assertEqual(tr[0]["trace_id"], "tr_001")

    def test_sensitivity_filters_candidates(self):
        m = make_memory()
        s = m.session("s1")
        from deusmith.models import MemoryNode
        secret = MemoryNode(type=NodeType.fact, content="用户的银行卡密码是123456", scope=SCOPE,
                            tier=Tier.middle, embedding_model="hash-64",
                            sensitivity=3,
                            content_vector=m.embedder.embed("用户的银行卡密码是123456"))
        m.store.upsert_node(secret)
        res = m.retrieve(s, "用户的银行卡密码")
        self.assertNotIn(secret.id, [n.id for n, _, _ in res.selected])
        self.assertTrue(any("sensitivity" in str(r) for r in res.filtered_reasons))


class TestInjectPath(unittest.TestCase):
    def test_segments_split_by_tier_and_role(self):
        m = make_memory()
        s = m.session("s1")
        from deusmith.models import MemoryNode
        identity = MemoryNode(type=NodeType.identity_policy, content="用户名：李雷；助理需诚实",
                              scope=SCOPE, tier=Tier.deep, embedding_model="hash-64",
                              human_confirmed=True,
                              content_vector=m.embedder.embed("用户名：李雷；助理需诚实"))
        m.store.upsert_node(identity)
        m.write(s, "我喜欢摄影。")
        out = m.inject_context(s, "用户的爱好")
        self.assertIn("李雷", out["system"])               # 深层身份 → system 段
        self.assertIn("摄影", out["user"])                 # 相关偏好 → user 段
        self.assertGreaterEqual(out["tokens"], 1)

    def test_contested_injected_as_conflict_hint_only(self):
        m = make_memory()
        m.write(m.session("s1"), "我喜欢摄影。")
        m.write(m.session("s2"), "我不再喜欢摄影了。")
        out = m.inject_context(m.session("s3"), "用户的爱好")
        self.assertIn("冲突", out["user"])                 # 只提示存在冲突
        self.assertNotIn("我喜欢摄影。", out["user"])      # 不注入争议原文

    def test_budget_respected_and_degradation(self):
        m = make_memory()
        s = m.session("s1")
        for topic in ("摄影", "跑步", "烹饪", "书法", "围棋", "滑雪", "钓鱼", "爬山"):
            m.write(s, f"我喜欢{topic}。")
        out = m.inject_context(s, "用户的爱好", token_budget=120)
        self.assertLessEqual(out["tokens"], 120)
        self.assertTrue(out["degraded"])                   # 超预算触发降级

    def test_session_context_in_user_segment(self):
        m = make_memory()
        s = m.session("s1")
        m.write(s, "我喜欢摄影。")
        out = m.inject_context(s, "用户的爱好")
        self.assertIn("摄影", out["user"])


class TestShallowSession(unittest.TestCase):
    def test_rounds_window_and_rolling_summary(self):
        m = make_memory()
        s = m.session("s1")
        for i in range(12):                                # 超过 N=10 轮
            m.write(s, f"第{i}轮：随便聊聊。")
        self.assertLessEqual(len(s.rounds), 10)            # 滚动窗口
        self.assertTrue(s.rolling_summary)                 # 超窗触发滚动摘要

    def test_context_text_within_shallow_budget(self):
        m = make_memory()
        s = m.session("s1")
        for i in range(10):
            m.write(s, f"第{i}轮内容。")
        text = s.to_text(token_budget=60)
        from deusmith.algos import estimate_tokens
        self.assertLessEqual(estimate_tokens(text), 60)


class TestEndSession(unittest.TestCase):
    def test_end_session_promotes_and_logs(self):
        m = make_memory()
        s = m.session("s1")
        m.write(s, "请记住：我对花粉过敏。")
        nodes = [n for n in m.store.active_nodes(SCOPE)]
        self.assertTrue(all(n.tier != Tier.shallow for n in nodes))  # 显式记住即晋升中层
        m.end_session(s)
        events = m.store.recent_events(SCOPE, limit=20)
        self.assertTrue(any(e["event_type"] == "session_end" for e in events))


if __name__ == "__main__":
    unittest.main()
