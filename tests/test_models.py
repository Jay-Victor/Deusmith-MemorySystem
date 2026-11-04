# -*- coding: utf-8 -*-
"""节点/边数据模型单测 —— 对应《Deusmith-Memory》§三 节点与链路模型 + 评审补充的槽位字段。

RED 阶段：deusmith.models 尚不存在，本文件应当 import 失败。
"""
import unittest
from datetime import datetime, timezone


def utcnow():
    return datetime.now(timezone.utc)


class TestEnums(unittest.TestCase):
    def test_node_types_match_doc(self):
        from deusmith.models import NodeType
        expected = {"event", "fact", "preference", "emotion", "summary",
                    "task", "tool_result", "identity_policy", "conflict"}
        self.assertEqual({t.value for t in NodeType}, expected)

    def test_statuses_match_doc(self):
        from deusmith.models import NodeStatus
        expected = {"active", "archived", "contested", "deprecated", "deleted_logical"}
        self.assertEqual({s.value for s in NodeStatus}, expected)

    def test_edge_types_match_doc(self):
        from deusmith.models import EdgeType
        expected = {"semantic_similar", "temporal_next", "causal", "co_occur",
                    "coref", "supports", "contradicts", "derives_from", "summary_of"}
        self.assertEqual({t.value for t in EdgeType}, expected)

    def test_sensitivity_levels(self):
        from deusmith.models import Sensitivity
        self.assertEqual([s.value for s in Sensitivity], [0, 1, 2, 3])


class TestMemoryNodeValidation(unittest.TestCase):
    def _base_kwargs(self, **over):
        from deusmith.models import NodeType, Tier
        kw = dict(
            type=NodeType.fact, content="用户偏好早上开会", scope="t0/u1",
            tier=Tier.middle, embedding_model="hash-256",
        )
        kw.update(over)
        return kw

    def test_minimal_node_gets_id_hash_and_timestamps(self):
        from deusmith.models import MemoryNode
        n = MemoryNode(**self._base_kwargs())
        self.assertTrue(n.id)
        self.assertTrue(n.created_at)
        self.assertIsNotNone(n.content_hash)
        self.assertEqual(n.activation_count, 0)
        self.assertEqual(n.version, 1)

    def test_same_content_same_hash_regardless_of_whitespace(self):
        from deusmith.models import MemoryNode
        a = MemoryNode(**self._base_kwargs(content="用户  喜欢  早上开会"))
        b = MemoryNode(**self._base_kwargs(content="用户喜欢早上开会"))
        self.assertEqual(a.content_hash, b.content_hash)

    def test_scope_required_invariant_i1(self):
        from deusmith.models import MemoryNode
        with self.assertRaises(ValueError):
            MemoryNode(**self._base_kwargs(scope=""))

    def test_weight_bounds(self):
        from deusmith.models import MemoryNode
        with self.assertRaises(ValueError):
            MemoryNode(**self._base_kwargs(weight=1.5))
        with self.assertRaises(ValueError):
            MemoryNode(**self._base_kwargs(weight=-0.1))

    def test_confidence_importance_emotion_bounds(self):
        from deusmith.models import MemoryNode
        for field in ("confidence", "importance", "emotion", "source_auth"):
            with self.assertRaises(ValueError):
                MemoryNode(**self._base_kwargs(**{field: 1.2}))
            with self.assertRaises(ValueError):
                MemoryNode(**self._base_kwargs(**{field: -0.2}))

    def test_sensitivity_range(self):
        from deusmith.models import MemoryNode
        with self.assertRaises(ValueError):
            MemoryNode(**self._base_kwargs(sensitivity=4))
        with self.assertRaises(ValueError):
            MemoryNode(**self._base_kwargs(sensitivity=-1))

    def test_polarity_values(self):
        from deusmith.models import MemoryNode
        with self.assertRaises(ValueError):
            MemoryNode(**self._base_kwargs(polarity=2))
        n = MemoryNode(**self._base_kwargs(polarity=-1))
        self.assertEqual(n.polarity, -1)

    def test_slot_key_is_scope_subject_predicate(self):
        from deusmith.models import MemoryNode
        n = MemoryNode(**self._base_kwargs(subject="u1", predicate="喜欢"))
        self.assertEqual(n.slot_key(), ("t0/u1", "u1", "喜欢"))


class TestContradiction(unittest.TestCase):
    def _mk(self, subject, predicate, obj, polarity, content):
        from deusmith.models import NodeType, Tier, MemoryNode
        return MemoryNode(
            type=NodeType.preference, content=content, scope="t0/u1",
            tier=Tier.middle, embedding_model="hash-256",
            subject=subject, predicate=predicate, object_value=obj, polarity=polarity,
        )

    def test_same_slot_opposite_polarity_is_contradiction(self):
        from deusmith.algos import is_contradiction
        a = self._mk("u1", "喜欢", "摄影", +1, "我喜欢摄影")
        b = self._mk("u1", "喜欢", "摄影", -1, "我不再喜欢摄影了")
        self.assertTrue(is_contradiction(a, b))

    def test_same_slot_different_object_is_contradiction(self):
        from deusmith.algos import is_contradiction
        a = self._mk("u1", "住在", "北京", 0, "我住在北京")
        b = self._mk("u1", "住在", "上海", 0, "我现在住在上海")
        self.assertTrue(is_contradiction(a, b))

    def test_different_predicate_is_not_contradiction(self):
        from deusmith.algos import is_contradiction
        a = self._mk("u1", "喜欢", "跑步", +1, "我喜欢跑步")
        b = self._mk("u1", "讨厌", "跑步", +1, "我讨厌跑步")
        self.assertFalse(is_contradiction(a, b))  # 谓语不同：是两条记忆，不是冲突

    def test_multi_valued_predicate_different_object_not_contradiction(self):
        """多值谓语（喜欢）换宾语 = 两条并存偏好，不是冲突（错误#1 完整版）。"""
        from deusmith.algos import is_contradiction
        a = self._mk("u1", "喜欢", "摄影", +1, "我喜欢摄影")
        b = self._mk("u1", "喜欢", "跑步", +1, "我喜欢跑步")
        self.assertFalse(is_contradiction(a, b))

    def test_multi_valued_opposite_polarity_different_object_not_contradiction(self):
        from deusmith.algos import is_contradiction
        a = self._mk("u1", "喜欢", "摄影", +1, "我喜欢摄影")
        b = self._mk("u1", "喜欢", "跑步", -1, "我不喜欢跑步")
        self.assertFalse(is_contradiction(a, b))  # 宾语不同：喜好对象都不同，无从冲突

    def test_identical_slot_and_value_is_duplicate(self):
        from deusmith.algos import is_duplicate
        a = self._mk("u1", "喜欢", "摄影", +1, "我喜欢摄影")
        b = self._mk("u1", "喜欢", "摄影", +1, "我挺喜欢摄影的")
        self.assertTrue(is_duplicate(a, b))


class TestMemoryEdgeValidation(unittest.TestCase):
    def test_edge_basics(self):
        from deusmith.models import EdgeType, MemoryEdge
        e = MemoryEdge(src_id="a" * 32, dst_id="b" * 32, type=EdgeType.semantic_similar,
                       scope="t0/u1")
        self.assertTrue(e.edge_id)
        self.assertEqual(e.direction, "directed")
        self.assertEqual(e.weight, 0.5)

    def test_self_loop_rejected(self):
        from deusmith.models import EdgeType, MemoryEdge
        with self.assertRaises(ValueError):
            MemoryEdge(src_id="a" * 32, dst_id="a" * 32, type=EdgeType.co_occur, scope="t0/u1")

    def test_scope_required_invariant_i1(self):
        from deusmith.models import EdgeType, MemoryEdge
        with self.assertRaises(ValueError):
            MemoryEdge(src_id="a" * 32, dst_id="b" * 32, type=EdgeType.co_occur, scope="")

    def test_weight_bounds(self):
        from deusmith.models import EdgeType, MemoryEdge
        with self.assertRaises(ValueError):
            MemoryEdge(src_id="a" * 32, dst_id="b" * 32, type=EdgeType.co_occur,
                       weight=1.5, scope="t0/u1")

    def test_contradicts_edge_is_bidirectional_marker(self):
        from deusmith.models import EdgeType, MemoryEdge
        e = MemoryEdge(src_id="a" * 32, dst_id="b" * 32, type=EdgeType.contradicts,
                       scope="t0/u1")
        self.assertEqual(e.direction, "undirected")


if __name__ == "__main__":
    unittest.main()
