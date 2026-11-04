# -*- coding: utf-8 -*-
"""核心算法单测 —— 对应《Deusmith-Memory》§五。

关键钉死项：
  - m102 数值演算（文档 §9.3 示例）由代码复算：S≈2.936，recency≈0.711，score≈0.349；
    文档自称 0.39 是漏乘 importance^ε 的结果（评审错误#2）。
  - 不变量 I4/I6（绝对刻度归一、衰减单调性）。
  - 置信度增益：无增益时 sigmoid 上限 0.711 < 晋升阈值 0.9，深层永远饥饿（评审修改项）。
"""
import math
import unittest
from datetime import datetime, timedelta, timezone

from deusmith.models import (
    EdgeType, MemoryEdge, MemoryNode, NodeStatus, NodeType, Tier,
)
from deusmith.params import Params


def mk_node(**over):
    kw = dict(
        type=NodeType.fact, content="示例记忆", scope="t0/u1",
        tier=Tier.middle, embedding_model="hash-256",
    )
    kw.update(over)
    return MemoryNode(**kw)


NOW = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)


class TestDecayRecency(unittest.TestCase):
    def test_zero_dt_recency_is_one(self):
        from deusmith.algos import decay_recency
        n = mk_node()
        self.assertAlmostEqual(decay_recency(n, n.last_access_at), 1.0, places=9)

    def test_m102_example_pinned(self):
        """文档 §9.3：Δt=7天, importance=0.8, emotion=0.5, confidence=0.9, activation=3
        → S≈2.936, recency=exp(-7/(7*S))≈0.711。"""
        from deusmith.algos import decay_recency, decay_stretch
        n = mk_node(importance=0.8, emotion=0.5, confidence=0.9, activation_count=3,
                    last_access_at=NOW - timedelta(days=7))
        S = decay_stretch(n)
        self.assertAlmostEqual(S, 2.9359, places=3)
        self.assertAlmostEqual(decay_recency(n, NOW), 0.7113, places=3)

    def test_monotonicity_invariant_i6(self):
        """importance/emotion/confidence/activation 任意↑ ⇒ recency↑（τ_eff↑）。"""
        from deusmith.algos import decay_recency
        base = dict(importance=0.5, emotion=0.3, confidence=0.5, activation_count=1,
                    last_access_at=NOW - timedelta(days=30))
        r0 = decay_recency(mk_node(**base), NOW)
        for over in (dict(importance=0.9), dict(emotion=0.9), dict(confidence=0.95),
                     dict(activation_count=10)):
            self.assertGreater(decay_recency(mk_node(**{**base, **over}), NOW), r0,
                               msg=f"提高 {over} 后 recency 应上升")

    def test_long_dormancy_approaches_zero(self):
        from deusmith.algos import decay_recency
        n = mk_node(last_access_at=NOW - timedelta(days=3650))
        self.assertLess(decay_recency(n, NOW), 0.05)

    def test_archive_after_days_derived_from_formula(self):
        """归档期限是导出量：t = -τ_eff·ln(θ_archive)，θ=0.2、S=1 → 7·1.609=11.27 天。"""
        from deusmith.algos import archive_after_days
        n = mk_node(importance=0.0, emotion=0.0, confidence=0.0, activation_count=0)
        self.assertAlmostEqual(archive_after_days(n), 7.0 * math.log(5.0), places=6)
        hot = mk_node(importance=0.8, emotion=0.5, confidence=0.9, activation_count=3)
        self.assertGreater(archive_after_days(hot), archive_after_days(n))


class TestReinforce(unittest.TestCase):
    def test_weight_saturation_formula(self):
        """w ← clip(w + η(1-w)·s, 0, 1)：w=0.5, η=0.3, s=1 → 0.65。"""
        from deusmith.algos import reinforce
        n = mk_node(weight=0.5)
        reinforce(n, signal_strength=1.0, now=NOW)
        self.assertAlmostEqual(n.weight, 0.65, places=9)

    def test_never_exceeds_one(self):
        from deusmith.algos import reinforce
        n = mk_node(weight=0.99)
        for _ in range(50):
            reinforce(n, signal_strength=1.0, now=NOW)
        self.assertLessEqual(n.weight, 1.0)
        self.assertGreater(n.weight, 0.99)

    def test_activation_and_access_time_update(self):
        from deusmith.algos import reinforce
        n = mk_node(activation_count=0, last_access_at=NOW - timedelta(days=10))
        reinforce(n, signal_strength=0.5, now=NOW)
        self.assertEqual(n.activation_count, 1)
        self.assertEqual(n.last_access_at, NOW)

    def test_confidence_updates_with_consistent_evidence(self):
        """强化附带按证据更新置信度（§五.2），使深层晋升阈值可达。"""
        from deusmith.algos import reinforce
        n = mk_node(confidence=0.6, consistency=0.9)
        c0 = n.confidence
        reinforce(n, signal_strength=1.0, now=NOW)
        self.assertGreater(n.confidence, c0)
        self.assertLessEqual(n.confidence, 1.0)


class TestEstimateConfidence(unittest.TestCase):
    def test_prior_reaches_promotion_threshold_with_gain(self):
        """增益 k=3：全 1 因子 → sigmoid(2.7)≈0.937 ≥ 0.9（无增益时上限 0.711，晋升不可达）。"""
        from deusmith.algos import estimate_confidence
        c = estimate_confidence(mk_node(source_auth=1.0, consistency=1.0), evidence_count=5,
                                freshness=1.0)
        self.assertAlmostEqual(c, 1.0 / (1.0 + math.exp(-2.7)), places=6)
        self.assertGreaterEqual(c, 0.9)

    def test_no_gain_ceiling_below_threshold(self):
        from deusmith.algos import estimate_confidence
        raw = 0.35 + 0.25 + 0.2 + 0.1  # 文档原式：sigmoid(0.9)=0.711
        self.assertLess(1.0 / (1.0 + math.exp(-raw)), 0.9)

    def test_conflict_penalty_reduces_confidence(self):
        from deusmith.algos import estimate_confidence
        base = dict(source_auth=0.8, consistency=0.8)
        c_clean = estimate_confidence(mk_node(**base), evidence_count=2, freshness=0.8)
        c_conflict = estimate_confidence(mk_node(**base), evidence_count=2, freshness=0.8,
                                         conflict_penalty=0.5)
        self.assertLess(c_conflict, c_clean)

    def test_monotonic_in_each_factor(self):
        from deusmith.algos import estimate_confidence
        c0 = estimate_confidence(mk_node(source_auth=0.5, consistency=0.5), 1, 0.5)
        for kw in (dict(source_auth=0.9), ):
            self.assertGreater(estimate_confidence(mk_node(**{**dict(source_auth=0.5, consistency=0.5), **kw}), 1, 0.5), c0)
        self.assertGreater(estimate_confidence(mk_node(source_auth=0.5, consistency=0.5), 5, 0.5), c0)
        self.assertGreater(estimate_confidence(mk_node(source_auth=0.5, consistency=0.5), 1, 0.9), c0)


class TestRetrievalScore(unittest.TestCase):
    def test_all_ones_scores_tier_boost(self):
        from deusmith.algos import retrieval_score
        n = mk_node(weight=1.0, confidence=1.0, importance=1.0,
                    last_access_at=NOW)
        self.assertAlmostEqual(retrieval_score(1.0, n, NOW), 1.0, places=9)

    def test_m102_score_pinned(self):
        """rel=0.8, weight=0.6, recency=0.7113, conf=0.9, imp=0.8, 中层 → 0.3489。
        文档值 0.39 = 漏乘 0.8^0.5。"""
        from deusmith.algos import retrieval_score
        n = mk_node(weight=0.6, confidence=0.9, importance=0.8,
                    last_access_at=NOW - timedelta(days=7),
                    activation_count=3, emotion=0.5)
        s = retrieval_score(0.8, n, NOW)
        self.assertAlmostEqual(s, 0.3489, places=3)

    def test_factor_floor_prevents_zero_score(self):
        from deusmith.algos import retrieval_score
        n = mk_node(weight=0.0, confidence=0.0, importance=0.0,
                    last_access_at=NOW - timedelta(days=3650))
        self.assertGreater(retrieval_score(0.5, n, NOW), 0.0)

    def test_sensitivity_gate_zeroes_restricted(self):
        from deusmith.algos import retrieval_score
        n = mk_node(sensitivity=3)
        self.assertEqual(retrieval_score(0.9, n, NOW, allowed_sensitivity=2), 0.0)
        self.assertGreater(retrieval_score(0.9, n, NOW, allowed_sensitivity=3), 0.0)

    def test_tier_boost_ordering_applies(self):
        from deusmith.algos import retrieval_score
        shallow = mk_node(tier=Tier.shallow, weight=1.0, confidence=1.0, importance=1.0)
        deep = mk_node(tier=Tier.deep, weight=1.0, confidence=1.0, importance=1.0)
        self.assertGreater(retrieval_score(1.0, deep, NOW), retrieval_score(1.0, shallow, NOW))


class TestRRFFusion(unittest.TestCase):
    def test_first_in_all_channels_is_exactly_one(self):
        """绝对参考点（不变量 I4）：全通道第一 → rel=1.0，与查询无关。"""
        from deusmith.algos import rrf_fuse
        rankings = {
            "vector": ["m1", "m2", "m3"],
            "graph": ["m1", "m3"],
            "keyword": ["m1", "m2"],
            "recent": ["m1"],
            "explicit": ["m1"],
        }
        rel = rrf_fuse("m1", rankings)
        self.assertAlmostEqual(rel, 1.0, places=9)

    def test_per_query_max_normalization_is_not_used(self):
        """单通道第一名不是 1.0（按当次最大值归一的坏做法在此处会得 1.0）。"""
        from deusmith.algos import rrf_fuse
        rankings = {"vector": ["m2", "m1"]}
        rel = rrf_fuse("m2", rankings)
        self.assertLess(rel, 1.0)

    def test_missing_from_all_channels_is_zero(self):
        from deusmith.algos import rrf_fuse
        self.assertEqual(rrf_fuse("mX", {"vector": ["m1"]}), 0.0)


class TestMMR(unittest.TestCase):
    def test_selects_diverse_over_duplicate(self):
        from deusmith.algos import mmr_select
        # v0 与 v1 几乎相同；v2 方向不同
        v0 = (1.0, 0.0)
        v1 = (0.99, 0.01)
        v2 = (0.0, 1.0)
        cands = [("a", 0.9, v0), ("b", 0.85, v1), ("c", 0.80, v2)]
        picked = mmr_select(cands, k=2, lam=0.7)
        self.assertIn("c", [p[0] for p in picked])
        self.assertNotIn("b", [p[0] for p in picked])

    def test_empty_candidates(self):
        from deusmith.algos import mmr_select
        self.assertEqual(mmr_select([], k=3, lam=0.7), [])

    def test_k_limits_selection(self):
        from deusmith.algos import mmr_select
        cands = [(f"m{i}", 0.9 - i * 0.01, (float(i), 1.0)) for i in range(10)]
        self.assertEqual(len(mmr_select(cands, k=3, lam=0.7)), 3)


class TestTierTransition(unittest.TestCase):
    def test_explicit_remember_promotes_shallow_to_middle(self):
        from deusmith.algos import tier_transition
        n = mk_node(tier=Tier.shallow, explicit_remember=True)
        self.assertEqual(tier_transition(n), "promote_to_middle")

    def test_cross_session_repeat_promotes(self):
        from deusmith.algos import tier_transition
        n = mk_node(tier=Tier.shallow, cross_session_count=2)
        self.assertEqual(tier_transition(n), "promote_to_middle")

    def test_importance_promotes(self):
        from deusmith.algos import tier_transition
        n = mk_node(tier=Tier.shallow, importance=0.75)
        self.assertEqual(tier_transition(n), "promote_to_middle")

    def test_middle_promotes_to_deep_when_all_thresholds_met(self):
        from deusmith.algos import tier_transition
        n = mk_node(tier=Tier.middle, confidence=0.92, source_auth=0.85,
                    cross_session_count=6, human_confirmed=True)
        self.assertEqual(tier_transition(n), "promote_to_deep")

    def test_identity_policy_requires_human_confirmation(self):
        from deusmith.algos import tier_transition
        n = mk_node(tier=Tier.middle, type=NodeType.identity_policy,
                    confidence=0.95, source_auth=0.9, cross_session_count=9,
                    human_confirmed=False)
        self.assertEqual(tier_transition(n), "needs_human_confirmation")

    def test_low_confidence_blocks_deep(self):
        from deusmith.algos import tier_transition
        n = mk_node(tier=Tier.middle, confidence=0.7, source_auth=0.85,
                    cross_session_count=6, human_confirmed=True)
        self.assertEqual(tier_transition(n), "keep")

    def test_stale_low_weight_archives(self):
        from deusmith.algos import tier_transition
        n = mk_node(tier=Tier.middle, weight=0.15,
                    last_access_at=NOW - timedelta(days=120))
        self.assertEqual(tier_transition(n, now=NOW), "archive")

    def test_contested_kept_not_archived(self):
        from deusmith.algos import tier_transition
        n = mk_node(tier=Tier.middle, weight=0.15, status=NodeStatus.contested,
                    last_access_at=NOW - timedelta(days=120))
        self.assertEqual(tier_transition(n, now=NOW), "keep")


class TestEdgeMath(unittest.TestCase):
    def test_saturation_merge(self):
        from deusmith.algos import merge_edge_weight
        self.assertAlmostEqual(merge_edge_weight(0.5, 0.2), 0.6, places=9)
        self.assertAlmostEqual(merge_edge_weight(0.9, 0.9), 1 - (1 - 0.9) * (1 - 0.9), places=9)

    def test_edge_decay_exponential(self):
        from deusmith.algos import decay_edge_weight
        self.assertAlmostEqual(decay_edge_weight(0.5, 60), 0.5 * math.exp(-0.01 * 60), places=9)

    def test_prune_rule_requires_all_three(self):
        from deusmith.algos import should_prune_edge
        old = NOW - timedelta(days=70)
        self.assertTrue(should_prune_edge(0.05, old, 0.2, now=NOW))
        self.assertFalse(should_prune_edge(0.05, old, 0.5, now=NOW))   # confidence 高
        self.assertFalse(should_prune_edge(0.05, NOW - timedelta(days=10), 0.2, now=NOW))
        self.assertFalse(should_prune_edge(0.5, old, 0.2, now=NOW))    # weight 高

    def test_contradicts_edge_weakened_not_deleted(self):
        """冲突边保留并降权（§五.4），status=weakened。"""
        from deusmith.algos import weaken_contradiction_edge
        e = MemoryEdge(src_id="a" * 32, dst_id="b" * 32, type=EdgeType.contradicts, weight=0.8)
        weaken_contradiction_edge(e)
        self.assertEqual(e.status.value, "weakened")
        self.assertLess(e.weight, 0.8)


class TestCosineAndTokens(unittest.TestCase):
    def test_cosine_identical_and_orthogonal(self):
        from deusmith.algos import cosine, estimate_tokens
        self.assertAlmostEqual(cosine((1, 0), (1, 0)), 1.0)
        self.assertAlmostEqual(cosine((1, 0), (0, 1)), 0.0)
        self.assertAlmostEqual(cosine((1, 0), (-1, 0)), -1.0)

    def test_estimate_tokens_cjk_vs_ascii(self):
        from deusmith.algos import estimate_tokens
        self.assertEqual(estimate_tokens("用户偏好会议时间"), 8)      # 8 个汉字
        self.assertEqual(estimate_tokens("hello world"), 3)          # 11 ASCII / 4
        self.assertEqual(estimate_tokens(""), 0)


class TestCompressMerge(unittest.TestCase):
    def test_cluster_by_type_scope_similarity(self):
        from deusmith.algos import cluster_nodes
        v = (1.0, 0.0)
        v2 = (0.95, 0.05)
        far = (0.0, 1.0)
        nodes = [
            mk_node(id="m1", type=NodeType.event, content="a", content_vector=v),
            mk_node(id="m2", type=NodeType.event, content="b", content_vector=v2),
            mk_node(id="m3", type=NodeType.fact, content="c", content_vector=v),   # 类型不同
            mk_node(id="m4", type=NodeType.event, content="d", content_vector=far,
                    scope="t0/u2"),                                              # scope 不同
        ]
        clusters = cluster_nodes(nodes, threshold=0.9)
        ids = {frozenset(c["ids"]) for c in clusters}
        self.assertIn(frozenset({"m1", "m2"}), ids)
        # m3、m4 各自成单例簇
        all_ids = sorted(i for cl in clusters for i in cl["ids"])
        self.assertEqual(all_ids, ["m1", "m2", "m3", "m4"])

    def test_summary_generation_exempts_from_further_clustering(self):
        from deusmith.algos import cluster_nodes
        v = (1.0, 0.0)
        nodes = [
            mk_node(id="s1", type=NodeType.summary, content="sum1", content_vector=v, generation=2),
            mk_node(id="s2", type=NodeType.summary, content="sum2", content_vector=v, generation=2),
        ]
        clusters = cluster_nodes(nodes, threshold=0.9)
        self.assertEqual(clusters, [])


class TestCycleResolution(unittest.TestCase):
    def test_tarjan_finds_cycle(self):
        from deusmith.algos import resolve_cycles
        edges = [
            MemoryEdge(src_id="a", dst_id="b", type=EdgeType.derives_from),
            MemoryEdge(src_id="b", dst_id="c", type=EdgeType.derives_from),
            MemoryEdge(src_id="c", dst_id="a", type=EdgeType.derives_from),
            MemoryEdge(src_id="d", dst_id="a", type=EdgeType.derives_from),
        ]
        actions = resolve_cycles(edges)
        removed = [e for e, act in actions if act == "remove"]
        self.assertEqual(len(removed), 1)          # 3-环去掉一条最弱边
        self.assertIn(removed[0].src_id, {"a", "b", "c"})
        kept = [e for e, act in actions if act == "keep"]
        self.assertEqual(len(kept), 3)


if __name__ == "__main__":
    unittest.main()
