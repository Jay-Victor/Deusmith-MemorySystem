# -*- coding: utf-8 -*-
"""参数默认值单测 —— 与《Deusmith-Memory》§四 默认超参表逐项钉死。

RED 阶段：deusmith.params 尚不存在，本文件应当 import 失败。
"""
import unittest


class TestDecayDefaults(unittest.TestCase):
    def test_tau_base_and_coeffs(self):
        from deusmith.params import Params
        p = Params()
        self.assertEqual(p.decay.tau_base_days, 7.0)
        self.assertEqual(p.decay.k_importance, 0.8)
        self.assertEqual(p.decay.k_emotion, 0.5)
        self.assertEqual(p.decay.k_confidence, 0.7)
        self.assertEqual(p.decay.k_activation, 0.3)

    def test_reinforce_defaults(self):
        from deusmith.params import Params
        p = Params()
        self.assertEqual(p.reinforce.eta, 0.3)
        self.assertEqual(p.reinforce.eta_edge, 0.2)
        self.assertEqual(p.reinforce.lambda_edge_per_day, 0.01)


class TestConfidenceDefaults(unittest.TestCase):
    def test_weights_match_doc(self):
        from deusmith.params import Params
        p = Params()
        self.assertEqual(p.confidence.w_source_auth, 0.35)
        self.assertEqual(p.confidence.w_consistency, 0.25)
        self.assertEqual(p.confidence.w_evidence, 0.2)
        self.assertEqual(p.confidence.w_freshness, 0.1)
        self.assertEqual(p.confidence.w_conflict_penalty, 0.3)

    def test_invariant_weights_in_unit_range_and_penalty_positive(self):
        from deusmith.params import Params
        c = Params().confidence
        for w in (c.w_source_auth, c.w_consistency, c.w_evidence, c.w_freshness):
            self.assertGreaterEqual(w, 0.0)
            self.assertLessEqual(w, 1.0)
        self.assertGreater(c.w_conflict_penalty, 0.0)


class TestScoreDefaults(unittest.TestCase):
    def test_exponents_match_doc(self):
        from deusmith.params import Params
        p = Params()
        self.assertEqual(p.score.alpha, 1.0)   # rel
        self.assertEqual(p.score.beta, 0.8)    # weight
        self.assertEqual(p.score.gamma, 0.6)   # recency
        self.assertEqual(p.score.delta, 1.0)   # confidence
        self.assertEqual(p.score.epsilon, 0.5) # importance
        self.assertEqual(p.score.mmr_lambda, 0.7)
        self.assertEqual(p.score.factor_floor, 0.01)

    def test_tier_not_a_ranking_factor(self):
        """tier_boost 已从打分公式移除（v0.2.1）：tier 管留存与注入段位，不进排序。"""
        from deusmith.params import Params
        self.assertFalse(hasattr(Params().score, "tier_boost"))


class TestRetrievalDefaults(unittest.TestCase):
    def test_recall_params_match_doc(self):
        from deusmith.params import Params
        p = Params()
        self.assertEqual(p.retrieval.topk_recall, 50)
        self.assertEqual(p.retrieval.graph_hops, 2)
        self.assertEqual(p.retrieval.ppr_damping, 0.85)
        self.assertEqual(p.retrieval.knn_edge_topk, 20)
        self.assertEqual(p.retrieval.token_budget, 4000)
        self.assertEqual(p.retrieval.rrf_k, 60)


class TestTierDefaults(unittest.TestCase):
    def test_shallow_params(self):
        from deusmith.params import Params
        t = Params().tier
        self.assertEqual(t.shallow_window_rounds, 10)
        self.assertEqual(t.shallow_context_share, 0.30)
        self.assertEqual(t.shallow_cleanup_delay_min, 30.0)

    def test_promotion_thresholds(self):
        from deusmith.params import Params
        t = Params().tier
        self.assertEqual(t.promote_mid_repeat, 2)
        self.assertEqual(t.promote_mid_importance, 0.7)
        self.assertEqual(t.promote_deep_confidence, 0.9)
        self.assertEqual(t.promote_deep_source_auth, 0.8)
        self.assertEqual(t.promote_deep_sessions, 5)

    def test_archive_and_prune_thresholds(self):
        from deusmith.params import Params
        t = Params().tier
        self.assertEqual(t.middle_reference_ttl_days, 90)
        self.assertEqual(t.archive_score_floor, 0.2)
        self.assertEqual(t.edge_prune_weight, 0.1)
        self.assertEqual(t.edge_prune_days, 60)
        self.assertEqual(t.edge_prune_confidence, 0.3)
        self.assertEqual(t.cluster_threshold, 0.9)
        self.assertEqual(t.max_summary_generation, 2)


class TestTunableDims(unittest.TestCase):
    def test_tunable_dims_capped(self):
        """可调维度收敛在 7（评审 §四：30 个自由参数必然过拟合探针集）。"""
        from deusmith.params import TUNABLE_DIMS, MAX_TUNABLE_DIMS
        self.assertLessEqual(len(TUNABLE_DIMS), MAX_TUNABLE_DIMS)
        self.assertEqual(len(set(TUNABLE_DIMS)), len(TUNABLE_DIMS))


class TestStageTwoParams(unittest.TestCase):
    def test_channel_weights_no_explicit(self):
        """常驻通道（explicit）不参与 RRF 融合（技能库 §4：常驻节点会在
        所有查询里排第一，挤掉真正相关的内容）；常驻记忆走无条件注入。"""
        from deusmith.params import Params
        cw = Params().retrieval.channel_weights
        self.assertNotIn("explicit", cw)
        for w in cw.values():
            self.assertGreaterEqual(w, 0)

    def test_recency_span(self):
        from deusmith.params import Params
        sp = Params().score
        self.assertEqual(sp.recency_span, 0.25)   # 有界弱时效：(1-s)+s·rec^γ
        self.assertLess(sp.recency_span, 0.5)


if __name__ == "__main__":
    unittest.main()
