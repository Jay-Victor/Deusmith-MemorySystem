# -*- coding: utf-8 -*-
"""可解释性单测 —— "你为什么记住了这个/注入了这条"（评审第三梯队 #9）。"""
import unittest


def make_mem():
    from deusmith.embeddings import HashEmbedding
    from deusmith.llm import RuleBasedLLM
    from deusmith.pipeline import DeusmithMemory
    from deusmith.store import InMemoryStore
    return DeusmithMemory(store=InMemoryStore(), embedder=HashEmbedding(dim=32),
                          llm=RuleBasedLLM(), scope="t0/u1")


class TestExplain(unittest.TestCase):
    def test_explain_injection_by_trace(self):
        m = make_mem()
        s = m.session("s1")
        m.write(s, "我喜欢摄影。")
        m.inject_context(s, "用户的爱好", trace_id="tr_x1")
        ex = m.explain_injection("tr_x1")
        self.assertTrue(ex["found"])
        self.assertEqual(ex["query"], "用户的爱好")
        self.assertIn("inject", ex["events"])
        self.assertTrue(ex["selected"])                  # 选中列表
        self.assertIsInstance(ex["tokens"], int)

    def test_missing_trace(self):
        m = make_mem()
        self.assertFalse(m.explain_injection("nope")["found"])

    def test_why_selected_finds_traces(self):
        m = make_mem()
        s = m.session("s1")
        nodes = m.write(s, "我喜欢摄影。")
        m.inject_context(s, "用户的爱好", trace_id="tr_y2")
        why = m.why_selected(nodes[0].id)
        self.assertTrue(any(w["trace_id"] == "tr_y2" for w in why))
        self.assertTrue(all("score" in w for w in why))

    def test_explain_score_factor_breakdown(self):
        m = make_mem()
        nodes = m.write(m.session("s1"), "我喜欢摄影。")
        ex = m.explain_score(nodes[0].id)
        for key in ("weight", "recency", "confidence", "importance", "tier_boost"):
            self.assertIn(key, ex["factors"])
        self.assertGreater(ex["score_at_full_rel"], 0)
        self.assertLessEqual(ex["score_at_full_rel"], 1.0)
        self.assertIn("摄影", ex["content"])

    def test_explain_score_missing_node(self):
        m = make_mem()
        self.assertFalse(m.explain_score("nope")["found"])


if __name__ == "__main__":
    unittest.main()
