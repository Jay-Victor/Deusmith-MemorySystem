# -*- coding: utf-8 -*-
"""LLM 挂钩单测 —— 规则引擎为默认实现（离线保底），OpenAI 兼容接口可选。

用户前提：记忆系统多层次管理途中也需要 LLM（抽取/摘要/仲裁），
但默认必须纯规则可跑（评审前提：任何 LLM 不可用时全链路可跑）。
"""
import unittest


class TestRuleBasedExtraction(unittest.TestCase):
    def _extract(self, text):
        from deusmith.llm import RuleBasedLLM
        return RuleBasedLLM().extract(text)

    def test_positive_preference(self):
        items = self._extract("我喜欢摄影，周末常去拍夜景。")
        prefs = [i for i in items if i["type"] == "preference"]
        self.assertTrue(prefs)
        self.assertEqual(prefs[0]["subject"], "user")
        self.assertEqual(prefs[0]["predicate"], "喜欢")
        self.assertEqual(prefs[0]["object_value"], "摄影")
        self.assertEqual(prefs[0]["polarity"], 1)

    def test_negative_preference(self):
        items = self._extract("我讨厌加班。")
        prefs = [i for i in items if i["type"] == "preference"]
        self.assertTrue(prefs)
        self.assertEqual(prefs[0]["predicate"], "讨厌")
        self.assertEqual(prefs[0]["polarity"], -1)

    def test_negated_preference(self):
        """『不再喜欢』必须产出 polarity=-1 —— 错误#1 的写入侧防线。"""
        items = self._extract("我不再喜欢摄影了。")
        prefs = [i for i in items if i["type"] == "preference"]
        self.assertTrue(prefs)
        self.assertEqual(prefs[0]["predicate"], "喜欢")
        self.assertEqual(prefs[0]["polarity"], -1)
        self.assertEqual(prefs[0]["object_value"], "摄影")

    def test_residence_fact(self):
        items = self._extract("我现在住在上海。")
        facts = [i for i in items if i["type"] == "fact"]
        self.assertTrue(facts)
        self.assertEqual(facts[0]["predicate"], "住在")
        self.assertEqual(facts[0]["object_value"], "上海")

    def test_identity(self):
        items = self._extract("我叫李雷，是一名后端工程师。")
        ids = [i for i in items if i["type"] == "identity_policy"]
        self.assertTrue(ids)
        self.assertEqual(ids[0]["object_value"], "李雷")

    def test_explicit_remember_marker(self):
        items = self._extract("请记住：我对花粉过敏。")
        self.assertTrue(any(i.get("explicit_remember") for i in items))

    def test_emotion_detected(self):
        items = self._extract("今天升职了，特别开心！")
        self.assertTrue(any(i.get("emotion", 0) >= 0.5 for i in items))

    def test_importance_ordering(self):
        """显式记住 > 身份/偏好 > 事实 > 事件。"""
        a = self._extract("请记住我对花粉过敏")[0]
        b = self._extract("上周三下了一场雨")[0]
        self.assertGreaterEqual(a["importance"], b["importance"])

    def test_plain_event_fallback(self):
        items = self._extract("上周三下了一场雨。")
        self.assertTrue(items)
        self.assertEqual(items[0]["type"], "event")

    def test_empty_text(self):
        self.assertEqual(self._extract(""), [])


class TestRuleBasedSummarize(unittest.TestCase):
    def test_summarize_joins_and_truncates(self):
        from deusmith.llm import RuleBasedLLM
        s = RuleBasedLLM().summarize(["用户喜欢摄影", "用户住在上海", "用户讨厌加班"],
                                      max_tokens=20)
        self.assertIn("摄影", s)
        self.assertIn("上海", s)

    def test_summarize_empty(self):
        from deusmith.llm import RuleBasedLLM
        self.assertEqual(RuleBasedLLM().summarize([], max_tokens=20), "")


class TestRuleBasedArbitration(unittest.TestCase):
    def test_higher_authority_wins(self):
        from deusmith.llm import RuleBasedLLM
        from deusmith.models import NodeType, Tier, MemoryNode
        a = MemoryNode(type=NodeType.fact, content="旧说法", scope="t/u", tier=Tier.middle,
                       embedding_model="x", source_auth=0.9, confidence=0.7)
        b = MemoryNode(type=NodeType.fact, content="新说法", scope="t/u", tier=Tier.middle,
                       embedding_model="x", source_auth=0.5, confidence=0.9)
        self.assertEqual(RuleBasedLLM().arbitrate(a, b), "prefer_a")

    def test_close_call_marks_contested(self):
        from deusmith.llm import RuleBasedLLM
        from deusmith.models import NodeType, Tier, MemoryNode
        a = MemoryNode(type=NodeType.fact, content="说法A", scope="t/u", tier=Tier.middle,
                       embedding_model="x", source_auth=0.7, confidence=0.7)
        b = MemoryNode(type=NodeType.fact, content="说法B", scope="t/u", tier=Tier.middle,
                       embedding_model="x", source_auth=0.7, confidence=0.72)
        self.assertEqual(RuleBasedLLM().arbitrate(a, b), "contested")

    def test_deep_node_never_loses_silently(self):
        """深层记忆即使权威低也不被规则仲裁直接替代（红线：不允许静默覆盖深层记忆）。"""
        from deusmith.llm import RuleBasedLLM
        from deusmith.models import NodeType, Tier, MemoryNode
        deep = MemoryNode(type=NodeType.fact, content="深层事实", scope="t/u", tier=Tier.deep,
                          embedding_model="x", source_auth=0.5, confidence=0.6)
        new = MemoryNode(type=NodeType.fact, content="挑战深层", scope="t/u", tier=Tier.middle,
                         embedding_model="x", source_auth=0.95, confidence=0.95)
        self.assertEqual(RuleBasedLLM().arbitrate(deep, new), "contested")  # 只能争议，等人工


class TestOpenAICompatibleLLM(unittest.TestCase):
    def test_config_and_prompt_building_offline(self):
        from deusmith.llm import OpenAICompatibleLLM
        llm = OpenAICompatibleLLM(base_url="https://api.example.com/v1",
                                  model="gpt-test", api_key="sk-x")
        msgs = llm._build_messages("system-prompt", "user-input")
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[-1]["content"], "user-input")
        self.assertIn("/chat/completions", llm.endpoint)


if __name__ == "__main__":
    unittest.main()
