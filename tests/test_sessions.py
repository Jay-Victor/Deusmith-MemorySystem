# -*- coding: utf-8 -*-
"""浅层会话持久化单测 —— 进程重启恢复（评审第二梯队 #5）。"""
import unittest


def make_mem(store):
    from deusmith.embeddings import HashEmbedding
    from deusmith.llm import RuleBasedLLM
    from deusmith.pipeline import DeusmithMemory
    return DeusmithMemory(store=store, embedder=HashEmbedding(dim=32),
                          llm=RuleBasedLLM(), scope="t0/u1")


class TestSessionPersistence(unittest.TestCase):
    def test_roundtrip_across_instances(self):
        from deusmith.store import InMemoryStore
        store = InMemoryStore()
        m = make_mem(store)
        s = m.session("sess-p1")
        m.write(s, "我喜欢摄影。")
        m.write(s, "我现在住在北京。")
        m.end_session(s)
        # 新实例共享 store —— 模拟进程重启
        m2 = make_mem(store)
        s2 = m2.session("sess-p1")
        self.assertEqual(len(s2.rounds), 2)
        self.assertEqual(s2.rounds[0].content, "我喜欢摄影。")
        self.assertEqual(s2.rounds[-1].content, "我现在住在北京。")
        self.assertIsNotNone(s2.closed_at)

    def test_full_history_beyond_window(self):
        from deusmith.store import InMemoryStore
        store = InMemoryStore()
        m = make_mem(store)
        s = m.session("sess-p2")
        for i in range(12):                       # 超过窗口 N=10
            m.write(s, f"第{i}轮。")
        raw = m.store.load_session_raw("t0/u1", "sess-p2")
        self.assertEqual(len(raw["rounds"]), 12)  # 全量落库
        self.assertIsNotNone(raw["summary"])      # 滚动摘要持久化
        m2 = make_mem(store)
        s2 = m2.session("sess-p2")
        self.assertLessEqual(len(s2.rounds), 10)  # 内存窗口
        self.assertEqual(s2.rounds[-1].content, "第11轮。")

    def test_scope_isolation(self):
        from deusmith.store import InMemoryStore
        store = InMemoryStore()
        m = make_mem(store)
        m.write(m.session("sess-x"), "你好。")
        raw = store.load_session_raw("t0/other", "sess-x")
        self.assertIsNone(raw)                    # 跨 scope 不可见（I1）

    def test_unknown_session_creates_new(self):
        from deusmith.store import InMemoryStore
        m = make_mem(InMemoryStore())
        s = m.session("no-such")
        self.assertEqual(len(s.rounds), 0)
        self.assertEqual(s.rolling_summary, "")


if __name__ == "__main__":
    unittest.main()
