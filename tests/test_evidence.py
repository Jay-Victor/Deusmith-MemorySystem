# -*- coding: utf-8 -*-
"""证据存储单测 —— 本地内容寻址写入器 + 管线挂钩（评审第三梯队 #8）。"""
import tempfile
import unittest


class TestLocalEvidenceStore(unittest.TestCase):
    def test_save_read_verify_roundtrip(self):
        from deusmith.evidence import LocalEvidenceStore
        with tempfile.TemporaryDirectory() as d:
            es = LocalEvidenceStore(d)
            ev = es.save("用户说他喜欢摄影", source="chat")
            self.assertTrue(ev["evidence_id"].startswith("ev_"))
            self.assertTrue(ev["pointer_uri"].startswith("file://"))
            self.assertEqual(es.read(ev["evidence_id"]), "用户说他喜欢摄影")
            self.assertTrue(es.verify(ev["evidence_id"]))

    def test_tamper_detected(self):
        from deusmith.evidence import LocalEvidenceStore
        with tempfile.TemporaryDirectory() as d:
            es = LocalEvidenceStore(d)
            ev = es.save("原始证据内容", source="chat")
            # 找到落盘文件并篡改
            files = list(es.root.rglob("*.txt"))
            self.assertEqual(len(files), 1)
            files[0].write_text("被篡改的内容", encoding="utf-8")
            self.assertFalse(es.verify(ev["evidence_id"]))

    def test_same_content_dedupes(self):
        from deusmith.evidence import LocalEvidenceStore
        with tempfile.TemporaryDirectory() as d:
            es = LocalEvidenceStore(d)
            ev1 = es.save("同一段话")
            ev2 = es.save("同一段话")
            self.assertEqual(ev1["evidence_id"], ev2["evidence_id"])
            files = list(es.root.rglob("*.txt"))
            self.assertEqual(len(files), 1)


class TestPipelineEvidence(unittest.TestCase):
    def test_write_links_evidence_to_node(self):
        from deusmith.embeddings import HashEmbedding
        from deusmith.evidence import LocalEvidenceStore
        from deusmith.llm import RuleBasedLLM
        from deusmith.pipeline import DeusmithMemory
        from deusmith.store import InMemoryStore
        with tempfile.TemporaryDirectory() as d:
            store = InMemoryStore()
            mem = DeusmithMemory(store=store, embedder=HashEmbedding(dim=32),
                                 llm=RuleBasedLLM(), scope="t0/u1",
                                 evidence_store=LocalEvidenceStore(d))
            nodes = mem.write(mem.session("s1"), "我喜欢摄影。")
            self.assertTrue(nodes)
            self.assertTrue(nodes[0].evidence_refs)          # 节点挂了证据引用
            ev_id = nodes[0].evidence_refs[0]
            records = store.list_evidence(nodes[0].id)       # evidence 表有行
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["evidence_id"], ev_id)
            # 证据可回读
            self.assertEqual(mem.evidence_store.read(ev_id), "我喜欢摄影")

    def test_no_evidence_store_is_optional(self):
        from deusmith.embeddings import HashEmbedding
        from deusmith.llm import RuleBasedLLM
        from deusmith.pipeline import DeusmithMemory
        from deusmith.store import InMemoryStore
        mem = DeusmithMemory(store=InMemoryStore(), embedder=HashEmbedding(dim=32),
                             llm=RuleBasedLLM(), scope="t0/u1")
        nodes = mem.write(mem.session("s1"), "我喜欢摄影。")
        self.assertEqual(nodes[0].evidence_refs, ())


if __name__ == "__main__":
    unittest.main()
