# -*- coding: utf-8 -*-
"""Embedding Provider 单测。

HashEmbedding 是零依赖离线默认实现（评审前提：默认全链路离线可跑）；
本地模型与 API Provider 走同一接口（用户前提：支持本地模型或 API）。
"""
import math
import unittest


class TestHashEmbedding(unittest.TestCase):
    def test_deterministic(self):
        from deusmith.embeddings import HashEmbedding
        e = HashEmbedding(dim=128)
        self.assertEqual(e.embed("我喜欢摄影"), e.embed("我喜欢摄影"))

    def test_dimension_and_model_name(self):
        from deusmith.embeddings import HashEmbedding
        e = HashEmbedding(dim=64)
        self.assertEqual(len(e.embed("hello")), 64)
        self.assertIn("hash", e.model_name)
        self.assertEqual(e.dim, 64)

    def test_l2_normalized(self):
        from deusmith.embeddings import HashEmbedding
        e = HashEmbedding(dim=256)
        v = e.embed("用户偏好会议时间安排在早晨")
        norm = math.sqrt(sum(x * x for x in v))
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_similar_beats_dissimilar(self):
        from deusmith.embeddings import cosine
        from deusmith.embeddings import HashEmbedding
        e = HashEmbedding(dim=256)
        a = e.embed("我喜欢摄影")
        b = e.embed("我热爱摄影")
        c = e.embed("今天的股价大跌")
        self.assertGreater(cosine(a, b), cosine(a, c))

    def test_paraphrase_stays_related(self):
        from deusmith.embeddings import cosine
        from deusmith.embeddings import HashEmbedding
        e = HashEmbedding(dim=256)
        q = e.embed("用户的会议时间偏好")
        m = e.embed("用户喜欢把会议安排在早上九点")
        unrelated = e.embed("量子计算机的纠错码")
        self.assertGreater(cosine(q, m), cosine(q, unrelated))

    def test_english_works(self):
        from deusmith.embeddings import cosine
        from deusmith.embeddings import HashEmbedding
        e = HashEmbedding(dim=256)
        self.assertGreater(
            cosine(e.embed("I like photography"), e.embed("I love photography")),
            cosine(e.embed("I like photography"), e.embed("stock market crash")),
        )

    def test_negation_is_similar_lexically(self):
        """词法层：否定句与肯定句高度相似——这正是冲突判定不能用相似度的证据（错误#1），
        此测试把该事实钉死，防止后人用向量相似度判冲突。"""
        from deusmith.embeddings import cosine
        from deusmith.embeddings import HashEmbedding
        e = HashEmbedding(dim=256)
        sim = cosine(e.embed("我喜欢摄影"), e.embed("我不喜欢摄影了"))
        self.assertGreater(sim, 0.6)

    def test_empty_text_returns_zero_vector(self):
        from deusmith.embeddings import HashEmbedding
        e = HashEmbedding(dim=64)
        v = e.embed("")
        self.assertEqual(len(v), 64)
        self.assertEqual(sum(x * x for x in v), 0.0)

    def test_embed_batch_consistent_with_single(self):
        from deusmith.embeddings import HashEmbedding
        e = HashEmbedding(dim=64)
        batch = e.embed_batch(["甲", "乙"])
        self.assertEqual(batch[0], e.embed("甲"))
        self.assertEqual(batch[1], e.embed("乙"))


class TestProviderFactory(unittest.TestCase):
    def test_hash_provider(self):
        from deusmith.embeddings import HashEmbedding, get_provider
        p = get_provider({"embedding": {"provider": "hash", "dim": 96}})
        self.assertIsInstance(p, HashEmbedding)
        self.assertEqual(p.dim, 96)

    def test_default_is_hash(self):
        from deusmith.embeddings import HashEmbedding, get_provider
        self.assertIsInstance(get_provider({}), HashEmbedding)

    def test_openai_provider_builds_endpoint_without_network(self):
        from deusmith.embeddings import OpenAICompatibleEmbedding, get_provider
        p = get_provider({"embedding": {
            "provider": "openai",
            "base_url": "https://api.example.com/v1",
            "model": "text-embedding-3-small",
            "api_key": "sk-test",
            "dim": 1536,
        }})
        self.assertIsInstance(p, OpenAICompatibleEmbedding)
        self.assertEqual(p.model_name, "text-embedding-3-small")
        self.assertIn("/embeddings", p.endpoint)

    def test_openai_provider_reads_env(self):
        import os
        from deusmith.embeddings import OpenAICompatibleEmbedding, get_provider
        os.environ["DEUSMITH_EMBEDDING_API_KEY"] = "sk-env"
        try:
            p = get_provider({"embedding": {"provider": "openai", "model": "m"}})
            self.assertIsInstance(p, OpenAICompatibleEmbedding)
            self.assertEqual(p.api_key, "sk-env")
        finally:
            del os.environ["DEUSMITH_EMBEDDING_API_KEY"]

    def test_unknown_provider_raises(self):
        from deusmith.embeddings import get_provider
        with self.assertRaises(ValueError):
            get_provider({"embedding": {"provider": "nope"}})

    def test_local_provider_optional(self):
        """本地模型 Provider：未安装 sentence-transformers 时报 ImportError 而非崩溃整个包。"""
        from deusmith.embeddings import get_provider
        try:
            p = get_provider({"embedding": {"provider": "local", "model_name": "BAAI/bge-small-zh"}})
            self.assertTrue(p.model_name)
        except ImportError:
            pass  # 环境未装可选依赖，允许


if __name__ == "__main__":
    unittest.main()
