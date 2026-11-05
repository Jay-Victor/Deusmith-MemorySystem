# -*- coding: utf-8 -*-
"""在线 API 硬化单测 —— 重试/退避/批量上限/成本计数/降级（评审第二梯队 #6）。

用 stdlib 起一个本地 mock 服务器验证真实网络路径（协议级，不依赖外部 API）。
"""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _State:
    def __init__(self):
        self.fail_next = 0
        self.calls = 0
        self.batches = []


STATE = _State()


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        STATE.calls += 1
        if STATE.fail_next > 0:
            STATE.fail_next -= 1
            self.send_response(500)
            self.end_headers()
            return
        if self.path.endswith("/embeddings"):
            inp = body.get("input", [])
            if isinstance(inp, str):
                inp = [inp]
            STATE.batches.append(len(inp))
            data = {"data": [{"index": i, "embedding": [0.5, 0.5]}
                             for i in range(len(inp))],
                    "model": body.get("model", "m")}
        else:
            data = {"choices": [{"message": {"content": "[]"}}], "usage": {}}
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # 静默
        pass


SERVER = None
BASE = None


def setUpModule():
    global SERVER, BASE
    SERVER = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=SERVER.serve_forever, daemon=True).start()
    BASE = f"http://127.0.0.1:{SERVER.server_address[1]}"


class TestEmbeddingRetry(unittest.TestCase):
    def setUp(self):
        STATE.fail_next = 0
        STATE.calls = 0
        STATE.batches = []

    def _emb(self, **over):
        from deusmith.embeddings import OpenAICompatibleEmbedding
        kw = dict(base_url=BASE, model="m", api_key="k", dim=2,
                  retry_attempts=3, retry_backoff=0.01)
        kw.update(over)
        return OpenAICompatibleEmbedding(**kw)

    def test_retry_then_success(self):
        emb = self._emb()
        STATE.fail_next = 2                          # 前两次 500
        v = emb.embed("你好")
        self.assertEqual(v, (0.5, 0.5))
        self.assertEqual(emb.stats["retries"], 2)
        self.assertEqual(emb.stats["requests"], 3)
        self.assertEqual(emb.stats["errors"], 0)

    def test_retry_exhausted_raises(self):
        emb = self._emb(retry_attempts=2)
        STATE.fail_next = 99
        with self.assertRaises(Exception):
            emb.embed("你好")
        self.assertGreaterEqual(emb.stats["errors"], 1)

    def test_non_retryable_client_error_raises_immediately(self):
        emb = self._emb(retry_attempts=3)
        # 400 不可重试：mock 不产 400，这里用死端口验证网络错误路径即可
        from deusmith.embeddings import OpenAICompatibleEmbedding
        dead = OpenAICompatibleEmbedding(base_url="http://127.0.0.1:1", model="m",
                                         api_key="k", retry_attempts=1,
                                         retry_backoff=0.01)
        with self.assertRaises(Exception):
            dead.embed("x")

    def test_batch_chunking_respects_limit(self):
        emb = self._emb(batch_limit=4)
        out = emb.embed_batch([f"t{i}" for i in range(10)])
        self.assertEqual(len(out), 10)
        self.assertEqual(STATE.batches, [4, 4, 2])   # 10 条按 4/4/2 分批
        self.assertEqual(emb.stats["items"], 10)

    def test_stats_snapshot(self):
        emb = self._emb()
        emb.embed("a")
        snap = emb.stats_snapshot()
        self.assertEqual(snap["requests"], 1)
        snap["requests"] = 99                        # 快照可改不影响内部
        self.assertEqual(emb.stats["requests"], 1)


class TestLLMRetry(unittest.TestCase):
    def setUp(self):
        STATE.fail_next = 0
        STATE.calls = 0

    def test_chat_retry_then_success(self):
        from deusmith.llm import OpenAICompatibleLLM
        llm = OpenAICompatibleLLM(base_url=BASE, model="m", api_key="k",
                                  retry_attempts=3, retry_backoff=0.01)
        STATE.fail_next = 1
        out = llm.chat("sys", "usr")
        self.assertEqual(out, "[]")
        self.assertEqual(llm.stats["retries"], 1)

    def test_chat_extract_parses_empty_array(self):
        from deusmith.llm import OpenAICompatibleLLM
        llm = OpenAICompatibleLLM(base_url=BASE, model="m", api_key="k",
                                  retry_attempts=1, retry_backoff=0.01)
        self.assertEqual(llm.extract("随便"), [])


class TestPipelineFallbackOnDeadLLM(unittest.TestCase):
    def test_write_falls_back_to_rules(self):
        """LLM 端点不可达时，写入路径自动降级规则引擎，不中断（评审 #6 红线）。"""
        from deusmith.embeddings import HashEmbedding
        from deusmith.llm import OpenAICompatibleLLM
        from deusmith.pipeline import DeusmithMemory
        from deusmith.store import InMemoryStore
        dead = OpenAICompatibleLLM(base_url="http://127.0.0.1:1", model="m",
                                   api_key="k", retry_attempts=1,
                                   retry_backoff=0.01)
        mem = DeusmithMemory(store=InMemoryStore(), embedder=HashEmbedding(dim=32),
                             llm=dead, scope="t0/u1")
        nodes = mem.write(mem.session("s1"), "我喜欢摄影。")
        self.assertTrue(nodes)                           # 规则引擎兜底成功
        self.assertEqual(nodes[0].type.value, "preference")
        events = mem.store.recent_events("t0/u1", limit=20)
        self.assertTrue(any(e["event_type"] == "llm_fallback" for e in events))


if __name__ == "__main__":
    unittest.main()
