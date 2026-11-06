# -*- coding: utf-8 -*-
"""Deusmith 可视化控制台 —— 本地 Web 服务（stdlib http.server，零第三方依赖）。

用法：
  python -m deusmith.webconsole                          # http://127.0.0.1:8765
  deusmith-web --port 9000 --scope tenant0/u1            # pip install -e . 后的入口点
  deusmith --web                                         # console 委托启动
  python -m deusmith.webconsole --smoke                  # 自检（CI 用）

页面：单文件 SPA（web/index.html，暗色主题）——写入/检索/注入预览、记忆浏览器、
遗忘曲线、记忆图谱、事件流、槽位版本链、离线治理、scope 切换。

API（JSON，同源）：
  GET  /                    index.html
  GET  /api/state           概览：配置/三层分布/总量/基尼/参数摘要
  POST /api/write           {text, source_auth} → 写入（完整管线：抽取→冲突→分层）
  POST /api/search          {query, topk} → 检索（score/rel/精排明细）
  POST /api/inject          {query, token_budget} → 上下文注入预览
  GET  /api/nodes           ?tier=&status=&q=&limit= 节点列表
  GET  /api/node/<id>       节点全参数 + explain_score 因子分解
  GET  /api/node/<id>/forget  遗忘曲线（S/τ_eff/归档期限/曲线点/冷热参考）
  GET  /api/node/<id>/why   命中历史（why_selected）
  GET  /api/graph           图谱数据（节点+边，服务端截断）
  GET  /api/events          ?limit= 近期事件
  GET  /api/slot            ?subject=&predicate= 槽位版本链（双时间）
  GET  /api/params          全参数 dump
  POST /api/govern          离线治理（衰减/剪枝/压缩/流转/审计）
  POST /api/scope           {scope} 切换 scope（store 共享，门面重建）
  POST /api/session/end     结束当前会话（摘要+晋升+持久化）

安全：默认只绑 127.0.0.1（本机工具，不做鉴权）；不要暴露到公网。
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from dataclasses import asdict
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from .algos import (
    archive_after_days, decay_recency, decay_stretch, gini, tau_eff_days,
)
from .console import build_memory_system, load_config
from .models import MemoryNode, NodeType, Tier, utcnow
from .offline import OfflineGovernor
from .params import Params
from .pipeline import DeusmithMemory

INDEX_PATH = Path(__file__).parent / "web" / "index.html"
SESSION_ID = "web-console"
_GRAPH_NODE_CAP = 400
_GRAPH_EDGE_CAP = 600


def _iso(dt) -> str | None:
    return dt.isoformat(timespec="seconds") if dt else None


def _node_json(n: MemoryNode, light: bool = False) -> dict:
    if light:
        return {
            "id": n.id, "content": n.content, "tier": n.tier.value,
            "status": n.status.value, "type": n.type.value,
            "weight": round(n.weight, 4), "confidence": round(n.confidence, 4),
            "importance": round(n.importance, 4),
            "activation_count": n.activation_count,
            "last_access_at": _iso(n.last_access_at),
            "subject": n.subject, "predicate": n.predicate,
            "sensitivity": n.sensitivity,
        }
    return {
        "id": n.id, "type": n.type.value, "content": n.content,
        "tier": n.tier.value, "status": n.status.value,
        "subject": n.subject, "predicate": n.predicate,
        "object_value": n.object_value, "polarity": n.polarity,
        "created_at": _iso(n.created_at), "updated_at": _iso(n.updated_at),
        "valid_from": _iso(n.valid_from), "valid_to": _iso(n.valid_to),
        "last_access_at": _iso(n.last_access_at),
        "weight": round(n.weight, 4), "confidence": round(n.confidence, 4),
        "importance": round(n.importance, 4), "emotion": round(n.emotion, 4),
        "source": n.source, "source_auth": round(n.source_auth, 3),
        "consistency": round(n.consistency, 3),
        "activation_count": n.activation_count,
        "cross_session_count": n.cross_session_count,
        "sensitivity": n.sensitivity,
        "explicit_remember": n.explicit_remember,
        "human_confirmed": n.human_confirmed,
        "supersedes": n.supersedes, "contested_by": n.contested_by,
        "evidence_refs": list(n.evidence_refs),
        "embedding_model": n.embedding_model,
        "dim": len(n.content_vector),
        "session_id": n.session_id, "generation": n.generation,
    }


_CURVE_DAYS = (0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365)


def _ref_node(importance: float, confidence: float, activation: int,
              emotion: float = 0.0) -> MemoryNode:
    return MemoryNode(type=NodeType.fact, content="遗忘曲线参考", scope="web/ref",
                      tier=Tier.middle, embedding_model="ref",
                      importance=importance, emotion=emotion,
                      confidence=confidence, activation_count=activation)


class WebApp:
    """控制台应用状态：store/embedder/llm/reranker 共享，按 scope 缓存门面。"""

    def __init__(self, store, embedder, llm, reranker, params: Params | None,
                 default_scope: str, backend_label: str = "memory"):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.reranker = reranker
        self.params = params or Params()
        self.backend = backend_label
        self.scope = default_scope
        self._mems: dict[str, DeusmithMemory] = {}
        self._lock = threading.RLock()

    @classmethod
    def from_memory(cls, mem: DeusmithMemory, backend_label: str = "memory") -> "WebApp":
        app = cls(mem.store, mem.embedder, mem.llm, mem.reranker, mem.params,
                  mem.scope, backend_label)
        app._mems[mem.scope] = mem
        return app

    def memory_for(self, scope: str) -> DeusmithMemory:
        with self._lock:
            mem = self._mems.get(scope)
            if mem is None:
                mem = DeusmithMemory(store=self.store, embedder=self.embedder,
                                     llm=self.llm, params=self.params,
                                     scope=scope, reranker=self.reranker)
                self._mems[scope] = mem
            return mem

    @property
    def mem(self) -> DeusmithMemory:
        return self.memory_for(self.scope)

    # ------------------------------------------------------------ 路由
    def handle_get(self, raw_path: str):
        u = urlparse(raw_path)
        path, qs = u.path, {k: v[0] for k, v in parse_qs(u.query).items()}
        if path in ("/", "/index.html"):
            return 200, "text/html; charset=utf-8", INDEX_PATH.read_bytes()
        if path == "/api/state":
            return 200, "application/json", _dump(self._state())
        if path == "/api/nodes":
            return 200, "application/json", _dump(self._nodes(qs))
        if path == "/api/graph":
            return 200, "application/json", _dump(self._graph())
        if path == "/api/events":
            return 200, "application/json", _dump(self._events(qs))
        if path == "/api/slot":
            return 200, "application/json", _dump(self._slot(qs))
        if path == "/api/params":
            return 200, "application/json", _dump(asdict(self.params))
        m = re.fullmatch(r"/api/node/([^/]+)", path)
        if m:
            return self._maybe_json(self._node_detail(m.group(1)))
        m = re.fullmatch(r"/api/node/([^/]+)/forget", path)
        if m:
            return self._maybe_json(self._forget(m.group(1)))
        m = re.fullmatch(r"/api/node/([^/]+)/why", path)
        if m:
            with self._lock:
                hits = self.mem.why_selected(m.group(1), limit=10)
            return 200, "application/json", _dump({"hits": hits})
        return 404, "application/json", _dump({"error": f"未知路径: {path}"})

    def handle_post(self, raw_path: str, body: dict):
        path = urlparse(raw_path).path
        if path == "/api/write":
            return self._maybe_json(self._write(body))
        if path == "/api/search":
            return self._maybe_json(self._search(body))
        if path == "/api/inject":
            return self._maybe_json(self._inject(body))
        if path == "/api/govern":
            return self._maybe_json(self._govern())
        if path == "/api/scope":
            return self._maybe_json(self._set_scope(body))
        if path == "/api/session/end":
            with self._lock:
                self.mem.end_session(self.mem.session(SESSION_ID))
            return 200, "application/json", _dump({"ok": True})
        return 404, "application/json", _dump({"error": f"未知路径: {path}"})

    @staticmethod
    def _maybe_json(result):
        """子处理器返回 (status, obj) 或 (status, obj) 含 error。"""
        status, obj = result
        return status, "application/json", _dump(obj)

    # ------------------------------------------------------------ 处理器
    def _state(self) -> dict:
        nodes = self.store.active_nodes(self.scope)
        ws = [n.weight for n in nodes]
        return {
            "scope": self.scope, "backend": self.backend,
            "embedding": getattr(self.embedder, "model_name",
                                 type(self.embedder).__name__),
            "llm": type(self.llm).__name__,
            "rerank": (getattr(self.reranker, "model_name", None)
                       if self.reranker else None),
            "counts": self.store.counts(self.scope),
            "nodes_active": len(nodes),
            "tier_stats": {f"{t}|{s}": c for (t, s), c
                           in self.store.tier_stats(self.scope).items()},
            "gini": round(gini(ws), 4) if ws else 0.0,
            "gini_alert": self.params.tier.gini_alert,
            "allowed_sensitivity": self.mem.allowed_sensitivity,
            "params": {
                "tau_base_days": self.params.decay.tau_base_days,
                "rrf_k": self.params.retrieval.rrf_k,
                "rerank_weight": self.params.retrieval.rerank_weight,
                "mmr_lambda": self.params.score.mmr_lambda,
                "token_budget": self.params.retrieval.token_budget,
            },
        }

    def _write(self, body: dict):
        text = (body.get("text") or "").strip()
        if not text:
            return 400, {"error": "text 必填"}
        try:
            auth = float(body.get("source_auth", 0.8))
        except (TypeError, ValueError):
            auth = 0.8
        auth = min(1.0, max(0.0, auth))
        with self._lock:
            s = self.mem.session(SESSION_ID)
            nodes = self.mem.write(s, text, source_auth=auth)
        return 200, {"written": [_node_json(n) for n in nodes]}

    def _search(self, body: dict):
        query = (body.get("query") or "").strip()
        if not query:
            return 400, {"error": "query 必填"}
        topk = max(1, min(30, int(body.get("topk", 8) or 8)))
        with self._lock:
            s = self.mem.session(SESSION_ID)
            res = self.mem.retrieve(s, query, topk=topk)
        return 200, {
            "results": [{
                "id": n.id, "content": n.content,
                "score": round(sc, 4), "rel": round(rel, 4),
                "tier": n.tier.value, "status": n.status.value,
                "last_access_at": _iso(n.last_access_at),
            } for n, sc, rel in res.selected],
            "conflict_flags": res.conflict_flags,
            "rerank": res.trace.get("rerank"),
            "trace_id": res.trace.get("trace_id"),
        }

    def _inject(self, body: dict):
        query = (body.get("query") or "").strip()
        if not query:
            return 400, {"error": "query 必填"}
        budget = int(body.get("token_budget", 0) or 0) or None
        with self._lock:
            s = self.mem.session(SESSION_ID)
            out = self.mem.inject_context(s, query, token_budget=budget)
        return 200, {
            "system": out["system"], "user": out["user"], "tool": out["tool"],
            "tokens": out["tokens"], "degraded": out["degraded"],
            "conflict_flags": out["conflict_flags"],
            "selected": out["selected"], "trace_id": out["trace_id"],
        }

    def _nodes(self, qs: dict) -> dict:
        tier = qs.get("tier") or None
        status = qs.get("status") or None
        limit = max(1, min(500, int(qs.get("limit", 200) or 200)))
        q = (qs.get("q") or "").strip().lower()
        nodes = self.store.list_nodes(self.scope, tier=tier, status=status,
                                      limit=500 if q else limit)
        if q:
            nodes = [n for n in nodes if q in (n.content or "").lower()][:limit]
        return {"nodes": [_node_json(n, light=True) for n in nodes]}

    def _node_detail(self, node_id: str):
        n = self.store.get_node(node_id)
        if n is None:
            return 404, {"error": f"节点不存在: {node_id}"}
        with self._lock:
            explain = self.mem.explain_score(node_id)
        return 200, {"node": _node_json(n), "explain": explain}

    def _forget(self, node_id: str):
        n = self.store.get_node(node_id)
        if n is None:
            return 404, {"error": f"节点不存在: {node_id}"}
        P = self.params
        now = utcnow()
        probe = copy.copy(n)
        probe.last_access_at = now

        def curve(node):
            p = copy.copy(node)
            p.last_access_at = now
            return [{"days": d,
                     "recency": round(decay_recency(p, now + timedelta(days=d), P), 4)}
                    for d in _CURVE_DAYS]

        cold = _ref_node(0.3, 0.4, 0)
        hot = _ref_node(0.8, 0.9, 3, emotion=0.5)
        return 200, {
            "node": _node_json(n, light=True),
            "S": round(decay_stretch(n, P), 4),
            "tau_days": round(tau_eff_days(n, P), 2),
            "archive_days": round(archive_after_days(n, P), 1),
            "archive_floor": P.tier.archive_score_floor,
            "current_dt_days": round(
                (now - n.last_access_at).total_seconds() / 86400.0, 2),
            "current_recency": round(decay_recency(n, now, P), 4),
            "curve": curve(n),
            "refs": {
                "cold": {"S": round(decay_stretch(cold, P), 3),
                         "curve": curve(cold)},
                "hot": {"S": round(decay_stretch(hot, P), 3),
                        "curve": curve(hot)},
            },
        }

    def _graph(self) -> dict:
        nodes = self.store.list_nodes(self.scope, limit=_GRAPH_NODE_CAP)
        truncated = False
        if len(nodes) >= _GRAPH_NODE_CAP:
            truncated = True
        ids = {n.id for n in nodes}
        edges = [e for e in self.store.all_edges(self.scope)
                 if e.src_id in ids and e.dst_id in ids][:_GRAPH_EDGE_CAP]
        return {
            "nodes": [{"id": n.id, "content": n.content, "tier": n.tier.value,
                       "status": n.status.value, "weight": round(n.weight, 3),
                       "type": n.type.value} for n in nodes],
            "edges": [{"src": e.src_id, "dst": e.dst_id,
                       "type": e.type.value, "weight": round(e.weight, 3)}
                      for e in edges],
            "truncated": truncated,
        }

    def _events(self, qs: dict) -> dict:
        limit = max(1, min(200, int(qs.get("limit", 60) or 60)))
        out = []
        for e in self.store.recent_events(self.scope, limit=limit):
            payload = e.get("payload")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except (ValueError, TypeError):
                    pass
            out.append({"ts": _iso(e.get("ts")), "type": e.get("event_type"),
                        "payload": payload})
        return {"events": out}

    def _slot(self, qs: dict) -> dict:
        subject = (qs.get("subject") or "").strip()
        predicate = (qs.get("predicate") or "").strip()
        if not subject or not predicate:
            return {"versions": [],
                    "hint": "需要 subject 与 predicate 参数"}
        with self._lock:
            versions = self.mem.slot_history(subject, predicate)
        return {"versions": versions}

    def _govern(self):
        with self._lock:
            gov = OfflineGovernor(self.store, self.embedder, self.llm,
                                  self.params)
            report = gov.run(self.scope)
        return 200, {"report": report}

    def _set_scope(self, body: dict):
        scope = (body.get("scope") or "").strip()
        if not scope:
            return 400, {"error": "scope 必填"}
        self.scope = scope
        self.memory_for(scope)          # 预热门面
        return 200, self._state()


def _dump(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")


def _make_handler(app: WebApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DeusmithWeb/0.3"
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, ctype: str, body: bytes):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            try:
                status, ctype, body = app.handle_get(self.path)
            except Exception as exc:
                status, ctype, body = 500, "application/json", _dump(
                    {"error": f"{type(exc).__name__}: {exc}"})
            self._send(status, ctype, body)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length) if length else b"{}"
                body = json.loads(raw.decode("utf-8") or "{}")
                status, ctype, payload = app.handle_post(self.path, body)
            except json.JSONDecodeError as exc:
                status, ctype, payload = 400, "application/json", _dump(
                    {"error": f"JSON 解析失败: {exc}"})
            except Exception as exc:
                status, ctype, payload = 500, "application/json", _dump(
                    {"error": f"{type(exc).__name__}: {exc}"})
            self._send(status, ctype, payload)

        def log_message(self, format, *args):    # noqa: A002 — 基类签名
            pass

    return Handler


def make_server(app: WebApp, host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _make_handler(app))


# ================================================================ 自检

def smoke() -> int:
    """Web 控制台自检：起临时服务，打全部 API 端点断言。"""
    from .embeddings import HashEmbedding
    from .llm import RuleBasedLLM
    from .store import InMemoryStore

    checks: list[tuple[str, bool, str]] = []

    def ck(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    mem = DeusmithMemory(store=InMemoryStore(), embedder=HashEmbedding(dim=64),
                         llm=RuleBasedLLM(), scope="websmoke/u1")
    app = WebApp.from_memory(mem, "memory")
    srv = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def get(p: str):
        with urllib.request.urlopen(base + p) as r:
            return r.status, r.read()

    def get_json(p: str):
        s, b = get(p)
        return s, json.loads(b.decode("utf-8"))

    def post(p: str, obj: dict):
        req = urllib.request.Request(
            base + p, data=json.dumps(obj).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read().decode("utf-8"))

    try:
        # ---- 页面
        s, body = get("/")
        ck("GET / 返回 200 且含页面标记", s == 200 and b"Deusmith" in body)

        # ---- state
        s, st = get_json("/api/state")
        ck("state: scope/backend/tier_stats/counts 齐备",
           s == 200 and st["scope"] == "websmoke/u1" and st["backend"] == "memory"
           and "tier_stats" in st and "counts" in st)

        # ---- 写入
        s, w1 = post("/api/write", {"text": "请记住：我对花粉过敏。"})
        s2, w2 = post("/api/write", {"text": "我喜欢摄影，周末常去拍夜景。"})
        s3, w3 = post("/api/write", {"text": "我现在住在北京。", "source_auth": 0.95})
        ck("write: 三条写入成功",
           s == s2 == s3 == 200 and all(len(w["written"]) >= 1
                                        for w in (w1, w2, w3)))
        ck("write: 返回节点带 tier/status/weight",
           "tier" in w1["written"][0] and "weight" in w1["written"][0])

        # ---- 检索
        s, res = post("/api/search", {"query": "用户的过敏情况", "topk": 5})
        ck("search: 有命中", s == 200 and len(res["results"]) >= 1)
        ck("search: top1 是过敏记忆",
           "花粉" in res["results"][0]["content"], res["results"][0]["content"][:20])
        ck("search: 结果带 score/rel",
           "score" in res["results"][0] and "rel" in res["results"][0])
        nid = res["results"][0]["id"]

        # ---- nodes
        s, ns = get_json("/api/nodes?limit=50")
        ck("nodes: 列表 ≥3", s == 200 and len(ns["nodes"]) >= 3)
        s, ns_m = get_json("/api/nodes?tier=middle")
        ck("nodes: tier 过滤生效",
           all(n["tier"] == "middle" for n in ns_m["nodes"]))
        s, ns_q = get_json("/api/nodes?q=" + quote("摄影"))
        ck("nodes: 文本过滤生效",
           all("摄影" in n["content"] for n in ns_q["nodes"])
           and len(ns_q["nodes"]) >= 1)

        # ---- node detail
        s, nd = get_json(f"/api/node/{nid}")
        ck("node: 全参数 + explain 因子",
           s == 200 and "valid_from" in nd["node"]
           and "factors" in nd["explain"] and "score_at_full_rel" in nd["explain"])

        # ---- forget
        s, fg = get_json(f"/api/node/{nid}/forget")
        ck("forget: S/τ/归档期限/曲线齐备",
           s == 200 and fg["S"] > 1 and fg["tau_days"] > 0
           and fg["archive_days"] > 0 and len(fg["curve"]) >= 10)
        ck("forget: 冷热参考曲线", "cold" in fg["refs"] and "hot" in fg["refs"]
           and fg["refs"]["hot"]["S"] > fg["refs"]["cold"]["S"])
        ck("forget: 当前 recency ∈ [0,1]",
           0.0 <= fg["current_recency"] <= 1.0)

        # ---- why
        s, wy = get_json(f"/api/node/{nid}/why")
        ck("why: 检索命中历史 ≥1", s == 200 and len(wy["hits"]) >= 1)

        # ---- graph
        s, gr = get_json("/api/graph")
        ck("graph: 节点 ≥3 且带 weight",
           s == 200 and len(gr["nodes"]) >= 3 and "weight" in gr["nodes"][0])

        # ---- events
        s, ev = get_json("/api/events?limit=20")
        ck("events: 含 write 事件",
           s == 200 and any(e["type"] == "write" for e in ev["events"]))

        # ---- slot
        s, sl = get_json("/api/slot?subject=user&predicate=" + quote("过敏"))
        ck("slot: 版本链 ≥1", s == 200 and len(sl["versions"]) >= 1)

        # ---- inject
        s, ij = post("/api/inject", {"query": "用户的过敏情况"})
        ck("inject: 三段 + tokens/degraded",
           s == 200 and all(k in ij for k in ("system", "user", "tool",
                                              "tokens", "degraded")))

        # ---- govern
        s, gv = post("/api/govern", {})
        rep = gv.get("report", {})
        ck("govern: 报告键齐备",
           s == 200 and "edges_decayed" in rep and "gini_weight" in rep
           and "summaries_created" in rep)

        # ---- 404
        try:
            get_json("/api/nope")
            ck("404: 未知路径被拒", False)
        except urllib.error.HTTPError as e:
            ck("404: 未知路径被拒", e.code == 404)
        try:
            get_json("/api/node/m_missing_x")
            ck("404: 节点不存在被拒", False)
        except urllib.error.HTTPError as e:
            ck("404: 节点不存在被拒", e.code == 404)

        # ---- scope 切换
        s, st2 = post("/api/scope", {"scope": "websmoke/u2"})
        ck("scope 切换: state 更新且新 scope 为空",
           s == 200 and st2["scope"] == "websmoke/u2"
           and st2["counts"]["nodes_total"] == 0)

        # ---- session end
        s, se = post("/api/session/end", {})
        ck("session/end: ok", s == 200 and se.get("ok") is True)
    finally:
        srv.shutdown()
        srv.server_close()

    fails = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))
    print(f"\n== webconsole smoke：{len(checks) - len(fails)}/{len(checks)} 通过 ==")
    return 1 if fails else 0


# ================================================================ 入口

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="deusmith-web",
        description="Deusmith-MemorySystem 可视化控制台（本地 Web 服务）")
    ap.add_argument("--config", help="配置文件（默认 ./config.json 或 $DEUSMITH_CONFIG）")
    ap.add_argument("--scope", default="console/default", help="初始 scope")
    ap.add_argument("--memory", action="store_true", help="强制内存后端")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1，勿暴露公网）")
    ap.add_argument("--port", type=int, default=8765, help="端口（默认 8765）")
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    ap.add_argument("--smoke", action="store_true", help="自检模式（CI/回归用）")
    args = ap.parse_args(argv)

    if args.smoke:
        return smoke()

    try:
        cfg = load_config(args.config)
        mem, backend = build_memory_system(cfg, args.scope,
                                           force_memory=args.memory)
    except Exception as exc:
        print(f"[启动失败] {exc}", file=sys.stderr)
        return 2

    app = WebApp.from_memory(mem, backend)
    try:
        srv = make_server(app, args.host, args.port)
    except OSError as exc:
        print(f"[端口占用] {args.host}:{args.port}（{exc}）", file=sys.stderr)
        mem.store.close()
        return 2
    url = f"http://{args.host}:{srv.server_address[1]}/"
    print("=" * 56)
    print("  Deusmith · 记忆系统可视化控制台")
    print(f"  地址:    {url}")
    print(f"  后端:    {backend} · 嵌入: {getattr(app.embedder, 'model_name', '?')}"
          f" · 精排: {getattr(app.reranker, 'model_name', '未启用') if app.reranker else '未启用'}")
    print(f"  scope:   {app.scope}（页面右上可切换）")
    print("  退出:    Ctrl+C")
    print("=" * 56)
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[退出]")
    finally:
        srv.server_close()
        mem.store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
