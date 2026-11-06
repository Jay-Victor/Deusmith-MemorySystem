# -*- coding: utf-8 -*-
"""Deusmith 控制台 —— CLI 一次性命令 + 交互式 REPL（零第三方依赖）。

用法：
  python -m deusmith.console                      # 交互 REPL（无 config.json 时内存模式）
  python -m deusmith.console write "我对花粉过敏"   # 一次性命令后退出
  python -m deusmith.console --smoke              # 控制台/精排自检（CI 用）
  deusmith …                                      # pip install -e . 后的入口点

命令（REPL 与一次性模式同集）：
  write <文本>            写入记忆（经抽取→冲突仲裁→分层落库）
  search <查询> [-k N]    多路召回+融合+精排检索，显示得分明细
  inject <查询>           模拟上下文注入（system/user/tool 三段 + 预算降级）
  tiers                   三层 × 状态 节点分布
  node <id>               节点全参数（时间五元组/权重/置信/敏感级…）
  slot <主体> <谓语>      槽位版本链（双时间窗口）
  why <node_id>           该记忆为何被选中（检索命中历史）
  explain <trace_id>      还原一次注入的决策
  stats                   总量/权重基尼/近期事件
  forget <node_id>        该记忆的遗忘曲线与归档期限（τ_eff 导出）
  govern                  跑一轮离线治理（衰减/剪枝/压缩/流转/审计）
  session [id]            查看/切换会话
  config                  当前生效配置（store/embedding/llm/rerank）

配置：--config 指定（默认找 cwd 的 config.json，或 DEUSMITH_CONFIG），
模型与后端全部由用户配置：embedding.provider=hash|local|openai、
rerank.provider=none|local|openai（模板见 config.example.json）。
无任何配置时内存模式可跑通全流程（数据不持久化）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from cmd import Cmd
from datetime import timedelta

from .algos import (
    archive_after_days, decay_recency, decay_stretch, gini, tau_eff_days,
)
from .embeddings import get_provider
from .llm import get_llm
from .models import Tier, utcnow
from .offline import OfflineGovernor
from .params import Params
from .pipeline import DeusmithMemory
from .rerank import IdentityReranker, get_reranker
from .store import InMemoryStore

COMMANDS = ("write", "search", "inject", "tiers", "node", "slot", "why",
            "explain", "stats", "forget", "govern", "session", "config",
            "web", "help", "quit")

_TIER_ORDER = (Tier.shallow, Tier.middle, Tier.deep)


def _trunc(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[: max(1, n - 1)] + "…"


def _iso(dt) -> str:
    return dt.isoformat(timespec="seconds") if dt else "-"


# ================================================================ 组装

def load_config(path: str | None) -> dict:
    """定位并读取配置：--config > DEUSMITH_CONFIG > ./config.json。缺省返回 {}。"""
    p = path or os.environ.get("DEUSMITH_CONFIG") or "config.json"
    if p and os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _probe_dim(embedder) -> int:
    """取嵌入维度：先看属性（hash/openai 已知），未知则预热一次（local 惰性加载）。"""
    dim = getattr(embedder, "dim", None)
    if dim:
        return int(dim)
    try:
        return len(embedder.embed("维度探测"))
    except Exception:
        return 256


def _pg_vector_dim(store) -> int | None:
    """读 PG 里 content_vector 列的实际维度（表不存在返回 None）。"""
    try:
        with store.cursor() as cur:
            cur.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'memory_node'::regclass "
                "AND attname = 'content_vector'")
            row = cur.fetchone()
        v = row["atttypmod"] if isinstance(row, dict) else row[0]  # dict_row 工厂
        return int(v) if v else None
    except Exception:
        return None


def build_memory_system(cfg: dict, scope: str, force_memory: bool = False):
    """按配置组装 DeusmithMemory。返回 (memory, backend)。"""
    st = (cfg or {}).get("storage", {}) or {}
    backend = st.get("backend", "")
    if force_memory:
        backend = "memory"
    elif not backend and not cfg and not os.environ.get("DEUSMITH_PG_DSN"):
        backend = "memory"      # 零配置首跑：内存模式，零依赖可演示
    elif not backend:
        backend = "postgres"

    if backend == "memory":
        store = InMemoryStore()
    else:
        from .store import PostgresStore
        store = PostgresStore(dsn=st.get("dsn"))

    embedder = get_provider(cfg)
    llm = get_llm(cfg)
    reranker = get_reranker(cfg)
    mem = DeusmithMemory(store=store, embedder=embedder, llm=llm,
                         scope=scope, reranker=reranker)
    if backend != "memory":
        dim = _probe_dim(embedder)             # local 模型此处完成加载
        try:
            store.ensure_schema(dim)
        except Exception as exc:                # 连不上/无扩展：给出可行动提示
            raise RuntimeError(
                f"PostgreSQL 初始化失败（{exc}）。"
                f"请检查 DSN/pgvector，或先用 --memory 体验内存模式。") from exc
        vd = _pg_vector_dim(store)
        if vd is not None and vd != dim:
            raise RuntimeError(
                f"向量维度不匹配：表 memory_node.content_vector 是 vector({vd})，"
                f"当前嵌入模型输出 {dim} 维。换嵌入模型需重建表"
                f"（DROP TABLE memory_node 等后重启），或改回匹配的模型。")
    return mem, backend


# ================================================================ REPL

class DeusmithConsole(Cmd):
    """交互控制台。所有 do_* 方法同时被一次性模式复用（onecmd）。"""

    intro = ("Deusmith-MemorySystem 控制台 —— 输入 help 查看命令；"
             "write/search/inject 走完整管线")
    prompt = "deusmith> "

    def __init__(self, mem: DeusmithMemory, cfg: dict, backend: str = "memory",
                 session_id: str = "console-1"):
        super().__init__()
        self.mem = mem
        self.cfg = cfg
        self.backend = backend
        self.sess = mem.session(session_id)

    # ---- 基础覆写：空行不重复上一条命令
    def emptyline(self):
        return False

    # ================================================== 写入
    def do_write(self, arg: str):
        if not arg.strip():
            print("用法: write <文本>（如：write 请记住：我对花粉过敏。）")
            return
        nodes = self.mem.write(self.sess, arg)
        if not nodes:
            print("（未抽取到记忆条目）")
            return
        for n in nodes:
            print(f"  写入 {n.id[-8:]} [{n.tier.value}/{n.status.value}] "
                  f"{_trunc(n.content, 48)}")

    # ================================================== 检索
    def do_search(self, arg: str):
        args = (arg or "").split()
        topk = 8
        if "-k" in args:
            i = args.index("-k")
            try:
                topk = max(1, int(args[i + 1]))
                del args[i:i + 2]
            except (IndexError, ValueError):
                print("用法: search <查询> [-k N]")
                return
        query = " ".join(args).strip()
        if not query:
            print("用法: search <查询> [-k N]")
            return
        res = self.mem.retrieve(self.sess, query, topk=topk)
        rr = (res.trace.get("rerank") or {})
        if rr.get("used"):
            print(f"  精排: {rr.get('model')} (pool={rr.get('pool')}, "
                  f"w={rr.get('weight')})")
        elif rr.get("error"):
            print(f"  精排降级: {rr['error']}")
        if not res.selected:
            print("  （无命中）")
            return
        print(f"  {'#':<2} {'score':>7} {'rel':>6} {'层':<7} {'状态':<9} 内容")
        for i, (n, s, rel) in enumerate(res.selected, 1):
            print(f"  {i:<2} {s:>7.4f} {rel:>6.3f} {n.tier.value:<7} "
                  f"{n.status.value:<9} {_trunc(n.content, 40)}")
        if res.conflict_flags:
            print(f"  ⚠ 记忆冲突（待确认）: "
                  f"{', '.join(x[-8:] for x in res.conflict_flags)}")

    # ================================================== 注入
    def do_inject(self, arg: str):
        query = (arg or "").strip()
        if not query:
            print("用法: inject <查询>")
            return
        out = self.mem.inject_context(self.sess, query)
        print(f"  [system] {out['system'] or '（空）'}")
        print(f"  [user]   {out['user'] or '（空）'}")
        print(f"  [tool]   {out['tool'] or '（空）'}")
        print(f"  tokens={out['tokens']}  degraded={out['degraded']}  "
              f"conflicts={len(out['conflict_flags'])}  "
              f"trace={out['trace_id']}")

    # ================================================== 层级与统计
    def do_tiers(self, arg: str):
        stats = self.mem.store.tier_stats(self.mem.scope)
        statuses = sorted({s for (_, s) in stats} | {"active"})
        print(f"  {'层':<8}" + "".join(f"{s:<12}" for s in statuses) + "小计")
        for tier in _TIER_ORDER:
            row = [stats.get((tier.value, s), 0) for s in statuses]
            print(f"  {tier.value:<8}" + "".join(f"{c:<12}" for c in row)
                  + f"{sum(row)}")
        total = sum(stats.values())
        print(f"  合计 {total} 个节点")

    def do_stats(self, arg: str):
        scope = self.mem.scope
        c = self.mem.store.counts(scope)
        nodes = self.mem.store.active_nodes(scope)
        ws = [n.weight for n in nodes]
        print(f"  节点 {c['nodes_total']} / 边 {c['edges_total']} / 活跃 {len(nodes)}")
        if ws:
            print(f"  weight: mean={sum(ws)/len(ws):.3f} "
                  f"max={max(ws):.3f} min={min(ws):.3f} "
                  f"gini={gini(ws):.3f}"
                  f"{' ⚠垄断告警' if gini(ws) > self.mem.params.tier.gini_alert else ''}")
        for e in self.mem.store.recent_events(scope, limit=5):
            print(f"  [{_iso(e['ts'])}] {e['event_type']}")

    def do_govern(self, arg: str):
        gov = OfflineGovernor(self.mem.store, self.mem.embedder, self.mem.llm,
                              self.mem.params)
        rep = gov.run(self.mem.scope)
        for k, v in rep.items():
            print(f"  {k}: {v}")
        if rep.get("weight_monopoly_warning"):
            print("  ⚠ 权重基尼超阈值（垄断风险），详见审计日志")

    def do_forget(self, arg: str):
        nid = (arg or "").strip()
        if not nid:
            print("用法: forget <node_id>（显示该记忆的遗忘曲线与归档期限）")
            return
        n = self.mem.store.get_node(nid)
        if n is None:
            print(f"  未找到节点 {nid}")
            return
        import copy
        P = self.mem.params
        now = utcnow()
        print(f"  节点 {n.id[-8:]}  {_trunc(n.content, 40)}")
        print(f"  S={decay_stretch(n, P):.3f}  τ_eff={tau_eff_days(n, P):.2f}天  "
              f"归档期限={archive_after_days(n, P):.1f}天  "
              f"(w={n.weight:.3f} 门槛={P.tier.archive_weight_floor})")
        print(f"  {'Δt(天)':>8} | {'recency':>8}   （距今实际 Δt="
              f"{(now - n.last_access_at).total_seconds()/86400:.1f} 天）")
        probe = copy.copy(n)                  # 浅拷贝：只改 last_access_at
        probe.last_access_at = now
        for d in (0, 1, 7, 14, 30, 90, 180, 365):
            r = decay_recency(probe, now + timedelta(days=d), P)
            print(f"  {d:>8} | {r:>8.4f}")

    # ================================================== 节点与解释
    def do_node(self, arg: str):
        nid = (arg or "").strip()
        if not nid:
            print("用法: node <id>")
            return
        n = self.mem.store.get_node(nid)
        if n is None:
            print(f"  未找到节点 {nid}")
            return
        rows = [
            ("id", n.id), ("type/tier/status",
                           f"{n.type.value}/{n.tier.value}/{n.status.value}"),
            ("content", _trunc(n.content, 60)),
            ("slot", f"{n.subject} | {n.predicate} | {n.object_value} "
                     f"(polarity={n.polarity})"),
            ("created_at", _iso(n.created_at)),
            ("updated_at", _iso(n.updated_at)),
            ("valid_from/to", f"{_iso(n.valid_from)} → {_iso(n.valid_to)}"),
            ("last_access_at", _iso(n.last_access_at)),
            ("weight/confidence", f"{n.weight:.3f} / {n.confidence:.3f}"),
            ("importance/emotion", f"{n.importance:.3f} / {n.emotion:.3f}"),
            ("source/auth/consistency",
             f"{n.source} / {n.source_auth:.2f} / {n.consistency:.2f}"),
            ("activation/cross_session", f"{n.activation_count} / {n.cross_session_count}"),
            ("sensitivity/explicit", f"{n.sensitivity} / {n.explicit_remember}"),
            ("supersedes/contested_by", f"{n.supersedes or '-'} / {n.contested_by or '-'}"),
            ("evidence_refs", ", ".join(n.evidence_refs[:3]) or "-"),
            ("embedding", f"{n.embedding_model} dim={len(n.content_vector)}"),
            ("session/generation", f"{n.session_id} / gen={n.generation}"),
        ]
        for k, v in rows:
            print(f"  {k:<24} {v}")

    def do_slot(self, arg: str):
        parts = (arg or "").split()
        if len(parts) < 2:
            print("用法: slot <主体> <谓语>（如: slot user 住在）")
            return
        subject, predicate = parts[0], parts[1]
        hist = self.mem.slot_history(subject, predicate)
        if not hist:
            print(f"  槽位 ({subject}, {predicate}) 无记录")
            return
        print(f"  槽位 ({subject}, {predicate}) 版本链（双时间）：")
        for h in hist:
            print(f"  {h['node_id'][-8:]} [{h['status']:<10}] "
                  f"{_iso(h['valid_from'])} → {_iso(h['valid_to'])}  "
                  f"{_trunc(h['content'], 36)}")

    def do_why(self, arg: str):
        nid = (arg or "").strip()
        if not nid:
            print("用法: why <node_id>")
            return
        hits = self.mem.why_selected(nid)
        if not hits:
            print(f"  节点 {nid} 暂无检索命中记录")
            return
        for h in hits:
            print(f"  [{h['trace_id']}] score={h['score']} "
                  f"query={_trunc(h['query'] or '', 40)}")

    def do_explain(self, arg: str):
        tid = (arg or "").strip()
        if not tid:
            print("用法: explain <trace_id>")
            return
        exp = self.mem.explain_injection(tid)
        if not exp.get("found"):
            print(f"  trace {tid} 无注入记录")
            return
        print(f"  query={exp.get('query')}  tokens={exp.get('tokens')}  "
              f"degraded={exp.get('degraded')}")
        print(f"  注入记忆: {', '.join((exp.get('selected') or [])[:8])}")
        print(f"  事件链: {' → '.join(exp.get('events', []))}")

    # ================================================== 会话与配置
    def do_session(self, arg: str):
        sid = (arg or "").strip()
        if sid:
            self.sess = self.mem.session(sid)
        s = self.sess
        print(f"  会话 {s.id}  轮次 {len(s.rounds)}  "
              f"closed={'是' if s.closed_at else '否'}")
        if s.rolling_summary:
            print(f"  滚动摘要: {_trunc(s.rolling_summary, 60)}")

    def do_config(self, arg: str):
        emb = self.mem.embedder
        rr = self.mem.reranker
        print(f"  scope:      {self.mem.scope}")
        print(f"  store:      {type(self.mem.store).__name__} ({self.backend})")
        print(f"  embedding:  {getattr(emb, 'model_name', emb.__class__.__name__)}")
        print(f"  llm:        {type(self.mem.llm).__name__}")
        print(f"  rerank:     "
              f"{getattr(rr, 'model_name', type(rr).__name__) if rr else '未启用'}")
        p = self.mem.params
        print(f"  params:     τ_base={p.decay.tau_base_days}天  "
              f"rrf_k={p.retrieval.rrf_k}  rerank_w={p.retrieval.rerank_weight}  "
              f"mmr_λ={p.score.mmr_lambda}")
        print(f"  敏感上限:   L{self.mem.allowed_sensitivity}")

    # ================================================== Web 控制台
    def do_web(self, arg: str):
        """web [端口] —— 启动可视化控制台（后台线程，REPL 继续可用）。"""
        if getattr(self, "_web_srv", None) is not None:
            print("  可视化控制台已在运行")
            return
        from .webconsole import WebApp, make_server
        port = 8765
        if arg.strip():
            try:
                port = int(arg.strip())
            except ValueError:
                print("用法: web [端口]")
                return
        app = WebApp.from_memory(self.mem, backend_label=self.backend)
        try:
            srv = make_server(app, "127.0.0.1", port)
        except OSError as exc:
            print(f"  端口 {port} 被占用（{exc}）")
            return
        self._web_srv = srv
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{srv.server_address[1]}/"
        print(f"  可视化控制台已启动: {url}（Ctrl+C 退出 REPL 时一并关闭）")
        try:
            webbrowser.open(url)
        except Exception:
            pass

    # ================================================== 退出
    def do_quit(self, arg: str):
        return True

    def do_EOF(self, arg: str):
        print()
        return True


# ================================================================ 自检

class _FakeReranker:
    """假精排器：命中关键词的文档打 1.0，其余 0.05（验证管线混合逻辑）。"""

    model_name = "fake-reranker"

    def __init__(self, keyword: str):
        self.keyword = keyword
        self.calls = 0

    def rerank(self, query, documents):
        self.calls += 1
        return [1.0 if self.keyword in d else 0.05 for d in documents]


class _RaisingReranker:
    model_name = "boom-reranker"

    def rerank(self, query, documents):
        raise RuntimeError("rerank endpoint down")


def _smoke_write(mem: DeusmithMemory, sid: str):
    s = mem.session(sid)
    mem.write(s, "请记住：我对花粉过敏。")
    mem.write(s, "我喜欢摄影。")
    mem.write(s, "我现在住在上海。", source_auth=0.4)
    mem.write(s, "我现在住在北京。", source_auth=0.95)
    return s


def smoke() -> int:
    """控制台 + 精排链路自检：基线不回归、rerank 生效/降级、命令面全通。"""
    from .embeddings import HashEmbedding
    from .llm import RuleBasedLLM

    checks: list[tuple[str, bool, str]] = []

    def ck(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # ---- 参数默认
    P = Params()
    ck("params: rerank_weight 默认 0.6",
       abs(P.retrieval.rerank_weight - 0.6) < 1e-9)
    ck("params: rerank_pool 默认 24", P.retrieval.rerank_pool == 24)

    # ---- 基线（无 reranker）：行为不回归
    mem = DeusmithMemory(store=InMemoryStore(), embedder=HashEmbedding(dim=64),
                         llm=RuleBasedLLM(), scope="smoke/u1")
    s = _smoke_write(mem, "smoke-1")
    res = mem.retrieve(s, "用户的居住地", reinforce_hits=False)
    ck("基线检索有命中", len(res.selected) >= 1, f"n={len(res.selected)}")
    ck("基线 trace 无 rerank 键", "rerank" not in res.trace)

    # ---- rerank 生效：假精排器把「摄影」顶到首位
    mem2 = DeusmithMemory(store=InMemoryStore(), embedder=HashEmbedding(dim=64),
                          llm=RuleBasedLLM(), scope="smoke/u2",
                          reranker=_FakeReranker("摄影"))
    s2 = _smoke_write(mem2, "smoke-2")
    res2 = mem2.retrieve(s2, "用户的居住地", reinforce_hits=False)
    rr = res2.trace.get("rerank") or {}
    ck("rerank: trace 留痕 used=True", rr.get("used") is True)
    top = res2.selected[0][0].content if res2.selected else ""
    ck("rerank: 命中文档被顶到首位", "摄影" in top, _trunc(top, 30))
    scores = list(rr.get("rerank_scores", {}).values())
    ck("rerank: 分数域 [0,1]", bool(scores) and all(0.0 <= v <= 1.0 for v in scores))
    ck("rerank: 混合后 rel∈[0,1]",
       all(0.0 <= rel <= 1.0 for _, _, rel in res2.selected))

    # ---- rerank 失败 → 降级不炸检索
    mem3 = DeusmithMemory(store=InMemoryStore(), embedder=HashEmbedding(dim=64),
                          llm=RuleBasedLLM(), scope="smoke/u3",
                          reranker=_RaisingReranker())
    s3 = _smoke_write(mem3, "smoke-3")
    res3 = mem3.retrieve(s3, "用户的居住地", reinforce_hits=False)
    ck("rerank 失败: 检索不抛错", len(res3.selected) >= 1)
    rr3 = res3.trace.get("rerank") or {}
    ck("rerank 失败: trace.used=False", rr3.get("used") is False)
    ck("rerank 失败: rerank_fallback 事件留痕",
       any(e["event_type"] == "rerank_fallback"
           for e in mem3.store.recent_events("smoke/u3", limit=20)))

    # ---- IdentityReranker → 不参与精排
    mem4 = DeusmithMemory(store=InMemoryStore(), embedder=HashEmbedding(dim=64),
                          llm=RuleBasedLLM(), scope="smoke/u4",
                          reranker=IdentityReranker())
    s4 = _smoke_write(mem4, "smoke-4")
    res4 = mem4.retrieve(s4, "居住地", reinforce_hits=False)
    ck("identity reranker: trace 无 rerank 键（rerank→None）",
       "rerank" not in res4.trace)

    # ---- 配置工厂
    ck("get_reranker: 无配置 → None", get_reranker({}) is None)
    ck("get_reranker: provider=none → None",
       get_reranker({"rerank": {"provider": "none"}}) is None)
    try:
        get_reranker({"rerank": {"provider": "???"}})
        ck("get_reranker: 未知 provider 抛错", False)
    except ValueError:
        ck("get_reranker: 未知 provider 抛错", True)

    # ---- 控制台命令面（内存后端全链路）
    mem5, backend = build_memory_system({}, "smoke/u5", force_memory=True)
    ck("build: 零配置内存后端", backend == "memory")
    con = DeusmithConsole(mem5, {}, backend="memory", session_id="smoke-5")
    con.onecmd('write 请记住：我对花粉过敏。')
    con.onecmd("search 花粉 -k 3")
    con.onecmd("inject 用户的过敏情况")
    con.onecmd("tiers")
    con.onecmd("stats")
    con.onecmd("session")
    con.onecmd("config")
    con.onecmd("slot user 过敏")
    con.onecmd("explain tr_none")          # 找不到 trace → 优雅提示
    con.onecmd("node m_missing")           # 找不到节点 → 优雅提示
    con.onecmd("govern")
    nodes = mem5.store.active_nodes("smoke/u5")
    ck("console: write 后有活跃节点", len(nodes) >= 1, f"n={len(nodes)}")
    if nodes:
        con.onecmd(f"node {nodes[0].id}")
        con.onecmd(f"why {nodes[0].id}")
        con.onecmd(f"forget {nodes[0].id}")
    ck("console: tier_stats 可用",
       isinstance(mem5.store.tier_stats("smoke/u5"), dict))

    fails = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))
    print(f"\n== console smoke：{len(checks) - len(fails)}/{len(checks)} 通过 ==")
    return 1 if fails else 0


# ================================================================ 入口

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="deusmith", description="Deusmith-MemorySystem 控制台（CLI/REPL）")
    ap.add_argument("--config", help="配置文件路径（默认 ./config.json 或 $DEUSMITH_CONFIG）")
    ap.add_argument("--scope", default="console/default", help="租户/用户（默认 console/default）")
    ap.add_argument("--session", default="console-1", help="会话 ID")
    ap.add_argument("--memory", action="store_true", help="强制内存后端（不落 PG）")
    ap.add_argument("--smoke", action="store_true", help="自检模式（CI/回归用）")
    ap.add_argument("--web", action="store_true",
                    help="启动可视化 Web 控制台（代替 REPL）")
    ap.add_argument("--port", type=int, default=8765, help="--web 端口（默认 8765）")
    ap.add_argument("cmd", nargs="*", help="一次性命令及参数（省略则进 REPL）")
    args = ap.parse_args(argv)

    if args.smoke:
        return smoke()

    if args.web:
        from .webconsole import main as web_main
        wargv = ["--scope", args.scope, "--port", str(args.port)]
        if args.config:
            wargv += ["--config", args.config]
        if args.memory:
            wargv.append("--memory")
        return web_main(wargv)

    try:
        cfg = load_config(args.config)
        mem, backend = build_memory_system(cfg, args.scope,
                                           force_memory=args.memory)
    except Exception as exc:
        print(f"[启动失败] {exc}", file=sys.stderr)
        return 2

    if not args.cmd:
        if backend == "memory":
            print("[提示] 内存模式：数据不持久化。配置 config.json 可接 PG/真实模型。")
        try:
            DeusmithConsole(mem, cfg, backend, args.session).cmdloop()
        except (KeyboardInterrupt, EOFError):
            print()
        finally:
            mem.store.close()
        return 0

    name, rest = args.cmd[0], " ".join(args.cmd[1:])
    if name not in COMMANDS:
        print(f"未知命令: {name}\n可用: {' '.join(COMMANDS)}", file=sys.stderr)
        mem.store.close()
        return 2
    if backend == "memory":
        print("[提示] 内存模式下一次性命令的数据随进程退出丢失；"
              "跨调用请用 PG 后端（config.json）或交互 REPL。", file=sys.stderr)
    try:
        DeusmithConsole(mem, cfg, backend, args.session).onecmd(
            f"{name} {rest}".strip())
    except Exception as exc:
        print(f"[命令失败] {name}: {exc}", file=sys.stderr)
        mem.store.close()
        return 1
    mem.store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
