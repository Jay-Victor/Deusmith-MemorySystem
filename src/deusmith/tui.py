# -*- coding: utf-8 -*-
"""Deusmith TUI —— 终端图形界面（Textual，可选依赖）。

安装：pip install deusmith-memory[tui]（即 textual>=1.0）
用法：
  deusmith-tui                          # 终端 TUI（无 config.json 时内存模式）
  deusmith --tui                        # console 委托启动
  python -m deusmith.tui --smoke        # 无头自检（CI 用）

五个标签页（与 Web 控制台同构）：
  ① 写入/检索：多行写入（来源权威可调）、检索（score/rel/精排明细）、注入预览、结束会话
  ② 记忆浏览器：tier/status/文本过滤 + 行选详情（全参数 + explain 因子条）
  ③ 遗忘曲线：exp(−Δt/τ_eff) 三条曲线（本节点/冷/热参考）+ 归档期限
  ④ 事件流：近期事件日志（类型着色）
  ⑤ 离线治理：一键执行 + 报告（线程 worker，不冻结 UI）

键位：Ctrl+Q 退出；标签页点击或 Tab/方向键切换；表格 ↑↓ + Enter 选行。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys

from rich.markup import escape
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, Label, RichLog, Select,
    Static, TabbedContent, TabPane, TextArea,
)

from .algos import (
    archive_after_days, decay_recency, decay_stretch, gini, tau_eff_days,
)
from .console import build_memory_system, load_config
from .models import MemoryNode, NodeType, Tier, utcnow
from .offline import OfflineGovernor
from .params import Params
from .pipeline import DeusmithMemory

SESSION_ID = "tui-console"
_LEVELS = "▁▂▃▄▅▆▇█"


# ---------------------------------------------------------------- 工具

def _ref_node(importance: float, confidence: float, activation: int,
              emotion: float = 0.0) -> MemoryNode:
    return MemoryNode(type=NodeType.fact, content="参考", scope="tui/ref",
                      tier=Tier.middle, embedding_model="ref",
                      importance=importance, emotion=emotion,
                      confidence=confidence, activation_count=activation)


def _ago(dt) -> str:
    if not dt:
        return "—"
    s = (utcnow() - dt).total_seconds()
    if s < 0:
        return dt.strftime("%m-%d %H:%M")
    if s < 90:
        return f"{s:.0f}秒前"
    if s < 7200:
        return f"{s / 60:.0f}分前"
    if s < 172800:
        return f"{s / 3600:.1f}时前"
    return f"{s / 86400:.1f}天前"


def _fmt_payload(p) -> str:
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except (ValueError, TypeError):
            return p[:100]
    try:
        s = json.dumps(p, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        s = str(p)
    return s[:110]


def _bar(v: float) -> str:
    n = int(round(max(0.0, min(1.0, v)) * 10))
    return "█" * n + "░" * (10 - n)


def _sel_value(sel) -> object | None:
    """Select 当前值归一：空哨兵（BLANK/NULL/False/""）→ None。

    textual 各版本的空值哨兵不同（BLANK=""、NULL=对象），且 NULL 为真值，
    `or None` 拦不住——不归一直接传给 Tier()/get_node() 会抛错。"""
    v = sel.value
    nulls = {None, False, ""}
    for name in ("BLANK", "NULL"):
        sentinel = getattr(Select, name, None)
        if sentinel is not None:
            nulls.add(sentinel)
    return None if v in nulls or v is None else v


def _spark(tau_days: float, width: int = 56, days: float = 365.0) -> str:
    """指数衰减曲线的 Unicode 火花线：每列一天数桶，8 级块字符。"""
    out = []
    for i in range(width):
        d = days * i / (width - 1)
        r = math.exp(-d / tau_days) if tau_days > 0 else 0.0
        out.append(_LEVELS[min(7, int(r * 8))])
    return "".join(out)


def _axis(width: int = 56, days: float = 365.0) -> tuple[str, str]:
    """刻度线 + 标签行（0/30/90/180/270/365 天）。"""
    ticks = (0, 30, 90, 180, 270, 365)
    xs = [round(d / days * (width - 1)) for d in ticks]
    tick_line = ["─"] * width
    for x in xs:
        tick_line[x] = "┬"
    label_line = [" "] * (width + 6)
    for x, d in zip(xs, ticks):
        lab = f"{d}天" if d == 365 else str(d)
        for i, ch in enumerate(lab):
            if 0 <= x + i < len(label_line):
                label_line[x + i] = ch
    return "└" + "".join(tick_line), "".join(label_line).rstrip()


def _forget_chart(node: MemoryNode, P: Params) -> str:
    """本节点 + 冷/热参考三条遗忘曲线 + 归档期限标注。"""
    now = utcnow()
    S = decay_stretch(node, P)
    tau = tau_eff_days(node, P)
    arch = archive_after_days(node, P)
    cold = _ref_node(0.3, 0.4, 0)
    hot = _ref_node(0.8, 0.9, 3, emotion=0.5)
    w = 56
    pad = 20
    lines = [
        f"{'本节点':　<4} S={S:.2f} τ={tau:5.1f}天 1.0│{_spark(tau, w)}",
        f"{'冷参考':　<4} S={decay_stretch(cold, P):.2f}            1.0│"
        f"{_spark(tau_eff_days(cold, P), w)}",
        f"{'热参考':　<4} S={decay_stretch(hot, P):.2f}            1.0│"
        f"{_spark(tau_eff_days(hot, P), w)}",
    ]
    tick, labels = _axis(w)
    lines.append(" " * pad + " 0" + tick)
    lines.append(" " * (pad + 2) + labels)
    dt_days = (now - node.last_access_at).total_seconds() / 86400.0
    cur = decay_recency(node, now, P)
    lines.append(
        f"θ={P.tier.archive_score_floor} 归档阈值穿越于第 [b]{arch:.1f}[/b] 天"
        f"（归档期限）；当前 Δt={dt_days:.1f} 天，recency={cur:.3f}")
    lines.append(
        "[dim]S = 1 + 0.8·importance + 0.5·emotion + 0.7·confidence "
        "+ 0.3·ln(1+activation)；τ_eff = τ_base·S；归档期限 = −τ_eff·ln(θ)[/dim]")
    return "\n".join(lines)


# ---------------------------------------------------------------- 应用

class DeusmithTUI(App):
    """Deusmith 终端控制台（Textual）。"""

    TITLE = "Deusmith · 记忆系统"
    SUB_TITLE = "三层记忆 · 向量检索 · 遗忘曲线"
    CSS = """
    #status { height: 3; padding: 0 1; }
    #st-chips { width: 1fr; content-align: left middle; }
    #scope-in { width: 26; }
    Label { margin: 1 0 0 0; color: $text-muted; }
    TextArea#write-text { height: 4; }
    #write-auth { width: 10; }
    #search-k { width: 6; }
    #search-q { width: 1fr; }
    DataTable { height: auto; max-height: 16; }
    #browse-table { height: 1fr; }
    #detail-wrap { height: 20; border: round $primary; padding: 0 1; }
    #events-log { height: 1fr; }
    #govern-out { padding: 1 0; }
    Horizontal { height: auto; align-vertical: middle; }
    Button { margin-right: 1; }
    Select { margin-right: 1; }
    #f-q { width: 1fr; }
    """
    BINDINGS = [("ctrl+q", "quit", "退出")]

    def __init__(self, mem: DeusmithMemory, backend: str = "memory"):
        super().__init__()
        self.store = mem.store
        self.embedder = mem.embedder
        self.llm = mem.llm
        self.reranker = mem.reranker
        self.params = mem.params
        self.backend = backend
        self.scope = mem.scope
        self._mems: dict[str, DeusmithMemory] = {mem.scope: mem}

    # ------------------------------------------------------------ 门面
    def memory_for(self, scope: str) -> DeusmithMemory:
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

    # ------------------------------------------------------------ 布局
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="status"):
            yield Static("", id="st-chips")
            yield Input(placeholder="scope 切换", id="scope-in")
            yield Button("切换", id="btn-scope")
        with TabbedContent(initial="tab-write"):
            with TabPane("① 写入/检索", id="tab-write"):
                with VerticalScroll():
                    yield Label("写入记忆（多行文本 · 来源权威 0–1）")
                    yield TextArea(id="write-text")
                    with Horizontal():
                        yield Input(value="0.8", id="write-auth")
                        yield Button("写入", id="btn-write", variant="primary")
                        yield Button("结束会话", id="btn-session-end")
                    yield Static("", id="write-out")
                    yield Label("检索 / 注入预览")
                    with Horizontal():
                        yield Input(placeholder="查询，如：用户的居住地和爱好",
                                    id="search-q")
                        yield Input(value="8", id="search-k")
                        yield Button("检索", id="btn-search", variant="primary")
                        yield Button("注入预览", id="btn-inject")
                    yield Static("", id="search-info")
                    yield DataTable(id="search-table", cursor_type="row")
                    yield Static("", id="inject-out")
            with TabPane("② 记忆浏览器", id="tab-browse"):
                with Horizontal():
                    yield Select(
                        [("全部层", ""), ("浅层", "shallow"),
                         ("中层", "middle"), ("深层", "deep")],
                        id="f-tier", allow_blank=True)
                    yield Select(
                        [("全部状态", ""), ("active", "active"),
                         ("archived", "archived"), ("contested", "contested"),
                         ("deprecated", "deprecated")],
                        id="f-status", allow_blank=True)
                    yield Input(placeholder="内容包含…", id="f-q")
                    yield Button("筛选", id="btn-filter", variant="primary")
                yield DataTable(id="browse-table", cursor_type="row")
                with VerticalScroll(id="detail-wrap"):
                    yield Static("（↑↓ 选择行，Enter 查看节点详情）",
                                 id="node-detail")
            with TabPane("③ 遗忘曲线", id="tab-forget"):
                with VerticalScroll():
                    with Horizontal():
                        yield Select([], id="forget-node", allow_blank=True)
                        yield Button("绘制", id="btn-forget", variant="primary")
                    yield Static("", id="forget-stats")
                    yield Static("", id="forget-chart")
            with TabPane("④ 事件流", id="tab-events"):
                with Horizontal():
                    yield Button("刷新", id="btn-events-refresh")
                yield RichLog(id="events-log", markup=True, wrap=True,
                              max_lines=500)
            with TabPane("⑤ 离线治理", id="tab-govern"):
                with VerticalScroll():
                    yield Button("执行一轮治理", id="btn-govern",
                                 variant="primary")
                    yield Static("边衰减/剪枝 → 环路消解 → 压缩合并 → "
                                 "权重慢衰减 → 层级流转 → 一致性审计",
                                 id="govern-hint")
                    yield Static("", id="govern-out")
        yield Footer()

    # ------------------------------------------------------------ 挂载
    def on_mount(self) -> None:
        st = self.query_one("#search-table", DataTable)
        st.add_columns("层", "状态", "score", "rel", "内容")
        bt = self.query_one("#browse-table", DataTable)
        bt.add_columns("ID", "内容", "层", "状态", "weight", "激活",
                       "最近访问")
        self.refresh_state()
        self.refresh_browse()
        self.populate_forget()

    @on(TabbedContent.TabActivated)
    def _tab_activated(self, event: TabbedContent.TabActivated) -> None:
        pid = event.pane.id
        if pid == "tab-events":
            self.refresh_events()
        elif pid == "tab-forget":
            self.populate_forget()
            self.draw_forget()
        elif pid == "tab-browse":
            self.refresh_browse()

    # ------------------------------------------------------------ 状态条
    def refresh_state(self) -> None:
        nodes = self.store.active_nodes(self.scope)
        ws = [n.weight for n in nodes]
        c = self.store.counts(self.scope)
        g = gini(ws) if ws else 0.0
        rr = getattr(self.reranker, "model_name", None) if self.reranker else None
        chips = (
            f"[b cyan]{escape(self.scope)}[/] · "
            f"后端 [b]{escape(self.backend)}[/] · "
            f"嵌入 [b]{escape(getattr(self.embedder, 'model_name', '?'))}[/] · "
            f"精排 [b]{escape(rr or '未启用')}[/] · "
            f"节点 [b]{c['nodes_total']}[/] 边 [b]{c['edges_total']}[/] · "
            f"基尼 [b]{g:.3f}[/]"
            + (" [b yellow]⚠垄断[/]" if g > self.params.tier.gini_alert else ""))
        self.query_one("#st-chips", Static).update(chips)
        scope_in = self.query_one("#scope-in", Input)
        if not scope_in.has_focus:
            scope_in.value = self.scope

    # ------------------------------------------------------------ 写入
    @on(Button.Pressed, "#btn-write")
    def do_write(self) -> None:
        text = self.query_one("#write-text", TextArea).text.strip()
        if not text:
            self.notify("请输入要写入的内容", severity="warning")
            return
        try:
            auth = min(1.0, max(0.0,
                        float(self.query_one("#write-auth", Input).value or 0.8)))
        except ValueError:
            auth = 0.8
        try:
            nodes = self.mem.write(self.mem.session(SESSION_ID), text,
                                   source_auth=auth)
        except Exception as exc:
            self.notify(f"写入失败: {exc}", severity="error")
            return
        if not nodes:
            self.query_one("#write-out", Static).update("[dim]（未抽取到记忆条目）[/]")
        else:
            lines = [f"写入 [b]{len(nodes)}[/b] 条："]
            for n in nodes:
                lines.append(f"  [green]{n.tier.value}[/]/{n.status.value} "
                             f"{escape(n.content[:44])}")
            self.query_one("#write-out", Static).update("\n".join(lines))
        self.query_one("#write-text", TextArea).text = ""
        self.notify(f"写入 {len(nodes)} 条记忆")
        self.refresh_state()
        self.refresh_browse()
        self.populate_forget()

    @on(Button.Pressed, "#btn-session-end")
    def do_session_end(self) -> None:
        self.mem.end_session(self.mem.session(SESSION_ID))
        self.notify("会话已结束：摘要 + 晋升评估 + 持久化")

    # ------------------------------------------------------------ 检索
    @on(Button.Pressed, "#btn-search")
    def do_search(self) -> None:
        q = self.query_one("#search-q", Input).value.strip()
        if not q:
            self.notify("请输入查询", severity="warning")
            return
        try:
            topk = max(1, min(30, int(self.query_one("#search-k", Input).value or 8)))
        except ValueError:
            topk = 8
        try:
            res = self.mem.retrieve(self.mem.session(SESSION_ID), q, topk=topk)
        except Exception as exc:
            self.notify(f"检索失败: {exc}", severity="error")
            return
        rr = res.trace.get("rerank") or {}
        if rr.get("used"):
            info = (f"精排: {rr.get('model')}（pool={rr.get('pool')}, "
                    f"w={rr.get('weight')}）")
        elif rr.get("error"):
            info = f"[yellow]精排降级: {escape(str(rr['error']))}[/]"
        else:
            info = "RRF 多路融合（精排未启用）"
        info += f" · trace {res.trace.get('trace_id')}"
        self.query_one("#search-info", Static).update(info)
        table = self.query_one("#search-table", DataTable)
        table.clear()
        for n, sc, rel in res.selected:
            table.add_row(n.tier.value, n.status.value, f"{sc:.4f}",
                          f"{rel:.3f}", escape(n.content[:46]), key=n.id)
        if res.conflict_flags:
            self.notify(f"⚠ 记忆冲突待确认: "
                        f"{', '.join(x[-8:] for x in res.conflict_flags)}",
                        severity="warning")

    @on(DataTable.RowSelected, "#search-table")
    def _search_row(self, event: DataTable.RowSelected) -> None:
        if event.row_key is not None:
            self.query_one(TabbedContent).active = "tab-browse"
            self._show_detail(str(event.row_key.value))

    @on(Button.Pressed, "#btn-inject")
    def do_inject(self) -> None:
        q = self.query_one("#search-q", Input).value.strip()
        if not q:
            self.notify("请输入查询", severity="warning")
            return
        try:
            out = self.mem.inject_context(self.mem.session(SESSION_ID), q)
        except Exception as exc:
            self.notify(f"注入失败: {exc}", severity="error")
            return
        text = (
            f"[b]tokens[/] {out['tokens']}   [b]降级[/] "
            f"{'[yellow]是[/]' if out['degraded'] else '否'}   "
            f"[b]冲突[/] {len(out['conflict_flags'])}\n"
            f"[bold cyan]── system：深层身份 / 策略 ──[/]\n"
            f"{escape(out['system'] or '（空）')}\n"
            f"[bold cyan]── user：相关记忆 + 会话投影 ──[/]\n"
            f"{escape(out['user'] or '（空）')}\n"
            f"[bold cyan]── tool：证据指针 ──[/]\n"
            f"{escape(out['tool'] or '（空）')}")
        self.query_one("#inject-out", Static).update(text)

    # ------------------------------------------------------------ 浏览器
    def refresh_browse(self) -> None:
        tier = _sel_value(self.query_one("#f-tier", Select))
        status = _sel_value(self.query_one("#f-status", Select))
        q = self.query_one("#f-q", Input).value.strip().lower()
        try:
            nodes = self.store.list_nodes(self.scope, tier=tier, status=status,
                                          limit=500 if q else 200)
        except Exception as exc:
            self.notify(f"列表查询失败: {exc}", severity="error")
            nodes = []
        if q:
            nodes = [n for n in nodes if q in (n.content or "").lower()][:200]
        table = self.query_one("#browse-table", DataTable)
        table.clear()
        for n in nodes:
            table.add_row(n.id[-8:], escape(n.content[:44]), n.tier.value,
                          n.status.value, f"{n.weight:.2f}",
                          str(n.activation_count), _ago(n.last_access_at),
                          key=n.id)

    @on(Button.Pressed, "#btn-filter")
    def do_filter(self) -> None:
        self.refresh_browse()

    @on(Select.Changed, "#f-tier")
    @on(Select.Changed, "#f-status")
    def _filter_changed(self, event: Select.Changed) -> None:
        self.refresh_browse()

    @on(DataTable.RowSelected, "#browse-table")
    def _browse_row(self, event: DataTable.RowSelected) -> None:
        if event.row_key is not None:
            self._show_detail(str(event.row_key.value))

    def _show_detail(self, node_id: str) -> None:
        n = self.store.get_node(node_id)
        if n is None:
            self.query_one("#node-detail", Static).update(
                f"[red]节点不存在: {escape(node_id)}[/]")
            return
        try:
            ex = self.mem.explain_score(node_id)
        except Exception:
            ex = {}
        f = ex.get("factors") or {}
        rows = [
            ("ID / 类型", f"{n.id} · {n.type.value}"),
            ("层 / 状态", f"{n.tier.value} / {n.status.value}"),
            ("内容", n.content),
            ("槽位", f"{n.subject or '—'} · {n.predicate or '—'} · "
                     f"{n.object_value or '—'}（polarity "
                     f"{n.polarity:+d}）" if n.subject else "—"),
            ("创建 / 更新", f"{_ago(n.created_at)} / {_ago(n.updated_at)}"),
            ("生效窗口", f"{n.valid_from or '—'} → "
                         f"{n.valid_to or '开放'}"),
            ("最近访问", _ago(n.last_access_at)),
            ("weight / confidence", f"{n.weight:.3f} / {n.confidence:.3f}"),
            ("importance / emotion", f"{n.importance:.2f} / {n.emotion:.2f}"),
            ("来源", f"{n.source}（auth {n.source_auth:.2f}，"
                     f"consistency {n.consistency:.2f}）"),
            ("激活 / 跨会话", f"{n.activation_count} / {n.cross_session_count}"),
            ("敏感 / 显式", f"L{n.sensitivity} / "
                            f"{'是' if n.explicit_remember else '否'}"),
            ("替代 / 争议", f"{(n.supersedes or '—')[-8:]} / "
                            f"{(n.contested_by or '—')[-8:]}"),
            ("证据", ", ".join(n.evidence_refs[:3]) or "—"),
            ("嵌入 / 世代", f"{n.embedding_model} {len(n.content_vector)}维 · "
                            f"gen{n.generation}"),
        ]
        text = "\n".join(f"[dim]{k:<20}[/] {escape(str(v))}" for k, v in rows)
        if f:
            text += "\n\n[dim]检索得分因子（explain_score）[/]"
            for k in ("weight", "recency", "confidence", "importance"):
                if k in f:
                    text += (f"\n[dim]{k:<20}[/] {_bar(float(f[k]))} "
                             f"{float(f[k]):.3f}")
            if ex.get("score_at_full_rel") is not None:
                text += (f"\n[dim]{'rel=1 上限分':<20}[/] "
                         f"[b]{ex['score_at_full_rel']}[/]")
        self.query_one("#node-detail", Static).update(text)

    # ------------------------------------------------------------ 遗忘曲线
    def populate_forget(self) -> None:
        nodes = self.store.list_nodes(self.scope, limit=100)
        sel = self.query_one("#forget-node", Select)
        sel.set_options([(f"{n.id[-8:]} · {n.content[:32]}", n.id)
                         for n in nodes])
        cur = _sel_value(sel)
        if nodes and (cur is None or str(cur) not in {n.id for n in nodes}):
            sel.value = nodes[0].id

    @on(Select.Changed, "#forget-node")
    def _forget_node_changed(self, event: Select.Changed) -> None:
        self.draw_forget()

    @on(Button.Pressed, "#btn-forget")
    def do_forget(self) -> None:
        self.draw_forget()

    def draw_forget(self) -> None:
        nid = _sel_value(self.query_one("#forget-node", Select))
        if not nid:
            nodes = self.store.list_nodes(self.scope, limit=1)
            if not nodes:
                self.query_one("#forget-stats", Static).update(
                    "[dim]（无节点——先写入几条记忆）[/]")
                self.query_one("#forget-chart", Static).update("")
                return
            nid = nodes[0].id
        n = self.store.get_node(str(nid))
        if n is None:
            return
        P = self.params
        stats = (
            f"[b]{escape(n.content[:40])}[/]\n"
            f"S = [b]{decay_stretch(n, P):.3f}[/]   "
            f"τ_eff = [b]{tau_eff_days(n, P):.2f}[/] 天   "
            f"归档期限 = [bold yellow]{archive_after_days(n, P):.1f}[/] 天   "
            f"（weight {n.weight:.3f} / 门槛 {P.tier.archive_weight_floor}）")
        self.query_one("#forget-stats", Static).update(stats)
        self.query_one("#forget-chart", Static).update(
            _forget_chart(n, P))

    # ------------------------------------------------------------ 事件流
    def refresh_events(self) -> None:
        log = self.query_one("#events-log", RichLog)
        log.clear()
        colors = {"write": "cyan", "retrieval": "magenta", "inject": "green",
                  "conflict": "red", "duplicate_write": "blue",
                  "offline_run": "yellow", "rerank_fallback": "yellow",
                  "llm_fallback": "yellow"}
        for e in self.store.recent_events(self.scope, limit=60):
            ts = e.get("ts")
            ts_s = ts.strftime("%m-%d %H:%M:%S") if ts else "?"
            color = colors.get(e.get("event_type"), "white")
            log.write(f"[dim]{ts_s}[/dim] [{color}]{e.get('event_type')}[/] "
                      f"{escape(_fmt_payload(e.get('payload')))}")

    @on(Button.Pressed, "#btn-events-refresh")
    def do_events_refresh(self) -> None:
        self.refresh_events()

    # ------------------------------------------------------------ 治理
    @on(Button.Pressed, "#btn-govern")
    def do_govern(self) -> None:
        self.query_one("#govern-out", Static).update("[yellow]治理中…[/]")
        self.run_worker(self._govern_job, thread=True, exclusive=True)

    def _govern_job(self) -> None:
        gov = OfflineGovernor(self.store, self.embedder, self.llm,
                              self.params)
        try:
            rep = gov.run(self.scope)
        except Exception as exc:
            self.call_from_thread(self._govern_failed, str(exc))
            return
        self.call_from_thread(self._govern_done, rep)

    def _govern_failed(self, msg: str) -> None:
        self.query_one("#govern-out", Static).update(
            f"[red]治理失败: {escape(msg)}[/]")
        self.notify("治理失败", severity="error")

    def _govern_done(self, rep: dict) -> None:
        lines = []
        warn_keys = ("weight_monopoly_warning",)
        for k, v in rep.items():
            mark = "[yellow]" if k in warn_keys and v else ""
            end = "[/]" if mark else ""
            lines.append(f"[dim]{k:<26}[/] {mark}{v}{end}")
        self.query_one("#govern-out", Static).update("\n".join(lines))
        self.notify(f"治理完成：归档 {rep.get('archived', 0)} · "
                    f"摘要 {rep.get('summaries_created', 0)} · "
                    f"晋升深层 {rep.get('promoted_deep', 0)}")
        self.refresh_state()
        self.refresh_browse()
        self.populate_forget()

    # ------------------------------------------------------------ scope
    @on(Button.Pressed, "#btn-scope")
    def do_scope(self) -> None:
        s = self.query_one("#scope-in", Input).value.strip()
        if not s:
            self.notify("scope 不能为空", severity="warning")
            return
        self.scope = s
        self.memory_for(s)
        self.notify(f"已切换 scope: {s}")
        self.refresh_state()
        self.refresh_browse()
        self.populate_forget()


# ================================================================ 自检

def smoke() -> int:
    """TUI 无头自检（Textual pilot）：五标签页 + 全按钮 + scope 切换。"""
    from .embeddings import HashEmbedding
    from .llm import RuleBasedLLM
    from .store import InMemoryStore

    checks: list[tuple[str, bool, str]] = []

    def ck(name: str, ok: bool, detail: str = ""):
        checks.append((name, bool(ok), detail))

    async def goto_tab(app, pilot, pane: str, wait_sel: str | None = None) -> None:
        """切换标签页并等布局就绪（目标控件非零区域）。

        两个 textual 8.x 测试垫坑（实测）：
        1. 点击会把焦点留给被点控件；焦点在 pane 内控件上时，
           TabbedContent.active 的程序化赋值不生效——先 set_focus(None)。
        2. 程序化切换后 pane 内容可能尚未布局（region=0×0），pilot.click
           的点击点会落到 (0,0)——正中 Header，而 Header 点击语义是打开
           CommandPalette（遮挡后续点击）。轮询等待布局完成。"""
        app.set_focus(None)
        app.query_one(TabbedContent).active = pane
        for _ in range(20):
            await pilot.pause()
            if wait_sel is None:
                return
            try:
                w = app.query_one(wait_sel)
            except Exception:
                continue
            r = getattr(w, "region", None)
            if r is not None and r.width > 0 and r.height > 0:
                return

    async def run() -> int:
        mem = DeusmithMemory(store=InMemoryStore(),
                             embedder=HashEmbedding(dim=64),
                             llm=RuleBasedLLM(), scope="tuismoke/u1")
        app = DeusmithTUI(mem, "memory")
        async with app.run_test(size=(120, 46)) as pilot:
            await pilot.pause()

            # ---- 状态条
            chips = str(app.query_one("#st-chips", Static).content)
            ck("状态条含 scope/后端/嵌入",
               "tuismoke/u1" in chips and "memory" in chips
               and "hash" in chips, chips[:60])

            # ---- 写入（首击走 pilot.click 验证按钮接线；后续直接调 handler
            #      ——textual 8.x pilot 对同一按钮的重复点击不稳定，测试垫问题）
            app.query_one("#write-text", TextArea).text = "请记住：我对花粉过敏。"
            clicked = await pilot.click("#btn-write")
            await pilot.pause()
            wout = str(app.query_one("#write-out", Static).content)
            ck("写入：点击触发 + 结果回显",
               clicked and "写入" in wout and "花粉" in wout, wout[:40])
            for txt in ("我喜欢摄影，周末常去拍夜景。", "我现在住在北京。"):
                app.query_one("#write-text", TextArea).text = txt
                app.do_write()
                await pilot.pause()
            ck("写入：库内 3 节点",
               app.store.counts("tuismoke/u1")["nodes_total"] == 3)

            # ---- 检索
            app.query_one("#search-q", Input).value = "用户的过敏情况"
            await pilot.click("#btn-search")
            await pilot.pause()
            stable = app.query_one("#search-table", DataTable)
            ck("检索：结果表有行", stable.row_count >= 1,
               f"rows={stable.row_count}")
            sinfo = str(app.query_one("#search-info", Static).content)
            ck("检索：RRF/精排信息", "RRF" in sinfo or "精排" in sinfo)
            top_cells = stable.get_row(list(stable.rows)[0])
            ck("检索：内容列含花粉", "花粉" in str(top_cells[4]),
               str(top_cells[4])[:24])
            # ---- 注入预览
            await pilot.click("#btn-inject")
            await pilot.pause()
            ij = str(app.query_one("#inject-out", Static).content)
            ck("注入预览：三段齐备",
               "system" in ij and "user" in ij and "tool" in ij)

            # ---- 浏览器
            await goto_tab(app, pilot, "tab-browse")
            btable = app.query_one("#browse-table", DataTable)
            ck("浏览器：≥3 行", btable.row_count >= 3,
               f"rows={btable.row_count}")
            key0 = list(btable.rows)[0].value
            app._show_detail(str(key0))
            await pilot.pause()
            detail = str(app.query_one("#node-detail", Static).content)
            ck("节点详情：全参数 + 因子",
               "weight" in detail and "槽位" in detail)

            # ---- 遗忘曲线
            await goto_tab(app, pilot, "tab-forget")
            chart = str(app.query_one("#forget-chart", Static).content)
            stats = str(app.query_one("#forget-stats", Static).content)
            ck("遗忘曲线：火花线渲染", "█" in chart and "┬" in chart)
            ck("遗忘曲线：S/τ/归档期限", "S =" in stats
               and "τ_eff" in stats and "归档期限" in stats)

            # ---- 事件流
            await goto_tab(app, pilot, "tab-events")
            log = app.query_one("#events-log", RichLog)
            ck("事件流：有记录", len(log.lines) >= 1,
               f"lines={len(log.lines)}")

            # ---- 治理
            await goto_tab(app, pilot, "tab-govern", "#btn-govern")
            await pilot.click("#btn-govern")
            await app.workers.wait_for_complete()
            await pilot.pause()
            gov = str(app.query_one("#govern-out", Static).content)
            ck("治理：报告键齐备",
               "edges_decayed" in gov and "gini_weight" in gov)

            # ---- scope 切换
            app.query_one("#scope-in", Input).value = "tuismoke/u2"
            await pilot.click("#btn-scope")
            await pilot.pause()
            chips2 = str(app.query_one("#st-chips", Static).content)
            ck("scope 切换：状态条更新", "tuismoke/u2" in chips2)
            ck("scope 切换：新 scope 为空",
               app.query_one("#browse-table", DataTable).row_count == 0)

            # ---- 会话结束
            await goto_tab(app, pilot, "tab-write", "#btn-session-end")
            await pilot.click("#btn-session-end")
            await pilot.pause()
            ck("结束会话：无异常", True)
        return 0

    rc = asyncio.run(run())
    fails = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))
    print(f"\n== tui smoke：{len(checks) - len(fails)}/{len(checks)} 通过 ==")
    return 1 if fails else rc


# ================================================================ 入口

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="deusmith-tui",
        description="Deusmith-MemorySystem 终端控制台（Textual TUI）")
    ap.add_argument("--config", help="配置文件（默认 ./config.json 或 $DEUSMITH_CONFIG）")
    ap.add_argument("--scope", default="console/default", help="初始 scope")
    ap.add_argument("--memory", action="store_true", help="强制内存后端")
    ap.add_argument("--smoke", action="store_true", help="无头自检（CI/回归用）")
    args = ap.parse_args(argv)

    if args.smoke:
        return smoke()

    try:
        from textual.widgets import TextArea  # noqa: F401 — 提前验证依赖
    except ImportError:
        print("[缺少依赖] TUI 需要 textual：pip install deusmith-memory[tui]",
              file=sys.stderr)
        return 2

    try:
        cfg = load_config(args.config)
        mem, backend = build_memory_system(cfg, args.scope,
                                           force_memory=args.memory)
    except Exception as exc:
        print(f"[启动失败] {exc}", file=sys.stderr)
        return 2

    app = DeusmithTUI(mem, backend)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
