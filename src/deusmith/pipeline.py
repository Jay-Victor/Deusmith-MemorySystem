# -*- coding: utf-8 -*-
"""管线层 —— 写入/读取/注入三路径（§五.8 调用顺序）。

写入路径：解析抽取 → 可信度评估 → 幂等/冲突处理 → 节点落库 → 即时建边 → 晋升评估 → 日志
读取路径：查询解析 → 多路召回（向量/图/关键词/最近/显式）→ 融合去重 → 绝对刻度打分
          → MMR 裁剪 → 命中强化 → 上下文注入
注入策略（§六.3）：system 段=深层身份/策略；user 段=相关记忆+冲突提示；
          tool 段=证据指针；超窗走降级阶梯（裁低分→裁tool→裁浅层→请求确认）。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .algos import (
    estimate_confidence, estimate_tokens, is_contradiction, mmr_select,
    reinforce, retrieval_score, rrf_fuse, tier_transition,
)
from .llm import RuleBasedLLM
from .models import (
    EdgeType, MemoryEdge, MemoryNode, NodeStatus, NodeType, Tier, utcnow,
)
from .params import Params
from .shallow import ShallowSession


@dataclass
class RetrievalResult:
    selected: list = field(default_factory=list)      # [(node, score, rel)]
    filtered_reasons: list = field(default_factory=list)
    conflict_flags: list = field(default_factory=list)
    trace: dict = field(default_factory=dict)


class DeusmithMemory:
    """记忆系统门面：绑定一个 scope（tenant/user，I1）。"""

    def __init__(self, store, embedder, llm=None, params: Params | None = None,
                 scope: str = "default/user", allowed_sensitivity: int = 2):
        if not scope or not scope.strip():
            raise ValueError("scope 必填（不变量 I1）")
        self.store = store
        self.embedder = embedder
        self.llm = llm or RuleBasedLLM()
        self.params = params or Params()
        self.scope = scope
        self.allowed_sensitivity = allowed_sensitivity
        self._sessions: dict[str, ShallowSession] = {}

    # ================================================================ 会话
    def session(self, session_id: str | None = None) -> ShallowSession:
        sid = session_id or ("sess_" + uuid.uuid4().hex[:12])
        if sid not in self._sessions:
            self._sessions[sid] = ShallowSession(self.scope, sid, self.params, self.llm)
        return self._sessions[sid]

    def end_session(self, session: ShallowSession, trace_id: str | None = None) -> None:
        """会话结束（§二.1 触发）：终态摘要 → 浅层晋升评估 → 记日志 → 登记 30min 清理。"""
        session.maybe_update_summary()
        session.close()
        for node in self.store.active_nodes(self.scope, tier=Tier.shallow):
            if node.session_id == session.id:
                action = tier_transition(node)
                if action == "promote_to_middle":
                    node.tier = Tier.middle
                    self.store.upsert_node(node)
        self.store.log_event("session_end", {
            "session_id": session.id, "summary": session.rolling_summary[:200],
        }, trace_id=trace_id, scope=self.scope)

    def cleanup_expired_sessions(self, now: datetime | None = None) -> list[str]:
        """清理超过 30 分钟的会话原文（浅层 TTL）。返回被清理的会话 id。"""
        now = now or utcnow()
        expired = [sid for sid, s in self._sessions.items()
                   if s.expiry_at() and now >= s.expiry_at()]
        for sid in expired:
            s = self._sessions.pop(sid)
            s.rounds.clear()
            s.tool_results.clear()
        return expired

    # ================================================================ 写入路径
    def write(self, session: ShallowSession, text: str, source: str = "user",
              source_auth: float = 0.8, trace_id: str | None = None) -> list[MemoryNode]:
        session.add_round("user", text)
        session.maybe_update_summary()
        items = self.llm.extract(text)
        written: list[MemoryNode] = []
        for item in items:
            node = self._build_node(session, item, source, source_auth)
            result = self._ingest(session, node, trace_id)
            if result is not None:
                written.append(result)
        self.store.log_event("write", {
            "text": text[:200], "items": len(items),
            "node_ids": [n.id for n in written],
        }, trace_id=trace_id, scope=self.scope)
        return written

    def _build_node(self, session: ShallowSession, item: dict, source: str,
                    source_auth: float) -> MemoryNode:
        vec = self.embedder.embed(item["content"])
        node = MemoryNode(
            type=NodeType(item["type"]), content=item["content"], scope=self.scope,
            tier=Tier.shallow, content_vector=vec,
            embedding_model=self.embedder.model_name,
            subject=item.get("subject"), predicate=item.get("predicate"),
            object_value=item.get("object_value"),
            polarity=int(item.get("polarity", 0) or 0),
            source=source, source_auth=source_auth,
            importance=float(item.get("importance", 0.5)),
            emotion=float(item.get("emotion", 0.0)),
            sensitivity=int(item.get("sensitivity", 1)),
            explicit_remember=bool(item.get("explicit_remember")),
            session_id=session.id,
        )
        node.confidence = estimate_confidence(
            node, evidence_count=1, freshness=1.0,
            conflict_penalty=0.0, params=self.params)
        return node

    def _ingest(self, session: ShallowSession, node: MemoryNode,
                trace_id: str | None) -> MemoryNode | None:
        now = utcnow()
        # 1) 幂等（I8）：同 hash 活跃节点 → 强化而非重复写入
        dup = self.store.find_active_by_hash(self.scope, node.content_hash,
                                             node.embedding_model)
        if dup is not None and not is_contradiction(dup, node):
            reinforce(dup, signal_strength=0.7, now=now, params=self.params)
            if dup.session_id != node.session_id:
                dup.cross_session_count += 1
            self.store.upsert_node(dup)
            self.store.log_event("duplicate_write", {"node_id": dup.id},
                                 trace_id=trace_id, scope=self.scope)
            return dup

        # 2) 冲突（I3）：槽位身份 + 极性/宾语判定，不走相似度
        if node.subject and node.predicate:
            for old in self.store.find_by_slot(self.scope, node.subject, node.predicate):
                if is_contradiction(old, node):
                    return self._handle_conflict(session, old, node, now, trace_id)

        # 3) 新节点落库（先浅层），即时晋升评估
        action = tier_transition(node, now=now, params=self.params)
        if action == "promote_to_middle":
            node.tier = Tier.middle
        self.store.upsert_node(node)
        session.node_ids.append(node.id)
        self._immediate_edges(session, node, now)
        return node

    def _handle_conflict(self, session: ShallowSession, old: MemoryNode,
                         new: MemoryNode, now: datetime,
                         trace_id: str | None) -> MemoryNode:
        """冲突仲裁（§五.5）：深层红线 → 只争议不覆盖；权威明确 → supersedes；
        势均力敌 → 双版本 contested 请求确认。"""
        new.tier = Tier.middle   # 挑战者不因冲突直接进入旧节点层级（晋升走正规门槛）

        if old.tier == Tier.deep:
            # 红线：不允许静默覆盖深层记忆
            new.status = NodeStatus.contested
            new.contested_by = old.id
            self.store.upsert_node(new)
            self._add_contradicts_edge(new, old)
            self.store.audit("conflict_deep", old.id, {
                "challenger": new.id, "content": new.content,
            }, trace_id=trace_id)
            self.store.log_event("conflict", {"old": old.id, "new": new.id,
                                              "verdict": "deep_protected"},
                                 trace_id=trace_id, scope=self.scope)
            return new

        verdict = self.llm.arbitrate(old, new)
        if verdict == "prefer_b":
            new.supersedes = old.id
            new.status = NodeStatus.active
            old.status = NodeStatus.deprecated
            self.store.upsert_node(old)
            self.store.upsert_node(new)
            self._add_contradicts_edge(new, old)
            self.store.audit("conflict_resolved", old.id, {
                "winner": new.id, "verdict": "prefer_b",
            }, trace_id=trace_id)
        else:  # prefer_a / contested：双版本保留，等确认
            new.status = NodeStatus.contested
            new.contested_by = old.id
            old.status = NodeStatus.contested
            old.contested_by = new.id
            self.store.upsert_node(old)
            self.store.upsert_node(new)
            self._add_contradicts_edge(new, old)
            self.store.audit("conflict_contested", old.id, {
                "challenger": new.id, "verdict": verdict,
            }, trace_id=trace_id)
        self.store.log_event("conflict", {"old": old.id, "new": new.id,
                                          "verdict": verdict},
                             trace_id=trace_id, scope=self.scope)
        session.node_ids.append(new.id)
        return new

    def _add_contradicts_edge(self, new: MemoryNode, old: MemoryNode) -> None:
        edge = MemoryEdge(src_id=new.id, dst_id=old.id, type=EdgeType.contradicts,
                          weight=0.5, scope=self.scope)
        edge.weight = edge.weight * 0.5          # 冲突边保留并降权（§五.4）
        from .models import EdgeStatus
        edge.status = EdgeStatus.weakened
        self.store.upsert_edge(edge)

    def _immediate_edges(self, session: ShallowSession, node: MemoryNode,
                         now: datetime) -> None:
        """即时建边（§三.3）：kNN 语义边 + 同会话时序边。"""
        k = self.params.retrieval.knn_edge_topk
        hits = self.store.vector_search(self.scope, node.content_vector,
                                        topk=k + 1,
                                        allowed_sensitivity=3)
        for other, sim in hits:
            if other.id == node.id or sim < 0.3:
                continue
            self.store.upsert_edge(MemoryEdge(
                src_id=node.id, dst_id=other.id, type=EdgeType.semantic_similar,
                weight=min(1.0, sim), confidence=node.confidence, scope=self.scope))
        if session.last_node_id and session.last_node_id != node.id:
            self.store.upsert_edge(MemoryEdge(
                src_id=session.last_node_id, dst_id=node.id,
                type=EdgeType.temporal_next, weight=0.6, scope=self.scope))
        session.last_node_id = node.id

    # ================================================================ 读取路径
    def retrieve(self, session: ShallowSession, query: str, topk: int = 8,
                 trace_id: str | None = None) -> RetrievalResult:
        now = utcnow()
        trace_id = trace_id or ("tr_" + uuid.uuid4().hex[:10])
        qvec = self.embedder.embed(query)
        rp = self.params.retrieval

        # ---- 多路召回（§六.1）
        vhits = self.store.vector_search(self.scope, qvec, topk=rp.topk_recall,
                                         allowed_sensitivity=self.allowed_sensitivity)
        vector_rank = [n.id for n, _ in vhits]
        seed_ids = vector_rank[:5]
        seed_sims = [sim for _, sim in vhits[:5]]
        graph_ids: set = set()
        graph_scores: dict[str, float] = {}
        if seed_ids:
            graph_ids, edges = self.store.neighbors(self.scope, seed_ids,
                                                    hops=rp.graph_hops)
            graph_scores = self._walk_scores(seed_ids, seed_sims, edges,
                                             rp.ppr_damping)
        kw_hits = self.store.keyword_search(self.scope, query, limit=20,
                                            allowed_sensitivity=self.allowed_sensitivity)
        recent = self.store.recent_active(self.scope, 20)
        resident = self.store.resident_nodes(self.scope)

        rankings = {
            "vector": vector_rank,
            "graph": sorted(graph_scores, key=graph_scores.get, reverse=True),
            "keyword": [n.id for n in kw_hits],
            "recent": [n.id for n in recent],
            "explicit": [n.id for n in resident],
        }

        # ---- 候选收集 + 敏感门控（I10）
        candidates: dict[str, MemoryNode] = {}
        filtered_reasons: list[str] = []
        for ids in rankings.values():
            for nid in ids:
                if nid in candidates:
                    continue
                node = self.store.get_node(nid)
                if node is None:
                    continue
                if node.sensitivity > self.allowed_sensitivity:
                    filtered_reasons.append(f"sensitivity:{nid}:level={node.sensitivity}")
                    continue
                candidates[nid] = node

        # ---- 融合去重 + 打分（I4：绝对刻度）
        scored = []
        for nid, node in candidates.items():
            rel = rrf_fuse(nid, rankings, self.params)
            score = retrieval_score(rel, node, now, self.allowed_sensitivity,
                                    self.params)
            scored.append((node, score, rel))
        scored.sort(key=lambda x: x[1], reverse=True)

        # ---- MMR 多样性裁剪（§六.2）
        pool = [(n.id, s, n.content_vector) for n, s, _ in scored[: max(topk * 3, 12)]]
        picked = mmr_select(pool, k=topk, lam=self.params.score.mmr_lambda)
        pick_ids = {p[0] for p in picked}
        selected = [(n, s, r) for n, s, r in scored if n.id in pick_ids]
        selected.sort(key=lambda x: x[1], reverse=True)

        # ---- 命中强化（§二.2 检索触发）
        for node, _, _ in selected:
            reinforce(node, signal_strength=0.5, now=now, params=self.params)
            self.store.upsert_node(node)

        conflict_flags = [n.id for n, _, _ in selected
                          if n.status == NodeStatus.contested]
        trace = {
            "trace_id": trace_id, "query": query[:120],
            "recall": {ch: len(ids) for ch, ids in rankings.items()},
            "candidate_ids": list(candidates.keys())[:50],
            "selected_ids": [n.id for n, _, _ in selected],
            "scores": {n.id: round(s, 4) for n, s, _ in selected},
            "conflict_flags": conflict_flags,
            "filtered_reasons": filtered_reasons[:10],
        }
        self.store.log_event("retrieval", trace, trace_id=trace_id, scope=self.scope)
        return RetrievalResult(selected=selected, filtered_reasons=filtered_reasons,
                               conflict_flags=conflict_flags, trace=trace)

    def _walk_scores(self, seed_ids, seed_sims, edges, damping: float) -> dict:
        """图游走评分（PPR 简化版）：score = max over paths damping^hop·Πw。"""
        best = {sid: float(sim) for sid, sim in zip(seed_ids, seed_sims)}
        adj: dict[str, list] = {}
        for e in edges:
            adj.setdefault(e.src_id, []).append((e.dst_id, e.weight))
            adj.setdefault(e.dst_id, []).append((e.src_id, e.weight))
        for _ in range(self.params.retrieval.graph_hops):
            nxt = dict(best)
            for nid, sc in best.items():
                for nbr, w in adj.get(nid, []):
                    val = damping * sc * w
                    if val > nxt.get(nbr, 0.0):
                        nxt[nbr] = val
            best = nxt
        return best

    # ================================================================ 注入路径
    def inject_context(self, session: ShallowSession, query: str,
                       token_budget: int | None = None,
                       trace_id: str | None = None) -> dict:
        """§六.3 注入策略 + 超窗降级阶梯（I7）。"""
        budget = token_budget or self.params.retrieval.token_budget
        res = self.retrieve(session, query, topk=10, trace_id=trace_id)

        # system 段：深层身份/策略 + 全局约束摘要（常驻通道）
        system_parts: list[str] = []
        for node in self.store.resident_nodes(self.scope):
            if (node.tier == Tier.deep and node.status == NodeStatus.active
                    and node.sensitivity <= self.allowed_sensitivity):
                system_parts.append(node.content)

        # user 段：按分数排序的相关记忆；争议只注入冲突提示
        user_parts: list[str] = []
        for node, score, _ in res.selected:
            if node.status == NodeStatus.contested:
                topic = node.predicate or node.content[:12]
                user_parts.append(f"[记忆冲突] 关于「{topic}」存在相互矛盾的记忆，需要用户确认")
                continue
            if node.tier == Tier.deep:
                continue  # 深层已进 system 段
            if estimate_tokens(node.content) > 60:
                user_parts.append(f"[记忆#{node.id[-6:]}] {node.content[:120]}…")
            else:
                user_parts.append(node.content)

        # tool 段：证据指针
        tool_parts: list[str] = []
        for node, _, _ in res.selected:
            if node.evidence_refs:
                tool_parts.append("证据: " + ", ".join(node.evidence_refs[:5]))

        # 浅层会话投影（≤30% 预算）
        shallow_budget = max(1, int(budget * self.params.tier.shallow_context_share))
        shallow_text = session.to_text(shallow_budget)

        def tok(parts):
            return sum(estimate_tokens(p) for p in parts)

        total = tok(system_parts) + tok(user_parts) + tok(tool_parts) \
            + estimate_tokens(shallow_text)
        degraded = False
        # 降级阶梯 1：裁低分记忆候选（user 段按分数序，尾部最低分）
        while total > budget and user_parts:
            dropped = user_parts.pop()
            total -= estimate_tokens(dropped)
            degraded = True
        # 阶梯 2：裁 tool 段
        if total > budget and tool_parts:
            total -= tok(tool_parts)
            tool_parts = []
            degraded = True
        # 阶梯 3：裁浅层投影
        if total > budget and shallow_text:
            total -= estimate_tokens(shallow_text)
            shallow_text = ""
            degraded = True
        # 阶梯 4：请求用户分步确认
        if total > budget:
            user_parts = ["[记忆系统] 相关记忆超出预算，请细化问题或分步检索"]
            total = tok(system_parts) + estimate_tokens(user_parts[0])
            degraded = True

        out = {
            "system": "\n".join(system_parts),
            "user": "\n".join(filter(None, [*user_parts, shallow_text])),
            "tool": "\n".join(tool_parts),
            "tokens": total,
            "degraded": degraded,
            "conflict_flags": res.conflict_flags,
            "selected": [(n.id, round(s, 4)) for n, s, _ in res.selected],
            "trace_id": res.trace.get("trace_id"),
        }
        self.store.log_event("inject", {
            "trace_id": res.trace.get("trace_id"), "tokens": total,
            "degraded": degraded, "query": query[:120],
            "selected_ids": out["selected"],
        }, trace_id=res.trace.get("trace_id"), scope=self.scope)
        return out
