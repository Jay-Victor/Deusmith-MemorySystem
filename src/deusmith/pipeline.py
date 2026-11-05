# -*- coding: utf-8 -*-
"""管线层 —— 写入/读取/注入三路径（§五.8 调用顺序）+ 阶段二扩展。

写入路径：解析抽取（LLM 失败自动降级规则引擎，评审 #6）→ 可信度评估
          → 幂等/冲突处理（supersede 写双时间窗口，评审 #7）→ 节点落库
          （证据挂钩，评审 #8）→ 即时建边 → 晋升评估 → 会话持久化（评审 #5）→ 日志
读取路径：查询 → 五路召回（向量/图/关键词/最近/显式）→ 融合去重
          → 绝对刻度打分（过期记忆降权）→ MMR 裁剪 → 命中强化 → 上下文注入
可解释性（评审 #9）：explain_injection / explain_score / why_selected / slot_history。
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from .algos import (
    decay_recency, estimate_confidence, estimate_tokens, is_contradiction,
    mmr_select, reinforce, retrieval_score, rrf_fuse, tier_transition,
)
from .llm import RuleBasedLLM
from .models import (
    EdgeType, MemoryEdge, MemoryNode, NodeStatus, NodeType, Tier, utcnow,
)
from .params import Params
from .shallow import Round, ShallowSession


@dataclass
class RetrievalResult:
    selected: list = field(default_factory=list)      # [(node, score, rel)]
    filtered_reasons: list = field(default_factory=list)
    conflict_flags: list = field(default_factory=list)
    trace: dict = field(default_factory=dict)


def _payload_of(p):
    if isinstance(p, str):
        try:
            return json.loads(p)
        except (ValueError, TypeError):
            return {}
    return p or {}


class DeusmithMemory:
    """记忆系统门面：绑定一个 scope（tenant/user，I1）。"""

    def __init__(self, store, embedder, llm=None, params: Params | None = None,
                 scope: str = "default/user", allowed_sensitivity: int = 2,
                 evidence_store=None, reranker=None):
        if not scope or not scope.strip():
            raise ValueError("scope 必填（不变量 I1）")
        self.store = store
        self.embedder = embedder
        self.llm = llm or RuleBasedLLM()
        self._fallback_llm = RuleBasedLLM()      # 在线 LLM 失败时的离线保底
        self.params = params or Params()
        self.scope = scope
        self.allowed_sensitivity = allowed_sensitivity
        self.evidence_store = evidence_store
        self.reranker = reranker                 # None → 精排关闭（默认，零开销）
        self._sessions: dict[str, ShallowSession] = {}

    # ================================================================ 会话
    def session(self, session_id: str | None = None) -> ShallowSession:
        """取会话；DB 中已存在则恢复（进程重启不丢，评审 #5），否则新建。"""
        sid = session_id or ("sess_" + uuid.uuid4().hex[:12])
        if sid not in self._sessions:
            raw = self.store.load_session_raw(self.scope, sid)
            if raw is not None:
                s = ShallowSession(self.scope, sid, self.params, self.llm)
                s.created_at = raw["created_at"]
                s.last_active_at = raw["last_active_at"]
                s.closed_at = raw["closed_at"]
                s.rolling_summary = raw["summary"] or ""
                for role, content, ts in raw["rounds"]:
                    s.rounds.append(Round(role=role, content=content, ts=ts))
            else:
                s = ShallowSession(self.scope, sid, self.params, self.llm)
            self._sessions[sid] = s
        return self._sessions[sid]

    def end_session(self, session: ShallowSession, trace_id: str | None = None) -> None:
        """会话结束（§二.1 触发）：终态摘要 → 浅层晋升评估 → 持久化 → 记日志。"""
        session.maybe_update_summary()
        session.close()
        for node in self.store.active_nodes(self.scope, tier=Tier.shallow):
            if node.session_id == session.id:
                action = tier_transition(node)
                if action == "promote_to_middle":
                    node.tier = Tier.middle
                    self.store.upsert_node(node)
        self.store.save_session(session)
        self.store.log_event("session_end", {
            "session_id": session.id, "summary": session.rolling_summary[:200],
        }, trace_id=trace_id, scope=self.scope)

    def cleanup_expired_sessions(self, now: datetime | None = None) -> list[str]:
        """清理超过 30 分钟的会话原文（浅层 TTL；DB 行保留作审计）。"""
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
        try:
            items = self.llm.extract(text)
        except Exception as exc:                     # 在线 LLM 失败 → 规则兜底
            self.store.log_event("llm_fallback", {"error": str(exc)[:200]},
                                 trace_id=trace_id, scope=self.scope)
            items = self._fallback_llm.extract(text)
        written: list[MemoryNode] = []
        for item in items:
            node = self._build_node(session, item, source, source_auth)
            ev = None
            if self.evidence_store is not None:
                ev = self.evidence_store.save(item["content"], source=source)
                node.evidence_refs = node.evidence_refs + (ev["evidence_id"],)
            result = self._ingest(session, node, trace_id)
            if result is not None:
                if ev is not None:
                    self.store.record_evidence(
                        result.id, ev["evidence_id"], source,
                        ev["content_hash"], ev["pointer_uri"])
                written.append(result)
        self.store.save_session(session)
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
        """冲突仲裁（§五.5）：深层红线 → 只争议不覆盖；权威明确 → supersedes
        （写双时间窗口：old.valid_to=now，new.valid_from=now）；势均力敌 →
        双版本 contested 请求确认。"""
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
            new.valid_from = new.valid_from or now       # 双时间（评审 #7）
            old.status = NodeStatus.deprecated
            old.valid_to = now                            # 旧版本失效时间
            self.store.upsert_node(old)
            self.store.upsert_node(new)
            self._add_contradicts_edge(new, old)
            self.store.audit("conflict_resolved", old.id, {
                "winner": new.id, "verdict": "prefer_b",
            }, trace_id=trace_id)
        else:  # prefer_a / contested：双版本保留，等确认（不写时间窗）
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
    def _keyword_terms(self, query: str) -> list[str]:
        """关键词通道切词：CJK 二元组 + 拉丁词。
        中文整句子串匹配几乎永远落空（无分词），二元组是最小可用粒度。"""
        terms: set[str] = set()
        for m in re.finditer(r"[A-Za-z0-9]+", query or ""):
            terms.add(m.group(0).lower())
        for m in re.finditer(r"[\u4e00-\u9fff]+", query or ""):
            s = m.group(0)
            if len(s) == 1:
                terms.add(s)
            for i in range(len(s) - 1):
                terms.add(s[i:i + 2])
        return sorted(terms)[:8]

    def retrieve(self, session: ShallowSession, query: str, topk: int = 8,
                 trace_id: str | None = None, reinforce_hits: bool = True
                 ) -> RetrievalResult:
        now = utcnow()
        trace_id = trace_id or ("tr_" + uuid.uuid4().hex[:10])
        qvec = self.embedder.embed_query(query)     # 查询侧嵌入（bge 类带指令前缀）
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
            graph_ids, edges = self.store.neighbors(
                self.scope, seed_ids, hops=rp.graph_hops, max_expand=100)
            graph_scores = self._walk_scores(seed_ids, seed_sims, edges,
                                             rp.ppr_damping)
        # keyword 通道：只收稀有 term（df<10）。通用二元组（"用户"命中全部节点）
        # 的排名与查询语义无关，进 RRF 会让高频节点在所有"用户…"查询里拿名次分
        # （RRF 只看名次不看分差）。连续性由浅层会话投影承担，recent 不进融合。
        kw_scores: dict[str, float] = {}
        for term in self._keyword_terms(query):
            hits = self.store.keyword_search(self.scope, term, limit=10,
                                             allowed_sensitivity=self.allowed_sensitivity)
            if len(hits) >= 10:
                continue          # 通用词（df 触顶）：不参与关键词通道
            if not hits:
                continue
            w = 1.0 / len(hits)
            for n in hits:
                kw_scores[n.id] = kw_scores.get(n.id, 0.0) + w
        recent = self.store.recent_active(self.scope, 20)

        # 四路召回融合（vector/graph/keyword/recent）。常驻通道（explicit）不进
        # 排序：常驻记忆无条件注入（inject_context 的 [常驻记忆] 段），不靠相关性。
        rankings = {
            "vector": vector_rank,
            "graph": sorted(graph_scores, key=graph_scores.get, reverse=True),
            "keyword": sorted(kw_scores, key=kw_scores.get, reverse=True),
            "recent": [n.id for n in recent],
        }

        # ---- 候选收集 + 敏感门控（I10）+ 状态门控
        # （图通道的边不过滤对端节点状态：deprecated/归档版本可经边游走
        #   回到候选池，必须在此统一拦截，否则被替代的旧值重新参选）
        candidates: dict[str, MemoryNode] = {}
        filtered_reasons: list[str] = []
        seen_ids: set = set()
        for ids in rankings.values():
            for nid in ids:
                if nid in candidates or nid in seen_ids:
                    continue
                seen_ids.add(nid)
                node = self.store.get_node(nid)
                if node is None:
                    continue
                if node.status not in (NodeStatus.active, NodeStatus.contested):
                    continue
                if node.sensitivity > self.allowed_sensitivity:
                    filtered_reasons.append(f"sensitivity:{nid}:level={node.sensitivity}")
                    continue
                candidates[nid] = node

        # ---- 融合去重 + 打分（I4：绝对刻度；过期记忆降权）
        scored = []
        for nid, node in candidates.items():
            rel = rrf_fuse(nid, rankings, self.params)
            score = retrieval_score(rel, node, now, self.allowed_sensitivity,
                                    self.params)
            scored.append((node, score, rel))
        scored.sort(key=lambda x: x[1], reverse=True)

        # ---- 精排（可选 reranker：cross-encoder / API；失败降级回 RRF 排序）
        rerank_info: dict | None = None
        if self.reranker is not None:
            scored, rerank_info = self._apply_rerank(query, scored, now, trace_id)

        # ---- MMR 多样性裁剪（§六.2）
        pool = [(n.id, s, n.content_vector) for n, s, _ in scored[: max(topk * 3, 12)]]
        picked = mmr_select(pool, k=topk, lam=self.params.score.mmr_lambda)
        pick_ids = {p[0] for p in picked}
        selected = [(n, s, r) for n, s, r in scored if n.id in pick_ids]
        selected.sort(key=lambda x: x[1], reverse=True)

        # ---- 命中强化（§二.2 检索触发；评估模式可关闭以测冷启动排序）
        if reinforce_hits:
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
        if rerank_info:
            trace["rerank"] = rerank_info
        self.store.log_event("retrieval", trace, trace_id=trace_id, scope=self.scope)
        return RetrievalResult(selected=selected, filtered_reasons=filtered_reasons,
                               conflict_flags=conflict_flags, trace=trace)

    def _apply_rerank(self, query: str, scored: list, now: datetime,
                      trace_id: str | None):
        """精排：rel' = (1-w)·rel_rrf + w·rerank_score（v0.3，§六.1 读取路径）。

        - 只精排头部 rerank_pool 条（cross-encoder 代价 O(pool)），尾部不动；
        - rerank 分数缺失的条目保持原 rel（分批 API 部分失败不炸整轮检索）；
        - 任何异常（网络/模型加载）降级回 RRF 排序，记 rerank_fallback 事件
          （与 LLM 失败降级同语义，评审 #6）；
        - 两者同为绝对刻度（不变量 I4 域内）：混合值仍与查询措辞无关，
          min_score_threshold 等绝对阈值继续可移植。
        """
        rp = self.params.retrieval
        if self.reranker is None:            # 独立调用守卫（类型收窄）
            return scored, None
        pool = scored[: max(1, int(rp.rerank_pool))]
        try:
            rscores = self.reranker.rerank(query, [n.content for n, _, _ in pool])
        except Exception as exc:
            self.store.log_event("rerank_fallback", {"error": str(exc)[:200]},
                                 trace_id=trace_id, scope=self.scope)
            return scored, {"used": False, "error": str(exc)[:120]}
        if not rscores:
            return scored, None     # IdentityReranker：不参与精排，零痕迹

        w = min(1.0, max(0.0, rp.rerank_weight))
        blended: list = []
        for i, (node, _old_score, rel) in enumerate(pool):
            rs = rscores[i] if i < len(rscores) else None
            if rs is None:
                blended.append((node, _old_score, rel))
                continue
            new_rel = (1.0 - w) * rel + w * float(rs)
            new_score = retrieval_score(new_rel, node, now,
                                        self.allowed_sensitivity, self.params)
            blended.append((node, new_score, new_rel))
        out = blended + scored[len(pool):]
        out.sort(key=lambda x: x[1], reverse=True)
        info = {
            "used": True, "model": getattr(self.reranker, "model_name", "?"),
            "pool": len(pool), "weight": w,
            "rerank_scores": {n.id: round(float(s), 4)
                              for (n, _, _), s in zip(pool, rscores)
                              if s is not None},
        }
        return out, info

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

        # user 段开头：常驻记忆无条件注入（不进 RRF 排序，技能库 §4）。
        # 深层身份已在 system 段；此处是 explicit_remember 的中层记忆，限量 5。
        resident_parts: list[str] = []
        for node in self.store.resident_nodes(self.scope):
            if (node.tier != Tier.deep and node.status == NodeStatus.active
                    and node.sensitivity <= self.allowed_sensitivity):
                resident_parts.append(f"[常驻记忆] {node.content}")

        # user 段：常驻 → 冲突提示 → 按分数排序的相关记忆（争议按槽位去重）
        user_parts: list[str] = list(resident_parts[:5])
        hinted_slots: set = set()
        for node, score, _ in res.selected:
            if node.status == NodeStatus.contested:
                key = (node.subject, node.predicate)
                if key in hinted_slots:
                    continue
                hinted_slots.add(key)
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
        # 降级阶梯 1：裁低分记忆候选
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

    # ================================================================ 可解释性（评审 #9）
    def explain_injection(self, trace_id: str) -> dict:
        """"这次注入了什么、为什么"：按 trace_id 还原注入事件。"""
        events = self.store.trace_events(trace_id)
        inject = [e for e in events if e["event_type"] == "inject"]
        if not inject:
            return {"trace_id": trace_id, "found": False}
        payload = _payload_of(inject[-1].get("payload"))
        return {
            "trace_id": trace_id, "found": True,
            "query": payload.get("query"), "tokens": payload.get("tokens"),
            "degraded": payload.get("degraded"),
            "selected": payload.get("selected_ids"),
            "events": [e["event_type"] for e in events],
        }

    def why_selected(self, node_id: str, limit: int = 10) -> list[dict]:
        """"你为什么记住了这个"：哪些检索轮次选中了该节点、当时得分。"""
        out = []
        for e in self.store.recent_events(self.scope, limit=200):
            if e["event_type"] != "retrieval":
                continue
            p = _payload_of(e.get("payload"))
            if node_id in (p.get("selected_ids") or []):
                out.append({"trace_id": e.get("trace_id"),
                            "query": p.get("query"),
                            "score": (p.get("scores") or {}).get(node_id)})
        return out[:limit]

    def explain_score(self, node_id: str) -> dict:
        """"这条记忆为什么排这么高"：因子分解 + rel=1.0 时的上限分。"""
        n = self.store.get_node(node_id)
        if n is None:
            return {"found": False}
        now = utcnow()
        rec = decay_recency(n, now, self.params)
        return {
            "found": True, "node_id": node_id, "type": n.type.value,
            "tier": n.tier.value, "status": n.status.value, "content": n.content,
            "factors": {
                "weight": round(n.weight, 4), "recency": round(rec, 4),
                "confidence": round(n.confidence, 4),
                "importance": round(n.importance, 4),
                "activation_count": n.activation_count,
                "valid_to": n.valid_to.isoformat() if n.valid_to else None,
            },
            "score_at_full_rel": round(
                retrieval_score(1.0, n, now, self.allowed_sensitivity,
                                self.params), 4),
            "last_access_at": n.last_access_at.isoformat(),
        }

    def slot_history(self, subject: str, predicate: str) -> list[dict]:
        """时序历史（双时间，评审 #7）：同槽位全部版本按生效时间排列。"""
        nodes = self.store.slot_history(self.scope, subject, predicate)
        return [{
            "node_id": n.id, "content": n.content, "status": n.status.value,
            "valid_from": n.valid_from.isoformat() if n.valid_from else None,
            "valid_to": n.valid_to.isoformat() if n.valid_to else None,
        } for n in nodes]
