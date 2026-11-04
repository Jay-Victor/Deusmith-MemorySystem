# -*- coding: utf-8 -*-
"""离线治理 —— §五.8 离线路径。

衰减重算（边）→ 压缩合并 → 层级流转 → 边剪枝/环路消解 → 一致性校验 → 归档与审计。

说明（评审 §四）：节点不设独立 weight 衰减——时效由检索期的 recency（连续函数）
承担；离线只对【边】做 λ_e 衰减与剪枝，节点侧做归档判定（导出公式），
避免 TTL 与连续衰减双轨制打架。
"""
from __future__ import annotations

from datetime import datetime

from .algos import (
    cluster_nodes, decay_edge_weight, resolve_cycles, should_prune_edge,
    tier_transition,
)
from .models import EdgeStatus, MemoryNode, NodeStatus, NodeType, Tier, utcnow
from .params import Params


class OfflineGovernor:
    """对单个 scope 执行一轮离线治理。建议节奏：每日或每 6 小时（§三.3）。"""

    def __init__(self, store, embedder, llm=None, params: Params | None = None):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.params = params or Params()

    def run(self, scope: str, now: datetime | None = None) -> dict:
        now = now or utcnow()
        report = {
            "edges_decayed": 0, "edges_pruned": 0, "cycles_resolved": 0,
            "summaries_created": 0, "promoted_deep": 0, "archived": 0,
            "consistency_warnings": 0,
        }
        self._decay_and_prune_edges(scope, now, report)
        self._resolve_cycles(scope, report)
        self._compress_merge(scope, report)
        self._tier_transitions(scope, now, report)
        self._consistency_check(scope, report)
        self.store.log_event("offline_run", report, scope=scope)
        return report

    # ---------------------------------------------------------------- 边衰减与剪枝
    def _decay_and_prune_edges(self, scope: str, now: datetime, report: dict) -> None:
        for e in self.store.all_edges(scope):
            days = max(0.0, (now - e.last_reinforced_at).total_seconds() / 86400.0)
            if days <= 0:
                continue
            e.weight = decay_edge_weight(e.weight, days, params=self.params)
            report["edges_decayed"] += 1
            if should_prune_edge(e.weight, e.last_reinforced_at, e.confidence,
                                 now, self.params):
                e.status = EdgeStatus.pruned
                report["edges_pruned"] += 1
            self.store.update_edge(e)

    # ---------------------------------------------------------------- 环路消解
    def _resolve_cycles(self, scope: str, report: dict) -> None:
        edges = self.store.all_edges(scope)
        if not edges:
            return
        for e, action in resolve_cycles(edges):
            if action == "remove":
                e.status = EdgeStatus.pruned
                self.store.update_edge(e)
                report["cycles_resolved"] += 1

    # ---------------------------------------------------------------- 压缩合并
    def _compress_merge(self, scope: str, report: dict) -> None:
        nodes = [n for n in self.store.active_nodes(scope, tier=Tier.middle)
                 if n.type != NodeType.identity_policy]
        if not nodes:
            return
        for cluster in cluster_nodes(nodes, params=self.params):
            if len(cluster["ids"]) < 2:
                continue
            members = [self.store.get_node(mid) for mid in cluster["ids"]]
            members = [m for m in members if m is not None]
            if len(members) < 2:
                continue
            content = (self.llm.summarize([m.content for m in members],
                                          max_tokens=self.params.tier.summary_max_tokens)
                       if self.llm else "；".join(m.content for m in members))
            summary = MemoryNode(
                type=NodeType.summary, content=content or "（空摘要）", scope=scope,
                tier=Tier.middle, content_vector=self.embedder.embed(content or "摘要"),
                embedding_model=self.embedder.model_name,
                weight=max(m.weight for m in members),
                confidence=max(m.confidence for m in members),
                importance=max(m.importance for m in members),
                emotion=max(m.emotion for m in members),
                generation=max(m.generation for m in members) + 1,
                evidence_refs=tuple(dict.fromkeys(
                    ref for m in members for ref in m.evidence_refs)),
            )
            from .models import EdgeType, MemoryEdge
            self.store.upsert_node(summary)
            for m in members:
                self.store.upsert_edge(MemoryEdge(
                    src_id=summary.id, dst_id=m.id, type=EdgeType.summary_of,
                    weight=1.0, scope=scope))
                m.status = NodeStatus.archived      # 原文归档，可回查
                self.store.upsert_node(m)
            self.store.audit("compress_merge", summary.id, {
                "members": [m.id for m in members],
            })
            report["summaries_created"] += 1

    # ---------------------------------------------------------------- 层级流转
    def _tier_transitions(self, scope: str, now: datetime, report: dict) -> None:
        for node in self.store.active_nodes(scope):
            action = tier_transition(node, now=now, params=self.params)
            if action == "promote_to_middle":
                node.tier = Tier.middle
                self.store.upsert_node(node)
            elif action == "promote_to_deep":
                node.tier = Tier.deep
                self.store.upsert_node(node)
                self.store.audit("promote_deep", node.id, {
                    "confidence": node.confidence, "source_auth": node.source_auth,
                    "cross_session_count": node.cross_session_count,
                })
                report["promoted_deep"] += 1
            elif action == "archive":
                node.status = NodeStatus.archived
                self.store.upsert_node(node)
                report["archived"] += 1
            elif action == "needs_human_confirmation":
                self.store.log_event("awaiting_human_confirmation",
                                     {"node_id": node.id}, scope=scope)

    # ---------------------------------------------------------------- 一致性校验
    def _consistency_check(self, scope: str, report: dict) -> None:
        for node in self.store.active_nodes(scope, tier=Tier.deep):
            if not node.evidence_refs:
                self.store.audit("consistency_warning", node.id, {
                    "reason": "deep node without evidence refs",
                })
                report["consistency_warnings"] += 1
