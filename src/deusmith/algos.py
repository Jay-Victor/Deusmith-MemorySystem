# -*- coding: utf-8 -*-
"""核心算法 —— 对应《Deusmith-Memory》§五。

全部为纯函数（无 IO、无全局状态），默认超参见 params.py。
签名与文档的对应：
  decay_recency(memory, now)              §五.1 遗忘衰减，O(1)
  reinforce(memory, signal_strength, now) §五.2 激活与强化，O(1)
  tier_transition(memory, stats, now)     §五.3 层级流转，O(1)
  update_edges 的纯函数部分：merge_edge_weight / decay_edge_weight /
    should_prune_edge / weaken_contradiction_edge / resolve_cycles   §五.4
  estimate_confidence(node, evidence)     §五.5 可信度评估，O(E)
  retrieval_score(query, memory, now)     §五.6 检索打分，O(1)/条
    （query 以召回层算好的 rel_norm 传入，见 rrf_fuse）
  compress_merge 的纯函数部分：cluster_nodes                §五.7

评审修正（相对原文档）：
  1. estimate_confidence 增加 sigmoid 增益 k=3：原式 sigmoid(≤0.9) 上限 0.711，
     永远达不到深层晋升阈值 0.9，深层记忆会饿死（数值见 docs/04）。
  2. retrieval_score 的 rel_norm 用绝对参考点归一（rrf_fuse），不用当次查询
     min-max/rank（防绝对阈值失效，不变量 I4）。
  3. 归档期限是导出量 archive_after_days = -τ_eff·ln(θ)，不与连续衰减双轨并存。
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from .models import EdgeType, EdgeStatus, MemoryEdge, MemoryNode, NodeStatus, NodeType, Tier
from .params import Params

_PARAMS = Params()


# ---------------------------------------------------------------- 槽位与冲突（不变量 I3）

def is_contradiction(a: MemoryNode, b: MemoryNode) -> bool:
    """冲突判定：槽位身份相同 + （极性相反 或 宾语不同）。

    语义相似度不参与判定——cos("我喜欢摄影","我不喜欢摄影了")≈0.95，
    相似度把否定当重复（评审错误#1）。无槽位节点返回 False（交给语义仲裁层）。
    """
    if a.slot_key() != b.slot_key() or a.subject is None:
        return False
    if a.polarity != 0 and b.polarity != 0 and a.polarity != b.polarity:
        return True
    if a.object_value and b.object_value and a.object_value != b.object_value:
        return True
    return False


def is_duplicate(a: MemoryNode, b: MemoryNode) -> bool:
    """去重判定：同槽位 + 同极性 + 同宾语（或均无宾语）+ 同类型；无槽位时退回内容 hash。"""
    if a.subject is not None and b.subject is not None:
        return (
            a.slot_key() == b.slot_key()
            and a.polarity == b.polarity
            and (a.object_value or None) == (b.object_value or None)
            and a.type == b.type
        )
    return a.content_hash == b.content_hash and a.scope == b.scope


def _days_between(a: datetime, b: datetime) -> float:
    return max(0.0, (a - b).total_seconds() / 86400.0)


# ---------------------------------------------------------------- §五.1 遗忘衰减

def decay_stretch(node: MemoryNode, params: Params = _PARAMS) -> float:
    """S = 1 + 0.8·importance + 0.5·emotion + 0.7·confidence + 0.3·ln(1+activation)。"""
    d = params.decay
    return (
        1.0
        + d.k_importance * node.importance
        + d.k_emotion * node.emotion
        + d.k_confidence * node.confidence
        + d.k_activation * math.log1p(node.activation_count)
    )


def tau_eff_days(node: MemoryNode, params: Params = _PARAMS) -> float:
    """有效时间常数 τ_eff = τ_base · S（S≥1 保证单调，不变量 I6）。"""
    return params.decay.tau_base_days * decay_stretch(node, params)


def decay_recency(node: MemoryNode, now: datetime, params: Params = _PARAMS) -> float:
    """recency = exp(-Δt / (τ_base · S))；Δt 为距 last_access_at 的天数。O(1)。

    边界：Δt=0 → 1；Δt→∞ → 0；未来时间钳为 0 天。
    """
    dt = _days_between(now, node.last_access_at)
    return math.exp(-dt / tau_eff_days(node, params))


def archive_after_days(node: MemoryNode, params: Params = _PARAMS,
                       theta: float | None = None) -> float:
    """导出归档期限 t = -τ_eff·ln(θ_archive)（评审 §二：TTL 是导出量非独立常数）。"""
    th = theta if theta is not None else params.tier.archive_score_floor
    return -tau_eff_days(node, params) * math.log(th)


# ---------------------------------------------------------------- §五.2 激活与强化

def reinforce(node: MemoryNode, signal_strength: float, now: datetime,
              params: Params = _PARAMS) -> MemoryNode:
    """activation_count+=1；last_access_at=now；
    weight ← clip(w + η(1-w)·s, 0, 1)；置信度按一致性证据上探；TTL 随访问滑动续期。O(1)。
    """
    eta = params.reinforce.eta
    node.activation_count += 1
    node.last_access_at = now
    node.weight = min(1.0, max(0.0, node.weight + eta * (1.0 - node.weight) * signal_strength))
    # 置信度按证据更新（§五.2 附带）：一致且强的信号使置信度上探，永不超过 1
    node.confidence = min(
        1.0,
        node.confidence + eta * (1.0 - node.confidence) * node.consistency * signal_strength,
    )
    node.updated_at = now
    return node


# ---------------------------------------------------------------- §五.5 可信度评估

def evidence_norm(evidence_count: int, reference: int = 5) -> float:
    """证据数量归一：ln(1+n)/ln(1+N_ref)，n≥N_ref 时为 1。"""
    if evidence_count <= 0:
        return 0.0
    return min(1.0, math.log1p(evidence_count) / math.log1p(reference))


def estimate_confidence(node: MemoryNode, evidence_count: int = 1, freshness: float = 1.0,
                        conflict_penalty: float = 0.0, params: Params = _PARAMS) -> float:
    """c = sigmoid(k·(w1·auth + w2·consistency + w3·evidence + w4·freshness − w5·penalty))。

    增益 k=3（评审修正）：无增益时输入上限 0.9 → sigmoid 上限 0.711 < 晋升阈值 0.9。
    冲突红线（不变量 I2/I3）：本函数只降分不覆盖；深层覆盖走仲裁+版本链。
    """
    cp = params.confidence
    x = cp.gain * (
        cp.w_source_auth * node.source_auth
        + cp.w_consistency * node.consistency
        + cp.w_evidence * evidence_norm(evidence_count)
        + cp.w_freshness * max(0.0, min(1.0, freshness))
        - cp.w_conflict_penalty * max(0.0, conflict_penalty)
    )
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------- §五.6 检索打分

def retrieval_score(rel_norm: float, node: MemoryNode, now: datetime,
                    allowed_sensitivity: int = 3, params: Params = _PARAMS) -> float:
    """score = rel^α · weight^β · recency^γ · confidence^δ · importance^ε
              · tier_boost · sensitivity_gate。每因子 clip 到 [floor, 1]（不变量 I4）。O(1)。
    """
    sp = params.score
    if node.sensitivity > allowed_sensitivity:
        return 0.0
    rec = decay_recency(node, now, params)
    tier = node.tier.value if hasattr(node.tier, "value") else str(node.tier)

    def clip01(v: float) -> float:
        return min(1.0, max(sp.factor_floor, v))

    return (
        clip01(rel_norm) ** sp.alpha
        * clip01(node.weight) ** sp.beta
        * clip01(rec) ** sp.gamma
        * clip01(node.confidence) ** sp.delta
        * clip01(node.importance) ** sp.epsilon
        * sp.tier_boost.get(tier, 1.0)
    )


def rrf_fuse(node_id: str, rankings: dict, params: Params = _PARAMS) -> float:
    """多路 RRF 融合 + 绝对参考点归一（不变量 I4）。

    rankings: {channel: [node_id 按名次]}。融合值 Σ_ch w_ch/(k+rank_ch)，
    除以参考点 Σ_ch w_ch/(k+1)（= 全通道第一名），得到与查询无关的绝对刻度 rel∈[0,1]。
    """
    rp = params.retrieval
    fused = 0.0
    for channel, ids in rankings.items():
        w = rp.channel_weights.get(channel, 0.0)
        if w <= 0 or not ids:
            continue
        try:
            rank = ids.index(node_id) + 1
        except ValueError:
            continue
        fused += w / (rp.rrf_k + rank)
    denom = sum(w / (rp.rrf_k + 1) for w in rp.channel_weights.values() if w > 0)
    if denom <= 0:
        return 0.0
    return fused / denom


# ---------------------------------------------------------------- 相似度与预算

def cosine(a, b) -> float:
    """纯 Python 余弦相似度；零向量返回 0。"""
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (na * nb)


def estimate_tokens(text: str) -> int:
    """Token 估算：CJK 字符按 1 token/字，其余按 4 字符≈1 token。仅用于预算裁剪。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def mmr_select(candidates, k: int, lam: float):
    """MMR 多样性选择：score_mmr = λ·score − (1−λ)·max_sim_to_selected。O(N·K)。

    candidates: [(id, score, vector)]，按 score 降序贪心选取。
    """
    remaining = sorted(candidates, key=lambda c: c[1], reverse=True)
    selected: list = []
    while remaining and len(selected) < k:
        if not selected:
            selected.append(remaining.pop(0))
            continue
        best_i, best_val = 0, -math.inf
        for i, (cid, sc, vec) in enumerate(remaining):
            max_sim = max(cosine(vec, svec) for _, _, svec in selected)
            val = lam * sc - (1.0 - lam) * max_sim
            if val > best_val:
                best_i, best_val = i, val
        selected.append(remaining.pop(best_i))
    return selected


# ---------------------------------------------------------------- §五.3 层级流转

def tier_transition(node: MemoryNode, stats: dict | None = None, now: datetime | None = None,
                    params: Params = _PARAMS) -> str:
    """输出：promote_to_middle / promote_to_deep / needs_human_confirmation /
    archive / deprecate / keep。O(1)。

    浅→中：显式记住 ∨ 跨会话重复≥2 ∨ importance≥0.7。
    中→深：confidence≥0.9 ∧ source_auth≥0.8 ∧ 跨会话≥5；identity_policy 必须人工确认。
    归档：超过导出归档期限 ∧ weight<0.2 ∧ 非争议。
    """
    now = now or datetime.now(timezone.utc)
    t = params.tier
    if node.status == NodeStatus.contested:
        return "keep"  # 冲突未决：不归档不晋升（§二.4）
    if node.tier == Tier.shallow:
        if node.explicit_remember or node.cross_session_count >= t.promote_mid_repeat \
                or node.importance >= t.promote_mid_importance:
            return "promote_to_middle"
        return "keep"
    if node.tier == Tier.middle:
        if (node.confidence >= t.promote_deep_confidence
                and node.source_auth >= t.promote_deep_source_auth
                and node.cross_session_count >= t.promote_deep_sessions):
            if node.type == NodeType.identity_policy and not node.human_confirmed:
                return "needs_human_confirmation"
            return "promote_to_deep"
        stale_days = _days_between(now, node.last_access_at)
        if stale_days > archive_after_days(node, params) and node.weight < t.archive_weight_floor:
            return "archive"
        return "keep"
    if node.tier == Tier.deep:
        return "keep"  # 深层只经仲裁/人工废止（不变量 I2），不走自动降级
    return "keep"


# ---------------------------------------------------------------- §五.4 链路更新（纯函数部分）

def merge_edge_weight(w: float, delta: float) -> float:
    """饱和强化：w' = 1 − (1−w)(1−Δ)，同 (src,dst,type) 合并幂等（不变量 I5）。"""
    return min(1.0, 1.0 - (1.0 - w) * (1.0 - delta))


def decay_edge_weight(w: float, days: float, lambda_per_day: float | None = None,
                      params: Params = _PARAMS) -> float:
    """边权指数衰减：w·exp(−λ_e·days)。"""
    lam = lambda_per_day if lambda_per_day is not None else params.reinforce.lambda_edge_per_day
    return w * math.exp(-lam * max(0.0, days))


def should_prune_edge(weight: float, last_reinforced_at: datetime, confidence: float,
                      now: datetime, params: Params = _PARAMS) -> bool:
    """剪枝三条件与门：w<0.1 ∧ 超过 prune_days 未强化 ∧ confidence<0.3。"""
    t = params.tier
    days = _days_between(now, last_reinforced_at)
    return (weight < t.edge_prune_weight
            and days > t.edge_prune_days
            and confidence < t.edge_prune_confidence)


def weaken_contradiction_edge(edge: MemoryEdge) -> MemoryEdge:
    """冲突边保留并降权（§五.4：冲突是信息不是噪音）。"""
    edge.status = EdgeStatus.weakened
    edge.weight = edge.weight * 0.5
    return edge


def resolve_cycles(edges: list[MemoryEdge]) -> list[tuple[MemoryEdge, str]]:
    """Tarjan 强连通分量检测环路；每环去掉一条最弱边（weight 最小，平局取 edge_id 序），
    其余保留。无向边（contradicts）不参与环路。返回 [(edge, 'remove'|'keep')]。
    """
    directed = [e for e in edges if e.type != EdgeType.contradicts]
    adj: dict[str, list[MemoryEdge]] = {}
    nodes: set[str] = set()
    for e in directed:
        adj.setdefault(e.src_id, []).append(e)
        nodes.update((e.src_id, e.dst_id))

    index_counter = [0]
    stack: list[str] = []
    lowlink: dict[str, int] = {}
    index: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    sccs: list[list[str]] = []

    def strongconnect(v: str):
        index[v] = lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for e in adj.get(v, []):
            w = e.dst_id
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w, False):
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in sorted(nodes):
        if v not in index:
            strongconnect(v)

    cycle_nodes: set[str] = set()
    for comp in sccs:
        if len(comp) > 1:
            cycle_nodes.update(comp)
    to_remove: set[str] = set()
    for comp in sccs:
        if len(comp) <= 1:
            continue
        in_cycle = [e for e in directed if e.src_id in cycle_nodes and e.dst_id in cycle_nodes]
        if in_cycle:
            weakest = min(in_cycle, key=lambda e: (e.weight, e.edge_id))
            to_remove.add(weakest.edge_id)

    return [(e, "remove" if e.edge_id in to_remove else "keep") for e in edges]


# ---------------------------------------------------------------- §五.7 压缩与合并（纯函数部分）

def cluster_nodes(nodes: list[MemoryNode], threshold: float | None = None,
                  params: Params = _PARAMS) -> list[dict]:
    """按 (type, scope) 分组，组内按质心余弦 ≥ threshold 贪心聚类。

    generation ≥ max_summary_generation 的节点豁免（防级联压缩，评审 §五）。
    返回 [{"ids", "type", "scope", "centroid"}]，含单例簇；无向量节点自成单例。
    """
    th = threshold if threshold is not None else params.tier.cluster_threshold
    eligible = [n for n in nodes if n.generation < params.tier.max_summary_generation]
    groups: dict[tuple, list[MemoryNode]] = {}
    for n in eligible:
        groups.setdefault((n.type, n.scope), []).append(n)

    clusters: list[dict] = []
    for (ntype, scope), members in groups.items():
        local: list[dict] = []
        for n in members:
            placed = False
            if n.content_vector:
                for cl in local:
                    if cl["centroid"] and cosine(n.content_vector, cl["centroid"]) >= th:
                        cl["ids"].append(n.id)
                        cl["members"].append(n)
                        dim = len(cl["centroid"])
                        cl["centroid"] = tuple(
                            sum(m.content_vector[i] for m in cl["members"]) / len(cl["members"])
                            for i in range(dim)
                        )
                        placed = True
                        break
            if not placed:
                local.append({
                    "ids": [n.id], "members": [n],
                    "centroid": tuple(n.content_vector) if n.content_vector else (),
                    "type": ntype, "scope": scope,
                })
        for cl in local:
            clusters.append({
                "ids": cl["ids"], "type": ntype, "scope": scope,
                "centroid": cl["centroid"],
            })
    return clusters
