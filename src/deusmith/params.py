# -*- coding: utf-8 -*-
"""参数体系 —— 单一事实源。

对应《Deusmith-Memory》§四 参数体系与耦合关系。
评审结论：
  - decay_rate 不作为节点存储字段（可由本模块导出），节点表只存可选的 ttl 覆盖值；
  - tunable 维度收敛为 7 个（TUNABLE_DIMS），其余为结构常数；
  - 本模块默认值与 docs/03-params.yaml 由 scripts/check_params.py 保证一致。

耦合关系（不变量 I6，由 tests/test_algos.py::test_tau_eff_monotonic 钉死）：
  activation_count↑ → weight↑ → retrieval_score↑
  last_access_at 越近 → recency↑ → retrieval_score↑
  importance/emotion/confidence↑ → S↑ → τ_eff↑ → 有效 TTL↑
  source_auth+consistency+evidence↑ → confidence↑ → weight↑
  conflict_penalty↑ → confidence↓，节点标记 contested
  sensitivity↑ → 注入权限与脱敏约束↑，候选可能被过滤
"""
from dataclasses import dataclass, field

MAX_TUNABLE_DIMS = 7

# 评审 §四：探针集 < 300 条时自由参数必然过拟合，收敛为 7 个独立维度。
TUNABLE_DIMS = (
    "decay.tau_base_days",      # 衰减时间基准
    "score.exponents",          # 打分指数组 (alpha,beta,gamma,delta,epsilon)
    "reinforce.etas",           # 强化学习率组 (eta, eta_edge)
    "score.mmr_lambda",         # 多样性-相关性折衷
    "retrieval.recall",         # 召回组 (topk, hops, budget)
    "tier.promotion",           # 晋升阈值组
    "tier.cluster_threshold",   # 压缩聚类阈值
)


@dataclass(frozen=True)
class DecayParams:
    """遗忘衰减（Ebbinghaus 指数式，在线 O(1)；Act-R 幂律校准在演进路径）。"""
    tau_base_days: float = 7.0
    k_importance: float = 0.8
    k_emotion: float = 0.5
    k_confidence: float = 0.7
    k_activation: float = 0.3


@dataclass(frozen=True)
class ReinforceParams:
    eta: float = 0.3                  # 节点强化学习率
    eta_edge: float = 0.2             # 边强化学习率
    lambda_edge_per_day: float = 0.01  # 边日衰减率


@dataclass(frozen=True)
class ConfidenceParams:
    """可信度五因子（sigmoid 组合；权重和=1.2，非 softmax 不必归一）。

    gain=3.0 为评审修正：sigmoid 增益。无增益时输入上限 0.9 → sigmoid 上限 0.711，
    深层晋升阈值 0.9 永远不可达（数值演算见 docs/04）。
    """
    w_source_auth: float = 0.35
    w_consistency: float = 0.25
    w_evidence: float = 0.2
    w_freshness: float = 0.1
    w_conflict_penalty: float = 0.3
    gain: float = 3.0


@dataclass(frozen=True)
class ScoreParams:
    alpha: float = 1.0    # rel_norm 指数
    beta: float = 0.8     # weight 指数
    gamma: float = 0.6    # recency 指数
    delta: float = 1.0    # confidence 指数
    epsilon: float = 0.5  # importance 指数
    factor_floor: float = 0.01   # 每因子下限，防零（不变量 I4）
    mmr_lambda: float = 0.7
    tier_boost: dict = field(default_factory=lambda: {
        "shallow": 0.8, "middle": 1.0, "deep": 1.15,
    })
    min_score_threshold: float = 0.45  # 绝对下限：低于此视为"没记住"


@dataclass(frozen=True)
class RetrievalParams:
    topk_recall: int = 50
    graph_hops: int = 2
    ppr_damping: float = 0.85
    knn_edge_topk: int = 20
    token_budget: int = 4000
    rrf_k: int = 60
    # 多路融合通道权重（向量/图/关键词/最近访问/显式引用），和不必为 1（RRF 归一在分母）
    channel_weights: dict = field(default_factory=lambda: {
        "vector": 1.0, "graph": 0.8, "keyword": 0.6, "recent": 0.5, "explicit": 1.0,
    })
    # 绝对参考点：全部通道均排第一时的 RRF 融合值（归一分母，不变量 I4）
    # max_fused = sum(w_ch) / (rrf_k + 1)


@dataclass(frozen=True)
class TierParams:
    shallow_window_rounds: int = 10        # 最近 N 轮原文
    shallow_context_share: float = 0.30    # 浅层 ≤ 上下文窗口 30%
    shallow_cleanup_delay_min: float = 30.0
    promote_mid_repeat: int = 2            # 跨会话重复 ≥2 → 晋升中层
    promote_mid_importance: float = 0.7
    promote_deep_confidence: float = 0.9
    promote_deep_source_auth: float = 0.8
    promote_deep_sessions: int = 5
    middle_reference_ttl_days: int = 90    # 监控参考值；权威归档期限由导出公式计算
    archive_score_floor: float = 0.2       # θ_archive：recency 低于此进入归档评估
    archive_weight_floor: float = 0.2
    edge_prune_weight: float = 0.1
    edge_prune_days: int = 60
    edge_prune_confidence: float = 0.3
    cluster_threshold: float = 0.9
    max_summary_generation: int = 2        # 摘要级联深度上限（评审 §五）
    summary_max_tokens: int = 512


@dataclass(frozen=True)
class Params:
    decay: DecayParams = field(default_factory=DecayParams)
    reinforce: ReinforceParams = field(default_factory=ReinforceParams)
    confidence: ConfidenceParams = field(default_factory=ConfidenceParams)
    score: ScoreParams = field(default_factory=ScoreParams)
    retrieval: RetrievalParams = field(default_factory=RetrievalParams)
    tier: TierParams = field(default_factory=TierParams)
