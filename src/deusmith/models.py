# -*- coding: utf-8 -*-
"""节点与边数据模型 —— 对应《Deusmith-Memory》§三 + 评审补充的槽位字段。

单表 + 类型判别（不变量 I9）：图游走/PPR 需要同构 ID 空间，不做一类型一表。
槽位 (subject/predicate/object_value/polarity) 是冲突判定的前提（评审错误#1）：
  冲突 = (scope, subject, predicate) 相同 而 极性相反 或 宾语不同；
  语义相似度不参与冲突判定。
"""
import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum

_WS = re.compile(r"\s+")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_content(text: str) -> str:
    """归一化：去全部空白后取小写，用于内容 hash。"""
    return _WS.sub("", text or "").lower()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_content(text).encode("utf-8")).hexdigest()


def new_node_id() -> str:
    return "m" + uuid.uuid4().hex


def new_edge_id() -> str:
    return "e" + uuid.uuid4().hex


class NodeType(str, Enum):
    event = "event"
    fact = "fact"
    preference = "preference"
    emotion = "emotion"
    summary = "summary"
    task = "task"
    tool_result = "tool_result"
    identity_policy = "identity_policy"
    conflict = "conflict"


class NodeStatus(str, Enum):
    active = "active"
    archived = "archived"
    contested = "contested"
    deprecated = "deprecated"
    deleted_logical = "deleted_logical"


class Tier(str, Enum):
    shallow = "shallow"
    middle = "middle"
    deep = "deep"


class EdgeType(str, Enum):
    semantic_similar = "semantic_similar"
    temporal_next = "temporal_next"
    causal = "causal"
    co_occur = "co_occur"
    coref = "coref"
    supports = "supports"
    contradicts = "contradicts"
    derives_from = "derives_from"
    summary_of = "summary_of"


class EdgeStatus(str, Enum):
    active = "active"
    weakened = "weakened"      # 冲突边保留并降权（§五.4）
    pruned = "pruned"


class Sensitivity(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3


def _check_unit(v: float, name: str) -> None:
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"{name} 必须在 [0,1]，得到 {v}")


@dataclass
class MemoryNode:
    # 身份与内容
    id: str = field(default_factory=new_node_id)
    type: NodeType = NodeType.fact
    content: str = ""
    scope: str = ""                      # tenant/user（不变量 I1：必填）
    tier: Tier = Tier.middle
    content_vector: tuple = ()           # 惰性填充；tuple 保证不可变快照
    embedding_model: str = ""
    # 槽位（冲突判定前提）
    subject: str | None = None
    predicate: str | None = None
    object_value: str | None = None
    polarity: int = 0                    # +1 肯定 / -1 否定 / 0 中性
    # 时间
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    last_access_at: datetime = field(default_factory=utcnow)
    # 来源与可信
    source: str = "user"
    source_auth: float = 0.5
    confidence: float = 0.5
    evidence_refs: tuple = ()
    consistency: float = 0.5
    # 权重与使用
    weight: float = 0.5
    activation_count: int = 0
    importance: float = 0.5
    emotion: float = 0.0                 # 0..1 情绪强度（归一）
    # 治理
    sensitivity: int = 1                 # Sensitivity.INTERNAL
    status: NodeStatus = NodeStatus.active
    version: int = 1
    supersedes: str | None = None        # 指向被本节点替代的旧版本
    contested_by: str | None = None
    content_hash: str = ""
    # 流转统计
    session_id: str | None = None
    cross_session_count: int = 0
    explicit_remember: bool = False
    human_confirmed: bool = False
    # 摘要谱系（compress_merge）
    generation: int = 0                  # 0=原文，1=一次摘要；>max_summary_generation 豁免聚类
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.scope or not self.scope.strip():
            raise ValueError("scope 必填（不变量 I1：多租户隔离）")
        if self.type == NodeType.identity_policy and self.tier == Tier.deep and not self.human_confirmed:
            # 深层身份/策略必须人工确认（§二.3 硬门）——直接构造也拦截
            raise ValueError("identity_policy 进入 deep 层必须 human_confirmed=True")
        _check_unit(self.weight, "weight")
        _check_unit(self.confidence, "confidence")
        _check_unit(self.importance, "importance")
        _check_unit(self.emotion, "emotion")
        _check_unit(self.source_auth, "source_auth")
        _check_unit(self.consistency, "consistency")
        if not (0 <= self.sensitivity <= 3):
            raise ValueError(f"sensitivity 必须在 0..3，得到 {self.sensitivity}")
        if self.polarity not in (-1, 0, 1):
            raise ValueError(f"polarity 必须是 -1/0/1，得到 {self.polarity}")
        if self.activation_count < 0 or self.cross_session_count < 0:
            raise ValueError("计数字段不能为负")
        if self.version < 1:
            raise ValueError("version 从 1 起")
        if not self.content_hash:
            self.content_hash = content_hash(self.content)
        if isinstance(self.type, str):
            self.type = NodeType(self.type)
        if isinstance(self.tier, str):
            self.tier = Tier(self.tier)
        if isinstance(self.status, str):
            self.status = NodeStatus(self.status)
        self.content_vector = tuple(self.content_vector)
        self.evidence_refs = tuple(self.evidence_refs)

    def slot_key(self) -> tuple:
        """槽位身份：(scope, subject, predicate)。无槽位时含 content_hash 兜底。"""
        if self.subject is None or self.predicate is None:
            return (self.scope, None, self.content_hash)
        return (self.scope, self.subject, self.predicate)


@dataclass
class MemoryEdge:
    edge_id: str = field(default_factory=new_edge_id)
    src_id: str = ""
    dst_id: str = ""
    type: EdgeType = EdgeType.co_occur
    direction: str = ""                  # directed / undirected
    weight: float = 0.5
    confidence: float = 0.5
    created_at: datetime = field(default_factory=utcnow)
    last_reinforced_at: datetime = field(default_factory=utcnow)
    evidence_refs: tuple = ()
    status: EdgeStatus = EdgeStatus.active

    def __post_init__(self):
        if not self.src_id or not self.dst_id:
            raise ValueError("src_id/dst_id 必填")
        if self.src_id == self.dst_id:
            raise ValueError("自环边不允许")
        _check_unit(self.weight, "edge.weight")
        _check_unit(self.confidence, "edge.confidence")
        if isinstance(self.type, str):
            self.type = EdgeType(self.type)
        if isinstance(self.status, str):
            self.status = EdgeStatus(self.status)
        if not self.direction:
            # 方向约定（§三.2）：冲突无向，其余有向
            self.direction = "undirected" if self.type == EdgeType.contradicts else "directed"
        self.evidence_refs = tuple(self.evidence_refs)
