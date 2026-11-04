# -*- coding: utf-8 -*-
"""浅层（工作）记忆 —— §二.1。

定位：当前会话工作记忆（最近对话原文、工具结果、滚动摘要、任务状态）。
存储：进程内（评审修改：单机形态不默认 Redis；Redis 为可选演进适配器）。
容量：最近 N=10 轮原文（滚动窗口）；超出部分进滚动摘要。
TTL：会话结束 30 分钟后原文可清理（由 DeusmithMemory.cleanup_expired_sessions 执行）；
     滚动摘要保留至会话结束并进入中层候选。
"""
from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .algos import estimate_tokens
from .models import utcnow
from .params import Params


@dataclass
class Round:
    role: str          # user / assistant / tool
    content: str
    ts: datetime = field(default_factory=utcnow)


class ShallowSession:
    """一个会话的浅层工作记忆。"""

    def __init__(self, scope: str, session_id: str | None, params: Params, llm=None):
        self.scope = scope
        self.id = session_id or ("sess_" + uuid.uuid4().hex[:12])
        self.params = params
        self.llm = llm
        self.rounds: deque[Round] = deque(maxlen=params.tier.shallow_window_rounds)
        self.tool_results: deque[str] = deque(maxlen=20)
        self.rolling_summary: str = ""
        self.last_node_id: str | None = None      # 本会话最近写入的节点（temporal_next 用）
        self.node_ids: list[str] = []             # 本会话写入的全部节点
        self.created_at = utcnow()
        self.last_active_at = utcnow()
        self.closed_at: datetime | None = None

    # ---- 写入
    def add_round(self, role: str, content: str) -> None:
        self.rounds.append(Round(role=role, content=content))
        self.last_active_at = utcnow()

    def add_tool_result(self, content: str) -> None:
        self.tool_results.append(content)
        self.last_active_at = utcnow()

    def maybe_update_summary(self) -> str:
        """滚动摘要：窗口滚满且尚无摘要时生成一次；此后每滚满 N 轮刷新。
        LLM 不可用时 llm.summarize 为规则式（保底可用）。"""
        n = self.params.tier.shallow_window_rounds
        if len(self.rounds) >= n and self.llm is not None:
            refreshed = self.llm.summarize([r.content for r in self.rounds],
                                           max_tokens=256)
            if refreshed:
                self.rolling_summary = refreshed
        return self.rolling_summary

    # ---- 读取
    def to_text(self, token_budget: int) -> str:
        """浅层投影：滚动摘要 + 最近 N 轮原文，总量受 token 预算约束（I7）。
        超预算时从最老的轮次开始丢弃。"""
        parts: list[str] = []
        if self.rolling_summary:
            parts.append(f"[会话摘要] {self.rolling_summary}")
        parts.extend(f"{r.role}: {r.content}" for r in self.rounds)
        text = "\n".join(parts)
        while estimate_tokens(text) > token_budget and len(parts) > 1:
            # 先裁最老轮次（保摘要与最近轮）
            drop_idx = 1 if parts and parts[0].startswith("[会话摘要]") else 0
            parts.pop(drop_idx)
            text = "\n".join(parts)
        return text

    def close(self) -> None:
        self.closed_at = utcnow()

    def expiry_at(self) -> datetime | None:
        """会话结束后 30 分钟，原文可清理。"""
        if self.closed_at is None:
            return None
        return self.closed_at + timedelta(
            minutes=self.params.tier.shallow_cleanup_delay_min)
