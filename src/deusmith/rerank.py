# -*- coding: utf-8 -*-
"""Rerank Provider —— 检索精排（召回之后的第二阶段重排序）。

管线位置：多路召回 → RRF 绝对刻度融合 → 打分 →【精排（本模块）】→ MMR 裁剪。
动机：双塔/单塔嵌入的 query-doc 相似度分不清"表面相似"与"查询相关"
（cos 高只说明话题近，不说明回答了这个查询）；cross-encoder 把
query 和 doc 拼接过同一个 transformer，是对"相关"的直接度量。

三种实现，同一接口（用户前提：本地模型或 API 均可，自行下载配置）：
  IdentityReranker           默认空实现（rerank 返回 None → 管线跳过精排，零开销）
  LocalCrossEncoderReranker  sentence-transformers CrossEncoder 本地重排模型
                             （BAAI/bge-reranker-base、bge-reranker-v2-m3 等，
                              model_name 支持本地目录或 HF ID，惰性加载）
  OpenAICompatibleReranker   OpenAI 兼容 /rerank API（Jina / Cohere /
                             SiliconFlow 风格：{results:[{index,relevance_score}]}）

分数约定：rerank(query, documents) -> list[float] | None
  - 返回与 documents 等长的相关性分数，每项 ∈ [0,1]；
  - 绝对刻度（0.9=强相关，与查询措辞无关），与 RRF rel 同域，可线性混合；
  - 返回 None 表示该实现不打分（IdentityReranker），管线保持原排序；
  - 抛异常 → pipeline 捕获，降级回 RRF 排序并记 rerank_fallback 事件
    （与 LLM 失败降级规则引擎同款语义，评审 #6）。
"""
from __future__ import annotations

import math
import os

from .retryhttp import post_json_with_retry


class BaseReranker:
    """接口基类：model_name 供 trace 留痕；rerank 默认不打分。"""

    model_name = "identity"

    def rerank(self, query: str, documents: list[str]):
        """返回 [0,1] 分数列表（与 documents 等长）；None = 不参与精排。"""
        return None


class IdentityReranker(BaseReranker):
    """空重排（默认）：管线行为与未启用 rerank 完全一致。"""


class LocalCrossEncoderReranker(BaseReranker):
    """本地 cross-encoder 重排（可选依赖 sentence-transformers，惰性加载）。

    score = CrossEncoder.predict([[query, doc]])。bge-reranker 系单 logit
    输出经默认 sigmoid 激活已 ∈[0,1]；若模型返回未激活 logit（越界），
    统一补一层 sigmoid（单调变换，排序不变但拉回 [0,1] 绝对刻度）。
    模型在首次 rerank 时才加载（构造便宜，provider 切换不触碰网络/磁盘）。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base",
                 batch_size: int = 16, max_length: int = 512):
        self._model = None
        self._model_name = model_name
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(64, int(max_length))
        self.stats = {"requests": 0, "items": 0}

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers.cross_encoder import CrossEncoder  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "local rerank 需要 sentence-transformers："
                    "pip install sentence-transformers（并自行下载重排模型，"
                    "如 BAAI/bge-reranker-base）") from e
            self._model = CrossEncoder(self._model_name, max_length=self.max_length)
        return self._model

    def rerank(self, query: str, documents: list[str]):
        docs = [d or "" for d in (documents or [])]
        if not docs:
            return []
        model = self._ensure_model()
        pairs = [[query or "", d] for d in docs]
        out = model.predict(pairs, batch_size=self.batch_size)
        scores = [float(x) for x in out]
        self.stats["requests"] += 1
        self.stats["items"] += len(docs)
        if any(s < 0.0 or s > 1.0 for s in scores):   # logit 形态 → 补 sigmoid
            scores = [1.0 / (1.0 + math.exp(-s)) for s in scores]
        return [min(1.0, max(0.0, s)) for s in scores]


class OpenAICompatibleReranker(BaseReranker):
    """OpenAI 兼容 /rerank 端点（Jina / Cohere / SiliconFlow 风格）。

    请求：POST {base_url}/rerank
          {"model", "query", "documents": [str...], "top_n": len(documents)}
    响应：{"results": [{"index": int, "relevance_score": float}, ...]}
    健壮性与 embeddings 同款（评审 #6）：5xx/429/网络错误指数退避重试；
    按 batch_limit 分批（部分服务对单次 documents 数有上限）；
    stats 计数 requests/retries/errors/items。
    api_key 参数优先，其次 DEUSMITH_RERANK_API_KEY 环境变量。
    """

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout: float = 30.0, retry_attempts: int = 3,
                 retry_backoff: float = 0.5, batch_limit: int = 64):
        if not base_url or not model:
            raise ValueError("openai rerank provider 需要 base_url 与 model")
        self.base_url = base_url.rstrip("/")
        self.endpoint = self.base_url + "/rerank"
        self.model = model
        self.model_name = model
        self.api_key = api_key or os.environ.get("DEUSMITH_RERANK_API_KEY", "")
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff
        self.batch_limit = max(1, int(batch_limit))
        self.stats = {"requests": 0, "retries": 0, "errors": 0, "items": 0}

    def _headers(self) -> dict:
        return {"Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"}

    def rerank(self, query: str, documents: list[str]):
        docs = [d or "" for d in (documents or [])]
        if not docs:
            return []
        out = [0.0] * len(docs)
        for i in range(0, len(docs), self.batch_limit):
            chunk = docs[i:i + self.batch_limit]
            data = post_json_with_retry(
                self.endpoint,
                {"model": self.model, "query": query or "",
                 "documents": chunk, "top_n": len(chunk)},
                self._headers(), self.timeout,
                attempts=self.retry_attempts, backoff=self.retry_backoff,
                stats=self.stats)
            for r in data.get("results") or []:
                j = int(r.get("index", -1))
                if 0 <= j < len(chunk) and r.get("relevance_score") is not None:
                    out[i + j] = min(1.0, max(0.0, float(r["relevance_score"])))
            self.stats["items"] += len(chunk)
        return out


def get_reranker(cfg: dict):
    """cfg 形如 {"rerank": {"provider": "none"|"local"|"openai", ...}}。

    未启用（默认）返回 None：pipeline 的 reranker=None 分支零开销，
    检索行为与历史版本逐位一致（向后兼容，不破坏既有探针门禁）。
    """
    rr_cfg = (cfg or {}).get("rerank", {}) or {}
    provider = rr_cfg.get("provider", "none")
    if provider in ("none", "identity", ""):
        return None
    if provider == "local":
        return LocalCrossEncoderReranker(
            model_name=rr_cfg.get("model_name", "BAAI/bge-reranker-base"),
            batch_size=int(rr_cfg.get("batch_size", 16)),
            max_length=int(rr_cfg.get("max_length", 512)),
        )
    if provider == "openai":
        return OpenAICompatibleReranker(
            base_url=rr_cfg.get("base_url", ""),
            model=rr_cfg.get("model", ""),
            api_key=rr_cfg.get("api_key"),
            timeout=float(rr_cfg.get("timeout", 30.0)),
            retry_attempts=int(rr_cfg.get("retry_attempts", 3)),
            retry_backoff=float(rr_cfg.get("retry_backoff", 0.5)),
            batch_limit=int(rr_cfg.get("batch_limit", 64)),
        )
    raise ValueError(f"未知 rerank provider: {provider}")
