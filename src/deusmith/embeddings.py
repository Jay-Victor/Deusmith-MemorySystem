# -*- coding: utf-8 -*-
"""Embedding Provider —— 三种实现，同一接口（用户前提：本地模型或 API 均可）。

  HashEmbedding          零依赖离线默认（特征哈希，CJK 一元+二元、拉丁词元）
  OpenAICompatibleEmbedding  OpenAI 兼容 /embeddings API（重试/退避/批量/成本计数）
  LocalModelEmbedding     sentence-transformers 本地模型（可选依赖，缺则 ImportError）

注意：HashEmbedding 是占位级词法嵌入，只保证"相似>不相关"的排序性，
检索质量评估请换真实模型（eval_probes.py --embed local）。
查询侧统一走 embed_query()：bge 类模型需要查询指令前缀，文档侧不加。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request

from .algos import cosine  # noqa: F401  (re-export)
from .retryhttp import post_json_with_retry

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+")

DEFAULT_DIM = 256


def _tokenize(text: str):
    """产出 (token, weight)：CJK 一元(0.5)+二元(1.0)；拉丁小写词(1.0)。"""
    out = []
    for m in _CJK_RUN.finditer(text or ""):
        s = m.group(0)
        if "\u4e00" <= s[0] <= "\u9fff":
            for ch in s:
                out.append((ch, 0.5))
            for i in range(len(s) - 1):
                out.append((s[i:i + 2], 1.0))
        else:
            out.append((s.lower(), 1.0))
    return out


def _hash_token(token: str, dim: int):
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    idx = int.from_bytes(h[:4], "little") % dim
    sign = 1 if (h[4] & 1) == 0 else -1
    return idx, sign


class HashEmbedding:
    """签名特征哈希嵌入：确定性、无依赖、L2 归一。dim 为 2 的幂效果更好。"""

    def __init__(self, dim: int = DEFAULT_DIM, model_tag: str = "hash"):
        if dim <= 8:
            raise ValueError("dim 至少 8")
        self.dim = dim
        self.model_name = f"{model_tag}-{dim}"

    def embed(self, text: str) -> tuple:
        vec = [0.0] * self.dim
        for tok, w in _tokenize(text):
            idx, sign = _hash_token(tok, self.dim)
            vec[idx] += sign * w
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return tuple(vec)

    def embed_query(self, text: str) -> tuple:
        """查询侧嵌入（哈希嵌入与文档侧一致）。"""
        return self.embed(text)

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class OpenAICompatibleEmbedding:
    """OpenAI 兼容 /embeddings API。

    健壮性（评审 #6）：5xx/429/网络错误指数退避重试；embed_batch 按
    batch_limit 分批；stats 计数 requests/retries/errors/items。
    api_key 参数优先，其次 DEUSMITH_EMBEDDING_API_KEY。
    """

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 dim: int = DEFAULT_DIM, timeout: float = 30.0,
                 retry_attempts: int = 3, retry_backoff: float = 0.5,
                 batch_limit: int = 64):
        if not base_url or not model:
            raise ValueError("openai provider 需要 base_url 与 model")
        self.base_url = base_url.rstrip("/")
        self.endpoint = self.base_url + "/embeddings"
        self.model_name = model
        self.api_key = api_key or os.environ.get("DEUSMITH_EMBEDDING_API_KEY", "")
        self.dim = dim
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff
        self.batch_limit = max(1, int(batch_limit))
        self.stats = {"requests": 0, "retries": 0, "errors": 0, "items": 0}

    def _headers(self) -> dict:
        return {"Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"}

    def _post(self, payload: dict) -> dict:
        return post_json_with_retry(
            self.endpoint, payload, self._headers(), self.timeout,
            attempts=self.retry_attempts, backoff=self.retry_backoff,
            stats=self.stats)

    def stats_snapshot(self) -> dict:
        return dict(self.stats)

    @staticmethod
    def _vec_from(data: dict, i: int) -> tuple:
        for item in data["data"]:
            if item["index"] == i:
                return tuple(float(x) for x in item["embedding"])
        raise ValueError("embedding 响应缺少 index")

    def embed(self, text: str) -> tuple:
        data = self._post({"model": self.model_name, "input": [text],
                           "dimensions": self.dim})
        self.stats["items"] += 1
        return self._vec_from(data, 0)

    def embed_query(self, text: str) -> tuple:
        return self.embed(text)

    def embed_batch(self, texts):
        texts = list(texts)
        out: list[tuple] = []
        for i in range(0, len(texts), self.batch_limit):
            chunk = texts[i:i + self.batch_limit]
            data = self._post({"model": self.model_name, "input": chunk,
                               "dimensions": self.dim})
            out.extend(self._vec_from(data, j) for j in range(len(chunk)))
            self.stats["items"] += len(chunk)
        return out


class LocalModelEmbedding:
    """本地 sentence-transformers 模型（可选依赖，惰性加载）。

    query_prefix：bge 系检索模型查询侧指令（如
    "为这个句子生成表示以用于检索相关文章："），文档侧不加。
    模型在首次 embed 时才加载（构造便宜，provider 选择不触碰网络/磁盘）。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5",
                 query_prefix: str = ""):
        self._model = None
        self._model_name = model_name
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.dim = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "local provider 需要 sentence-transformers："
                    "pip install sentence-transformers") from e
            self._model = SentenceTransformer(self._model_name)
            # sentence-transformers ≥6 改名 get_embedding_dimension，旧版兼容
            get_dim = (getattr(self._model, "get_embedding_dimension", None)
                       or self._model.get_sentence_embedding_dimension)
            self.dim = get_dim()
        return self._model

    def embed(self, text: str) -> tuple:
        model = self._ensure_model()
        return tuple(float(x) for x in model.encode([text])[0])

    def embed_query(self, text: str) -> tuple:
        model = self._ensure_model()
        return tuple(float(x) for x in
                     model.encode([self.query_prefix + text])[0])

    def embed_batch(self, texts):
        model = self._ensure_model()
        return [tuple(float(x) for x in v) for v in model.encode(list(texts))]


def get_provider(cfg: dict):
    """cfg 形如 {"embedding": {"provider": "hash", "dim": 256, ...}}。默认 hash。"""
    emb_cfg = (cfg or {}).get("embedding", {}) or {}
    provider = emb_cfg.get("provider", "hash")
    if provider == "hash":
        return HashEmbedding(dim=int(emb_cfg.get("dim", DEFAULT_DIM)))
    if provider == "openai":
        return OpenAICompatibleEmbedding(
            base_url=emb_cfg.get("base_url", "https://api.openai.com/v1"),
            model=emb_cfg.get("model", ""),
            api_key=emb_cfg.get("api_key"),
            dim=int(emb_cfg.get("dim", DEFAULT_DIM)),
            retry_attempts=int(emb_cfg.get("retry_attempts", 3)),
            retry_backoff=float(emb_cfg.get("retry_backoff", 0.5)),
            batch_limit=int(emb_cfg.get("batch_limit", 64)),
        )
    if provider == "local":
        return LocalModelEmbedding(
            model_name=emb_cfg.get("model_name", "BAAI/bge-small-zh-v1.5"),
            query_prefix=emb_cfg.get("query_prefix", ""),
        )
    raise ValueError(f"未知 embedding provider: {provider}")
