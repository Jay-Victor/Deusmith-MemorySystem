# -*- coding: utf-8 -*-
"""Embedding Provider —— 三种实现，同一接口（用户前提：本地模型或 API 均可）。

  HashEmbedding          零依赖离线默认（特征哈希，CJK 一元+二元、拉丁词元）
  OpenAICompatibleEmbedding  OpenAI 兼容 /embeddings API（urllib，stdlib）
  LocalModelEmbedding     sentence-transformers 本地模型（可选依赖，缺则 ImportError）

注意：HashEmbedding 是占位级词法嵌入，只保证"相似>不相关"的排序性，
检索质量评估请换真实模型。它是让全链路离线可跑的保底，不是终点。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request

from .algos import cosine  # noqa: F401  (re-export)

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

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class OpenAICompatibleEmbedding:
    """OpenAI 兼容 /embeddings API。api_key 参数优先，其次 DEUSMITH_EMBEDDING_API_KEY。"""

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 dim: int = DEFAULT_DIM, timeout: float = 30.0):
        if not base_url or not model:
            raise ValueError("openai provider 需要 base_url 与 model")
        self.base_url = base_url.rstrip("/")
        self.endpoint = self.base_url + "/embeddings"
        self.model_name = model
        self.api_key = api_key or os.environ.get("DEUSMITH_EMBEDDING_API_KEY", "")
        self.dim = dim
        self.timeout = timeout

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _vec_from(data: dict, i: int) -> tuple:
        for item in data["data"]:
            if item["index"] == i:
                return tuple(float(x) for x in item["embedding"])
        raise ValueError("embedding 响应缺少 index")

    def embed(self, text: str) -> tuple:
        return self._vec_from(self._post({
            "model": self.model_name, "input": [text], "dimensions": self.dim,
        }), 0)

    def embed_batch(self, texts):
        data = self._post({"model": self.model_name, "input": list(texts),
                           "dimensions": self.dim})
        return [self._vec_from(data, i) for i in range(len(texts))]


class LocalModelEmbedding:
    """本地 sentence-transformers 模型（可选依赖）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "local provider 需要 sentence-transformers："
                "pip install sentence-transformers") from e
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> tuple:
        return tuple(float(x) for x in self._model.encode([text])[0])

    def embed_batch(self, texts):
        return [tuple(float(x) for x in v) for v in self._model.encode(list(texts))]


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
        )
    if provider == "local":
        return LocalModelEmbedding(model_name=emb_cfg.get("model_name", "BAAI/bge-small-zh-v1.5"))
    raise ValueError(f"未知 embedding provider: {provider}")
