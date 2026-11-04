# -*- coding: utf-8 -*-
"""LLM 挂钩 —— RuleBasedLLM 为离线保底，OpenAICompatibleLLM 为可选在线实现。

职责（写读路径中的三处语义环节）：
  extract(text)     写入侧：从原文抽取记忆候选（类型/槽位/极性/重要性/情绪）
  summarize(texts)  离线与浅层：滚动摘要、压缩合并的摘要生成
  arbitrate(a, b)   冲突仲裁：prefer_a / prefer_b / contested

红线（不变量 I2/I3）：
  - arbitrate 对深层记忆只返回 contested（不允许静默覆盖深层记忆），
    深层变更必须走人工确认 + 版本链 + 审计；
  - extract 的否定词处理是错误#1 的写入侧防线：『不再喜欢』必须产出 polarity=-1。
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from .algos import estimate_tokens
from .models import MemoryNode

_SENT_SPLIT = re.compile(r"[。！？；;\n]+")
_EMOTION_WORDS = {
    "开心": 0.8, "高兴": 0.8, "兴奋": 0.9, "激动": 0.8, "满意": 0.6,
    "难过": 0.8, "伤心": 0.8, "生气": 0.9, "愤怒": 0.9, "沮丧": 0.7,
    "焦虑": 0.7, "害怕": 0.7, "累": 0.4, "烦": 0.6,
}
_NEG_PREF = re.compile(
    r"我(?:现在)?(?:不再|不|很|特别|比较|有点|超)?(喜欢|讨厌|爱|恨|偏爱|钟爱)([\u4e00-\u9fffA-Za-z0-9]+)"
)
_RESIDENCE = re.compile(r"我(?:现在)?(?:住在|位于|就职于|工作在)([\u4e00-\u9fffA-Za-z0-9]+)")
_IDENTITY = re.compile(r"我(?:叫|是)([\u4e00-\u9fffA-Za-z0-9]{1,20})")
_ALLERGY = re.compile(r"我对(.{1,12}?)过敏")
_SLOT_FACT = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9]{1,10})的([\u4e00-\u9fff]{1,8})是([\u4e00-\u9fffA-Za-z0-9]{1,20})"
)
_EXPLICIT = re.compile(r"^(?:请|帮我|麻烦)?(?:你)?记住[:：]?\s*")
_POS_VERBS = {"喜欢", "爱", "偏爱", "钟爱"}
_TRAILING = "了的地得啊吧呢哦呀，。, "


def _strip_trailing(s: str) -> str:
    return s.rstrip(_TRAILING).strip()


def _emotion_of(sentence: str) -> float:
    best = 0.0
    for w, v in _EMOTION_WORDS.items():
        if w in sentence:
            best = max(best, v)
    return best


class RuleBasedLLM:
    """规则引擎：中文优先的正则抽取 + 截断式摘要 + 权威/置信仲裁。全离线、确定性。"""

    def extract(self, text: str) -> list[dict]:
        items: list[dict] = []
        if not text or not text.strip():
            return items
        for sent in _SENT_SPLIT.split(text):
            sent = sent.strip()
            if not sent:
                continue
            explicit = bool(_EXPLICIT.match(sent))
            if explicit:
                sent = _EXPLICIT.sub("", sent).strip()
                if not sent:
                    continue
            emotion = _emotion_of(sent)
            item = self._match_one(sent, explicit, emotion)
            if item:
                items.append(item)
        return items

    def _match_one(self, sent: str, explicit: bool, emotion: float) -> dict | None:
        # 偏好（含否定）
        m = _NEG_PREF.search(sent)
        if m:
            verb, obj = m.group(1), _strip_trailing(m.group(2))
            neg = re.search(r"(不再|不)", sent[:m.start(1)] or "")
            base_pol = 1 if verb in _POS_VERBS else -1
            polarity = -base_pol if neg else base_pol
            return dict(
                type="preference", content=sent, subject="user",
                predicate=verb, object_value=obj, polarity=polarity,
                importance=0.75 + (0.15 if explicit else 0.0),
                emotion=emotion, explicit_remember=explicit, sensitivity=1,
            )
        # 身份
        m = _IDENTITY.search(sent)
        if m:
            return dict(
                type="identity_policy", content=sent, subject="user",
                predicate="叫" if sent[m.start():m.start() + 2] == "我叫" else "是",
                object_value=_strip_trailing(m.group(1)), polarity=0,
                importance=0.85 + (0.05 if explicit else 0.0),
                emotion=emotion, explicit_remember=explicit, sensitivity=2,
            )
        # 居住/任职事实
        m = _RESIDENCE.search(sent)
        if m:
            return dict(
                type="fact", content=sent, subject="user",
                predicate="住在" if "住" in sent[m.start():m.start() + 4] else "位于",
                object_value=_strip_trailing(m.group(1)), polarity=0,
                importance=0.6, emotion=emotion,
                explicit_remember=explicit, sensitivity=1,
            )
        # 过敏等敏感事实
        m = _ALLERGY.search(sent)
        if m:
            return dict(
                type="fact", content=sent, subject="user",
                predicate="过敏", object_value=_strip_trailing(m.group(1)), polarity=0,
                importance=0.8 + (0.1 if explicit else 0.0),
                emotion=emotion, explicit_remember=explicit, sensitivity=2,
            )
        # 通用槽位事实：『X的Y是Z』（第三人称/复合主语的事实与身份类陈述）
        m = _SLOT_FACT.search(sent)
        if m:
            return dict(
                type="fact", content=sent, subject=m.group(1), predicate=m.group(2),
                object_value=_strip_trailing(m.group(3)), polarity=0,
                importance=0.6 + (0.2 if explicit else 0.0),
                emotion=emotion, explicit_remember=explicit, sensitivity=1,
            )
        # 兜底：事件
        return dict(
            type="event", content=sent, subject=None, predicate=None,
            object_value=None, polarity=0,
            importance=0.4 + (0.3 if explicit else 0.0),
            emotion=emotion, explicit_remember=explicit, sensitivity=1,
        )

    def summarize(self, texts, max_tokens: int = 512) -> str:
        seen, parts = set(), []
        for t in texts:
            t = (t or "").strip()
            key = t.rstrip(_TRAILING)
            if t and key not in seen:
                seen.add(key)
                parts.append(t)
        if not parts:
            return ""
        summary = "；".join(parts)
        # 截断到 token 预算（粗粒度，保头不保尾）
        while parts and estimate_tokens("；".join(parts)) > max_tokens:
            parts.pop()
            summary = "；".join(parts)
        return "摘要：" + summary if summary else ""

    def arbitrate(self, a: MemoryNode, b: MemoryNode) -> str:
        """a=既有记忆，b=挑战者。深层参与 → 永远 contested（红线）。
        否则按 0.6·source_auth + 0.4·confidence 打分（来源权威优先，§五.5），
        差距 < 0.15 视为势均力敌，保留双版本待确认。"""
        from .models import Tier
        if a.tier == Tier.deep or b.tier == Tier.deep:
            return "contested"
        sa = 0.6 * a.source_auth + 0.4 * a.confidence
        sb = 0.6 * b.source_auth + 0.4 * b.confidence
        if abs(sa - sb) < 0.15:
            return "contested"
        return "prefer_a" if sa > sb else "prefer_b"


class OpenAICompatibleLLM:
    """OpenAI 兼容 /chat/completions。任何错误抛给调用方，由 pipeline 决定是否降级到规则。"""

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout: float = 60.0, temperature: float = 0.2):
        if not base_url or not model:
            raise ValueError("openai llm 需要 base_url 与 model")
        self.base_url = base_url.rstrip("/")
        self.endpoint = self.base_url + "/chat/completions"
        self.model = model
        self.api_key = api_key or os.environ.get("DEUSMITH_LLM_API_KEY", "")
        self.timeout = timeout
        self.temperature = temperature

    def _build_messages(self, system: str, user: str) -> list[dict]:
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    def chat(self, system: str, user: str) -> str:
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps({
                "model": self.model,
                "messages": self._build_messages(system, user),
                "temperature": self.temperature,
            }).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    # ---- 三个语义环节的 LLM 版本（prompt 契约与 RuleBasedLLM 输出一致）----

    _EXTRACT_SYSTEM = (
        "你是记忆抽取器。从用户消息中抽取记忆条目，输出 JSON 数组，每项含字段："
        "type(event/fact/preference/emotion/identity_policy)、content(原文子句)、"
        "subject、predicate、object_value、polarity(1/-1/0)、importance(0-1)、"
        "emotion(0-1)、sensitivity(0-3)、explicit_remember(bool)。只输出 JSON。"
    )

    def extract(self, text: str) -> list[dict]:
        raw = self.chat(self._EXTRACT_SYSTEM, text)
        try:
            start, end = raw.find("["), raw.rfind("]")
            items = json.loads(raw[start:end + 1])
            return [i for i in items if isinstance(i, dict) and i.get("content")]
        except (ValueError, TypeError):
            return []

    def summarize(self, texts, max_tokens: int = 512) -> str:
        raw = self.chat(
            "你是摘要器。把多条记忆合并为一条不超过 %d token 的中文摘要，保留偏好、事实与时间线索。" % max_tokens,
            "\n".join(texts),
        )
        return raw.strip()

    def arbitrate(self, a: MemoryNode, b: MemoryNode) -> str:
        raw = self.chat(
            "你是记忆冲突仲裁器。A=既有记忆，B=挑战记忆。输出 prefer_a / prefer_b / contested 三者之一。"
            "身份、价值观、安全策略类深层记忆一律 contested。只输出判定词。",
            f"A: {a.content} (tier={a.tier.value}, auth={a.source_auth}, conf={a.confidence})\n"
            f"B: {b.content} (tier={b.tier.value}, auth={b.source_auth}, conf={b.confidence})",
        )
        verdict = raw.strip().lower()
        for k in ("prefer_a", "prefer_b", "contested"):
            if k in verdict:
                return k
        return "contested"


def get_llm(cfg: dict):
    """cfg 形如 {"llm": {"provider": "rule"|"openai", ...}}。默认 rule（离线保底）。"""
    llm_cfg = (cfg or {}).get("llm", {}) or {}
    provider = llm_cfg.get("provider", "rule")
    if provider == "rule":
        return RuleBasedLLM()
    if provider == "openai":
        return OpenAICompatibleLLM(
            base_url=llm_cfg.get("base_url", ""),
            model=llm_cfg.get("model", ""),
            api_key=llm_cfg.get("api_key"),
            timeout=float(llm_cfg.get("timeout", 60.0)),
        )
    raise ValueError(f"未知 llm provider: {provider}")
