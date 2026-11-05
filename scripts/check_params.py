# -*- coding: utf-8 -*-
"""check_params.py —— 参数表 ↔ 代码默认值一致性校验 + 不变式。

  1. 解析 docs/03-params.yaml（受限扁平点号格式，无需 PyYAML）；
  2. 与 src/deusmith/params.py 的 dataclass 默认值逐项比对（漂移即失败）；
  3. 断言参数不变式（权重区间、tier_boost 序、tunable 维度上限等）。

用法：.venv/Scripts/python.exe scripts/check_params.py
"""
from __future__ import annotations

import sys
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deusmith.params import MAX_TUNABLE_DIMS, TUNABLE_DIMS, Params  # noqa: E402

PARAMS_YAML = ROOT / "docs" / "03-params.yaml"

PASS, FAIL = 0, 0


def check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}" + (f" —— {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" —— {detail}" if detail else ""))


def parse_flat_yaml(path: Path) -> dict:
    """解析受限格式：`a.b.c: value`，# 注释，值为 int/float/str。"""
    out: dict = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{lineno} 缺少冒号: {line!r}")
        key, _, raw = line.partition(":")
        key = key.strip()
        raw = raw.split("#", 1)[0].strip()
        try:
            val = int(raw)
        except ValueError:
            try:
                val = float(raw)
            except ValueError:
                val = raw
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out


def dataclass_defaults(cls) -> dict:
    """嵌套 dataclass → 嵌套 dict（只取默认值）。"""
    out = {}
    for f in fields(cls):
        if f.default_factory is not MISSING:
            default = f.default_factory()
        elif f.default is not MISSING:
            default = f.default
        else:
            continue
        if is_dataclass(default):
            out[f.name] = dataclass_defaults(type(default))
        elif isinstance(default, dict):
            out[f.name] = dict(default)
        else:
            out[f.name] = default
    return out


def flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten(v, key + "."))
        else:
            flat[key] = v
    return flat


def diff(a: dict, b: dict) -> list[str]:
    fa, fb = flatten(a), flatten(b)
    issues = []
    for k in sorted(set(fa) | set(fb)):
        va, vb = fa.get(k, "<缺失>"), fb.get(k, "<缺失>")
        if isinstance(va, float) or isinstance(vb, float):
            try:
                equal = abs(float(va) - float(vb)) < 1e-9
            except (TypeError, ValueError):
                equal = False
        else:
            equal = va == vb
        if not equal:
            issues.append(f"{k}: yaml={va} code={vb}")
    return issues


def main() -> int:
    print("== Deusmith-MemorySystem · 参数一致性校验 ==")
    yaml_cfg = parse_flat_yaml(PARAMS_YAML)
    code_cfg = dataclass_defaults(Params)

    issues = diff(yaml_cfg, code_cfg)
    check("docs/03-params.yaml ↔ params.py 默认值逐项一致", not issues,
          "; ".join(issues[:5]) if issues else f"{len(flatten(yaml_cfg))} 项全部匹配")

    p = Params()

    # ---- 不变式
    check("tunable 维度 ≤ 上限且不重复",
          len(TUNABLE_DIMS) == len(set(TUNABLE_DIMS)) <= MAX_TUNABLE_DIMS,
          f"{len(TUNABLE_DIMS)}/{MAX_TUNABLE_DIMS}: {', '.join(TUNABLE_DIMS)}")

    c = p.confidence
    ws = [c.w_source_auth, c.w_consistency, c.w_evidence, c.w_freshness,
          c.w_conflict_penalty]
    check("confidence 权重均在 [0,1] 且惩罚项为正",
          all(0 <= w <= 1 for w in ws) and c.w_conflict_penalty > 0)
    check("sigmoid 增益使深层晋升阈值可达（全1因子 ≥ 0.9）",
          c.gain * (c.w_source_auth + c.w_consistency + c.w_evidence
                    + c.w_freshness) >= 2.197,
          f"gain={c.gain}，输入上限={c.gain * 0.9:.2f}（需 ≥2.197 才能到 0.9）")

    check("打分指数全部非负", all(v >= 0 for v in
          (p.score.alpha, p.score.beta, p.score.gamma, p.score.delta, p.score.epsilon)))
    check("factor_floor > 0（I4 防零）", p.score.factor_floor > 0)
    check("MMR λ∈[0,1]", 0 <= p.score.mmr_lambda <= 1)
    check("recency_span∈(0,0.5)（时效有界弱化）",
          0 < p.score.recency_span < 0.5)
    check("通道权重全部非负", all(w >= 0 for w in p.retrieval.channel_weights.values()))

    t = p.tier
    check("晋升阈值有序：中→深 conf/auth 高于浅→中 importance",
          t.promote_deep_confidence > t.promote_mid_importance
          and t.promote_deep_source_auth >= 0.5)
    check("剪枝三条件阈值处于单位区间",
          0 < t.edge_prune_weight < 1 and t.edge_prune_days > 0
          and 0 < t.edge_prune_confidence < 1)
    check("浅层窗口预算占比 ≤ 1（I7）", 0 < t.shallow_context_share <= 1)
    check("摘要级联深度 ≥ 1", t.max_summary_generation >= 1)

    total = PASS + FAIL
    print(f"\n== 结果：{PASS}/{total} 项通过 ==")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
