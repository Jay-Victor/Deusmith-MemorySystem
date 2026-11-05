# -*- coding: utf-8 -*-
"""demo.py —— 数值演算与边界自检（交付物 §9.3/9.4）。

用法（项目 venv）：
  PYTHONPATH=src .venv/Scripts/python.exe demo.py --selftest   # 边界断言，失败退出码 1
  PYTHONPATH=src .venv/Scripts/python.exe demo.py --demo       # 打印全部数值表（进 docs/04）

文档中的每个数字都由本脚本产出（评审错误#2：手算数字不进正文）。
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone

if __name__ == "__main__":
    sys.path.insert(0, "src")

from deusmith.algos import (  # noqa: E402
    archive_after_days, cosine, decay_recency, decay_stretch, estimate_confidence,
    estimate_tokens, merge_edge_weight, mmr_select, reinforce, retrieval_score,
    rrf_fuse, tau_eff_days,
)
from deusmith.embeddings import HashEmbedding  # noqa: E402
from deusmith.llm import RuleBasedLLM  # noqa: E402
from deusmith.models import MemoryNode, NodeType, Tier  # noqa: E402
from deusmith.params import Params  # noqa: E402
from deusmith.store import InMemoryStore  # noqa: E402

NOW = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)
P = Params()


def m102_node() -> MemoryNode:
    """文档 §9.3 的 m102：importance=0.8, emotion=0.5, confidence=0.9, activation=3, Δt=7天。"""
    return MemoryNode(
        type=NodeType.fact, content="m102 示例", scope="demo/u1", tier=Tier.middle,
        embedding_model="hash", importance=0.8, emotion=0.5, confidence=0.9,
        activation_count=3, last_access_at=NOW - timedelta(days=7), weight=0.6,
    )


def mk(**over) -> MemoryNode:
    kw = dict(type=NodeType.fact, content="边界", scope="demo/u1",
              tier=Tier.middle, embedding_model="hash")
    kw.update(over)
    return MemoryNode(**kw)


# ================================================================ selftest

def selftest() -> int:
    checks: list[tuple[str, bool, str]] = []

    def ck(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # ---- m102 数值钉死（±0.001）
    n = m102_node()
    S = decay_stretch(n, P)
    rec = decay_recency(n, NOW, P)
    score = retrieval_score(0.8, n, NOW, params=P)
    ck("m102: S=2.9359", abs(S - 2.9359) < 5e-4, f"S={S:.4f}")
    ck("m102: recency=0.7113", abs(rec - 0.7113) < 5e-4, f"recency={rec:.4f}")
    ck("m102: score=0.4082（v0.2.1 时效有界化后；原公式 0.3489）",
       abs(score - 0.4082) < 5e-4, f"score={score:.4f}")

    # ---- 衰减边界
    ck("Δt=0 → recency=1", decay_recency(mk(last_access_at=NOW), NOW, P) == 1.0)
    ck("Δt=3650天 → recency<0.05",
       decay_recency(mk(last_access_at=NOW - timedelta(days=3650)), NOW, P) < 0.05)
    ck("未来时间钳为 0 天（recency=1）",
       decay_recency(mk(last_access_at=NOW + timedelta(days=3)), NOW, P) == 1.0)
    cold = mk(importance=0.0, emotion=0.0, confidence=0.0, activation_count=0)
    ck("S=1 时归档期限=7·ln5=11.27天",
       abs(archive_after_days(cold, P) - 7.0 * math.log(5)) < 1e-9,
       f"{archive_after_days(cold, P):.2f}天")

    # ---- 打分边界
    zeroed = mk(weight=0.0, confidence=0.0, importance=0.0,
                last_access_at=NOW - timedelta(days=3650))
    ck("全零因子经 floor 后 score>0", retrieval_score(0.5, zeroed, NOW, params=P) > 0)
    restricted = mk(sensitivity=3)
    ck("sensitivity 超门控 → score=0",
       retrieval_score(0.9, restricted, NOW, allowed_sensitivity=2, params=P) == 0.0)

    # ---- MMR 边界
    ck("空候选集 → []", mmr_select([], k=3, lam=0.7) == [])
    ties = [(f"m{i}", 0.8, (1.0, 0.0)) for i in range(6)]
    ck("全同分同向量 → 仍返回 k 条（确定性退化）",
       len(mmr_select(ties, k=3, lam=0.7)) == 3)
    single = [("only", 0.9, (1.0, 0.0))]
    ck("单条候选 → 返回该条", mmr_select(single, k=3, lam=0.7) == single)

    # ---- RRF 绝对刻度
    rk = {"vector": ["a", "b"], "graph": ["a", "c"], "keyword": ["a"],
          "recent": ["a"], "explicit": ["a"]}
    ck("全通道第一 → rel=1.0", rrf_fuse("a", rk, P) == 1.0)
    single_ch = {"vector": ["x", "y"]}
    ck("单通道第一 → rel<1（不按当次最大值归一）", rrf_fuse("x", single_ch, P) < 1.0)
    ck("不在任何通道 → rel=0", rrf_fuse("zz", rk, P) == 0.0)

    # ---- 可信度
    ones = mk(source_auth=1.0, consistency=1.0)
    c_hi = estimate_confidence(ones, evidence_count=5, freshness=1.0, params=P)
    c_pen = estimate_confidence(ones, evidence_count=5, freshness=1.0,
                                conflict_penalty=0.5, params=P)
    ck("全1因子+增益 → ≥0.9（晋升可达）", c_hi >= 0.9, f"{c_hi:.4f}")
    ck("冲突惩罚 ↓ 置信度", c_pen < c_hi, f"{c_pen:.4f}")
    raw_ceiling = 1.0 / (1.0 + math.exp(-0.9))
    ck("无增益上限 0.711 < 0.9（文档原式的缺陷）", raw_ceiling < 0.9, f"{raw_ceiling:.4f}")

    # ---- 强化饱和
    m = mk(weight=0.5)
    reinforce(m, 1.0, NOW, P)
    ck("w=0.5,η=0.3,s=1 → 0.65", abs(m.weight - 0.65) < 1e-9, f"{m.weight:.4f}")
    w = 0.5
    for _ in range(1000):
        w = merge_edge_weight(w, 0.2)
    ck("边合并 1000 次 < 1（饱和不上溢）", w < 1.0, f"距 1 还差 {1 - w:.2e}")

    # ---- 相似度与 token
    ck("零向量 cosine=0", cosine((0, 0), (1, 2)) == 0.0)
    ck("CJK token=字数", estimate_tokens("用户偏好会议时间") == 8)
    ck("ASCII 4字符≈1token", estimate_tokens("hello world") == 3)
    ck("空串 token=0", estimate_tokens("") == 0)

    # ---- I1 守卫
    from deusmith.pipeline import DeusmithMemory
    try:
        DeusmithMemory(store=InMemoryStore(), embedder=HashEmbedding(dim=8),
                       scope="")
        ck("scope 必填守卫", False, "未抛错")
    except ValueError:
        ck("scope 必填守卫（I1）", True)

    fails = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))
    print(f"\n== selftest：{len(checks) - len(fails)}/{len(checks)} 通过 ==")
    return 1 if fails else 0


# ================================================================ demo

def demo() -> int:
    line = "-" * 78

    # ---- A. m102 数值演算（文档 §9.3，由代码复算）
    n = m102_node()
    S = decay_stretch(n, P)
    tau = tau_eff_days(n, P)
    rec = decay_recency(n, NOW, P)
    sp = P.score
    f_rel, f_w = 0.8 ** sp.alpha, n.weight ** sp.beta
    rec_factor = (1 - sp.recency_span) + sp.recency_span * rec ** sp.gamma
    f_c = n.confidence ** sp.delta
    f_i = n.importance ** sp.epsilon
    score = retrieval_score(0.8, n, NOW, params=P)
    print("A. m102 数值演算（Δt=7天, imp=0.8, emo=0.5, conf=0.9, act=3, w=0.6, rel=0.8）")
    print(f"   S = 1 + 0.8×0.8 + 0.5×0.5 + 0.7×0.9 + 0.3×ln(4) = {S:.4f}")
    print(f"   τ_eff = 7 × S = {tau:.4f} 天")
    print(f"   recency = exp(-7/τ_eff) = {rec:.4f}")
    print(f"   因子分解：rel^α={f_rel:.4f}  weight^β={f_w:.4f}  "
          f"rec_factor=(1-0.25)+0.25·rec^γ={rec_factor:.4f}  "
          f"conf^δ={f_c:.4f}  imp^ε={f_i:.4f}")
    print(f"   score = {score:.4f}")
    print(f"   【勘误】原文档写 ≈0.39 —— 原公式（无界 rec^γ）复算={f_rel * f_w * rec ** sp.gamma * f_c * f_i:.4f}；"
          f"v0.2.1 时效有界化后={score:.4f}")

    # ---- B. 衰减曲线
    print(f"\n{line}\nB. 遗忘衰减曲线 recency=exp(-Δt/(7·S))")
    cold = mk(importance=0.3, emotion=0.0, confidence=0.4, activation_count=0)
    hot = m102_node()
    print(f"   {'Δt(天)':>8} | {'冷节点(S=%.2f)' % decay_stretch(cold, P):>14} | "
          f"{'热节点(S=%.2f)' % decay_stretch(hot, P):>14}")
    for d in (0, 1, 3, 7, 14, 30, 90, 180, 365):
        r_c = decay_recency(mk(importance=0.3, emotion=0.0, confidence=0.4,
                               last_access_at=NOW - timedelta(days=d)), NOW, P)
        r_h = decay_recency(mk(importance=0.8, emotion=0.5, confidence=0.9,
                               activation_count=3,
                               last_access_at=NOW - timedelta(days=d)), NOW, P)
        print(f"   {d:>8} | {r_c:>14.4f} | {r_h:>14.4f}")

    # ---- C. 归档期限（导出公式）
    print(f"\n{line}\nC. 归档期限（导出量）t = -τ_eff·ln(0.2) —— 不与连续衰减双轨并存")
    for label, node in (("冷节点", cold), ("温节点(默认)", mk()),
                        ("m102 热节点", hot)):
        print(f"   {label:<10} S={decay_stretch(node, P):.2f}  τ_eff={tau_eff_days(node, P):6.2f}天  "
              f"归档期限={archive_after_days(node, P):6.2f}天  "
              f"(90天参考TTL={'超过' if archive_after_days(node, P) < 90 else '未到'})")

    # ---- D. 可信度
    print(f"\n{line}\nD. 可信度估计 c=sigmoid(k·(w1·auth+w2·cons+w3·ev+w4·fresh−w5·pen))")
    raw = 0.35 + 0.25 + 0.2 + 0.1
    print(f"   无增益（k=1）：sigmoid({raw:.2f}) = {1 / (1 + math.exp(-raw)):.4f}  ← 上限低于晋升阈值 0.9")
    ones = mk(source_auth=1.0, consistency=1.0)
    print(f"   增益 k=3，全1因子：sigmoid({3 * raw:.2f}) = "
          f"{estimate_confidence(ones, 5, 1.0, params=P):.4f}  ← 阈值可达")
    mid = mk(source_auth=0.8, consistency=0.8)
    print(f"   增益 k=3，0.8因子：{estimate_confidence(mid, 2, 0.8, params=P):.4f}  "
          f"(需跨会话强化才能过 0.9)")
    print(f"   同上 + 冲突惩罚0.5：{estimate_confidence(mid, 2, 0.8, 0.5, P):.4f}  ← 冲突显著降分")

    # ---- E. RRF 绝对刻度
    print(f"\n{line}\nE. 多路 RRF 融合（绝对参考点归一，不变量 I4）")
    rk = {"vector": ["m1", "m2", "m3"], "graph": ["m1", "m3"],
          "keyword": ["m1", "m2"], "recent": ["m1"], "explicit": ["m1"]}
    for nid in ("m1", "m2", "m3", "mX"):
        print(f"   rel({nid}) = {rrf_fuse(nid, rk, P):.4f}")
    print("   m1=全通道第一（绝对参考点 1.0）；mX 不在任何通道（0）；")
    print("   单通道第一 ≈ 0.26 —— 与查询无关的绝对刻度，阈值可跨查询移植")

    # ---- F. 端到端场景
    print(f"\n{line}\nF. 端到端场景（InMemoryStore + HashEmbedding + RuleBasedLLM）")
    from deusmith.offline import OfflineGovernor
    from deusmith.pipeline import DeusmithMemory
    emb = HashEmbedding(dim=128)
    store = InMemoryStore()
    mem = DeusmithMemory(store=store, embedder=emb, llm=RuleBasedLLM(),
                         params=P, scope="demo/user")
    s = mem.session("demo-s1")
    mem.write(s, "请记住：我对花粉过敏。")
    mem.write(s, "我喜欢摄影。")
    mem.write(s, "我现在住在上海。", source_auth=0.4)
    mem.write(s, "我现在住在北京。", source_auth=0.95)   # 高权威纠正 → supersedes 版本链
    mem.write(s, "我不再喜欢摄影了。")                     # 否定 → 冲突而非覆盖
    out = mem.inject_context(s, "用户的居住地和爱好")
    print(f"   [system] {out['system']!r}")
    print(f"   [user]   {out['user']!r}")
    print(f"   [tool]   {out['tool']!r}")
    print(f"   tokens={out['tokens']}  degraded={out['degraded']}  "
          f"conflicts={len(out['conflict_flags'])}")
    rep = OfflineGovernor(store, emb, RuleBasedLLM(), P).run("demo/user")
    print(f"   离线治理报告：{rep}")
    counts = store.counts("demo/user")
    print(f"   库内统计：{counts}")

    print(f"\n{line}\nG. 边界情况（与 --selftest 同源，摘要）")
    print("   t=0→recency=1；Δt=10年→0.0017；全零因子→floor(0.01) 兜底；")
    print("   空候选→[]；全同分→确定性取 k 条；单通道第一→0.26 而非 1.0；")
    print("   敏感超门控→score=0；边合并1000次→饱和<1；scope缺失→抛错(I1)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    if "--demo" in sys.argv:
        raise SystemExit(demo())
    print(__doc__)
    raise SystemExit(0)
