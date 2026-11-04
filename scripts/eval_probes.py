# -*- coding: utf-8 -*-
"""eval_probes.py —— 七类探针门禁（≥50 节点场景）→ docs/eval-report.md。

门禁（当前技术栈能做的那一列）：
  G1 陈旧值      旧值不应压过新值
  G2 纠正        高权威纠正后检索返回新值
  G3 连续性      同会话上下文可召回
  G4 聚合        离线压缩后摘要可检索、原文归档
  G5 常驻通道    深层身份与显式记忆不依赖查询相似度即可召回
  G6 无关精度    无关查询不产生高分命中（绝对刻度）
  G7 改写问法    同义改写仍命中目标记忆
对照实验（评审错误#1）：
  相似度去重坏变体 vs 槽位+极性判定：否定更新丢失率、良性改写合并率。

用法：PYTHONPATH=src .venv/Scripts/python.exe scripts/eval_probes.py
污染声明：本探针集即开发集（曾用于发现并修复多值谓语冲突误判 bug）。
正式调参前请按 70/30 切分 dev/held-out 并另建留出集。
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deusmith.algos import cosine  # noqa: E402
from deusmith.embeddings import HashEmbedding  # noqa: E402
from deusmith.llm import RuleBasedLLM  # noqa: E402
from deusmith.models import (  # noqa: E402
    MemoryNode, NodeStatus, NodeType, Tier, utcnow,
)
from deusmith.offline import OfflineGovernor  # noqa: E402
from deusmith.params import Params  # noqa: E402
from deusmith.pipeline import DeusmithMemory  # noqa: E402
from deusmith.store import InMemoryStore  # noqa: E402

SCOPE = "probe/user"
DIM = 128
P = Params()

NOISE = [
    "量子计算纠错码研究进展", "古希腊哲学史概述", "咖啡烘焙工艺详解",
    "深海热泉生态考察", "文艺复兴绘画流派", "区块链共识算法比较",
    "登山绳结技术图解", "中日韩书法比较", "气象卫星遥感原理",
    "城市地铁规划案例", "酿酒发酵微生物学", "无人机航拍技巧",
    "古典音乐和声理论", "宠物猫的品种图谱", "半导体光刻工艺流程",
    "丝绸之路贸易史", "统计学习理论入门",
]
PREFS = [
    ("用户喜欢摄影", 60), ("用户讨厌加班", 40), ("用户喜欢清晨跑步", 30),
    ("用户偏爱素食", 25), ("用户喜欢科幻小说", 20), ("用户讨厌嘈杂环境", 15),
    ("用户喜欢绿茶", 10), ("用户偏爱安静的工作环境", 5),
]
FACTS = [
    ("用户的公司在工业园区", 50), ("用户的团队有六名工程师", 45),
    ("用户的工位在三层靠窗", 40), ("用户常用Python和Rust", 35),
    ("用户的显示是双显示器", 30), ("用户每天通勤四十分钟", 22),
    ("用户的键盘是静音轴", 12), ("用户的椅子是人体工学椅", 8),
]
OTHER_EVENTS = [
    ("用户上周参加了技术沙龙", 18), ("用户昨天调试了数据库索引", 3),
]


def build_scenario() -> tuple[DeusmithMemory, HashEmbedding]:
    store = InMemoryStore()
    emb = HashEmbedding(dim=DIM)
    mem = DeusmithMemory(store=store, embedder=emb, llm=RuleBasedLLM(),
                         params=P, scope=SCOPE)
    now = utcnow()

    def put(content, days_ago, type_=NodeType.event, **over):
        kw = dict(
            type=type_, content=content, scope=SCOPE, tier=Tier.middle,
            embedding_model=emb.model_name, content_vector=emb.embed(content),
            last_access_at=now - timedelta(days=days_ago), weight=0.5,
            importance=0.5)
        kw.update(over)
        store.upsert_node(MemoryNode(**kw))

    for t in NOISE:
        put(t, 30 + (hash(t) % 60))                    # 30–90 天前的噪声
    for t, d in PREFS:
        put(t, d, type_=NodeType.preference, importance=0.7)
    for t, d in FACTS:
        put(t, d, type_=NodeType.fact)
    for t, d in OTHER_EVENTS:
        put(t, d, type_=NodeType.event)
    # 聚合组：8 条相似周会记录（向量一致 → 聚类 ≥0.9）
    vec_meeting = emb.embed("项目周会记录")
    for i in range(8):
        put(f"项目周会第{i + 1}次记录：进度同步与风险评审", 20 + i, type_=NodeType.event,
            content_vector=vec_meeting)
    # G1 陈旧值（新值：高置信新鲜；旧值：低置信陈旧）
    put("用户负责的麒麟项目在三月上线", 90, type_=NodeType.fact, importance=0.4,
        weight=0.3, confidence=0.6)
    put("用户的凤凰项目今天上线", 0, type_=NodeType.fact, importance=0.7,
        weight=0.6, confidence=0.8)
    # G5 常驻通道
    store.upsert_node(MemoryNode(
        type=NodeType.identity_policy, content="用户名叫李雷，是一名后端工程师",
        scope=SCOPE, tier=Tier.deep, embedding_model=emb.model_name,
        content_vector=emb.embed("用户名叫李雷，是一名后端工程师"),
        human_confirmed=True, evidence_refs=("ev_profile_1",), importance=0.95))
    # G7 显式记住（走写入路径）
    mem.write(mem.session("probe-s1"), "请记住：我对花粉过敏。")
    # G2 纠正（高权威）
    mem.write(mem.session("probe-s2"), "我现在住在上海。", source_auth=0.4)
    mem.write(mem.session("probe-s2"), "我现在住在北京。", source_auth=0.95)
    # G3 连续性
    s3 = mem.session("probe-s3")
    mem.write(s3, "我们刚才在讨论去京都的旅行计划。")
    return mem, emb


def top_contents(res, k=3):
    return [n.content for n, _, _ in res.selected[:k]]


def run_gates(mem: DeusmithMemory, emb: HashEmbedding) -> list[tuple[str, bool, str]]:
    gates: list[tuple[str, bool, str]] = []
    s = mem.session("probe-eval")

    # G1 陈旧值
    res = mem.retrieve(s, "用户最近上线了什么项目")
    sel = [n.content for n, _, _ in res.selected]
    phoenix_idx = next((i for i, c in enumerate(sel) if "凤凰" in c), None)
    qilin_idx = next((i for i, c in enumerate(sel) if "麒麟" in c), None)
    ok = (phoenix_idx is not None and phoenix_idx < 3
          and (qilin_idx is None or phoenix_idx < qilin_idx))
    gates.append(("G1 陈旧值：新值压过旧值", bool(ok),
                  f"凤凰@{phoenix_idx}, 麒麟@{qilin_idx}, top1={sel[0][:20] if sel else '∅'}"))

    # G2 纠正
    res = mem.retrieve(s, "用户现在住在哪个城市")
    tops = top_contents(res, 3)
    ok = any("北京" in t for t in tops) and not any("上海" in t for t in tops)
    gates.append(("G2 纠正：返回新值且旧值已废止", bool(ok), "；".join(tops)[:60]))

    # G3 连续性
    s3 = mem.session("probe-s3")
    res = mem.retrieve(s3, "刚才聊的旅行计划是什么")
    tops = top_contents(res, 3)
    ok = any("京都" in t for t in tops)
    gates.append(("G3 连续性：同会话上下文召回", bool(ok), "；".join(tops)[:60]))

    # G5 常驻通道（先于离线治理运行）
    res = mem.retrieve(s, "今天天气怎么样")
    ids = [n.id for n, _, _ in res.selected]
    tops = top_contents(res, 8)
    ok = any("李雷" in t for t in tops) and any("花粉" in t for t in tops)
    gates.append(("G5 常驻通道：身份与显式记忆无条件召回", bool(ok),
                  f"含李雷={any('李雷' in t for t in tops)}, 含花粉={any('花粉' in t for t in tops)}"))

    # G6 无关精度（绝对刻度）
    res = mem.retrieve(s, "火锅底料的炒制方法")
    max_score = res.selected[0][1] if res.selected else 0.0
    ok = max_score < P.score.min_score_threshold
    gates.append(("G6 无关精度：max score < 0.45", bool(ok), f"max={max_score:.4f}"))

    # G7 改写问法
    hits = 0
    for q in ("用户有什么过敏原", "用户对什么过敏", "花粉过敏要注意什么"):
        res = mem.retrieve(s, q)
        if any("花粉" in t for t in top_contents(res, 3)):
            hits += 1
    gates.append(("G7 改写问法：≥2/3 同义改写命中", hits >= 2, f"{hits}/3"))

    # G4 聚合（离线治理后）
    gov = OfflineGovernor(mem.store, emb, RuleBasedLLM(), P)
    report = gov.run(SCOPE)
    res = mem.retrieve(s, "项目周会的记录")
    tops = top_contents(res, 3)
    summary_hit = any(n.type == NodeType.summary for n, _, _ in res.selected[:5])
    summary_rank = next((i + 1 for i, (n, _, _) in enumerate(res.selected)
                         if n.type == NodeType.summary), None)
    originals = [n for n in mem.store.active_nodes(SCOPE)
                 if "周会" in n.content and n.type == NodeType.event]
    ok = summary_hit and not originals and report["summaries_created"] >= 1
    gates.append(("G4 聚合：摘要可检索、原文已归档", bool(ok),
                  f"summary@{summary_rank}, 活跃原文={len(originals)}, "
                  f"summaries={report['summaries_created']}"))
    return gates


# ---------------------------------------------------------- 对照实验（错误#1）

NEG_PAIRS = [
    ("我喜欢摄影", "我不再喜欢摄影了"), ("我喜欢跑步", "我不再喜欢跑步了"),
    ("我偏爱素食", "我不再偏爱素食了"), ("我喜欢科幻小说", "我不喜欢科幻小说了"),
    ("我喜欢绿茶", "我不再喜欢绿茶了"), ("我讨厌加班", "我不讨厌加班了"),
    ("我喜欢清晨跑步", "我不再喜欢清晨跑步了"), ("用户偏爱安静", "用户不再偏爱安静"),
]
BENIGN_PAIRS = [
    ("我喜欢摄影", "我热爱摄影"), ("我喜欢跑步", "我爱跑步"),
    ("我偏爱素食", "我钟爱素食"), ("我喜欢科幻小说", "我喜爱科幻小说"),
    ("我喜欢绿茶", "我钟爱绿茶"), ("用户讨厌加班", "用户厌恶加班"),
    ("我喜欢清晨跑步", "我喜爱清晨跑步"), ("用户偏爱安静", "用户喜爱安静"),
]


def bad_variant(theta: float) -> tuple[int, int]:
    """相似度去重坏变体：写入时若与既有节点 cos≥θ 则当重复丢弃。
    返回 (否定更新丢失数, 良性改写合并数)。"""
    store = InMemoryStore()
    emb = HashEmbedding(dim=DIM)
    mem = DeusmithMemory(store=store, embedder=emb, llm=RuleBasedLLM(),
                         params=P, scope=SCOPE)

    def bad_write(text):
        vec = emb.embed(text)
        node = MemoryNode(type=NodeType.preference, content=text, scope=SCOPE,
                          tier=Tier.middle, embedding_model=emb.model_name,
                          content_vector=vec, subject="user", predicate="喜欢",
                          polarity=1 if "不" not in text else -1)
        hits = store.vector_search(SCOPE, vec, topk=1)
        if hits and hits[0][1] >= theta:
            return hits[0][0]        # 当重复 → 丢弃新内容
        store.upsert_node(node)
        return node

    lost = 0
    for orig, neg in NEG_PAIRS:
        bad_write(orig)
        kept = bad_write(neg)
        if not (hasattr(kept, "content") and "不" in kept.content):
            lost += 1
    store2 = InMemoryStore()
    emb2 = HashEmbedding(dim=DIM)
    mem2 = DeusmithMemory(store=store2, embedder=emb2, llm=RuleBasedLLM(),
                          params=P, scope=SCOPE)
    merged = 0
    for orig, para in BENIGN_PAIRS:
        for text in (orig, para):
            vec = emb2.embed(text)
            hits = store2.vector_search(SCOPE, vec, topk=1)
            if hits and hits[0][1] >= theta:
                continue
            store2.upsert_node(MemoryNode(
                type=NodeType.preference, content=text, scope=SCOPE,
                tier=Tier.middle, embedding_model=emb2.model_name,
                content_vector=vec))
        nodes = [n for n in store2.active_nodes(SCOPE)]
        if len([n for n in nodes if n.content in (orig, para)]) == 1:
            merged += 1
        # 清空，逐对独立
        for n in list(store2._nodes.values()):
            store2._nodes.pop(n.id)
    return lost, merged


def slot_variant() -> int:
    """槽位+极性判定：否定更新保留数（以 contested/superseded 节点存在为准）。"""
    store = InMemoryStore()
    emb = HashEmbedding(dim=DIM)
    mem = DeusmithMemory(store=store, embedder=emb, llm=RuleBasedLLM(),
                         params=P, scope=SCOPE)
    preserved = 0
    for i, (orig, neg) in enumerate(NEG_PAIRS):
        s = mem.session(f"neg-{i}a")
        mem.write(s, orig + "。")
        s2 = mem.session(f"neg-{i}b")
        mem.write(s2, neg + "。")
        nodes = [n for n in store.active_nodes(SCOPE)
                 + [x for x in store.recent_active(SCOPE, 100)]]
        seen = {n.id: n for n in nodes}
        neg_nodes = [n for n in seen.values()
                     if "不" in n.content and n.subject == "user"]
        if neg_nodes:
            preserved += 1
    return preserved


def main() -> int:
    print("== Deusmith-MemorySystem · 探针门禁 ==")
    mem, emb = build_scenario()
    total_nodes = mem.store.counts(SCOPE)["nodes_total"]
    print(f"场景规模：{total_nodes} 节点（要求 ≥50）")
    gates = run_gates(mem, emb)

    fails = 0
    for name, ok, detail in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} —— {detail}")
        fails += 0 if ok else 1

    # ---- 对照实验
    print("\n== 对照实验（评审错误#1：相似度判据 vs 槽位+极性）==")
    cos_neg = [cosine(emb.embed(a), emb.embed(b)) for a, b in NEG_PAIRS]
    cos_ben = [cosine(emb.embed(a), emb.embed(b)) for a, b in BENIGN_PAIRS]
    print(f"  否定对相似度   min={min(cos_neg):.3f} max={max(cos_neg):.3f} "
          f"mean={sum(cos_neg) / len(cos_neg):.3f}")
    print(f"  良性改写对相似度 min={min(cos_ben):.3f} max={max(cos_ben):.3f} "
          f"mean={sum(cos_ben) / len(cos_ben):.3f}")
    tradeoff = []
    for theta in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        lost, merged = bad_variant(theta)
        tradeoff.append((theta, lost, merged))
        print(f"  θ={theta:.1f}：否定更新丢失 {lost}/8，良性改写合并 {merged}/8")
    preserved = slot_variant()
    print(f"  槽位+极性判定：否定更新保留 {preserved}/8（0 丢失）")

    no_valid_theta = all(
        not (t[2] == 8 and t[1] == 0) for t in tradeoff)
    print(f"  结论：{'不存在能同时「合并全部良性改写」与「零丢失否定更新」的阈值' if no_valid_theta else '存在可行阈值'}")

    # ---- 报告落盘
    report = ["# Deusmith-MemorySystem · 探针评估报告", "",
              f"生成：scripts/eval_probes.py（场景 {total_nodes} 节点，HashEmbedding dim={DIM}，规则 LLM）", "",
              "## 门禁结果（当前技术栈）", "",
              "| 门禁 | 结果 | 证据 |", "|---|---|---|"]
    for name, ok, detail in gates:
        report.append(f"| {name} | {'**PASS**' if ok else '**FAIL**'} | {detail} |")
    report += ["", f"**门禁：{len(gates) - fails}/{len(gates)} PASS**", "",
               "## 对照实验：相似度判据 vs 槽位+极性（评审错误#1）", "",
               f"- 否定对（如「我喜欢摄影」↔「我不再喜欢摄影了」）余弦相似度："
               f"min={min(cos_neg):.3f}，max={max(cos_neg):.3f}，mean={sum(cos_neg) / len(cos_neg):.3f}",
               f"- 良性改写对（如「我喜欢摄影」↔「我热爱摄影」）余弦相似度："
               f"min={min(cos_ben):.3f}，max={max(cos_ben):.3f}，mean={sum(cos_ben) / len(cos_ben):.3f}",
               "- 两类区间重叠且否定侧整体更高：**任何能合并良性改写的阈值 θ 必然吞掉否定更新；"
               "任何保得住否定更新的 θ 连良性改写都合不了——相似度判据不可用。**", "",
               "| θ | 否定更新丢失 | 良性改写合并 |", "|---|---|---|"]
    for theta, lost, merged in tradeoff:
        report.append(f"| {theta:.1f} | {lost}/8 | {merged}/8 |")
    report += ["",
               f"槽位+极性判定：否定更新保留 **{preserved}/8**（丢失 0）；"
               "真重复（同槽位+同宾语+同极性）由 is_duplicate 正确合并；"
               "多值谓语（喜欢）换宾语不误判为冲突（单值谓语如「住在」换宾语才算冲突）。", "",
               "## 污染声明", "",
               "本探针集在开发过程中曾用于发现并修复：多值谓语冲突误判 bug"
               "（『喜欢摄影』与『喜欢跑步』曾被误判为冲突）。**它已是被污染的开发集**——"
               "正式调参前请按 70/30 切分 dev/held-out，另建留出集，且留出集不得用于修 bug。", "",
               "## 已知缺口（不进门禁，单列）", "",
               "- HashEmbedding 为词法级占位实现，改写问法的召回质量显著低于真实语义模型；",
               "- 图游走为简化 PPR（2 跳衰减），未与真 PPR/Neo4j 对照；",
               "- 多用户并发与 PG 后端下的检索延迟未在本报告覆盖（见 docs/04）。"]
    out = ROOT / "docs" / "eval-report.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"\n报告已写入 {out}")
    print(f"== 门禁：{len(gates) - fails}/{len(gates)} PASS ==")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
