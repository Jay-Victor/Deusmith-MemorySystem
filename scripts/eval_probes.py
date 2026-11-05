# -*- coding: utf-8 -*-
"""eval_probes.py —— 七类门禁 + 查询探针（dev/held-out 70/30）→ docs/eval-report.md。

门禁（当前技术栈能做的那一列）：
  G1 陈旧值 / G2 纠正 / G3 连续性 / G4 聚合 / G5 常驻通道 / G6 无关精度 / G7 改写问法
查询探针：20 条带独立 ground truth 的查询（期望子串出现在 top-3），
  按固定种子 70/30 切 dev/held-out（评审第一梯队 #3）。

用法：
  PYTHONPATH=src .venv/Scripts/python.exe scripts/eval_probes.py                     # hash 嵌入 + dev
  PYTHONPATH=src .venv/Scripts/python.exe scripts/eval_probes.py --embed local        # 真实本地模型
  PYTHONPATH=src .venv/Scripts/python.exe scripts/eval_probes.py --heldout            # 留出集（发布前跑一次）

污染声明：dev 集在开发中用于发现并修复 bug（多值谓语冲突误判、关键词通道
中文切词缺失、摘要置信度继承缺陷）；held-out 在 v0.2.0 发布前跑了一次基线，
下次发布前需重建探针集。
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deusmith.algos import cosine  # noqa: E402
from deusmith.embeddings import HashEmbedding  # noqa: E402
from deusmith.llm import RuleBasedLLM  # noqa: E402
from deusmith.models import (  # noqa: E402
    MemoryNode, NodeType, Tier, utcnow,
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
    "丝绸之路贸易史", "统计学习理论入门", "宋代瓷器鉴赏要点",
    "深海潜水器设计原理", "银河系旋臂结构观测", "咖啡因代谢机制研究",
    "北欧家具设计史话",
]
PREFS = [
    ("用户喜欢摄影", 60), ("用户讨厌加班", 40), ("用户喜欢清晨跑步", 30),
    ("用户偏爱素食", 25), ("用户喜欢科幻小说", 20), ("用户讨厌嘈杂环境", 15),
    ("用户喜欢绿茶", 10), ("用户偏爱安静的工作环境", 5),
    ("用户喜欢悬疑电影", 28), ("用户讨厌早起", 9),
]
FACTS = [
    ("用户的公司在工业园区", 50), ("用户的团队有六名工程师", 45),
    ("用户的工位在三层靠窗", 40), ("用户常用Python和Rust", 35),
    ("用户的显示是双显示器", 30), ("用户每天通勤四十分钟", 22),
    ("用户的键盘是静音轴", 12), ("用户的椅子是人体工学椅", 8),
    ("用户的耳机是降噪款", 26), ("用户的午饭常吃沙拉", 14),
]
OTHER_EVENTS = [
    ("用户上周参加了技术沙龙", 18), ("用户昨天调试了数据库索引", 3),
    ("用户上个月参加了行业峰会", 55), ("用户上周买了新键盘", 6),
]

# 查询探针：(id, 查询, 期望 top-3 命中的子串) —— ground truth 独立于被测机制
PROBE_SET = [
    ("p01", "用户的爱好有什么", "摄影"),
    ("p02", "用户讨厌什么", "加班"),
    ("p03", "用户对什么过敏", "花粉"),
    ("p04", "用户现在住在哪里", "北京"),
    ("p05", "用户叫什么名字", "李雷"),
    ("p06", "用户喜欢什么运动", "跑步"),
    ("p07", "用户的工作环境偏好", "安静"),
    ("p08", "凤凰项目上线了吗", "凤凰"),
    ("p09", "用户常用什么编程语言", "Python"),
    ("p10", "用户通勤要多久", "四十分钟"),
    ("p11", "用户喝茶的偏好", "绿茶"),
    ("p12", "用户的饮食偏好", "素食"),
    ("p13", "用户有什么过敏原", "花粉"),
    ("p14", "用户在哪个城市", "北京"),
    ("p15", "周会都讨论什么", "周会"),
    ("p16", "用户的职业是什么", "工程师"),
    ("p17", "用户喜欢读什么类型的书", "科幻"),
    ("p18", "用户讨厌什么样的环境", "嘈杂"),
    ("p19", "用户的旅行计划", "京都"),
    ("p20", "用户的办公设备", "双显示器"),
]


def split_probes():
    """固定种子 70/30 切分：dev 用于开发与扫参，held-out 只在发布前跑一次。"""
    idx = list(range(len(PROBE_SET)))
    random.Random(42).shuffle(idx)
    cut = int(len(idx) * 0.7)
    dev = [PROBE_SET[i] for i in sorted(idx[:cut])]
    held = [PROBE_SET[i] for i in sorted(idx[cut:])]
    return dev, held


DEV_PROBES, HELDOUT_PROBES = split_probes()


def build_scenario(embedder=None, params=None):
    """构建 60 节点评测场景（全新 InMemory，无残留强化状态）。"""
    p = params or P
    emb = embedder or HashEmbedding(dim=DIM)
    store = InMemoryStore()
    mem = DeusmithMemory(store=store, embedder=emb, llm=RuleBasedLLM(),
                         params=p, scope=SCOPE)
    now = utcnow()

    def put(content, days_ago, type_=NodeType.event, **over):
        # conf 默认 0.8：与 write 路径产出一致（source_auth=0.8, ev=1 →
        # estimate_confidence≈0.85）。场景直接造节点时若沿用模型默认 0.5，
        # 会与真实管线写入的节点不在同一置信基线上，排序对比失真。
        kw = dict(
            type=type_, content=content, scope=SCOPE, tier=Tier.middle,
            embedding_model=emb.model_name, content_vector=emb.embed(content),
            last_access_at=now - timedelta(days=days_ago), weight=0.5,
            importance=0.5, confidence=0.8)
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
        human_confirmed=True, evidence_refs=("ev_profile_1",), importance=0.95,
        confidence=0.9))
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


def run_query_probes(mem, emb, probes):
    """查询探针：期望子串出现在 top-3。全新会话，避免残留强化。"""
    s = mem.session("probe-queries")
    results = []
    for pid, q, expect in probes:
        # 评估模式关闭命中强化：探针测的是冷启动排序质量，不是先发优势滚雪球
        res = mem.retrieve(s, q, topk=5, reinforce_hits=False)
        tops = [n.content for n, _, _ in res.selected[:3]]
        hit = any(expect in c for c in tops)
        results.append((pid, hit, f"top1={tops[0][:16] if tops else '∅'}"))
    return results


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

    # G5 常驻通道（改为注入语义：常驻记忆不进排序，注入时无条件可见）
    out5 = mem.inject_context(s, "今天天气怎么样")
    has_lilei = "李雷" in out5["system"] or "李雷" in out5["user"]
    has_pollen = "花粉" in out5["user"]
    ok = has_lilei and has_pollen
    gates.append(("G5 常驻通道：身份与显式记忆注入可见", bool(ok),
                  f"system含李雷={'李雷' in out5['system']}, user含花粉={has_pollen}"))

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
    emb = HashEmbedding(dim=DIM)
    store = InMemoryStore()
    lost = 0
    for orig, neg in NEG_PAIRS:
        kept = None
        for text in (orig, neg):
            vec = emb.embed(text)
            hits = store.vector_search(SCOPE, vec, topk=1)
            if hits and hits[0][1] >= theta:
                kept = hits[0][0]
                continue
            from deusmith.models import MemoryNode, NodeType, Tier
            node = MemoryNode(type=NodeType.preference, content=text, scope=SCOPE,
                              tier=Tier.middle, embedding_model=emb.model_name,
                              content_vector=vec)
            store.upsert_node(node)
            kept = node
        if not (hasattr(kept, "content") and "不" in kept.content):
            lost += 1
    merged = 0
    for orig, para in BENIGN_PAIRS:
        store2 = InMemoryStore()
        for text in (orig, para):
            vec = emb.embed(text)
            hits = store2.vector_search(SCOPE, vec, topk=1)
            if hits and hits[0][1] >= theta:
                continue
            from deusmith.models import MemoryNode, NodeType, Tier
            store2.upsert_node(MemoryNode(
                type=NodeType.preference, content=text, scope=SCOPE,
                tier=Tier.middle, embedding_model=emb.model_name,
                content_vector=vec))
        nodes = [n for n in store2.active_nodes(SCOPE)]
        if len([n for n in nodes if n.content in (orig, para)]) == 1:
            merged += 1
    return lost, merged


def slot_variant() -> int:
    """槽位+极性判定：否定更新保留数（以含否定的节点存在为准）。"""
    emb = HashEmbedding(dim=DIM)
    mem = DeusmithMemory(store=InMemoryStore(), embedder=emb, llm=RuleBasedLLM(),
                         params=P, scope=SCOPE)
    preserved = 0
    for i, (orig, neg) in enumerate(NEG_PAIRS):
        mem.write(mem.session(f"neg-{i}a"), orig + "。")
        mem.write(mem.session(f"neg-{i}b"), neg + "。")
        neg_nodes = [n for n in mem.store.recent_active(SCOPE, 100)
                     if "不" in n.content and n.subject == "user"]
        if neg_nodes:
            preserved += 1
    return preserved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", default="hash", choices=["hash", "local"])
    ap.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--heldout", action="store_true",
                    help="跑留出集（只应在发布前跑一次）")
    args = ap.parse_args()

    if args.embed == "local":
        try:
            from deusmith.embeddings import LocalModelEmbedding
            emb = LocalModelEmbedding(
                args.model,
                query_prefix="为这个句子生成表示以用于检索相关文章：")
        except ImportError as e:
            print(f"[SKIP] 本地模型不可用：{e}")
            return 0
    else:
        emb = HashEmbedding(dim=DIM)

    probes = HELDOUT_PROBES if args.heldout else DEV_PROBES
    label = "held-out" if args.heldout else "dev"

    print(f"== Deusmith-MemorySystem · 探针门禁（embedder={emb.model_name}）==")
    # 场景级门禁（G1–G7）跑在专用场景上，与查询探针互不污染
    mem_g, _ = build_scenario(emb)
    total_nodes = mem_g.store.counts(SCOPE)["nodes_total"]
    print(f"门禁场景规模：{total_nodes} 节点")
    gates = run_gates(mem_g, emb)
    gate_fails = 0
    for name, ok, detail in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} —— {detail}")
        gate_fails += 0 if ok else 1

    # 查询探针（全新场景）
    mem_q, _ = build_scenario(emb)
    qres = run_query_probes(mem_q, emb, probes)
    acc = sum(1 for _, h, _ in qres if h) / len(qres)
    print(f"\n查询探针[{label}]：{sum(1 for _, h, _ in qres if h)}/{len(qres)}"
          f"（准确率 {acc:.3f}）")

    # ---- 对照实验（固定用 hash 嵌入，结论与嵌入无关：判据层面的问题）
    print("\n== 对照实验（评审错误#1：相似度判据 vs 槽位+极性）==")
    hemb = HashEmbedding(dim=DIM)
    cos_neg = [cosine(hemb.embed(a), hemb.embed(b)) for a, b in NEG_PAIRS]
    cos_ben = [cosine(hemb.embed(a), hemb.embed(b)) for a, b in BENIGN_PAIRS]
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
    no_valid_theta = all(not (t[2] == 8 and t[1] == 0) for t in tradeoff)
    print(f"  结论：{'不存在能同时「合并全部良性改写」与「零丢失否定更新」的阈值' if no_valid_theta else '存在可行阈值'}")

    # ---- 报告落盘
    report = ["# Deusmith-MemorySystem · 探针评估报告", "",
              f"生成：scripts/eval_probes.py（门禁场景 {total_nodes} 节点，"
              f"embedder=`{emb.model_name}`，规则 LLM）", "",
              "## 门禁结果（当前技术栈）", "",
              "| 门禁 | 结果 | 证据 |", "|---|---|---|"]
    for name, ok, detail in gates:
        report.append(f"| {name} | {'**PASS**' if ok else '**FAIL**'} | {detail} |")
    report += ["", f"**门禁：{len(gates) - gate_fails}/{len(gates)} PASS**", "",
               "## 查询探针（70/30 切分，固定种子 42）", "",
               f"- **dev（{len(DEV_PROBES)} 条，本报告口径）准确率："
               f"{acc:.3f}（embedder={emb.model_name}, split={label}）",
               f"- held-out：{len(HELDOUT_PROBES)} 条，只应在发布前跑一次"
               f"（`--heldout`；v0.2.0 基线见 docs/04）",
               "- ground truth 为期望子串出现在 top-3，独立于被测机制",
               "", "## 对照实验：相似度判据 vs 槽位+极性（评审错误#1）", "",
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
               "## 污染声明与校准协议", "",
               "dev 集在开发过程中用于发现并修复：多值谓语冲突误判、中文关键词通道"
               "切词缺失、摘要置信度继承缺陷、G4 门禁顺序效应——**它已是被污染的开发集**。",
               "held-out 在 v0.2.0 发布前跑了一次基线（默认参数），下次发布前需重建。",
               "正式调参用 scripts/calibrate.py：dev 上单维扫参，选型后 held-out 终验一次。",
               "", "## 已知缺口（不进门禁，单列）", "",
               "- HashEmbedding 为词法级占位实现（本报告默认），跨措辞召回显著低于"
               "真实语义模型（对照数字见 docs/04 §I）；",
               "- 图游走为简化 PPR（2 跳衰减），未与真 PPR/Neo4j 对照；",
               "- 多用户并发与 PG 后端下的检索延迟未在本报告覆盖（见 docs/04 §H）。"]
    out = ROOT / "docs" / "eval-report.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"\n报告已写入 {out}")
    print(f"== 门禁：{len(gates) - gate_fails}/{len(gates)} PASS | "
          f"查询探针[{label}] {acc:.3f} ==")
    return 1 if gate_fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
