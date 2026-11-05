# -*- coding: utf-8 -*-
"""诊断 v3：G3 查询全链路解剖（bge）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_probes as ep
from deusmith.embeddings import LocalModelEmbedding
from deusmith.algos import cosine

MODEL = r"E:/ALL_FOR_MY_PROJECTS/Deusmith/Deusmith-MemorySystem/bge-small-zh-v1.5"
emb = LocalModelEmbedding(MODEL, query_prefix="为这个句子生成表示以用于检索相关文章：")

mem, _ = ep.build_scenario(emb)
s3 = mem.session("probe-s3")
q = "刚才聊的旅行计划是什么"

# 先模拟 G1/G2 已跑过的强化（gates 顺序：G1、G2 在 G3 之前）
mem.retrieve(mem.session("probe-eval"), "用户最近上线了什么项目")
mem.retrieve(mem.session("probe-eval"), "用户现在住在哪个城市")

res = mem.retrieve(s3, q, topk=8)
print(f"query={q!r}")
for i, (n, sc, rel) in enumerate(res.selected[:8]):
    mark = "★" if "京都" in n.content else " "
    print(f" {mark}{i+1}. score={sc:.4f} rel={rel:.3f} w={n.weight:.3f} "
          f"conf={n.confidence:.2f} imp={n.importance:.2f} tier={n.tier.value:8s} {n.content[:22]}")

target = [n for n in mem.store.active_nodes(ep.SCOPE) if "京都" in n.content]
if target:
    t = target[0]
    qv = emb.embed_query(q)
    vhits = mem.store.vector_search(ep.SCOPE, qv, topk=65)
    vrank = next((i+1 for i, (n, sim) in enumerate(vhits) if n.id == t.id), None)
    vsim = next((sim for n, sim in vhits if n.id == t.id), None)
    print(f"\n目标「{t.content[:20]}」向量名次={vrank}/{len(vhits)} cos={vsim:.4f} "
          f"w={t.weight:.3f} conf={t.confidence:.2f} imp={t.importance:.2f} tier={t.tier.value}")
    print(f"top5 向量名次及其 cos：")
    for i, (n, sim) in enumerate(vhits[:5]):
        print(f"  v{i+1}. cos={sim:.4f} {n.content[:26]}")
    # 京都是否进了 selected 但排在后面？
    allsel = [(i+1, n, sc) for i, (n, sc, _) in enumerate(res.selected)]
    pos = next((p for p, n, _ in allsel if "京都" in n.content), None)
    print(f"京都 selected 名次: {pos}")
