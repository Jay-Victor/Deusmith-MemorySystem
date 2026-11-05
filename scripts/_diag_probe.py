# -*- coding: utf-8 -*-
"""诊断：本地模型下查询探针的排序细节。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_probes as ep
from deusmith.embeddings import LocalModelEmbedding

MODEL = r"E:/ALL_FOR_MY_PROJECTS/Deusmith/Deusmith-MemorySystem/bge-small-zh-v1.5"
emb = LocalModelEmbedding(MODEL, query_prefix="为这个句子生成表示以用于检索相关文章：")

mem, _ = ep.build_scenario(emb)
s = mem.session("probe-diag")

for pid, q, expect in ep.DEV_PROBES[:5]:
    res = mem.retrieve(s, q, topk=5)
    print(f"\n[{pid}] q={q!r} 期望含 {expect!r}")
    for i, (n, sc, rel) in enumerate(res.selected[:5]):
        mark = "★" if expect in n.content else " "
        print(f"  {mark}{i+1}. score={sc:.4f} rel={rel:.3f} type={n.type.value:11s} {n.content[:22]}")
    if not any(expect in n.content for n, _, _ in res.selected[:3]):
        # 目标落选：看它在候选池的分数与名次
        target = [n for n in mem.store.active_nodes(ep.SCOPE) if expect in n.content]
        if target:
            t = target[0]
            from deusmith.models import utcnow
            import deusmith.algos as A
            now = utcnow()
            sc = A.retrieval_score(0.5, t, now, 2, mem.params)
            vhits = mem.store.vector_search(ep.SCOPE, emb.embed_query(q), topk=60)
            vrank = next((i+1 for i, (n, sim) in enumerate(vhits) if n.id == t.id), None)
            vsim = next((sim for n, sim in vhits if n.id == t.id), None)
            print(f"    → 目标「{t.content[:18]}」: 向量名次={vrank} cos={vsim:.3f} "
                  f"weight={t.weight:.2f} recency={A.decay_recency(t, now):.4f} "
                  f"rec^γ={A.decay_recency(t, now)**0.6:.4f} rel半估分={sc:.4f}")
