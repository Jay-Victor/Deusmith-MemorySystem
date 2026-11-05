# -*- coding: utf-8 -*-
"""calibrate.py —— 单维参数校准框架（评审第一梯队 #3）。

协议（与 docs/eval-report.md 的污染声明配套）：
  1. dev 集（70%）扫参：一次只动一个维度，其余固定默认值；
  2. 每个候选值在全新场景上跑查询探针（场景与探针都不复用强化状态）；
  3. held-out（30%）只在最终选型后跑一次（--final），跑过即作废，需重建。

用法：
  PYTHONPATH=src .venv/Scripts/python.exe scripts/calibrate.py                  # 默认扫参表
  PYTHONPATH=src .venv/Scripts/python.exe scripts/calibrate.py --param score.mmr_lambda --values 0.5,0.6,0.7,0.8
  PYTHONPATH=src .venv/Scripts/python.exe scripts/calibrate.py --final          # held-out 终验（慎用）
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import eval_probes as ep  # noqa: E402
from deusmith.params import Params  # noqa: E402

DEFAULT_SWEEPS = {
    "score.mmr_lambda": [0.5, 0.6, 0.7, 0.8, 0.9],
    "score.beta": [0.6, 0.8, 1.0],
    "retrieval.topk_recall": [30, 50, 80],
}


def apply_param(params: Params, dotted: str, value):
    section, key = dotted.split(".", 1)
    return replace(params,
                   **{section: replace(getattr(params, section), **{key: value})})


def accuracy(params: Params, probes, embedder) -> float:
    mem, emb = ep.build_scenario(embedder=embedder, params=params)
    res = ep.run_query_probes(mem, emb, probes)
    return sum(1 for _, hit, _ in res if hit) / len(res)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", help="形如 score.mmr_lambda 的点号键")
    ap.add_argument("--values", help="逗号分隔候选值")
    ap.add_argument("--embed", default="hash", choices=["hash", "local"],
                    help="embedder：hash（离线默认）或 local（真实模型）")
    ap.add_argument("--model", default="BAAI/bge-small-zh-v1.5",
                    help="local 模式下的模型路径/ID")
    ap.add_argument("--final", action="store_true",
                    help="held-out 终验（跑过即作废）")
    args = ap.parse_args()

    if args.embed == "local":
        try:
            from deusmith.embeddings import LocalModelEmbedding
            emb = LocalModelEmbedding(
                args.model,
                query_prefix="为这个句子生成表示以用于检索相关文章：")
        except ImportError as e:
            raise SystemExit(f"[SKIP] 本地模型不可用：{e}")
    else:
        emb = None       # build_scenario 默认 hash

    base = Params()
    if args.final:
        acc = accuracy(base, ep.HELDOUT_PROBES, emb)
        print(f"[held-out 终验] 默认参数准确率：{acc:.3f}"
              f"（{len(ep.HELDOUT_PROBES)} 条）")
        print("注意：held-out 跑过即被消费；下次发布前请重建探针集。")
        return 0

    if args.param:
        if not args.values:
            raise SystemExit("--param 需要 --values")
        values = [float(v) if "." in v else int(v)
                  for v in args.values.split(",")]
        sweeps = {args.param: values}
    else:
        sweeps = DEFAULT_SWEEPS

    base_acc = accuracy(base, ep.DEV_PROBES, emb)
    emb_label = emb.model_name if emb else "hash"
    print(f"基线（默认参数，embedder={emb_label}，dev {len(ep.DEV_PROBES)} 条）：{base_acc:.3f}")
    for dotted, values in sweeps.items():
        print(f"\n== 扫参 {dotted} ==")
        for v in values:
            acc = accuracy(apply_param(base, dotted, v), ep.DEV_PROBES, emb)
            delta = (acc - base_acc) * 100
            print(f"  {dotted}={v!s:>6} → dev 准确率 {acc:.3f}（{delta:+.1f}pp）")
    print("\n协议：选型后用 --final 在 held-out 上终验一次；跑过即作废。")
    print("提醒：dev=14 时 ±1 条探针 = ±7.1pp，小于 2 条的波动是噪声；"
          "无多点一致提升（>2pp 且跨相邻值稳定）时保持默认——防过拟合。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
