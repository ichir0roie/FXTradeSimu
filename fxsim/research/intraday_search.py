"""E055: 日足未満のテクニカル 168 通りを、発見(主要 10 ペア・2010〜2018)と確認(全 28 ペア・2018〜2026)に分けて探す。

    python -m fxsim.research.intraday_search discover      # 発見だけ(選ぶ基準を当てた一覧)
    python -m fxsim.research.intraday_search confirm       # 発見で選ばれたものを、確認で 1 回だけ

設計と基準は docs/proposals/experiments/E055-intraday-split-search.md(結果を見る前に固定)。
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

from .intraday_limit import CROSSES, FAMILIES, PAIRS, month_t, simulate

TFS = [1, 2, 4, 8]
GATES = {"all": {}, "high": {"vr_min": 1.2}, "low": {"vr_max": 0.8}}
ENTRIES = [0.0, 0.5]
SPLIT = int(np.datetime64("2018-11-01").astype("datetime64[s]").astype(np.int64))
T_SELECT = 3.5
OUT = Path(__file__).with_name("intraday_search_selected.json")
NULL_SEEDS = 6


def _opts(fam: str, gate: str, extra: dict | None = None) -> dict:
    return {"family": fam, "rolling": True, "max_bars": 6, "stop": 3.0, **GATES[gate], **(extra or {})}


def _run(jobs, workers):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(simulate, jobs, chunksize=1))


def discover(workers: int) -> None:
    configs = list(product(FAMILIES, TFS, GATES))
    jobs = [(p, k, 0, _opts(fam, g, {"until": SPLIT})) for fam, k, g in configs for p in PAIRS]
    outs = _run(jobs, workers)
    rows = []
    for ci, (fam, k, g) in enumerate(configs):
        per = outs[ci * len(PAIRS):(ci + 1) * len(PAIRS)]
        for d in ENTRIES:
            tr = np.vstack([o[d] for o in per])
            if len(tr) < 30:
                rows.append((fam, k, g, d, len(tr), 0.0, 0.0, 0.0, 0))
                continue
            v, t = tr[:, 0], tr[:, 1]
            rows.append((fam, k, g, d, len(tr), float(tr[:, 5].mean()), float(v.mean()), month_t(v, t),
                         sum(o[d][:, 0].sum() > 0 for o in per)))
    rows.sort(key=lambda r: -r[7])
    print(f"発見: 主要 10 ペア、2010-10〜2018-10。{len(rows)} 通り。選ぶ基準 = 1 取引 > 0 かつ月ごとの t ≥ {T_SELECT}\n")
    print("| 型 | 足 | 荒さ | 入り | 取引 / 年 | 1 取引 pip | bp | t(月) | プラスのペア | 選ぶ |\n|---|---:|---|---|---:|---:|---:|---:|---:|---|")
    sel = []
    for fam, k, g, d, n, pips, bp, t, pp in rows:
        ok = bp > 0 and t >= T_SELECT
        if ok:
            sel.append({"family": fam, "tf": k, "gate": g, "limit": d})
        print(f"| {fam} | {k} | {g} | {'成行' if d == 0 else f'指値 {d}'} | {n / 8.1:.0f} | {pips:+.2f} | {bp:+.2f} | {t:+.2f} | {pp}/10 | {'o' if ok else ''} |")
    OUT.write_text(json.dumps(sel, ensure_ascii=False, indent=1))
    print(f"\n選ばれたもの {len(sel)} 通り -> {OUT.name}")


def confirm(workers: int) -> None:
    sel = json.loads(OUT.read_text())
    pairs = PAIRS + CROSSES
    print(f"確認: 全 {len(pairs)} ペア、2018-11〜2026-11。発見で選ばれた {len(sel)} 通りを 1 回だけ\n")
    print("| 型 | 足 | 荒さ | 入り | 取引 / 年 | 1 取引 pip | bp / t(月) | プラスのペア | でたらめの bp | z | 合格 |\n|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|")
    for s in sel:
        jobs = [(p, s["tf"], seed, _opts(s["family"], s["gate"], {"since": SPLIT})) for seed in range(NULL_SEEDS + 1) for p in pairs]
        outs = _run(jobs, workers)
        d = s["limit"]
        per = [o[d] for (p, k, seed, _), o in zip(jobs, outs, strict=True) if seed == 0]
        tr = np.vstack(per)
        nulls = np.array([np.vstack([o[d] for (p, k, seed, _), o in zip(jobs, outs, strict=True) if seed == sd])[:, 0].mean()
                          for sd in range(1, NULL_SEEDS + 1)])
        v, t = tr[:, 0], tr[:, 1]
        tt = month_t(v, t)
        pp = sum(x[:, 0].sum() > 0 for x in per if len(x))
        npairs = sum(len(x) > 0 for x in per)
        z = (v.mean() - nulls.mean()) / nulls.std(ddof=1)
        ok = v.mean() > 0 and tt >= 2 and pp >= 0.6 * npairs and z >= 2
        tag = "参考(pullback は汚れている)" if s["family"] == "pullback" else ("o" if ok else "x")
        print(f"| {s['family']} | {s['tf']} | {s['gate']} | {'成行' if d == 0 else f'指値 {d}'} | {len(tr) / 8.0:.0f} | {tr[:, 5].mean():+.2f} "
              f"| {v.mean():+.2f} / {tt:+.2f} | {pp}/{npairs} | {nulls.mean():+.2f} ± {nulls.std(ddof=1):.2f} | {z:+.2f} | {tag} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["discover", "confirm"])
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    discover(args.workers) if args.step == "discover" else confirm(args.workers)


if __name__ == "__main__":
    main()
