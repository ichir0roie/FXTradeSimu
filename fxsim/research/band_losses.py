"""E060 の負けどころ: 日中の主の形を出口の理由ごとに(実データと偽薬)、週足の逆張りを年ごとに。

    python -m fxsim.research.band_losses
"""
# ruff: noqa: E501
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fxsim.research.vol_band_fade import CROSSES, HOLDOUT, MAIN, PAIRS, simulate

WHY = {0: "時間切れ", 1: "損切り(2w)", 2: "利確(P に戻る)", 3: "両側で確定"}

def job(a):
    return simulate(*a)

def part1():
    print("## 1 時間足・4 時間足の主の形(指値・q 0.9・tp_stop・3 本)、主要 10 ペア。bp / 取引")
    for tf in ("H1", "H4"):
        jobs = [(p, tf, 0, g, s, 0.0, [MAIN]) for p in PAIRS for g, s in ((False, 0), (True, 0), (True, 1))]
        with ProcessPoolExecutor(6) as ex:
            outs = list(ex.map(job, jobs))
        res = {}
        for j, o in zip(jobs, outs, strict=True):
            res.setdefault((j[3], j[4]), []).append(o["res"][MAIN])
        net, gross, plc = (np.vstack(res[k]) for k in ((False, 0), (True, 0), (True, 1)))
        print(f"\n### {tf}: 取引 {len(net)}、費用込み {net[:,0].mean():+.2f} = 素 {gross[:,0].mean():+.2f} − 費用 {gross[:,0].mean()-net[:,0].mean():.2f}。偽薬の素 {plc[:,0].mean():+.2f}")
        print("| 出口 | 実データ: 割合 / 素の bp / 費用込み bp / 合計(費用込み) | 偽薬: 割合 / 素の bp |")
        print("|---|---|---|")
        for w, lab in WHY.items():
            m, mg, mp = net[:, 3] == w, gross[:, 3] == w, plc[:, 3] == w
            print(f"| {lab} | {m.mean():.0%} / {gross[mg,0].mean():+.1f} / {net[m,0].mean():+.1f} / {net[m,0].sum():+.0f} | {mp.mean():.0%} / {plc[mp,0].mean():+.1f} |")

def part2():
    print("\n## 週足の逆張り(2010 年以降で一番だった形)の年ごと。bp の合計(費用込み + キャリー)と取引数")
    for cfg in (("fade", ("kc", 20, 2.0), "tp", 10, "market"), ("fade", ("bb", 20, 2.5), "tp", 10, "market"), ("fade", 0.9, "tp_stop", 3, 100)):
        jobs = [(p, "W1", 0, False, 0, 0.0, [cfg]) for p in PAIRS + CROSSES] + [(p, "W1d", 0, False, 0, 0.0, [cfg], HOLDOUT) for p in PAIRS]
        with ProcessPoolExecutor(6) as ex:
            outs = list(ex.map(job, jobs))
        tr = np.vstack([o["res"][cfg] for o in outs])
        yrs = (tr[:, 1] // (365.25 * 86400) + 1970).astype(int)
        tot = tr[:, 0] + tr[:, 4]
        line = "、".join(f"{y}: {tot[yrs==y].sum():+.0f}({(yrs==y).sum()})" for y in sorted(set(yrs)))
        win, loss = tot[tot > 0], tot[tot < 0]
        print(f"\n- {cfg[1]} {cfg[4] if isinstance(cfg[1], tuple) else 'limit'} {cfg[2]} {cfg[3]}: 勝ち平均 {win.mean():+.0f} / 負け平均 {loss.mean():+.0f} bp、最悪 {tot.min():+.0f}")
        print("  " + line)

if __name__ == "__main__":
    part1()
    part2()
