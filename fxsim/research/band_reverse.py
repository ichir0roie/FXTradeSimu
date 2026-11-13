"""E060 を裏返す。日中は 5 分足の経路(2025-07〜、28 ペア)で、指値の逆張りと同じ値の逆指値の順張りを比べる
(元の利確の位置 P に損切り、元の損切りの位置に利確)。週足は逆張りの最良の形を順張りにして(平均線に損切り)、2010〜 と 1994〜2010。

    python -m fxsim.research.band_reverse
"""
# ruff: noqa: E501
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fxsim.research.vol_band_fade import CROSSES, HOLDOUT, PAIRS, simulate

M5 = (5, int(np.datetime64("2025-07-01").astype("datetime64[s]").astype(np.int64)), None)

def job(a):
    return simulate(*a)

def month_t(v, tt):
    if len(v) < 10:
        return 0.0
    mon = (tt // (86400 * 30.4375)).astype(int)
    s = np.bincount(mon - mon.min(), weights=v)[np.unique(mon - mon.min())]
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))

def table(cfgs, tfs, pairs, src, title, carry=False):
    print(f"\n## {title}")
    print("| 足 | 形 | 取引 / 年(1 ペア) | 費用込み bp | 素 bp | 勝率 | t(月) |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for tf in tfs:
        jobs = [(p, tf, 0, g, 0, 0.0, cfgs, src) for p in pairs for g in (False, True)]
        with ProcessPoolExecutor(6) as ex:
            outs = list(ex.map(job, jobs))
        for cfg, lab in cfgs_labels[tuple(cfgs)]:
            net = np.vstack([o["res"][cfg] for j, o in zip(jobs, outs, strict=True) if not j[3]])
            gr = np.vstack([o["res"][cfg] for j, o in zip(jobs, outs, strict=True) if j[3]])
            yrs = sum(o["years"] for j, o in zip(jobs, outs, strict=True) if not j[3])
            v = net[:, 0] + (net[:, 4] if carry else 0)
            print(f"| {tf} | {lab} | {len(net)/yrs:.0f} | {v.mean():+.2f} | {gr[:,0].mean()+(gr[:,4].mean() if carry else 0):+.2f} | {(v>0).mean():.0%} | {month_t(v, net[:,1]):+.2f} |")
    return

intraday = [(("fade", 0.9, "tp_stop", 3, 100), "元: 指値で逆張り(q 0.9、P で利確、2w 先で損切り、3 本)"),
            (("break", 0.9, "tp_stop", 3, 100), "裏返し: 逆指値で順張り(P で損切り、2w 先で利確、3 本)"),
            (("fade", 0.9, "time", 3, 100), "元: 指値で逆張り、3 本で出る"),
            (("break", 0.9, "time", 3, 100), "裏返し: 逆指値で順張り、3 本で出る")]
weekly = [(("fade", ("kc", 20, 2.0), "tp", 10, "market"), "元: ケルトナー 20・2 の外で逆張り、平均線で利確、10 週"),
          (("break", ("kc", 20, 2.0), "tp", 10, "market"), "裏返し: 同じ所で順張り、平均線で損切り、10 週"),
          (("fade", ("bb", 20, 2.5), "tp", 10, "market"), "元: ボリンジャー 20・2.5 の外で逆張り"),
          (("break", ("bb", 20, 2.5), "tp", 10, "market"), "裏返し: 同じ所で順張り、平均線で損切り")]
cfgs_labels = {tuple(c for c, _ in intraday): intraday, tuple(c for c, _ in weekly): weekly}

if __name__ == "__main__":
    table([c for c, _ in intraday], ["H1", "H4"], PAIRS + CROSSES, M5, "日中(5 分足の経路、2025-07〜2026-11、28 ペア)")
    table([c for c, _ in weekly], ["W1"], PAIRS + CROSSES, None, "週足 2010-10〜(1 時間足の経路、28 ペア、キャリー込み)", carry=True)
    table([c for c, _ in weekly], ["W1d"], PAIRS, HOLDOUT, "週足 初見の 1994〜2010-09(日足の経路、主要 10 ペア、キャリー込み)", carry=True)
