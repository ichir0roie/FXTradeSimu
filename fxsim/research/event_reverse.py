"""指標発表の片側の形を裏返す: 向きを逆にし、元の利確の位置に損切りを置く。出口(calm / 5d)は同じ。

    python -m fxsim.research.event_reverse

約定と決まりは event_nfp(E062)と同じ。雇用統計と新規失業保険申請件数(E067)。
"""
# ruff: noqa: E501
from __future__ import annotations

from datetime import date

import numpy as np

from fxsim.calendar import claims_dates, nfp_bar_start, nfp_dates
from fxsim.research.event_nfp import PAIRS, SPLIT, Path, load, wv_series

FORMS = [(d, k, ex) for d in ("trend", "random") for k in (0.25, 0.5) for ex in ("calm", "5d")]

def month_t(v, tt):
    mon = (np.asarray(tt) // (86400 * 30.4375)).astype(int)
    s = np.bincount(mon - mon.min(), weights=v)[np.unique(mon - mon.min())]
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))

for event in ("nfp", "claims"):
    dates = (nfp_dates if event == "nfp" else claims_dates)(date(2010, 10, 1), date(2026, 11, 13))
    occ = [nfp_bar_start(d) for d in dates]
    rows = {(f, rev): [] for f in FORMS for rev in (False, True)}
    for pair in PAIRS:
        x = load(pair)
        wvs = wv_series(x, occ)
        n = len(x["t"])
        rng = np.random.default_rng(PAIRS.index(pair) + 7)
        for ev in occ:
            k = x["idx"].get(ev)
            if k is None or ev not in wvs or k + 200 >= n or np.isnan(x["sma"][k]) or x["bad"][k]:
                continue
            P0 = x["o"][k]
            wv = wvs[ev] * P0
            bp = 1e4 / P0
            rnd = int(rng.choice([-1, 1]))
            for f in FORMS:
                dname, kk, ex = f
                side0 = (int(x["trend20"][k]) or 1) if dname == "trend" else rnd
                for rev in (False, True):
                    out = []
                    for gross in (False, True):
                        p = Path(x, 0.0, k)
                        if gross:
                            p.sp[:] = 0.0
                            p.s_mkt = p.s_stop = 0.0
                        side = -side0 if rev else side0
                        e = p.market_in(k, side)
                        lvl = P0 + side0 * kk * wv                 # 元の利確の位置
                        take, stop = (None, lvl) if rev else (lvl, None)
                        j, px = p.run_exit(k, side, e, k + 120, take, stop, True, calm=(ex == "calm"))
                        out.append(side * (px - e) * bp)
                    rows[(f, rev)].append((out[0], out[1], ev))
    print(f"\n## {event}: 元の形と、裏返した形(向きを逆、元の利確の位置に損切り)。bp / 取引")
    print("| 向き | k | 出口 | 元: 費用込み / 素 | 裏返し: 費用込み / 素 / 勝率 / t(月) | 裏返しの発見 / 確認 |")
    print("|---|---:|---|---|---|---|")
    for f in FORMS:
        a = np.array(rows[(f, False)])
        b = np.array(rows[(f, True)])
        sa, sb = b[:, 2] < SPLIT, b[:, 2] >= SPLIT
        print(f"| {f[0]} | {f[1]} | {f[2]} | {a[:,0].mean():+.2f} / {a[:,1].mean():+.2f} | {b[:,0].mean():+.2f} / {b[:,1].mean():+.2f} / {(b[:,0]>0).mean():.0%} / {month_t(b[:,0], b[:,2]):+.2f} "
              f"| {b[sa,0].mean():+.2f} / {b[sb,0].mean():+.2f} |")
