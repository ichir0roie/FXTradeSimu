"""E062・E067 の負けどころ: 片側・利確のみ・逆指値なしの形を、出口の理由・負けの裾・発表の足の向きと大きさで分ける。

    python -m fxsim.research.event_losses

約定と決まりは event_nfp(E062)と同じ。素の損益はスプレッドと滑りを 0 にして回し直した値。
"""
# ruff: noqa: E501
from __future__ import annotations

from datetime import date

import numpy as np

from fxsim.calendar import claims_dates, nfp_bar_start, nfp_dates
from fxsim.research.event_nfp import PAIRS, Path, load, wv_series

FORMS = [("trend", 0.5, "calm"), ("trend", 0.25, "5d"), ("random", 0.25, "5d")]

def run(event):
    dates = (nfp_dates if event == "nfp" else claims_dates)(date(2010, 10, 1), date(2026, 11, 13))
    occ = [nfp_bar_start(d) for d in dates]
    rows = {f: [] for f in FORMS}
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
            size = (x["h"][k] - x["l"][k]) / wv           # 発表の足の幅 / 物差し
            for f in FORMS:
                dname, ktp, ex = f
                side = (int(x["trend20"][k]) or 1) if dname == "trend" else int(rng.choice([-1, 1]))
                for gross in (False, True):
                    p = Path(x, 0.0, k)
                    if gross:
                        p.sp[:] = 0.0
                        p.s_mkt = p.s_stop = 0.0
                    e = p.market_in(k, side)
                    take = P0 + side * ktp * wv
                    j_end = k + 120
                    j, px = p.run_exit(k, side, e, j_end, take, None, True, calm=(ex == "calm"))
                    v = side * (px - e) * bp
                    if gross:
                        g = v
                    else:
                        net = v
                        hit_tp = (side > 0 and x["h"][j] >= take) or (side < 0 and x["l"][j] + p.sp[j] <= take)
                        reason = "tp" if hit_tp and j < p.next_good(min(j_end, n - 1)) else ("calm" if ex == "calm" and j < p.next_good(min(j_end, n - 1)) else "timeout")
                cont = np.sign(x["c"][k] - P0) == side          # 発表の足が自分の向きに終わったか
                rows[f].append((net, g, reason, size, cont, j - k))
    return rows

for event in ("nfp", "claims"):
    rows = run(event)
    for f, rs in rows.items():
        net = np.array([r[0] for r in rs])
        g = np.array([r[1] for r in rs])
        reason = np.array([r[2] for r in rs])
        size = np.array([r[3] for r in rs])
        cont = np.array([r[4] for r in rs])
        held = np.array([r[5] for r in rs])
        cost = g - net
        print(f"\n### {event} {f}: 取引 {len(net)}、勝率 {(net>0).mean():.0%}、1 取引 費用込み {net.mean():+.2f} bp = 素 {g.mean():+.2f} − 費用 {cost.mean():.2f}")
        loss = net[net < 0]
        k5 = max(1, len(net) // 20)
        print(f"- 勝ちの合計 {net[net>0].sum():+.0f} / 負けの合計 {loss.sum():+.0f} bp。下位 5%({k5} 本)の負け {np.sort(net)[:k5].sum():+.0f} bp(負けの合計の {np.sort(net)[:k5].sum()/loss.sum():.0%})、最悪 {net.min():+.0f}")
        for rsn in ("tp", "calm", "timeout"):
            m = reason == rsn
            if m.any():
                print(f"- 出口 {rsn}: {m.sum()} 本({m.mean():.0%})、平均 {net[m].mean():+.1f} bp、合計 {net[m].sum():+.0f}、勝率 {(net[m]>0).mean():.0%}、持った本数の中央 {np.median(held[m]):.0f}")
        for lab, m in (("発表の足が自分の向きに終わった", cont), ("逆向きに終わった", ~cont)):
            print(f"- {lab}: {m.sum()} 本、平均 {net[m].mean():+.1f} bp、勝率 {(net[m]>0).mean():.0%}")
        q = np.quantile(size, [1/3, 2/3])
        for lab, m in (("発表の足が小さい 1/3", size <= q[0]), ("中", (size > q[0]) & (size <= q[1])), ("大きい 1/3", size > q[1])):
            print(f"- {lab}: 平均 {net[m].mean():+.1f} bp、勝率 {(net[m]>0).mean():.0%}、最悪 {net[m].min():+.0f}")
