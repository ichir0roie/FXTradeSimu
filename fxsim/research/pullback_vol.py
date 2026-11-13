"""E051: 押し目(指値で待つ版)の利益は、ボラが大きい時期に寄っているか。そこだけ入る形は成り立つか。

    python -m fxsim.research.pullback_vol

ボラの物差しは、入る日の ATR(20) / ATR(250)(区切りは 0.8・1.0・1.2 に固定。結果を見て動かさない)。
    1  1 時間足の経路(E050 と同じ約定、2010-10〜2026-11): 成行と指値 ATR x 0.5 の取引を、ボラの比で分ける。
       bp(同じ通貨数)と R(損益 / その日の ATR。ボラで量をそろえた単位)。でたらめな入り 20 通りも同じ区分で
    2  日足だけ(ta_scan のデータ、1996〜2026): 成行の押し目を同じ区分で。1996〜2010 年は 1 の研究で使っていない
発見 = 2010-10〜2018-10、確認 = 2018-11〜。
"""
# ruff: noqa: E501
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..strategies.indicators import atr, rsi, sma
from . import exit_grid as g
from .ta_scan import COST, Data, hold

BINS = [(0.0, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 9.9)]
KEYS = [(0.0, 3.0, 0.0, True), (0.5, 3.0, 0.0, True)]
NULL_SEEDS = 20


def _job(args):
    pair, kind, seed = args
    g.ENTRIES, g.STOPS, g.TAKES, g.SIGNAL_EXIT = [0.0, 0.5], [3.0], [0.0], [True]
    return g.simulate(pair, kind, seed)


def cell(v, t) -> str:
    return f"{v.mean():+.2f} / {g.month_t(v, t):+.2f}" if len(v) >= 10 else "—"


def hourly() -> None:
    jobs = [(p, "pullback", 0) for p in g.PAIRS] + [(p, "random", s) for s in range(1, NULL_SEEDS + 1) for p in g.PAIRS]
    with ProcessPoolExecutor(10) as ex:
        outs = list(ex.map(_job, jobs))
    for key in KEYS:
        tr = np.vstack([o[key] for j, o in zip(jobs, outs, strict=True) if j[1] == "pullback"])
        nulls = [np.vstack([o[key] for j, o in zip(jobs, outs, strict=True) if j[1] == "random" and j[2] == s])
                 for s in range(1, NULL_SEEDS + 1)]
        v, t, vr, rel = tr[:, 0], tr[:, 1], tr[:, 4], tr[:, 5]
        r = v / (rel * 1e4)
        print(f"\n### {'成行' if key[0] == 0 else '指値 ATR x 0.5'}(損切り ATR x 3・利確なし・信号の出あり)\n")
        print("| ATR(20) / ATR(250) | 取引 / 年 | 勝率 | 1 取引 bp / t(月) | 1 取引 R / t(月) | 発見 bp / t | 確認 bp / t | 発見 R | 確認 R "
              "| でたらめ 20 通りの bp(平均 ± 振れ) | z |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for lo, hi in BINS + [(0.0, 9.9), (1.0, 9.9)]:
            m = (vr >= lo) & (vr < hi)
            a, b = m & (t < g.SPLIT), m & (t >= g.SPLIT)
            nm = np.array([n[(n[:, 4] >= lo) & (n[:, 4] < hi), 0].mean() for n in nulls])
            name = "全部" if (lo, hi) == (0.0, 9.9) else "1.0 以上" if (lo, hi) == (1.0, 9.9) else f"{lo}〜{hi}" if hi < 9 else f"{lo} 以上"
            print(f"| {name} | {m.sum() / 16.1:.0f} | {(v[m] > 0).mean():.0%} | {cell(v[m], t[m])} | {cell(r[m], t[m])} "
                  f"| {cell(v[a], t[a])} | {cell(v[b], t[b])} | {r[a].mean():+.3f} | {r[b].mean():+.3f} "
                  f"| {nm.mean():+.2f} ± {nm.std(ddof=1):.2f} | {(v[m].mean() - nm.mean()) / nm.std(ddof=1):+.2f} |")


def daily() -> None:
    d = Data()
    c = d.c
    ma, r = d.col(sma, c, n=200), d.col(rsi, c, n=2)
    a20, a250 = d.col(atr, d.h, d.l, c, n=20), d.col(atr, d.h, d.l, c, n=250)
    vr = a20 / a250
    e = np.where((c > ma) & (r < 10), 1, np.where((c < ma) & (r > 90), -1, 0))
    eras = [("1996〜2010(初見)", "1996-03-01", "2010-10-31"), ("2010〜2018", "2010-11-01", "2018-10-31"),
            ("2018〜2026", "2018-11-01", "2026-12-31")]
    ts = lambda s: np.datetime64(s).astype("datetime64[s]").astype(np.int64)   # noqa: E731
    print("\n### 日足だけ・成行(1996〜2026、10 ペア、値動き − 費用。年率 % / t(月)/ 取引 / 年)\n")
    print("| ATR(20) / ATR(250) | " + " | ".join(n for n, _, _ in eras) + " | 全期間 |\n|---|---:|---:|---:|---:|")
    for lo, hi in BINS + [(0.0, 9.9), (1.0, 9.9)]:
        pos = hold(np.where((vr >= lo) & (vr < hi), e, 0), r > 50, r < 50, 10)
        pos = np.nan_to_num(pos)
        pos[~np.isfinite(c)] = 0
        net = np.zeros(pos.shape)
        net[1:] = pos[:-1] * np.nan_to_num(d.ret)[1:]
        prev = np.vstack([np.zeros((1, pos.shape[1])), pos[:-1]])
        turn = np.abs(pos - prev)
        net -= turn * COST
        cells = []
        for _, s0, s1 in eras + [("", "1996-03-01", "2026-12-31")]:
            m = (d.t >= ts(s0)) & (d.t <= ts(s1))
            yrs = m.sum() / 260
            tot = net[m].sum(axis=1)
            cells.append(f"{tot.sum() / yrs * 100:+.2f} / {g.month_t(tot, d.t[m]):+.2f} / {turn[m].sum() / 2 / yrs:.0f}")
        name = "全部" if (lo, hi) == (0.0, 9.9) else "1.0 以上" if (lo, hi) == (1.0, 9.9) else f"{lo}〜{hi}" if hi < 9 else f"{lo} 以上"
        print(f"| {name} | " + " | ".join(cells) + " |")


def daily_limit(limit_atr: float = 0.5, margin_pips: float = 3.0, spread_pips: float = 2.0) -> None:
    """指値の版を日足の高安だけで近似する(1 時間足の無い 1996〜2010 年を見るため)。

    bid の日足の安値はロールオーバーでスプレッドが開いた分だけ下に伸びるので、買いの指値は
    「安値 + スプレッド + margin_pips ≤ 指値」のときだけ約定させる(売りは高値 ≥ 指値 + margin_pips)。
    損切りは日足の高安で判定し、入った日も見る。手仕舞いは終値(スプレッドと滑り 0.5 pip を払う)。"""
    d = Data()
    eras = [("1996〜2010(初見)", "1996-03-01", "2010-10-31"), ("2010〜2018", "2010-11-01", "2018-10-31"),
            ("2018〜2026", "2018-11-01", "2026-12-31"), ("全期間", "1996-03-01", "2026-12-31")]
    ts = lambda x: np.datetime64(x).astype("datetime64[s]").astype(np.int64)   # noqa: E731
    rows = []
    for j, pair in enumerate(d.pairs):
        ok = np.isfinite(d.c[:, j])
        o, h, lo, c, t = d.o[ok, j], d.h[ok, j], d.l[ok, j], d.c[ok, j], d.t[ok]
        pip = 0.01 if pair.endswith("JPY") else 0.0001
        sp, mg, slip = spread_pips * pip, margin_pips * pip, 0.5 * pip
        ma, r = sma(c, 200), rsi(c, 2)
        a, a250 = atr(h, lo, c, 20), atr(h, lo, c, 250)
        i = 250
        while i < len(c) - 2:
            side = 1 if (c[i] > ma[i] and r[i] < 10) else -1 if (c[i] < ma[i] and r[i] > 90) else 0
            if side == 0 or not np.isfinite(a250[i]):
                i += 1
                continue
            lim = c[i] - side * limit_atr * a[i]
            k = i + 1
            hit = (lo[k] + sp + mg <= lim) if side > 0 else (h[k] >= lim + mg)
            if not hit:
                i += 1
                continue
            entry = min(lim, o[k] + sp) if side > 0 else max(lim, o[k])
            stop = entry - side * 3.0 * a[i]
            px = None
            for m in range(k, min(k + 10, len(c) - 1) + 1):
                if (lo[m] <= stop) if side > 0 else (h[m] + sp >= stop):
                    px = (min(stop, o[m]) - slip) if side > 0 else (max(stop, o[m] + sp) + slip)
                    break
                if (r[m] > 50 if side > 0 else r[m] < 50) or m == min(k + 10, len(c) - 1):
                    px = (c[m] - slip) if side > 0 else (c[m] + sp + slip)
                    break
            rows.append((side * (px - entry) / entry * 1e4, t[m], a[i] / a250[i], side * (px - entry) / a[i]))
            i = m + 1
    tr = np.array(rows)
    print(f"\n### 日足の高安での近似・指値 ATR x {limit_atr}(約定には {margin_pips} pip の余分な抜けが要る。"
          "1 取引 bp / t(月)/ R / 取引 / 年)\n")
    print("| ATR(20) / ATR(250) | " + " | ".join(n for n, _, _ in eras) + " |\n|---|---:|---:|---:|---:|")
    for lo_, hi_ in BINS + [(0.0, 9.9), (1.0, 9.9)]:
        cells = []
        for _, s0, s1 in eras:
            m = (tr[:, 2] >= lo_) & (tr[:, 2] < hi_) & (tr[:, 1] >= ts(s0)) & (tr[:, 1] <= ts(s1))
            yrs = (min(ts(s1), d.t[-1]) - ts(s0)) / 86400 / 365.25
            cells.append(f"{cell(tr[m, 0], tr[m, 1])} / {tr[m, 3].mean():+.2f} / {m.sum() / yrs:.0f}" if m.sum() >= 10 else "—")
        name = "全部" if (lo_, hi_) == (0.0, 9.9) else "1.0 以上" if (lo_, hi_) == (1.0, 9.9) else f"{lo_}〜{hi_}" if hi_ < 9 else f"{lo_} 以上"
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    import sys

    if "--daily-only" not in sys.argv:
        hourly()
        daily()
    daily_limit()
    daily_limit(margin_pips=10.0)
