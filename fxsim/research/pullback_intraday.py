"""E049: 押し目(S048)と同じ決まりを、5 分足〜1 時間足で両方向に回すとどうなるか。

    python -m fxsim.research.pullback_intraday

決まりは日足と同じ: 終値が slow 本の移動平均より上で RSI(2) が 10 を割ったら買い(売りは鏡)、RSI(2) が 50 を越えるか
10 本で出る。約定は信号の足の終値(mid)。費用は入りと出の足の記録のスプレッドの平均 + 往復 0.2 pip。
ロールオーバーの前後の足では入らない・出ない(出は次の足に回す)。
    5 分足・15 分足   相場の DB の 5 分足(2025-07〜2026-11)。15 分足は 5 分足から作る
    1 時間足・4 時間足 相場の DB の 1 時間足(2010-10〜2026-11)。4 時間足は 1 時間足から作る
t は、全ペアの損益を日ごとに足してから数える。
"""
# ruff: noqa: E501
from __future__ import annotations

import numpy as np

from ..storage import prices
from ..strategies.indicators import rsi, sma
from .intraday_edge import BAD_HOURS, SOURCE

PAIRS = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD", "EURJPY", "GBPJPY", "AUDJPY"]


def resample(t, c, sp, k):
    """k 本ごとにまとめる(終値と、最後の足のスプレッド)。区切りは時刻で決める。"""
    step = int(np.median(np.diff(t))) * k
    key = t // step
    last = np.r_[np.flatnonzero(np.diff(key)), len(t) - 1]
    return t[last], c[last], sp[last]


def trades(t, c, sp, pip, slow, enter=10.0, exit_=50.0, max_bars=10, trend=True):
    ma, r = sma(c, slow), rsi(c, 2)
    bad = np.isin((t // 3600) % 24, BAD_HOURS)
    out = []
    cur, age, i0 = 0, 0, 0
    for i in range(slow, len(c)):
        if cur != 0:
            age += 1
            if ((r[i] > exit_ if cur > 0 else r[i] < 100 - exit_) or age >= max_bars) and not bad[i]:
                out.append((cur * (c[i] - c[i0]) / pip, (sp[i] + sp[i0]) / 2 + 0.2, t[i], i - i0))
                cur = 0
        if cur == 0 and not bad[i] and np.isfinite(ma[i]) and np.isfinite(r[i]):
            up = c[i] > ma[i] if trend else True
            dn = c[i] < ma[i] if trend else True
            e = 1 if (up and r[i] < enter) else -1 if (dn and r[i] > 100 - enter) else 0
            if e:
                cur, age, i0 = e, 0, i
    return out


def day_t(net, t) -> float:
    day = (np.asarray(t) // 86400).astype(int)
    s = np.bincount(day - day.min(), weights=net)[np.unique(day - day.min())]
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 2 and s.std() > 0 else 0.0


def main() -> None:
    con = prices.open_market_db()
    rows = {}
    for label, tf, k, slow in (("5 分足", 5, 1, 200), ("15 分足", 5, 3, 200), ("1 時間足", 60, 1, 200),
                               ("4 時間足", 60, 4, 200), ("5 分足・平均なし(ただの逆張り)", 5, 1, 0),
                               ("1 時間足・平均なし(ただの逆張り)", 60, 1, 0)):
        acc = []
        years = 0.0
        for p in PAIRS:
            r = prices.read_rates(con, SOURCE, p, tf)
            pip = 0.01 if p.endswith("JPY") else 0.0001
            point = pip / 10
            t, c, sp = r["time"], r["close"] + r["spread"] * point / 2, r["spread"] * point / pip
            if k > 1:
                t, c, sp = resample(t, c, sp, k)
            years = max(years, (t[-1] - t[0]) / 86400 / 365.25)
            acc += trades(t, c, sp, pip, slow or 200, trend=bool(slow))
        a = np.array(acc)
        rows[label] = (a, years)
    con.close()
    print("押し目(RSI(2) < 10、50 で出る、両方向)を足の長さを変えて。10 ペアの合計、1 取引の pip")
    print("| 足 | 期間(年) | 取引 / 年(10 ペア) | 持つ本数 | 勝率(費用の前) | 素の損益 | 費用 | 差し引き | t(日) | 年の費用(pip) | 前半 | 後半 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, (a, years) in rows.items():
        net = a[:, 0] - a[:, 1]
        h = a[:, 2] < np.median(a[:, 2])
        print(f"| {label} | {years:.1f} | {len(a) / years:,.0f} | {a[:, 3].mean():.1f} | {(a[:, 0] > 0).mean():.0%} "
              f"| {a[:, 0].mean():+.2f} | {a[:, 1].mean():.2f} | {net.mean():+.2f} | {day_t(net, a[:, 2]):+.2f} "
              f"| {a[:, 1].sum() / years:,.0f} | {net[h].mean():+.2f} | {net[~h].mean():+.2f} |")


if __name__ == "__main__":
    main()
