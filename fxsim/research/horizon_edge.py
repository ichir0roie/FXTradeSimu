"""E021: 純粋な値動きの戦略が費用を越えうるのは、どの時間の長さからか。

    python -m fxsim.research.horizon_edge
    python -m fxsim.research.horizon_edge --swap-side    # スワップを受け取る向きの取引だけ(E022)

5 分から半年まで、同じ形の一番素直な順張り「直近 h の値動きの向きに入り、h 持つ」(逆張りはこの符号を逆にしたもの)を、
主要 10 ペアの実データで測る。取引は重ねない(h ごとに 1 回)。
    h(足)       データ
    5〜60 分     5 分足(2025-05〜2026-11)
    1〜24 時間   1 時間足(2010-10〜2026-11)
    1〜126 日    日足(1996〜2026-11。EURUSD・GBPUSD・USDCHF は 1996 から、ほかは 1993 から)
素の損益は mid どうしの pip。費用は今の OANDA の条件: 1 分足の平均スプレッド(ロールオーバーを除く)+ 往復 0.2 pip
+ 持つ日数分のスワップの業者の取り分(受け取り 0.44% と払い 1.10% の平均 0.77% / 年)。金利差そのもの(キャリー)は
値動きの戦略の外なので入れない。
有意さは、同じ時刻に入った全ペアの損益を足してから時刻ごとに数えた t(ペアどうしの相関を数えすぎないため)。

--swap-side: 順張りの向きがスワップを受け取る向き(金利差 x 向き − 受け取り側の取り分 0.44% > 0)のときだけ入る。
スワップは費用ではなく収入になるので、値動きの素の損益とスワップの受け取り(pip)を分けて出し、費用はスプレッドと滑りだけにする。
"""
from __future__ import annotations

import argparse

import numpy as np

from ..data import load_rates
from ..data.rates import MULTI_RATES
from ..storage import prices
from .intraday_edge import BAD_HOURS, PAIRS, load_panel

MARKUP = (0.0044 + 0.011) / 2
RECV = 0.0044


def measure(panel, h: int, bar_minutes: int, swap_side: bool = False) -> dict:
    lm = np.log(panel.mid)
    k = len(lm)
    past = np.full(lm.shape, np.nan)
    past[h:] = lm[h:] - lm[:-h]
    fwd = np.full(lm.shape, np.nan)
    fwd[:-h] = (panel.mid[h:] - panel.mid[:-h]) / panel.pip
    rows = np.arange(h, k - h, h)                  # 重ねない
    if bar_minutes < 1440:
        hour = (panel.t // 3600) % 24
        ok = ~np.isin(hour[rows], BAD_HOURS) & ~np.isin(hour[rows + h], BAD_HOURS)
        step = int(np.median(np.diff(panel.t)))
        ok &= (panel.t[rows + h] - panel.t[rows]) <= h * step + 3 * 86400      # 週末をまたぐのは許す(日中の h は除く)
        if bar_minutes * h < 1440:
            ok &= (panel.t[rows + h] - panel.t[rows]) <= h * step + 600
        rows = rows[ok]
    side = np.sign(past[rows])
    g = side * fwd[rows]
    valid = np.isfinite(g) & (side != 0)
    days = h * bar_minutes / 1440 * 7 / 5          # 持つ取引日 -> スワップの付く暦日の目安(週末の分を含む)
    yen_per_pip = np.nanmedian(panel.mid, axis=0) / panel.pip
    swap_pips = np.zeros(g.shape)
    if swap_side:
        t_ns = panel.t[rows].astype(np.int64) * 1_000_000_000
        rt = load_rates(MULTI_RATES)
        diff = np.column_stack([rt.pair_differential(p.lower(), t_ns) for p in panel.pairs])
        annual = side * diff - RECV
        valid &= annual > 0
        swap_pips = np.where(valid, annual * yen_per_pip * days / 365, 0.0)
        cost = panel.spread_pips + 0.2
    else:
        cost = panel.spread_pips + 0.2 + MARKUP * yen_per_pip * days / 365       # ペアごと
    per_time = np.where(valid, g, 0.0).sum(axis=1)
    n_time = valid.sum(axis=1)
    use = n_time > 0
    pt = per_time[use]
    t = float(pt.mean() / (pt.std(ddof=1) / np.sqrt(len(pt)))) if len(pt) > 2 and pt.std() > 0 else 0.0
    n = int(valid.sum())
    gross = float(np.nansum(np.where(valid, g, 0.0)) / max(n, 1))
    sd = float(np.nanstd(np.where(valid, fwd[rows], np.nan)))
    c = float(np.sum(np.where(valid, cost[None, :], 0.0)) / max(n, 1))
    by_pair = {p: float(g[valid[:, j], j].mean()) for j, p in enumerate(panel.pairs) if valid[:, j].any()}
    swap = float(swap_pips.sum() / max(n, 1))
    net_time = np.where(valid, g + swap_pips - cost[None, :], 0.0).sum(axis=1)[use]
    t_net = float(net_time.mean() / (net_time.std(ddof=1) / np.sqrt(len(net_time)))) \
        if len(net_time) > 2 and net_time.std() > 0 else 0.0
    return {"n": n, "gross": gross, "t": t, "sd": sd, "cost": c, "pos_pairs": sum(v > 0 for v in by_pair.values()),
            "n_pairs": len(by_pair), "swap": swap, "t_net": t_net}


def verdict(r: dict) -> str:
    if r["gross"] > r["cost"]:
        return "越える"
    return "逆張りなら越える" if -r["gross"] > r["cost"] else "越えない"


def label(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes} 分"
    if minutes < 1440:
        return f"{minutes // 60} 時間"
    return f"{minutes // 1440} 日"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap-side", action="store_true")
    a = ap.parse_args(argv)
    if a.swap_side:
        return main_swap_side()
    print("| 持つ時間 | 取引(10 ペア) | 1 取引の値動きの標準偏差(pip) | 順張りの素の損益(pip / 取引) | t "
          "| 順張りがプラスのペア "
          "| 費用(pip / 取引) | 費用 / 標準偏差 | 判定 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    plan = [(5, [1, 3, 6, 12]), (60, [1, 2, 4, 8, 24]), (1440, [1, 2, 5, 10, 21, 63, 126])]
    for tf, hs in plan:
        panel = load_panel(prices.parse_timeframe(tf), PAIRS)
        for h in hs:
            r = measure(panel, h, tf)
            print(f"| {label(h * tf)} | {r['n']:,} | {r['sd']:.1f} | {r['gross']:+.2f} | {r['t']:+.2f} | "
                  f"{r['pos_pairs']}/{r['n_pairs']} | {r['cost']:.2f} | {r['cost'] / r['sd']:.1%} | "
                  f"{verdict(r)} |")



def main_swap_side() -> None:
    print("| 持つ時間 | 取引(10 ペア) | 値動きの素の損益(pip / 取引) | t | スワップの受け取り(pip / 取引) "
          "| 費用(pip / 取引) | 差し引き(pip / 取引) | 差し引きの t |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    plan = [(5, [1, 3, 6, 12]), (60, [1, 2, 4, 8, 24]), (1440, [1, 2, 5, 10, 21, 63, 126])]
    for tf, hs in plan:
        panel = load_panel(prices.parse_timeframe(tf), PAIRS)
        for h in hs:
            r = measure(panel, h, tf, swap_side=True)
            net = r["gross"] + r["swap"] - r["cost"]
            print(f"| {label(h * tf)} | {r['n']:,} | {r['gross']:+.2f} | {r['t']:+.2f} | {r['swap']:+.2f} | "
                  f"{r['cost']:.2f} | {net:+.2f} | {r['t_net']:+.2f} |")


if __name__ == "__main__":
    main()
