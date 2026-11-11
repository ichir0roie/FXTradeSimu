"""E020: 分足の上下(短い時間の平均回帰)を取れるか。実データ(相場の DB)の 1 分足・5 分足、主要 10 ペアで測る。

    python -m fxsim.research.minute_reversion --tf M5
    python -m fxsim.research.minute_reversion --tf M1

数時間の偏りが無くても、その間の上下を取れるかは「行き過ぎたら戻る癖」があるかで決まる。値動きがでたらめなら、
安く買って高く売れる波と、買ってさらに下がる波が同じだけあり、取れる分は 0(費用の分だけ負ける)。

1. 分散比 VR(k) = k 本の値動きの分散 / (k x 1 本の分散)。1 より小さければ戻る癖(平均回帰)、大きければ続く癖。
   「戻る癖で取れる大きさ」の目安として、k 本の値動きの標準偏差 x (1 − √VR) も pip で出す
2. 逆張りの事象: 足 i の終値が、直近 n 本の平均から z 標準偏差以上離れたら逆に入り、h 本後の終値で出る。
   素の損益(mid、pip)と、その時刻の費用(1 分足のサーバーの時ごとの平均スプレッド + 往復 0.2 pip)
3. 選ぶ段と確かめる段: 期間の前半で素の損益 − 費用が一番良かった組み合わせを、後半でもう一度測る

形(n・z・h・時間帯)は結果を見る前に決めた小さな格子。試した数は表の行数。ロールオーバーの前後(UTC 20〜22 時)の
入り・出は数えない(E015 の作り物)。有意さは日ごとにまとめた損益の t。
"""
from __future__ import annotations

import argparse

import numpy as np

from ..calendar import server_hour
from ..markets.components.real import _hourly_spread
from ..storage import prices
from .intraday_edge import BAD_HOURS, PAIRS, load_panel

SESSIONS = {"東京 0-6": range(0, 7), "ロンドン 7-11": range(7, 12), "NY 12-16": range(12, 17),
            "夜 17-19": range(17, 20)}


def cost_pips(panel, slip_pips: float = 0.1) -> np.ndarray:
    """(足 x ペア)の往復の費用(pip): その時刻のサーバーの時の平均スプレッド + 往復の滑り。"""
    out = np.zeros(panel.mid.shape)
    t_ns = panel.t.astype(np.int64) * 1_000_000_000
    for j, p in enumerate(panel.pairs):
        found = _hourly_spread(p)
        if found is None:
            out[:, j] = panel.spread_pips[j] + 2 * slip_pips
            continue
        prof, rule = found
        out[:, j] = prof[server_hour(t_ns, rule)] / panel.pip[j] + 2 * slip_pips
    return out


def variance_ratios(panel, ks: list[int]) -> dict:
    lm = np.log(panel.mid)
    r1 = np.diff(lm, axis=0)
    hour = (panel.t[1:] // 3600) % 24
    ok = ~np.isin(hour, BAD_HOURS)
    out = {}
    for j, p in enumerate(panel.pairs):
        x = r1[:, j][ok & np.isfinite(r1[:, j])]
        v1 = np.var(x)
        row = {}
        for k in ks:
            rk = lm[k:, j] - lm[:-k, j]
            hk = (panel.t[k:] // 3600) % 24
            sel = np.isfinite(rk) & ~np.isin(hk, BAD_HOURS)
            vk = np.var(rk[sel])
            vr = vk / (k * v1)
            sd_pips = np.sqrt(vk) * np.nanmedian(panel.mid[:, j]) / panel.pip[j]
            row[k] = (float(vr), float(sd_pips * (1 - np.sqrt(min(vr, 1.0)))))
        out[p] = row
    return out


def reversion_events(panel, n: int, z: float, h: int, cost: np.ndarray, sel_rows: np.ndarray) -> dict:
    """逆張りの事象の、ペアと時間帯ごとの (取引数, 素の損益の和, 費用の和, 日ごとの損益)。"""
    c = panel.mid
    lm = np.log(c)
    k = len(c)
    mean = np.full(c.shape, np.nan)
    sd = np.full(c.shape, np.nan)
    cs = np.nancumsum(np.nan_to_num(lm), axis=0)
    cs2 = np.nancumsum(np.nan_to_num(lm) ** 2, axis=0)
    mean[n:] = (cs[n:] - cs[:-n]) / n
    sd[n:] = np.sqrt(np.maximum((cs2[n:] - cs2[:-n]) / n - mean[n:] ** 2, 0))
    dev = (lm - mean) / np.where(sd > 0, sd, np.nan)
    side = np.where(dev > z, -1, np.where(dev < -z, 1, 0))
    fwd = np.full(c.shape, np.nan)
    fwd[:-h] = (c[h:] - c[:-h]) / panel.pip
    hour = (panel.t // 3600) % 24
    bad = np.isin(hour, BAD_HOURS)
    bad_exit = np.ones(k, bool)
    bad_exit[:-h] = bad[h:]
    gap = np.ones(k, bool)
    step = int(np.median(np.diff(panel.t)))
    gap[:-h] = (panel.t[h:] - panel.t[:-h]) > h * step + 600
    take = (side != 0) & np.isfinite(fwd) & ~(bad | bad_exit | gap)[:, None] & sel_rows[:, None]
    # 重ならないように: 同じペアで前の取引を持っている間(h 本)は入らない
    for j in range(c.shape[1]):
        idx = np.flatnonzero(take[:, j])
        keep = np.zeros(len(idx), bool)
        last = -10 ** 9
        for m, i in enumerate(idx):
            if i >= last + h:
                keep[m] = True
                last = i
        take[idx[~keep], j] = False
    gross = np.where(take, side * np.nan_to_num(fwd), 0.0)
    cst = np.where(take, cost, 0.0)
    out = {}
    days = panel.t // 86400
    for sname, hrs in SESSIONS.items():
        hs = np.isin(hour, list(hrs))
        for j, p in enumerate(panel.pairs):
            m = take[:, j] & hs
            out[(p, sname)] = (int(m.sum()), float(gross[m, j].sum()), float(cst[m, j].sum()),
                               days[m], gross[m, j] - cst[m, j])
    return out


def _t_by_day(days: np.ndarray, net: np.ndarray) -> float:
    if len(days) < 3:
        return 0.0
    u, inv = np.unique(days, return_inverse=True)
    d = np.zeros(len(u))
    np.add.at(d, inv, net)
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 and d.std() > 0 else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="M5")
    a = ap.parse_args(argv)
    tf = prices.parse_timeframe(a.tf)
    panel = load_panel(tf, PAIRS)
    print(f"{a.tf}: {len(panel.pairs)} pairs, {len(panel.t)} bars, {np.datetime64(int(panel.t[0]), 's')} .. "
          f"{np.datetime64(int(panel.t[-1]), 's')}")
    cost = cost_pips(panel)
    # 1. 分散比
    ks = [k for k in (1, 3, 6, 12, 36) if k > 1] if tf == 5 else [5, 15, 30, 60]
    vr = variance_ratios(panel, ks)
    print("\n## 分散比(1 より小さい = 戻る癖)と、戻る癖で取れる大きさの目安(pip)\n")
    print("| ペア | " + " | ".join(f"{k * tf} 分" for k in ks) + " | 平均の費用(pip) |")
    print("|---|" + "---:|" * (len(ks) + 1))
    for j, p in enumerate(panel.pairs):
        cells = " | ".join(f"{vr[p][k][0]:.3f}({vr[p][k][1]:.2f})" for k in ks)
        print(f"| {p} | {cells} | {np.nanmean(cost[:, j]):.2f} |")
    # 2・3. 逆張りの事象(前半で選び、後半で確かめる)
    half = panel.t >= panel.t[0] + (panel.t[-1] - panel.t[0]) // 2
    # 平均を取る長さ n と持つ長さ h は分で決め、足の本数に直す
    # (5 分足: n 60・180 分、h 15・30・60 分。1 分足: n 15・60 分、h 5・15・30 分)
    minutes = ((60, 180), (15, 30, 60)) if tf == 5 else ((15, 60), (5, 15, 30))
    grid = [(n // tf, z, h // tf) for n in minutes[0] for z in (2.0, 3.0) for h in minutes[1]]
    rows = []
    for n, z, h in grid:
        first = reversion_events(panel, n, z, h, cost, ~half)
        second = reversion_events(panel, n, z, h, cost, half)
        for key in first:
            a1, a2 = first[key], second[key]
            if a1[0] < 30:
                continue
            rows.append(dict(n=n, z=z, h=h, pair=key[0], sess=key[1], n1=a1[0], g1=a1[1] / a1[0], c1=a1[2] / a1[0],
                             t1=_t_by_day(a1[3], a1[4]), n2=a2[0], g2=a2[1] / max(a2[0], 1),
                             c2=a2[2] / max(a2[0], 1), t2=_t_by_day(a2[3], a2[4])))
    print(f"\n## 逆張りの事象(試した組み合わせ {len(rows)}、ペア x 時間帯 x n・z・h)\n")
    print("素の損益 − 費用(pip / 取引)が前半で一番良かった 15 個と、その後半:\n")
    print("| n(分) | z | 持つ(分) | ペア | 時間帯 | 前半 取引 | 前半 素の損益 | 前半 費用 | 前半 t "
          "| 後半 取引 | 後半 素の損益 | 後半 費用 | 後半 t |")
    print("|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows.sort(key=lambda r: r["g1"] - r["c1"], reverse=True)
    for r in rows[:15]:
        print(f"| {r['n'] * tf} | {r['z']:g} | {r['h'] * tf} | {r['pair']} | {r['sess']} | {r['n1']} | "
              f"{r['g1']:+.2f} | "
              f"{r['c1']:.2f} | {r['t1']:+.2f} | {r['n2']} | {r['g2']:+.2f} | {r['c2']:.2f} | {r['t2']:+.2f} |")
    g = np.array([r["g1"] for r in rows])
    w = np.array([r["n1"] for r in rows])
    g2 = np.average([r["g2"] for r in rows], weights=[max(r["n2"], 1) for r in rows])
    print(f"\n全組み合わせの素の損益の取引数での加重平均: 前半 {np.average(g, weights=w):+.2f} pip、"
          f"後半 {g2:+.2f} pip。"
          f"費用を越えた組み合わせ(前半): {sum(r['g1'] > r['c1'] for r in rows)} / {len(rows)}、"
          f"そのうち後半も越えた: {sum(r['g1'] > r['c1'] and r['g2'] > r['c2'] for r in rows)}")


if __name__ == "__main__":
    main()
