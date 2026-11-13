"""E015: 数時間で手仕舞う売買に、費用を越える素の偏りがあるかを、実データ(相場の DB の MT5 の足)でまとめて測る。

バックテストの前のふるい。足 i の終値で向きを決め、足 i + H の終値で手仕舞う、と考えたときの
「向き x 先の値動き」(mid、pip)を、案ごと・ペアごとに数える。費用は別に、ペアごとの実際のスプレッドから見積もる。

    python -m fxsim.research.intraday_edge                 # 1 時間足(2010〜)
    python -m fxsim.research.intraday_edge --tf M5 --hold 12,24,48

有意さは日ごとにまとめた損益の t(同じ日の取引・重なった建玉は独立でないので、取引単位では数えない)。
案は結果を見る前に決めた形だけ(しきい値の探索はしない。試した数は表の行数)。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from ..storage import prices

SOURCE = "OANDA-Japan MT5 Demo"
PAIRS = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD", "EURJPY", "GBPJPY", "AUDJPY"]
CURRENCIES = ["USD", "EUR", "GBP", "AUD", "NZD", "CHF", "CAD", "JPY"]
ROLLOVER_UTC = 21   # ロールオーバー(夏 21 時・冬 22 時 UTC)。スプレッドが 5〜6 倍に広がる
# 入りと出の足がこの時間(足の始まりの UTC の時)にかかる取引は数えない。足の記録は bid と足の中の最小スプレッドなので、
# ロールオーバーの前後は mid が低く出て、「前に売って後に買い戻す」が作り物の勝ちになる(E015)
BAD_HOURS = (20, 21, 22)


def pip_of(sym: str) -> float:
    return 0.01 if sym.endswith("JPY") else 0.0001


@dataclass
class Panel:
    pairs: list[str]
    t: np.ndarray            # (n,) 足の始まり(UTC 秒)
    mid: np.ndarray          # (n, k) 終値の mid。欠けは nan
    pip: np.ndarray          # (k,)
    spread_pips: np.ndarray  # (k,) 1 分足の平均スプレッド(ロールオーバーを除く)。費用の見積もり


def load_panel(tf: int, pairs: list[str], since: int | None = None) -> Panel:
    con = prices.open_market_db()
    data = {}
    for p in pairs:
        r = prices.read_rates(con, SOURCE, p, tf, start=since)
        if len(r["time"]):
            data[p] = r
    pairs = list(data)
    t = np.unique(np.concatenate([data[p]["time"] for p in pairs]))
    mid = np.full((len(t), len(pairs)), np.nan)
    pip = np.array([pip_of(p) for p in pairs])
    spread = np.zeros(len(pairs))
    for j, p in enumerate(pairs):
        r = data[p]
        point = 0.001 if p.endswith("JPY") else 0.00001
        idx = np.searchsorted(t, r["time"])
        mid[idx, j] = r["close"] + r["spread"] * point / 2
        m1 = prices.read_rates(con, SOURCE, p, 1)
        h = (m1["time"] // 3600) % 24
        ok = (h != ROLLOVER_UTC) & (h != ROLLOVER_UTC + 1)
        spread[j] = float(np.mean(m1["spread"][ok])) * point / pip[j] if ok.any() else np.nan
    con.close()
    return Panel(pairs, t, mid, pip, spread)


def currency_strength(logret: np.ndarray, pairs: list[str]) -> np.ndarray:
    """ペアの log リターン(n x k)から通貨の強さ(n x 8、和 0)。r(ペア) = s(基軸) − s(決済) の最小二乗。"""
    a = np.zeros((len(pairs), len(CURRENCIES)))
    for j, p in enumerate(pairs):
        a[j, CURRENCIES.index(p[:3])] += 1
        a[j, CURRENCIES.index(p[3:])] -= 1
    a = np.vstack([a, np.ones(len(CURRENCIES))])           # 和 0 の制約
    pinv = np.linalg.pinv(a)[:, :len(pairs)]
    x = np.nan_to_num(logret) @ pinv.T
    x[~np.isfinite(logret).all(axis=1)] = np.nan
    return x


def fitted_pair(strength: np.ndarray, pairs: list[str]) -> np.ndarray:
    return np.column_stack([strength[:, CURRENCIES.index(p[:3])] - strength[:, CURRENCIES.index(p[3:])]
                            for p in pairs])


def lag_logret(logmid: np.ndarray, n: int) -> np.ndarray:
    out = np.full(logmid.shape, np.nan)
    out[n:] = logmid[n:] - logmid[:-n]
    return out


def signals(panel: Panel, bars_per_hour: int) -> dict[str, np.ndarray]:
    """案ごとの向き(n x k、+1 / −1 / 0)。どれも足 i の終値までで決まる。"""
    lm = np.log(panel.mid)
    hb = bars_per_hour
    t = panel.t
    hour = (t // 3600) % 24
    out: dict[str, np.ndarray] = {}
    vol = np.full(lm.shape, np.nan)     # 1 本の log リターンの標準偏差(直近 20 日相当)
    r1 = lag_logret(lm, 1)
    w = 20 * 24 * hb
    csum = np.nancumsum(np.nan_to_num(r1) ** 2, axis=0)
    vol[w:] = np.sqrt((csum[w:] - csum[:-w]) / w)
    for L in (1, 4, 12):
        r = lag_logret(lm, L * hb)
        z = r / (vol * np.sqrt(L * hb))
        # 1 ペアごとの時系列(反転)。大きく動いたときだけ
        out[f"ts_rev_{L}h_z1"] = np.where(z > 1, -1, np.where(z < -1, 1, 0))
        # 通貨の強さの順位(反転): 一番強い通貨を売り、一番弱い通貨を買う向きのペアだけ
        s = currency_strength(r, panel.pairs)
        f = fitted_pair(s, panel.pairs)
        fz = f / (vol * np.sqrt(L * hb))
        out[f"xs_rev_{L}h_z1"] = np.where(fz > 1, -1, np.where(fz < -1, 1, 0))
        # 同じ形の順張り(反転の逆。比べるため)は、表の符号を逆に読めばよいので作らない
    # 時間帯の癖: 同じ時刻(UTC の時)の、過去 2 年の先 H 時間の平均の向き(H は呼ぶ側で決めるので 1 時間で代表)
    fwd1 = np.vstack([lm[hb:] - lm[:-hb], np.full((hb, lm.shape[1]), np.nan)])
    season = np.zeros(lm.shape, dtype=int)
    day_bars = 24 * hb
    look = 2 * 260
    for i in range(len(t)):
        # 同じ時刻の過去の足(1 日ずつ遡る)。足 i の終値で分かる先の値動きは i − hb まで
        js = np.arange(i - day_bars, max(i - day_bars * look, 0) - 1, -day_bars)
        js = js[(js >= 0) & (js + hb <= i)]
        if len(js) < 200:
            continue
        x = fwd1[js]
        m = np.nanmean(x, axis=0)
        se = np.nanstd(x, axis=0) / np.sqrt(np.isfinite(x).sum(axis=0))
        season[i] = np.where(m > 2 * se, 1, np.where(m < -2 * se, -1, 0))
    out["season_hour_t2"] = season
    # セッションの勢い: 東京(0〜7 時 UTC)の動きの向きに、ロンドンの始まり(7 時)で乗る / 逆らう
    tok = lag_logret(lm, 7 * hb)
    sess = np.zeros(lm.shape, dtype=int)
    at7 = (hour == 6) & ((t // 60) % 60 == (60 - 60 // hb) % 60)   # 6 時台の最後の足の終値 = 7 時
    sess[at7] = np.sign(np.nan_to_num(tok[at7])).astype(int)
    out["session_tokyo_mom"] = sess
    return out


def evaluate(panel: Panel, sig: np.ndarray, hold_bars: int, slip_pips: float) -> dict:
    lm_mid = panel.mid
    fwd = np.full(lm_mid.shape, np.nan)
    fwd[:-hold_bars] = (lm_mid[hold_bars:] - lm_mid[:-hold_bars]) / panel.pip     # pip
    # 週をまたぐ建玉は数えない(週末の窓)
    gap = np.zeros(len(panel.t), bool)
    gap[:-hold_bars] = (panel.t[hold_bars:] - panel.t[:-hold_bars]) > (hold_bars * (panel.t[1] - panel.t[0]) + 6 * 3600)
    # 入り(足 i の終値)と出(足 i + H の終値)の足がロールオーバーの時間にかかる取引は数えない
    hour = (panel.t // 3600) % 24
    bad = np.isin(hour, BAD_HOURS)
    bad_exit = np.ones(len(bad), bool)
    bad_exit[:-hold_bars] = bad[hold_bars:]
    take = (sig != 0) & np.isfinite(fwd) & ~gap[:, None] & ~(bad | bad_exit)[:, None]
    pnl = np.where(take, sig * np.nan_to_num(fwd), 0.0)
    n = int(take.sum())
    days = panel.t // 86400
    ud, inv = np.unique(days, return_inverse=True)
    daily = np.zeros(len(ud))
    np.add.at(daily, inv, pnl.sum(axis=1))
    active = np.zeros(len(ud))
    np.add.at(active, inv, take.sum(axis=1))
    dd = daily[active > 0]
    t_day = float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))) if len(dd) > 2 else np.nan
    cost = (panel.spread_pips[None, :] + 2 * slip_pips) * take
    by_pair = {p: (float(pnl[:, j].sum() / max(take[:, j].sum(), 1)), int(take[:, j].sum()))
               for j, p in enumerate(panel.pairs)}
    return {"trades": n, "per_day": n / max(len(ud), 1), "gross": float(pnl.sum() / max(n, 1)),
            "cost": float(cost.sum() / max(n, 1)), "t_day": t_day, "by_pair": by_pair}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--hold", default="1,2,4,8", help="手仕舞いまでの時間(時間)")
    ap.add_argument("--since", default=None, help="この日付から(UTC、YYYY-MM-DD)")
    ap.add_argument("--slip", type=float, default=0.1, help="片道の滑り(pip)")
    ap.add_argument("--pairs", action="store_true", help="ペアごとの内訳も出す")
    a = ap.parse_args(argv)
    tf = prices.parse_timeframe(a.tf)
    since = int(np.datetime64(a.since, "s").astype(np.int64)) if a.since else None
    panel = load_panel(tf, PAIRS, since)
    hb = 60 // tf
    print(f"{a.tf}: {len(panel.pairs)} pairs, {len(panel.t)} bars, "
          f"{np.datetime64(int(panel.t[0]), 's')} .. {np.datetime64(int(panel.t[-1]), 's')}")
    spreads = {p: round(float(s), 2) for p, s in zip(panel.pairs, panel.spread_pips, strict=True)}
    print("spread (pip, M1 mean):", spreads)
    sigs = signals(panel, hb)
    print("| 案 | 持つ時間 | 取引 / 日 | 素の損益 pip / 取引 | 費用 pip / 取引 | 差 | t(日ごと) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, s in sigs.items():
        for h in [int(x) for x in a.hold.split(",")]:
            r = evaluate(panel, s, h * hb, a.slip)
            print(f"| {name} | {h}h | {r['per_day']:.1f} | {r['gross']:+.2f} | {r['cost']:.2f} | "
                  f"{r['gross'] - r['cost']:+.2f} | {r['t_day']:+.2f} |")
            if a.pairs:
                cells = ", ".join(f"{p} {g:+.1f} ({n})" for p, (g, n) in r["by_pair"].items())
                print(f"|  |  |  | {cells} | | | |")


if __name__ == "__main__":
    main()
