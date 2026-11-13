"""E052: 指値で待つ押し目(E050・E051)を、日足未満の足(1〜8 時間足)に持ち込む。

    python -m fxsim.research.intraday_limit                  # 1・2・4・8 時間足
    python -m fxsim.research.intraday_limit --tfs 1 4 --null-seeds 20

日足の版(E050)と同じ決まりを、足だけ短くする(数字は日足のまま。結果を見て選ばない):
    信号   k 時間足の終値が 200 本の移動平均より上で RSI(2) < 10 なら買い(売りは鏡)
    入り   成行(次の 1 時間足の始値)/ 指値(信号の足の終値から ATR(20) x d だけ有利な値。信号の足 1 本ぶんの時間だけ有効)
    出     k 時間足の RSI(2) が 50 を越えた次の 1 時間足の始値で成行、または 10 本。損切りは建値から ATR x 3 の逆指値
    荒さ   信号の足の ATR(20) / ATR(250)。区切りは 0.8・1.0・1.2(E051 と同じ)
約定は 1 時間足の bid の高安と、その足の記録のスプレッドで追う(買いは ask、売りは bid。成行の滑り 0.1 pip、逆指値 0.3 pip)。
ロールオーバーの前後の足では、指値は約定させず、成行は次の足に回す。同じ足で損切りと指値の約定が重なれば損切りを先に見る。
相場の DB の主要 10 ペア、2010-10〜2026-11。発見 = 〜2018-10、確認 = 2018-11〜。
対照: 同じ回数を、でたらめな足・でたらめな向きで(--null-seeds 通り)。
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..storage import prices
from ..strategies.indicators import atr, ema, highest, lowest, rolling_std, rsi, sma
from .exit_grid import PAIRS, SPLIT, month_t
from .intraday_edge import BAD_HOURS, SOURCE

LIMITS = [0.0, 0.25, 0.5, 1.0]
BINS = [(0.0, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 9.9), (0.0, 9.9)]
MAX_BARS = 10
SLIP_MKT, SLIP_STOP = 0.1, 0.3
YEARS = 16.1
NOW = int(np.datetime64("2026-11-10").astype("datetime64[s]").astype(np.int64))


CURRENCIES = ["EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY"]      # クロスの名前の並び(前が基軸)
CROSSES = [a + b for i, a in enumerate(CURRENCIES) for b in CURRENCIES[i + 1:] if a + b not in PAIRS]
ALL = PAIRS + CROSSES + ["syn:" + x for x in PAIRS + CROSSES]


def _usd_leg(cur: str, tf: int = 60) -> dict:
    """通貨 cur の対ドルの値(1 cur = 何ドルか)の足(mid)と、相対のスプレッド。"""
    direct = cur + "USD" in PAIRS
    pair = cur + "USD" if direct else "USD" + cur
    con = prices.open_market_db()
    r = prices.read_rates(con, SOURCE, pair, tf)
    con.close()
    point = (0.01 if pair.endswith("JPY") else 0.0001) / 10
    half = r["spread"] * point / 2
    o, hi, lo, c = (r[k] + half for k in ("open", "high", "low", "close"))
    rel = r["spread"] * point / c
    if not direct:
        o, hi, lo, c = 1 / o, 1 / lo, 1 / hi, 1 / c
    return {"time": r["time"], "open": o, "high": hi, "low": lo, "close": c, "rel": rel}


def load_h1(pair: str, wick: float = 0.0, use_m5: bool = True) -> dict:
    """相場の DB の 1 時間足(bid とスプレッド)。DB に無いクロスは、対ドルの 2 本から合成する(E053)。

    合成クロス: 始値・終値は 2 本の mid の比で正確。足の中の高安は分からないので、指値・利確の約定には
    始値と終値の高い方・低い方(確かに付いた値だけ)、損切りの判定には 2 本の高安から出る一番広い値を使う
    (どちらも不利な側)。スプレッドは 2 本の相対スプレッドの和(実際のクロスの口座より広い)。"""
    name = pair.removeprefix("syn:")
    if not pair.startswith("syn:"):                      # 相場の DB にあれば本物を使う
        con = prices.open_market_db()
        h = prices.read_rates(con, SOURCE, pair, 60)
        con.close()
        if len(h["time"]):
            return h
    a, b = (CURRENCIES[4], name[3:]) if name[:3] == "USD" else (name[:3], name[3:])
    x, y = _usd_leg(name[:3]), _usd_leg(name[3:])
    t = np.intersect1d(x["time"], y["time"])
    ix, iy = np.searchsorted(x["time"], t), np.searchsorted(y["time"], t)
    o, c = x["open"][ix] / y["open"][iy], x["close"][ix] / y["close"][iy]
    spread = (x["rel"][ix] + y["rel"][iy]) * c
    pip = 0.01 if name.endswith("JPY") else 0.0001
    half = spread / 2
    hi_c, lo_c = np.maximum(o, c), np.minimum(o, c)
    hi_w, lo_w = x["high"][ix] / y["low"][iy], x["low"][ix] / y["high"][iy]
    # wick: 足の中の高安を、終値ベース(0)と一番広い値(1)の間のどこに置くか。本物の 3 クロスでは中央 0.40〜0.45、
    # 25% 点 0.21〜0.25、10% 点 0.08〜0.10(E053)。小さいほど「確かに付いた値」だけで約定を見る
    out = {"time": t, "open": o - half, "close": c - half, "high": hi_c + wick * (hi_w - hi_c) - half,
           "low": lo_c - wick * (lo_c - lo_w) - half, "high_wide": hi_w - half,
           "low_wide": lo_w - half, "spread": spread / pip * 10, "legs": (a, b)}
    if not use_m5:
        return out
    # 5 分足のある期間(2025-07〜)は、足の中の高安を 5 分足の始値・終値から作る(指値の約定を測れるようにする)
    x5, y5 = _usd_leg(name[:3], 5), _usd_leg(name[3:], 5)
    t5 = np.intersect1d(x5["time"], y5["time"])
    if len(t5):
        i5, j5 = np.searchsorted(x5["time"], t5), np.searchsorted(y5["time"], t5)
        o5, c5 = x5["open"][i5] / y5["open"][j5], x5["close"][i5] / y5["close"][j5]
        hw5, lw5 = x5["high"][i5] / y5["low"][j5], x5["low"][i5] / y5["high"][j5]
        hour = t5 // 3600 * 3600
        st = np.r_[0, np.flatnonzero(np.diff(hour)) + 1]
        hrs = hour[st]
        pos = np.searchsorted(t, hrs)
        ok = (pos < len(t)) & (t[np.minimum(pos, len(t) - 1)] == hrs)
        cnt = np.diff(np.r_[st, len(hour)])
        ok &= cnt >= 10                                   # 5 分足がほぼそろっている時間だけ
        pos = pos[ok]
        hf = half[pos]
        out["high"][pos] = np.maximum.reduceat(np.maximum(o5, c5), st)[ok] - hf
        out["low"][pos] = np.minimum.reduceat(np.minimum(o5, c5), st)[ok] - hf
        out["high_wide"][pos] = np.maximum.reduceat(hw5, st)[ok] - hf
        out["low_wide"][pos] = np.minimum.reduceat(lw5, st)[ok] - hf
    return out


FAMILIES = ["pullback", "momentum", "breakout", "breakout_trend", "fade_z", "bb_reentry", "ema_cross"]


def _prev(x: np.ndarray) -> np.ndarray:
    return np.r_[np.nan, x[:-1]]


def family_signals(fam: str, c, hi, lo, slow: int, enter: float, trend: bool):
    """型の名前 -> (入りの向き, 買いの出, 売りの出)。全部、足の終値までで決まる(E055)。

    pullback        長い向き(slow 本平均)へ戻る RSI(2) の行き過ぎ(E052 の型)。RSI(2) が 50 を越えたら出る
    momentum        長い向きへの RSI(2) の行き過ぎに乗る(pullback の逆)。RSI(2) が 50 に戻ったら出る
    breakout        直近 20 本の高値・安値を終値で抜けたら抜けた向き。反対側の 10 本のチャネルで出る
    breakout_trend  breakout のうち、長い向きと同じものだけ
    fade_z          20 本平均から 2 標準偏差離れたら逆へ。平均に戻ったら出る
    bb_reentry      2 標準偏差の外から内側へ戻った足で、平均の向きへ。平均に戻ったら出る
    ema_cross       EMA10 が EMA40 を抜けた向き。逆に抜けたら出る"""
    ma = sma(c, slow)
    up, dn = (c > ma), (c < ma)
    if not trend:
        up = dn = np.isfinite(ma)
    if fam in ("pullback", "momentum"):
        r = rsi(c, 2)
        ok = np.isfinite(ma) & np.isfinite(r)
        if fam == "pullback":
            e = np.where(ok & up & (r < enter), 1, np.where(ok & dn & (r > 100 - enter), -1, 0))
            return e, r > 50, r < 50
        e = np.where(ok & up & (r > 100 - enter), 1, np.where(ok & dn & (r < enter), -1, 0))
        return e, r < 50, r > 50
    if fam in ("breakout", "breakout_trend"):
        h20, l20 = _prev(highest(hi, 20)), _prev(lowest(lo, 20))
        h10, l10 = _prev(highest(hi, 10)), _prev(lowest(lo, 10))
        ok = np.isfinite(h20) & np.isfinite(ma)
        bu, bd = ok & (c > h20), ok & (c < l20)
        if fam == "breakout_trend":
            bu, bd = bu & (c > ma), bd & (c < ma)
        return np.where(bu, 1, np.where(bd, -1, 0)), c < l10, c > h10
    if fam in ("fade_z", "bb_reentry"):
        m, sd = sma(c, 20), rolling_std(c, 20)
        z = (c - m) / np.where(sd > 0, sd, np.nan)
        ok = np.isfinite(z) & np.isfinite(ma)
        if fam == "fade_z":
            e = np.where(ok & (z < -2), 1, np.where(ok & (z > 2), -1, 0))
        else:
            zp = _prev(z)
            e = np.where(ok & (zp < -2) & (z >= -2), 1, np.where(ok & (zp > 2) & (z <= 2), -1, 0))
        return e, z > 0, z < 0
    if fam == "ema_cross":
        d = ema(c, 10) - ema(c, 40)
        dp = _prev(d)
        ok = np.isfinite(d) & np.isfinite(dp) & np.isfinite(ma)
        return np.where(ok & (dp <= 0) & (d > 0), 1, np.where(ok & (dp >= 0) & (d < 0), -1, 0)), d < 0, d > 0
    raise ValueError(fam)


def simulate(args) -> dict:
    pair, k, seed, opts = args
    stop_atr = opts.get("stop", 3.0)
    tp_atr = opts.get("tp", 0.0)
    max_bars = opts.get("max_bars", MAX_BARS)
    enter, slow, trend = opts.get("enter", 10.0), opts.get("slow", 200), opts.get("trend", True)
    fm = opts.get("margin", 0.0)                     # 指値の約定に要る余分な抜け(pip)
    h = load_h1(pair, opts.get("wick", 0.0), opts.get("use_m5", True))
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    ht, ho, hh, hl = h["time"], h["open"], h["high"], h["low"]
    hh_stop, hl_stop = h.get("high_wide", hh), h.get("low_wide", hl)     # 損切りの判定に使う高安(合成クロスは広い側)
    sp = h["spread"] * pip / 10
    bad = np.isin((ht // 3600) % 24, BAD_HOURS)
    nh = len(ht)
    hc = h["close"]

    def next_ok(j: int) -> int:                          # 成行はロールオーバーの前後を避けて次の足へ
        while j < nh - 1 and bad[j]:
            j += 1
        return j

    def phase(off: int) -> dict:
        """区切りを off 時間ずらした k 時間足と、その信号。"""
        key = (ht + off * 3600) // (k * 3600)
        starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
        ends = np.r_[starts[1:], nh]                     # その足の次の 1 時間足の位置(= 判断が約定する足)
        c = hc[ends - 1]
        hi = np.maximum.reduceat(hh, starts)
        lo = np.minimum.reduceat(hl, starts)
        a = atr(hi, lo, c, 20)
        vr = a / atr(hi, lo, c, 250)
        e, out_l, out_s = family_signals(opts.get("family", "pullback"), c, hi, lo, slow, enter, trend)
        e = np.where(np.isfinite(vr), e, 0)
        n = len(c)
        if seed:
            rng = np.random.default_rng(1000 * seed + ALL.index(pair) + 10 * k + 7 * off)
            cnt = int((e != 0).sum())
            e = np.zeros(n, dtype=int)
            e[rng.choice(np.arange(260, n), size=cnt, replace=False)] = rng.choice([-1, 1], size=cnt)
            out_l, out_s = rng.random(n) < 0.4, rng.random(n) < 0.4
        e[:max(260, slow + 10)] = 0
        e[n - 2:] = 0
        return {"ends": ends, "c": c, "a": a, "vr": vr, "e": e, "out_l": out_l, "out_s": out_s, "n": n}

    def trade(ph: dict, i: int, d: float):
        """位相 ph の足 i の信号で 1 回。戻り値は (記録, 出た 1 時間足の位置)。指値が届かなければ None。"""
        ends, c, a = ph["ends"], ph["c"], ph["a"]
        side = int(ph["e"][i])
        j0 = int(ends[i])
        if j0 >= nh - 1:
            return None
        last = min(i + max_bars, ph["n"] - 2)
        xs = np.flatnonzero((ph["out_l"] if side > 0 else ph["out_s"])[i + 1:last + 1])
        x_bar = i + 1 + int(xs[0]) if len(xs) else last
        j_end = next_ok(min(int(ends[x_bar]), nh - 1))
        if d == 0:
            je = next_ok(j0)
            entry = ho[je] + sp[je] + SLIP_MKT * pip if side > 0 else ho[je] - SLIP_MKT * pip
        else:
            lim = c[i] - side * d * a[i]
            j1 = min(int(ends[i + 1]), j_end)
            seg = slice(j0, j1)
            reach = ((hl[seg] + sp[seg] <= lim - fm * pip) if side > 0 else (hh[seg] >= lim + fm * pip)) & ~bad[seg]
            if not reach.any():
                return None
            je = j0 + int(np.argmax(reach))
            entry = min(lim, ho[je] + sp[je]) if side > 0 else max(lim, ho[je])
        if je >= j_end:
            return None
        seg = slice(je, j_end)
        m = j_end - je
        js = jt = m
        stop = entry - side * stop_atr * a[i]
        hit = (hl_stop[seg] <= stop) if side > 0 else (hh_stop[seg] + sp[seg] >= stop)
        if hit.any():
            js = int(np.argmax(hit))
        if tp_atr > 0:
            take = entry + side * tp_atr * a[i]
            hit = ((hh[seg] >= take) if side > 0 else (hl[seg] + sp[seg] <= take)) & ~bad[seg]
            hit[0] = False                                # 入った足では利確を見ない
            if hit.any():
                jt = int(np.argmax(hit))
        if js <= jt and js < m:
            j = je + js
            px = (min(stop, ho[j]) - SLIP_STOP * pip if side > 0 else max(stop, ho[j] + sp[j]) + SLIP_STOP * pip)
        elif jt < m:
            j = je + jt
            px = max(take, ho[j]) if side > 0 else min(take, ho[j] + sp[j])
        else:
            j = j_end
            px = ho[j] - SLIP_MKT * pip if side > 0 else ho[j] + sp[j] + SLIP_MKT * pip
        pnl = side * (px - entry)
        return (pnl / entry * 1e4, ht[j], ph["vr"][i], pnl / a[i], (ht[j] - ht[je]) / 3600, pnl / pip,
                (ht[je] // 3600) % 24), j

    # rolling: 区切りを 0〜k−1 時間ずらした全部の位相を 1 つの戦略にまとめる(毎時、直近 k 時間の足で判断する)
    phases = [phase(off) for off in (range(k) if opts.get("rolling") else [int(opts.get("offset", 0))])]
    since = opts.get("since", 0)                         # この時刻(UTC の秒)以降の信号だけ
    vr_min = opts.get("vr_min", 0.0)
    vr_max = opts.get("vr_max", 99.0)                    # < 99 なら、静かな時期の信号だけ
    until = opts.get("until", 2**62)                     # この時刻(UTC の秒)より前の信号だけ                     # > 0 なら、荒い時期の信号だけを候補にする(1 度に 1 建玉なので結果が変わる)
    cands = sorted((int(ph["ends"][i]), pi, int(i)) for pi, ph in enumerate(phases)
                   for i in np.flatnonzero((ph["e"] != 0) & (np.nan_to_num(ph["vr"]) >= vr_min)
                                   & (np.nan_to_num(ph["vr"], nan=99.0) < vr_max))
                   if since <= ht[min(int(ph["ends"][i]), nh - 1)] < until)
    res = {}
    for d in LIMITS:
        tr = []
        busy = -1                                         # この 1 時間足までは建玉がある
        for j0, pi, i in cands:
            if j0 <= busy:
                continue
            got = trade(phases[pi], i, d)
            if got is None:
                continue
            tr.append(got[0])
            busy = got[1]
        res[d] = np.array(tr).reshape(-1, 7)
    return res


def stat(tr: np.ndarray) -> dict:
    v, t = tr[:, 0], tr[:, 1]
    a, b = t < SPLIT, t >= SPLIT
    f = lambda x: float(x.mean()) if len(x) else 0.0      # noqa: E731
    return {"n": len(v), "avg": f(v), "t": month_t(v, t), "a": f(v[a]), "ta": month_t(v[a], t[a]), "b": f(v[b]),
            "tb": month_t(v[b], t[b]), "win": f(v > 0), "r": f(tr[:, 3]), "hours": f(tr[:, 4]), "pips": f(tr[:, 5])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--null-seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--stop", type=float, default=3.0)
    ap.add_argument("--tp", type=float, default=0.0)
    ap.add_argument("--max-bars", type=int, default=MAX_BARS)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--rolling", action="store_true", help="区切りをずらした全部の位相をまとめる(毎時判断)")
    ap.add_argument("--vr-min", type=float, default=0.0)
    ap.add_argument("--pairs", nargs="+", default=None, help="ペア(既定は主要 10)。crosses = 合成クロス 18 本")
    ap.add_argument("--wick", type=float, default=0.0, help="合成クロスの足の中の高安の位置(0〜1)")
    ap.add_argument("--no-m5", action="store_true", help="合成クロスの高安に 5 分足を使わない")
    ap.add_argument("--since", default=None, help="この日付以降の信号だけ(例 2025-07-15)")
    ap.add_argument("--only-high", action="store_true", help="ATR 比 1.2 以上と全部の行だけ出す")
    args = ap.parse_args()
    opts = {"stop": args.stop, "tp": args.tp, "max_bars": args.max_bars, "margin": args.margin, "rolling": args.rolling,
            "vr_min": args.vr_min, "wick": args.wick, "use_m5": not args.no_m5,
            "since": int(np.datetime64(args.since).astype("datetime64[s]").astype(np.int64)) if args.since else 0}
    pairs = (PAIRS if not args.pairs else CROSSES if args.pairs == ["crosses"]
             else ["syn:" + x for x in CROSSES] if args.pairs == ["syn-crosses"] else args.pairs)
    jobs = [(p, k, s, opts) for k in args.tfs for s in range(args.null_seeds + 1) for p in pairs]
    with ProcessPoolExecutor(args.workers) as ex:
        outs = list(ex.map(simulate, jobs, chunksize=2))
    global YEARS
    if args.since:
        YEARS = (NOW - opts["since"]) / 86400 / 365.25
    print(f"損切り ATR x {args.stop}、利確 {'なし' if args.tp == 0 else f'ATR x {args.tp}'}、最長 {args.max_bars} 本、指値の余分な抜け {args.margin} pip")
    for k in args.tfs:
        print(f"\n## {k} 時間足({len(pairs)} ペア: {' '.join(pairs) if args.pairs else '主要 10 ペア'}、2010-10〜2026-11)\n")
        print("| 入り | ATR(20) / ATR(250) | 取引 / 年 | 持つ時間 | 勝率 | 1 取引 pip | bp / t(月) | R | 発見 bp / t | 確認 bp / t "
              "| でたらめの bp(平均 ± 振れ) | z | プラスのペア |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for d in LIMITS:
            per = [o[d] for j, o in zip(jobs, outs, strict=True) if j[1] == k and j[2] == 0]
            tr = np.vstack(per)
            nulls = [np.vstack([o[d] for j, o in zip(jobs, outs, strict=True) if j[1] == k and j[2] == s])
                     for s in range(1, args.null_seeds + 1)]
            for lo, hi in BINS:
                m = (tr[:, 2] >= lo) & (tr[:, 2] < hi)
                if m.sum() < 20 or (args.only_high and lo not in (1.2,) and (lo, hi) != (0.0, 9.9)):
                    continue
                s = stat(tr[m])
                nm = np.array([n_[(n_[:, 2] >= lo) & (n_[:, 2] < hi), 0].mean() for n_ in nulls]) if nulls else np.zeros(1)
                z = (s["avg"] - nm.mean()) / nm.std(ddof=1) if len(nm) > 2 and nm.std() > 0 else float("nan")
                pp = sum(x[(x[:, 2] >= lo) & (x[:, 2] < hi), 0].sum() > 0 for x in per)
                name = "全部" if (lo, hi) == (0.0, 9.9) else f"{lo}〜{hi}" if hi < 9 else f"{lo} 以上"
                print(f"| {'成行' if d == 0 else f'指値 {d}'} | {name} | {s['n'] / YEARS:.0f} | {s['hours']:.1f} | {s['win']:.0%} "
                      f"| {s['pips']:+.2f} | {s['avg']:+.2f} / {s['t']:+.2f} | {s['r']:+.3f} | {s['a']:+.2f} / {s['ta']:+.2f} "
                      f"| {s['b']:+.2f} / {s['tb']:+.2f} | {nm.mean():+.2f} ± {nm.std(ddof=1) if len(nm) > 1 else 0:.2f} | {z:+.2f} | {pp}/{len(per)} |")


if __name__ == "__main__":
    main()
