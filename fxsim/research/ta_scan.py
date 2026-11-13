"""E048: 日足のテクニカルの型を、主要 10 ペアの実データで広くふるいにかける。

    python -m fxsim.research.ta_scan            # 全部の型
    python -m fxsim.research.ta_scan --top 40   # 上位だけ

これまでに落ちた型(両方向の順張り・1 時間足以下)の外を探すためのふるい。合否の判定ではない(判定は verify と pair-eval)。
建玉は足 t の終値で決め、t → t+1 の値動きを取る(因果的)。単位は建玉比の年率 %。

    値動き    建玉 x 翌日の log リターン
    費用      建玉の変化 1 単位あたり 0.015%(往復 0.03%。OANDA・約定 2 時で 3〜4.5 pip、E017)
    スワップ  建玉 x 金利差 − 業者の取り分(受け取り 0.44%、払い 1.10%)。判定には入れない(値動きの優位性を探すので)

発見(1996〜2013 年)と確認(2014〜2026 年)に分け、両方で「値動き − 費用」がプラスのものだけを候補にする。
t はペアを足してから月ごとに数える(同じ月のペアどうしは相関するので)。
試した数が多いので、片方の期間だけの t 2〜3 は偶然と区別しない。
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np

from ..data import load_rates
from ..data.rates import MULTI_RATES
from ..storage import prices
from ..strategies.indicators import adx, atr, ema, highest, lowest, macd, rolling_std, rsi, sma, stochastic
from .intraday_edge import CURRENCIES, SOURCE

PAIRS = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD", "EURJPY", "GBPJPY", "AUDJPY"]
COST = 0.00015            # 建玉の変化 1 単位あたり(往復 0.03%)
RECV, PAY = 0.0044, 0.011
SPLIT = np.datetime64("2014-01-01").astype("datetime64[s]").astype(np.int64)
FIRST_DAY = int(np.datetime64("1994-06-01").astype("datetime64[D]").astype(np.int64))
START = np.datetime64("1996-03-01").astype("datetime64[s]").astype(np.int64)


class Data:
    def __init__(self, pairs=PAIRS):
        con = prices.open_market_db()
        raw = {p: prices.read_rates(con, SOURCE, p, 1440) for p in pairs}
        con.close()
        # 足の始まりは 21 / 22 時 UTC。日付(翌日の暦日)にそろえる
        day = {p: (raw[p]["time"] + 6 * 3600) // 86400 for p in pairs}
        days = np.unique(np.concatenate(list(day.values())))
        days = days[days >= FIRST_DAY]
        raw = {p: {k: v[day[p] >= FIRST_DAY] for k, v in raw[p].items()} for p in pairs}
        day = {p: v[v >= FIRST_DAY] for p, v in day.items()}
        self.pairs, self.t = pairs, days * 86400
        n, k = len(days), len(pairs)
        self.o, self.h, self.l, self.c = (np.full((n, k), np.nan) for _ in range(4))
        for j, p in enumerate(pairs):
            idx = np.searchsorted(days, day[p])
            for name, arr in (("open", self.o), ("high", self.h), ("low", self.l), ("close", self.c)):
                arr[idx, j] = raw[p][name]
        self.ret = np.full((n, k), np.nan)
        self.ret[1:] = np.log(self.c[1:] / self.c[:-1])
        rt = load_rates(MULTI_RATES)
        t_ns = self.t.astype(np.int64) * 1_000_000_000
        self.diff = np.column_stack([rt.pair_differential(p, t_ns) for p in pairs])
        self.dow = ((days + 3) % 7).astype(int)          # 0 = 月曜
        self.month = (days // 30.4375).astype(int)
        self.gap = np.r_[1, np.diff(days)]

    def col(self, f, *arrs, **kw):
        """1 ペアずつの指標を全ペアに。欠けている足は飛ばして計算し、元の位置に戻す。"""
        out = None
        for j in range(len(self.pairs)):
            ok = np.isfinite(self.c[:, j])
            res = f(*[a[ok, j] for a in arrs], **kw)
            res = res if isinstance(res, tuple) else (res,)
            if out is None:
                out = [np.full(self.c.shape, np.nan) for _ in res]
            for o, r in zip(out, res, strict=True):
                o[ok, j] = r
        return out[0] if len(out) == 1 else out


def shift(x, n=1):
    if n == 0:
        return x
    out = np.full(x.shape, np.nan)
    out[n:] = x[:-n]
    return out


def hold(entry, exit_long, exit_short, max_days=0):
    """entry(+1 / −1 / 0)で入り、exit の条件か max_days で出る建玉(列ごとの状態機械)。"""
    n, k = entry.shape
    pos = np.zeros((n, k))
    cur = np.zeros(k)
    age = np.zeros(k)
    for i in range(n):
        e = entry[i]
        out = ((cur > 0) & exit_long[i]) | ((cur < 0) & exit_short[i])
        if max_days:
            out |= (cur != 0) & (age >= max_days)
        cur = np.where(out, 0.0, cur)
        new = (cur == 0) & (e != 0) & ~out
        flip = (cur != 0) & (e != 0) & (e != cur)
        age = np.where(new | flip, 0, age + 1)
        cur = np.where(new | flip, e, cur)
        pos[i] = cur
    return pos


def evaluate(d: Data, pos: np.ndarray) -> dict:
    pos = np.nan_to_num(pos)
    pos[~np.isfinite(d.c)] = 0
    r = np.nan_to_num(d.ret)
    price = np.zeros(pos.shape)
    price[1:] = pos[:-1] * r[1:]
    prev = np.vstack([np.zeros((1, pos.shape[1])), pos[:-1]])
    cost = np.abs(pos - prev) * COST
    carry = np.where(pos > 0, d.diff - RECV, np.where(pos < 0, -d.diff - RECV, 0.0))
    carry = np.where(carry < 0, carry - (PAY - RECV), carry)      # 払う側は取り分が大きい
    swap = np.zeros(pos.shape)
    swap[1:] = (np.abs(pos) * carry)[:-1] * d.gap[1:, None] / 365.25
    net = price - cost
    live = d.t >= START
    out = {}
    for name, m in (("all", live), ("a", live & (d.t < SPLIT)), ("b", d.t >= SPLIT)):
        years = m.sum() / 260
        tot = net[m].sum(axis=1)
        mon = d.month[m]
        ms = np.bincount(mon - mon.min(), weights=tot)[np.unique(mon - mon.min())]
        t = ms.mean() / (ms.std(ddof=1) / np.sqrt(len(ms))) if ms.std() > 0 else 0.0
        pp = net[m].sum(axis=0) / years * 100
        out[name] = {"net": tot.sum() / years * 100, "t": t, "swap": swap[m].sum() / years * 100,
                     "trades": np.abs(pos - prev)[m].sum() / 2 / years, "pairs_pos": int((pp > 0).sum()),
                     "expo": np.abs(pos[m]).mean() * pos.shape[1]}
    return out


def signals(d: Data) -> dict:
    """型の名前 -> 建玉(n x k)。全部、足 t の終値までの情報だけ。"""
    c, h, lo, o = d.c, d.h, d.l, d.o
    S = {}
    a14 = d.col(atr, h, lo, c, n=14)
    vol20 = d.col(rolling_std, d.ret, n=20)
    false = np.zeros(c.shape, dtype=bool)
    sma200 = d.col(sma, c, n=200)
    up200, dn200 = c > sma200, c < sma200

    # 1 順張り: チャネル抜け・移動平均・モメンタム(対照。E046 で落ちている)
    for n_in, n_out in ((10, 5), (20, 10), (55, 20), (100, 50), (200, 100)):
        hi, lw = shift(d.col(highest, h, n=n_in)), shift(d.col(lowest, lo, n=n_in))
        xl, xs = shift(d.col(lowest, lo, n=n_out)), shift(d.col(highest, h, n=n_out))
        S[f"donchian {n_in}/{n_out}"] = hold(np.where(c > hi, 1, np.where(c < lw, -1, 0)), c < xl, c > xs)
    for f, s in ((5, 20), (10, 50), (20, 100), (50, 200)):
        S[f"ema {f}/{s}"] = np.sign(d.col(ema, c, n=f) - d.col(ema, c, n=s))
    for n in (5, 10, 21, 63, 126, 252):
        S[f"mom {n}"] = np.sign(c - shift(c, n))
    m, sg = d.col(macd, c)
    S["macd"] = np.sign(m - sg)
    ax, pdi, mdi = d.col(adx, h, lo, c, n=14)
    S["adx>25 di"] = np.where(ax > 25, np.sign(pdi - mdi), 0)
    S["adx<20 di fade"] = np.where(ax < 20, -np.sign(pdi - mdi), 0)

    # 2 短期の逆張り: 大きく動いた翌日以降に逆へ
    for k in (1, 2, 3, 5):
        z = (c - shift(c, k)) / (c * vol20 * np.sqrt(k))
        for th in (1.0, 1.5, 2.0, 2.5):
            e = np.where(z > th, -1, np.where(z < -th, 1, 0))
            for hd in (1, 3, 5, 10):
                S[f"fade z{k}>{th} hold{hd}"] = hold(e, false, false, hd)
    # RSI(2)・RSI(14)・ストキャスの逆張り(トレンドの絞りあり・なし)
    for n, lo_th, hi_th in ((2, 10, 90), (2, 5, 95), (3, 15, 85), (14, 30, 70)):
        r = d.col(rsi, c, n=n)
        e = np.where(r < lo_th, 1, np.where(r > hi_th, -1, 0))
        S[f"rsi{n} fade exit50"] = hold(e, r > 50, r < 50, 10)
        et = np.where((r < lo_th) & up200, 1, np.where((r > hi_th) & dn200, -1, 0))
        S[f"rsi{n} pullback sma200"] = hold(et, r > 50, r < 50, 10)
        ec = np.where((r < lo_th) & dn200, 1, np.where((r > hi_th) & up200, -1, 0))
        S[f"rsi{n} fade counter200"] = hold(ec, r > 50, r < 50, 10)
    kk, dd = d.col(stochastic, h, lo, c)
    S["stoch fade"] = hold(np.where(kk < 20, 1, np.where(kk > 80, -1, 0)), kk > 50, kk < 50, 10)
    # ボリンジャー: 逆張り(中心で出る)と抜け
    for n, w in ((20, 2.0), (20, 2.5), (50, 2.0), (10, 2.0)):
        mid, sd = d.col(sma, c, n=n), d.col(rolling_std, c, n=n)
        e = np.where(c < mid - w * sd, 1, np.where(c > mid + w * sd, -1, 0))
        S[f"bb{n}x{w} fade"] = hold(e, c > mid, c < mid, 3 * n)
        S[f"bb{n}x{w} break"] = hold(-e, c < mid, c > mid, 3 * n)
        width = 2 * w * sd / mid
        rank = width < d.col(lambda x: lowest(x, 120) * 1.25, width)
        S[f"bb{n}x{w} squeeze break"] = hold(np.where(shift(rank.astype(float)) > 0, -e, 0), c < mid, c > mid, 3 * n)

    # 3 ローソク足・1 日の形
    rng = h - lo
    body = c - o
    pc, ph, pl, po = shift(c), shift(h), shift(lo), shift(o)
    pos_in_range = (c - lo) / np.where(rng > 0, rng, np.nan)
    for hd in (1, 3, 5):
        eng = np.where((body > 0) & (pc < po) & (c > po) & (o < pc), 1,
                       np.where((body < 0) & (pc > po) & (c < po) & (o > pc), -1, 0))
        S[f"engulfing hold{hd}"] = hold(eng, false, false, hd)
        pin = np.where((pos_in_range > 0.7) & (np.minimum(o, c) - lo > 0.6 * rng) & (rng > a14), 1,
                       np.where((pos_in_range < 0.3) & (h - np.maximum(o, c) > 0.6 * rng) & (rng > a14), -1, 0))
        S[f"pin bar hold{hd}"] = hold(pin, false, false, hd)
        S[f"pin bar fade hold{hd}"] = hold(-pin, false, false, hd)
        inside = (h < ph) & (lo > pl)
        ib = np.where(shift(inside.astype(float)) > 0, np.where(c > ph, 1, np.where(c < pl, -1, 0)), 0)
        S[f"inside break hold{hd}"] = hold(ib, false, false, hd)
        outside = (h > ph) & (lo < pl) & (rng > 1.5 * a14)
        S[f"outside day follow hold{hd}"] = hold(np.where(outside, np.sign(body), 0), false, false, hd)
        S[f"outside day fade hold{hd}"] = hold(np.where(outside, -np.sign(body), 0), false, false, hd)
        top = np.where((pos_in_range > 0.9) & (rng > a14), 1, np.where((pos_in_range < 0.1) & (rng > a14), -1, 0))
        S[f"close at extreme follow hold{hd}"] = hold(top, false, false, hd)
        S[f"close at extreme fade hold{hd}"] = hold(-top, false, false, hd)
        nr7 = rng <= d.col(lowest, rng, n=7)
        nb = np.where(shift(nr7.astype(float)) > 0, np.where(c > ph, 1, np.where(c < pl, -1, 0)), 0)
        S[f"nr7 break hold{hd}"] = hold(nb, false, false, hd)
        # 新安値を付けてから前日の終値より上で引ける(キーリバーサル)
        l20, h20 = shift(d.col(lowest, lo, n=20)), shift(d.col(highest, h, n=20))
        key = np.where((lo < l20) & (c > pc), 1, np.where((h > h20) & (c < pc), -1, 0))
        S[f"key reversal hold{hd}"] = hold(key, false, false, hd)
        # 抜けの失敗(タートルスープ): 20 日の安値を割ったのに、その中に戻して引ける
        soup = np.where((lo < l20) & (c > l20), 1, np.where((h > h20) & (c < h20), -1, 0))
        S[f"false break hold{hd}"] = hold(soup, false, false, hd)
    for k in (3, 4, 5):
        upk = np.all(np.stack([shift(d.ret, i) > 0 for i in range(k)]), axis=0)
        dnk = np.all(np.stack([shift(d.ret, i) < 0 for i in range(k)]), axis=0)
        e = np.where(upk, 1, np.where(dnk, -1, 0))
        for hd in (1, 3):
            S[f"{k} in a row follow hold{hd}"] = hold(e, false, false, hd)
            S[f"{k} in a row fade hold{hd}"] = hold(-e, false, false, hd)

    # 4 押し目・戻り: 長いトレンドの向きに、短い逆行で入る
    for trend_n in (50, 100, 200):
        tr = np.sign(c - d.col(sma, c, n=trend_n))
        for pb in (3, 5, 10):
            dip = np.sign(c - shift(c, pb))
            e = np.where((tr > 0) & (dip < 0) & (c <= shift(d.col(lowest, lo, n=pb))), 1,
                         np.where((tr < 0) & (dip > 0) & (c >= shift(d.col(highest, h, n=pb))), -1, 0))
            S[f"pullback sma{trend_n} low{pb}"] = hold(e, c > shift(d.col(highest, h, n=pb)),
                                                        c < shift(d.col(lowest, lo, n=pb)), 20)
    # 52 週の高値に近い(高値圏の順張り)・遠い
    h252, l252 = d.col(highest, h, n=252), d.col(lowest, lo, n=252)
    loc = (c - l252) / (h252 - l252)
    S["52w loc >0.9 long / <0.1 short"] = np.where(loc > 0.9, 1, np.where(loc < 0.1, -1, 0))
    S["52w loc fade"] = np.where(loc > 0.95, -1, np.where(loc < 0.05, 1, 0))

    # 5 ボラの局面と向き
    vr = vol20 / d.col(rolling_std, d.ret, n=250)
    m63 = np.sign(c - shift(c, 63))
    S["mom63 when vol low"] = np.where(vr < 0.8, m63, 0)
    S["mom63 when vol high"] = np.where(vr > 1.2, m63, 0)
    S["mom63 scaled 1/vol"] = m63 * np.clip(0.006 / vol20, 0, 3)

    # 6 暦: 曜日ごとの買い(対照)。売りは符号を返したもの
    for w, nm in enumerate(["mon", "tue", "wed", "thu", "fri"]):
        S[f"long on {nm} close"] = np.where((d.dow == w)[:, None], 1.0, 0.0) * np.ones(c.shape)

    # 7 通貨の強弱(8 通貨)。直近 n 日の強さの順位で、強い通貨を買う(順張り)か売る(逆張り)
    a = np.zeros((len(d.pairs), len(CURRENCIES)))
    for j, p in enumerate(d.pairs):
        a[j, CURRENCIES.index(p[:3])] += 1
        a[j, CURRENCIES.index(p[3:])] -= 1
    pinv = np.linalg.pinv(np.vstack([a, np.ones(len(CURRENCIES))]))[:, :len(d.pairs)]
    full = np.isfinite(d.ret).all(axis=1)
    strength = np.where(full[:, None], np.nan_to_num(d.ret) @ pinv.T, 0.0)
    cum = np.cumsum(strength, axis=0)
    for n in (1, 3, 5, 10, 21, 63, 126, 252):
        s_n = cum - shift(cum, n)
        pair_score = np.column_stack([s_n[:, CURRENCIES.index(p[:3])] - s_n[:, CURRENCIES.index(p[3:])]
                                      for p in d.pairs])
        rk = np.argsort(np.argsort(pair_score, axis=1, kind="stable"), axis=1, kind="stable")
        xs = np.where(rk >= 7, 1, np.where(rk <= 2, -1, 0)).astype(float)
        xs[~np.isfinite(pair_score).all(axis=1)] = 0
        S[f"strength {n}d follow (top3-bottom3)"] = xs
        S[f"strength {n}d fade (top3-bottom3)"] = -xs
    S["_strength"] = cum
    return S


def crosses(d: Data) -> dict:
    """合成クロス(8 通貨の 28 組)の平均回帰。建玉は通貨の強さの差の向き(2 本の脚に分けた概算の費用 x 2)。"""
    S = signals_strength(d)
    out = {}
    for n, th in ((60, 2.0), (120, 2.0), (250, 2.0), (120, 1.5), (250, 1.5), (20, 2.0)):
        for i, j in combinations(range(len(CURRENCIES)), 2):
            x = S[:, i] - S[:, j]
            mid, sd = sma(x, n), rolling_std(x, n)
            z = (x - mid) / np.where(sd > 0, sd, np.nan)
            e = np.where(z > th, -1, np.where(z < -th, 1, 0))[:, None]
            pos = hold(e, (z > 0)[:, None], (z < 0)[:, None], 2 * n)[:, 0]
            out[(n, th, CURRENCIES[i] + CURRENCIES[j])] = (pos, x)
    return out


def signals_strength(d: Data) -> np.ndarray:
    a = np.zeros((len(d.pairs), len(CURRENCIES)))
    for j, p in enumerate(d.pairs):
        a[j, CURRENCIES.index(p[:3])] += 1
        a[j, CURRENCIES.index(p[3:])] -= 1
    pinv = np.linalg.pinv(np.vstack([a, np.ones(len(CURRENCIES))]))[:, :len(d.pairs)]
    full = np.isfinite(d.ret).all(axis=1)
    return np.cumsum(np.where(full[:, None], np.nan_to_num(d.ret) @ pinv.T, 0.0), axis=0)


def eval_cross(d: Data, pos: np.ndarray, x: np.ndarray) -> dict:
    """クロスの建玉 1 本の成績(値動き − 費用、建玉比の年率 %)。費用は脚 2 本ぶん。"""
    price = np.r_[0.0, pos[:-1] * np.diff(x)]
    cost = np.abs(np.diff(np.r_[0.0, pos])) * COST * 2
    net = price - cost
    out = {}
    for name, m in (("all", d.t >= START), ("a", (d.t >= START) & (d.t < SPLIT)), ("b", d.t >= SPLIT)):
        out[name] = (net[m].sum() / (m.sum() / 260) * 100, net[m], d.month[m])
    return out


def month_t(v, mon) -> float:
    ms = np.bincount(mon - mon.min(), weights=v)[np.unique(mon - mon.min())]
    return float(ms.mean() / (ms.std(ddof=1) / np.sqrt(len(ms)))) if ms.std() > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0)
    args = ap.parse_args()
    d = Data()
    S = signals(d)
    S.pop("_strength")
    rows = []
    for name, pos in S.items():
        for label, p in ((name, pos), ):
            r = evaluate(d, p)
            rows.append((label, r))
    rows.sort(key=lambda x: -min(x[1]["a"]["t"], x[1]["b"]["t"]))
    if args.top:
        rows = rows[:args.top]
    print(f"型の数 {len(S)}。値動き − 費用(10 ペアの合計、建玉比の年率 %)。発見 = 1996〜2013、確認 = 2014〜2026")
    print("| 型 | 全期間 | t | 発見 | t | 確認 | t | スワップ(全) | 取引 / 年 | プラスのペア | 平均の建玉数 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in rows:
        print(f"| {name} | {r['all']['net']:+.2f} | {r['all']['t']:+.2f} | {r['a']['net']:+.2f} | {r['a']['t']:+.2f} "
              f"| {r['b']['net']:+.2f} | {r['b']['t']:+.2f} | {r['all']['swap']:+.2f} | {r['all']['trades']:.0f} "
              f"| {r['all']['pairs_pos']}/10 | {r['all']['expo']:.1f} |")

    print("\n合成クロスの平均回帰(28 組の合計。z が ±th を越えたら逆へ、0 に戻るか 2n 日で出る)")
    print("| n / th | 全期間 | t | 発見 | t | 確認 | t | プラスの組 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    cr = crosses(d)
    keys = sorted({(n, th) for n, th, _ in cr})
    per_cross = {}
    for n, th in keys:
        agg = {"all": None, "a": None, "b": None}
        npos = 0
        for (n2, th2, nm), (pos, x) in cr.items():
            if (n2, th2) != (n, th):
                continue
            e = eval_cross(d, pos, x)
            npos += e["all"][0] > 0
            per_cross[(n, th, nm)] = e
            for k in agg:
                agg[k] = e[k][1].copy() if agg[k] is None else agg[k] + e[k][1]
        mons = {k: per_cross[(n, th, "USDEUR")][k][2] for k in agg}
        cells = " | ".join(f"{agg[k].sum() / (len(agg[k]) / 260) * 100:+.2f} | {month_t(agg[k], mons[k]):+.2f}"
                           for k in ("all", "a", "b"))
        print(f"| {n} / {th} | {cells} | {npos}/28 |")
    print("\nクロスごと(n 120 / th 2.0。年率 %、発見 / 確認)")
    for (n, th, nm), e in sorted(per_cross.items(), key=lambda kv: -kv[1]["all"][0]):
        if (n, th) == (120, 2.0):
            print(f"  {nm}: 全 {e['all'][0]:+.2f} (t {month_t(e['all'][1], e['all'][2]):+.2f}) "
                  f"発見 {e['a'][0]:+.2f} 確認 {e['b'][0]:+.2f}")


if __name__ == "__main__":
    main()
