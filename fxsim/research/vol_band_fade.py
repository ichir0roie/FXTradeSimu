"""E060: 直近の振れの q 点に、上下両方の指値を置く(1 時間足〜週足)。

    python -m fxsim.research.vol_band_fade                    # 第 1 段: 主要 10 ペア、全部の格子
    python -m fxsim.research.vol_band_fade --pairs crosses    # 初見のクロス 18 本
    python -m fxsim.research.vol_band_fade --placebo 5        # 1 時間足の形を並べ替えた相場(seed 1〜5)
    python -m fxsim.research.vol_band_fade --fill-margin 1    # 指値は 1 pip 余分に抜けたときだけ約定
    python -m fxsim.research.vol_band_fade --holdout3         # 第 5 段: 第 3 段の日足・週足の逆張りを初見の 1994〜2010 年と偽薬で
    python -m fxsim.research.vol_band_fade --stage3           # 第 3 段: 平均線・ボリンジャー・ケルトナー・ドンチアンなどを基準にした帯
    python -m fxsim.research.vol_band_fade --holdout          # 第 4 段: 第 2 段の週足・逆張りを初見の 1994〜2010 年(日足の経路)で
    python -m fxsim.research.vol_band_fade --stage2           # 第 2 段: 日足・週足、28 ペア、振り返り x 持つ長さを広げ、
                                                              #   発見(〜2018-10)で選んで確認(2018-11〜)で確かめる

判断の足(1 時間足・4 時間足・日足・週足)は相場の DB の 1 時間足をまとめて作る。足 i の終値 P から上下に w を取り、
次の足の間だけ有効な売りの指値 P + w と買いの指値 P − w を置く(fade)。鏡(break)は同じ値に逆指値を置く順張り。
w は直近 lb 本の「高値 − 始値」と「始値 − 安値」を合わせた分布の q 点。建玉がある間は新しい注文を出さない。
出口(H 本 = 判断の足の本数)
    time     H 本後の足の始まりで成行
    tp       fade は P に戻ったら利確(指値)。break は P に戻ったら損切り(逆指値)。届かなければ H 本後に成行
    tp_stop  tp に、fade は建値から 2w 逆の逆指値、break は建値から 2w 先の利確を足す
    片側の建玉が出る前にもう片側が刺さったら、両方で 1 取引(売りと買いの値の差で確定)
約定は E050(`exit_grid.py`)と同じく不利な側に倒す: 買いは ask(bid + その足の記録のスプレッド)、売りは bid、
ロールオーバー(20〜22 時 UTC)の足では指値・逆指値を約定させず、時間切れの成行はその後の最初の足に送る。
逆指値は届いた値か始値の悪い方 + 0.3 pip、成行は 0.1 pip。同じ足で損切りと利確に届いたら損切りが先、入った足では利確を見ない。
--gross の版はスプレッドと滑りを 0 にする(素の損益)。
金利差の受け払い(キャリー)は別の列に出す: FRED の短期金利の月次(入った日の前の月の値)の差 x 持った日数。
業者の上乗せは入れていない。
単位は 1 取引の bp(P 比)と、帯の幅 w を 1 とした値。t は全ペアの損益を月ごとに足してから数える。
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..storage import prices
from ..strategies.indicators import atr, ema, highest, lowest, rolling_std, sma
from .intraday_edge import BAD_HOURS, SOURCE

PAIRS = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD", "EURJPY", "GBPJPY", "AUDJPY"]
CROSSES = ["EURGBP", "EURAUD", "EURNZD", "EURCAD", "EURCHF", "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF",
           "AUDNZD", "AUDCAD", "AUDCHF", "NZDCAD", "NZDCHF", "NZDJPY", "CADCHF", "CADJPY", "CHFJPY"]
TFS = {"H1": [0], "H4": [0, 1, 2, 3], "D1": [0], "W1": [0]}      # 足と、区切りのずらし(時間)
QS = [0.75, 0.9, 0.95]
EXITS = ["time", "tp", "tp_stop"]
HOLDS = [1, 3]
KINDS = ["fade", "break"]
LOOKBACK = 100
GRID = [(k, q, e, h, LOOKBACK) for k, q, e, h in product(KINDS, QS, EXITS, HOLDS)]
MAIN = ("fade", 0.9, "tp_stop", 3, LOOKBACK)
STAGE2_TFS = {"D1": [0], "W1": [0]}
STAGE2_GRID = [(k, q, e, h, lb) for k, q, e, h, lb in product(KINDS, QS, EXITS, [1, 2, 3, 6], [26, 52, 100])
               if k == "fade" or e == "time"]          # 鏡は時間で出る形だけ(P の損切りは同じ足の順番で不利に倒れすぎる)
STAGE3_TFS = {"H1": [0], "H4": [0, 2], "D1": [0], "W1": [0]}
BANDS = [("maq", 20, 0.9, 100), ("maq", 50, 0.9, 100), ("maq", 20, 0.75, 100), ("maq", 20, 0.95, 100),
         ("bb", 20, 2.0), ("bb", 20, 2.5), ("bb", 20, 3.0), ("bb", 50, 2.0),
         ("kc", 20, 1.5), ("kc", 20, 2.0), ("kc", 20, 3.0),
         ("dev", 20, 0.9, 100), ("dev", 20, 0.95, 100), ("dev", 50, 0.9, 100),
         ("dc", 20), ("dc", 55)]
BAND_NAMES = {"maq": "平均線 + 伸びの q 点", "bb": "ボリンジャー", "kc": "ケルトナー", "dev": "平均からの離れの q 点", "dc": "ドンチアン"}
STAGE3_GRID = ([("fade", b, e, h, m) for b, m, e, h in product(BANDS, ["limit", "market"], EXITS, [1, 3, 10])]
               + [("break", b, "time", h, m) for b, m, h in product(BANDS, ["limit", "market"], [1, 3, 10])])
SLIP_MKT, SLIP_STOP = 0.1, 0.3
SPLIT = int(np.datetime64("2018-11-01").astype("datetime64[s]").astype(np.int64))
FROM = int(np.datetime64("2010-10-01").astype("datetime64[s]").astype(np.int64))
MONTH = 86400 * 30.4375
RATES = "data/real/short_rates_monthly_fred.csv"


def load(pair: str, seed: int = 0, src: tuple | None = None) -> dict:
    """src = (足の分数, 始まり, 終わり)。既定は 1 時間足の 2010-10〜。"""
    tf_min, start, end = src or (60, FROM, None)
    con = prices.open_market_db()
    h = prices.read_rates(con, SOURCE, pair, tf_min, start=start, end=end)
    con.close()
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    t, o, hi, lo, c = h["time"], h["open"], h["high"], h["low"], h["close"]
    sp = h["spread"] * pip / 10
    if seed:      # 偽薬: 1 時間足の形(始値からの高・安・終値)とスプレッドを並べ替える。時刻はそのまま
        rng = np.random.default_rng(seed * 1000 + (PAIRS + CROSSES).index(pair))
        perm = rng.permutation(len(t))
        up, dn, cl, sp = np.log(hi / o)[perm], np.log(lo / o)[perm], np.log(c / o)[perm], sp[perm]
        o = o[0] * np.exp(np.r_[0.0, np.cumsum(cl)[:-1]])
        hi, lo, c = o * np.exp(up), o * np.exp(dn), o * np.exp(cl)
    bad = np.isin((t // 3600) % 24, BAD_HOURS) if tf_min < 1440 else np.zeros(len(t), dtype=bool)
    return {"t": t, "o": o, "h": hi, "l": lo, "c": c, "sp": sp, "pip": pip, "bad": bad}


def rate_diff(pair: str) -> tuple[np.ndarray, np.ndarray]:
    """(有効になる時刻、基軸 − 決済の金利 % )。月 M の平均は翌月 1 日から使う。"""
    with open(RATES) as f:
        rows = list(csv.DictReader(f))
    b, q = pair[:3].lower(), pair[3:].lower()
    eff = np.array([np.datetime64(r["date"], "s").astype(np.int64) + 31 * 86400 for r in rows])
    diff = np.array([float(r[b]) - float(r[q]) for r in rows])
    return eff, diff


def tf_keys(t: np.ndarray, tf: str, off: int) -> np.ndarray:
    if tf == "H1":
        return t // 3600
    if tf == "H4":
        return (t - off * 3600) // 14400
    if tf == "D1":
        return (t + 7200) // 86400                           # 22:00 UTC 区切り
    if tf == "D1d":                                          # 日足そのもの(日足の時刻は 21:00 / 22:00 UTC)
        return (t + 3 * 3600) // 86400
    if tf == "W1d":                                          # 日足から作る週足(日足の時刻は 21:00 / 22:00 UTC)
        return (t + 3 * 3600 - 4 * 86400) // (7 * 86400)
    return (t + 7200 - 4 * 86400) // (7 * 86400)             # 日曜 22:00 UTC 区切り


def parse(cfg: tuple) -> tuple[tuple, str]:
    """格子の 1 行から(帯の作り方、入り方)。第 1・2 段の行は (kind, q, exit, hold, lb)。"""
    if isinstance(cfg[1], tuple):
        return cfg[1], cfg[4]
    return ("q", cfg[1], cfg[4]), "limit"


def make_band(spec: tuple, O, H, L, C, cache: dict) -> tuple[np.ndarray, ...] | None:
    """判断の足 i の終値までで決まる(真ん中、上、下、半分の幅)。足りなければ None。
    q    (q, lb)      終値 ± 直近 lb 本の始値からの伸びの q 点
    maq  (n, q, lb)   SMA(n) ± 同じ伸びの q 点
    bb   (n, k)       SMA(n) ± k x 標準偏差(n)(ボリンジャー)
    kc   (n, k)       EMA(n) ± k x ATR(n)(ケルトナー)
    dev  (n, q, lb)   SMA(n) ± 直近 lb 本の |終値 − SMA(n)| の q 点
    dc   (n,)         一つ前までの n 本の高値・安値(ドンチアン)、真ん中はその間
    """
    if spec in cache:
        return cache[spec]
    kind, nT = spec[0], len(C)

    def exq(q: float, lb: int):
        key = ("exq", lb)
        if key not in cache:
            if nT < lb + 5:
                cache[key] = None
            else:
                ex = np.concatenate([sliding_window_view(H - O, lb), sliding_window_view(O - L, lb)], axis=1)
                W = np.full((len(QS), nT), np.nan)
                W[:, lb - 1:] = np.quantile(ex, QS, axis=1)
                cache[key] = W
        W = cache[key]
        return None if W is None else W[QS.index(q)]

    out = None
    if kind == "q":
        w = exq(spec[1], spec[2])
        out = None if w is None else (C, C + w, C - w, w)
    elif kind == "maq":
        w = exq(spec[2], spec[3])
        m = sma(C, spec[1])
        out = None if w is None else (m, m + w, m - w, w)
    elif kind == "bb":
        m, sd = sma(C, spec[1]), rolling_std(C, spec[1])
        out = (m, m + spec[2] * sd, m - spec[2] * sd, spec[2] * sd)
    elif kind == "kc":
        m, a = ema(C, spec[1]), atr(H, L, C, spec[1])
        out = (m, m + spec[2] * a, m - spec[2] * a, spec[2] * a)
    elif kind == "dev":
        m = sma(C, spec[1])
        w = pd.Series(np.abs(C - m)).rolling(spec[3]).quantile(spec[2]).to_numpy(copy=True)
        out = (m, m + w, m - w, w)
    elif kind == "dc":
        up, dn = np.r_[np.nan, highest(H, spec[1])[:-1]], np.r_[np.nan, lowest(L, spec[1])[:-1]]
        out = ((up + dn) / 2, up, dn, (up - dn) / 2)
    if out is not None and nT < 30:
        out = None
    cache[spec] = out
    return out


def simulate(pair: str, tf: str, off: int, gross: bool, seed: int, fill_margin: float, grid: list | None = None,
             src: tuple | None = None) -> dict:
    grid = grid or GRID
    x = load(pair, seed, src)
    t, ho, hh, hl, hc, bad, pip = x["t"], x["o"], x["h"], x["l"], x["c"], x["bad"], x["pip"]
    n = len(t)
    sp = np.zeros(n) if gross else x["sp"]
    s_mkt, s_stop = (0.0, 0.0) if gross else (SLIP_MKT * pip, SLIP_STOP * pip)
    fm = fill_margin * pip
    r_eff, r_diff = rate_diff(pair)
    keys = tf_keys(t, tf, off)
    S = np.flatnonzero(np.r_[True, np.diff(keys) != 0])
    E = np.r_[S[1:], n]
    tf_of = np.cumsum(np.r_[0, np.diff(keys) != 0])
    O, H, L, C = ho[S], np.maximum.reduceat(hh, S), np.minimum.reduceat(hl, S), hc[E - 1]
    SPX = np.maximum.reduceat(sp, S)
    nT = len(S)
    years = (t[-1] - t[0]) / (365.25 * 86400)
    cache: dict = {}
    good = np.flatnonzero(~bad)
    nb_next = np.r_[good, n][np.searchsorted(good, np.arange(n + 1))]     # j 以降で最初の約定できる足
    BIG = n + 10

    def first(mask: np.ndarray, a: int) -> int:
        return a + int(np.argmax(mask)) if mask.any() else BIG

    def carry(side: int, j0: int, j1: int) -> float:
        k = int(np.searchsorted(r_eff, t[j0], side="right")) - 1
        return side * r_diff[max(k, 0)] / 100 * (t[min(j1, n - 1)] - t[j0]) / (365.25 * 86400) * 1e4

    def leg_exit(s1: int, e1: float, j1: int, j_end: int, P: float, w: float, kind: str, exit_: str,
                 take_in_first: bool = False) -> tuple[float, int, int, float]:
        """j1 で入った片側の建玉の出口。(損益、出た足、理由、キャリー)。理由 0 時間 / 1 損切り / 2 利確。"""
        take = stop = None
        if exit_ != "time":
            if kind == "fade":
                take = P
                stop = e1 - s1 * 2 * w if exit_ == "tp_stop" else None
            else:
                stop = P
                take = e1 + s1 * 2 * w if exit_ == "tp_stop" else None
        lg = slice(j1, j_end)
        m = j_end - j1
        js = jt = m
        if stop is not None:
            hit = (hl[lg] <= stop) if s1 > 0 else (hh[lg] + sp[lg] >= stop)
            if hit.any():
                js = int(np.argmax(hit))
        if take is not None:
            hit = ((hh[lg] >= take) if s1 > 0 else (hl[lg] + sp[lg] <= take)) & ~bad[lg]
            if not take_in_first:
                hit[0] = False                             # 指値で入った足では利確を見ない
            if hit.any():
                jt = int(np.argmax(hit))
        if js <= jt and js < m:
            jx = j1 + js
            px = min(stop, ho[jx]) - s_stop if s1 > 0 else max(stop, ho[jx] + sp[jx]) + s_stop
            why = 1
        elif jt < m:
            jx = j1 + jt
            px = max(take, ho[jx]) if s1 > 0 else min(take, ho[jx] + sp[jx])
            why = 2
        else:
            jx = j_end
            px = ho[jx] - s_mkt if s1 > 0 else ho[jx] + sp[jx] + s_mkt
            why = 0
        return s1 * (px - e1), jx, why, carry(s1, j1, jx)

    out = {}
    for cfg in grid:
        kind, _, exit_, hold, _ = cfg
        spec, entry = parse(cfg)
        band = make_band(spec, O, H, L, C, cache)
        if band is None:
            out[cfg] = np.zeros((0, 5))
            continue
        CEN, UP, DN, WH = band
        tr = []
        i = spec[2] - 1 if spec[0] == "q" else 0
        while i < nT - 1 - hold:
            P, up, dn, w = CEN[i], UP[i], DN[i], WH[i]
            if not (w > 0):
                i += 1
                continue
            if entry == "market":                          # 終値が帯の外なら、次の足の最初の約定できる足で成行
                side0 = 0
                if C[i] > up:
                    side0 = -1 if kind == "fade" else 1
                elif C[i] < dn:
                    side0 = 1 if kind == "fade" else -1
                if side0 == 0:
                    i += 1
                    continue
                j1 = int(nb_next[S[i + 1]])
                j_end = int(nb_next[S[i + 1 + hold]])
                if j_end >= n:
                    break
                if j1 >= j_end:
                    i += 1
                    continue
                s1 = side0
                e1 = ho[j1] + sp[j1] + s_mkt if s1 > 0 else ho[j1] - s_mkt
                pnl, jx, why, cy = leg_exit(s1, e1, j1, j_end, P, w, kind, exit_, take_in_first=True)
                tr.append((pnl / C[i] * 1e4, t[min(jx, n - 1)], pnl / w, why, cy))
                i = max(int(tf_of[min(jx, n - 1)]), i + 1)
                continue
            if up <= C[i]:                                 # 帯の外にいる側には指値を置かない
                up = np.inf
            if dn >= C[i]:
                dn = -np.inf
            if kind == "fade":
                if H[i + 1] < up + fm and L[i + 1] > dn - fm:
                    i += 1
                    continue
            elif H[i + 1] + SPX[i + 1] < up and L[i + 1] > dn:
                i += 1
                continue
            a, b = int(S[i + 1]), int(E[i + 1])
            seg = slice(a, b)
            if kind == "fade":     # 上で売り(−1)、下で買い(+1)
                ju = first((hh[seg] >= up + fm) & ~bad[seg], a)
                jd = first((hl[seg] + sp[seg] <= dn - fm) & ~bad[seg], a)
            else:                  # 上で買い(+1)、下で売り(−1)
                ju = first((hh[seg] + sp[seg] >= up) & ~bad[seg], a)
                jd = first((hl[seg] <= dn) & ~bad[seg], a)
            if ju == BIG and jd == BIG:
                i += 1
                continue
            j_end = int(nb_next[S[i + 1 + hold]])
            if j_end >= n:
                break

            def fill(is_up: bool, j: int, kind=kind, up=up, dn=dn) -> tuple[int, float]:
                if kind == "fade":
                    return (-1, max(up, ho[j])) if is_up else (1, min(dn, ho[j] + sp[j]))
                return (1, max(up, ho[j] + sp[j]) + s_stop) if is_up else (-1, min(dn, ho[j]) - s_stop)

            if ju == jd:                                   # 同じ足で両側が刺さった
                (s1, e1), (s2, e2) = fill(True, ju), fill(False, jd)
                pnl, jx, why, cy = -(s1 * e1 + s2 * e2), ju, 3, 0.0
            else:
                is_up = ju < jd
                j1, j2 = (ju, jd) if is_up else (jd, ju)
                s1, e1 = fill(is_up, j1)
                pnl1, jx, why, cy1 = leg_exit(s1, e1, j1, j_end, P, w, kind, exit_)
                if j2 < jx:                                # 出る前にもう片側も刺さった: 両方で確定
                    s2, e2 = fill(not is_up, j2)
                    pnl, why, cy = -(s1 * e1 + s2 * e2), 3, carry(s1, j1, j2)
                    jx = j2
                else:
                    pnl, cy = pnl1, cy1
            tr.append((pnl / C[i] * 1e4, t[min(jx, n - 1)], pnl / w, why, cy))
            i = max(int(tf_of[min(jx, n - 1)]), i + 1)
        out[cfg] = np.array(tr).reshape(-1, 5)
    return {"res": out, "years": years}


def month_t(v: np.ndarray, tt: np.ndarray) -> float:
    if len(v) < 10:
        return 0.0
    mon = (tt // MONTH).astype(int)
    s = np.bincount(mon - mon.min(), weights=v)[np.unique(mon - mon.min())]
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 2 and s.std() > 0 else 0.0


def summarize(parts: list[np.ndarray], years: float, n_off: int) -> dict:
    tr = np.vstack(parts) if parts else np.zeros((0, 5))
    v, tt, wu, why, cy = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3], tr[:, 4]
    vc = v + cy
    a, b = tt < SPLIT, tt >= SPLIT
    pos = sum(p[:, 0].sum() > 0 for p in parts if len(p))
    has = len(v) > 0
    return {"n": len(v), "per_year": len(v) / years / n_off if years else 0, "win": (v > 0).mean() if has else 0,
            "avg": v.mean() if has else 0, "w": wu.mean() if has else 0,
            "t": month_t(v, tt), "tw": month_t(wu, tt),
            "a": v[a].mean() if a.any() else 0, "ta": month_t(v[a], tt[a]),
            "b": v[b].mean() if b.any() else 0, "tb": month_t(v[b], tt[b]),
            "carry": cy.mean() if has else 0, "avg_c": vc.mean() if has else 0, "t_c": month_t(vc, tt),
            "a_c": vc[a].mean() if a.any() else 0, "ta_c": month_t(vc[a], tt[a]),
            "b_c": vc[b].mean() if b.any() else 0, "tb_c": month_t(vc[b], tt[b]),
            "lock": (why == 3).mean() if has else 0, "tp": (why == 2).mean() if has else 0,
            "sl": (why == 1).mean() if has else 0, "pos": pos, "npairs": len(parts)}


def run(pairs: list[str], gross_list: list[bool], seeds: list[int], fill_margin: float, workers: int,
        tfs: dict | None = None, grid: list | None = None, src: tuple | None = None) -> dict:
    tfs, grid = tfs or TFS, grid or GRID
    jobs = [(p, tf, off, g, s, fill_margin, grid, src) for s in seeds for g in gross_list for tf, offs in tfs.items()
            for off in offs for p in pairs]
    with ProcessPoolExecutor(workers) as ex:
        outs = list(ex.map(simulate, *zip(*jobs, strict=True)))
    stats = {}
    for s in seeds:
        for g in gross_list:
            for tf, offs in tfs.items():
                sel = [(j, o) for j, o in zip(jobs, outs, strict=True) if j[4] == s and j[3] == g and j[1] == tf]
                years = sum(o["years"] for _, o in sel) / len(offs)
                for cfg in grid:
                    per_pair = {}
                    for j, o in sel:                      # ペアごとに(区切りのずらしは合わせて)まとめる
                        per_pair.setdefault(j[0], []).append(o["res"][cfg])
                    parts = [np.vstack(v) for v in per_pair.values()]
                    stats[(s, g, tf, cfg)] = summarize(parts, years, len(offs))
    return stats


def label(cfg) -> str:
    kind, q, exit_, hold, lb = cfg
    return f"{'指値(逆張り)' if kind == 'fade' else '逆指値(順張り)'} | {q} | {exit_} | {hold}" + ("" if lb == LOOKBACK else f" | {lb}")


def report(stats: dict, seeds: list[int], title: str) -> None:
    s0 = seeds[0]
    print(f"\n## {title}: 主の形(指値・q 0.9・tp_stop・H 3)\n")
    print("| 足 | 取引 / 年(1 ペア) | 勝率 | 両側で確定 | 利確 | 損切り | 費用込み bp | 素の bp | 費用込み w | t(月) | t(月、w) "
          "| 発見 bp / t | 確認 bp / t | プラスのペア |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tf in TFS:
        s, g = stats.get((s0, False, tf, MAIN)), stats.get((s0, True, tf, MAIN))
        if s is None:
            s = g
        print(f"| {tf} | {s['per_year']:.0f} | {s['win']:.0%} | {s['lock']:.0%} | {s['tp']:.0%} | {s['sl']:.0%} "
              f"| {s['avg']:+.2f} | {g['avg'] if g else float('nan'):+.2f} | {s['w']:+.3f} | {s['t']:+.2f} | {s['tw']:+.2f} "
              f"| {s['a']:+.2f} / {s['ta']:+.2f} | {s['b']:+.2f} / {s['tb']:+.2f} | {s['pos']}/{s['npairs']} |")
    for g in sorted({k[1] for k in stats}):
        print(f"\n### 全部の格子({'素の損益' if g else '費用込み'}、seed {s0})\n")
        print("| 足 | 注文 | q | 出口 | H | 取引 / 年 | 勝率 | 両側で確定 | bp | w | t(月) | t(月、w) | 発見 bp / t | 確認 bp / t | プラスのペア |")
        print("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for tf in TFS:
            for cfg in GRID:
                s = stats[(s0, g, tf, cfg)]
                print(f"| {tf} | {label(cfg)} | {s['per_year']:.0f} | {s['win']:.0%} | {s['lock']:.0%} | {s['avg']:+.2f} "
                      f"| {s['w']:+.3f} | {s['t']:+.2f} | {s['tw']:+.2f} | {s['a']:+.2f} / {s['ta']:+.2f} "
                      f"| {s['b']:+.2f} / {s['tb']:+.2f} | {s['pos']}/{s['npairs']} |")


def report_placebo(stats: dict, seeds: list[int]) -> None:
    print(f"\n## 偽薬(1 時間足の形を並べ替えた相場、seed {seeds[0]}〜{seeds[-1]}、素の損益)\n")
    print("| 足 | 注文 | q | 出口 | H | bp(seed ごと) | 平均 | w の平均 | t(月)の範囲 |\n|---|---|---:|---|---:|---|---:|---:|---|")
    for tf in TFS:
        for cfg in GRID:
            ss = [stats[(s, True, tf, cfg)] for s in seeds]
            each = " ".join(f"{x['avg']:+.2f}" for x in ss)
            print(f"| {tf} | {label(cfg)} | {each} | {np.mean([x['avg'] for x in ss]):+.2f} "
                  f"| {np.mean([x['w'] for x in ss]):+.3f} | {min(x['t'] for x in ss):+.2f}〜{max(x['t'] for x in ss):+.2f} |")


def report_stage2(stats: dict, top: int) -> None:
    """発見の期間(〜2018-10)の t(費用込み + キャリー)で選び、確認の期間で確かめる。"""
    print("\n## 第 2 段: 日足・週足、主要 10 ペア + クロス 18 本。費用込み + キャリー(業者の上乗せ無し)\n")
    for tf in STAGE2_TFS:
        for kind in KINDS:
            cfgs = [c for c in STAGE2_GRID if c[0] == kind]
            ss = {c: stats[(0, False, tf, c)] for c in cfgs}
            conf_pos = sum(s["b_c"] > 0 for s in ss.values())
            print(f"\n### {tf} {'指値(逆張り)' if kind == 'fade' else '逆指値(順張り)'}: 格子 {len(cfgs)} 通り。"
                  f"格子全体の平均 費用込み {np.mean([s['avg'] for s in ss.values()]):+.2f} bp、キャリー {np.mean([s['carry'] for s in ss.values()]):+.2f} bp。"
                  f"確認でプラス {conf_pos}/{len(cfgs)}、確認で t ≥ 2 は {sum(s['tb_c'] >= 2 for s in ss.values())}\n")
            print("| q | 出口 | H | 振り返り | 取引 / 年(1 ペア) | 勝率 | 費用込み bp | キャリー bp | 合計 bp | t(月) | 発見 bp / t | 確認 bp / t | プラスのペア |")
            print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for c, s in sorted(ss.items(), key=lambda kv: -kv[1]["ta_c"])[:top]:
                print(f"| {c[1]} | {c[2]} | {c[3]} | {c[4]} | {s['per_year']:.1f} | {s['win']:.0%} | {s['avg']:+.2f} | {s['carry']:+.2f} "
                      f"| {s['avg_c']:+.2f} | {s['t_c']:+.2f} | {s['a_c']:+.2f} / {s['ta_c']:+.2f} | {s['b_c']:+.2f} / {s['tb_c']:+.2f} "
                      f"| {s['pos']}/{s['npairs']} |")


def report_stage3(stats: dict, top: int) -> None:
    """平均線などを基準にした帯。発見(〜2018-10)の t(費用込み + キャリー)で選び、確認で確かめる。"""
    print("\n## 第 3 段: 帯の基準(平均線・ボリンジャー・ケルトナー・平均からの離れ・ドンチアン)、28 ペア、費用込み + キャリー\n")
    for tf in STAGE3_TFS:
        print(f"\n### {tf}\n")
        print("| 注文 | 入り | 格子 | 平均 bp(費用込み + キャリー) | 確認でプラス | 確認で t ≥ 2 | 発見の t の最大 |\n|---|---|---:|---:|---:|---:|---:|")
        for kind in KINDS:
            for entry in ("limit", "market"):
                ss = [stats[(0, False, tf, c)] for c in STAGE3_GRID if c[0] == kind and c[4] == entry]
                print(f"| {'指値(逆張り)' if kind == 'fade' else '逆指値(順張り)'} | {'指値' if entry == 'limit' else '帯の外で成行'} | {len(ss)} "
                      f"| {np.mean([s['avg_c'] for s in ss]):+.2f} | {sum(s['b_c'] > 0 for s in ss)} | {sum(s['tb_c'] >= 2 for s in ss)} "
                      f"| {max(s['ta_c'] for s in ss):+.2f} |")
        print("\n帯の種類ごと(逆張り、入り x 出口 x 持つ長さの平均。bp は費用込み + キャリー)\n")
        print("| 帯 | 入り | 平均 bp | 発見 bp | 確認 bp | 確認でプラス | 取引 / 年(1 ペア) |\n|---|---|---:|---:|---:|---:|---:|")
        for b in BANDS:
            for entry in ("limit", "market"):
                ss = [stats[(0, False, tf, c)] for c in STAGE3_GRID if c[0] == "fade" and c[1] == b and c[4] == entry]
                print(f"| {BAND_NAMES[b[0]]} {b[1:]} | {'指値' if entry == 'limit' else '成行'} | {np.mean([s['avg_c'] for s in ss]):+.2f} "
                      f"| {np.mean([s['a_c'] for s in ss]):+.2f} | {np.mean([s['b_c'] for s in ss]):+.2f} | {sum(s['b_c'] > 0 for s in ss)}/{len(ss)} "
                      f"| {np.mean([s['per_year'] for s in ss]):.0f} |")
        print(f"\n発見の t の上位 {top}(確認は選ぶのに使っていない)\n")
        print("| 注文 | 帯 | 入り | 出口 | H | 取引 / 年 | 勝率 | 費用込み bp | キャリー | t(月) | 発見 bp / t | 確認 bp / t | プラスのペア |")
        print("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        ranked = sorted(((c, stats[(0, False, tf, c)]) for c in STAGE3_GRID), key=lambda kv: -kv[1]["ta_c"])
        for c, s in ranked[:top]:
            print(f"| {c[0]} | {BAND_NAMES[c[1][0]]} {c[1][1:]} | {c[4]} | {c[2]} | {c[3]} | {s['per_year']:.1f} | {s['win']:.0%} "
                  f"| {s['avg']:+.2f} | {s['carry']:+.2f} | {s['t_c']:+.2f} | {s['a_c']:+.2f} / {s['ta_c']:+.2f} | {s['b_c']:+.2f} / {s['tb_c']:+.2f} "
                  f"| {s['pos']}/{s['npairs']} |")


HOLDOUT = (1440, int(np.datetime64("1994-01-01").astype("datetime64[s]").astype(np.int64)),
           int(np.datetime64("2010-09-30").astype("datetime64[s]").astype(np.int64)))


def report_holdout(stats: dict, top_cfgs: list) -> None:
    """第 2 段の週足・逆張りを、初見の 1996〜2010 年(日足の経路、主要 10 ペア)で。"""
    ss = {c: stats[(0, False, "W1d", c)] for c in STAGE2_GRID if c[0] == "fade"}
    print("\n## 第 4 段: 初見の 1994〜2010-09(相場の DB の日足から週足を作る。主要 10 ペア、費用込み + キャリー)\n")
    print(f"- 週足の逆張り 108 通りの平均: 費用込み {np.mean([s['avg'] for s in ss.values()]):+.2f} bp、"
          f"キャリー {np.mean([s['carry'] for s in ss.values()]):+.2f} bp、合計 {np.mean([s['avg_c'] for s in ss.values()]):+.2f} bp。"
          f"プラス {sum(s['avg_c'] > 0 for s in ss.values())}/108、t ≥ 2 は {sum(s['t_c'] >= 2 for s in ss.values())}\n")
    print("| 選び方 | q | 出口 | H | 振り返り | 取引 / 年(1 ペア) | 勝率 | 費用込み bp | キャリー bp | 合計 bp | t(月) | プラスのペア |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for why, c in top_cfgs:
        s = ss[c]
        print(f"| {why} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | {s['per_year']:.1f} | {s['win']:.0%} | {s['avg']:+.2f} | {s['carry']:+.2f} "
              f"| {s['avg_c']:+.2f} | {s['t_c']:+.2f} | {s['pos']}/{s['npairs']} |")


def report_holdout3(s3: dict, hold: dict, plc: dict, seeds: list[int], top: int) -> None:
    """第 3 段の日足・週足の逆張りを、初見の 1994〜2010-09(日足の経路、主要 10 ペア)と偽薬で。"""
    print("\n## 第 5 段: 第 3 段の日足・週足の逆張りを、初見の 1994〜2010-09(主要 10 ペア、日足の経路)と偽薬で\n")
    print("bp は費用込み + キャリー。偽薬は 1 時間足の形を並べ替えた 2010-10〜の相場(主要 10 ペア、素の損益、seed の平均)\n")
    for tf, tfh in (("D1", "D1d"), ("W1", "W1d")):
        print(f"\n### {tf}\n")
        print("| 帯 | 入り | 2010〜(第 3 段)bp | 初見 1994〜2010 bp | 初見でプラス | 初見で t ≥ 2 | 偽薬 bp |\n|---|---|---:|---:|---:|---:|---:|")
        for b in BANDS:
            for entry in ("limit", "market"):
                cs = [c for c in STAGE3_GRID if c[0] == "fade" and c[1] == b and c[4] == entry]
                a = np.mean([s3[(0, False, tf, c)]["avg_c"] for c in cs])
                h = [hold[(0, False, tfh, c)] for c in cs]
                pl = np.mean([plc[(s, True, tf, c)]["avg"] for c in cs for s in seeds])
                print(f"| {BAND_NAMES[b[0]]} {b[1:]} | {'指値' if entry == 'limit' else '成行'} | {a:+.2f} | {np.mean([x['avg_c'] for x in h]):+.2f} "
                      f"| {sum(x['avg_c'] > 0 for x in h)}/{len(h)} | {sum(x['t_c'] >= 2 for x in h)} | {pl:+.2f} |")
        print(f"\n発見(2010-10〜2018-10)の t の上位 {top} を、確認と初見で\n")
        print("| 帯 | 入り | 出口 | H | 2010〜 取引 / 年 | 発見 bp / t | 確認 bp / t | 初見 bp / t | 初見 取引 / 年 | 初見でプラスのペア | 偽薬 bp |")
        print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        cs = [c for c in STAGE3_GRID if c[0] == "fade"]
        for c in sorted(cs, key=lambda c: -s3[(0, False, tf, c)]["ta_c"])[:top]:
            a, h = s3[(0, False, tf, c)], hold[(0, False, tfh, c)]
            pl = np.mean([plc[(s, True, tf, c)]["avg"] for s in seeds])
            print(f"| {BAND_NAMES[c[1][0]]} {c[1][1:]} | {c[4]} | {c[2]} | {c[3]} | {a['per_year']:.1f} | {a['a_c']:+.2f} / {a['ta_c']:+.2f} "
                  f"| {a['b_c']:+.2f} / {a['tb_c']:+.2f} | {h['avg_c']:+.2f} / {h['t_c']:+.2f} | {h['per_year']:.1f} | {h['pos']}/{h['npairs']} | {pl:+.2f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="majors", choices=["majors", "crosses"])
    ap.add_argument("--placebo", type=int, default=0)
    ap.add_argument("--fill-margin", type=float, default=0.0)
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--stage3", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--holdout3", action="store_true")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2 - 2))
    args = ap.parse_args()
    if args.stage2:
        stats = run(PAIRS + CROSSES, [False], [0], args.fill_margin, args.workers, STAGE2_TFS, STAGE2_GRID)
        report_stage2(stats, args.top)
        return
    if args.holdout3:
        fade3 = [c for c in STAGE3_GRID if c[0] == "fade"]
        s3 = run(PAIRS + CROSSES, [False], [0], args.fill_margin, args.workers, {"D1": [0], "W1": [0]}, fade3)
        hold = run(PAIRS, [False], [0], args.fill_margin, args.workers, {"D1d": [0], "W1d": [0]}, fade3, HOLDOUT)
        seeds = [1, 2, 3]
        plc = run(PAIRS, [True], seeds, args.fill_margin, args.workers, {"D1": [0], "W1": [0]}, fade3)
        report_holdout3(s3, hold, plc, seeds, args.top)
        return
    if args.holdout:
        # 第 2 段(2010-10〜)の週足・逆張りから、発見の期間の t で選んだ上位 8 と、主の形
        s2 = run(PAIRS + CROSSES, [False], [0], args.fill_margin, args.workers, {"W1": [0]},
                 [c for c in STAGE2_GRID if c[0] == "fade"])
        ranked = sorted((c for c in STAGE2_GRID if c[0] == "fade"), key=lambda c: -s2[(0, False, "W1", c)]["ta_c"])
        picks = [(f"第 2 段の発見で {k + 1} 位", c) for k, c in enumerate(ranked[:8])] + [("主の形", MAIN)]
        stats = run(PAIRS, [False], [0], args.fill_margin, args.workers, {"W1d": [0]},
                    [c for c in STAGE2_GRID if c[0] == "fade"], HOLDOUT)
        report_holdout(stats, picks)
        return
    if args.stage3:
        stats = run(PAIRS + CROSSES, [False], [0], args.fill_margin, args.workers, STAGE3_TFS, STAGE3_GRID)
        report_stage3(stats, args.top)
        return
    pairs = PAIRS if args.pairs == "majors" else CROSSES
    if args.placebo:
        seeds = list(range(1, args.placebo + 1))
        report_placebo(run(pairs, [True], seeds, args.fill_margin, args.workers), seeds)
        return
    stats = run(pairs, [False, True], [0], args.fill_margin, args.workers)
    report(stats, [0], f"{args.pairs}(約定の余裕 {args.fill_margin} pip)")


if __name__ == "__main__":
    main()
