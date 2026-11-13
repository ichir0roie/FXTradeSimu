"""E062: 指標発表(米雇用統計)の前後を狙う売買は費用を越えるか。

    python -m fxsim.research.event_nfp                       # 全部
    python -m fxsim.research.event_nfp --extra-pips 2        # 発表の足と次の足でスプレッドをさらに 2 pip 広く見る
    python -m fxsim.research.event_nfp --event claims        # 米新規失業保険申請件数(毎週木曜 8:30、E067)

発表の日時は AI に選ばせず、BLS の決まりで作る: 対象月の 12 日を含む週(日〜土)が終わった土曜の 20 日後の金曜、
8:30 ニューヨーク時間(1 月 3 日以前なら 1 週後、7 月 4 日・7 月 3 日(振替)の金曜なら前の木曜)。2013-10 と 2025-10〜12 は
政府の閉鎖で日がずれたので、覚えている公表日(2013-10-22、2013-11-08、2025-11-20、2025-12-16)に置き換えた。
日がずれていても、その回は「発表の無い時刻」が混ざるだけで、向きにも振れの大きさにも偏りは入らない(選ぶのに相場を見ない)。
相場は相場の DB の 1 時間足(2010-10〜)、対ドル 7 ペア。発表は 1 時間足の真ん中(:30)に来る。
発表の足の始値 P0(発表の 30 分前)を基準にし、振れの物差し wv は、そのペアの直前 12 回の発表の足の(高値 − 安値)/ 始値の中央値。

A 片側・逆指値なし(利用者の最初の案): 発表の足の始値に成行で入り、P0 から k x wv で利確(指値。入った足から見る)。
  逆指値は置かない。出口: calm(発表の後、足の幅が直近 480 本の中央値以下に戻った最初の足の終値)/ 4h / 24h /
  tp_or_5d(利確まで最長 5 日待つ。負けは戻るまで持つ)。向き: でたらめ(seed 1〜20)/ 20 日の流れの向き / その逆
B 発表をまたぐ両側の注文: P0 ± k x wv に、指値(fade、逆張り)か逆指値(break、順張り)を発表の足と次の足の間だけ置く。
  出口: fade は tp(P0 に戻ったら利確、最長 24h)/ tp_stop(tp + 建値から 2k x wv 逆の逆指値)/ 2h / 24h、break は 2h / 24h
  (break の損切りを近くに置くと、1 時間足では同じ足の順番が分からず不利に倒れすぎるので、時間で出る形だけ)
C 跳ねた後の押し目: 発表の足の終値が P0 から 0.5 x wv 以上動いたら、その動きの半分戻った値に指値(with = 動いた向き、
  against = 逆向き)を 4 時間置く。出口 24h / 72h / stop_p0(P0 で損切り、最長 24h)。trend は 200 日の平均の向きと
  動いた向きがそろうときだけ
対照: 同じ決まりを、発表の無い金曜の同じ時刻(8:30 ニューヨーク)に当てる。
約定は E050 と同じく不利な側に倒す(買いは ask、ロールオーバーの足では約定させない、成行 0.1 pip・逆指値 0.3 pip の滑り、
同じ足で損切りと利確なら損切りが先、指値で入った足では利確を見ない)。
1 回の損益は、7 ペアの平均を発表 1 回分として数える(ドルのペアは発表で一緒に動くので、独立なのは発表の回数)。
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..calendar import claims_dates, nfp_dates
from ..calendar import nfp_bar_start as at_830
from ..storage import prices
from .intraday_edge import BAD_HOURS, SOURCE

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]
NY = ZoneInfo("America/New_York")
SPLIT = int(datetime(2018, 11, 1, tzinfo=timezone.utc).timestamp())
SLIP_MKT, SLIP_STOP = 0.1, 0.3
SEEDS = 20


def load(pair: str) -> dict:
    con = prices.open_market_db()
    h = prices.read_rates(con, SOURCE, pair, 60)
    con.close()
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    t = h["time"]
    rng_ = h["high"] - h["low"]
    med = pd.Series(rng_).rolling(480).median().shift(1).to_numpy(copy=True)   # 足 i の前の 480 本
    c = h["close"]
    trend20 = np.full(len(t), 0.0)
    trend20[480:] = np.sign(c[480:] - c[:-480])
    sma = np.full(len(t), np.nan)
    cs = np.cumsum(np.r_[0.0, c])
    sma[4800:] = (cs[4800:-1] - cs[:-4801]) / 4800                      # 足 i の前の 4800 本(200 日)
    return {"t": t, "o": h["open"], "h": h["high"], "l": h["low"], "c": c, "sp": h["spread"] * pip / 10,
            "pip": pip, "bad": np.isin((t // 3600) % 24, BAD_HOURS), "med": med, "trend20": trend20, "sma": sma,
            "idx": {int(x): k for k, x in enumerate(t)}}


class Path:
    """1 本のペアの 1 時間足の上で、1 つの建玉を追う(不利な側に倒す約定)。"""

    def __init__(self, x: dict, extra: float, ev: int):
        self.x, self.n = x, len(x["t"])
        self.sp = x["sp"].copy()
        self.sp[ev:ev + 2] += extra * x["pip"]           # 発表の足と次の足のスプレッドを広く見る
        self.s_mkt, self.s_stop = SLIP_MKT * x["pip"], SLIP_STOP * x["pip"]

    def next_good(self, j: int) -> int:
        bad = self.x["bad"]
        while j < self.n and bad[j]:
            j += 1
        return j

    def market_in(self, j: int, side: int) -> float:
        o = self.x["o"][j]
        return o + self.sp[j] + self.s_mkt if side > 0 else o - self.s_mkt

    def market_out(self, j: int, side: int, at_close: bool = False) -> float:
        p = self.x["c"][j] if at_close else self.x["o"][j]
        sp = self.sp[j]
        return p - self.s_mkt if side > 0 else p + sp + self.s_mkt

    def limit_fill(self, a: int, b: int, side: int, lvl: float) -> tuple[int, float] | None:
        x, sp = self.x, self.sp
        for j in range(a, min(b, self.n)):
            if x["bad"][j]:
                continue
            if side > 0 and x["l"][j] + sp[j] <= lvl:
                return j, min(lvl, x["o"][j] + sp[j])
            if side < 0 and x["h"][j] >= lvl:
                return j, max(lvl, x["o"][j])
        return None

    def stop_fill(self, a: int, b: int, side: int, lvl: float) -> tuple[int, float] | None:
        """side の向きに入る逆指値(買いは ask が lvl 以上、売りは bid が lvl 以下)。"""
        x, sp = self.x, self.sp
        for j in range(a, min(b, self.n)):
            if x["bad"][j]:
                continue
            if side > 0 and x["h"][j] + sp[j] >= lvl:
                return j, max(lvl, x["o"][j] + sp[j]) + self.s_stop
            if side < 0 and x["l"][j] <= lvl:
                return j, min(lvl, x["o"][j]) - self.s_stop
        return None

    def run_exit(self, j1: int, side: int, entry: float, j_end: int, take: float | None, stop: float | None,
                 take_in_entry_bar: bool, calm: bool = False) -> tuple[int, float]:
        """j1 で入った建玉を、利確・損切り・(calm なら落ち着いた足の終値)・j_end の始値の成行で出す。"""
        x, sp = self.x, self.sp
        j_end = self.next_good(min(j_end, self.n - 1))
        if j_end >= self.n:
            j_end = self.n - 1
        for j in range(j1, j_end):
            if stop is not None:
                hit = x["l"][j] <= stop if side > 0 else x["h"][j] + sp[j] >= stop
                if hit:
                    px = min(stop, x["o"][j]) - self.s_stop if side > 0 else max(stop, x["o"][j] + sp[j]) + self.s_stop
                    return j, px
            if take is not None and (j > j1 or take_in_entry_bar) and not x["bad"][j]:
                hit = x["h"][j] >= take if side > 0 else x["l"][j] + sp[j] <= take
                if hit:
                    px = max(take, x["o"][j]) if side > 0 else min(take, x["o"][j] + sp[j])
                    return j, px
            if calm and j > j1 and not x["bad"][j] and x["h"][j] - x["l"][j] <= x["med"][j]:
                return j, self.market_out(j, side, at_close=True)
        return j_end, self.market_out(j_end, side)


def wv_series(x: dict, occ: list[int]) -> dict[int, float]:
    """各回の直前 12 回の、その時刻の足の(高値 − 安値)/ 始値の中央値。"""
    out, hist = {}, []
    for ev in occ:
        k = x["idx"].get(ev)
        if len(hist) >= 6:
            out[ev] = float(np.median(hist[-12:]))
        if k is not None:
            hist.append((x["h"][k] - x["l"][k]) / x["o"][k])
    return out


def simulate(pair: str, occ: list[int], extra: float) -> dict:
    x = load(pair)
    wvs = wv_series(x, occ)
    res: dict[tuple, list] = {}

    def add(key, ev, v):
        res.setdefault(key, []).append((ev, v))

    for ev in occ:
        k = x["idx"].get(ev)
        if k is None or ev not in wvs or k + 200 >= len(x["t"]) or np.isnan(x["sma"][k]) or x["bad"][k]:
            continue
        P0 = x["o"][k]
        wv = wvs[ev] * P0
        p = Path(x, extra, k)
        bp = 1e4 / P0
        # A 片側・逆指値なし
        rng = np.random.default_rng([PAIRS.index(pair), ev])
        dirs = [("random", s, int(rng.choice([-1, 1]))) for s in range(1, SEEDS + 1)]
        tr = int(x["trend20"][k]) or 1
        dirs += [("trend", 0, tr), ("anti", 0, -tr)]
        for dname, seed, side in dirs:
            e = p.market_in(k, side)
            for ktp in (0.25, 0.5, 1.0, 0.0):
                take = P0 + side * ktp * wv if ktp else None
                for ex in ("calm", "4h", "24h", "tp_or_5d"):
                    j_end = k + {"calm": 120, "4h": 4, "24h": 24, "tp_or_5d": 120}[ex]
                    j, px = p.run_exit(k, side, e, j_end, take, None, True, calm=(ex == "calm"))
                    add(("A", dname, seed, ktp, ex), ev, (side * (px - e) * bp, (j - k)))
        # B 両側の注文
        for kind in ("fade", "break"):
            for kb in (0.5, 1.0):
                up, dn = P0 + kb * wv, P0 - kb * wv
                if kind == "fade":
                    fu, fd = p.limit_fill(k, k + 2, -1, up), p.limit_fill(k, k + 2, 1, dn)
                    su, sd = -1, 1
                else:
                    fu, fd = p.stop_fill(k, k + 2, 1, up), p.stop_fill(k, k + 2, -1, dn)
                    su, sd = 1, -1
                exits = ("tp", "tp_stop", "2h", "24h") if kind == "fade" else ("2h", "24h")
                for ex in exits:
                    if fu is None and fd is None:
                        add(("B", kind, kb, ex), ev, (0.0, 0))
                        continue
                    if fu is not None and fd is not None and fu[0] == fd[0]:
                        add(("B", kind, kb, ex), ev, ((-(su * fu[1] + sd * fd[1])) * bp, 0))
                        continue
                    first_up = fd is None or (fu is not None and fu[0] < fd[0])
                    (j1, e1), s1 = (fu, su) if first_up else (fd, sd)
                    other = fd if first_up else fu
                    s2 = sd if first_up else su
                    take = P0 if ex in ("tp", "tp_stop") else None
                    stop = e1 - s1 * 2 * kb * wv if ex == "tp_stop" else None
                    hold = 2 if ex == "2h" else 24
                    j, px = p.run_exit(j1, s1, e1, k + hold, take, stop, kind != "fade")
                    if other is not None and other[0] < j:
                        add(("B", kind, kb, ex), ev, ((-(s1 * e1 + s2 * other[1])) * bp, 0))
                    else:
                        add(("B", kind, kb, ex), ev, (s1 * (px - e1) * bp, j - k))
        # C 跳ねた後の押し目
        mv = x["c"][k] - P0
        if abs(mv) >= 0.5 * wv:
            sgn = int(np.sign(mv))
            aligned = np.sign(x["c"][k] - x["sma"][k]) == sgn
            for who in ("with", "against"):
                side = sgn if who == "with" else -sgn
                lvl = x["c"][k] - sgn * 0.5 * abs(mv) if who == "with" else x["c"][k] + sgn * 0.25 * abs(mv)
                f = p.limit_fill(k + 1, k + 5, side, lvl)
                for flt in ("all", "trend"):
                    for ex in ("24h", "72h", "stop_p0"):
                        key = ("C", who, flt, ex)
                        if f is None or (flt == "trend" and not aligned):
                            add(key, ev, (0.0, 0))
                            continue
                        j1, e1 = f
                        hold = {"24h": 24, "72h": 72, "stop_p0": 24}[ex]
                        stop = P0 if (ex == "stop_p0" and who == "with") else (x["c"][k] + sgn * abs(mv) if ex == "stop_p0" else None)
                        j, px = p.run_exit(j1, side, e1, j1 + hold, None, stop, False)
                        add(key, ev, (side * (px - e1) * bp, j - k))
        else:
            for who in ("with", "against"):
                for flt in ("all", "trend"):
                    for ex in ("24h", "72h", "stop_p0"):
                        add(("C", who, flt, ex), ev, (0.0, 0))
    return res


def pool(results: list[dict]) -> dict[tuple, dict[int, list[float]]]:
    """ペアごとの結果を、回(発表)ごとに集める。0.0 は取引の無いペア。"""
    out: dict[tuple, dict[int, list[float]]] = {}
    for r in results:
        for key, rows in r.items():
            d = out.setdefault(key, {})
            for ev, (v, _) in rows:
                d.setdefault(ev, []).append(v)
    return out


def describe(d: dict[int, list[float]]) -> dict:
    evs = sorted(d)
    per_ev = np.array([np.mean(d[e]) for e in evs])
    allv = np.concatenate([np.asarray(d[e]) for e in evs])
    trades = allv[allv != 0]
    ts_ = np.array(evs)
    a, b = per_ev[ts_ < SPLIT], per_ev[ts_ >= SPLIT]

    def t_of(v):
        return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 2 and v.std() > 0 else 0.0

    return {"n_ev": len(evs), "n_tr": len(trades), "mean": float(per_ev.mean()), "t": t_of(per_ev),
            "a": float(a.mean()) if len(a) else 0.0, "ta": t_of(a), "b": float(b.mean()) if len(b) else 0.0, "tb": t_of(b),
            "per_tr": float(trades.mean()) if len(trades) else 0.0, "win": float((trades > 0).mean()) if len(trades) else 0.0,
            "worst": float(trades.min()) if len(trades) else 0.0, "q01": float(np.quantile(trades, 0.01)) if len(trades) else 0.0,
            "avg_win": float(trades[trades > 0].mean()) if (trades > 0).any() else 0.0,
            "avg_loss": float(trades[trades < 0].mean()) if (trades < 0).any() else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-pips", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--event", default="nfp", choices=["nfp", "claims"])
    args = ap.parse_args()
    if args.event == "nfp":
        dates = nfp_dates(date(2010, 10, 1), date(2026, 11, 13))
        ctrl_day, ctrl_name = 4, "対照(発表の無い金曜)"
    else:                     # 米新規失業保険申請件数(毎週木曜)。対照は発表の無い火曜の同じ時刻
        dates = claims_dates(date(2010, 10, 1), date(2026, 11, 13))
        ctrl_day, ctrl_name = 1, "対照(発表の無い火曜)"
    ev_set = set(dates)
    fridays = []
    d = date(2010, 10, 1)
    while d <= date(2026, 11, 13):
        if d.weekday() == ctrl_day and d not in ev_set:
            fridays.append(d)
        d += timedelta(days=1)
    occ = {"発表": [at_830(d) for d in dates], ctrl_name: [at_830(d) for d in fridays]}
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(args.workers) as ex:
        out = {name: pool(list(ex.map(simulate, PAIRS, [o] * len(PAIRS), [args.extra_pips] * len(PAIRS))))
               for name, o in occ.items()}
    # 発表の足が本当に荒れているか(日付の決まりの確認。選ぶのには使わない)
    x = load("EURUSD")

    def rng_bp(ts_list):
        v = [(x["h"][x["idx"][t]] - x["l"][x["idx"][t]]) / x["o"][x["idx"][t]] * 1e4 for t in ts_list if t in x["idx"]]
        return np.array(v)

    re, rc = rng_bp(occ["発表"]), rng_bp(occ[ctrl_name])
    print(f"\n{args.event}: 発表 {len(dates)} 回(対照 {len(fridays)} 回)。EURUSD の発表の足の幅: 中央 {np.median(re):.1f} bp"
          f"(対照 {np.median(rc):.1f} bp)。対照の中央値より狭い発表の足 {int((re < np.median(rc)).sum())} 回。追加のスプレッド {args.extra_pips} pip")
    hdr = ("| 形 | 回: 発表 / 対照 | 1 回(7 ペア平均)bp: 発表 / 対照 | t: 発表 / 対照 | 発見 bp / t | 確認 bp / t | 取引あたり bp | 勝率 "
           "| 平均の勝ち / 負け | 1% 点 / 最悪 |\n|---|---|---|---|---|---|---:|---:|---|---|")

    def line(name, s, c):
        return (f"| {name} | {s['n_ev']} / {c['n_ev']} | {s['mean']:+.2f} / {c['mean']:+.2f} | {s['t']:+.2f} / {c['t']:+.2f} "
                f"| {s['a']:+.2f} / {s['ta']:+.2f} | {s['b']:+.2f} / {s['tb']:+.2f} | {s['per_tr']:+.2f} | {s['win']:.0%} "
                f"| {s['avg_win']:+.1f} / {s['avg_loss']:+.1f} | {s['q01']:+.0f} / {s['worst']:+.0f} |")

    ev, ct = out["発表"], out[ctrl_name]
    print("\n## A 片側・逆指値なし(発表の 30 分前に成行、利確 k x wv、負けは出口まで持つ)\n")
    print("でたらめな向きは seed 1〜20 の平均(t は seed ごとの t の平均、括弧は seed ごとの 1 回の bp の範囲)\n")
    print(hdr)
    for ktp in (0.25, 0.5, 1.0, 0.0):
        for exn in ("calm", "4h", "24h", "tp_or_5d"):
            ss = [describe(ev[("A", "random", s, ktp, exn)]) for s in range(1, SEEDS + 1)]
            cs = [describe(ct[("A", "random", s, ktp, exn)]) for s in range(1, SEEDS + 1)]
            avg = {k2: float(np.mean([s[k2] for s in ss])) for k2 in ss[0]}
            cavg = {k2: float(np.mean([s[k2] for s in cs])) for k2 in cs[0]}
            avg["worst"] = min(s["worst"] for s in ss)
            lo, hi = min(s["mean"] for s in ss), max(s["mean"] for s in ss)
            print(line(f"でたらめ・利確 {ktp or 'なし'}・{exn}({lo:+.1f}〜{hi:+.1f})", avg, cavg))
    for dname in ("trend", "anti"):
        for ktp in (0.25, 0.5, 1.0, 0.0):
            for exn in ("calm", "4h", "24h", "tp_or_5d"):
                key = ("A", dname, 0, ktp, exn)
                print(line(f"{'20 日の流れ' if dname == 'trend' else '流れの逆'}・利確 {ktp or 'なし'}・{exn}", describe(ev[key]), describe(ct[key])))
    print("\n## B 発表をまたぐ両側の注文(P0 ± k x wv、発表の足と次の足の間だけ)\n")
    print(hdr)
    for kind in ("fade", "break"):
        for kb in (0.5, 1.0):
            for exn in (("tp", "tp_stop", "2h", "24h") if kind == "fade" else ("2h", "24h")):
                key = ("B", kind, kb, exn)
                print(line(f"{'指値(逆張り)' if kind == 'fade' else '逆指値(順張り)'}・k {kb}・{exn}", describe(ev[key]), describe(ct[key])))
    print("\n## C 跳ねた後の押し目(発表の足で 0.5 x wv 以上動いたら、半分戻りに指値を 4 時間)\n")
    print(hdr)
    for who in ("with", "against"):
        for flt in ("all", "trend"):
            for exn in ("24h", "72h", "stop_p0"):
                key = ("C", who, flt, exn)
                print(line(f"{'動いた向き' if who == 'with' else '逆向き'}・{'全部' if flt == 'all' else '200 日の向きとそろう'}・{exn}",
                           describe(ev[key]), describe(ct[key])))


if __name__ == "__main__":
    main()
