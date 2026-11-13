"""E061: 暦で決まる時刻の癖(五十日の仲値、月末のロンドン 16 時のフィキシング)は費用を越えるか。

    python -m fxsim.research.calendar_effects

「日時で AI に判断させる」案のうち、未来が漏れずにバックテストできる部分(暦の決まりだけで作れる信号)を測る。
相場は相場の DB(OANDA のデモ口座の足、時刻は UTC)。1 時間足 2010-10〜、5 分足 2025-07〜。

五十日(gotobi): 毎月 5・10・15・20・25 日と月末(土日なら前の金曜。日本の祝日は扱わない)。
    東京の仲値は 9:55 JST = 00:55 UTC(日本に夏時間は無い)。
    買い: 前日 23:00 UTC(8:00 JST)の始値で USDJPY を買い、仲値の前後で売る。
        1 時間足: 00:00 UTC の始値(9:00 JST)で売る版 / 01:00 UTC(00:00 の足の終値、10:00 JST)で売る版
        5 分足: 00:55 UTC(00:50 の足の終値)で売る版
    売り(仲値の後の戻り): 仲値から 2 時間後まで売る(1 時間足は 01:00 → 03:00 UTC、5 分足は 00:55 → 02:55 UTC)
    対照: 同じ時刻の取引を、五十日でない平日に行う
月末のフィキシング: 月の最後の平日の 16:00 ロンドン(夏時間は 15:00 UTC、冬は 16:00 UTC)。
    その前の 1 時間(フィキシングまで)と後の 1 時間で、ドルの向き(対ドル 7 ペアの平均)を測る。
    ドルを買う(無条件)と、月初からのドルの動きの逆に張る(リバランスの代わり。株価のデータが無いので、ドル自身の動きで代える)。
    対照: 同じ時刻・同じ決まりを、月末でない平日に行う
費用: 買いは ask(bid + その足の記録のスプレッド)、売りは bid。記録は足の中の最小のスプレッドなので甘めの見積もり。
単位は 1 回の bp。t は 1 回ごとの損益(月末はペアの平均を 1 回)の平均 ÷ 標準誤差。
"""
# ruff: noqa: E501
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from ..storage import prices
from .intraday_edge import SOURCE

USD_PAIRS = {"EURUSD": -1, "GBPUSD": -1, "AUDUSD": -1, "NZDUSD": -1, "USDJPY": 1, "USDCHF": 1, "USDCAD": 1}
JPY_PAIRS = ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY"]
SPLIT = date(2018, 11, 1)
LONDON = ZoneInfo("Europe/London")


def load(pair: str, tf: int) -> dict:
    con = prices.open_market_db()
    r = prices.read_rates(con, SOURCE, pair, tf)
    con.close()
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    r["sp"] = r["spread"] * pip / 10
    r["idx"] = {int(t): k for k, t in enumerate(r["time"])}
    return r


def ts(d: date, hh: int, mm: int = 0) -> int:
    return int(datetime(d.year, d.month, d.day, hh, mm, tzinfo=timezone.utc).timestamp())


def gotobi_days(d0: date, d1: date) -> set[date]:
    out = set()
    d = date(d0.year, d0.month, 1)
    while d <= d1:
        nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        last = nxt - timedelta(days=1)
        for day in [date(d.year, d.month, k) for k in (5, 10, 15, 20, 25)] + [last]:
            while day.weekday() >= 5:
                day -= timedelta(days=1)
            out.add(day)
        d = nxt
    return out


def stats(v: list[float]) -> tuple[int, float, float, float]:
    a = np.asarray(v, dtype=float)
    if len(a) < 3:
        return len(a), float("nan"), float("nan"), float("nan")
    return len(a), float(a.mean()), float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))), float((a > 0).mean())


def trade(r: dict, t_in: int, t_out: int, side: int, open_in: bool = True, open_out: bool = True) -> tuple[float, float] | None:
    """t_in の足の始値(open_in=False なら終値)で入り、t_out の足の始値(False なら終値)で出る。(素の bp、費用込みの bp)。"""
    i, j = r["idx"].get(t_in), r["idx"].get(t_out)
    if i is None or j is None or j < i:
        return None
    p0 = r["open"][i] if open_in else r["close"][i]
    p1 = r["open"][j] if open_out else r["close"][j]
    gross = side * (p1 - p0) / p0 * 1e4
    cost = (r["sp"][i] + r["sp"][j]) / 2 / p0 * 1e4     # 往復でスプレッド 1 回分
    return gross, gross - cost


def gotobi() -> None:
    print("\n## 五十日の仲値(USDJPY ほか円のペア)\n")
    print("| ペア | 足 | 取引 | 五十日: 回数 / 素の bp / 費用込み bp / t(素) / 勝率 | 発見(〜2018-10)素の bp / t | 確認 素の bp / t "
          "| 対照(五十日でない平日): 回数 / 素の bp / t | 差の t |\n|---|---|---|---|---|---|---|---:|")
    for pair in JPY_PAIRS:
        for tf in (60, 5):
            r = load(pair, tf)
            t = r["time"]
            d0 = datetime.fromtimestamp(int(t[0]), timezone.utc).date()
            d1 = datetime.fromtimestamp(int(t[-1]), timezone.utc).date()
            gdays = gotobi_days(d0, d1)
            if tf == 60:
                variants = [("買い 23→00 UTC(9:00 JST まで)", 1, (-1, 23, 0), (0, 0, 0), True, True),
                            ("買い 23→01 UTC(10:00 JST まで)", 1, (-1, 23, 0), (0, 0, 0), True, False),
                            ("売り 01→03 UTC(仲値の後)", -1, (0, 1, 0), (0, 3, 0), True, True)]
            else:
                variants = [("買い 23:00→00:55 UTC(仲値まで)", 1, (-1, 23, 0), (0, 0, 50), True, False),
                            ("売り 00:55→02:55 UTC(仲値の後)", -1, (0, 0, 50), (0, 2, 50), False, False)]
            for name, side, (din, hin, min_), (dout, hout, mout), oi, oo in variants:
                res = {"g": [], "c": [], "ga": [], "gb": [], "n": [], "dates": []}
                d = d0 + timedelta(days=1)
                while d <= d1:
                    if d.weekday() < 5:
                        x = trade(r, ts(d + timedelta(days=din), hin, min_), ts(d + timedelta(days=dout), hout, mout), side, oi, oo)
                        if x is not None:
                            if d in gdays:
                                res["g"].append(x[0])
                                res["c"].append(x[1])
                                (res["ga"] if d < SPLIT else res["gb"]).append(x[0])
                            else:
                                res["n"].append(x[0])
                    d += timedelta(days=1)
                n, mg, tg, win = stats(res["g"])
                _, mc, _, _ = stats(res["c"])
                na, ma, ta, _ = stats(res["ga"])
                nb, mb, tb, _ = stats(res["gb"])
                nn, mn, tn, _ = stats(res["n"])
                g, c = np.asarray(res["g"]), np.asarray(res["n"])
                tdiff = (g.mean() - c.mean()) / np.sqrt(g.var(ddof=1) / len(g) + c.var(ddof=1) / len(c)) if len(g) > 2 else float("nan")
                print(f"| {pair} | {'H1' if tf == 60 else 'M5'} | {name} | {n} / {mg:+.2f} / {mc:+.2f} / {tg:+.2f} / {win:.0%} "
                      f"| {ma:+.2f} / {ta:+.2f}({na}) | {mb:+.2f} / {tb:+.2f}({nb}) | {nn} / {mn:+.2f} / {tn:+.2f} | {tdiff:+.2f} |")


def month_end() -> None:
    data = {p: load(p, 60) for p in USD_PAIRS}
    t = data["USDJPY"]["time"]
    d0 = datetime.fromtimestamp(int(t[0]), timezone.utc).date()
    d1 = datetime.fromtimestamp(int(t[-1]), timezone.utc).date()

    def fix_hour(d: date) -> int:                     # 16:00 ロンドンの UTC の時
        return datetime(d.year, d.month, d.day, 16, tzinfo=LONDON).astimezone(timezone.utc).hour

    def usd_ret(h_from: int, h_to: int, d: date, close_from: bool = False) -> tuple[float, float] | None:
        """ドルを買ったときの、7 ペア平均の(素の bp、費用込みの bp)。h_from の足の始値 → h_to の足の始値。"""
        g, c = [], []
        for p, sgn in USD_PAIRS.items():
            x = trade(data[p], ts(d, h_from), ts(d, h_to), sgn)
            if x is None:
                return None
            g.append(x[0])
            c.append(x[1])
        return float(np.mean(g)), float(np.mean(c))

    def usd_level(d: date, h: int) -> float | None:
        v = []
        for p, sgn in USD_PAIRS.items():
            k = data[p]["idx"].get(ts(d, h))
            if k is None:
                return None
            v.append(sgn * np.log(data[p]["open"][k]))
        return float(np.mean(v))

    # 月の最後の平日
    ends, d = set(), date(d0.year, d0.month, 1)
    while d <= d1:
        nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        last = nxt - timedelta(days=1)
        while last.weekday() >= 5:
            last -= timedelta(days=1)
        ends.add(last)
        d = nxt
    rows = {}
    prev_level = None
    d = d0 + timedelta(days=1)
    month_start_level: dict[tuple[int, int], float] = {}
    while d <= d1:
        if d.weekday() < 5:
            fh = fix_hour(d)
            lvl = usd_level(d, fh - 1)
            key = (d.year, d.month)
            if lvl is not None and key not in month_start_level and prev_level is not None:
                month_start_level[key] = prev_level           # 前の月末のフィキシングの 1 時間前の値から
            mtd = (lvl - month_start_level[key]) * 1e4 if lvl is not None and key in month_start_level else None
            before = usd_ret(fh - 1, fh, d)
            after = usd_ret(fh, fh + 1, d)
            tag = "end" if d in ends else "other"
            for win_name, x in (("前の 1 時間", before), ("後の 1 時間", after)):
                if x is None:
                    continue
                rows.setdefault((tag, win_name, "ドル買い"), []).append((x[0], x[1], d))
                if mtd is not None and mtd != 0:
                    s = -np.sign(mtd)
                    rows.setdefault((tag, win_name, "月初からの逆"), []).append((s * x[0], s * x[0] - (x[0] - x[1]), d))
            if d in ends and lvl is not None:
                prev_level = lvl
        d += timedelta(days=1)
    print("\n## 月末のロンドン 16 時のフィキシング(対ドル 7 ペアの平均、1 時間足)\n")
    print("| 窓 | 張り方 | 月末: 回数 / 素の bp / 費用込み bp / t(素) / 勝率 | 発見 素の bp / t | 確認 素の bp / t | 対照(月末でない平日): 回数 / 素の bp / t | 差の t |")
    print("|---|---|---|---|---|---|---:|")
    for win_name in ("前の 1 時間", "後の 1 時間"):
        for how in ("ドル買い", "月初からの逆"):
            e = rows.get(("end", win_name, how), [])
            o = rows.get(("other", win_name, how), [])
            g = [x[0] for x in e]
            n, mg, tg, win = stats(g)
            mc = float(np.mean([x[1] for x in e])) if e else float("nan")
            ga = [x[0] for x in e if x[2] < SPLIT]
            gb = [x[0] for x in e if x[2] >= SPLIT]
            _, ma, ta, _ = stats(ga)
            _, mb, tb, _ = stats(gb)
            go = [x[0] for x in o]
            nn, mn, tn, _ = stats(go)
            a, b = np.asarray(g), np.asarray(go)
            tdiff = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            print(f"| {win_name} | {how} | {n} / {mg:+.2f} / {mc:+.2f} / {tg:+.2f} / {win:.0%} | {ma:+.2f} / {ta:+.2f} "
                  f"| {mb:+.2f} / {tb:+.2f} | {nn} / {mn:+.2f} / {tn:+.2f} | {tdiff:+.2f} |")


def main() -> None:
    gotobi()
    month_end()


if __name__ == "__main__":
    main()
