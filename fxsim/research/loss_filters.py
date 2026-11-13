"""E035: E033 で見つけた負けどころを避けると、短期の順張りはどうなるか(USDJPY の実データ、時間足ごとに 1 本、OANDA)。

    python -m fxsim.research.loss_filters [--workers N]

順張り 8 本(E033 と同じ、既定のパラメータ)に、対策を 1 つずつと組み合わせで掛け、
方向でたらめ版にも同じ対策を掛けて対で回す。
    1  ロールオーバー前後(日本時間 5〜8 時台)は新規を出さない(取引時間。持ち高は持ち越す)
    1' 同じ時間帯に入る足で持ち高をたたみ、新規も出さない(取引時間の外は持たない)
    4  荒い局面(ATR(14) / 直前 500 本の ATR の中央値 > 1.25)では新規を出さない(entry_filter)
    5a 一方向の局面(20 本の効率比 > 0.4)では新規を出さない(entry_filter、5 の切り分け用)
    5b スワップを受け取る向きだけ(EngineConfig.swap_side_only。USDJPY では買いだけ)
    6  金曜(日本時間。土曜の早朝 = NY の金曜を含む)は新規を出さない(持ち高は持ち越す)
基準も同じ書き方(日本時間、曜日の制限なし)で回す。E033 の基準(UTC の月〜金)とは日曜の夜の扱いだけ違う。
5b の勝ちには円安(期間のドリフト)とスワップが入る。ドリフト分 = 期間の平均の値上がり x 持っていた時間 x 向き。
"""
from __future__ import annotations

import argparse
import contextlib
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.portfolio import build_backtester
from fxsim.backtest.schedule import DAY_NAMES, TradingSchedule
from fxsim.catalog import *  # noqa: F403
from fxsim.experiments.config import default_workers
from fxsim.markets import MARKETS
from fxsim.research.loss_timing import CASES, JA, JST, NS_MIN, PAIR, TREND, _week_t, bar_features
from fxsim.strategies import STRATEGIES
from fxsim.strategies.indicators import atr as _atr

MAIN_CASES = CASES[:5]
ALL_DAYS = list(DAY_NAMES)
NO_FRIDAY = ["sun", "mon", "tue", "wed", "thu"]          # 日本時間の金曜と土曜(NY の金曜の午後)を外す
OUT_ROLLOVER = [["09:00", "05:00"]]                      # 日本時間 5:00〜8:59 を外す
VOL, ER = 1.25, 0.4

# (ラベル, schedule, entry_filter の引数, swap_side_only)
VARIANTS = [
    ("基準", {}, {}, False),
    ("1 ロールオーバーは新規なし", {"hours": OUT_ROLLOVER}, {}, False),
    ("1' ロールオーバーの前にたたむ", {"hours": OUT_ROLLOVER, "outside": "close"}, {}, False),
    ("4 荒いときは新規なし", {}, {"max_vol_ratio": VOL}, False),
    ("6 金曜は新規なし", {"days": NO_FRIDAY}, {}, False),
    ("5a 一方向では新規なし", {}, {"max_er": ER}, False),
    ("5b スワップを受け取る向きだけ", {}, {}, True),
    ("1'+4+6", {"hours": OUT_ROLLOVER, "outside": "close", "days": NO_FRIDAY}, {"max_vol_ratio": VOL}, False),
    ("1'+4+6+5b", {"hours": OUT_ROLLOVER, "outside": "close", "days": NO_FRIDAY}, {"max_vol_ratio": VOL}, True),
    # 結果を見てから足した版(4 は単独で悪く、6 は効かなかったので外したもの)
    ("1'+5b(後から)", {"hours": OUT_ROLLOVER, "outside": "close"}, {}, True),
]


def schedule(spec: dict) -> TradingSchedule:
    return TradingSchedule(tz="JST", days=spec.get("days", ALL_DAYS), hours=spec.get("hours"),
                           outside=spec.get("outside", "hold"))


def job(arg):
    label, tf, data_tf, start, end, method = arg
    with contextlib.suppress(OSError):
        os.nice(10)
    md = MARKETS.create("real_pair", {"source": {"path": f"db:{PAIR}/{data_tf}", "start": start,
                                                 "end": end}}).generate(seed=1)
    bars = md.bars(tf)
    _, er, _ = bar_features(bars)
    lc = np.log(bars.close)
    mu = (lc[-1] - lc[0]) / float(bars.t[-1] - bars.t[0])        # 1 ns あたりの平均の対数の値上がり
    a = _atr(bars.high, bars.low, bars.close, 14)
    out = []
    for vlabel, sched, filt, swap in VARIANTS:
        for kind in ("model", "null"):
            inner, inner_params = (method, {}) if kind == "model" else ("direction_null", {"inner": method})
            s = STRATEGIES.create("entry_filter", {"inner": inner, "inner_params": inner_params, "timeframe": tf,
                                                   **filt})
            broker = BrokerConfig.preset("oanda", {"mt5": {"preset": "jp_domestic"}})
            bt = build_backtester(md, s, broker, EngineConfig(units=10_000, swap_side_only=swap), seed=1,
                                  schedule=schedule(sched))
            bt.run()
            trades = []
            for t in bt.trades:
                k = int(np.searchsorted(bars.t, t.entry_t, "right")) - 2
                if k < 0 or t.reason == "end":
                    continue
                held = (t.exit_t - t.entry_t) / (tf * NS_MIN)
                ext = (t.side * (lc[k] - lc[k - 20]) / (a[k] / bars.close[k])) if k >= 20 and a[k] > 0 else np.nan
                trades.append({
                    "pnl": float(t.pnl), "pct": float(t.pnl / (abs(t.units) * t.entry_price) * 100),
                    "pips": float(t.pips), "gross": float(t.gross_pips), "cost": float(t.cost_pips),
                    "swap": float(t.swap), "side": int(t.side), "held": float(held), "reason": t.reason,
                    "drift": float(t.side * mu * (t.exit_t - t.entry_t) * t.entry_price / md.pip),
                    "week": int((t.entry_t + JST) // (7 * 1440 * NS_MIN)),
                    "er": float(er[k]) if np.isfinite(er[k]) else np.nan, "ext": float(ext)})
            out.append({"case": label, "method": method, "variant": vlabel, "kind": kind, "trades": trades})
    # 曜日ごとの値動きの大きさ(日本時間。1 本の |対数リターン| の平均を全体の平均で割る)と、荒い足の割合
    r = np.abs(np.diff(lc, prepend=lc[0]))
    jst = bars.t + bars.timeframe * NS_MIN + JST
    dow = ((jst // (1440 * NS_MIN)) + 3) % 7
    vr, _, _ = bar_features(bars)
    week = {}
    for d in range(7):
        m = dow == d
        if m.sum() > 50:
            week[DAY_NAMES[d]] = (float(r[m].mean() / r.mean()), float(np.nanmean(vr[m] > VOL)), int(m.sum()))
    return out, (label, method, week)


def stats(tr) -> dict:
    if not tr:
        return {"n": 0}
    pnl = np.array([t["pnl"] for t in tr])
    pct = np.array([t["pct"] for t in tr])
    pips = np.array([t["pips"] for t in tr])
    win = pnl > 0
    wp, lp = pips[win], pips[~win]
    return {"n": len(tr), "win": win.mean(), "win_pct": pct[win].mean() if win.any() else 0.0,
            "loss_pct": pct[~win].mean() if (~win).any() else 0.0,
            "be": float(-lp.mean() / (wp.mean() - lp.mean())) if len(wp) and len(lp) else float("nan"),
            "exp": pips.mean(), "gross": np.mean([t["gross"] for t in tr]), "cost": np.mean([t["cost"] for t in tr]),
            "total": pnl.sum() / 1e4, "swap": np.mean([t["swap"] for t in tr]) / 100,      # 円 → pip(1 万通貨)
            "drift": np.mean([t["drift"] for t in tr]), "t": _week_t(tr),
            "fake": np.mean([t["held"] <= 3 for t in tr]), "sl": np.mean([t["reason"] == "sl" for t in tr])}


def _pool(rows, case, variant, kind, methods=TREND):
    return [t for r in rows if r["case"] == case and r["variant"] == variant and r["kind"] == kind
            and r["method"] in methods for t in r["trades"]]


def table_variants(rows) -> list[str]:
    out = []
    for case in [c[0] for c in MAIN_CASES]:
        out += [f"### {case}", "",
                "| 対策 | 取引 | 勝率 | 勝ちの平均 | 負けの平均 | 損益分岐の勝率 | 期待値(pip) | 素(pip) | 費用(pip) | "
                "スワップ(pip) | ドリフト分(pip) | 帰無の期待値(pip) | 合計(万円) | 週の t | 3 本以内 | 損切り |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for v, *_ in VARIANTS:
            s, sn = stats(_pool(rows, case, v, "model")), stats(_pool(rows, case, v, "null"))
            if s["n"] == 0:
                out.append(f"| {v} | 0 |" + " - |" * 14)
                continue
            out.append(f"| {v} | {s['n']} | {s['win']:.0%} | {s['win_pct']:+.3f}% | {s['loss_pct']:+.3f}% | "
                       f"{s['be']:.0%} | {s['exp']:+.2f} | {s['gross']:+.2f} | {s['cost']:.2f} | {s['swap']:+.2f} | "
                       f"{s['drift']:+.2f} | {sn.get('exp', 0):+.2f} | {s['total']:+.1f} | {s['t']:+.1f} | "
                       f"{s['fake']:.0%} | {s['sl']:.0%} |")
        out.append("")
    return out


def table_methods(rows, variants) -> list[str]:
    out = ["| 足 | 手法 | " + " | ".join(variants) + " |", "|---|---|" + "---:|" * len(variants)]
    for case in [c[0] for c in MAIN_CASES]:
        for m in TREND:
            cells = []
            for v in variants:
                s = stats(_pool(rows, case, v, "model", [m]))
                cells.append(f"{s['exp']:+.2f}(t {s['t']:+.1f}、{s['n']})" if s["n"] else "-")
            out.append(f"| {case} | {JA[m]} | " + " | ".join(cells) + " |")
    return out


def table_oneway(rows) -> list[str]:
    """5 の切り分け: 基準の取引を、一方向かどうか x 向き(買い = スワップを受け取る向き)x 入る前の伸び で分ける。"""
    out = ["| 足 | 区分 | 取引 | 勝率 | 期待値(pip) | 素(pip) | 帰無の素(pip) | 入る前 20 本の伸び(ATR) |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for case in [c[0] for c in MAIN_CASES]:
        tr = _pool(rows, case, "基準", "model")
        nu = _pool(rows, case, "基準", "null")
        groups = [
            ("一方向以外", lambda t: t["er"] <= ER),
            ("一方向 x 買い", lambda t: t["er"] > ER and t["side"] > 0),
            ("一方向 x 売り", lambda t: t["er"] > ER and t["side"] < 0),
            ("一方向 x 伸び 4 ATR 未満", lambda t: t["er"] > ER and abs(t["ext"]) < 4),
            ("一方向 x 伸び 4〜8 ATR", lambda t: t["er"] > ER and 4 <= abs(t["ext"]) < 8),
            ("一方向 x 伸び 8 ATR 以上", lambda t: t["er"] > ER and abs(t["ext"]) >= 8),
        ]
        for name, f in groups:
            sub = [t for t in tr if np.isfinite(t["er"]) and np.isfinite(t["ext"]) and f(t)]
            nsub = [t for t in nu if np.isfinite(t["er"]) and np.isfinite(t["ext"]) and f(t)]
            if len(sub) < 20:
                continue
            s = stats(sub)
            out.append(f"| {case} | {name} | {s['n']} | {s['win']:.0%} | {s['exp']:+.2f} | {s['gross']:+.2f} | "
                       f"{np.mean([t['gross'] for t in nsub]) if nsub else 0:+.2f} | "
                       f"{np.mean([t['side'] * t['ext'] for t in sub]):+.1f} |")
    return out


def table_week(weeks) -> list[str]:
    days = ["mon", "tue", "wed", "thu", "fri", "sat"]
    ja = {"mon": "月", "tue": "火", "wed": "水", "thu": "木", "fri": "金", "sat": "土(早朝)"}
    out = ["| 足 | " + " | ".join(ja[d] for d in days) + " |", "|---|" + "---:|" * len(days)]
    seen = set()
    for label, _m, w in weeks:
        if label in seen:
            continue
        seen.add(label)
        out.append(f"| {label} | " + " | ".join(f"{w[d][0]:.2f} / {w[d][1]:.0%}" if d in w else "-" for d in days)
                   + " |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=default_workers())
    a = ap.parse_args()
    rows, weeks = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for r, w in ex.map(job, [(*c, m) for c in MAIN_CASES for m in TREND]):
            rows += r
            weeks.append(w)
    lines = ["## 1. 対策ごとの成績(順張り 8 本の取引の合計)", ""] + table_variants(rows)
    lines += ["## 2. 手法ごとの期待値(pip / 取引、週の t、取引数)", ""]
    shown = ["基準", "1' ロールオーバーの前にたたむ", "1'+4+6", "1'+4+6+5b", "1'+5b(後から)"]
    lines += table_methods(rows, shown) + [""]
    lines += ["## 3. 一方向の局面で順張りが負ける理由(基準の取引)", ""] + table_oneway(rows) + [""]
    lines += ["## 4. 曜日ごとの値動きの大きさ(1 本の |リターン| の平均 ÷ 全体の平均 / 荒い足の割合)", ""]
    lines += table_week(weeks)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
