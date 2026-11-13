"""E033: 短期の基本のテクニカル手法は、いつ負けるのか(USDJPY の実データ、時間足ごとに 1 本、OANDA)。

    python -m fxsim.research.loss_timing [--workers N]

手法は既定のパラメータのまま timeframe だけを変え、同じ時刻に入って出る方向でたらめ版と対で回す。
約定を足の中まで見るため、時間足ごとに相場の DB の一番細かい足を使う(1 分 = M1、5 分・15 分 = M5、1 時間・4 時間 = H1)。
1 時間・4 時間は 5 分足と同じ期間でも回す(足の違いと期間の違いを分ける参考)。
スワップを受け取る向きだけの決まり(E022)は掛けない(短期の手法の性質を見るため。掛けると USDJPY は買いだけになる)。

取引ごとに、入る判断をした確定足の局面を付ける(どれも因果的。閾値は回す前に提案書に書いたもの):
    時間帯   約定時刻(日本時間): 東京朝 7〜10 / 東京 10〜15 / 欧州 15〜21 / NY 21〜翌 2 / 早朝 2〜7
    曜日     日本時間: 月曜 / 火〜木 / 金曜(土曜の早朝は金曜)
    ボラ     ATR(14) / 直前 500 本の ATR の中央値: 静か < 0.8 / 中 / 荒い > 1.25
    一方向   20 本の効率比: もみ合い < 0.2 / 中 / 一方向 > 0.4
    流れ     60 本の対数リターン / (20 本のボラ x √60) を建てた向きで: 逆行 < −0.5 / 中立 / 順行 > +0.5
    持つ長さ 戦略の足で 3 本以下 / 4〜20 本 / 21 本以上
    決済     signal / reverse / sl / その他
勝ち = 円の損益(スワップ・費用込み)> 0。利益率・損失率は建玉の額(数量 x 約定価格)に対する %。
t は週ごとの損益の合計で数える(同じ週の取引は独立でない。E001)。
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
from fxsim.catalog import *  # noqa: F403
from fxsim.experiments.config import default_workers
from fxsim.markets import MARKETS
from fxsim.strategies import STRATEGIES

PAIR = "USDJPY"
TREND = ["ma_cross", "macd_cross", "perfect_order", "adx_trend", "ichimoku", "parabolic_sar", "heikin_ashi",
         "donchian_breakout"]
REVERSION = ["rsi_reversion", "bollinger_reversion", "stochastic_reversion"]
METHODS = TREND + REVERSION
JA = {"ma_cross": "MA クロス", "macd_cross": "MACD クロス", "perfect_order": "パーフェクトオーダー",
      "adx_trend": "ADX", "ichimoku": "一目均衡表", "parabolic_sar": "パラボリック SAR", "heikin_ashi": "平均足",
      "donchian_breakout": "ブレイクアウト", "rsi_reversion": "RSI 逆張り", "bollinger_reversion": "ボリンジャー逆張り",
      "stochastic_reversion": "ストキャス逆張り"}
# (ラベル, 戦略の足(分), 相場の DB の足, 開始, 終了)
CASES = [
    ("1 分", 1, "M1", "2026-08-05", "2026-11-11"),
    ("5 分", 5, "M5", "2025-07-09", "2026-11-11"),
    ("15 分", 15, "M5", "2025-07-09", "2026-11-11"),
    ("1 時間", 60, "H1", "2010-10-06", "2026-11-11"),
    ("4 時間", 240, "H1", "2010-10-06", "2026-11-11"),
    ("1 時間(5 分足と同じ期間)", 60, "H1", "2025-07-14", "2026-11-10"),
    ("4 時間(5 分足と同じ期間)", 240, "H1", "2025-07-14", "2026-11-10"),
]
MAIN = [c[0] for c in CASES[:5]]
FEATURES = {
    "時間帯": ("東京朝", "東京", "欧州", "NY", "早朝"),
    "曜日": ("月曜", "火〜木", "金曜"),
    "ボラ": ("静か", "中", "荒い"),
    "一方向": ("もみ合い", "中", "一方向"),
    "流れ": ("逆行", "中立", "順行"),
    "持つ長さ": ("〜3 本", "4〜20 本", "21 本〜"),
    "決済": ("signal", "reverse", "sl", "その他"),
}
NS_MIN = 60_000_000_000
JST = 9 * 60 * NS_MIN


def bar_features(bars):
    c, h, lo = bars.close, bars.high, bars.low
    n = len(c)
    lc = np.log(c)
    r = np.diff(lc, prepend=lc[0])
    prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev), np.abs(lo - prev)))
    atr = np.full(n, np.nan)
    a = tr[:14].mean()
    for i in range(14, n):
        a = a + (tr[i] - a) / 14
        atr[i] = a
    vr = np.full(n, np.nan)
    for i in range(514, n):
        vr[i] = atr[i] / np.median(atr[i - 500:i])
    cs = np.cumsum(np.abs(r))
    er = np.full(n, np.nan)
    er[20:] = np.abs(lc[20:] - lc[:-20]) / np.maximum(cs[20:] - cs[:-20], 1e-12)
    vol20 = np.full(n, np.nan)
    for i in range(20, n):
        vol20[i] = r[i - 19:i + 1].std()
    trend = np.full(n, np.nan)
    trend[60:] = (lc[60:] - lc[:-60]) / np.maximum(vol20[60:] * np.sqrt(60), 1e-12)
    return vr, er, trend


def _cut(v, lo, hi, names):
    if not np.isfinite(v):
        return None
    return names[0] if v < lo else names[2] if v > hi else names[1]


def session(hour: int) -> str:
    if 7 <= hour < 10:
        return "東京朝"
    if 10 <= hour < 15:
        return "東京"
    if 15 <= hour < 21:
        return "欧州"
    if hour >= 21 or hour < 2:
        return "NY"
    return "早朝"


def tag(t, k: int, tf: int, f) -> dict:
    vr, er, trend = f
    jst = t.entry_t + JST
    hour = int((jst // (60 * NS_MIN)) % 24)
    dow = int(((jst // (1440 * NS_MIN)) + 3) % 7)          # 1970-01-01 は木曜。0 = 月曜
    held = (t.exit_t - t.entry_t) / (tf * NS_MIN)
    return {
        "時間帯": session(hour),
        "曜日": "月曜" if dow == 0 else "金曜" if dow in (4, 5) else "火〜木" if dow in (1, 2, 3) else None,
        "ボラ": _cut(vr[k], 0.8, 1.25, FEATURES["ボラ"]),
        "一方向": _cut(er[k], 0.2, 0.4, FEATURES["一方向"]),
        "流れ": _cut(trend[k] * t.side, -0.5, 0.5, FEATURES["流れ"]),
        "持つ長さ": "〜3 本" if held <= 3 else "4〜20 本" if held <= 20 else "21 本〜",
        "決済": t.reason if t.reason in ("signal", "reverse", "sl") else "その他",
    }


def job(arg):
    label, tf, data_tf, start, end, method = arg
    with contextlib.suppress(OSError):
        os.nice(10)
    md = MARKETS.create("real_pair", {"source": {"path": f"db:{PAIR}/{data_tf}", "start": start,
                                                 "end": end}}).generate(seed=1)
    bars = md.bars(tf)
    f = bar_features(bars)
    out = []
    for kind in ("model", "null"):
        if kind == "model":
            s = STRATEGIES.create(method, {"timeframe": tf})
        else:
            s = STRATEGIES.create("direction_null", {"inner": method, "inner_params": {}, "timeframe": tf})
        broker = BrokerConfig.preset("oanda", {"mt5": {"preset": "jp_domestic"}})
        bt = build_backtester(md, s, broker, EngineConfig(units=10_000), seed=1)
        bt.run()
        trades = []
        for t in bt.trades:
            k = int(np.searchsorted(bars.t, t.entry_t, "right")) - 2      # 入る判断をした確定足
            if k < 0 or t.reason == "end":
                continue
            trades.append({"pnl": float(t.pnl), "pct": float(t.pnl / (abs(t.units) * t.entry_price) * 100),
                           "pips": float(t.pips), "gross": float(t.gross_pips), "cost": float(t.cost_pips),
                           "week": int((t.entry_t + JST) // (7 * 1440 * NS_MIN)),
                           "hour": int(((t.entry_t + JST) // (60 * NS_MIN)) % 24), **tag(t, k, tf, f)})
        out.append({"case": label, "method": method, "kind": kind, "trades": trades})
    return out


def _week_t(trades) -> float:
    if len(trades) < 3:
        return 0.0
    weeks: dict[int, float] = {}
    for t in trades:
        weeks[t["week"]] = weeks.get(t["week"], 0.0) + t["pnl"]
    v = np.array(list(weeks.values()))
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 2 and v.std() > 0 else 0.0


def _max_streak(trades) -> int:
    best = cur = 0
    for t in trades:
        cur = cur + 1 if t["pnl"] <= 0 else 0
        best = max(best, cur)
    return best


def stats(trades) -> dict:
    if not trades:
        return {"n": 0}
    pnl = np.array([t["pnl"] for t in trades])
    pct = np.array([t["pct"] for t in trades])
    pips = np.array([t["pips"] for t in trades])
    gross = np.array([t["gross"] for t in trades])
    cost = np.array([t["cost"] for t in trades])
    win = pnl > 0
    lose = ~win
    gw, gl = pnl[win].sum(), -pnl[lose].sum()
    return {
        "n": len(trades), "win": win.mean(),
        "win_pct": pct[win].mean() if win.any() else 0.0, "loss_pct": pct[lose].mean() if lose.any() else 0.0,
        "win_pip": pips[win].mean() if win.any() else 0.0, "loss_pip": pips[lose].mean() if lose.any() else 0.0,
        "ratio": (pips[win].mean() / -pips[lose].mean()) if win.any() and lose.any() and pips[lose].mean() < 0
        else float("nan"),
        "pf": gw / gl if gl > 0 else float("inf"),
        "exp_pip": pips.mean(), "exp_pct": pct.mean(), "gross": gross.mean(), "cost": cost.mean(),
        "cost_loss": float(((gross > 0) & lose).sum() / max(lose.sum(), 1)),
        "be_win": float(-pips[lose].mean() / (pips[win].mean() - pips[lose].mean()))
        if win.any() and lose.any() else float("nan"),
        "t": _week_t(trades), "streak": _max_streak(trades),
    }


def _get(rows, case, method, kind):
    return next(r["trades"] for r in rows if r["case"] == case and r["method"] == method and r["kind"] == kind)


def table_methods(rows, case: str) -> list[str]:
    lines = [f"### {case}", "",
             "| 手法 | 取引 | 勝率 | 平均利益率 | 平均損失率 | 平均利益(pip) | 平均損失(pip) | 損益比 | "
             "損益分岐の勝率 | 期待値(pip) | 素(pip) | 費用(pip) | 帰無の期待値(pip) | 週の t | 費用負け | 最大連敗 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        s = stats(_get(rows, case, m, "model"))
        if s["n"] == 0:
            lines.append(f"| {JA[m]} | 0 | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")
            continue
        nu = stats(_get(rows, case, m, "null"))
        lines.append(
            f"| {JA[m]} | {s['n']} | {s['win']:.0%} | {s['win_pct']:+.3f}% | {s['loss_pct']:+.3f}% | "
            f"{s['win_pip']:+.1f} | {s['loss_pip']:+.1f} | {s['ratio']:.2f} | {s['be_win']:.0%} | "
            f"{s['exp_pip']:+.2f} | {s['gross']:+.2f} | {s['cost']:.2f} | "
            f"{nu.get('exp_pip', 0):+.2f} | {s['t']:+.1f} | {s['cost_loss']:.0%} | {s['streak']} |")
    return lines


def table_cases(rows) -> list[str]:
    lines = ["| 足 | 家族 | 取引 | 勝率 | 平均利益率 | 平均損失率 | 損益比 | 損益分岐の勝率 | 期待値(pip) | 素(pip) | "
             "費用(pip) | 費用 / 1 取引の値幅 | 帰無の期待値(pip) | 費用負け |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for case in [c[0] for c in CASES]:
        for fam, ms in (("順張り", TREND), ("逆張り", REVERSION)):
            tr = [t for m in ms for t in _get(rows, case, m, "model")]
            nu = [t for m in ms for t in _get(rows, case, m, "null")]
            s, sn = stats(tr), stats(nu)
            if s["n"] == 0:
                continue
            width = np.mean([abs(t["gross"]) for t in tr])
            lines.append(f"| {case} | {fam} | {s['n']} | {s['win']:.0%} | {s['win_pct']:+.3f}% | "
                         f"{s['loss_pct']:+.3f}% | {s['ratio']:.2f} | {s['be_win']:.0%} | {s['exp_pip']:+.2f} | "
                         f"{s['gross']:+.2f} | "
                         f"{s['cost']:.2f} | {s['cost'] / width:.0%} | {sn['exp_pip']:+.2f} | {s['cost_loss']:.0%} |")
    return lines


def table_feature(rows, feat: str, by_method: bool = False) -> list[str]:
    cats = FEATURES[feat]
    head = "| 足 | " + ("手法" if by_method else "家族") + " | " + " | ".join(cats) + " |"
    lines = [head, "|---|---|" + "---:|" * len(cats)]
    groups = [(JA[m], [m]) for m in METHODS] if by_method else [("順張り", TREND), ("逆張り", REVERSION)]
    for case in MAIN:
        for name, ms in groups:
            tr = [t for m in ms for t in _get(rows, case, m, "model")]
            nu = [t for m in ms for t in _get(rows, case, m, "null")]
            if not tr:
                continue
            loss_total = -sum(t["pnl"] for t in tr if t["pnl"] < 0)
            cells = []
            for c in cats:
                sub = [t for t in tr if t[feat] == c]
                nsub = [t for t in nu if t[feat] == c]
                if len(sub) < 20:
                    cells.append(f"-({len(sub)})")
                    continue
                win = np.mean([t["pnl"] > 0 for t in sub])
                exp = np.mean([t["pips"] for t in sub])
                nexp = np.mean([t["pips"] for t in nsub]) if nsub else 0.0
                share_loss = -sum(t["pnl"] for t in sub if t["pnl"] < 0) / max(loss_total, 1e-9)
                share_n = len(sub) / len(tr)
                cells.append(f"{win:.0%} / {exp:+.1f}(帰無 {nexp:+.1f})/ 負け {share_loss / share_n:.2f}x")
            lines.append(f"| {case} | {name} | " + " | ".join(cells) + " |")
    return lines


ROLLOVER = range(5, 9)      # 日本時間 5〜8 時台(NY の引けのスプレッド拡大。夏は 6 時、冬は 7 時が山)


def table_rollover(rows) -> list[str]:
    """結果を見てから足した切り口(E033): ロールオーバー前後に入った取引を除くと、負けは消えるか。"""
    lines = ["| 足 | 家族 | 5〜8 時台の取引の割合 | その費用(pip) | ほかの時間の費用(pip) | 期待値(pip、全部) | "
             "期待値(pip、5〜8 時台を除く) | 素(pip、除く) | 勝率(除く) | 損益分岐の勝率(除く) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for case in MAIN:
        for fam, ms in (("順張り", TREND), ("逆張り", REVERSION)):
            tr = [t for m in ms for t in _get(rows, case, m, "model")]
            ro = [t for t in tr if t["hour"] in ROLLOVER]
            ok = [t for t in tr if t["hour"] not in ROLLOVER]
            s, so = stats(tr), stats(ok)
            lines.append(f"| {case} | {fam} | {len(ro) / len(tr):.0%} | "
                         f"{np.mean([t['cost'] for t in ro]) if ro else 0:.2f} | {so['cost']:.2f} | "
                         f"{s['exp_pip']:+.2f} | {so['exp_pip']:+.2f} | {so['gross']:+.2f} | {so['win']:.0%} | "
                         f"{so['be_win']:.0%} |")
    return lines


def report(rows) -> str:
    out = ["## 1. 時間足ごとのまとめ(手法を順張り 8 本・逆張り 3 本にまとめた取引の合計)", ""]
    out += table_cases(rows)
    out += ["", "## 1b. ロールオーバー前後(日本時間 5〜8 時台)に入った取引を除くと(結果を見てから足した切り口)", ""]
    out += table_rollover(rows)
    out += ["", "## 2. 手法 x 時間足の成績", ""]
    for case in [c[0] for c in CASES]:
        out += table_methods(rows, case) + [""]
    out += ["## 3. 負けの集まり方(勝率 / 期待値 pip(同じ区分の方向でたらめ版)/ 負け額の割合 ÷ 取引の割合)", "",
            "負け x は、その区分の負け額の割合を取引の割合で割ったもの。1 より大きいほど、その区分に負けが集まる。"
            "20 取引未満の区分は -(取引数)。", ""]
    for feat in FEATURES:
        out += [f"### {feat}", ""] + table_feature(rows, feat) + [""]
    out += ["## 4. 時間帯(手法ごと)", ""] + table_feature(rows, "時間帯", by_method=True) + [""]
    out += ["## 5. 一方向(手法ごと)", ""] + table_feature(rows, "一方向", by_method=True) + [""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=default_workers())
    a = ap.parse_args()
    args = [(*c, m) for c in CASES for m in METHODS]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(job, args):
            rows += r
    print(report(rows))


if __name__ == "__main__":
    main()
