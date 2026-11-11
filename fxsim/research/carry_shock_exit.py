"""E029: スワップ狙い(B047 の形)に「急変ですぐ切る」を足すと、各通貨の急変から逃げられるか。

    python -m fxsim.research.carry_shock_exit

相場・口座は E026 と同じ(直近の実データ 2014-11〜2026-11 + ランダム 8 本、古い実データ 1996〜2014-10 +
古いランダム 16 本、10 万円、OANDA、約定は 2 時、swap_side_only)。
持つ期間は 1 週間以上なら短くなってよい(スワップは減る)。
土台は 2 つ: B047(金利差 2%・c2r 0.4 の下限)と、急変を浴びやすい「下限の緩い版」(金利差 1%・c2r なし)。
実データの急変の期間ごとに、口座の増減(期間の前の日の有効証拠金に対する割合)も出す。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.metrics import daily_equity
from fxsim.backtest.portfolio import build_backtester
from fxsim.catalog import *  # noqa: F403
from fxsim.markets import MARKETS
from fxsim.research.carry_breakout_opt import markets, metrics
from fxsim.strategies import STRATEGIES

B047 = {"n_in": 20, "n_out": 10, "min_carry": 0.02, "c2r_min": 0.4, "units": 2000, "margin_cap": 0.5}
LOOSE = {**B047, "min_carry": 0.01, "c2r_min": 0.0}
EXITS = [
    ("そのまま", {}),
    ("急変用の逆指値を近く(ATR x 4)", {"emergency_atr": 4.0}),
    ("終値の急変(1 日で ATR x 2.5 逆行)で切る", {"shock_atr": 2.5}),
    ("終値の急変で切る + 全ペア + 10 日休む", {"shock_atr": 2.5, "shock_all": True, "cooldown": 10}),
    ("5 日ボラが 250 日の 2 倍で切る + 10 日休む", {"vol_ratio": 2.0, "cooldown": 10}),
    ("急変・ボラの両方 + 全ペア + 10 日休む", {"shock_atr": 2.5, "vol_ratio": 2.0, "shock_all": True, "cooldown": 10}),
    ("出のチャネルを 5 日に", {"n_out": 5}),
]
VARIANTS = [(f"{bn}: {en}", {**base, **ex}) for bn, base in (("B047", B047), ("緩い版", LOOSE)) for en, ex in EXITS]
EVENTS = [("1998-10 円の急騰", "1998-10-05", "1998-10-09"), ("2008-09〜10 リーマン", "2008-09-15", "2008-10-27"),
          ("2011-08 フラン高", "2011-08-01", "2011-08-10"), ("2015-01 SNB", "2015-01-15", "2015-01-16"),
          ("2016-06 英国の離脱", "2016-06-23", "2016-06-27"), ("2020-03 コロナ", "2020-03-06", "2020-03-19"),
          ("2024-08 円キャリーの巻き戻し", "2024-07-25", "2024-08-06")]
NS_DAY = 86_400_000_000_000


def event_moves(bt) -> dict[str, float]:
    t, eq = daily_equity(bt)
    # 日足はサーバーの 0 時(UTC の前の日の 21〜22 時)に始まるので、2 時間足して取引日の日付にする
    d = (np.asarray(t).astype(np.int64) + 2 * 3_600_000_000_000).astype("datetime64[ns]").astype("datetime64[D]")
    out = {}
    for name, a, b in EVENTS:
        i0 = int(np.searchsorted(d, np.datetime64(a), "left")) - 1
        i1 = int(np.searchsorted(d, np.datetime64(b), "right")) - 1
        if 0 <= i0 < i1 < len(eq):
            out[name] = float(eq[i1] / eq[i0] - 1)
    return out


def job(spec):
    stage, name, model, params = spec
    params = dict(params)
    seed = params.pop("_seed", 1)
    md = MARKETS.create(model, params).generate(seed=seed)
    cfg = EngineConfig(initial_balance=100_000.0, min_units=1000.0, swap_side_only=True)
    out = []
    for k, (_, p) in enumerate(VARIANTS):
        s = STRATEGIES.create("main_carry_breakout", p)
        bt = build_backtester(md, s, BrokerConfig.preset("oanda"), cfg, seed=1)
        bt.run()
        years = (bt.bars.t[-1] - bt.bars.t[0]) / (365.25 * NS_DAY)
        days = [(t.exit_t - t.entry_t) / NS_DAY for t in bt.trades]
        worst = min((t.pnl for t in bt.trades), default=0.0)
        out.append({"stage": stage, "market": name, "k": k, **metrics(bt), "per_year": len(bt.trades) / years,
                    "hold_med": float(np.median(days)) if days else 0.0, "worst_trade": float(worst),
                    "events": event_moves(bt) if name.startswith("real") else {}, "blocked": dict(s.blocked)})
    return out


def main() -> None:
    with ProcessPoolExecutor(6) as ex:
        rows = [r for part in ex.map(job, markets()) for r in part]
    cols = ["設定", "直近の実データ シャープ / 年率 / 最大 DD", "ランダム 8 本 シャープ / 年率 / 最大 DD(中央・最悪)",
            "古い実データ シャープ / 年率 / 最大 DD", "古いランダム 16 本 シャープ / 年率 / 最大 DD(中央・最悪)",
            "取引 / 年・持つ日数の中央(直近 / 古い実データ)", "最悪の 1 取引(円、実データ 2 本)"]
    print("| " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * (len(cols) - 1))

    def rnd(x):
        med = {f: float(np.median([r[f] for r in x])) for f in ("sharpe", "cagr", "dd")}
        return f"{med['sharpe']:+.2f} / {med['cagr']:+.1%} / {med['dd']:.0%}・{max(r['dd'] for r in x):.0%}"
    for k, (label, _) in enumerate(VARIANTS):
        rs = [r for r in rows if r["k"] == k]
        ri = [r for r in rs if r["stage"] == "is" and not r["market"].startswith("real")]
        rq = [r for r in rs if r["stage"] == "oos" and not r["market"].startswith("real")]
        a = [r for r in rs if r["market"] == "real_recent"][0]
        b = [r for r in rs if r["market"] == "real_old"][0]
        print(f"| {label} | {a['sharpe']:+.2f} / {a['cagr']:+.1%} / {a['dd']:.0%} | {rnd(ri)} | "
              f"{b['sharpe']:+.2f} / {b['cagr']:+.1%} / {b['dd']:.0%} | {rnd(rq)} | "
              f"{a['per_year']:.1f}・{a['hold_med']:.0f} 日 / {b['per_year']:.1f}・{b['hold_med']:.0f} 日 | "
              f"{min(a['worst_trade'], b['worst_trade']):+,.0f} |")
    print("\n## 実データの急変の期間の口座の増減(期間の前の日の有効証拠金に対して)\n")
    print("| 設定 | " + " | ".join(e[0] for e in EVENTS) + " | 合計 |")
    print("|---|" + "---:|" * (len(EVENTS) + 1))
    for k, (label, _) in enumerate(VARIANTS):
        ev: dict[str, float] = {}
        for r in rows:
            if r["k"] == k and r["market"] in ("real_recent", "real_old"):
                ev.update(r["events"])
        cells = [f"{ev[e[0]]:+.1%}" if e[0] in ev else "-" for e in EVENTS]
        print(f"| {label} | " + " | ".join(cells) + f" | {sum(ev.values()):+.1%} |")


if __name__ == "__main__":
    main()
