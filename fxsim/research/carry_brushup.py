"""E031: 本命の 2 週版(B049)を磨く。急変で切る(B050)、急変で全部切って休む(B051)、2 週と 4 週を半分ずつ。

    python -m fxsim.research.carry_brushup

相場・口座・物差しは E026 / E029 と同じ(直近の実データ 2014-11〜2026-11 + ランダム 8 本、古い実データ 1996〜2014-10 +
古いランダム 16 本、10 万円、OANDA、約定は 2 時、swap_side_only)。全期間で測る(E030 のように最初の 3 年は除かない)。

「半分ずつ」は、口座を 5 万円ずつに分けた 2 本(2 週のチャネル・4 週のチャネル、どちらも 1 ペア 1,000 通貨、
証拠金の上限・口座の床はそれぞれの半分の口座で)の有効証拠金を足したもの。1 つの口座で同じペアを 2 本のチャネルで持つと
証拠金を共有するので少し違うが、採るかどうかを決めるにはこれで足りる(採るならエンジンと常駐に積み増しが要る)。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.portfolio import build_backtester
from fxsim.catalog import *  # noqa: F403
from fxsim.markets import MARKETS
from fxsim.research.carry_breakout_opt import markets, metrics
from fxsim.research.carry_shock_exit import EVENTS, NS_DAY, event_moves
from fxsim.strategies import STRATEGIES

BASE = {"min_carry": 0.02, "c2r_min": 0.4, "units": 2000, "margin_cap": 0.5}
W2 = {**BASE, "n_in": 10, "n_out": 5}
W4 = {**BASE, "n_in": 20, "n_out": 10}
SHOCK = {"shock_atr": 2.5}
SHOCK_ALL = {"shock_atr": 2.5, "shock_all": True, "cooldown": 10}
HALF = {"units": 1000}
# (名前, [(パラメータ, 口座)])。口座が 2 つなら、それぞれ 5 万円で回して有効証拠金を足す
VARIANTS = [
    ("B047: 4 週", [(W4, 100_000.0)]),
    ("B049: 2 週", [(W2, 100_000.0)]),
    ("B050: 2 週 + 急変で切る", [({**W2, **SHOCK}, 100_000.0)]),
    ("B051: 2 週 + 急変で全部切る・10 日休む", [({**W2, **SHOCK_ALL}, 100_000.0)]),
    ("2 週と 4 週を半分ずつ", [({**W2, **HALF}, 50_000.0), ({**W4, **HALF}, 50_000.0)]),
    ("2 週と 4 週を半分ずつ + 急変で全部切る・10 日休む",
     [({**W2, **HALF, **SHOCK_ALL}, 50_000.0), ({**W4, **HALF, **SHOCK_ALL}, 50_000.0)]),
]


def combine(bts: list):
    """複数の口座を足した、metrics / event_moves に渡せるもの。"""
    if len(bts) == 1:
        return bts[0]
    return SimpleNamespace(bars=bts[0].bars, equity_curve=sum(b.equity_curve for b in bts),
                           trades=[t for b in bts for t in b.trades],
                           stats={"losscut": sum(b.stats.get("losscut", 0) for b in bts)},
                           ruined=any(b.ruined for b in bts))


def job(spec):
    stage, name, model, params = spec
    params = dict(params)
    seed = params.pop("_seed", 1)
    md = MARKETS.create(model, params).generate(seed=seed)
    out = []
    for k, (_, legs) in enumerate(VARIANTS):
        bts = []
        for p, bal in legs:
            cfg = EngineConfig(initial_balance=bal, min_units=1000.0, swap_side_only=True)
            bt = build_backtester(md, STRATEGIES.create("main_carry_breakout", p), BrokerConfig.preset("oanda"), cfg,
                                  seed=1)
            bt.run()
            bts.append(bt)
        bt = combine(bts)
        years = (bt.bars.t[-1] - bt.bars.t[0]) / (365.25 * NS_DAY)
        days = [(t.exit_t - t.entry_t) / NS_DAY for t in bt.trades]
        pnl = sum(t.pnl for t in bt.trades)
        out.append({"stage": stage, "market": name, "k": k, **metrics(bt), "per_year": len(bt.trades) / years,
                    "hold_med": float(np.median(days)) if days else 0.0,
                    "worst_trade": float(min((t.pnl for t in bt.trades), default=0.0)),
                    "swap_share": float(sum(t.swap for t in bt.trades) / pnl) if pnl > 0 else float("nan"),
                    "events": event_moves(bt) if name.startswith("real") else {}})
    return out


def main() -> None:
    with ProcessPoolExecutor(6) as ex:
        rows = [r for part in ex.map(job, markets()) for r in part]
    cols = ["設定", "点数(直近の実データとランダムの半々)", "直近の実データ シャープ / 年率 / 最大 DD",
            "ランダム 8 本 シャープ / 年率 / 最大 DD(中央・最悪)", "古い実データ シャープ / 年率 / 最大 DD",
            "古いランダム 16 本 シャープ / 年率 / 最大 DD(中央・最悪)",
            "取引 / 年・持つ日数の中央・スワップの割合(直近 / 古い実データ)", "最悪の 1 取引(円、実データ 2 本)",
            "ロスカット・破綻"]
    print("| " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * (len(cols) - 1))

    def rnd(x):
        med = {f: float(np.median([r[f] for r in x])) for f in ("sharpe", "cagr", "dd")}
        return f"{med['sharpe']:+.2f} / {med['cagr']:+.1%} / {med['dd']:.0%}・{max(r['dd'] for r in x):.0%}"

    def share(r):
        return "-" if not np.isfinite(r["swap_share"]) else f"{r['swap_share']:.0%}"
    for k, (label, _) in enumerate(VARIANTS):
        rs = [r for r in rows if r["k"] == k]
        ri = [r for r in rs if r["stage"] == "is" and not r["market"].startswith("real")]
        rq = [r for r in rs if r["stage"] == "oos" and not r["market"].startswith("real")]
        a = [r for r in rs if r["market"] == "real_recent"][0]
        b = [r for r in rs if r["market"] == "real_old"][0]
        score = 0.5 * a["sharpe"] + 0.5 * float(np.median([r["sharpe"] for r in ri]))
        bad = sum(bool(r["losscut"] or r["ruined"]) for r in rs)
        print(f"| {label} | {score:+.2f} | {a['sharpe']:+.2f} / {a['cagr']:+.1%} / {a['dd']:.0%} | {rnd(ri)} | "
              f"{b['sharpe']:+.2f} / {b['cagr']:+.1%} / {b['dd']:.0%} | {rnd(rq)} | "
              f"{a['per_year']:.1f}・{a['hold_med']:.0f} 日・{share(a)} / "
              f"{b['per_year']:.1f}・{b['hold_med']:.0f} 日・{share(b)} | "
              f"{min(a['worst_trade'], b['worst_trade']):+,.0f} | {bad}/{len(rs)} |")
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
