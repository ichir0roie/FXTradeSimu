"""E018: スワップを払う向きには新規で入らないゲートの効果(日足、OANDA、約定 2 時、10 ペア x 5 期間)。

    python -m fxsim.research.swap_gate

戦略と方向でたらめ版のそれぞれに、ゲートあり・なしを回す。ゲートは注文の窓口(ctx)を包み、
「向き x 金利差 − 業者の受け取り側の取り分」が 0 以下の新規を止める(逆の建玉があれば決済だけする)。
"""
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.metrics import compute_metrics
from fxsim.backtest.portfolio import build_backtester
from fxsim.cases.loader import load_cases
from fxsim.catalog import *  # noqa
from fxsim.data import load_rates
from fxsim.data.rates import MULTI_RATES
from fxsim.markets import MARKETS
from fxsim.pair_eval import PAIRS, RANGES
from fxsim.strategies import STRATEGIES

BEHAVIORS = ["B004", "B007", "B010", "B035", "B045"]
RECV = 0.0044          # OANDA の受け取る側の取り分


class _Gate:
    def __init__(self, ctx, ok):
        self._ctx, self._ok = ctx, ok

    def __getattr__(self, k):
        return getattr(self._ctx, k)

    def _order(self, side, name, *a, **kw):
        if self._ok(side):
            return getattr(self._ctx, name)(*a, **kw)
        if hasattr(self._ctx, "side") and not callable(self._ctx.side) and self._ctx.side == -side:
            self._ctx.close()
        return SimpleNamespace(ok=False)

    def buy(self, *a, **kw):
        return self._order(1, "buy", *a, **kw)

    def sell(self, *a, **kw):
        return self._order(-1, "sell", *a, **kw)


def gated(strategy):
    inner_prepare, inner_on_bar = strategy.prepare, strategy.on_bar

    def prepare(bars, rng):
        strategy._diff = load_rates(MULTI_RATES).pair_differential(bars.pair, bars.t)
        inner_prepare(bars, rng)

    def on_bar(i, ctx):
        inner_on_bar(i, _Gate(ctx, lambda side: side * strategy._diff[i] - RECV > 0))

    strategy.prepare, strategy.on_bar = prepare, on_bar
    return strategy


def job(arg):
    pair, (start, end) = arg
    book = load_cases()
    md = MARKETS.create("real_pair", {"source": {"path": f"db:{pair}/D1", "start": start, "end": end,
                                                 "daily_fill_hour": 2}}).generate(seed=1)
    out = []
    for bid in BEHAVIORS:
        b = book.behaviors[bid]
        broker = BrokerConfig.preset("oanda", (b.broker_spec.get("overrides") or {}))
        for kind in ("model", "null"):
            for gate in (False, True):
                if kind == "model":
                    s = STRATEGIES.create(b.strategy, b.strategy_params())
                else:
                    tf = b.strategy_params().get("timeframe")
                    s = STRATEGIES.create(b.null_strategy(), {"inner": b.strategy, "inner_params": dict(b.params),
                                                              **({"timeframe": int(tf)} if tf else {})})
                if gate:
                    s = gated(s)
                bt = build_backtester(md, s, broker, EngineConfig(**b.engine), seed=1)
                bt.run()
                m = compute_metrics(bt)
                out.append(dict(bid=bid, pair=pair, start=start, kind=kind, gate=gate, pnl=m["final_equity"] - 1e6,
                                swap=m["costs"]["swap"], gross=m["costs"]["gross_pnl"], trades=m["n_trades"]))
    return out


def _t(v) -> float:
    v = np.asarray(v, float)
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 1 and v.std() > 0 else 0.0


def report(rows: list[dict]) -> str:
    def key(r):
        return r["pair"], r["start"]

    lines = ["| 動作ケース | 版 | 平均(万円 / 6 年) | 中央値 | プラス | t | スワップの平均 | 素の損益の平均 "
             "| 最悪のパターン | 取引 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for bid in BEHAVIORS:
        d = {}
        for kind in ("model", "null"):
            for g in (False, True):
                rs = sorted([r for r in rows if r["bid"] == bid and r["kind"] == kind and r["gate"] == g], key=key)
                d[(kind, g)] = rs
                p = np.array([r["pnl"] for r in rs])
                name = ("戦略" if kind == "model" else "方向でたらめ版") + (" + ゲート" if g else "")
                lines.append(f"| {bid} | {name} | {p.mean() / 1e4:+.1f} | {np.median(p) / 1e4:+.1f} | "
                             f"{(p > 0).mean():.0%} | {_t(p):+.2f} | {np.mean([r['swap'] for r in rs]) / 1e4:+.1f} | "
                             f"{np.mean([r['gross'] for r in rs]) / 1e4:+.1f} | {p.min() / 1e4:+.1f} | "
                             f"{np.mean([r['trades'] for r in rs]):.0f} |")
        ge = [a["pnl"] - b["pnl"] for a, b in zip(d[("model", True)], d[("model", False)], strict=True)]
        gn = [a["pnl"] - b["pnl"] for a, b in zip(d[("null", True)], d[("null", False)], strict=True)]
        vs = [a["pnl"] - b["pnl"] for a, b in zip(d[("model", True)], d[("null", True)], strict=True)]
        lines.append(f"| {bid} | ゲートの効果 | 戦略 {np.mean(ge) / 1e4:+.1f}(t {_t(ge):+.2f}) / "
                     f"帰無 {np.mean(gn) / 1e4:+.1f}(t {_t(gn):+.2f}) / "
                     f"ゲート付きの戦略 − ゲート付きの帰無 {np.mean(vs) / 1e4:+.1f}(t {_t(vs):+.2f}) | | | | | | | | |")
    return "\n".join(lines)


def main() -> None:
    args = [(p, w) for p in PAIRS for w in RANGES["D1"].windows]
    rows = []
    with ProcessPoolExecutor(6) as ex:
        for r in ex.map(job, args):
            rows += r
    print(report(rows))


if __name__ == "__main__":
    main()
