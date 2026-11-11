"""E024: キャリーを本体にしたときの、手仕舞いの決め方と「値が安定したペア」の選び方。

    python -m fxsim.research.carry_exits

入るのはいつでもよい(スワップを受け取れる向きで、業者の取り分を引いた受け取りが年 1% 以上のとき)。
手仕舞いの決め方を変えて比べる:
    持ち続ける           金利差が 1% を割るか向きが変わるまで持つ
    合計 +1% / +3% で利確 値動きの損益 + たまったスワップ が建玉の +1% / +3% に届いたら手仕舞い、
                         翌日また入る
    沈んだ後 0 に戻ったら  合計が一度 −2% 以下に沈み、その後 0 以上に戻ったら手仕舞い
                         (損失をスワップと戻りで埋めた時点)、翌日また入る
    損切り −10%          合計が −10% を割ったら手仕舞い、次の月まで入らない
    キャリー・トゥ・リスク 受け取りの年率 ÷ 60 日のボラ(年率)が 0.3 以上のときだけ持つ
                         (値の振れに比べて金利差が大きいペア・時期)
建玉は 1 ペア 100 万円分(レバレッジ 1 倍)、量は入るときに決めて変えない。相場とブローカーは E017 と同じ(real_pair の日足
10 ペア x 5 期間、OANDA、約定は 2 時、swap_side_only)。損益は建玉 100 万円に対する年率。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.metrics import compute_metrics, daily_equity
from fxsim.backtest.portfolio import build_backtester
from fxsim.catalog import *  # noqa: F403
from fxsim.data import load_rates
from fxsim.data.rates import MULTI_RATES
from fxsim.markets import MARKETS
from fxsim.pair_eval import PAIRS, RANGES

RECV = 0.0044
NOTIONAL = 1_000_000.0


@dataclass
class Rule:
    name: str
    tp: float | None = None          # 合計 / 建玉 がこれ以上で手仕舞い
    sink: float | None = None        # 一度これ以下に沈み、その後 0 以上に戻ったら手仕舞い
    stop: float | None = None        # 合計 / 建玉 がこれ以下で手仕舞い、次の月まで入らない
    c2r: float | None = None         # 受け取りの年率 / ボラ(年率)がこれ以上のときだけ持つ


RULES = [Rule("持ち続ける"), Rule("合計 +1% で利確", tp=0.01), Rule("合計 +3% で利確", tp=0.03),
         Rule("沈んだ後 0 に戻ったら手仕舞い", sink=-0.02), Rule("損切り −10%", stop=-0.10),
         Rule("キャリー・トゥ・リスク 0.3 以上", c2r=0.3),
         Rule("キャリー・トゥ・リスク 0.3 以上 + 損切り −10%", c2r=0.3, stop=-0.10)]


class CarryHold:
    """受け取れる向きを持ち、手仕舞いを Rule で決める(研究用。登録しない)。"""
    multi = False
    timeframe = 1440

    def __init__(self, rule: Rule, min_carry: float = 0.01):
        self.rule, self.min_carry = rule, min_carry

    def prepare(self, bars, rng):
        diff = load_rates(MULTI_RATES).pair_differential(bars.pair, bars.t)
        self.side = np.sign(diff).astype(int)
        self.carry = np.abs(diff) - RECV
        r = np.diff(np.log(bars.close), prepend=np.log(bars.close[0]))
        vol = np.full(len(r), np.nan)
        for i in range(60, len(r)):
            vol[i] = r[i - 59:i + 1].std() * np.sqrt(260)
        self.c2r = self.carry / vol
        self.close = bars.close
        self.t = bars.t
        self.warmup = 60
        self.sunk = False
        self.block_month = None
        self.exits = 0

    def on_bar(self, i, ctx):
        p = ctx.position
        want = int(self.side[i]) if self.carry[i] >= self.min_carry else 0
        if self.rule.c2r is not None and not (np.isfinite(self.c2r[i]) and self.c2r[i] >= self.rule.c2r):
            want = 0
        month = int(np.datetime64(int(self.t[i]), "ns").astype("datetime64[M]").astype(np.int64))
        if p is not None:
            total = ((self.close[i] - p.entry_price) * p.side * p.units + p.swap) / NOTIONAL
            if total <= (self.rule.sink if self.rule.sink is not None else -np.inf):
                self.sunk = True
            out = p.side != want
            if self.rule.tp is not None and total >= self.rule.tp:
                out = True
            if self.rule.sink is not None and self.sunk and total >= 0:
                out = True
            if self.rule.stop is not None and total <= self.rule.stop:
                out, self.block_month = True, month
            if out:
                ctx.close()
                self.sunk = False
                self.exits += 1
            return
        if want and self.block_month != month:
            units = np.floor(NOTIONAL / self.close[i] / 1000) * 1000
            (ctx.buy if want > 0 else ctx.sell)(units=units)


def job(arg):
    pair, (start, end) = arg
    md = MARKETS.create("real_pair", {"source": {"path": f"db:{pair}/D1", "start": start, "end": end,
                                                 "daily_fill_hour": 2}}).generate(seed=1)
    broker = BrokerConfig.preset("oanda")
    out = []
    for rule in RULES:
        s = CarryHold(rule)
        bt = build_backtester(md, s, broker, EngineConfig(swap_side_only=True, max_units=1e8), seed=1)
        bt.run()
        m = compute_metrics(bt)
        _, eq = daily_equity(bt)
        pnl = eq - 1e6
        peak = np.maximum.accumulate(np.r_[0.0, pnl])[1:]
        dd = float(((peak - pnl) / NOTIONAL).max()) if len(pnl) else 0.0
        held = sum(t.bars_held for t in bt.trades)
        out.append({"rule": rule.name, "pair": pair, "start": start, "ret": (eq[-1] - 1e6) / NOTIONAL / m["years"],
                    "swap": m["costs"]["swap"] / NOTIONAL / m["years"], "dd": dd, "trades": m["n_trades"],
                    "years": m["years"], "held": held / max(len(md.bars(1440)), 1),
                    "hold_days": held / max(m["n_trades"], 1)})
    return out


def _t(v) -> float:
    v = np.asarray(v, float)
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 2 and v.std() > 0 else 0.0


def main() -> None:
    args = [(p, w) for p in PAIRS for w in RANGES["D1"].windows]
    rows = []
    with ProcessPoolExecutor(6) as ex:
        for r in ex.map(job, args):
            rows += r
    print("| 手仕舞いの決め方 | 年率(建玉比、平均) | 中央値 | プラス | t | うちスワップ | 値動き "
          "| 最大 DD(建玉比、中央 / 最悪) "
          "| 持っていた割合 | 1 回に持つ日数 | 取引 / 6 年 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rule in RULES:
        rs = [r for r in rows if r["rule"] == rule.name]
        ret = np.array([r["ret"] for r in rs])
        sw = np.array([r["swap"] for r in rs])
        dd = np.array([r["dd"] for r in rs])
        print(f"| {rule.name} | {ret.mean():+.2%} | {np.median(ret):+.2%} | {(ret > 0).mean():.0%} | {_t(ret):+.2f} | "
              f"{sw.mean():+.2%} | {(ret - sw).mean():+.2%} | {np.median(dd):.1%} / {dd.max():.1%} | "
              f"{np.mean([r['held'] for r in rs]):.0%} | {np.mean([r['hold_days'] for r in rs]):.0f} | "
              f"{np.mean([r['trades'] for r in rs]):.1f} |")
    print("\n持ち続ける版の、ペアごとの年率(5 期間の平均)と最大 DD(最悪の期間):")
    base = [r for r in rows if r["rule"] == RULES[0].name]
    for p in PAIRS:
        rs = [r for r in base if r["pair"] == p]
        print(f"  {p}: 年率 {np.mean([r['ret'] for r in rs]):+.2%}、最大 DD {max(r['dd'] for r in rs):.1%}、"
              f"持っていた割合 {np.mean([r['held'] for r in rs]):.0%}")


if __name__ == "__main__":
    main()
