"""E052: 4 時間足の押し目・荒い時期だけ(B068〜B070)を、1 つの口座で主要 10 ペアに回したときの成績。

    python -m fxsim.research.pullback_h4_account

本物のエンジン(MT5 のテスター、OANDA、1 時間足の実データ)で 1 ペアずつ 2010-11〜2026-11 を通しで回し(100 万円、
損切りで有効証拠金の 0.5% を失う量)、取引の損益を決済の時刻で足し合わせる。10 ペアの損益の合計は
「100 万円の口座 1 つで、10 ペアそれぞれに 1 回 0.5% の危険で入る」場合の近似(複利と証拠金の取り合いは見ない)。
危険を r 倍にすれば、損益もドローダウンも r 倍になる。
"""
# ruff: noqa: E501, E701, E702, I001
import fxsim.markets.models  # noqa: F401
import fxsim.strategies.builtin  # noqa: F401
import fxsim.strategies.main  # noqa: F401
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.portfolio import build_backtester
from fxsim.markets import MARKETS
from fxsim.strategies import STRATEGIES

PAIRS = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD", "EURJPY", "GBPJPY", "AUDJPY"]
CASES = {"B069 成行(区切り固定)": {"limit_atr": 0.0}, "B068 指値 0.5(区切り固定)": {"limit_atr": 0.5},
         "B074 成行・毎時判断": {"limit_atr": 0.0, "rolling": True},
         "B072 指値 0.5・毎時判断": {"limit_atr": 0.5, "rolling": True},
         "B073 指値 1.0・毎時判断": {"limit_atr": 1.0, "rolling": True},
         "B075 8 時間・指値 0.5・毎時判断": {"limit_atr": 0.5, "rolling": True, "signal_hours": 8}}
BASE = {"signal_hours": 4, "vol_ratio_min": 1.2, "max_days": 6}
EQUITY = 1e6


def run(a):
    pair, name = a
    md = MARKETS.create("real_pair", {"source": {"path": f"db:{pair}/H1", "start": "2010-11-01", "end": "2026-11-10"}}).generate(seed=1)
    s = STRATEGIES.create("main_pullback_limit", {**BASE, **CASES[name]})
    bt = build_backtester(md, s, BrokerConfig.preset("oanda"), EngineConfig(initial_balance=EQUITY), seed=1)
    bt.run()
    return [(t.exit_t / 1e9, t.pnl, t.entry_t / 1e9, t.swap, t.side) for t in bt.trades]


def main() -> None:
    jobs = [(p, n) for n in CASES for p in PAIRS]
    with ProcessPoolExecutor(10) as ex:
        outs = list(ex.map(run, jobs))
    for name in CASES:
        per = {p: np.array(o).reshape(-1, 5) for (p, n), o in zip(jobs, outs, strict=True) if n == name}
        a = np.vstack(list(per.values()))
        a = a[np.argsort(a[:, 0])]
        eq = np.cumsum(a[:, 1])
        dd = (np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max()
        mon = (a[:, 0] // (86400 * 30.4375)).astype(int)
        full = np.zeros(mon.max() - mon.min() + 1)
        np.add.at(full, mon - mon.min(), a[:, 1])
        sharpe = full.mean() / full.std(ddof=1) * np.sqrt(12)
        yrs = (a[:, 0].astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970)
        years = len(full) / 12
        held = (a[:, 0] - a[:, 2]) / 3600
        # 同時に持っている数の最大(決済と新規の時刻から)
        ev = sorted([(x, 1) for x in a[:, 2]] + [(x, -1) for x in a[:, 0]])
        cur = mx = 0
        for _, d in ev:
            cur += d
            mx = max(mx, cur)
        risk = 0.005 * EQUITY
        print(f"\n### {name}(1 回の危険 0.5% = {risk:,.0f} 円)\n")
        print(f"- 取引 {len(a)}(年 {len(a) / years:.0f} 回)、勝率 {(a[:, 1] > 0).mean():.0%}、持つ時間 平均 {held.mean():.1f}・最長 {held.max():.0f} 時間、"
              f"買い {int((a[:, 4] > 0).sum())}・売り {int((a[:, 4] < 0).sum())}、同時に持つ数の最大 {mx}")
        print(f"- 損益の合計 {eq[-1]:+,.0f} 円(年 {eq[-1] / years:+,.0f} 円 = 年率 {eq[-1] / years / EQUITY:+.2%})、うちスワップ {a[:, 3].sum():+,.0f} 円")
        print(f"- 1 取引の平均 {a[:, 1].mean():+,.0f} 円(危険の {a[:, 1].mean() / risk:+.3f} 倍)、最悪の 1 取引 {a[:, 1].min():+,.0f} 円")
        print(f"- 最大ドローダウン {dd:,.0f} 円({dd / EQUITY:.2%})、月ごとのシャープ(年率){sharpe:+.2f}、プラスの月 {(full > 0).mean():.0%}"
              f"(取引の無い月 {(full == 0).mean():.0%})")
        print(f"- 危険を 2% にした場合の目安: 年率 {eq[-1] / years / EQUITY * 4:+.1%}、最大ドローダウン {dd / EQUITY * 4:.1%}")
        ys = sorted(set(yrs))
        print("- 年ごと(円): " + ", ".join(f"{y} {a[yrs == y, 1].sum():+,.0f}" for y in ys)
              + f" → プラスの年 {sum(a[yrs == y, 1].sum() > 0 for y in ys)}/{len(ys)}")
        print("- ペアごと(円): " + ", ".join(f"{p} {v[:, 1].sum():+,.0f}" for p, v in per.items()))


if __name__ == "__main__":
    main()
