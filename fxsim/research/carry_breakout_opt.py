"""E026: 本命 3 本目(main_carry_breakout、B047)のしきい値を、直近の実データとランダムな相場を半々に見て選ぶ。

    python -m fxsim.research.carry_breakout_opt

選ぶ段(点数 = 0.5 x 直近の実データ + 0.5 x ランダムな相場 8 本の中央値。成績は口座の日々の増減のシャープ、年率):
    直近の実データ    real_pairs 2014-11〜2026-11(12 年)
    ランダムな相場    real_pairs_bootstrap(同じ 12 年の時刻の並びに、1993〜2026 年の日付を 250 本のブロックでつなぐ。
                      値動きと金利差は日付と一緒、スプレッドは 2014〜2026 年の記録)乱数 1〜8
確かめる段(選ぶのに使わない):
    古い実データ      real_pairs 1996-01〜2014-10
    古いランダム      real_pairs_bootstrap の元を 1993〜2014-10 に限ったもの、乱数 101〜116
81 通り全部を両方の段で回し、選ぶ段の順位と確かめる段の順位がそろうか(順位の相関)を見る。
口座は 10 万円、1 ペア 1,000 通貨、証拠金の上限 50%、OANDA、約定は 2 時、swap_side_only。
"""
from __future__ import annotations

import itertools
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.metrics import daily_equity
from fxsim.backtest.portfolio import build_backtester
from fxsim.catalog import *  # noqa: F403
from fxsim.markets import MARKETS
from fxsim.strategies import STRATEGIES

RECENT = ("2014-11-01", "2026-11-10")
OLD = ("1996-01-01", "2014-10-31")
CHANNELS = [(20, 10), (40, 10), (40, 20), (55, 10), (55, 20), (55, 40), (80, 10), (80, 20), (80, 40)]
GRID = [{"n_in": a, "n_out": b, "min_carry": mc, "c2r_min": cr}
        for (a, b), mc, cr in itertools.product(CHANNELS, (0.005, 0.01, 0.02), (0.0, 0.2, 0.4))]
DEFAULT = {"n_in": 55, "n_out": 20, "min_carry": 0.01, "c2r_min": 0.2}
IS_SEEDS = list(range(1, 9))
OOS_SEEDS = list(range(101, 117))


def markets() -> list[tuple[str, str, str, dict]]:
    """(段, 名前, 市場モデル, パラメータ)。"""
    src = {"start": RECENT[0], "end": RECENT[1], "daily_fill_hour": 2}
    out = [("is", "real_recent", "real_pairs", {"source": src})]
    out += [("is", f"rand_{s}", "real_pairs_bootstrap", {"source": {**src, "block_days": 250}, "_seed": s})
            for s in IS_SEEDS]
    out += [("oos", "real_old", "real_pairs", {"source": {"start": OLD[0], "end": OLD[1], "daily_fill_hour": 2}})]
    out += [("oos", f"rand_old_{s}", "real_pairs_bootstrap",
             {"source": {**src, "block_days": 250, "pool_end": OLD[1]}, "_seed": s}) for s in OOS_SEEDS]
    return out


def metrics(bt) -> dict:
    _, eq = daily_equity(bt)
    r = np.diff(eq) / eq[:-1]
    r = r[np.isfinite(r)]
    years = (bt.bars.t[-1] - bt.bars.t[0]) / (365.25 * 86_400_000_000_000)
    peak = np.maximum.accumulate(eq)
    sharpe = float(r.mean() / r.std() * np.sqrt(260)) if len(r) > 2 and r.std() > 0 else 0.0
    return {"sharpe": sharpe, "cagr": float((max(eq[-1], 1.0) / eq[0]) ** (1 / years) - 1),
            "dd": float(((peak - eq) / peak).max()), "losscut": int(bt.stats.get("losscut", 0)),
            "ruined": bool(bt.ruined), "trades": len(bt.trades), "final": float(eq[-1])}


def job(spec):
    stage, name, model, params = spec
    params = dict(params)
    seed = params.pop("_seed", 1)
    md = MARKETS.create(model, params).generate(seed=seed)
    cfg = EngineConfig(initial_balance=100_000.0, min_units=1000.0, swap_side_only=True)
    broker = BrokerConfig.preset("oanda")
    rows = []
    for k, g in enumerate(GRID + [DEFAULT]):
        s = STRATEGIES.create("main_carry_breakout", g)
        bt = build_backtester(md, s, broker, cfg, seed=1)
        bt.run()
        rows.append({"stage": stage, "market": name, "k": k, **metrics(bt)})
    return rows


def _rank(v):
    order = np.argsort(np.argsort(v))
    return order.astype(float)


def report(rows: list[dict]) -> str:
    n = len(GRID) + 1

    def score(stage: str, k: int) -> dict:
        rs = [r for r in rows if r["stage"] == stage and r["k"] == k]
        real = [r for r in rs if r["market"].startswith("real")][0]
        rand = [r for r in rs if r["market"].startswith("rand")]
        sh = np.array([r["sharpe"] for r in rand])
        return {"score": 0.5 * real["sharpe"] + 0.5 * float(np.median(sh)), "real": real,
                "rand_sharpe": float(np.median(sh)), "rand_cagr": float(np.median([r["cagr"] for r in rand])),
                "rand_dd": float(np.median([r["dd"] for r in rand])), "rand_dd_max": float(max(r["dd"] for r in rand)),
                "rand_pos": float(np.mean([r["cagr"] > 0 for r in rand])),
                "rand_losscut": sum(r["losscut"] > 0 for r in rand), "n_rand": len(rand)}

    is_s = [score("is", k) for k in range(n)]
    oos_s = [score("oos", k) for k in range(n)]
    order = sorted(range(len(GRID)), key=lambda k: -is_s[k]["score"])

    def label(k):
        g = (GRID + [DEFAULT])[k]
        return (f"{g['n_in']} / {g['n_out']} 日、金利差 {g['min_carry']:.1%}、"
                f"c2r {g['c2r_min']:g}" + ("(既定)" if k == len(GRID) else ""))

    lines = ["## 選ぶ段の上位 10 と既定(B047)。右半分は確かめる段", "",
             "| 設定 | 点数 | 直近の実データ シャープ / 年率 / 最大 DD "
             "| ランダム 8 本 シャープ / 年率 / 最大 DD(中央・最悪) "
             "/ プラス / ロスカット | 確かめ: 点数(順位) | 古い実データ シャープ / 年率 / 最大 DD "
             "| 古いランダム 16 本 シャープ / 年率 / 最大 DD(中央・最悪) / プラス / ロスカット |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    oos_rank = {k: r for r, k in enumerate(sorted(range(len(GRID)), key=lambda k: -oos_s[k]["score"]), start=1)}
    for k in order[:10] + [len(GRID)]:
        a, b = is_s[k], oos_s[k]
        ra, rb = a["real"], b["real"]
        rk = oos_rank.get(k, "-")
        lines.append(
            f"| {label(k)} | {a['score']:+.2f} | {ra['sharpe']:+.2f} / {ra['cagr']:+.1%} / {ra['dd']:.0%} | "
            f"{a['rand_sharpe']:+.2f} / {a['rand_cagr']:+.1%} / {a['rand_dd']:.0%}・{a['rand_dd_max']:.0%} / "
            f"{a['rand_pos']:.0%} / {a['rand_losscut']}/{a['n_rand']} | {b['score']:+.2f}({rk}) | "
            f"{rb['sharpe']:+.2f} / {rb['cagr']:+.1%} / {rb['dd']:.0%} | "
            f"{b['rand_sharpe']:+.2f} / {b['rand_cagr']:+.1%} / {b['rand_dd']:.0%}・{b['rand_dd_max']:.0%} / "
            f"{b['rand_pos']:.0%} / {b['rand_losscut']}/{b['n_rand']} |")
    x = np.array([is_s[k]["score"] for k in range(len(GRID))])
    y = np.array([oos_s[k]["score"] for k in range(len(GRID))])
    rho = float(np.corrcoef(_rank(x), _rank(y))[0, 1])
    best = order[0]
    pct = float((y < y[best]).mean())
    lines += ["", f"選ぶ段と確かめる段の点数の順位の相関(81 通り): {rho:+.2f}。選ぶ段の一番は、確かめる段で 81 通り中 "
              f"{oos_rank[best]} 位(下から {pct:.0%})。確かめる段の点数の中央値 {np.median(y):+.2f}"]
    for name, key in (("チャネル", lambda g: (g["n_in"], g["n_out"])), ("金利差の下限", lambda g: g["min_carry"]),
                      ("キャリー・トゥ・リスク", lambda g: g["c2r_min"])):
        groups: dict = {}
        for k, g in enumerate(GRID):
            groups.setdefault(key(g), []).append(k)
        cells = ", ".join(f"{v}: 選ぶ {np.mean([x[k] for k in ks]):+.2f} / 確かめ {np.mean([y[k] for k in ks]):+.2f}"
                          for v, ks in groups.items())
        lines.append(f"- {name}ごとの点数の平均: {cells}")
    return "\n".join(lines)


def main() -> None:
    with ProcessPoolExecutor(6) as ex:
        rows = [r for part in ex.map(job, markets()) for r in part]
    out = Path(tempfile.gettempdir()) / "e026_rows.json"   # 表を作り直すための生の結果(コミットしない)
    out.write_text(json.dumps(rows))
    print(report(rows))


if __name__ == "__main__":
    main()
