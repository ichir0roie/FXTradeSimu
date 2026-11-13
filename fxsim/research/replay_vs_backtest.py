"""E027: 10 ペアの常駐の再生(trade-replay)で実口座に入った取引と、同じ戦略を相場の DB の日足でそのまま回した
バックテストの取引を突き合わせる。

    python -m fxsim.research.replay_vs_backtest data/trading/replay/E027-majors configs/trading/demo-majors-b047.yaml

常駐の中のシミュレーションと、別に回したバックテスト(real_pairs、日足、同じパラメータ、仮の口座 10 万円)が
同じ日に同じペアを同じ向きで建てていれば、「評価した戦略がそのまま実口座で動く」ことの確かめになる。
違いが出うるところ: 常駐の足は 1 分足から作った日足と DB の日足のつなぎ目、円への換算(常駐はその時の気配、
バックテストは期間の最初)、起動の時点ですでに持っていた建玉(常駐は途中から乗らない)、故障での取りこぼし。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.portfolio import build_backtester
from fxsim.cases.loader import load_cases
from fxsim.catalog import *  # noqa: F403
from fxsim.markets import MARKETS
from fxsim.strategies import STRATEGIES
from fxsim.trading import load_trader_config


def backtest_trades(cfg, start: str, end: str, history_days: int = 4400) -> list[dict]:
    b = load_cases().behaviors[cfg.behavior]
    params = {**b.strategy_params(), **(cfg.strategy_params or {}), "protect": 0.0}
    t1 = np.datetime64(end, "D")
    t0 = t1 - np.timedelta64(history_days, "D")
    md = MARKETS.create("real_pairs", {"pairs": [s[:6] for s in cfg.symbols],
                                       "source": {"start": str(t0), "end": str(t1), "daily_fill_hour": 2}}) \
        .generate(seed=1)
    s = STRATEGIES.create(b.strategy, params)
    bt = build_backtester(md, s, BrokerConfig.preset(b.broker_spec["preset"]), EngineConfig(**b.engine), seed=1)
    bt.run()
    lo = np.datetime64(start, "s").astype(np.int64) * 1_000_000_000
    out = []
    for t in bt.trades:
        if t.entry_t >= lo:
            out.append({"symbol": t.pair.upper(), "side": t.side,
                        "day": str(np.datetime64(int(t.entry_t), "ns").astype("datetime64[D]")),
                        "exit_day": str(np.datetime64(int(t.exit_t), "ns").astype("datetime64[D]")),
                        "reason": t.reason})
    return out


def replay_trades(out_dir: Path, cfg) -> list[dict]:
    """常駐の記録(journal の fill)で、実口座に通った新規。日付は、注文を出した時刻(サーバーの 2 時ごろ、UTC の
    前の日の 23 時か当日の 0 時)を UTC の日付に寄せたもの(バックテストの約定 = 日足の始まりの UTC の日付と同じ日)。"""
    jp = out_dir / cfg.name / "journal.jsonl"
    out = []
    for line in jp.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["kind"] != "fill":
            continue
        ok = {x["symbol"]: all(y["ok"] for y in x["results"]) for x in rec.get("results", [])}
        for a in rec["actions"]:
            if a["kind"] == "open" and ok.get(a["symbol"], False):
                out.append({"symbol": a["symbol"], "side": a["side"],
                            "day": str(np.datetime64(int(rec["ts"] + 3 * 3600), "s").astype("datetime64[D]"))})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("replay_dir")
    ap.add_argument("config")
    a = ap.parse_args(argv)
    out = Path(a.replay_dir)
    cfg = load_trader_config(a.config)
    r = json.loads((out / "result.json").read_text(encoding="utf-8"))
    real = replay_trades(out, cfg)
    bt = backtest_trades(cfg, r["start"], r["end"])
    key = lambda x: (x["symbol"], x["side"], x["day"])  # noqa: E731
    near = lambda x, ys: any(x["symbol"] == y["symbol"] and x["side"] == y["side"]  # noqa: E731
                             and abs((np.datetime64(x["day"]) - np.datetime64(y["day"])).astype(int)) <= 1 for y in ys)
    both = [x for x in bt if near(x, real)]
    only_bt = [x for x in bt if not near(x, real)]
    only_real = [x for x in real if not near(x, bt)]
    print(f"## {cfg.name}: 再生の実口座の新規 {len(real)} 回 / 同じ期間のバックテストの新規 {len(bt)} 回")
    print(f"- 両方にある(同じペア・同じ向き・±1 日): {len(both)}")
    print(f"- バックテストだけ: {len(only_bt)} {sorted(only_bt, key=key)}")
    print(f"- 実口座だけ: {len(only_real)} {sorted(only_real, key=key)}")


if __name__ == "__main__":
    main()
