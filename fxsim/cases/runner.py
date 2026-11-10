"""テストケースを実験に直して回し、記録する。"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from ..experiments.config import ExperimentConfig
from ..experiments.runner import git_sha, run_experiment
from ..experiments.stats import strategy_stats
from ..markets import MARKETS
from ..storage import Tx, queries, writer
from .loader import load_cases
from .model import BehaviorCase, TestCase

# テストケースの保存の細かさ(長期の 5〜10 年でも DB が膨らみすぎないように)
CASE_STORAGE = {"candle_timeframe": 240, "equity_detail_timeframe": 1440, "equity_timeframe": 1440,
                "trades": "detail"}


def to_experiment(t: TestCase, b: BehaviorCase) -> ExperimentConfig:
    """テストケースを実験の設定に機械的に直す。

    期間を start / end で指定したとき、実データ系はその期間を切り出し、合成相場は開始日を合わせる。
    """
    markets = []
    for p in t.patterns:
        params = dict(p.params)
        if "start" in t.period:
            if MARKETS.create(p.market, params).source is not None:
                params["source"] = {**params.get("source", {}), "start": str(t.period["start"]),
                                    "end": str(t.period["end"])}
            else:
                params["base"] = {**params.get("base", {}), "start": str(t.period["start"])}
        markets.append({"model": p.market, "label": p.display_label, "params": params, "seeds": p.seeds})
    first_seeds = sorted({p.seeds[0] for p in t.patterns})
    return ExperimentConfig.from_dict({
        "name": f"{t.id} {t.name}".strip(),
        "kind": "test_case",
        "proposal": t.id,
        "notes": f"behavior {b.id}: {b.name}",
        "years": t.years(),
        "seeds": first_seeds,
        "detail_seeds": t.detail_seeds if t.detail_seeds is not None else first_seeds,
        "workers": t.workers,
        "engine": b.engine,
        "storage": CASE_STORAGE,
        "brokers": [b.broker_spec],
        "markets": markets,
        "strategies": [{"name": b.strategy, "label": b.id, "params": b.strategy_params(), "schedule": b.schedule}],
    })


def summarize_run(con: sqlite3.Connection, run_id: int) -> dict:
    """case_runs.summary_json に残す要約: 市場ごとと総合の統計、パターンごとの損益。"""
    stats = strategy_stats(con, run_id)
    keys = ("pnl", "net_pips_per_year", "sharpe", "max_dd", "trades_per_year")
    out: dict = {"by_market": {}, "all": None, "patterns": []}
    for s in stats:
        item = {k: s[k] for k in keys}
        if s["market"] == "ALL":
            out["all"] = item
            out["patterns"] += s["patterns"]
        else:
            out["by_market"][s["market"]] = item
    return out


def run_cases(con: sqlite3.Connection, db_path: str, ids: list[str] | None = None, *, force: bool = False,
              root: Path | str | None = None, log=print) -> list[int]:
    """テストケースを回す。定義(動作・テスト)とコードの版が前回と同じなら飛ばす。回した run 番号を返す。"""
    book = load_cases(root)
    sha = git_sha()
    with Tx(con):
        bh = {k: writer.upsert_case_def(con, kind="behavior", id_=k, name=b.name, path=b.path,
                                        definition=b.definition()) for k, b in book.behaviors.items()}
        th = {k: writer.upsert_case_def(con, kind="test", id_=k, name=t.name, path=t.path,
                                        definition=t.definition()) for k, t in book.tests.items()}
    run_ids = []
    for tid in ids or sorted(book.tests):
        if tid not in book.tests:
            raise KeyError(f"unknown test case {tid}. known: {sorted(book.tests)}")
        t = book.tests[tid]
        b = book.behaviors[t.behavior]
        prev = queries.find_case_run(con, tid, th[tid], bh[b.id], sha)
        if prev is not None and not force:
            log(f"{tid}: 定義とコードが前回(run #{prev})と同じなので飛ばします(--force で回し直し)")
            continue
        cfg = to_experiment(t, b)
        case_run_id = writer.start_case_run(con, test_id=tid, behavior_id=b.id, test_hash=th[tid],
                                            behavior_hash=bh[b.id], code_sha=sha)
        log(f"{tid}: behavior {b.id} ({b.strategy}, tf={cfg.strategies[0].params.get('timeframe')}m) x "
            f"{sum(len(p.seeds) for p in t.patterns)} patterns, {cfg.years:.2f} years")
        run_id = run_experiment(cfg, db_path, log=log)
        summary = summarize_run(con, run_id)
        writer.finish_case_run(con, case_run_id, run_id=run_id, status=queries.run_status(con, run_id),
                               summary=summary)
        p = (summary["all"] or {}).get("pnl") or {}
        if p.get("n"):
            log(f"{tid}: run #{run_id} 損益 平均 {p['mean']:,.0f} / 中央 {p['median']:,.0f} / 合計 {p['total']:,.0f}"
                f" / プラス {p['positive_ratio'] * 100:.0f}% / t {p['t']:.2f} (n={p['n']})")
        run_ids.append(run_id)
    return run_ids
