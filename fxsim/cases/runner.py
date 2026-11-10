"""テストケースを実験に直して回し、記録する。"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from ..experiments.config import ExperimentConfig
from ..experiments.runner import git_sha, run_experiment
from ..experiments.stats import strategy_stats
from ..markets import MARKETS
from ..storage import Tx, queries, writer
from ..verification import NULL_LABEL, compute_verdict
from .loader import load_cases
from .model import BehaviorCase, TestCase

# テストケースの保存の細かさ(長期の 5〜10 年でも DB が膨らみすぎないように)
CASE_STORAGE = {"candle_timeframe": 240, "equity_detail_timeframe": 1440, "equity_timeframe": 1440,
                "trades": "detail"}
# 検証はパターンも帰無も多いので、資産曲線は詳細保存のパターンが週次、それ以外は約 30 日ごと
# (画面の資産曲線の束に十分)。足は日足(30 年の検証でも DB が膨らまないように)
VERIFY_STORAGE = {**CASE_STORAGE, "candle_timeframe": 1440, "equity_detail_timeframe": 10080,
                  "equity_timeframe": 43200, "events": False}


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
    strategies = [{"name": b.strategy, "label": b.id, "params": b.strategy_params(), "schedule": b.schedule}]
    if t.verify:
        tf = b.strategy_params().get("timeframe")
        null_params = {"inner": b.strategy, "inner_params": dict(b.params)}
        if tf:
            null_params["timeframe"] = int(tf)
        strategies.append({"name": "direction_null", "label": NULL_LABEL, "params": null_params,
                           "schedule": b.schedule})
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
        "storage": VERIFY_STORAGE if t.verify else CASE_STORAGE,
        "brokers": [b.broker_spec],
        "markets": markets,
        "strategies": strategies,
    })


def summarize_run(con: sqlite3.Connection, run_id: int, verify_label: str | None = None) -> dict:
    """case_runs.summary_json に残す要約: 市場ごとと総合の統計、パターンごとの損益。"""
    stats = strategy_stats(con, run_id)
    keys = ("pnl", "net_pips_per_year", "sharpe", "max_dd", "trades_per_year")
    out: dict = {"by_market": {}, "all": None, "patterns": []}
    if verify_label:
        stats = [s for s in stats if s["strategy"] == verify_label]   # 帰無(NULL)は要約に混ぜない
        out["verdict"] = compute_verdict(con, run_id, verify_label)
    for s in stats:
        item = {k: s[k] for k in keys}
        item["ruin_ratio"] = s.get("ruin_ratio", 0.0)
        if s["market"] == "ALL":
            out["all"] = item
            out["patterns"] += s["patterns"]
        else:
            out["by_market"][s["market"]] = item
    return out


def run_cases(con: sqlite3.Connection, db_path: str, ids: list[str] | None = None, *, force: bool = False,
              root: Path | str | None = None, extra: list[TestCase] = (), log=print) -> list[int]:
    """テストケースを回す。定義(動作・テスト)とコードの版が前回と同じなら飛ばす。回した run 番号を返す。

    extra: ファイルに書かずに回すテストケース(fxsim verify が作る)。
    """
    book = load_cases(root)
    for t in extra:
        t.validate(book.behaviors)
        book.tests[t.id] = t
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
        summary = summarize_run(con, run_id, verify_label=b.id if t.verify else None)
        writer.finish_case_run(con, case_run_id, run_id=run_id, status=queries.run_status(con, run_id),
                               summary=summary)
        from ..reports.summary import auto_summary

        auto_summary(con, db_path, run_id, log=log)
        p = (summary["all"] or {}).get("pnl") or {}
        if p.get("n"):
            log(f"{tid}: run #{run_id} 損益 平均 {p['mean']:,.0f} / 中央 {p['median']:,.0f} / 合計 {p['total']:,.0f}"
                f" / プラス {p['positive_ratio'] * 100:.0f}% / t {p['t']:.2f} (n={p['n']})")
        if "verdict" in summary:
            log(f"{tid}: 判定 {summary['verdict']['verdict']}")
        run_ids.append(run_id)
    return run_ids


def verification_case(behavior_id: str, *, years: float = 10.0, seeds: list[int] | None = None,
                      start: str | None = None, end: str | None = None, block_days: float = 20.0,
                      dataset: str | None = None) -> TestCase:
    """標準のパターンの組で、動作ケースを検証するテストケースを作る(ID は V-<動作ケース>)。

    期間を省くと、同梱の実データの末尾 years 年。合成相場の開始日も同じ日にそろえる。
    block_days は並べ替え(real_bootstrap)のブロックの長さ。並べ替えはブロックより長い構造を壊すので、
    戦略が見る期間(移動平均やモメンタムの日数)より長くする。
    dataset は実データ系のパターンに使う CSV(既定は同梱の USDJPY 日足)。
    """
    from ..data import load_ohlc
    from ..markets.components.real import DEFAULT_DATASET
    from ..verification import STANDARD_PATTERNS

    seeds = seeds or list(range(1, 9))
    if start is None or end is None:
        s = load_ohlc(dataset or DEFAULT_DATASET).window(years=years)
        start = str(np.datetime64(int(s.times[0]), "ns"))[:10]
        end = str(np.datetime64(int(s.times[-1]), "ns"))[:10]
    patterns = [{"market": m, "seeds": seeds[:1] if m == "real" else seeds} for m in STANDARD_PATTERNS]
    for p in patterns:
        if p["market"].startswith("real"):
            src = {"path": dataset} if dataset else {}
            if p["market"] == "real_bootstrap":
                src["block_days"] = float(block_days)
            if src:
                p["params"] = {"source": src}
    return TestCase(id=f"V-{behavior_id}", behavior=behavior_id, patterns=patterns,
                    name=f"{behavior_id} の長期検証(標準パターン、並べ替え {block_days:g} 日ブロック)",
                    period={"start": start, "end": end},
                    detail_seeds=seeds[:1], verify=True)
