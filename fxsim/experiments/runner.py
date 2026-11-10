"""実験の実行。

1. plan        runs / paths / jobs の行を先に全部作る(画面で全体の進み具合が見える)
2. run_path    市場の経路(市場の案 x シード)ごとに 1 プロセスで生成し、その上で ブローカー x 戦略 を順に回す
3. 途中経過と結果は storage.writer が逐次 SQLite に書く
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .. import catalog  # noqa: F401  (ワーカーでも登録を済ませる)
from ..backtest import BrokerConfig, compute_metrics
from ..backtest.portfolio import build_backtester
from ..data import MarketData
from ..markets import MARKETS
from ..markets.evaluation import compute_facts, score_facts
from ..paths import DEFAULT_DB, REPO_ROOT
from ..storage import Tx, checkpoint, connect, init_db, writer
from ..strategies import STRATEGIES
from .config import BrokerSpec, ExperimentConfig, MarketSpec, StrategySpec

WORKER_NICE = 10


def _lower_priority() -> None:
    """ワーカーの優先度を下げる(画面の操作を先に回す)。対応しない OS では何もしない。"""
    with contextlib.suppress(AttributeError, OSError):
        os.nice(WORKER_NICE)


def git_sha() -> str | None:
    """コードの版。fxsim/ に未コミットの変更があれば "+dirty" を付ける。"""
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                  timeout=5).stdout.strip()

        sha = git("rev-parse", "--short", "HEAD")
        return sha + ("+dirty" if git("status", "--porcelain", "fxsim") else "") if sha else None
    except Exception:  # pragma: no cover
        return None


@dataclass
class JobTask:
    job_id: int
    broker: BrokerSpec
    strategy: StrategySpec


@dataclass
class PathTask:
    path_id: int
    market: MarketSpec
    seed: int
    detail: bool
    jobs: list[JobTask] = field(default_factory=list)


@dataclass
class PathResult:
    path_id: int
    ok: bool
    sec: float = 0.0
    error: str = ""
    trace: str = ""


def plan(con, cfg: ExperimentConfig) -> tuple[int, list[PathTask]]:
    tasks: list[PathTask] = []
    with Tx(con):
        config = cfg.to_dict()
        # プリセットの中身は版によって変わるので、実際に使った値を記録しておく
        config["brokers_resolved"] = {b.label: BrokerConfig.preset(b.preset, b.overrides).to_dict()
                                      for b in cfg.brokers}
        run_id = writer.create_run(con, name=cfg.name, kind=cfg.kind, proposal=cfg.proposal, git_sha=git_sha(),
                                   years=cfg.years, seeds=cfg.seeds, config=config, notes=cfg.notes)
        for m in cfg.markets:
            model = MARKETS.create(m.model, m.params)
            seeds = m.seeds or cfg.seeds
            for seed in seeds[:1] if model.deterministic else seeds:
                detail = seed in cfg.detail_seeds
                path_id = writer.add_path(con, run_id, market=m.model, label=m.label, info=model.info(), seed=seed,
                                          detail=detail)
                task = PathTask(path_id, m, seed, detail)
                for b in cfg.brokers:
                    for s in cfg.strategies:
                        strat = STRATEGIES.create(s.name, s.params)
                        job_id = writer.add_job(con, run_id, path_id, broker=b.label, strategy=s.name, label=s.label,
                                                info=strat.info(), schedule=s.trading_schedule().to_dict(),
                                                timeframe=strat.timeframe)
                        task.jobs.append(JobTask(job_id, b, s))
                tasks.append(task)
    return run_id, tasks


def _candle_timeframe(cfg: ExperimentConfig, md: MarketData) -> int:
    tf = max(cfg.storage.candle_timeframe, md.step_minutes)
    return md.step_minutes if tf % md.step_minutes else tf


def _run_job(con, job: JobTask, task: PathTask, md: MarketData, cfg: ExperimentConfig, candle_tf: int) -> None:
    strat = STRATEGIES.create(job.strategy.name, job.strategy.params)
    tf = strat.timeframe
    if tf < md.step_minutes or tf % md.step_minutes:
        writer.mark_job(con, job.job_id, "skipped", f"timeframe {tf}m is finer than the data ({md.step_minutes}m)")
        return
    if strat.multi and not md.meta.get("legs"):
        writer.mark_job(con, job.job_id, "skipped", "multi-pair strategy needs a basket market (basket_*)")
        return
    bars = md.bars(tf)
    equity_minutes = (max(candle_tf, cfg.storage.equity_detail_timeframe) if task.detail
                      else cfg.storage.equity_timeframe)
    w = writer.JobWriter(con, job.job_id, bars, store_trades=cfg.storage.store_trades(task.detail),
                         equity_minutes=equity_minutes, delay_ms=cfg.live.delay_ms if task.detail else 0.0)
    bt = build_backtester(md, strat, BrokerConfig.preset(job.broker.preset, job.broker.overrides), cfg.engine,
                          seed=task.seed, schedule=job.strategy.trading_schedule())
    try:
        bt.run(progress=w.progress, every=cfg.live.flush_every)
    except Exception as e:
        writer.mark_job(con, job.job_id, "failed", f"{type(e).__name__}: {e}")
        raise
    w.finish(bt, compute_metrics(bt))


def run_path(db_path: str, task: PathTask, cfg: ExperimentConfig) -> PathResult:
    """ワーカーの仕事: 経路を 1 本作り、その上の全ジョブを回す。"""
    con = connect(db_path)
    t0 = time.time()
    try:
        md = MARKETS.create(task.market.model, task.market.params).generate(years=cfg.years, seed=task.seed)
        facts = compute_facts(md)
        score = score_facts(facts)
        facts["_checks"] = score["checks"]
        candle_tf = _candle_timeframe(cfg, md)
        writer.record_path(con, task.path_id, md, seed=task.seed, detail=task.detail, candle_tf=candle_tf,
                           facts=facts, score=score, events=cfg.storage.events)
        for job in task.jobs:
            _run_job(con, job, task, md, cfg, candle_tf)
        writer.set_path_status(con, task.path_id, "done")
        return PathResult(task.path_id, True, time.time() - t0)
    except Exception as e:
        writer.set_path_status(con, task.path_id, "failed")
        return PathResult(task.path_id, False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
    finally:
        con.close()


def run_experiment(cfg: ExperimentConfig, db_path: str | Path | None = None, log=print) -> int:
    db_path = str(db_path or DEFAULT_DB)
    con = init_db(db_path)
    t0 = time.time()
    run_id, tasks = plan(con, cfg)
    n_jobs = sum(len(t.jobs) for t in tasks)
    log(f"run #{run_id} '{cfg.name}': {len(tasks)} market paths x {n_jobs // max(1, len(tasks))} jobs "
        f"= {n_jobs} backtests, workers={cfg.workers}, db={db_path}")
    tasks.sort(key=lambda t: (not t.detail, t.path_id))   # 画面で追う詳細保存の経路を先に
    errors: list[PathResult] = []

    def done(r: PathResult) -> None:
        if r.ok:
            log(f"  path {r.path_id} done ({r.sec:.1f}s)")
        else:
            errors.append(r)
            log(f"  path {r.path_id} FAILED: {r.error}\n{r.trace}")

    if cfg.workers <= 1:
        for t in tasks:
            done(run_path(db_path, t, cfg))
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers, initializer=_lower_priority) as ex:
            for f in as_completed([ex.submit(run_path, db_path, t, cfg) for t in tasks]):
                done(f.result())
    status = "failed" if errors else "done"
    writer.finish_run(con, run_id, status, time.time() - t0)
    checkpoint(con)
    con.close()
    log(f"run #{run_id} {status} in {time.time() - t0:.1f}s")
    return run_id


__all__ = ["JobTask", "PathResult", "PathTask", "git_sha", "plan", "run_experiment", "run_path"]
