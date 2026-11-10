"""実験の実行。市場の経路(市場の案 x シード)ごとに 1 プロセスで生成し、
その経路の上で ブローカー x 戦略 のバックテストを順に回す。進捗は逐次 SQLite に書く。
"""
from __future__ import annotations

import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .. import catalog  # noqa: F401  (登録を済ませる)
from ..backtest import Backtester, BrokerConfig, EngineConfig, compute_metrics
from ..calendar import NS_MIN
from ..data import BAR_OFFSET_MINUTES
from ..markets import MARKETS
from ..markets.evaluation import compute_facts, score_facts
from ..storage import DEFAULT_DB, Tx, checkpoint, connect, dumps, init_db, now_iso
from ..strategies import STRATEGIES
from .config import ExperimentConfig

NS = 1_000_000_000


def git_sha() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                             timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "fxsim"], cwd=root, capture_output=True, text=True,
                               timeout=5).stdout.strip()
        return sha + ("+dirty" if dirty else "") if sha else None
    except Exception:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# 計画: run / paths / jobs の行を先に全部作る(画面で全体の進み具合が見える)
# ---------------------------------------------------------------------------

def plan(con, cfg: ExperimentConfig) -> tuple[int, list[dict]]:
    tasks = []
    with Tx(con):
        run_id = con.execute(
            "INSERT INTO runs(name, kind, proposal, status, created_at, git_sha, years, seeds_json, config_json, notes)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (cfg.name, cfg.kind, cfg.proposal, "running", now_iso(), git_sha(), cfg.years, dumps(cfg.seeds),
             dumps(cfg.to_dict()), cfg.notes)).lastrowid
        for m in cfg.markets:
            model = MARKETS.create(m.model, m.params)
            info = model.info()
            seeds = cfg.seeds[:1] if model.deterministic else cfg.seeds
            for seed in seeds:
                detail = int(seed in cfg.detail_seeds)
                path_id = con.execute(
                    "INSERT INTO paths(run_id, market, market_label, market_version, params_hash, params_json, seed,"
                    " detail, status) VALUES (?,?,?,?,?,?,?,?,?)",
                    (run_id, m.model, m.label, info["version"], info["params_hash"], dumps(info["params"]), seed,
                     detail, "queued")).lastrowid
                jobs = []
                for b in cfg.brokers:
                    for s in cfg.strategies:
                        strat = STRATEGIES.create(s.name, s.params)
                        si = strat.info()
                        job_id = con.execute(
                            "INSERT INTO jobs(run_id, path_id, broker, strategy, strategy_label, strategy_version,"
                            " params_hash, params_json, timeframe, status) VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (run_id, path_id, b.label, s.name, s.label, si["version"], si["params_hash"],
                             dumps(si["params"]), strat.timeframe, "queued")).lastrowid
                        jobs.append({"job_id": job_id, "broker": b.__dict__, "strategy": s.__dict__})
                tasks.append({"path_id": path_id, "market": m.__dict__, "seed": seed, "detail": bool(detail),
                              "jobs": jobs})
    return run_id, tasks


# ---------------------------------------------------------------------------
# 実行(ワーカー側)
# ---------------------------------------------------------------------------

def _bucket_last(t_ns: np.ndarray, minutes: int) -> tuple[np.ndarray, np.ndarray]:
    """足を minutes ごとに束ね、各束の最後の足の位置と、束の開始時刻(秒)を返す。"""
    off = BAR_OFFSET_MINUTES * NS_MIN
    key = (t_ns + off) // (minutes * NS_MIN)
    last = np.flatnonzero(np.r_[key[1:] != key[:-1], True])
    return last, (key[last] * minutes * NS_MIN - off) // NS


class LiveWriter:
    """バックテストの途中経過を SQLite に書く。"""

    def __init__(self, con, job_id: int, bars, *, store_trades: bool, equity_minutes: int, delay_ms: float):
        self.con = con
        self.job_id = job_id
        self.bars = bars
        self.store_trades = store_trades
        self.delay = delay_ms / 1000.0
        tf = max(equity_minutes, bars.timeframe)
        self.eq_idx, self.eq_t = _bucket_last(bars.t, tf)
        self.eq_ptr = 0
        self.trade_ptr = 0
        with Tx(con):
            con.execute("UPDATE jobs SET status='running', bars_total=?, started_at=?, updated_at=? WHERE id=?",
                        (len(bars), now_iso(), now_iso(), job_id))

    def progress(self, bt: Backtester, i: int) -> None:
        con = self.con
        with Tx(con):
            if self.store_trades and len(bt.trades) > self.trade_ptr:
                rows = []
                for k in range(self.trade_ptr, len(bt.trades)):
                    t = bt.trades[k]
                    rows.append((self.job_id, k, t.side, t.units, t.entry_t // NS, t.exit_t // NS, t.entry_price,
                                 t.exit_price, t.pnl, t.pips, t.gross_pips, t.cost_pips, t.swap, t.reason,
                                 t.bars_held))
                con.executemany("INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                self.trade_ptr = len(bt.trades)
            j = self.eq_ptr
            n_eq = len(self.eq_idx)
            rows = []
            while j < n_eq and self.eq_idx[j] <= i:
                k = int(self.eq_idx[j])
                rows.append((self.job_id, int(self.eq_t[j]), float(bt.balance_curve[k]), float(bt.equity_curve[k])))
                j += 1
            if rows:
                con.executemany("INSERT OR REPLACE INTO equity VALUES (?,?,?,?)", rows)
            self.eq_ptr = j
            con.execute("UPDATE jobs SET bars_done=?, sim_t=?, balance=?, equity=?, n_trades=?, updated_at=?"
                        " WHERE id=?",
                        (i + 1, int(self.bars.t[i] // NS), float(bt.balance), float(bt.equity), len(bt.trades),
                         now_iso(), self.job_id))
        if self.delay:
            time.sleep(self.delay)

    def finish(self, bt: Backtester, metrics: dict) -> None:
        m = metrics
        with Tx(self.con):
            self.con.execute(
                "UPDATE jobs SET status='done', finished_at=?, updated_at=?, metrics_json=?, sharpe=?, cagr=?,"
                " max_dd=?, profit_factor=?, win_rate=?, trades_per_year=?, net_pips_per_year=?,"
                " gross_pips_per_year=?, cost_pips_per_year=?, gross_sharpe=?, final_equity=?, pnl=?, ruined=?,"
                " hunted=?, balance=?, equity=?, n_trades=? WHERE id=?",
                (now_iso(), now_iso(), dumps(m), m["sharpe"], m["cagr"], m["max_dd"],
                 m["profit_factor"] if np.isfinite(m["profit_factor"]) else None, m["win_rate"],
                 m["trades_per_year"], m["net_pips_per_year"], m["gross_pips_per_year"], m["cost_pips_per_year"],
                 m["gross_sharpe"], m["final_equity"], m["final_equity"] - bt.cfg.initial_balance,
                 int(m["ruined"]), m["hunted"], bt.balance, bt.equity, m["n_trades"], self.job_id))


def _write_path(con, task: dict, md, cfg: dict) -> int:
    facts = compute_facts(md)
    sc = score_facts(facts)
    facts["_checks"] = sc["checks"]
    candle_tf = max(int(cfg["storage"]["candle_timeframe"]), md.step_minutes)
    if md.step_minutes > 1 and candle_tf % md.step_minutes:
        candle_tf = md.step_minutes
    with Tx(con):
        con.execute("UPDATE paths SET status='running', candle_tf=?, t_start=?, t_end=?, facts_json=?,"
                    " realism_score=?, n_fail=? WHERE id=?",
                    (candle_tf, int(md.times[0] // NS), int(md.times[-1] // NS), dumps(facts),
                     sc["realism_score"], sc["n_fail"], task["path_id"]))
        if task["detail"]:
            b = md.bars(candle_tf)
            pip = md.pip
            con.executemany(
                "INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?)",
                zip([task["path_id"]] * len(b), (b.t // NS).tolist(), b.open.tolist(), b.high.tolist(),
                    b.low.tolist(), b.close.tolist(), (b.spread_mean / pip).tolist()))
            ev = []
            for e in md.events:
                d = {k: v for k, v in e.items() if k not in ("idx", "kind")}
                pips = d.get("pips", d.get("final_pips", d.get("overshoot_pips")))
                ev.append((task["path_id"], int(md.times[e["idx"]] // NS), e["kind"], d.get("direction"),
                           pips, dumps(d)))
            con.executemany("INSERT INTO market_events VALUES (?,?,?,?,?,?)", ev)
    return candle_tf


def run_path(db_path: str, task: dict, cfg: dict) -> dict:
    con = connect(db_path)
    t0 = time.time()
    try:
        m = task["market"]
        model = MARKETS.create(m["model"], m["params"])
        md = model.generate(years=cfg["years"], seed=task["seed"])
        candle_tf = _write_path(con, task, md, cfg)
        eng = EngineConfig(**cfg["engine"])
        detail = task["detail"]
        trades_mode = cfg["storage"]["trades"]
        for job in task["jobs"]:
            s = job["strategy"]
            strat = STRATEGIES.create(s["name"], s["params"])
            tf = strat.timeframe
            if tf < md.step_minutes or tf % md.step_minutes:
                with Tx(con):
                    con.execute("UPDATE jobs SET status='skipped', note=?, updated_at=? WHERE id=?",
                                (f"timeframe {tf}m is finer than the data ({md.step_minutes}m)", now_iso(),
                                 job["job_id"]))
                continue
            b = job["broker"]
            broker = BrokerConfig.preset(b["preset"], b["overrides"])
            bars = md.bars(tf)
            writer = LiveWriter(
                con, job["job_id"], bars,
                store_trades=(trades_mode == "all") or (trades_mode == "detail" and detail),
                equity_minutes=candle_tf if detail else int(cfg["storage"]["equity_timeframe"]),
                delay_ms=float(cfg["live"]["delay_ms"]) if detail else 0.0)
            bt = Backtester(bars, strat, broker, eng, pip=md.pip, seed=task["seed"])
            try:
                bt.run(progress=writer.progress, every=int(cfg["live"]["flush_every"]))
                writer.finish(bt, compute_metrics(bt))
            except Exception as e:
                with Tx(con):
                    con.execute("UPDATE jobs SET status='failed', note=?, updated_at=? WHERE id=?",
                                (f"{type(e).__name__}: {e}", now_iso(), job["job_id"]))
                raise
        with Tx(con):
            con.execute("UPDATE paths SET status='done' WHERE id=?", (task["path_id"],))
        return {"path_id": task["path_id"], "ok": True, "sec": time.time() - t0}
    except Exception as e:
        with Tx(con):
            con.execute("UPDATE paths SET status='failed' WHERE id=?", (task["path_id"],))
        return {"path_id": task["path_id"], "ok": False, "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()}
    finally:
        con.close()


def run_experiment(cfg: ExperimentConfig, db_path: str | Path | None = None, log=print) -> int:
    db_path = str(db_path or DEFAULT_DB)
    con = init_db(db_path)
    t0 = time.time()
    run_id, tasks = plan(con, cfg)
    n_jobs = sum(len(t["jobs"]) for t in tasks)
    log(f"run #{run_id} '{cfg.name}': {len(tasks)} market paths x {n_jobs // max(1, len(tasks))} jobs "
        f"= {n_jobs} backtests, workers={cfg.workers}, db={db_path}")
    c = cfg.to_dict()
    errors = []
    # 詳細保存の経路(画面で追う)を先に回す
    tasks.sort(key=lambda t: (not t["detail"], t["path_id"]))
    if cfg.workers <= 1:
        results = (run_path(db_path, t, c) for t in tasks)
        for r in results:
            _log_result(r, log, errors)
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futs = [ex.submit(run_path, db_path, t, c) for t in tasks]
            for f in as_completed(futs):
                _log_result(f.result(), log, errors)
    status = "failed" if errors else "done"
    with Tx(con):
        con.execute("UPDATE runs SET status=?, finished_at=?, elapsed_sec=? WHERE id=?",
                    (status, now_iso(), time.time() - t0, run_id))
    checkpoint(con)
    con.close()
    log(f"run #{run_id} {status} in {time.time() - t0:.1f}s")
    return run_id


def _log_result(r: dict, log, errors: list) -> None:
    if r["ok"]:
        log(f"  path {r['path_id']} done ({r['sec']:.1f}s)")
    else:
        errors.append(r)
        log(f"  path {r['path_id']} FAILED: {r['error']}\n{r['trace']}")


__all__ = ["run_experiment", "plan", "run_path"]
