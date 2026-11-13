"""書き込み。実験の計画(runs / paths / jobs の行)、相場パターンの保存、ジョブの途中経過と結果。

時刻は DB では UNIX 秒、計算側では int64 ns。変換はこのモジュールだけで行う。
"""
from __future__ import annotations

import sqlite3
import time

import numpy as np

from ..calendar import NS_MIN
from ..data import BAR_OFFSET_MINUTES, MarketData
from ..registry import params_hash
from .db import Tx, dumps, make_run_key, now_iso

NS = 1_000_000_000
SKIP_EVENT_KINDS = {"regime"}   # 数が多く使わないので保存しない(facts の events_per_day には入る)


def _sec(t_ns) -> int:
    return int(t_ns // NS)


# ---- 計画 -------------------------------------------------------------------

def create_run(con: sqlite3.Connection, *, name: str, kind: str, proposal: str | None, git_sha: str | None,
               years: float, seeds: list[int], config: dict, notes: str) -> int:
    created = now_iso()
    return con.execute(
        "INSERT INTO runs(name, kind, proposal, status, created_at, git_sha, years, seeds_json, config_json, notes,"
        " run_key) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (name, kind, proposal, "running", created, git_sha, years, dumps(seeds), dumps(config), notes,
         make_run_key(proposal, kind, created))).lastrowid


def add_path(con: sqlite3.Connection, run_id: int, *, market: str, label: str, info: dict, seed: int,
             detail: bool) -> int:
    return con.execute(
        "INSERT INTO paths(run_id, market, market_label, market_version, params_hash, params_json, seed, detail,"
        " status) VALUES (?,?,?,?,?,?,?,?,?)",
        (run_id, market, label, info["version"], info["params_hash"], dumps(info["params"]), seed, int(detail),
         "queued")).lastrowid


def add_job(con: sqlite3.Connection, run_id: int, path_id: int, *, broker: str, strategy: str, label: str,
            info: dict, schedule: dict, timeframe: int) -> int:
    return con.execute(
        "INSERT INTO jobs(run_id, path_id, broker, strategy, strategy_label, strategy_version, params_hash,"
        " params_json, schedule_json, timeframe, status) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, path_id, broker, strategy, label, info["version"], info["params_hash"], dumps(info["params"]),
         dumps(schedule), timeframe, "queued")).lastrowid


def finish_run(con: sqlite3.Connection, run_id: int, status: str, elapsed_sec: float) -> None:
    with Tx(con):
        con.execute("UPDATE runs SET status=?, finished_at=?, elapsed_sec=? WHERE id=?",
                    (status, now_iso(), elapsed_sec, run_id))


# ---- 相場パターン -------------------------------------------------------------

def pattern_key(md: MarketData, seed: int, candle_tf: int) -> str:
    """相場パターンの中身のキー。同じモデル・パラメータ・シード・期間なら実験をまたいで同じになる。"""
    info = md.model_info
    return params_hash({"model": info["params_hash"], "name": info["name"], "version": info["version"],
                        "seed": seed, "n": len(md), "t0": int(md.times[0]), "t1": int(md.times[-1]),
                        "candle_tf": candle_tf})


def record_path(con: sqlite3.Connection, path_id: int, md: MarketData, *, seed: int, detail: bool, candle_tf: int,
                facts: dict, score: dict, events: bool = True) -> None:
    """経路の性質を書き、詳細保存なら相場パターンの中身(足・イベント)を 1 度だけ保存する。"""
    key = pattern_key(md, seed, candle_tf)
    with Tx(con):
        con.execute("UPDATE paths SET status='running', candle_tf=?, t_start=?, t_end=?, facts_json=?,"
                    " realism_score=?, n_fail=?, data_key=? WHERE id=?",
                    (candle_tf, _sec(md.times[0]), _sec(md.times[-1]), dumps(facts), score["realism_score"],
                     score["n_fail"], key if detail else None, path_id))
        if not detail:
            return
        known = con.execute("SELECT n_events FROM pattern_data WHERE data_key=?", (key,)).fetchone()
        if known is not None:
            # 先にイベント無しで保存されたパターンを、イベントを残したい実験が使うときは、イベントだけ足す
            if events and not known[0]:
                rows = _event_rows(key, md)
                con.executemany("INSERT INTO market_events VALUES (?,?,?,?,?,?)", rows)
                con.execute("UPDATE pattern_data SET n_events=? WHERE data_key=?", (len(rows), key))
            return
        b = md.bars(candle_tf)
        con.executemany(
            "INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?)",
            zip([key] * len(b), (b.t // NS).tolist(), b.open.tolist(), b.high.tolist(), b.low.tolist(),
                b.close.tolist(), (b.spread_mean / md.pip).tolist(), strict=True))
        rows = _event_rows(key, md) if events else []
        con.executemany("INSERT INTO market_events VALUES (?,?,?,?,?,?)", rows)
        con.execute("INSERT INTO pattern_data VALUES (?,?,?,?,?,?,?,?,?)",
                    (key, md.model_info["name"], seed, candle_tf, _sec(md.times[0]), _sec(md.times[-1]), len(b),
                     len(rows), now_iso()))


def _event_rows(key: str, md: MarketData) -> list[tuple]:
    rows = []
    for e in md.events:
        if e["kind"] in SKIP_EVENT_KINDS:
            continue
        d = {k: v for k, v in e.items() if k not in ("idx", "kind")}
        pips = d.get("pips", d.get("final_pips", d.get("overshoot_pips")))
        rows.append((key, _sec(md.times[e["idx"]]), e["kind"], d.get("direction"), pips, dumps(d)))
    return rows


def set_path_status(con: sqlite3.Connection, path_id: int, status: str) -> None:
    with Tx(con):
        con.execute("UPDATE paths SET status=? WHERE id=?", (status, path_id))


# ---- ジョブ -----------------------------------------------------------------

def _bucket_last(t_ns: np.ndarray, minutes: int) -> tuple[np.ndarray, np.ndarray]:
    """足を minutes ごとに束ね、各束の最後の足の位置と、束の開始時刻(秒)を返す。"""
    off = BAR_OFFSET_MINUTES * NS_MIN
    key = (t_ns + off) // (minutes * NS_MIN)
    last = np.flatnonzero(np.r_[key[1:] != key[:-1], True])
    return last, (key[last] * minutes * NS_MIN - off) // NS


def compact_equity(con: sqlite3.Connection, run_id: int, minutes: int, detail_minutes: int | None = None) -> int:
    """保存済みの資産曲線を粗い間隔に間引く(DB を小さくする)。消した行の数を返す。

    各束の最後の点だけを残し、時刻を束の開始にそろえる(書き込み時の間引きと同じ形)。
    詳細保存の経路のジョブは detail_minutes(None なら間引かない)。成績(jobs)は変えない。
    """
    off = BAR_OFFSET_MINUTES * NS_MIN
    jobs = con.execute("SELECT j.id, p.detail FROM jobs j JOIN paths p ON p.id = j.path_id WHERE j.run_id = ?",
                       (run_id,)).fetchall()
    removed = 0
    with Tx(con):
        for job_id, detail in jobs:
            m = detail_minutes if detail else minutes
            if not m:
                continue
            rows = con.execute("SELECT t, balance, equity FROM equity WHERE job_id = ? ORDER BY t",
                               (job_id,)).fetchall()
            if not rows:
                continue
            keep: dict[int, tuple] = {}
            for t, bal, eq in rows:
                keep[(t * NS + off) // (m * NS_MIN)] = (bal, eq)          # 束ごとに最後の点
            if len(keep) == len(rows):
                continue
            con.execute("DELETE FROM equity WHERE job_id = ?", (job_id,))
            con.executemany("INSERT INTO equity VALUES (?,?,?,?)",
                            [(job_id, (k * m * NS_MIN - off) // NS, bal, eq) for k, (bal, eq) in keep.items()])
            removed += len(rows) - len(keep)
    return removed


def mark_job(con: sqlite3.Connection, job_id: int, status: str, note: str) -> None:
    """skipped / failed など、バックテストが終わらなかったジョブの状態を書く。"""
    with Tx(con):
        con.execute("UPDATE jobs SET status=?, note=?, updated_at=? WHERE id=?", (status, note, now_iso(), job_id))


class JobWriter:
    """バックテスト 1 本の途中経過と結果を書く。``Backtester.run(progress=writer.progress)`` に渡す。"""

    def __init__(self, con: sqlite3.Connection, job_id: int, bars, *, store_trades: bool, equity_minutes: int,
                 delay_ms: float = 0.0):
        self.con = con
        self.job_id = job_id
        self.bars = bars
        self.store_trades = store_trades
        self.delay = delay_ms / 1000.0
        self.eq_idx, self.eq_t = _bucket_last(bars.t, max(equity_minutes, bars.timeframe))
        self.eq_ptr = 0
        self.trade_ptr = 0
        with Tx(con):
            con.execute("UPDATE jobs SET status='running', bars_total=?, started_at=?, updated_at=? WHERE id=?",
                        (len(bars), now_iso(), now_iso(), job_id))

    def _new_trades(self, bt) -> list[tuple]:
        rows = [(self.job_id, k, t.side, t.units, _sec(t.entry_t), _sec(t.exit_t), t.entry_price, t.exit_price,
                 t.pnl, t.pips, t.gross_pips, t.cost_pips, t.swap, t.reason, t.bars_held)
                for k, t in enumerate(bt.trades[self.trade_ptr:], start=self.trade_ptr)]
        self.trade_ptr = len(bt.trades)
        return rows

    def _new_equity(self, bt, i: int) -> list[tuple]:
        rows = []
        while self.eq_ptr < len(self.eq_idx) and self.eq_idx[self.eq_ptr] <= i:
            k = int(self.eq_idx[self.eq_ptr])
            rows.append((self.job_id, int(self.eq_t[self.eq_ptr]), float(bt.balance_curve[k]),
                         float(bt.equity_curve[k])))
            self.eq_ptr += 1
        return rows

    def progress(self, bt, i: int) -> None:
        con = self.con
        with Tx(con):
            if self.store_trades:
                trades = self._new_trades(bt)
                if trades:
                    con.executemany("INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", trades)
            equity = self._new_equity(bt, i)
            if equity:
                con.executemany("INSERT OR REPLACE INTO equity VALUES (?,?,?,?)", equity)
            con.execute("UPDATE jobs SET bars_done=?, sim_t=?, balance=?, equity=?, n_trades=?, updated_at=?"
                        " WHERE id=?",
                        (i + 1, _sec(self.bars.t[i]), float(bt.balance), float(bt.equity), len(bt.trades),
                         now_iso(), self.job_id))
        if self.delay:
            time.sleep(self.delay)

    def finish(self, bt, m: dict) -> None:
        pf = m["profit_factor"] if np.isfinite(m["profit_factor"]) else None
        with Tx(self.con):
            self.con.execute(
                "UPDATE jobs SET status='done', finished_at=?, updated_at=?, metrics_json=?, sharpe=?, cagr=?,"
                " max_dd=?, profit_factor=?, win_rate=?, trades_per_year=?, net_pips_per_year=?,"
                " gross_pips_per_year=?, cost_pips_per_year=?, gross_sharpe=?, final_equity=?, pnl=?, ruined=?,"
                " hunted=?, balance=?, equity=?, n_trades=? WHERE id=?",
                (now_iso(), now_iso(), dumps(m), m["sharpe"], m["cagr"], m["max_dd"], pf, m["win_rate"],
                 m["trades_per_year"], m["net_pips_per_year"], m["gross_pips_per_year"], m["cost_pips_per_year"],
                 m["gross_sharpe"], m["final_equity"], m["final_equity"] - bt.cfg.initial_balance,
                 int(m["ruined"]), m["hunted"], bt.balance, bt.equity, m["n_trades"], self.job_id))


# ---- ケース -----------------------------------------------------------------

def upsert_case_def(con: sqlite3.Connection, *, kind: str, id_: str, name: str, path: str, definition: dict) -> str:
    """定義のスナップショットを書き、ハッシュを返す。中身が同じなら updated_at は変えない。"""
    h = params_hash(definition)
    con.execute("INSERT INTO case_defs(id, kind, name, path, definition_json, hash, updated_at) VALUES (?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, name=excluded.name, path=excluded.path,"
                " definition_json=excluded.definition_json, hash=excluded.hash,"
                " updated_at=CASE WHEN case_defs.hash = excluded.hash THEN case_defs.updated_at"
                " ELSE excluded.updated_at END",
                (id_, kind, name, path, dumps(definition), h, now_iso()))
    return h


def start_case_run(con: sqlite3.Connection, *, test_id: str, behavior_id: str, test_hash: str, behavior_hash: str,
                   code_sha: str | None) -> int:
    with Tx(con):
        return con.execute(
            "INSERT INTO case_runs(test_id, behavior_id, test_hash, behavior_hash, code_sha, status, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (test_id, behavior_id, test_hash, behavior_hash, code_sha, "running", now_iso())).lastrowid


def update_case_summary(con: sqlite3.Connection, case_run_id: int, summary: dict) -> None:
    """記録済みの実行の要約だけを書き直す(判定の基準を変えたとき。相場と売買は回し直さない)。"""
    with Tx(con):
        con.execute("UPDATE case_runs SET summary_json=? WHERE id=?", (dumps(summary), case_run_id))


def finish_case_run(con: sqlite3.Connection, case_run_id: int, *, run_id: int, status: str, summary: dict) -> None:
    with Tx(con):
        con.execute("UPDATE case_runs SET run_id=?, status=?, finished_at=?, summary_json=? WHERE id=?",
                    (run_id, status, now_iso(), dumps(summary), case_run_id))
