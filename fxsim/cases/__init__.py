"""動作ケースとテストケース。

動作ケース(cases/behaviors/*.yaml): どう取引するか
    id: B001
    name: 説明
    strategy: ma_cross            # 戦略(登録名)
    params: {fast: 20, slow: 100} # 戦略のパラメータ
    timeframe: 60                 # 判断に使う足(分)
    schedule:                     # いつ判断し、いつ取引してよいか(fxsim/backtest/schedule.py)
      decision_every: 1           #   何本ごとに判断するか
      hours: [["07:00", "16:00"]] #   取引してよい時間帯(省略 = 終日)
      tz: UTC                     #   UTC / JST / LON / NY
      days: [mon, tue, wed, thu, fri]
      outside: hold               #   時間外の持ち高: hold / close
    broker: malicious             # プリセット名、または {preset: ..., overrides: {...}}
    engine: {units: 10000}

テストケース(cases/tests/*.yaml): ある動作ケースを、どの相場パターンで、どれだけの期間試すか
    id: T001
    name: 説明
    behavior: B001
    period: {years: 5}            # または {start: "2019-01-01", end: "2023-12-31"}
    patterns:
      - {market: real}                              # 実データそのもの
      - {market: real_bridge, seeds: [1, 2, 3]}     # 実相場の範囲でランダム化したパターン
      - {market: malicious, seeds: [1, 2]}          # 完全なランダム相場のパターン
      - {market: malicious, label: mal_strong, params: {stop_hunt: {prob: 0.12}}, seeds: [1]}
    detail_seeds: [1]             # ローソク足・取引まで保存するシード(省略 = 各パターンの最初)

``python -m fxsim cases run --all`` で全テストケースを回す。定義(動作ケース・テストケース)と
コードの版が前回と同じなら飛ばす(--force で回し直す)。結果は runs / paths / jobs に入り、
case_runs に「どの定義の版を、どの run で回したか」と損益の要約が残る。
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from .. import catalog  # noqa: F401
from ..backtest.broker import BROKER_PRESETS
from ..backtest.schedule import TradingSchedule
from ..experiments.config import ExperimentConfig
from ..experiments.stats import strategy_stats, summarize
from ..markets import MARKETS
from ..registry import params_hash
from ..storage import Tx, dumps, now_iso
from ..strategies import STRATEGIES

CASES_DIR = Path(__file__).resolve().parents[2] / "cases"


@dataclass
class BehaviorCase:
    id: str
    strategy: str
    name: str = ""
    params: dict = field(default_factory=dict)
    timeframe: int | None = None
    schedule: dict = field(default_factory=dict)
    broker: str | dict = "fair"
    engine: dict = field(default_factory=dict)
    path: str = ""

    def validate(self) -> None:
        STRATEGIES.get(self.strategy)
        TradingSchedule.from_dict(self.schedule)
        preset = self.broker if isinstance(self.broker, str) else self.broker.get("preset")
        if preset not in BROKER_PRESETS:
            raise ValueError(f"{self.id}: unknown broker preset {preset}")
        STRATEGIES.create(self.strategy, self.strategy_params())

    def strategy_params(self) -> dict:
        p = dict(self.params)
        if self.timeframe:
            p["timeframe"] = int(self.timeframe)
        return p

    def definition(self) -> dict:
        return {"id": self.id, "name": self.name, "strategy": self.strategy, "params": self.params,
                "timeframe": self.timeframe, "schedule": TradingSchedule.from_dict(self.schedule).to_dict(),
                "broker": self.broker, "engine": self.engine}


@dataclass
class Pattern:
    market: str
    label: str | None = None
    params: dict = field(default_factory=dict)
    seeds: list[int] = field(default_factory=lambda: [1])


@dataclass
class TestCase:
    id: str
    behavior: str
    patterns: list[Pattern]
    name: str = ""
    period: dict = field(default_factory=lambda: {"years": 3})
    detail_seeds: list[int] | None = None
    workers: int = 0
    path: str = ""

    def definition(self) -> dict:
        return {"id": self.id, "name": self.name, "behavior": self.behavior, "period": self.period,
                "patterns": [p.__dict__ for p in self.patterns], "detail_seeds": self.detail_seeds}

    def years(self) -> float:
        if "years" in self.period:
            return float(self.period["years"])
        a = date.fromisoformat(str(self.period["start"]))
        b = date.fromisoformat(str(self.period["end"]))
        return (b - a).days / 365.25


def _load_dir(sub: str) -> list[tuple[dict, Path]]:
    out = []
    for p in sorted((CASES_DIR / sub).glob("*.y*ml")):
        with open(p, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        if d:
            out.append((d, p))
    return out


def load_cases(root: Path | None = None) -> tuple[dict[str, BehaviorCase], dict[str, TestCase]]:
    global CASES_DIR
    if root is not None:
        CASES_DIR = Path(root)
    behaviors: dict[str, BehaviorCase] = {}
    for d, p in _load_dir("behaviors"):
        b = BehaviorCase(**d, path=str(p.relative_to(CASES_DIR.parent)))
        if b.id in behaviors:
            raise ValueError(f"duplicate behavior id {b.id} ({p})")
        b.validate()
        behaviors[b.id] = b
    tests: dict[str, TestCase] = {}
    for d, p in _load_dir("tests"):
        d = dict(d)
        pats = []
        for x in d.pop("patterns"):
            seeds = x.get("seeds", [1])
            if isinstance(seeds, str):
                a, b = seeds.split("-")
                seeds = list(range(int(a), int(b) + 1))
            pats.append(Pattern(x["market"], x.get("label"), x.get("params") or {}, list(seeds)))
        t = TestCase(**d, patterns=pats, path=str(p.relative_to(CASES_DIR.parent)))
        if t.id in tests:
            raise ValueError(f"duplicate test id {t.id} ({p})")
        if t.behavior not in behaviors:
            raise ValueError(f"{t.id}: unknown behavior {t.behavior}")
        for pat in pats:
            MARKETS.get(pat.market)
        if "years" not in t.period and not ("start" in t.period and "end" in t.period):
            raise ValueError(f"{t.id}: period needs years or start/end")
        tests[t.id] = t
    return behaviors, tests


def to_experiment(t: TestCase, b: BehaviorCase) -> ExperimentConfig:
    """テストケースを実験の設定に機械的に直す。"""
    markets = []
    labels = set()
    for p in t.patterns:
        params = dict(p.params)
        model = MARKETS.create(p.market, params)
        if "start" in t.period:
            if model.source is not None:     # 実データ系は期間を切り出す
                params["source"] = {**params.get("source", {}), "start": str(t.period["start"]),
                                    "end": str(t.period["end"])}
            else:                             # 合成相場は開始日を合わせる
                params["base"] = {**params.get("base", {}), "start": str(t.period["start"])}
        label = p.label or p.market
        if label in labels:
            raise ValueError(f"{t.id}: pattern label '{label}' is duplicated; set label")
        labels.add(label)
        markets.append({"model": p.market, "label": label, "params": params, "seeds": p.seeds})
    broker = b.broker if isinstance(b.broker, dict) else {"preset": b.broker}
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
        "storage": {"candle_timeframe": 240, "equity_detail_timeframe": 1440, "equity_timeframe": 1440,
                    "trades": "detail"},
        "brokers": [{"preset": broker["preset"], "label": broker.get("label", broker["preset"]),
                     "overrides": broker.get("overrides", {})}],
        "markets": markets,
        "strategies": [{"name": b.strategy, "label": b.id, "params": b.strategy_params(), "schedule": b.schedule}],
    })


def summarize_run(con: sqlite3.Connection, run_id: int) -> dict:
    stats = strategy_stats(con, run_id)
    out = {"by_market": {}, "all": None}
    for s in stats:
        item = {k: s[k] for k in ("pnl", "net_pips_per_year", "sharpe", "max_dd", "trades_per_year")}
        if s["market"] == "ALL":
            out["all"] = item
        else:
            out["by_market"][s["market"]] = item
    out["patterns"] = [p for s in stats if s["market"] == "ALL" for p in s["patterns"]]
    return out


def _upsert_def(con, kind: str, id_: str, name: str, path: str, d: dict) -> str:
    h = params_hash(d)
    con.execute("INSERT INTO case_defs(id, kind, name, path, definition_json, hash, updated_at) VALUES (?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, name=excluded.name, path=excluded.path,"
                " definition_json=excluded.definition_json, hash=excluded.hash,"
                " updated_at=CASE WHEN case_defs.hash = excluded.hash THEN case_defs.updated_at"
                " ELSE excluded.updated_at END",
                (id_, kind, name, path, dumps(d), h, now_iso()))
    return h


def run_cases(con: sqlite3.Connection, db_path: str, ids: list[str] | None = None, force: bool = False,
              log=print) -> list[int]:
    from ..experiments.runner import git_sha, run_experiment

    behaviors, tests = load_cases()
    sha = git_sha()
    with Tx(con):
        bh = {k: _upsert_def(con, "behavior", k, b.name, b.path, b.definition()) for k, b in behaviors.items()}
        th = {k: _upsert_def(con, "test", k, t.name, t.path, t.definition()) for k, t in tests.items()}
    targets = ids or sorted(tests)
    run_ids = []
    for tid in targets:
        if tid not in tests:
            raise KeyError(f"unknown test case {tid}. known: {sorted(tests)}")
        t = tests[tid]
        b = behaviors[t.behavior]
        prev = con.execute("SELECT run_id FROM case_runs WHERE test_id=? AND test_hash=? AND behavior_hash=?"
                           " AND code_sha=? AND status='done' ORDER BY id DESC LIMIT 1",
                           (tid, th[tid], bh[b.id], sha)).fetchone()
        if prev and not force:
            log(f"{tid}: 定義とコードが前回(run #{prev[0]})と同じなので飛ばします(--force で回し直し)")
            continue
        cfg = to_experiment(t, b)
        with Tx(con):
            case_run_id = con.execute(
                "INSERT INTO case_runs(test_id, behavior_id, test_hash, behavior_hash, code_sha, status, created_at)"
                " VALUES (?,?,?,?,?,?,?)", (tid, b.id, th[tid], bh[b.id], sha, "running", now_iso())).lastrowid
        log(f"{tid}: behavior {b.id} ({b.strategy}, tf={cfg.strategies[0].params.get('timeframe')}m) x "
            f"{sum(len(p.seeds) for p in t.patterns)} patterns, {cfg.years:.2f} years")
        run_id = run_experiment(cfg, db_path, log=log)
        status = con.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()[0]
        summary = summarize_run(con, run_id)
        with Tx(con):
            con.execute("UPDATE case_runs SET run_id=?, status=?, finished_at=?, summary_json=? WHERE id=?",
                        (run_id, status, now_iso(), dumps(summary), case_run_id))
        a = summary["all"]
        if a and a["pnl"].get("n"):
            p = a["pnl"]
            log(f"{tid}: run #{run_id} 損益 平均 {p['mean']:,.0f} / 中央 {p['median']:,.0f} / 合計 {p['total']:,.0f}"
                f" / プラス {p['positive_ratio'] * 100:.0f}% (n={p['n']})")
        run_ids.append(run_id)
    return run_ids


def list_cases(con: sqlite3.Connection) -> str:
    behaviors, tests = load_cases()
    lines = ["## 動作ケース", "| ID | 名前 | 戦略 | 足 | 判断の間隔 | 取引時間 | 時間外 | ブローカー |", "|---|---|---|---|---|---|---|---|"]
    for b in behaviors.values():
        s = TradingSchedule.from_dict(b.schedule)
        hours = ", ".join(f"{x}-{y}" for x, y in s.hours) + f" {s.tz}" if s.hours else "終日"
        tf = b.strategy_params().get("timeframe") or STRATEGIES.create(b.strategy).timeframe
        lines.append(f"| {b.id} | {b.name} | {b.strategy} | {tf}m | {s.decision_every} 本ごと | {hours} "
                     f"({','.join(s.days)}) | {s.outside} | {b.broker if isinstance(b.broker, str) else b.broker['preset']} |")
    lines += ["", "## テストケース", "| ID | 名前 | 動作 | 期間 | パターン | 最新 run | 損益 平均 | 中央 | 合計 | プラス |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for t in tests.values():
        row = con.execute("SELECT run_id, summary_json FROM case_runs WHERE test_id=? AND status='done'"
                          " ORDER BY id DESC LIMIT 1", (t.id,)).fetchone()
        period = f"{t.period['years']} 年" if "years" in t.period else f"{t.period['start']}〜{t.period['end']}"
        pats = ", ".join(f"{p.label or p.market}x{len(p.seeds)}" for p in t.patterns)
        if row and row[1]:
            import json

            a = (json.loads(row[1]).get("all") or {}).get("pnl") or {}
            cells = [f"#{row[0]}", f"{a.get('mean', 0):,.0f}", f"{a.get('median', 0):,.0f}",
                     f"{a.get('total', 0):,.0f}", f"{a.get('positive_ratio', 0) * 100:.0f}%"]
        else:
            cells = ["未実行", "-", "-", "-", "-"]
        lines.append(f"| {t.id} | {t.name} | {t.behavior} | {period} | {pats} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


__all__ = ["BehaviorCase", "Pattern", "TestCase", "list_cases", "load_cases", "run_cases", "summarize",
           "to_experiment"]
