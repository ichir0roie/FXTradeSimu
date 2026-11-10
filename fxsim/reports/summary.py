"""実験の概要。DB(手元だけ、コミットしない)とは別に、成績の要約を docs/results/<run_key>/ に書き出してコミットする。

    run.json   実験の設定・コードの版・テストケースの記録と要約(検証の判定を含む)
    paths.csv  相場パターンごとの性質(facts)と現実らしさ
    jobs.csv   バックテストごとの成績と費用の内訳(cost_*)、決済の理由と年ごとの損益(JSON)

資産曲線・取引の明細・ローソク足は持たない。結果はシードで決まるので、要るときは同じ設定で回し直す。
パターン単位の統計(平均・中央値・t 値など)は jobs.csv から取り直せる(read_summary)。
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..paths import DEFAULT_DB, SUMMARIES_DIR
from ..storage import queries

JOB_COLUMNS = ["job_id", "path_id", "market_label", "seed", "broker", "strategy_label", "strategy",
               "strategy_version", "params_hash", "timeframe", "status", "note", "n_trades", "pnl"]
PATH_COLUMNS = ["path_id", "market_label", "market", "market_version", "params_hash", "seed", "detail", "status",
                "start", "end", "realism_score", "n_fail"]


def _round(v):
    """JSON の中の小数も有効数字 10 桁に(円の損益で 1 銭まで残る)。"""
    if isinstance(v, float) and math.isfinite(v):
        return float(f"{v:.10g}")
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_round(x) for x in v]
    return v


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.10g}"
    if isinstance(v, (dict, list)):
        return json.dumps(_round(v), ensure_ascii=False, separators=(",", ":"), default=str)
    return v


def _date(t) -> str:
    return datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%d") if t is not None else ""


def _flatten_metrics(m: dict) -> dict:
    """数値はそのまま、costs は cost_<内訳> に、そのほかの dict(exit_reasons・yearly_pnl)は JSON のまま。"""
    out = {}
    for k, v in m.items():
        if k == "costs" and isinstance(v, dict):
            out.update({f"cost_{ck}": cv for ck, cv in v.items()})
        else:
            out[k] = v
    return out


def _write_csv(path: Path, fixed: list[str], rows: list[dict]) -> None:
    extra: list[str] = []
    for r in rows:
        extra += [k for k in r if k not in fixed and k not in extra]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fixed + extra)
        for r in rows:
            w.writerow([_cell(r.get(k)) for k in fixed + extra])


def export_summary(con: sqlite3.Connection, run_id: int, out_dir: Path | str = SUMMARIES_DIR) -> Path:
    """1 実行の概要を out_dir/<run_key>/ に書き出し、そのフォルダを返す(既にあれば書き直す)。"""
    run = queries.get_run(con, run_id)
    if run is None:
        raise KeyError(f"run #{run_id} がありません")
    folder = Path(out_dir) / run["run_key"]
    tmp = folder.with_name(f".{folder.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    meta = {k: run[k] for k in ("run_key", "name", "kind", "proposal", "status", "created_at", "finished_at",
                                "git_sha", "years", "notes", "elapsed_sec")}
    meta["seeds"] = json.loads(run["seeds_json"]) if run["seeds_json"] else None
    meta["config"] = json.loads(run["config_json"]) if run["config_json"] else None
    meta["case_runs"] = [{**{k: v for k, v in c.items() if k != "summary_json"},
                          "summary": json.loads(c["summary_json"]) if c["summary_json"] else None}
                         for c in queries.case_runs_of(con, run_id)]
    (tmp / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str) + "\n",
                                  encoding="utf-8")

    paths = []
    for p in queries.summary_paths(con, run_id):
        facts = json.loads(p.pop("facts_json") or "{}")
        row = {**p, "start": _date(p.pop("t_start")), "end": _date(p.pop("t_end"))}
        row.update({k: v for k, v in facts.items() if not k.startswith("_")})
        paths.append(row)
    _write_csv(tmp / "paths.csv", PATH_COLUMNS, paths)

    jobs = []
    for j in queries.summary_jobs(con, run_id):
        m = json.loads(j.pop("metrics_json") or "{}")
        jobs.append({**j, **_flatten_metrics(m)})
    _write_csv(tmp / "jobs.csv", JOB_COLUMNS, jobs)

    if folder.exists():
        shutil.rmtree(folder)
    tmp.rename(folder)
    return folder


def is_default_db(path: str | Path | None) -> bool:
    return Path(path or DEFAULT_DB).resolve() == DEFAULT_DB.resolve()


def auto_summary(con: sqlite3.Connection, db_path: str | Path | None, run_id: int, log=print) -> Path | None:
    """既定の DB に書いた実行(demo 以外)の概要を、終わったら docs/results に書き出す。"""
    if not is_default_db(db_path):
        return None
    row = queries.get_run(con, run_id)
    if row is None or row["kind"] == "demo" or row["status"] == "running":
        return None
    out = export_summary(con, run_id)
    log(f"run #{run_id} -> docs/results/{out.name}/")
    return out


# ---- 読み出し(分析用) ------------------------------------------------------

TEXT_COLUMNS = {"market_label", "market", "market_version", "params_hash", "broker", "strategy_label", "strategy",
                "strategy_version", "status", "note", "start", "end"}


def _parse(k: str, v: str):
    if v == "":
        return None
    if k in TEXT_COLUMNS:
        return v
    if v[0] in "[{":
        return json.loads(v)
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() and "inf" not in v else f


def read_summary(run_key: str, root: Path | str = SUMMARIES_DIR) -> dict:
    """docs/results/<run_key>/ を読む。{"run": dict, "paths": [dict], "jobs": [dict]}。前方一致も可。"""
    root = Path(root)
    folder = root / run_key
    if not folder.is_dir():
        hits = sorted(p for p in root.glob(f"{run_key}*") if p.is_dir())
        if len(hits) != 1:
            raise KeyError(f"概要 '{run_key}' が見つからないか、複数に当たります")
        folder = hits[0]
    out: dict = {"run": json.loads((folder / "run.json").read_text(encoding="utf-8"))}
    for name in ("paths", "jobs"):
        with (folder / f"{name}.csv").open(encoding="utf-8") as f:
            out[name] = [{k: _parse(k, v) for k, v in r.items()} for r in csv.DictReader(f)]
    return out
