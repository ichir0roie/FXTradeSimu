"""結果パック。コミットする結果は 1 実行 = 1 ファイル、相場パターン 1 つ = 1 ファイルに分ける。

    data/results/<run_key>.sqlite    1 実行の runs / paths / jobs / equity / trades / case_runs / case_defs
    data/patterns/<data_key>.sqlite  1 相場パターンの pattern_data / candles / market_events

統合 DB(data/fxsim.sqlite)はコミットしない。パックから作り直す(`fxsim db build`)。既定の DB を開くとき
(init_db)は、まだ取り込んでいないパックを自動で取り込む(sync_packs)。

- パックの中のスキーマは統合 DB と同じ(schema.SCHEMA)。番号(id)はパックの中だけのもので、取り込むときに振り直す
- 実行を指すときは、全環境で一意な runs.run_key を使う(統合 DB の番号は環境と取り込みの順で変わる)
- パックは一度書いたら変えない(compact で資産曲線を間引いたときだけ書き直す)。git の履歴に積まれるのは 1 回分
"""
from __future__ import annotations

import os
import re
import secrets
import sqlite3
from pathlib import Path

from ..paths import DEFAULT_DB, PATTERNS_DIR, RESULTS_DIR
from .db import Tx
from .schema import SCHEMA

RUN_TABLES = ("runs", "paths", "jobs", "equity", "trades", "case_runs", "case_defs")


# ---- run_key ----------------------------------------------------------------

def make_run_key(proposal: str | None, kind: str, created_at: str, suffix: str | None = None) -> str:
    """<提案 ID か種類>-<作成時刻 YYYYmmddTHHMMSS>-<6 桁>。suffix を省くと乱数(環境をまたいでぶつからない)。"""
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", proposal or kind or "run").strip("_") or "run"
    stamp = re.sub(r"[^0-9T]", "", created_at or "")[:15]
    return f"{label}-{stamp}-{suffix or secrets.token_hex(3)}"


def ensure_run_keys(con: sqlite3.Connection) -> None:
    """run_key の無い実行(パック導入前の DB)に、作成時刻と番号から決まるキーを付ける。"""
    rows = con.execute("SELECT id, proposal, kind, created_at FROM runs WHERE run_key IS NULL").fetchall()
    if not rows:
        return
    with Tx(con):
        for r in rows:
            con.execute("UPDATE runs SET run_key=? WHERE id=?",
                        (make_run_key(r["proposal"], r["kind"], r["created_at"], f"r{r['id']:02d}"), r["id"]))


def resolve_run(con: sqlite3.Connection, ref: str | int | None) -> int | None:
    """番号か run_key(前方一致も可)から、統合 DB の番号を返す。"""
    if ref is None:
        return None
    s = str(ref)
    if s.isdigit():
        return int(s)
    row = con.execute("SELECT id FROM runs WHERE run_key = ?", (s,)).fetchone()
    if row is None:
        rows = con.execute("SELECT id FROM runs WHERE run_key LIKE ? ORDER BY id", (s + "%",)).fetchall()
        if len(rows) != 1:
            raise KeyError(f"run '{s}' が見つからないか、複数に当たります")
        row = rows[0]
    return int(row["id"])


# ---- 共通 -------------------------------------------------------------------

def _db_file(con: sqlite3.Connection) -> str:
    for r in con.execute("PRAGMA database_list").fetchall():
        if r[1] == "main":
            return r[2]
    raise RuntimeError("main database not found")


def _cols(con: sqlite3.Connection, schema: str, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA {schema}.table_info({table})").fetchall()]


def _common(con: sqlite3.Connection, table: str, a: str, b: str, drop: tuple[str, ...] = ()) -> list[str]:
    cb = set(_cols(con, b, table))
    return [c for c in _cols(con, a, table) if c in cb and c not in drop]


def _new_pack(path: Path) -> sqlite3.Connection:
    """パック用の接続(WAL にしない。1 ファイルで完結させる)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=DELETE")
    con.executescript(SCHEMA)
    return con


def _finish_pack(con: sqlite3.Connection, tmp: Path, out: Path) -> None:
    con.execute("VACUUM")
    con.close()
    os.replace(tmp, out)


# ---- 書き出し ----------------------------------------------------------------

def export_run(con: sqlite3.Connection, run_id: int, results_dir: Path | str = RESULTS_DIR,
               patterns_dir: Path | str = PATTERNS_DIR) -> Path:
    """1 実行をパックに書き出し、使った相場パターンも(まだ無ければ)書き出す。書いたパックのパスを返す。"""
    ensure_run_keys(con)
    key = con.execute("SELECT run_key FROM runs WHERE id=?", (run_id,)).fetchone()["run_key"]
    results_dir = Path(results_dir)
    out = results_dir / f"{key}.sqlite"
    tmp = results_dir / f".{key}.sqlite.tmp"
    pk = _new_pack(tmp)
    pk.execute("ATTACH DATABASE ? AS src", (_db_file(con),))
    pk.execute("BEGIN")
    where = {
        "runs": ("id = ?", (run_id,)),
        "paths": ("run_id = ?", (run_id,)),
        "jobs": ("run_id = ?", (run_id,)),
        "equity": ("job_id IN (SELECT id FROM src.jobs WHERE run_id = ?)", (run_id,)),
        "trades": ("job_id IN (SELECT id FROM src.jobs WHERE run_id = ?)", (run_id,)),
        "case_runs": ("run_id = ?", (run_id,)),
        "case_defs": ("id IN (SELECT test_id FROM src.case_runs WHERE run_id = ?"
                      " UNION SELECT behavior_id FROM src.case_runs WHERE run_id = ?)", (run_id, run_id)),
    }
    for t in RUN_TABLES:
        cols = ", ".join(_common(pk, t, "main", "src"))
        cond, args = where[t]
        pk.execute(f"INSERT INTO main.{t} ({cols}) SELECT {cols} FROM src.{t} WHERE {cond}", args)
    pk.execute("COMMIT")
    keys = [r[0] for r in pk.execute("SELECT DISTINCT data_key FROM main.paths WHERE data_key IS NOT NULL")]
    pk.execute("DETACH DATABASE src")
    _finish_pack(pk, tmp, out)
    for k in keys:
        export_pattern(con, k, patterns_dir)
    return out


def export_pattern(con: sqlite3.Connection, data_key: str, patterns_dir: Path | str = PATTERNS_DIR) -> Path | None:
    """相場パターンを書き出す。既にあり、イベントの数も同じ以上なら何もしない。"""
    row = con.execute("SELECT n_events FROM pattern_data WHERE data_key=?", (data_key,)).fetchone()
    if row is None:
        return None
    patterns_dir = Path(patterns_dir)
    out = patterns_dir / f"{data_key}.sqlite"
    if out.exists() and _pattern_events(out) >= (row["n_events"] or 0):
        return out
    tmp = patterns_dir / f".{data_key}.sqlite.tmp"
    pk = _new_pack(tmp)
    pk.execute("ATTACH DATABASE ? AS src", (_db_file(con),))
    pk.execute("BEGIN")
    for t in ("pattern_data", "candles", "market_events"):
        cols = ", ".join(_common(pk, t, "main", "src"))
        pk.execute(f"INSERT INTO main.{t} ({cols}) SELECT {cols} FROM src.{t} WHERE data_key = ?", (data_key,))
    pk.execute("COMMIT")
    pk.execute("DETACH DATABASE src")
    _finish_pack(pk, tmp, out)
    return out


def _pattern_events(path: Path) -> int:
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = c.execute("SELECT n_events FROM pattern_data LIMIT 1").fetchone()
        return int(row[0] or 0) if row else -1
    finally:
        c.close()


# ---- 取り込み ----------------------------------------------------------------

def import_pack(con: sqlite3.Connection, path: Path | str, patterns_dir: Path | str = PATTERNS_DIR) -> int | None:
    """パックを統合 DB に取り込み、新しい番号を返す。同じ run_key が既にあれば何もしない(None)。"""
    path = Path(path)
    con.execute("ATTACH DATABASE ? AS p", (str(path),))
    try:
        r = con.execute("SELECT * FROM p.runs").fetchone()
        if r is None or con.execute("SELECT 1 FROM main.runs WHERE run_key=?", (r["run_key"],)).fetchone():
            return None
        with Tx(con):
            run_id = _copy_rows(con, "runs", "SELECT * FROM p.runs", {})[r["id"]]
            path_map = _copy_rows(con, "paths", "SELECT * FROM p.paths", {"run_id": {r["id"]: run_id}})
            job_map = _copy_rows(con, "jobs", "SELECT * FROM p.jobs",
                                 {"run_id": {r["id"]: run_id}, "path_id": path_map})
            con.execute("CREATE TEMP TABLE IF NOT EXISTS _job_map (old INTEGER PRIMARY KEY, new INTEGER)")
            con.execute("DELETE FROM temp._job_map")
            con.executemany("INSERT INTO temp._job_map VALUES (?, ?)", job_map.items())
            for t in ("equity", "trades"):
                cols = _common(con, t, "main", "p", drop=("job_id",))
                sel = ", ".join(f"x.{c}" for c in cols)
                con.execute(f"INSERT INTO main.{t} (job_id, {', '.join(cols)}) SELECT m.new, {sel} "
                            f"FROM p.{t} x JOIN temp._job_map m ON m.old = x.job_id")
            _copy_rows(con, "case_runs", "SELECT * FROM p.case_runs", {"run_id": {r["id"]: run_id}})
            for d in con.execute("SELECT * FROM p.case_defs").fetchall():
                cur = con.execute("SELECT updated_at FROM main.case_defs WHERE id=?", (d["id"],)).fetchone()
                if cur is None or (cur["updated_at"] or "") < (d["updated_at"] or ""):
                    cols = _common(con, "case_defs", "main", "p")
                    con.execute(f"INSERT OR REPLACE INTO main.case_defs ({', '.join(cols)}) "
                                f"VALUES ({', '.join('?' * len(cols))})", [d[c] for c in cols])
            keys = [k[0] for k in con.execute("SELECT DISTINCT data_key FROM p.paths WHERE data_key IS NOT NULL")]
    finally:
        con.execute("DETACH DATABASE p")
    for k in keys:
        import_pattern(con, k, patterns_dir)
    return run_id


def _copy_rows(con: sqlite3.Connection, table: str, select: str, remap: dict[str, dict]) -> dict[int, int]:
    """行を 1 つずつ写し、id の対応(旧 → 新)を返す。remap の列は対応表で置き換える。"""
    cols = _common(con, table, "main", "p", drop=("id",))
    sql = f"INSERT INTO main.{table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
    out = {}
    for row in con.execute(select).fetchall():
        vals = [remap[c][row[c]] if c in remap else row[c] for c in cols]
        out[row["id"]] = con.execute(sql, vals).lastrowid
    return out


def import_pattern(con: sqlite3.Connection, data_key: str, patterns_dir: Path | str = PATTERNS_DIR) -> bool:
    """相場パターンを取り込む。統合 DB に無いか、パックの方がイベントが多いときだけ。"""
    path = Path(patterns_dir) / f"{data_key}.sqlite"
    if not path.exists():
        return False
    cur = con.execute("SELECT n_events FROM pattern_data WHERE data_key=?", (data_key,)).fetchone()
    if cur is not None and (cur["n_events"] or 0) >= _pattern_events(path):
        return False
    con.execute("ATTACH DATABASE ? AS q", (str(path),))
    try:
        with Tx(con):
            for t in ("pattern_data", "candles", "market_events"):
                con.execute(f"DELETE FROM main.{t} WHERE data_key = ?", (data_key,))
                cols = ", ".join(_common(con, t, "main", "q"))
                con.execute(f"INSERT INTO main.{t} ({cols}) SELECT {cols} FROM q.{t}")
    finally:
        con.execute("DETACH DATABASE q")
    return True


def list_packs(results_dir: Path | str = RESULTS_DIR) -> list[tuple[str, str, Path]]:
    """(作成時刻, run_key, パス) を作成時刻の順に。取り込みの順(統合 DB の番号)をどの環境でも同じにするため。"""
    out = []
    for p in Path(results_dir).glob("*.sqlite"):
        c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        try:
            row = c.execute("SELECT created_at, run_key FROM runs LIMIT 1").fetchone()
        finally:
            c.close()
        if row:
            out.append((row[0] or "", row[1], p))
    return sorted(out)


def sync_packs(con: sqlite3.Connection, results_dir: Path | str = RESULTS_DIR,
               patterns_dir: Path | str = PATTERNS_DIR, log=None) -> list[int]:
    """まだ取り込んでいないパックを取り込む。取り込んだ実行の番号を返す。"""
    have = {r[0] for r in con.execute("SELECT run_key FROM runs WHERE run_key IS NOT NULL")}
    new = []
    for _, key, p in list_packs(results_dir):
        if key in have:
            continue
        rid = import_pack(con, p, patterns_dir)
        if rid is not None:
            new.append(rid)
            if log:
                log(f"imported {p.name} -> run #{rid}")
    return new


def is_default_db(path: str | Path | None) -> bool:
    return Path(path or DEFAULT_DB).resolve() == DEFAULT_DB.resolve()


def auto_export(con: sqlite3.Connection, db_path: str | Path | None, run_id: int, log=print) -> Path | None:
    """既定の DB に書いた実行(demo 以外)を、終わったらパックに書き出す。"""
    if not is_default_db(db_path):
        return None
    row = con.execute("SELECT kind, status FROM runs WHERE id=?", (run_id,)).fetchone()
    if row is None or row["kind"] == "demo" or row["status"] == "running":
        return None
    out = export_run(con, run_id)
    log(f"run #{run_id} -> {out.relative_to(out.parents[2]) if len(out.parents) > 2 else out}")
    return out
