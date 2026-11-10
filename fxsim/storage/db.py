"""SQLite への接続とトランザクション。書き込みは writer.py、読み出しは queries.py。"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..paths import DEFAULT_DB
from .schema import SCHEMA


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """WAL モードで開く。Python が書いている間も、画面(web/)は読み取り専用で読める。"""
    path = Path(path or DEFAULT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=120, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=120000")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db(path: str | Path | None = None) -> sqlite3.Connection:
    """接続してスキーマを作る(既にあれば何もしない。ビューは作り直す)。"""
    con = connect(path)
    con.executescript(SCHEMA)
    return con


class Tx:
    """BEGIN IMMEDIATE ... COMMIT。書き込みが他プロセスと重なっても待って再試行する。"""

    def __init__(self, con: sqlite3.Connection):
        self.con = con

    def __enter__(self):
        for attempt in range(50):
            try:
                self.con.execute("BEGIN IMMEDIATE")
                return self.con
            except sqlite3.OperationalError as e:  # pragma: no cover - 競合時のみ
                if "locked" not in str(e) and "busy" not in str(e):
                    raise
                time.sleep(0.05 * (attempt + 1))
        raise sqlite3.OperationalError("could not acquire write lock")

    def __exit__(self, exc_type, exc, tb):
        self.con.execute("ROLLBACK" if exc_type else "COMMIT")
        return False


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dumps(obj) -> str:
    """numpy の値も書ける json.dumps。"""
    return json.dumps(obj, ensure_ascii=False, default=_json_default, allow_nan=True)


def checkpoint(con: sqlite3.Connection) -> None:
    """WAL をメインのファイルへ書き戻す(コミット前に呼ぶ)。"""
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
