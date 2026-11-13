"""相場の DB(data/market.sqlite)。MT5 などから取った足を、時間足ごとのテーブルに溜める。

テーブル
    symbols      出どころ x 銘柄の桁数・ポイント・サーバー時刻の決まり
    rates_<tf>   足。rates_m1 / rates_m5 / rates_m15 / rates_m30 / rates_h1 / rates_h4 / rates_d1。
                 1 行 = 出どころ x 銘柄 x 足の始まり(time、UTC の秒)。価格は bid、spread はポイント。
                 server_time はブローカーのサーバー時刻のまま(時刻の決まりを直したときに引き直せるように)
    fetches      取ってきた記録(何本取り、何本増え、前回の分とつながったか)

出どころ(source)は MT5 のサーバー名(例: OANDA-Japan MT5 Demo)。同じ銘柄でも業者ごとに値とスプレッドが違う。
実験の DB(fxsim.sqlite)とは別のファイル。このモジュールは data に依存しない(data/real.py が db: のときだけ読みに来る)。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from .. import paths
from .db import Tx, connect, now_iso

# 時間足(分) -> テーブル名の後ろ。MT5 の TIMEFRAME_* と同じ並び
TIMEFRAMES = {1: "m1", 5: "m5", 15: "m15", 30: "m30", 60: "h1", 240: "h4", 1440: "d1"}
PRICE_COLUMNS = ("open", "high", "low", "close")
COLUMNS = ("time", "server_time", *PRICE_COLUMNS, "tick_volume", "spread", "real_volume")


def table(timeframe: int) -> str:
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"no table for a {timeframe}-minute timeframe (supported: {sorted(TIMEFRAMES)})")
    return f"rates_{TIMEFRAMES[timeframe]}"


def parse_timeframe(text: str | int) -> int:
    """'M1' / 'h1' / 'D1' / '60' -> 分。"""
    s = str(text).strip().lower()
    for minutes, name in TIMEFRAMES.items():
        if s in (name, str(minutes)):
            return minutes
    raise ValueError(f"unknown timeframe {text!r} (use one of {[n.upper() for n in TIMEFRAMES.values()]})")


def _rates_table(name: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {name} (
    source      TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    time        INTEGER NOT NULL,   -- 足の始まり(UTC、秒)
    server_time INTEGER NOT NULL,   -- 足の始まり(ブローカーのサーバー時刻、秒)
    open        REAL    NOT NULL,   -- bid
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    tick_volume INTEGER NOT NULL,
    spread      INTEGER NOT NULL,   -- ポイント(symbols.point を掛けると価格)
    real_volume INTEGER NOT NULL,
    PRIMARY KEY (source, symbol, time)
) WITHOUT ROWID;"""


SCHEMA = "\n".join([
    """
CREATE TABLE IF NOT EXISTS symbols (
    source     TEXT    NOT NULL,
    symbol     TEXT    NOT NULL,
    digits     INTEGER NOT NULL,
    point      REAL    NOT NULL,
    server_tz  TEXT    NOT NULL,   -- サーバー時刻の決まり(例: America/New_York+7)
    updated_at TEXT    NOT NULL,
    PRIMARY KEY (source, symbol)
);
CREATE TABLE IF NOT EXISTS fetches (
    id         INTEGER PRIMARY KEY,
    fetched_at TEXT    NOT NULL,
    source     TEXT    NOT NULL,
    symbol     TEXT    NOT NULL,
    timeframe  INTEGER NOT NULL,
    server_tz  TEXT    NOT NULL,
    n_rows     INTEGER NOT NULL,   -- 取った本数
    n_new      INTEGER NOT NULL,   -- そのうち DB に無かった本数
    first_time INTEGER NOT NULL,
    last_time  INTEGER NOT NULL,
    connected  INTEGER            -- 前回までの分とつながったか(1/0)。初回は NULL
);""",
    *(_rates_table(f"rates_{n}") for n in TIMEFRAMES.values()),
])


def open_market_db(path: str | Path | None = None) -> sqlite3.Connection:
    """相場の DB に接続してテーブルを作る(既にあれば何もしない)。"""
    con = connect(path or paths.MARKET_DB)
    con.executescript(SCHEMA)
    return con


def upsert_symbol(con: sqlite3.Connection, source: str, symbol: str, digits: int, point: float, server_tz: str) -> None:
    with Tx(con):
        con.execute(
            "INSERT INTO symbols (source, symbol, digits, point, server_tz, updated_at) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(source, symbol) DO UPDATE SET digits=excluded.digits, point=excluded.point, "
            "server_tz=excluded.server_tz, updated_at=excluded.updated_at",
            (source, symbol, int(digits), float(point), server_tz, now_iso()))


def symbol_info(con: sqlite3.Connection, source: str, symbol: str) -> dict | None:
    r = con.execute("SELECT * FROM symbols WHERE source=? AND symbol=?", (source, symbol)).fetchone()
    return dict(r) if r else None


def write_rates(con: sqlite3.Connection, source: str, symbol: str, timeframe: int, rows: dict,
                server_tz: str) -> dict:
    """足を書き足す。同じ時刻の足は上書きする(業者が足を直したときに合わせる)。

    rows は COLUMNS の配列を持つ dict(time は UTC の秒)。戻り値は fetches に書いた記録。
    """
    tbl = table(timeframe)
    t = np.asarray(rows["time"], dtype=np.int64)
    if len(t) == 0:
        raise ValueError("no rates to write")
    if np.any(np.diff(t) <= 0):
        raise ValueError("rates must be sorted by time without duplicates")
    first, last = int(t[0]), int(t[-1])
    data = [tuple(int(v) if c not in PRICE_COLUMNS else float(v) for c, v in zip(COLUMNS, vals, strict=True))
            for vals in zip(*(np.asarray(rows[c]) for c in COLUMNS), strict=True)]
    cols = ", ".join(COLUMNS)
    updates = ", ".join(f"{c}=excluded.{c}" for c in COLUMNS if c != "time")
    with Tx(con):
        prev_last = con.execute(f"SELECT MAX(time) FROM {tbl} WHERE source=? AND symbol=?",
                                (source, symbol)).fetchone()[0]
        existing = con.execute(f"SELECT COUNT(*) FROM {tbl} WHERE source=? AND symbol=? AND time BETWEEN ? AND ?",
                               (source, symbol, first, last)).fetchone()[0]
        con.executemany(
            f"INSERT INTO {tbl} (source, symbol, {cols}) VALUES (?, ?, {', '.join('?' * len(COLUMNS))}) "
            f"ON CONFLICT(source, symbol, time) DO UPDATE SET {updates}",
            [(source, symbol, *d) for d in data])
        rec = {"fetched_at": now_iso(), "source": source, "symbol": symbol, "timeframe": int(timeframe),
               "server_tz": server_tz, "n_rows": len(data), "n_new": len(data) - int(existing),
               "first_time": first, "last_time": last,
               # 取った最初の足が、前回までの最後の足より後なら、その間は取れていない(MT5 が遡れる範囲を越えた)
               "connected": None if prev_last is None else int(first <= int(prev_last))}
        con.execute(f"INSERT INTO fetches ({', '.join(rec)}) VALUES ({', '.join('?' * len(rec))})", tuple(rec.values()))
    return rec


def sources(con: sqlite3.Connection, symbol: str, timeframe: int) -> list[str]:
    return [r[0] for r in con.execute(f"SELECT DISTINCT source FROM {table(timeframe)} WHERE symbol=? ORDER BY source",
                                      (symbol,))]


def read_rates(con: sqlite3.Connection, source: str, symbol: str, timeframe: int,
               start: int | None = None, end: int | None = None) -> dict[str, np.ndarray]:
    """足を古い順に。start / end は足の始まり(UTC の秒、両端を含む)。"""
    q = f"SELECT {', '.join(COLUMNS)} FROM {table(timeframe)} WHERE source=? AND symbol=?"
    args: list = [source, symbol]
    if start is not None:
        q += " AND time >= ?"
        args.append(int(start))
    if end is not None:
        q += " AND time <= ?"
        args.append(int(end))
    rows = con.execute(q + " ORDER BY time", args).fetchall()
    out = {}
    for k, c in enumerate(COLUMNS):
        out[c] = np.array([r[k] for r in rows], dtype=float if c in PRICE_COLUMNS else np.int64)
    return out


def coverage(con: sqlite3.Connection) -> list[dict]:
    """溜まっている足の一覧(出どころ x 銘柄 x 時間足ごとの本数と期間、最後に取った時刻)。"""
    out = []
    for minutes in TIMEFRAMES:
        q = (f"SELECT source, symbol, COUNT(*) AS n, MIN(time) AS first_time, MAX(time) AS last_time "
             f"FROM {table(minutes)} GROUP BY source, symbol ORDER BY source, symbol")
        for r in con.execute(q):
            f = con.execute("SELECT MAX(fetched_at) FROM fetches WHERE source=? AND symbol=? AND timeframe=?",
                            (r["source"], r["symbol"], minutes)).fetchone()[0]
            out.append({**dict(r), "timeframe": minutes, "last_fetched_at": f})
    return out
