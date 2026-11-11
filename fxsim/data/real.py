"""実際の相場データの読み込み。

- 汎用 CSV: 列 time, close(必須)と open, high, low(任意)。time は UTC。
  日付だけの行は、その日の NY クローズ(22:00 UTC)の値とみなす。
- HistData Generic ASCII(format="histdata"): ``YYYYMMDD HHMMSS;O;H;L;C;V``(EST 固定 = UTC-5)。
- 相場の DB(data/market.sqlite、`python -m fxsim fetch-mt5` で溜める): パスの代わりに
  ``db:<銘柄>/<時間足>``(例: ``db:USDJPY/M1``)。出どころが複数あるときは ``db:<出どころ>/<銘柄>/<時間足>``。
  DB の価格は bid なので、足ごとのスプレッドの半分を足して mid にする。
パスはリポジトリ直下からの相対でもよい。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..calendar import NS_HOUR, NS_MIN
from ..paths import resolve

DB_PREFIX = "db:"


@dataclass(frozen=True)
class RealSeries:
    times: np.ndarray        # int64 ns(足の終わりの時刻)
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    has_ohlc: bool
    step_minutes: int        # 典型的な間隔
    source: str
    spread: np.ndarray | None = None   # 足ごとのスプレッド(価格)。DB から読んだときだけ

    def __len__(self) -> int:
        return len(self.times)

    def window(self, start: str | None = None, end: str | None = None, years: float | None = None) -> RealSeries:
        t = self.times
        lo = 0 if start is None else int(np.searchsorted(t, np.datetime64(start, "ns").astype(np.int64)))
        hi = len(t) if end is None else int(np.searchsorted(t, np.datetime64(end, "ns").astype(np.int64), "right"))
        if years is not None and start is None:
            first = t[hi - 1] - int(years * 365.25 * 24 * NS_HOUR)
            lo = max(lo, int(np.searchsorted(t, first)))
        sl = slice(lo, hi)
        return RealSeries(t[sl], self.open[sl], self.high[sl], self.low[sl], self.close[sl], self.has_ohlc,
                          self.step_minutes, self.source, None if self.spread is None else self.spread[sl])


@lru_cache(maxsize=8)
def load_ohlc(path: str, fmt: str = "auto") -> RealSeries:
    if str(path).startswith(DB_PREFIX):
        return _load_db(str(path)[len(DB_PREFIX):])
    p = resolve(path)
    if fmt == "histdata" or (fmt == "auto" and p.suffix.lower() == ".csv" and _looks_histdata(p)):
        df = pd.read_csv(p, sep=";", header=None, names=["dt", "open", "high", "low", "close", "vol"])
        t = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S") + pd.Timedelta(hours=5)   # EST -> UTC
        t = t + pd.Timedelta(minutes=1)    # 足の開始時刻 -> 終わりの時刻
        has = True
    else:
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        df = df.rename(columns={v: k for k, v in cols.items()})
        raw = df["time"].astype(str)
        date_only = raw.str.len() <= 10
        t = pd.to_datetime(raw, utc=True).dt.tz_localize(None)
        t = t.where(~date_only, t + pd.Timedelta(hours=22))
        has = all(c in df.columns for c in ("open", "high", "low"))
    df = df.assign(_t=t.values).dropna(subset=["close"]).sort_values("_t").drop_duplicates("_t")
    times = df["_t"].values.astype("datetime64[ns]").astype(np.int64)
    c = df["close"].to_numpy(dtype=float, copy=True)
    if has:
        o = df["open"].to_numpy(dtype=float, copy=True)
        h = df["high"].to_numpy(dtype=float, copy=True)
        lo = df["low"].to_numpy(dtype=float, copy=True)
    else:
        o = np.r_[c[0], c[:-1]]
        h = np.maximum(o, c)
        lo = np.minimum(o, c)
    step = int(np.median(np.diff(times)) // NS_MIN) if len(times) > 1 else 1440
    if step >= 1440 - 60:
        step = 1440
    return RealSeries(times, o, h, lo, c, has, max(1, step), str(p.name))


def _load_db(spec: str) -> RealSeries:
    """``[<出どころ>/]<銘柄>/<時間足>`` を相場の DB から読む。"""
    from ..storage import prices  # storage は data を読むので、ここで初めて読み込む(輪にしない)

    parts = spec.split("/")
    if len(parts) not in (2, 3) or not all(parts):
        raise ValueError(f"db source must be db:<symbol>/<timeframe> or db:<source>/<symbol>/<timeframe>: {spec!r}")
    *src, symbol, tf_text = parts
    tf = prices.parse_timeframe(tf_text)
    con = prices.open_market_db()
    try:
        if src:
            source = src[0]
        else:
            found = prices.sources(con, symbol, tf)
            if len(found) != 1:
                raise ValueError(f"db:{spec}: {'no rates' if not found else f'several sources {found}'} "
                                 f"for {symbol} {tf_text} (write db:<source>/{symbol}/{tf_text})")
            source = found[0]
        info = prices.symbol_info(con, source, symbol)
        r = prices.read_rates(con, source, symbol, tf)
    finally:
        con.close()
    if info is None or len(r["time"]) == 0:
        raise ValueError(f"db:{spec}: no rates in {source!r}")
    times = bar_end_ns(r["time"], tf)
    s = r["spread"].astype(float) * float(info["point"])
    half = s / 2
    return RealSeries(times, r["open"] + half, r["high"] + half, r["low"] + half, r["close"] + half, True, tf,
                      f"{DB_PREFIX}{source}/{symbol}/{tf_text}", s)


def bar_end_ns(start_sec: np.ndarray, timeframe: int) -> np.ndarray:
    """足の始まり(UTC の秒)-> 足の終わり(ns)。

    日足は業者のサーバーの 0 時(NY の 17 時)始まりで、UTC では夏は 21 時、冬は 22 時になる。
    このリポジトリの日足は 22:00 UTC 区切りなので、終わりをその日の 22:00 UTC に揃える(日付だけの CSV と同じ)。
    """
    t = np.asarray(start_sec, dtype=np.int64)
    if timeframe == 1440:
        day = 86_400
        end = (t + day) // day * day + 22 * 3600
    else:
        end = t + timeframe * 60
    return end * 1_000_000_000


def _looks_histdata(p: Path) -> bool:
    with open(p, encoding="utf-8", errors="ignore") as f:
        line = f.readline()
    return ";" in line and line[:8].isdigit()
