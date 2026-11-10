"""実際の相場データの読み込み。

- 汎用 CSV: 列 time, close(必須)と open, high, low(任意)。time は UTC。
  日付だけの行は、その日の NY クローズ(22:00 UTC)の値とみなす。
- HistData Generic ASCII(format="histdata"): ``YYYYMMDD HHMMSS;O;H;L;C;V``(EST 固定 = UTC-5)。
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
                          self.step_minutes, self.source)


@lru_cache(maxsize=8)
def load_ohlc(path: str, fmt: str = "auto") -> RealSeries:
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


def _looks_histdata(p: Path) -> bool:
    with open(p, encoding="utf-8", errors="ignore") as f:
        line = f.readline()
    return ";" in line and line[:8].isdigit()
