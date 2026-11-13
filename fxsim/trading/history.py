"""MT5 の足を相場の DB(data/market.sqlite)に溜める(python -m fxsim fetch-mt5)。

MT5 の端末が遡れる本数は「チャートの最大バー数」(既定の端末で 10 万本の少し手前)までで、1 分足なら 3 か月ほど。
それより古い足は MT5 からは取り直せないので、定期的に取って DB に書き足していく。同じ足は上書きするだけなので、
何度流してもよい。前回の分とつながらなかった(間が空きすぎた)ときは警告を出し、fetches に記録する。

設定(configs/market_data/*.yaml):
    account: demo                    # .env.<account> の口座でログインする(無ければ端末がいまログインしている口座)
    source: null                     # DB の出どころの名前。省略で口座のサーバー名(例: OANDA-Japan MT5 Demo)
    server_tz: America/New_York+7    # サーバー時刻の決まり(feed.server_to_utc)。固定のずれなら UTC+2 など
    symbols: [USDJPY]
    timeframes: [M1]
    bars: 90000                      # 1 回に遡る本数。最後の 1 本(作りかけ)は書かない
    mt5: {backend: rpyc, host: localhost, port: 18812, path: ...}
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ..paths import resolve
from ..storage import prices
from .config import MT5Config, _typed, load_credentials
from .feed import server_to_utc
from .mt5 import MT5Client, MT5Error

log = logging.getLogger(__name__)


@dataclass
class FetchConfig:
    account: str = "demo"
    source: str | None = None
    server_tz: str = "America/New_York+7"
    symbols: list[str] = field(default_factory=lambda: ["USDJPY"])
    timeframes: list[str | int] = field(default_factory=lambda: ["M1"])
    bars: int = 90_000
    offset_tolerance_sec: int = 0     # 気配から推定した今のずれと、server_tz のずれの許す差
    mt5: MT5Config = field(default_factory=MT5Config)

    def __post_init__(self):
        if isinstance(self.mt5, dict):
            self.mt5 = _typed(MT5Config, self.mt5)
        if isinstance(self.symbols, str):
            self.symbols = [self.symbols]
        if isinstance(self.timeframes, (str, int)):
            self.timeframes = [self.timeframes]
        self.timeframes = [prices.parse_timeframe(t) for t in self.timeframes]
        if not 2 <= int(self.bars) <= 99_999:
            raise ValueError("bars must be between 2 and 99999 (the terminal refuses 100000 or more by default)")


def load_fetch_config(path: str | Path) -> FetchConfig:
    with open(resolve(path), encoding="utf-8") as f:
        return _typed(FetchConfig, yaml.safe_load(f) or {})


def check_server_tz(client: MT5Client, symbol: str, rule: str, tolerance_sec: int = 0) -> int | None:
    """気配の時刻から推定した今のずれ(サーバー時刻 − UTC)が、server_tz の今のずれと合うか確かめる。

    市場が閉じていて推定できないときは None を返して確かめない。合わなければ MT5Error。
    """
    off = client.server_offset(symbol)
    if off is None:
        return None
    t = int(client.tick(symbol).time)
    expected = t - int(server_to_utc([t], rule)[0])
    if abs(expected - off) > tolerance_sec:
        raise MT5Error(f"server_tz {rule!r} gives server - UTC = {expected / 3600:+g} h now, "
                       f"but the quotes say {off / 3600:+g} h (fix server_tz in the config)")
    return off


def rates_rows(rates: np.ndarray, rule: str) -> dict[str, np.ndarray]:
    """MT5 の構造化配列 -> prices.write_rates の行(時刻を UTC に直す)。"""
    server = np.asarray(rates["time"], dtype=np.int64)
    rows = {c: np.asarray(rates[c]) for c in ("open", "high", "low", "close", "tick_volume", "spread")}
    has_real = "real_volume" in (rates.dtype.names or ())
    rows["real_volume"] = np.asarray(rates["real_volume"]) if has_real else np.zeros(len(server), np.int64)
    rows["server_time"] = server
    rows["time"] = server_to_utc(server, rule)
    return rows


def fetch_to_db(client: MT5Client, con: sqlite3.Connection, cfg: FetchConfig) -> list[dict]:
    """設定の銘柄 x 時間足を取って DB に書き足す。戻り値は fetches に書いた記録。"""
    source = cfg.source or str(client.account()["server"])
    out = []
    for symbol in cfg.symbols:
        spec = client.symbol(symbol)
        check_server_tz(client, symbol, cfg.server_tz, cfg.offset_tolerance_sec)
        prices.upsert_symbol(con, source, symbol, spec.digits, spec.point, cfg.server_tz)
        for tf in cfg.timeframes:
            r = client.rates(symbol, tf, cfg.bars)[:-1]      # 最後の 1 本は作りかけ(市場が閉じていれば次回に取る)
            if len(r) == 0:
                raise MT5Error(f"{symbol} M{tf}: no finished bars")
            rec = prices.write_rates(con, source, symbol, tf, rates_rows(r, cfg.server_tz), cfg.server_tz)
            if rec["connected"] == 0:
                log.warning("%s %s %s: the fetched bars do not reach the last stored bar; the bars in between are "
                            "missing (fetch more often than the terminal's history covers)",
                            source, symbol, prices.TIMEFRAMES[tf].upper())
            out.append(rec)
    return out


def run_fetch(cfg: FetchConfig, db: str | Path | None = None) -> list[dict]:
    """端末につないで取り、切って終わる。"""
    from .mt5 import load_module

    client = MT5Client(load_module(cfg.mt5), cfg.mt5)
    client.connect(load_credentials(cfg.account))
    con = prices.open_market_db(db)
    try:
        return fetch_to_db(client, con, cfg)
    finally:
        con.close()
        client.shutdown()
