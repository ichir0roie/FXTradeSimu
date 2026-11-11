"""相場の DB(storage/prices.py)と、MT5 からの取り込み(trading/history.py)、db: での読み込み(data/real.py)。"""
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from fxsim import paths
from fxsim.data import load_ohlc
from fxsim.data.real import bar_end_ns
from fxsim.markets import MARKETS
from fxsim.storage import prices
from fxsim.trading.feed import parse_server_tz, server_to_utc
from fxsim.trading.history import FetchConfig, check_server_tz, fetch_to_db
from fxsim.trading.mt5 import MT5Client, MT5Error

RULE = "America/New_York+7"


def _sec(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())


def _rates(server_start: str, n: int, minutes: int = 1, p0: float = 150.0) -> np.ndarray:
    dtype = [("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
             ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")]
    t = _sec(server_start) + np.arange(n) * minutes * 60
    c = p0 + np.arange(n) * 0.001
    r = np.zeros(n, dtype=dtype)
    r["time"], r["open"], r["close"] = t, c - 0.0005, c
    r["high"], r["low"] = c + 0.002, c - 0.002
    r["tick_volume"], r["spread"] = 10, 4
    return r


class FakeMT5:
    """MetaTrader5 モジュールの偽物(足と銘柄と気配だけ)。"""

    def __init__(self, rates, server="Broker-Demo", offset=None):
        self.rates = rates
        self.server = server
        self.offset = offset if offset is not None else _offset_now(RULE)

    def last_error(self):
        return (0, "ok")

    def account_info(self):
        return SimpleNamespace(login=1, server=self.server, currency="JPY", balance=0, equity=0, margin=0,
                               margin_free=0, leverage=25, trade_mode=0, margin_mode=2, company="x")

    def symbol_select(self, name, enable):
        return True

    def symbol_info(self, name):
        return SimpleNamespace(digits=3, point=0.001, trade_contract_size=100_000, volume_min=0.01, volume_max=100,
                               volume_step=0.01, filling_mode=1, trade_stops_level=0)

    def symbol_info_tick(self, name):
        return SimpleNamespace(time=int(time.time()) + self.offset, bid=150.0, ask=150.004)

    def copy_rates_from_pos(self, name, tf, start, count):
        return self.rates[max(0, len(self.rates) - count):]


def _offset_now(rule: str) -> int:
    """いまのサーバー時刻 − UTC(この決まりで)。"""
    now = int(time.time())
    for off in (7200, 10800):
        if (now + off) - int(server_to_utc([now + off], rule)[0]) == off:
            return off
    raise AssertionError("unexpected offset")


@pytest.fixture
def market_db(tmp_path, monkeypatch):
    path = tmp_path / "market.sqlite"
    monkeypatch.setattr(paths, "MARKET_DB", path)
    load_ohlc.cache_clear()
    yield path
    load_ohlc.cache_clear()


def test_server_time_follows_us_daylight_saving():
    # OANDA Japan: 週明けはサーバーの月曜 0 時。夏(〜11/1)は UTC+3、冬は UTC+2
    summer = server_to_utc([_sec("2026-08-17T00:00")], RULE)[0]
    winter = server_to_utc([_sec("2026-11-09T00:00")], RULE)[0]
    assert summer == _sec("2026-08-16T21:00")
    assert winter == _sec("2026-11-08T22:00")
    # 金曜の終わり(夏時間の最後の週)と月曜の始まり(冬時間の最初の週)
    fri, mon = server_to_utc([_sec("2026-10-30T23:59"), _sec("2026-11-02T00:00")], RULE)
    assert fri == _sec("2026-10-30T20:59") and mon == _sec("2026-11-01T22:00")
    assert server_to_utc([_sec("2026-08-17T00:00")], "UTC+2")[0] == _sec("2026-08-16T22:00")


def test_parse_server_tz():
    assert parse_server_tz("America/New_York+7") == ("America/New_York", 7 * 3600)
    assert parse_server_tz("UTC-5") == ("UTC", -5 * 3600)
    assert parse_server_tz("Etc/GMT-2") == ("Etc/GMT-2", 0)       # 符号まで地域の名前


def test_write_rates_upserts_and_records_whether_fetches_connect(market_db):
    con = prices.open_market_db(market_db)

    def rows(start, n):
        r = _rates(start, n)
        return {c: (server_to_utc(r["time"], RULE) if c == "time" else r["time"] if c == "server_time" else r[c])
                for c in prices.COLUMNS}

    a = prices.write_rates(con, "B", "USDJPY", 1, rows("2026-11-09T00:00", 100), RULE)
    assert a["n_new"] == 100 and a["connected"] is None
    b = prices.write_rates(con, "B", "USDJPY", 1, rows("2026-11-09T00:50", 100), RULE)   # 50 本重なる
    assert b["n_new"] == 50 and b["connected"] == 1
    c = prices.write_rates(con, "B", "USDJPY", 1, rows("2026-11-10T00:00", 10), RULE)    # 間が空いた
    assert c["n_new"] == 10 and c["connected"] == 0
    r = prices.read_rates(con, "B", "USDJPY", 1)
    assert len(r["time"]) == 160 and np.all(np.diff(r["time"]) > 0)
    assert r["time"][0] == _sec("2026-11-08T22:00") and r["server_time"][0] == _sec("2026-11-09T00:00")
    # 時間足ごとに別のテーブル
    assert prices.read_rates(con, "B", "USDJPY", 5)["time"].size == 0
    cov = prices.coverage(con)
    assert [(x["timeframe"], x["n"]) for x in cov] == [(1, 160)]
    with pytest.raises(ValueError):
        prices.table(2)


def test_fetch_to_db_then_read_with_db_path(market_db):
    raw = _rates("2026-10-30T23:50", 10) # 金曜(夏時間)
    raw = np.concatenate([raw, _rates("2026-11-02T00:00", 21, p0=151.0)])   # 月曜(冬時間)。最後の 1 本は作りかけ
    client = MT5Client(FakeMT5(raw))
    con = prices.open_market_db(market_db)
    recs = fetch_to_db(client, con, FetchConfig(symbols=["USDJPY"], timeframes=["M1"], bars=1000))
    assert recs[0]["source"] == "Broker-Demo" and recs[0]["n_rows"] == 30
    con.close()

    s = load_ohlc("db:USDJPY/M1")
    assert len(s) == 30 and s.step_minutes == 1 and s.has_ohlc
    # 時刻は足の終わり(UTC)。金曜は UTC+3、月曜は UTC+2
    assert s.times[0] // 10**9 == _sec("2026-10-30T20:51")
    assert s.times[10] // 10**9 == _sec("2026-11-01T22:01")
    # bid にスプレッドの半分を足して mid
    np.testing.assert_allclose(s.close, raw["close"][:30] + 0.002)
    np.testing.assert_allclose(s.spread, 0.004)
    assert load_ohlc("db:Broker-Demo/USDJPY/M1").close[-1] == s.close[-1]
    with pytest.raises(ValueError):
        load_ohlc("db:USDJPY/M5")                 # 溜めていない

    md = MARKETS.create("real", {"source": {"path": "db:USDJPY/M1"}}).generate(days=5, seed=1)
    assert md.step_minutes == 1
    assert md.mid[-1] == pytest.approx(s.close[-1])


def test_check_server_tz_rejects_a_wrong_rule():
    client = MT5Client(FakeMT5(_rates("2026-11-09T00:00", 5)))
    assert check_server_tz(client, "USDJPY", RULE) == _offset_now(RULE)
    wrong = "UTC+5"
    with pytest.raises(MT5Error):
        check_server_tz(client, "USDJPY", wrong)


def test_daily_bars_end_at_ny_close():
    # 日足はサーバーの 0 時始まり(夏 21:00 UTC、冬 22:00 UTC)。終わりは日付だけの CSV と同じ 22:00 UTC に揃える
    starts = np.array([_sec("2026-08-16T21:00"), _sec("2026-11-08T22:00")])
    ends = bar_end_ns(starts, 1440) // 10**9
    assert list(ends) == [_sec("2026-08-17T22:00"), _sec("2026-11-09T22:00")]
    assert bar_end_ns(starts[:1], 1)[0] // 10**9 == starts[0] + 60
