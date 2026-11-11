"""fxsim trade-replay(常駐を 1 分足の再生の上で動かす)。合成した 1 分足の相場の DB で、短い期間だけ回す。"""
import numpy as np
import pytest

from fxsim.storage import prices
from fxsim.trading.replay import Feed, ReplayOptions, run_replay

SOURCE = "Replay Test"
OFFSET = 2 * 3600                       # サーバー時刻 = UTC + 2 時間
M1_START = 1_772_409_600                # 2026-03-02 00:00 UTC(月曜)


def _rows(t, mid, spread_pts=4):
    o = np.r_[mid[0], mid[:-1]]
    c = mid
    return {"time": t, "server_time": t + OFFSET, "open": o, "high": np.maximum(o, c) + 0.005,
            "low": np.minimum(o, c) - 0.005, "close": c, "tick_volume": np.full(len(t), 10),
            "spread": np.full(len(t), spread_pts), "real_volume": np.zeros(len(t), dtype=np.int64)}


@pytest.fixture
def market_db(tmp_path):
    rng = np.random.default_rng(3)
    con = prices.open_market_db(tmp_path / "market.sqlite")
    # 1 分足: 月曜から 4 日(平日だけ、途切れなし)
    t = M1_START + 60 * np.arange(4 * 1440)
    mid = 150 + np.cumsum(rng.normal(0, 0.01, len(t)))
    prices.write_rates(con, SOURCE, "USDJPY", 1, _rows(t, mid), "UTC+2")
    # その前の 1 時間足(端末の履歴の代わり。B001 は 3,000 本遡る)
    h = M1_START - 3600 * np.arange(4000, 0, -1)
    hm = 150 + np.cumsum(rng.normal(0, 0.08, len(h)))
    hm += mid[0] - hm[-1]
    prices.write_rates(con, SOURCE, "USDJPY", 60, _rows(h, hm), "UTC+2")
    con.close()
    return tmp_path / "market.sqlite"


def _config(tmp_path, name="replay-b001", magic=5):
    p = tmp_path / f"{name}.yaml"
    p.write_text(f"name: {name}\nbehavior: B001\nsymbol: USDJPY\nmagic: {magic}\naccount: demo\n"
                 "history_bars: 3000\nprotect: 0.9\n", encoding="utf-8")
    return str(p)


def test_feed_builds_the_forming_bar_from_ticks(market_db):
    con = prices.open_market_db(market_db)
    feed = Feed(con, SOURCE, "USDJPY", 0.001)
    now = M1_START + 3 * 3600 + 30 * 60 + 25              # 03:30:25 UTC = サーバー 05:30:25
    r = feed.rates(60, now, 3)
    assert list(r["time"][-2:] - OFFSET) == [M1_START + 2 * 3600, M1_START + 3 * 3600]
    k = feed.tick_index(now)
    assert r["close"][-1] == pytest.approx(feed.tick_bid[k])      # 作りかけの足の終値 = 今のティック
    assert feed.market_open(now)


def test_replay_without_faults_decides_every_bar(market_db, tmp_path):
    out = tmp_path / "out"
    r = run_replay([_config(tmp_path)], ReplayOptions(start="2026-03-03T00:00", end="2026-03-05T00:00",
                                                      faults="none", out_dir=out), market_db=market_db)
    t = r["traders"][0]
    assert t["missed"] == 0 and t["bars_decided"] >= t["bars_expected"] >= 47
    assert t["latency_sec"]["max"] < 60                          # 1 時間足は 30 秒ごとに見る
    assert all(k == "ok" for k in t["order_results"])
    assert r["log"]["levels"].get("ERROR", 0) == 0
    term = r["terminals"][0]
    deals = term["deals_by_magic"].get("5", {})
    assert deals.get("in_client", 0) - sum(v for k, v in deals.items() if k.startswith("out")) \
        == term["open_positions"]
    assert (out / "report.md").exists()


def test_replay_with_faults_keeps_running(market_db, tmp_path):
    r = run_replay([_config(tmp_path), _config(tmp_path, "replay-b001-2", magic=6)],
                   ReplayOptions(start="2026-03-03T00:00", end="2026-03-06T00:00", faults="default", seed=4,
                                 out_dir=tmp_path / "out"), market_db=market_db)
    kinds = r["log"]["warnings_and_errors"]
    assert not any("unexpected error" in k for k in kinds)
    for t in r["traders"]:
        assert t["bars_decided"] > 0
        assert "両方の向きが混ざる" not in t["mismatch_kinds"]
