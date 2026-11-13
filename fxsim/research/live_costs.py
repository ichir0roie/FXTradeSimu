"""E040: デモ口座の常駐(1 銘柄)の実際の約定を、同じ期間・同じ戦略のシミュレーションと突き合わせて、
頻繁な取引が実口座でどれだけ削られるかを測る。

    python -m fxsim fetch-mt5 configs/market_data/oanda-demo-majors.yaml     # 先に相場の DB を今に追いつかせる
    python -m fxsim.research.live_costs configs/trading/demo-usdjpy-b056.yaml --since 2026-11-12

実口座: 端末の約定の履歴(history_deals_get)から、この常駐の印(magic)の建玉を、新規と決済の対にする。
シミュレーション: 同じ動作ケースを相場の DB の足(real_pair、記録のスプレッド、OANDA)で、同じ期間だけ回す
(指標が温まるように、期間の前の warmup_days 日から回し、期間の中で建てた取引だけを数える)。
突き合わせ: 同じ向きで、建てた時刻の差が match_minutes 分以内の取引を同じ取引とみなす。
滑り: 常駐の記録(journal.jsonl)の、注文を出したときの気配(request.price)と約定の値段の差。
pip は約定の値段どうし(スプレッド・滑り込み)。正が有利。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np

from fxsim.catalog import *  # noqa: F403
from fxsim.paths import DATA_DIR
from fxsim.runners import SimRunner
from fxsim.trading import load_trader_config
from fxsim.trading.mt5 import MT5Client, load_module

NS = 1_000_000_000
REASONS = {0: "client", 1: "mobile", 2: "web", 3: "expert", 4: "sl", 5: "tp", 6: "so"}


def real_trades(cfg, since: dt.datetime, offset_hours: float | None) -> tuple[list[dict], float]:
    """この常駐の建玉を、新規と決済の対にする。時刻は UTC の ns。"""
    m = load_module(cfg.mt5)
    client = MT5Client(m, cfg.mt5)
    off = client.server_offset(cfg.symbol) if offset_hours is None else int(offset_hours * 3600)
    if off is None:
        raise SystemExit("server offset unknown (market closed?): pass --offset-hours")
    lo = since - dt.timedelta(days=2)
    hi = dt.datetime.now() + dt.timedelta(days=2)
    deals = m.history_deals_get(lo, hi) or ()
    mine = {d.position_id for d in deals if d.magic == cfg.magic and d.entry == 0 and d.symbol == cfg.symbol}
    by_pos: dict[int, dict] = {}
    for d in sorted(deals, key=lambda d: d.time_msc):
        if d.position_id not in mine:
            continue
        p = by_pos.setdefault(d.position_id, {"profit": 0.0, "swap": 0.0, "commission": 0.0})
        t = (int(d.time_msc) // 1000 - off) * NS
        if d.entry == 0:
            p.update(side=1 if d.type == 0 else -1, entry_t=t, entry_price=float(d.price), volume=float(d.volume))
        else:
            p.update(exit_t=t, exit_price=float(d.price), reason=REASONS.get(int(d.reason), str(d.reason)))
        p["profit"] += float(d.profit)
        p["swap"] += float(d.swap)
        p["commission"] += float(d.commission)
    since_ns = int(since.timestamp()) * NS
    out = [p for p in by_pos.values() if "entry_t" in p and p["entry_t"] >= since_ns]
    return sorted(out, key=lambda p: p["entry_t"]), off / 3600


def sim_trades(cfg, start_ns: int, end_ns: int, timeframe: int, warmup_days: int) -> list:
    t0 = np.datetime64(start_ns - warmup_days * 86_400 * NS, "ns")
    t1 = np.datetime64(end_ns, "ns")
    src = {"path": f"db:{cfg.symbol}/{'M' + str(timeframe) if timeframe < 60 else 'H' + str(timeframe // 60)}",
           "start": str(t0), "end": str(t1)}
    r = SimRunner(cfg.behavior, broker="oanda").run("real_pair", params={"source": src})
    return [x for x in r.backtester.trades if start_ns <= x.entry_t <= end_ns]


def slippage(cfg, pip: float) -> list[float]:
    """常駐の記録から、新規と決済の注文の滑り(pip、正が有利)。"""
    path = DATA_DIR / "trading" / cfg.name / "journal.jsonl"
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        for rs in e.get("results") or []:
            for r in rs:
                q = r.get("request") or {}
                if r.get("ok") and r.get("price") and q.get("price"):
                    side = 1 if q.get("type") == 0 else -1
                    out.append((float(q["price"]) - float(r["price"])) * side / pip)
    return out


def summary(rows: list[float]) -> str:
    if not rows:
        return "-"
    a = np.asarray(rows)
    return f"{a.mean():+.2f}(中央 {np.median(a):+.2f}、{len(a)} 件)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--since", required=True, help="この日時(UTC)以降に建てた取引を数える。例 2026-11-12")
    ap.add_argument("--until", default=None, help="この日時(UTC)まで。既定は今")
    ap.add_argument("--offset-hours", type=float, default=None, help="サーバー時刻 − UTC。既定は気配から推定")
    ap.add_argument("--match-minutes", type=float, default=6.0)
    ap.add_argument("--warmup-days", type=int, default=10)
    ap.add_argument("--timeframe", type=int, default=5)
    ap.add_argument("--pip", type=float, default=0.01)
    a = ap.parse_args()
    cfg = load_trader_config(a.config)
    since = dt.datetime.fromisoformat(a.since).replace(tzinfo=dt.UTC)
    real, off = real_trades(cfg, since, a.offset_hours)
    closed = [p for p in real if "exit_t" in p]
    # 期間の終わり = 最後の取引か、--until(既定は今)。相場の DB がそこまで追いついていなければ、DB の最後まで
    until = dt.datetime.fromisoformat(a.until).replace(tzinfo=dt.UTC) if a.until else dt.datetime.now(dt.UTC)
    end_ns = max([p["exit_t"] for p in closed] + [int(until.timestamp()) * NS])
    sim = sim_trades(cfg, int(since.timestamp()) * NS, end_ns, a.timeframe, a.warmup_days)
    pip = a.pip

    def rp(p):
        return (p["exit_price"] - p["entry_price"]) * p["side"] / pip

    tol = a.match_minutes * 60 * NS
    used, pairs = set(), []
    for p in closed:
        best = None
        for k, x in enumerate(sim):
            if k in used or x.side != p["side"] or abs(x.entry_t - p["entry_t"]) > tol:
                continue
            if best is None or abs(x.entry_t - p["entry_t"]) < abs(sim[best].entry_t - p["entry_t"]):
                best = k
        if best is not None:
            used.add(best)
            pairs.append((p, sim[best]))
    days = max((end_ns - int(since.timestamp()) * NS) / (86_400 * NS), 1e-9)
    print(f"期間: {since:%Y-%m-%d %H:%M} UTC から {np.datetime64(end_ns, 'ns')} まで(サーバー時刻 UTC{off:+g})\n")
    print("| | 取引 | 1 日あたり | pip / 取引(平均) | 勝率 | 損益(円) |")
    print("|---|---:|---:|---:|---:|---:|")
    if closed:
        r = [rp(p) for p in closed]
        yen = sum(p["profit"] + p["swap"] + p["commission"] for p in closed)
        print(f"| 実口座 | {len(closed)} | {len(closed) / days:.1f} | {np.mean(r):+.2f} | "
              f"{np.mean([x > 0 for x in r]):.0%} | {yen:+,.0f} |")
    if sim:
        print(f"| シミュレーション | {len(sim)} | {len(sim) / days:.1f} | {np.mean([x.pips for x in sim]):+.2f} | "
              f"{np.mean([x.pips > 0 for x in sim]):.0%} | {sum(x.pnl for x in sim):+,.0f} |")
    print(f"\n同じ取引とみなせた組: {len(pairs)}(実口座 {len(closed)}・シミュレーション {len(sim)})")
    if pairs:
        de = [(s.entry_price - p["entry_price"]) * p["side"] / pip for p, s in pairs]
        dx = [(p["exit_price"] - s.exit_price) * p["side"] / pip for p, s in pairs]
        dn = [rp(p) - s.pips for p, s in pairs]
        same = np.mean([(p.get("reason") == s.reason) for p, s in pairs])
        print(f"- 建値の差(実口座 − シミュレーション、正が有利): {summary(de)}")
        print(f"- 決済の値段の差: {summary(dx)}")
        print(f"- 1 取引の pip の差: {summary(dn)}")
        print(f"- 決済の理由が同じ割合: {same:.0%}")
    print(f"- 注文を出したときの気配からの滑り(正が有利): {summary(slippage(cfg, pip))}")
    reasons = {}
    for p in closed:
        reasons[p.get("reason")] = reasons.get(p.get("reason"), 0) + 1
    print(f"- 実口座の決済の理由: {reasons}")
    print(f"- シミュレーションのスプレッドと滑り(pip / 取引): {np.mean([x.cost_pips for x in sim]):.2f}" if sim else "")
    held = [(p["exit_t"] - p["entry_t"]) / 60 / NS for p in closed]
    if held:
        print(f"- 実口座の持った時間(分、中央): {np.median(held):.0f}")


if __name__ == "__main__":
    main()
