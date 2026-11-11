"""複数ペアの戦略(Strategy.multi、例: B047 main_carry_breakout)を、1 つの MT5 口座で常駐して動かす。

LiveTrader(1 銘柄)と同じ流れを、symbols の全ペアでまとめて行う。
    1. 全ペアの足を取り、全ペアにある足だけを残す(作りかけの足がまだ来ていないペアがあれば、その回は待つ)
    2. 決済通貨が円でないペアは、今の気配から「決済通貨 1 単位の円の値」K を求めて価格に掛ける(real_pair と同じ。
       損益・証拠金・量が円にそろう)。PortfolioBacktester でシミュレーションを回し、最後の確定足の判断を写す
       (signals.simulate_portfolio)
    3. ペアごとに、実口座の建玉(magic が同じもの)と plan で突き合わせる(1 銘柄の常駐と同じ決まり。途中からは乗らない)。
       損切り・利確の幅は円に換算した値段なので、K で割って銘柄の値段に戻す
    4. fill_server_hour があれば、サーバーの時刻がその時になるまで注文を待つ(待っている注文は state.json の pending)。
       待っている間に次の足の判断が出たら、古い注文は捨てて新しい判断にする
口座の床(protect)は 1 銘柄の常駐と同じ決まりで、口座全体の有効証拠金で見る。割ったら symbols の全ペアの、
この常駐(magic)の建玉を決済する。シミュレーションの中の床は切る(実口座の有効証拠金で見るので)。
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, replace

import numpy as np

from ..calendar import NS_MIN
from ..data import Bars
from ..strategies.guards import EquityFloor, floor_lots
from .feed import pip_from_digits, rates_to_bars
from .mt5 import LivePosition, MT5Error, OrderResult
from .signals import PortfolioSnapshot, simulate_portfolio
from .trader import Action, LiveTrader, TraderState, lots_for, plan

log = logging.getLogger(__name__)

PRICE_FIELDS = ("open", "high", "low", "close", "bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high",
                "ask_low", "ask_close", "spread_mean", "spread_max")


def pair_key(symbol: str) -> str:
    """MT5 の銘柄名(接尾辞付きもある)-> 戦略のペア名(小文字 6 文字)。"""
    return symbol[:6].lower()


def scale_bars(b: Bars, k: float, pair: str) -> Bars:
    """価格に K を掛けた Bars(円に換算)。"""
    kw = {f: getattr(b, f) * k for f in PRICE_FIELDS}
    return replace(b, **kw, pair=pair)


class PortfolioTrader(LiveTrader):
    multi = True

    def __init__(self, cfg, client, behavior=None, strategy_factory: Callable[[], object] | None = None,
                 sleep=None, expected_login=None):
        if not cfg.symbols:
            raise ValueError(f"{cfg.name}: multi-pair behaviors need symbols in the config")
        super().__init__(cfg, client, behavior, strategy_factory, sleep, expected_login)
        if strategy_factory is None:
            from ..strategies import STRATEGIES

            params = self.behavior.strategy_params()
            probe = STRATEGIES.create(self.behavior.strategy, params)
            if hasattr(probe.p, "protect"):
                params = {**params, "protect": 0.0}       # 床は実口座の有効証拠金で見る
            self.make_strategy = lambda: STRATEGIES.create(self.behavior.strategy, params)
        self.symbols = list(cfg.symbols)

    # ---- 起動 ----
    def startup(self) -> dict:
        info = super().startup()
        specs = {s: asdict(self.client.symbol(s)) for s in self.symbols}
        self.journal("symbols", symbols=specs)
        return info

    # ---- 円への換算 ----
    def jpy_per_unit(self, ccy: str) -> float:
        """通貨 ccy の 1 単位の円の値(今の気配の mid。symbols の中のペアから)。"""
        if ccy == "JPY":
            return 1.0
        names = {pair_key(s).upper(): s for s in self.symbols}

        def mid(code: str) -> float | None:
            s = names.get(code)
            if s is None:
                return None
            t = self.client.tick(s)
            return (float(t.bid) + float(t.ask)) / 2

        usdjpy = mid("USDJPY")
        if usdjpy is None:
            raise MT5Error("symbols must include USDJPY to convert non-yen pairs to yen")
        if ccy == "USD":
            return usdjpy
        for code, f in ((f"{ccy}JPY", lambda v: v), (f"USD{ccy}", lambda v: usdjpy / v),
                        (f"{ccy}USD", lambda v: usdjpy * v)):
            v = mid(code)
            if v:
                return f(v)
        raise MT5Error(f"no symbol in symbols converts {ccy} to yen (add USD{ccy} or {ccy}JPY)")

    # ---- 足 ----
    def fetch_all(self, forming: int) -> dict[str, np.ndarray] | None:
        lead = self.fetch_rates(forming)                 # 起点(anchor)は先頭の銘柄で決める
        out = {self.cfg.symbol: lead}
        for s in self.symbols:
            if s == self.cfg.symbol:
                continue
            r = self.client.rates(s, self.timeframe, len(lead) + 10)
            out[s] = r[r["time"] >= self.state.anchor]
        common = None
        for r in out.values():
            t = np.asarray(r["time"], dtype=np.int64)
            common = t if common is None else np.intersect1d(common, t)
        if common is None or len(common) < 2 or int(common[-1]) != forming:
            missing = [s for s, r in out.items() if int(r["time"][-1]) != forming]
            log.info("waiting for the forming bar of %s", ", ".join(missing) or "all symbols")
            return None
        return {s: r[np.isin(np.asarray(r["time"], dtype=np.int64), common)] for s, r in out.items()}

    def decide(self, forming: int):
        rates = self.fetch_all(forming)
        if rates is None:
            return None
        off = self.utc_offset()
        bars, pips, ks, specs = {}, {}, {}, {}
        for s, r in rates.items():
            spec = self.client.symbol(s)
            k = self.jpy_per_unit(pair_key(s)[3:].upper())
            pair = pair_key(s)
            bars[pair] = scale_bars(rates_to_bars(r, self.timeframe, spec.point, off), k, pair)
            pips[pair] = (self.cfg.pip or pip_from_digits(spec.digits)) * k
            ks[s], specs[s] = k, spec
        order = [pair_key(s) for s in self.symbols]
        bars = {p: bars[p] for p in order}
        snap = simulate_portfolio(bars, self.make_strategy(), self.sim_broker, self.engine_cfg, self.schedule, pips)
        per: dict[str, tuple[list[LivePosition], list[Action]]] = {}
        for s in self.symbols:
            pair, k, spec = pair_key(s), ks[s], specs[s]
            prev = TraderState(**{**asdict(TraderState()), **self.state.pairs.get(s, {})})
            real = self.client.positions(s, self.cfg.magic)
            acts = plan(snap.pairs[pair], real, prev, lambda u, spec=spec: lots_for(u, spec, self.cfg.volume_scale))
            out = []
            for a in acts:                               # 円に換算した値段 -> 銘柄の値段
                out.append(replace(a, sl_dist=a.sl_dist / k if a.sl_dist else a.sl_dist,
                                   tp_dist=a.tp_dist / k if a.tp_dist else a.tp_dist,
                                   sl=a.sl / k if a.sl else a.sl, tp=a.tp / k if a.tp else a.tp))
            per[s] = (real, out)
        close_t = int(bars[order[0]].t[-2]) + self.timeframe * NS_MIN
        meta = {"bars": len(bars[order[0]]), "k": ks,
                "bar_close_utc": str(np.datetime64(close_t, "ns").astype("datetime64[s]"))}
        return snap, per, meta

    # ---- 口座の床 ----
    def check_floor(self) -> tuple[EquityFloor, float]:
        self._floor_checked = time.time()
        f = self.load_floor()
        equity = float(self.client.account()["equity"])
        if self.cfg.protect <= 0:
            return f, equity
        f.update(time.time_ns(), equity)
        tripped = f.check(equity)
        self.save_floor(f)
        if tripped:
            log.error("equity %.0f fell to the floor %.0f: closing all pairs and halting new entries for this month",
                      equity, f.floor)
            results = [] if self.cfg.dry_run else [r for s in self.symbols
                                                   for r in self.execute(Action("close_all", reason="floor"), s)]
            self.state.pending = []
            self.state.save(self.state_path)
            self.journal("floor", equity=equity, floor=f.floor, results=[asdict(r) for r in results],
                         dry_run=self.cfg.dry_run)
        return f, equity

    def apply_floor_all(self, items: list[tuple[str, Action]]) -> list[tuple[str, Action]]:
        if self.cfg.protect <= 0 or not any(a.kind in ("open", "add") for _, a in items):
            return items
        f, equity = self.check_floor()
        closes_own = any(a.kind == "close_all" for _, a in items)
        worst = self.account_worst_loss(self.cfg.magic if closes_own else None)
        room = f.room(equity, worst)
        out = []
        for s, a in items:
            if a.kind not in ("open", "add"):
                out.append((s, a))
                continue
            spec = self.client.symbol(s)
            tick = self.client.tick(s)
            px = tick.ask if a.side > 0 else tick.bid
            per_lot = (-self.client.profit(s, a.side, 1.0, px, px - a.side * a.sl_dist) * self.cfg.gap_mult
                       if a.sl_dist else float("inf"))
            lots = floor_lots(a.lots, room, per_lot, spec.volume_step)
            if lots < spec.volume_min:
                why = ("halted for the month" if f.halted else "no stop on some position" if worst is None
                       else "no stop on the order" if not a.sl_dist else f"room {room:.0f}")
                log.warning("%s %s %+d %.2f lots skipped by the equity floor (%s)", s, a.kind, a.side, a.lots, why)
                self.journal("floor_skip", symbol=s, action=asdict(a), room=room, floor=f.floor, equity=equity,
                             reason=why)
                continue
            if lots < a.lots:
                a = replace(a, lots=lots, reason=a.reason + " (floor)")
            room -= lots * per_lot
            out.append((s, a))
        return out

    # ---- 常駐 ----
    def server_hour(self) -> int:
        return int(self.client.tick(self.cfg.symbol).time // 3600 % 24)

    def fill_due(self) -> bool:
        h = self.cfg.fill_server_hour
        return h is None or self.server_hour() >= int(h)

    def run_pending(self) -> bool:
        """待っている注文を、出す時刻になっていれば出す。出したら True。"""
        s = self.state
        if not s.pending or not self.fill_due():
            return False
        items = [(p["symbol"], Action(**p["action"])) for p in s.pending]
        items = self.apply_floor_all(items)
        s.pending, s.pending_bar_t = [], None           # 出す前に消す(途中で落ちても二重に出さない)
        s.updated_at = time.time()
        s.save(self.state_path)
        results = [] if self.cfg.dry_run else [(sym, self.execute(a, sym)) for sym, a in items]
        self.journal("fill", actions=[{"symbol": sym, **asdict(a)} for sym, a in items],
                     results=[{"symbol": sym, "results": [asdict(r) for r in rs]} for sym, rs in results],
                     dry_run=self.cfg.dry_run)
        for sym, a in items:
            log.info("%s: %s %+d %s (%s)", sym, a.kind, a.side, a.lots or "", a.reason)
        for sym, rs in results:
            for r in rs:
                if not r.ok:
                    log.error("%s order failed: %s %s", sym, r.retcode, r.comment)
        return True

    def poll(self) -> bool:
        if self.cfg.protect > 0 and time.time() - self._floor_checked >= self.cfg.floor_check_seconds:
            self.check_floor()
        filled = self.run_pending()
        forming = self.new_bar()
        if forming is None:
            return filled
        decided = self.decide(forming)
        if decided is None:                         # 作りかけの足がまだ来ていないペアがある。次の回にやり直す
            return filled
        snap, per, meta = decided
        s = self.state
        if s.pending:
            self.journal("pending_dropped", pending=s.pending, bar_t=s.pending_bar_t)
        s.last_forming, s.last_bar_t = forming, snap.bar_t
        for sym in self.symbols:
            p = snap.pairs[pair_key(sym)]
            s.pairs[sym] = {"sim_position_t": p.position_t, "sim_sl": p.position.sl if p.position else None,
                            "sim_tp": p.position.tp if p.position else None}
        s.pending = [{"symbol": sym, "action": asdict(a)} for sym in self.symbols for a in per[sym][1]]
        s.pending_bar_t = snap.bar_t if s.pending else None
        s.updated_at = time.time()
        s.save(self.state_path)
        self.journal("bar", **meta, sim={"equity": snap.equity, "halted": snap.halted, "trades": snap.n_trades,
                                         "positions": {k: asdict(v.position) if v.position else None
                                                       for k, v in snap.pairs.items()}},
                     real={sym: [asdict(p) for p in per[sym][0]] for sym in self.symbols},
                     actions=s.pending, dry_run=self.cfg.dry_run)
        if s.pending:
            log.info("bar %s: %d order(s)%s", meta["bar_close_utc"], len(s.pending),
                     "" if self.fill_due() else f" waiting for server hour {self.cfg.fill_server_hour}")
        self.run_pending()
        return True

    def execute(self, a: Action, sym: str | None = None) -> list[OrderResult]:
        return super().execute(a, sym)


__all__ = ["PortfolioTrader", "PortfolioSnapshot", "pair_key", "scale_bars"]
