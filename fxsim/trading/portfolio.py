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

MT5 の機能で固くしているところ:
    建玉の同一性   新規の注文のコメントに「シミュレーションのどの建玉か」(建てる足の日付、例 fxsim:261111)を付け、
                  建てたチケットと一緒に state.pairs に残す。シミュレーションが同じ向きで持っていても、別の建玉
                  (コメントの日付が違う)に属する実口座の建玉は決済し、建て直さない(途中からは乗らない)
    出し直し      待っている注文は、出して成功を確かめてから 1 件ずつ消す(送信中に切れたら残る)。新規を出す前に、
                  同じコメントの建玉が口座にあれば出さない(約定した後に切れた注文を二重に出さない)
    金利差        判断の足とその次の足の金利差を、端末のスワップの点数(swap_mode 1)から「受け取れる年率 + 業者の取り分」
                  として出す(業者が実際に付けるスワップで向きと下限を決める。過去の足は FRED の短期金利のまま)。
                  OANDA の点数は次のロールオーバーで付く分(週末の分が付く日は 3 日分)なので、判断のたびの観測を
                  state.pairs に残し、直近 10 回の中央値を 1 日分とする
    円への換算    端末の 1 ティックの口座通貨での値(trade_tick_value / trade_tick_size / 1 ロットの通貨量)。
                  端末が返さなければ、ほかのペアの気配から
    スプレッド    新規を出す直前の気配のスプレッドが、直近 24 時間の 1 分足のスプレッドの中央値の
                  fill_max_spread_mult 倍を超えていたら待つ(fill_wait_hours まで)
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, replace

import numpy as np

from ..calendar import NS_MIN
from ..data import Bars, load_rates
from ..data.rates import MULTI_RATES
from ..strategies.guards import EquityFloor, floor_lots
from .feed import pip_from_digits, rates_to_bars
from .mt5 import LivePosition, MT5Error, OrderResult
from .signals import PortfolioSnapshot, SimSnapshot
from .trader import Action, LiveTrader, TraderState, lots_for, plan

log = logging.getLogger(__name__)

PRICE_FIELDS = ("open", "high", "low", "close", "bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high",
                "ask_low", "ask_close", "spread_mean", "spread_max")

NS_DAY = 86_400_000_000_000


def pair_key(symbol: str) -> str:
    """MT5 の銘柄名(接尾辞付きもある)-> 戦略のペア名(小文字 6 文字)。"""
    return symbol[:6].lower()


def in_symbol_prices(snap: SimSnapshot, k: float) -> SimSnapshot:
    """円に換算した値段のシミュレーションの判断を、銘柄の値段に戻す(建値・損切り・利確とその幅を K で割る)。
    換算は判断のたびにその時の気配でし直すので、円の値段のまま比べると、動いていない損切りが動いたように見える。"""
    pos = snap.position
    if pos is not None:
        pos = replace(pos, entry_price=pos.entry_price / k, entry_mid=pos.entry_mid / k,
                      sl=pos.sl / k if pos.sl is not None else None, tp=pos.tp / k if pos.tp is not None else None)
    orders = [replace(o, sl_dist=o.sl_dist / k if o.sl_dist else o.sl_dist,
                      tp_dist=o.tp_dist / k if o.tp_dist else o.tp_dist) for o in snap.orders]
    return replace(snap, position=pos, orders=orders)


def scale_bars(b: Bars, k: float, pair: str) -> Bars:
    """価格に K を掛けた Bars(円に換算)。"""
    kw = {f: getattr(b, f) * k for f in PRICE_FIELDS}
    return replace(b, **kw, pair=pair)


class PortfolioTrader(LiveTrader):
    multi = True

    def __init__(self, cfg, client, behavior=None, strategy_factory: Callable[[], object] | None = None,
                 sleep=None, expected_login=None, runner=None):
        if not cfg.symbols:
            raise ValueError(f"{cfg.name}: multi-pair behaviors need symbols in the config")
        super().__init__(cfg, client, behavior, strategy_factory, sleep, expected_login, runner)
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

    def k_of(self, sym: str, spec) -> float:
        """銘柄の決済通貨 1 単位の円の値。端末の 1 ティックの値があればそれから、無ければほかのペアの気配から。"""
        if spec.tick_value and spec.tick_size and spec.contract_size:
            return float(spec.tick_value) / float(spec.tick_size) / float(spec.contract_size)
        return self.jpy_per_unit(pair_key(sym)[3:].upper())

    def swap_points(self, sym: str, spec, day: int) -> tuple[float, float] | None:
        """1 日分のスワップの点数(買い, 売り)。今日の観測を state.pairs に足し、直近 10 回の中央値を返す。
        観測が足りなければ None。"""
        obs = self.state.pairs.setdefault(sym, {}).setdefault("swap_obs", [])
        if not obs or obs[-1][0] != day:
            obs.append([day, float(spec.swap_long), float(spec.swap_short)])
            del obs[:-10]
        if len(obs) < max(1, int(self.cfg.broker_swap_min_obs)):
            return None
        return float(np.median([o[1] for o in obs])), float(np.median([o[2] for o in obs]))

    def broker_diff(self, sym: str, spec, price: float, day: int = 0) -> float | None:
        """端末のスワップの点数から、戦略に渡す金利差
        (受け取る向きの符号 x (受け取れる年率 + 業者の受け取り側の取り分))。
        受け取れる向きが無ければ 0。点数で付ける業者(swap_mode 1)でない・観測が足りなければ None(FRED のまま)。"""
        if not self.cfg.broker_swap or spec.swap_mode != 1 or spec.swap_long is None or spec.swap_short is None:
            return None
        pts = self.swap_points(sym, spec, day)
        if pts is None:
            return None
        long_pct = pts[0] * spec.point * 365 / price
        short_pct = pts[1] * spec.point * 365 / price
        markup = self.sim_broker.swap_markup
        if long_pct > 0 and long_pct >= short_pct:
            return long_pct + markup
        if short_pct > 0:
            return -(short_pct + markup)
        return 0.0

    def carry_series(self, sym: str, t: np.ndarray, table: np.ndarray, live: float | None) -> np.ndarray:
        """シミュレーションに渡す足ごとの金利差。判断の足と次の足は、業者が実際に付けるスワップ(live)。
        前の日までの判断の足は、その日に使った値を状態に残して使い続ける(E027: 月次の金利の表に戻すと、
        前の日に建てた判断が次の日の計算で消え、建てた翌日に「シミュレーションは持っていない」で手仕舞っていた)。"""
        rec = self.state.pairs.setdefault(sym, {}).setdefault("live_diff", {})
        if live is not None:
            for x in t[-2:]:
                rec[str(int(x) // NS_DAY)] = round(live, 7)
        first = int(t[0]) // NS_DAY                     # 読んでいる足より前の分だけ捨てる(長く持つ建玉の判断も消さない)
        for d in [d for d in rec if int(d) < first]:
            del rec[d]
        if not rec:
            return table
        diff = np.array(table, dtype=float, copy=True)
        for j, x in enumerate(t):
            v = rec.get(str(int(x) // NS_DAY))
            if v is not None:
                diff[j] = v
        return diff

    def tag_for(self, bar_t_ns: int | None) -> str:
        """シミュレーションの建玉(建てた足の始まり、UTC ns)の印。注文のコメントに付ける。"""
        if bar_t_ns is None:
            return ""
        d = np.datetime64(int(bar_t_ns), "ns").astype("datetime64[D]").astype(object)
        return f"{self.cfg.comment}:{d:%y%m%d}"[:31]

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
        bars, pips, ks, specs, swaps = {}, {}, {}, {}, {}
        rates_table = load_rates(MULTI_RATES)
        for s, r in rates.items():
            spec = self.client.symbol(s)
            k = self.k_of(s, spec)
            pair = pair_key(s)
            b = scale_bars(rates_to_bars(r, self.timeframe, spec.point, off), k, pair)
            live = self.broker_diff(s, spec, float(r["close"][-1]), int(forming // 86_400))
            diff = self.carry_series(s, b.t, rates_table.pair_differential(pair, b.t), live)
            swaps[s] = live
            bars[pair] = replace(b, rate_diff=diff)
            pips[pair] = (self.cfg.pip or pip_from_digits(spec.digits)) * k
            ks[s], specs[s] = k, spec
        order = [pair_key(s) for s in self.symbols]
        bars = {p: bars[p] for p in order}
        snap = self.runner.decide(bars, broker=self.sim_broker, pips=pips)
        per: dict[str, tuple[list[LivePosition], list[Action]]] = {}
        forming_t = int(bars[order[0]].t[-1])
        for s in self.symbols:
            pair, k, spec = pair_key(s), ks[s], specs[s]
            sim = in_symbol_prices(snap.pairs[pair], k)
            prev = TraderState(**{**{f: v for f, v in asdict(TraderState()).items()},
                                  **{f: v for f, v in self.state.pairs.get(s, {}).items()
                                     if f in ("sim_position_t", "sim_sl", "sim_tp")}})
            real = self.client.positions(s, self.cfg.magic)
            # 建玉の同一性: 実口座の建玉のコメントの印が、いまのシミュレーションの建玉の印と違えば、別の建玉
            want_tag = self.tag_for(sim.position_t) if sim.position is not None else ""
            prefix = f"{self.cfg.comment}:"
            stale = [p for p in real if p.comment.startswith(prefix) and want_tag and p.comment != want_tag]
            acts: list[Action] = []
            if stale:
                acts.append(Action("close_all", reason=f"belongs to another sim position ({stale[0].comment})"))
                real_for_plan: list[LivePosition] = []
            else:
                real_for_plan = real
            acts += plan(sim, real_for_plan, prev, lambda u, spec=spec: lots_for(u, spec, self.cfg.volume_scale))
            # 新規には建玉の印(値段は in_symbol_prices で銘柄の値段に戻してある)
            out = [replace(a, tag=self.tag_for(forming_t)) if a.kind == "open" else a for a in acts]
            per[s] = (real, out)
        close_t = int(bars[order[0]].t[-2]) + self.timeframe * NS_MIN
        meta = {"bars": len(bars[order[0]]), "k": ks, "broker_diff": swaps,
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

    def spread_wide(self, sym: str) -> bool:
        """いまの気配のスプレッドが、直近 24 時間の 1 分足のスプレッドの中央値の fill_max_spread_mult 倍を
        超えているか。"""
        m = self.cfg.fill_max_spread_mult
        if m <= 0:
            return False
        spec = self.client.symbol(sym)
        r = self.client.rates(sym, 1, 1440)
        typical = float(np.median(r["spread"])) * spec.point if len(r) else 0.0
        tick = self.client.tick(sym)
        return typical > 0 and (float(tick.ask) - float(tick.bid)) > m * typical

    def _drop_pending(self, ids: set[int]) -> None:
        s = self.state
        s.pending = [p for p in s.pending if p.get("id") not in ids]
        if not s.pending:
            s.pending_bar_t, s.pending_due = None, None
        s.updated_at = time.time()
        s.save(self.state_path)

    def run_pending(self) -> bool:
        """待っている注文を、出す時刻になっていれば出す。1 件ずつ、出して確かめてから消す(送信中に切れたら残り、
        次の回にやり直す)。新規は、同じ印の建玉が口座にあれば出さない。何か出したら True。"""
        s = self.state
        if not s.pending or not self.fill_due():
            return False
        if s.pending_due is None:
            s.pending_due = time.time()
            s.save(self.state_path)
        late = time.time() - s.pending_due >= self.cfg.fill_wait_hours * 3600
        ready, waiting = [], []
        for p in s.pending:
            a = Action(**p["action"])
            if a.kind in ("open", "add") and not late and self.spread_wide(p["symbol"]):
                waiting.append(p["symbol"])
                continue
            ready.append((p["id"], p["symbol"], a))
        if waiting:
            log.info("waiting for the spread of %s to come back", ", ".join(sorted(set(waiting))))
        if not ready:
            return False
        done = []
        for i, sym, a in ready:
            kept = self.apply_floor_all([(sym, a)])     # 1 件ずつ(出した分だけ口座の余裕が減る)
            if not kept:
                self._drop_pending({i})
                continue
            a = kept[0][1]
            if self.cfg.dry_run:
                rs = []
            elif a.kind == "open" and a.tag and any(p.comment == a.tag
                                                    for p in self.client.positions(sym, self.cfg.magic)):
                log.info("%s: a position tagged %s is already open; not sending again", sym, a.tag)
                self.journal("dedupe", symbol=sym, tag=a.tag)
                rs = []
            else:
                rs = self.execute(a, sym)              # 切れたら例外で抜ける(残りは pending に残る)
            done.append((sym, a, rs))
            if a.kind == "open" and rs and rs[0].ok:
                self.state.pairs.setdefault(sym, {})["ticket"] = int(rs[0].order or rs[0].deal or 0)
                self.state.pairs[sym]["tag"] = a.tag
            self._drop_pending({i})
            log.info("%s: %s %+d %s (%s)", sym, a.kind, a.side, a.lots or "", a.reason)
            for r in rs:
                if not r.ok:
                    log.error("%s order failed: %s %s", sym, r.retcode, r.comment)
        self.journal("fill", actions=[{"symbol": sym, **asdict(a)} for sym, a, _ in done],
                     results=[{"symbol": sym, "results": [asdict(r) for r in rs]} for sym, _, rs in done],
                     dry_run=self.cfg.dry_run)
        return bool(done)

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
            p = in_symbol_prices(snap.pairs[pair_key(sym)], meta["k"][sym])
            s.pairs.setdefault(sym, {}).update({"sim_position_t": p.position_t,
                                                "sim_sl": p.position.sl if p.position else None,
                                                "sim_tp": p.position.tp if p.position else None})
        acts = [(sym, a) for sym in self.symbols for a in per[sym][1]]
        s.pending = [{"id": k, "symbol": sym, "action": asdict(a)} for k, (sym, a) in enumerate(acts)]
        s.pending_bar_t = snap.bar_t if s.pending else None
        s.pending_due = None
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
