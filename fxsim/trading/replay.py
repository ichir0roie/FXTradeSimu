"""fxsim trade の 24 時間運用を、相場の DB(data/market.sqlite)の 1 分足で再生して確かめる
(python -m fxsim trade-replay)。

常駐のコード(Service / LiveTrader / MT5Client)はそのまま動かし、差し替えるのは 2 つだけ:
    - MetaTrader5 のモジュール → ReplayTerminal(1 分足を再生する偽の端末。口座・建玉・サーバー側の逆指値・スワップ・
      ストップアウトを持つ。銘柄と口座の決まりは OANDA Japan のデモの端末から読んだ値)
    - 時計 → 仮想の時計(fxsim.trading の trader / service / mt5 の time を差し替える)。常駐は 1 本ずつスレッドで動かし、
      Scheduler が「次に起きる常駐」の時刻まで時計を進めて、1 本ずつ順番に起こす(同じ時刻なら名前順。結果は決定的)

価格の再生: 1 分足 1 本を 4 つのティック(0 秒 始値、20 秒・40 秒 高値と安値(陽線なら安値が先)、59 秒 終値)にする。
ask = bid + その分のスプレッド。上の時間足(H1・D1 など)は、1 分足をサーバー時刻で区切って作る(作りかけの足も今の
ティックまで)。1 分足より前の足は、DB に溜めた同じ時間足の足をそのまま使う(本物の端末が持っている履歴の代わり)。
1 分足の間が 10 分以上空いたところ(週末など)は市場が閉じているとみなし、成行は 10018 で断る。

サーバー側(時計が進むたびにティックの順に):
    - 逆指値・利確: 買いは bid、売りは ask で判定。分の始値のティックで越えていればその値段(窓)、
      分の中ならその価格で約定
    - スワップ: サーバー時刻の日付が変わるたびに、経過した日数分(週末をまたげば 3 日分)。ポイントで指定(swap_mode 1)
    - ストップアウト: 証拠金維持率が stop_out(%)以下なら、損の大きい建玉から決済
    - 証拠金: 1 ロット = 10 万通貨 x 今の値段 / レバレッジ。両建ては大きい側だけ

故障(--faults default、シードで決まる):
    terminal    端末が落ちる(全関数が None、initialize も失敗)。サーバー側の逆指値は動き続ける
    bridge      Wine の橋が切れる(全関数が EOFError)
    restart     常駐の再起動(systemd の再起動の代わり。止めて 30 秒後に起動し直す。状態はファイルから)
    order_lost  注文の送信中に切れる(送る前に切れる = 約定しない / 約定した後に切れる = 約定したのに失敗が返る)
    reject      一時的な拒否(10021 気配なし / 10012 時間切れ / 10024 要求が多すぎる)
    cold        ログインの直後、足の履歴が届くまで copy_rates が None を返す
    suspend     PC のスリープ(常駐も止まる。サーバー側は動き続ける)
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time as _rt
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..storage import prices

log = logging.getLogger(__name__)

# MT5 の TIMEFRAME_* の値 -> 分
TF_CODES = {1: 1, 5: 5, 15: 15, 30: 30, 16385: 60, 16388: 240, 16408: 1440}
RATE_DTYPE = [("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"),
              ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")]
TICK_SEC = (0, 20, 40, 59)
CLOSED_GAP_SEC = 600            # 1 分足の間がこれ以上空いたら市場が閉じているとみなす
DAY = 86_400


# ---- 銘柄・口座の決まり ------------------------------------------------------------

@dataclass
class SymbolProps:
    """OANDA-Japan MT5 Demo の USDJPY(2026-11-11 に端末から読んだ値)。"""
    name: str = "USDJPY"
    digits: int = 3
    point: float = 0.001
    contract_size: float = 100_000.0
    volume_min: float = 0.01
    volume_max: float = 10.0
    volume_step: float = 0.01
    filling_mode: int = 2          # IOC だけ
    stops_level: int = 5           # ポイント
    swap_long: float = 10.16       # ポイント / ロット / 日(swap_mode 1)
    swap_short: float = -25.06


@dataclass
class AccountProps:
    login: int = 1
    server: str = "OANDA-Japan MT5 Demo"
    currency: str = "JPY"
    balance: float = 3_000_000.0
    leverage: float = 25.0
    stop_out: float = 100.0        # 証拠金維持率(%)
    company: str = "OANDA (replay)"


# ---- 価格 ----------------------------------------------------------------------

class Feed:
    """相場の DB の 1 分足を、ティックと任意の時間足の足(MT5 の copy_rates の形)にする。"""

    def __init__(self, con: sqlite3.Connection, source: str, symbol: str, point: float,
                 history_timeframes=(60, 240, 1440)):
        m1 = prices.read_rates(con, source, symbol, 1)
        if m1 is None or len(m1["time"]) == 0:
            raise ValueError(f"no M1 rates for {source} {symbol} in the market DB (python -m fxsim fetch-mt5)")
        self.point = point
        self.t = np.asarray(m1["time"], dtype=np.int64)
        self.st = np.asarray(m1["server_time"], dtype=np.int64)
        self.o, self.h, self.l, self.c = (np.asarray(m1[k], dtype=float) for k in ("open", "high", "low", "close"))
        self.spr = np.asarray(m1["spread"], dtype=np.int64)
        self.vol = np.asarray(m1["tick_volume"], dtype=np.int64)
        n = len(self.t)
        # ティック(1 分に 4 つ)
        up = self.c >= self.o
        p1 = np.where(up, self.l, self.h)
        p2 = np.where(up, self.h, self.l)
        self.tick_bid = np.stack([self.o, p1, p2, self.c], axis=1).reshape(-1)
        self.tick_t = (self.t[:, None] + np.asarray(TICK_SEC)[None, :]).reshape(-1)
        self.tick_min = np.repeat(np.arange(n), 4)
        self.tick_pos = np.tile(np.arange(4), n)
        gap_after = np.append(np.diff(self.t), CLOSED_GAP_SEC + 1)
        self.gap_after = gap_after
        self._hist = {tf: prices.read_rates(con, source, symbol, tf) for tf in history_timeframes
                      if tf in prices.TIMEFRAMES}
        self._tf: dict[int, dict] = {}

    # ティック
    def tick_index(self, now: float) -> int:
        return int(np.searchsorted(self.tick_t, now, side="right")) - 1

    def market_open(self, now: float) -> bool:
        k = self.tick_index(now)
        if k < 0:
            return False
        m = self.tick_min[k]
        return now < self.t[m] + 60 or self.gap_after[m] < CLOSED_GAP_SEC

    def quote(self, k: int) -> tuple[float, float, int]:
        """ティック k の (bid, ask, サーバー時刻)。"""
        m = self.tick_min[k]
        bid = self.tick_bid[k]
        return bid, bid + self.spr[m] * self.point, int(self.st[m] + TICK_SEC[self.tick_pos[k]])

    def server_offset_at(self, now: float) -> int:
        k = max(self.tick_index(now), 0)
        m = self.tick_min[k]
        return int(self.st[m] - self.t[m])

    # 足
    def _prep(self, tf: int) -> dict:
        if tf in self._tf:
            return self._tf[tf]
        sec = tf * 60
        key = (self.st // sec) * sec
        starts, first = np.unique(key, return_index=True)
        last = np.append(first[1:], len(key)) - 1
        g_of_min = np.searchsorted(starts, key)
        o = self.o[first]
        h = np.maximum.reduceat(self.h, first)
        lo = np.minimum.reduceat(self.l, first)
        c = self.c[last]
        v = np.add.reduceat(self.vol, first)
        s = np.minimum.reduceat(self.spr, first)
        # 1 分足の最初の区切りは途中から始まるので捨て、それより前は DB の同じ時間足の足(端末の履歴の代わり)
        seam = starts[1] if len(starts) > 1 else starts[0] + sec
        hist = self._hist.get(tf)
        if hist is not None and len(hist["time"]):
            keep = np.asarray(hist["server_time"]) < seam
            hv = {k: np.asarray(hist[k])[keep] for k in ("server_time", "open", "high", "low", "close",
                                                         "tick_volume", "spread")}
        else:
            hv = {k: np.zeros(0) for k in ("server_time", "open", "high", "low", "close", "tick_volume", "spread")}
        closed = np.zeros(len(hv["server_time"]) + len(starts) - 1, dtype=RATE_DTYPE)
        nh = len(hv["server_time"])
        closed["time"][:nh] = hv["server_time"]
        for name, src in (("open", "open"), ("high", "high"), ("low", "low"), ("close", "close"),
                          ("tick_volume", "tick_volume"), ("spread", "spread")):
            closed[name][:nh] = hv[src]
        closed["time"][nh:] = starts[1:]
        closed["open"][nh:], closed["high"][nh:], closed["low"][nh:], closed["close"][nh:] = o[1:], h[1:], lo[1:], c[1:]
        closed["tick_volume"][nh:], closed["spread"][nh:] = v[1:], s[1:]
        d = {"starts": starts, "first": first, "g_of_min": g_of_min, "closed": closed, "nh": nh}
        self._tf[tf] = d
        return d

    def first_full_bar_utc(self, tf: int) -> int:
        """1 分足から作る最初の(途中から始まらない)足の始まり(UTC)。再生はここより後から始める。"""
        d = self._prep(tf)
        g = 1 if len(d["starts"]) > 1 else 0
        m = d["first"][g]
        return int(self.t[m])

    def rates(self, tf: int, now: float, count: int) -> np.ndarray | None:
        """now の時点で端末が返す足(最後の 1 本は作りかけ)。古い順に最後の count 本。"""
        k = self.tick_index(now)
        if k < 0:
            return None
        d = self._prep(tf)
        m, pos = self.tick_min[k], self.tick_pos[k]
        g = d["g_of_min"][m]
        if g == 0:
            raise ValueError("replay starts before the first full bar of the timeframe")
        f = d["first"][g]
        bids = self.tick_bid[k - pos:k + 1]
        hi = max(self.h[f:m].max(initial=-np.inf), bids.max())
        lo = min(self.l[f:m].min(initial=np.inf), bids.min())
        forming = np.zeros(1, dtype=RATE_DTYPE)
        forming["time"] = d["starts"][g]
        forming["open"], forming["high"], forming["low"], forming["close"] = self.o[f], hi, lo, bids[-1]
        forming["tick_volume"] = int(self.vol[f:m].sum()) + pos + 1
        forming["spread"] = int(self.spr[f:m + 1].min())
        n_closed = d["nh"] + g - 1
        take = max(0, min(count - 1, n_closed))
        return np.concatenate([d["closed"][n_closed - take:n_closed], forming])


# ---- 口座 ----------------------------------------------------------------------

@dataclass
class RPos:
    ticket: int
    symbol: str
    side: int
    volume: float
    price_open: float
    sl: float | None
    tp: float | None
    time: int               # サーバー時刻
    magic: int
    comment: str = ""
    swap: float = 0.0


@dataclass
class Fault:
    start: float
    end: float
    kind: str               # terminal | bridge | suspend


class ReplayTerminal:
    """1 つの口座の端末。どの常駐も同じ端末(同じ口座)を見る。常駐ごとの窓口は ModuleView。"""

    def __init__(self, feed: Feed, clock: VirtualClock, sym: SymbolProps | None = None,
                 acct: AccountProps | None = None, rng: np.random.Generator | None = None,
                 p_reject: float = 0.0, p_lost: float = 0.0, cold_calls: int = 0):
        self.feed, self.clock = feed, clock
        self.sym = sym or SymbolProps()
        self.acct = acct or AccountProps()
        self.balance = self.acct.balance
        self.positions: dict[int, RPos] = {}
        self.deals: list[dict] = []
        self.next_ticket = 1000
        self.rng = rng or np.random.default_rng(0)
        self.p_reject, self.p_lost, self.cold_calls = p_reject, p_lost, cold_calls
        self.faults: list[Fault] = []
        self.calls: Counter = Counter()
        self.events: Counter = Counter()
        self.views: list[ModuleView] = []
        self._k = max(feed.tick_index(clock.now), 0)      # 処理済みの最後のティック
        self._day = self._server_day(self._k)
        self.lock = threading.RLock()

    # -- 時計が進んだときのサーバー側 --
    def _server_day(self, k: int) -> int:
        return self.feed.quote(k)[2] // DAY

    def advance(self, now: float) -> None:
        with self.lock:
            k1 = self.feed.tick_index(now)
            if k1 <= self._k:
                return
            if not self.positions:
                self._k, self._day = k1, self._server_day(k1)
                return
            for k in range(self._k + 1, k1 + 1):
                self._on_tick(k)
            self._k = k1

    def _on_tick(self, k: int) -> None:
        bid, ask, st = self.feed.quote(k)
        day = st // DAY
        if day != self._day:
            days = day - self._day
            for p in self.positions.values():
                pts = self.sym.swap_long if p.side > 0 else self.sym.swap_short
                p.swap += pts * self.sym.point * self.sym.contract_size * p.volume * days
            self.events["swap_days"] += days
            self._day = day
        gap = self.feed.tick_pos[k] == 0
        tt = float(self.feed.tick_t[k])
        prev, self._k = self._k, k                   # 損益・証拠金はこのティックの値段で
        for p in list(self.positions.values()):
            px = bid if p.side > 0 else ask
            if p.sl is not None and (px - p.sl) * p.side <= 0:
                self._close(p, p.volume, px if gap else p.sl, "sl", tt)
            elif p.tp is not None and (px - p.tp) * p.side >= 0:
                self._close(p, p.volume, px if gap else p.tp, "tp", tt)
        while self.positions and self.margin() > 0 and self.equity() / self.margin() * 100 <= self.acct.stop_out:
            worst = min(self.positions.values(), key=self._profit)
            self._close(worst, worst.volume, bid if worst.side > 0 else ask, "stop_out", tt)
        self._k = max(prev, k)

    # -- 口座 --
    def _quote(self) -> tuple[float, float, int]:
        return self.feed.quote(max(self._k, 0))

    def _profit(self, p: RPos) -> float:
        bid, ask, _ = self._quote()
        px = bid if p.side > 0 else ask
        return (px - p.price_open) * p.side * p.volume * self.sym.contract_size

    def equity(self) -> float:
        return self.balance + sum(self._profit(p) + p.swap for p in self.positions.values())

    def margin(self) -> float:
        bid, ask, _ = self._quote()
        mid = (bid + ask) / 2
        side_vol = {1: 0.0, -1: 0.0}
        for p in self.positions.values():
            side_vol[p.side] += p.volume
        return max(side_vol.values()) * self.sym.contract_size * mid / self.acct.leverage

    def _close(self, p: RPos, volume: float, price: float, reason: str, t: float | None = None) -> float:
        frac = volume / p.volume
        profit = (price - p.price_open) * p.side * volume * self.sym.contract_size
        swap = p.swap * frac
        self.balance += profit + swap
        p.swap -= swap
        p.volume = round(p.volume - volume, 8)
        if p.volume <= 1e-9:
            del self.positions[p.ticket]
        self.deals.append({"t": self.clock.now if t is None else t, "ticket": p.ticket, "entry": "out", "side": p.side,
                           "volume": volume,
                           "price": price, "profit": profit, "swap": swap, "reason": reason, "magic": p.magic})
        self.events[f"close_{reason}"] += 1
        return price

    # -- 注文 --
    def order_send(self, req: dict, view: ModuleView):
        with self.lock:
            self.advance(self.clock.now)
            if self.p_reject and self.rng.random() < self.p_reject:
                code = int(self.rng.choice([10021, 10012, 10024]))
                self.events[f"reject_{code}"] += 1
                return _result(code, "rejected (replay fault)")
            lost = self.p_lost and self.rng.random() < self.p_lost
            if lost and self.rng.random() < 0.5:
                self.events["lost_before"] += 1
                raise EOFError("connection closed while sending the order (replay fault: not executed)")
            r = self._execute(req)
            if lost:
                self.events["lost_after"] += 1
                raise EOFError(f"connection closed after the order was executed (replay fault: retcode {r.retcode})")
            return r

    def _execute(self, req: dict):
        sym = self.sym
        bid, ask, st = self._quote()
        if req.get("action") == 6:                                   # TRADE_ACTION_SLTP
            p = self.positions.get(int(req.get("position", 0)))
            if p is None:
                return _result(10013, "invalid request (no position)")
            sl, tp = (req.get("sl") or None), (req.get("tp") or None)
            if (sl, tp) == (p.sl, p.tp):
                return _result(10025, "no changes")
            if not self._stops_ok(p.side, sl, tp, bid, ask):
                return _result(10016, "invalid stops")
            p.sl, p.tp = sl, tp
            self.events["modify"] += 1
            return _result(10009, "done")
        if req.get("action") != 1:
            return _result(10013, "invalid request")
        if not self.feed.market_open(self.clock.now):
            self.events["market_closed"] += 1
            return _result(10018, "market closed")
        vol = float(req["volume"])
        steps = vol / sym.volume_step
        if vol < sym.volume_min - 1e-9 or vol > sym.volume_max + 1e-9 or abs(steps - round(steps)) > 1e-6:
            return _result(10014, "invalid volume")
        if int(req.get("type_filling", 1)) != 1 or not (sym.filling_mode & 2):
            return _result(10030, "unsupported filling mode")
        side = 1 if int(req["type"]) == 0 else -1
        if "position" in req:
            p = self.positions.get(int(req["position"]))
            if p is None or p.side != -side or vol > p.volume + 1e-9:
                return _result(10013, "invalid request (close)")
            price = bid if p.side > 0 else ask
            self._close(p, vol, price, "client")
            return _result(10009, "done", price=price, volume=vol)
        price = ask if side > 0 else bid
        sl, tp = (req.get("sl") or None), (req.get("tp") or None)
        if not self._stops_ok(side, sl, tp, bid, ask):
            return _result(10016, "invalid stops")
        mid = (bid + ask) / 2
        extra = vol * sym.contract_size * mid / self.acct.leverage
        if self.equity() - (self.margin() + extra) < 0:
            return _result(10019, "no money")
        self.next_ticket += 1
        p = RPos(self.next_ticket, req["symbol"], side, vol, price, sl, tp, st, int(req.get("magic", 0)),
                 str(req.get("comment", "")))
        self.positions[p.ticket] = p
        self.deals.append({"t": self.clock.now, "ticket": p.ticket, "entry": "in", "side": side, "volume": vol,
                           "price": price, "profit": 0.0, "swap": 0.0, "reason": "client", "magic": p.magic})
        self.events["open"] += 1
        return _result(10009, "done", price=price, volume=vol, order=p.ticket, deal=p.ticket)

    def _stops_ok(self, side: int, sl, tp, bid: float, ask: float) -> bool:
        lvl = self.sym.stops_level * self.sym.point
        ref = bid if side > 0 else ask
        if sl is not None and (ref - sl) * side < lvl:
            return False
        return not (tp is not None and (tp - ref) * side < lvl)

    def add_manual(self, side: int, volume: float, sl_dist: float | None, magic: int = 0) -> int:
        """手動の建玉(ほかの EA・手で建てたもの)を足す。"""
        with self.lock:
            bid, ask, st = self._quote()
            price = ask if side > 0 else bid
            self.next_ticket += 1
            sl = price - side * sl_dist if sl_dist else None
            self.positions[self.next_ticket] = RPos(self.next_ticket, self.sym.name, side, volume, price, sl, None,
                                                    st, magic, "manual")
            return self.next_ticket

    # -- 故障 --
    def fault_now(self) -> str | None:
        now = self.clock.now
        for f in self.faults:
            if f.kind in ("terminal", "bridge") and f.start <= now < f.end:
                return f.kind
        return None


def _result(code: int, comment: str, price: float = 0.0, volume: float = 0.0, order: int = 0, deal: int = 0):
    return SimpleNamespace(retcode=code, comment=comment, order=order, deal=deal, price=price, volume=volume)


class ModuleView:
    """1 つの常駐(プロセス)から見た MetaTrader5 のモジュール。initialize してから使える。"""

    def __init__(self, term: ReplayTerminal, name: str):
        self.term, self.name = term, name
        self.initialized = False
        self.cold = 0
        self._err = (1, "Success")
        self._last_fault: str | None = None
        term.views.append(self)

    def _gate(self, fn: str) -> bool:
        """呼べるなら True。端末が落ちていれば False(None を返す)、橋が切れていれば EOFError。"""
        t = self.term
        t.calls[fn] += 1
        t.advance(t.clock.now)
        f = t.fault_now()
        if f != self._last_fault and f is not None:
            self.initialized = False            # 落ちた端末・切れた橋は、つなぎ直す(initialize)まで使えない
        self._last_fault = f
        if f == "bridge":
            raise EOFError("connection closed by peer (replay fault: bridge down)")
        if f == "terminal":
            self._err = (-10004, "No IPC connection")
            return False
        if not self.initialized and fn != "initialize":
            self._err = (-10004, "No IPC connection")
            return False
        return True

    # 接続
    def initialize(self, **kw):
        if not self._gate("initialize"):
            return False
        self.initialized = True
        self.cold = self.term.cold_calls
        return True

    def shutdown(self):
        self.term.calls["shutdown"] += 1
        self.initialized = False

    def last_error(self):
        return self._err

    def terminal_info(self):
        if not self._gate("terminal_info"):
            return None
        return SimpleNamespace(trade_allowed=True, connected=True)

    def account_info(self):
        if not self._gate("account_info"):
            return None
        t = self.term
        with t.lock:
            eq, mg = t.equity(), t.margin()
        a = t.acct
        return SimpleNamespace(login=a.login, server=a.server, currency=a.currency, balance=t.balance, equity=eq,
                               margin=mg, margin_free=eq - mg, leverage=int(a.leverage), trade_mode=0, margin_mode=2,
                               company=a.company)

    # 銘柄・足
    def symbol_select(self, name, enable):
        return self._gate("symbol_select")

    def symbol_info(self, name):
        if not self._gate("symbol_info"):
            return None
        s = self.term.sym
        return SimpleNamespace(digits=s.digits, point=s.point, trade_contract_size=s.contract_size,
                               volume_min=s.volume_min, volume_max=s.volume_max, volume_step=s.volume_step,
                               filling_mode=s.filling_mode, trade_stops_level=s.stops_level)

    def symbol_info_tick(self, name):
        if not self._gate("symbol_info_tick"):
            return None
        bid, ask, st = self.term._quote()
        return SimpleNamespace(time=st, bid=bid, ask=ask, last=0.0, volume=0, time_msc=st * 1000, flags=4)

    def copy_rates_from_pos(self, name, tf, start, count):
        if not self._gate("copy_rates_from_pos"):
            return None
        if self.cold > 0:                        # ログインの直後は履歴が届くまで失敗する
            self.cold -= 1
            self._err = (-1, "Terminal: Call failed")
            return None
        return self.term.feed.rates(TF_CODES[int(tf)], self.term.clock.now, int(count))

    def positions_get(self, symbol=None):
        if not self._gate("positions_get"):
            return None
        t = self.term
        with t.lock:
            return tuple(SimpleNamespace(ticket=p.ticket, symbol=p.symbol, type=0 if p.side > 0 else 1,
                                         volume=p.volume, price_open=p.price_open, sl=p.sl or 0.0, tp=p.tp or 0.0,
                                         time=p.time, magic=p.magic, swap=p.swap, profit=t._profit(p),
                                         comment=p.comment)
                         for p in t.positions.values() if symbol is None or p.symbol == symbol)

    def order_calc_profit(self, otype, symbol, volume, price_open, price_close):
        if not self._gate("order_calc_profit"):
            return None
        side = 1 if int(otype) == 0 else -1
        return (price_close - price_open) * side * volume * self.term.sym.contract_size

    def order_send(self, req):
        if not self._gate("order_send"):
            return None
        return self.term.order_send(dict(req), self)


# ---- 時計と順番 -----------------------------------------------------------------

class VirtualClock:
    def __init__(self, start: float):
        self.now = float(start)


class Scheduler:
    """常駐のスレッドを仮想の時計で 1 本ずつ動かす。眠る(sleep / wait)と次の常駐に順番が移る。"""

    def __init__(self, clock: VirtualClock, terminals: list[ReplayTerminal]):
        self.clock, self.terminals = clock, terminals
        self.cv = threading.Condition()
        self.wake: dict[str, float] = {}
        self.running: str | None = None
        self.threads: dict[str, threading.Thread] = {}
        self.stopping = False
        self.suspends: list[Fault] = []
        self.error: BaseException | None = None

    def sleep(self, sec: float) -> None:
        name = threading.current_thread().name
        if name not in self.threads:
            raise RuntimeError(f"virtual sleep from an unknown thread {name}")
        with self.cv:
            self.wake[name] = self.clock.now + max(float(sec), 0.0)
            self.running = None
            self.cv.notify_all()
            self.cv.wait_for(lambda: self.running == name)

    def spawn(self, name: str, fn) -> None:
        def body():
            with self.cv:
                self.cv.wait_for(lambda: self.running == name)
            try:
                fn()
            except BaseException as e:      # 常駐の外(再生の仕組み)の失敗は止めて知らせる
                log.exception("replay participant %s failed", name)
                self.error = e
            finally:
                with self.cv:
                    self.wake.pop(name, None)
                    self.running = None
                    self.cv.notify_all()
        th = threading.Thread(target=body, name=name, daemon=True)
        with self.cv:
            self.threads[name] = th
            self.wake[name] = self.clock.now
        th.start()

    def _move_clock(self, t: float) -> bool:
        """t まで時計を進める。途中で PC のスリープに入ったら、スリープの終わりまで進めて True
        (起こす相手を選び直す)。"""
        for s in list(self.suspends):
            if s.start <= t:                 # PC のスリープ: 常駐は止まり、サーバー側だけが進む
                for term in self.terminals:
                    term.advance(s.end)
                self.clock.now = max(self.clock.now, s.end)
                for n, w in self.wake.items():
                    self.wake[n] = max(w, s.end)
                self.suspends.remove(s)
                log.info("suspend %s .. %s", _iso(s.start), _iso(s.end))
                return True
        for term in self.terminals:
            term.advance(t)
        self.clock.now = max(self.clock.now, t)
        return False

    def run_until(self, end: float, on_end) -> None:
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.running is None)
                if self.error is not None:
                    raise RuntimeError("replay participant failed") from self.error
                if not self.wake:
                    return
                name, t = min(self.wake.items(), key=lambda kv: (kv[1], kv[0]))
                if t > end and not self.stopping:
                    self.stopping = True
                    on_end()
                    for n in self.wake:
                        self.wake[n] = self.clock.now
                    continue
                if not self.stopping and self._move_clock(t):
                    continue
                del self.wake[name]
                self.running = name
                self.cv.notify_all()


class VirtualEvent:
    """Service.stop_event の代わり。wait は仮想の時計で眠る。"""

    def __init__(self, sched: Scheduler):
        self.sched = sched
        self.flag = False

    def set(self):
        self.flag = True

    def is_set(self):
        return self.flag or self.sched.stopping

    def wait(self, timeout=None):
        self.sched.sleep(timeout or 0.0)
        return self.is_set()


class VirtualTime:
    """fxsim.trading の各モジュールの time の代わり(仮想の時計)。"""

    def __init__(self, clock: VirtualClock, sched: Scheduler):
        self.clock, self.sched = clock, sched

    def time(self) -> float:
        return self.clock.now

    def time_ns(self) -> int:
        return int(self.clock.now * 1e9)

    def monotonic(self) -> float:
        return self.clock.now

    def sleep(self, sec: float) -> None:
        self.sched.sleep(sec)


def _iso(t: float) -> str:
    return str(np.datetime64(int(t), "s"))


# ---- 再生 ----------------------------------------------------------------------

@dataclass
class ReplayOptions:
    start: str | None = None            # UTC。省略で 1 分足から作る最初の足の後の月曜
    end: str | None = None              # 省略で 1 分足の最後
    faults: str = "default"             # default | none
    seed: int = 1
    separate_accounts: bool = False     # 常駐ごとに別の口座(端末)にする
    balance: float = 3_000_000.0
    source: str | None = None
    out_dir: Path | None = None
    manual: list[dict] = field(default_factory=list)   # [{"at": ISO, "side": 1, "volume": 0.1, "sl_dist": 1.0}]


def _fault_plan(rng: np.random.Generator, t0: float, t1: float, names: list[str]) -> dict:
    """故障の予定(期間あたりの回数はおおよそ: 端末 4 日に 1 回、橋 7 日に 1 回、再起動 10 日に 1 回 / 常駐、
    スリープ 20 日に 1 回)。"""
    days = (t1 - t0) / DAY

    def times(rate_per_day):
        n = rng.poisson(rate_per_day * days)
        return np.sort(rng.uniform(t0, t1, n))

    plan = {"terminal": [], "bridge": [], "suspend": [], "restart": {n: [] for n in names}}
    for s in times(1 / 4):
        plan["terminal"].append(Fault(s, s + float(rng.uniform(60, 1800)), "terminal"))
    for s in times(1 / 7):
        plan["bridge"].append(Fault(s, s + float(rng.uniform(30, 300)), "bridge"))
    for s in times(1 / 20):
        plan["suspend"].append(Fault(s, s + float(rng.uniform(1800, 6 * 3600)), "suspend"))
    for n in names:
        plan["restart"][n] = list(times(1 / 10))
    return plan


def run_replay(config_paths: list[str], opts: ReplayOptions, market_db: str | Path | None = None) -> dict:
    """設定(configs/trading/*.yaml)の常駐を、1 分足の再生の上で動かす。結果(report の材料)を返す。"""
    import fxsim.trading.config as tconfig
    import fxsim.trading.mt5 as tmt5
    import fxsim.trading.service as tservice
    import fxsim.trading.trader as ttrader

    from .config import load_credentials, load_trader_config
    from .service import Service

    cfgs = [load_trader_config(p) for p in config_paths]
    con = prices.open_market_db(market_db)
    symbol = cfgs[0].symbol
    if any(c.symbol != symbol for c in cfgs):
        raise ValueError("all configs must trade the same symbol")
    srcs = prices.sources(con, symbol, 1)
    source = opts.source or (srcs[0] if srcs else None)
    if source is None:
        raise ValueError(f"no M1 rates for {symbol} in the market DB")
    sym = SymbolProps(name=symbol)
    feed = Feed(con, source, symbol, sym.point)
    tfs = sorted({_tf_of(c) for c in cfgs})
    first_ok = max(feed.first_full_bar_utc(tf) for tf in tfs)
    if opts.start:
        start = float(np.datetime64(opts.start, "s").astype(np.int64))
    else:                                       # 最初の足の後の月曜 0 時(UTC)
        d = np.datetime64(first_ok, "s").astype("datetime64[D]")
        wd = (d.astype(np.int64) + 3) % 7                 # 0 = 月曜
        start = float((d + np.timedelta64(int((7 - wd) % 7) or 7, "D")).astype("datetime64[s]").astype(np.int64))
    if start < first_ok:
        raise ValueError(f"start must be at or after {_iso(first_ok)} (first full bar built from M1)")
    end = float(np.datetime64(opts.end, "s").astype(np.int64)) if opts.end else float(feed.t[-1] + 59)
    stamp = _rt.strftime("%Y%m%dT%H%M%S")
    out = Path(opts.out_dir) if opts.out_dir else tconfig.TRADING_DIR / "replay" / stamp
    out.mkdir(parents=True, exist_ok=True)

    clock = VirtualClock(start)
    rng = np.random.default_rng(opts.seed)
    faulty = opts.faults != "none"
    n_terms = len(cfgs) if opts.separate_accounts else 1
    terminals = []
    for k in range(n_terms):
        creds = load_credentials(cfgs[k].account)
        acct = AccountProps(login=creds.login if creds else 1 + k, balance=opts.balance)
        terminals.append(ReplayTerminal(feed, clock, sym, acct, np.random.default_rng(opts.seed * 100 + k),
                                        p_reject=0.03 if faulty else 0.0, p_lost=0.02 if faulty else 0.0,
                                        cold_calls=2 if faulty else 0))
    sched = Scheduler(clock, terminals)
    names = [c.name for c in cfgs]
    plan = _fault_plan(rng, start, end, names) if faulty else {"terminal": [], "bridge": [], "suspend": [],
                                                                 "restart": {n: [] for n in names}}
    for term in terminals:
        term.faults = plan["terminal"] + plan["bridge"]
    sched.suspends = list(plan["suspend"])

    # 時計を差し替える(この関数の中だけ)
    vt = VirtualTime(clock, sched)
    saved = {m: m.time for m in (ttrader, tservice, tmt5)}
    saved_dir = tconfig.TRADING_DIR
    for m in saved:
        m.time = vt
    tconfig.TRADING_DIR = out

    # 計測: 判断(シミュレーション)にかかった実時間
    decide_cost: dict[str, list[float]] = {n: [] for n in names}
    orig_decide = ttrader.LiveTrader.decide

    def timed_decide(self, forming):
        t0 = _rt.process_time()
        try:
            return orig_decide(self, forming)
        finally:
            decide_cost.setdefault(self.cfg.name, []).append(_rt.process_time() - t0)
    ttrader.LiveTrader.decide = timed_decide

    handler = _LogCapture(clock, out / "replay.log")
    root = logging.getLogger("fxsim")
    prev_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    services: dict[str, Service] = {}
    restarts: Counter = Counter()

    def participant(cfg, term):
        pending = list(plan["restart"][cfg.name])
        while not sched.stopping:
            view = ModuleView(term, cfg.name)
            svc = Service(cfg, module=view, heartbeat_sec=3600.0)
            svc.stop_event = VirtualEvent(sched)
            services[cfg.name] = svc
            # 再起動の予定: その時刻に止める(SIGTERM と同じ)
            while pending and pending[0] <= clock.now:
                pending.pop(0)
            if pending:
                _arm_restart(sched, svc, pending.pop(0))
            svc.run()
            if sched.stopping:
                return
            restarts[cfg.name] += 1
            sched.sleep(30.0)                    # systemd の RestartSec

    for k, cfg in enumerate(cfgs):
        term = terminals[k if opts.separate_accounts else 0]
        sched.spawn(cfg.name, lambda cfg=cfg, term=term: participant(cfg, term))
    for m in opts.manual:
        sched.spawn(f"manual-{m['at']}", lambda m=m: _manual(sched, terminals[0], m))

    def on_end():
        for svc in services.values():
            svc.stop_event.set()

    wall0 = _rt.perf_counter()
    cpu0 = _rt.process_time()
    try:
        sched.run_until(end, on_end)
    finally:
        for m, t in saved.items():
            m.time = t
        tconfig.TRADING_DIR = saved_dir
        ttrader.LiveTrader.decide = orig_decide
        root.removeHandler(handler)
        root.setLevel(prev_level)
        handler.close()
    result = {
        "start": _iso(start), "end": _iso(end), "source": source, "symbol": symbol, "out_dir": str(out),
        "faults": opts.faults, "seed": opts.seed, "configs": config_paths,
        "wall_sec": _rt.perf_counter() - wall0, "cpu_sec": _rt.process_time() - cpu0,
        "plan": {k: ([asdict(f) for f in v] if k != "restart" else {n: list(map(float, ts)) for n, ts in v.items()})
                 for k, v in plan.items()},
        "restarts": dict(restarts), "decide_cpu": {n: _stats(v) for n, v in decide_cost.items()},
        "log": handler.counts(),
        "terminals": [_terminal_summary(t) for t in terminals],
        "traders": [_trader_summary(c, out, feed, start, end, plan) for c in cfgs],
    }
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(report(result), encoding="utf-8")
    return result


def _tf_of(cfg) -> int:
    from ..cases.loader import load_cases
    return int(load_cases().behaviors[cfg.behavior].timeframe)


def _arm_restart(sched: Scheduler, svc, at: float) -> None:
    def killer():
        sched.sleep(max(at - sched.clock.now, 0.0))
        if sched.stopping:
            return
        log.info("restart: stopping %s", svc.cfg.name)
        svc.stop_event.set()
    sched.spawn(f"restart-{svc.cfg.name}-{int(at)}", killer)


def _manual(sched: Scheduler, term: ReplayTerminal, m: dict) -> None:
    at = float(np.datetime64(m["at"], "s").astype(np.int64))
    sched.sleep(max(at - sched.clock.now, 0.0))
    if sched.stopping:
        return
    t = term.add_manual(int(m.get("side", 1)), float(m.get("volume", 0.1)), m.get("sl_dist"))
    log.info("manual position %s added (sl_dist %s)", t, m.get("sl_dist"))
    if m.get("close_after_days"):
        sched.sleep(float(m["close_after_days"]) * DAY)
        with term.lock:
            p = term.positions.get(t)
            if p is not None:
                bid, ask, _ = term._quote()
                term._close(p, p.volume, bid if p.side > 0 else ask, "manual")


class _LogCapture(logging.Handler):
    """ログを仮想の時刻付きでファイルに書き、レベルと内容の種類ごとに数える。"""

    def __init__(self, clock: VirtualClock, path: Path):
        super().__init__(logging.INFO)
        self.clock = clock
        self.fh = open(path, "w", encoding="utf-8")  # noqa: SIM115  再生の間ずっと開いておく
        self.levels: Counter = Counter()
        self.kinds: Counter = Counter()

    def emit(self, record):
        msg = record.getMessage()
        self.levels[record.levelname] += 1
        if record.levelno >= logging.WARNING:
            self.kinds[f"{record.levelname} {_kind(msg)}"] += 1
        thread = threading.current_thread().name
        self.fh.write(f"{_iso(self.clock.now)} {record.levelname} [{thread}] {record.name}: {msg}\n")
        if record.exc_info and record.levelno >= logging.ERROR:
            self.fh.write(logging.Formatter().formatException(record.exc_info) + "\n")

    def counts(self) -> dict:
        return {"levels": dict(self.levels), "warnings_and_errors": dict(self.kinds.most_common())}

    def close(self):
        self.fh.close()
        super().close()


def _kind(msg: str) -> str:
    """ログの文から、数字・時刻を除いた種類。"""
    import re
    s = re.sub(r"\d[\d.:\-T ]*", "#", msg)
    return s[:90]


def _stats(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    a = np.asarray(v)
    return {"n": len(a), "mean_ms": float(a.mean() * 1e3), "p95_ms": float(np.percentile(a, 95) * 1e3),
            "max_ms": float(a.max() * 1e3), "total_sec": float(a.sum())}


def _terminal_summary(t: ReplayTerminal) -> dict:
    deals = t.deals
    by_magic: dict[int, Counter] = {}
    for d in deals:
        by_magic.setdefault(d["magic"], Counter())[f"{d['entry']}_{d['reason']}"] += 1
    return {"login_masked": "***", "balance": t.balance, "equity": t.equity(), "open_positions": len(t.positions),
            "deals_by_magic": {str(k): dict(v) for k, v in by_magic.items()},
            "realized": sum(d["profit"] + d["swap"] for d in deals),
            "calls": dict(t.calls.most_common()), "events": dict(t.events)}


def _trader_summary(cfg, out: Path, feed: Feed, start: float, end: float, plan: dict) -> dict:
    d = out / cfg.name
    recs = []
    jp = d / "journal.jsonl"
    if jp.exists():
        recs = [json.loads(line) for line in jp.read_text(encoding="utf-8").splitlines() if line.strip()]
    bars = [r for r in recs if r["kind"] == "bar"]
    tf = _tf_of(cfg)
    # 判断の遅れ: 足が確定してから判断するまで(仮想の時刻)
    # 判断できるようになった時刻 = 足が確定した後の最初のティック(週末をまたぐ足は月曜の最初のティック)
    lat = []
    seen = set()
    for r in bars:
        close_t = float(np.datetime64(r["bar_close_utc"], "s").astype(np.int64))
        j = min(int(np.searchsorted(feed.t, close_t - 3600)), len(feed.t) - 1)
        while j < len(feed.t) - 1 and feed.t[j] < close_t:
            j += 1
        if r["ts"] >= start + 1:                 # 起動した最初の判断は数えない(古い足をまとめて見る)
            lat.append(r["ts"] - max(close_t, float(feed.t[j])))
        seen.add(int(close_t))
    # 確定した足のうち、判断しなかったもの(1 分足から作った分。足の始まり + 足の長さ = 確定の時刻とみなす)
    dd = feed._prep(tf)
    expected = []
    for g in range(1, len(dd["first"]) - 1):
        bar_start_utc = int(feed.t[dd["first"][g]])
        next_tick_utc = int(feed.t[dd["first"][g + 1]])
        if start + tf * 60 <= next_tick_utc <= end:
            st = int(dd["starts"][g])
            close_utc = st - (feed.st[dd["first"][g]] - feed.t[dd["first"][g]]) + tf * 60
            expected.append((int(close_utc), bar_start_utc, next_tick_utc))
    # 常駐が判断に使うのは 1 つのずれなので、夏時間の切り替えの後は確定の時刻が 1 時間ずれて見える。
    # ±1 時間まで同じ足とみなす
    missed = [e for e in expected if not any(abs(e[0] - s) <= 3600 for s in seen)]
    # シミュレーションと実口座の食い違い: 判断の時点(足の確定)のシミュレーションの建玉と、実口座の建玉(注文の前)の向き。
    # 起動(つなぎ直し)したときにシミュレーションが持っていた建玉は、設計どおり実口座では建てずに待つ
    mismatch = []
    kinds = Counter()
    session_ts = sorted(r["ts"] for r in recs if r["kind"] == "startup")
    k_session, held_at_start = -1, None
    for a in bars:
        k = int(np.searchsorted(session_ts, a["ts"], side="right")) - 1
        pos = a["sim"]["position"]
        ident = (pos["side"], round(pos["entry_price"], 6)) if pos else None
        if k != k_session:
            k_session, held_at_start = k, ident
        sim_side = pos["side"] if pos else 0
        real_sides = {p["side"] for p in a["real"]}
        real_side = real_sides.pop() if len(real_sides) == 1 else (0 if not real_sides else 9)
        if sim_side != real_side:
            kind = ("起動の前からの建玉(設計どおり待つ)" if real_side == 0 and ident == held_at_start
                    else "実口座だけ持たない" if real_side == 0 else "実口座だけ持つ" if sim_side == 0
                    else "両方の向きが混ざる" if real_side == 9 else "向きが逆")
            kinds[kind] += 1
            mismatch.append({"bar": a["bar_close_utc"], "sim": sim_side, "real": real_side})
    results = Counter()
    for r in bars:
        for rs in r.get("results", []):
            for x in rs:
                results["ok" if x["ok"] else f"fail_{x['retcode']}"] += 1
    actions = Counter(a["kind"] for r in bars for a in r["actions"])
    other = Counter(r["kind"] for r in recs if r["kind"] != "bar")
    floor_skip = Counter(r.get("reason", "") for r in recs if r["kind"] == "floor_skip")
    state = json.loads((d / "state.json").read_text(encoding="utf-8")) if (d / "state.json").exists() else {}
    return {"name": cfg.name, "behavior": cfg.behavior, "timeframe": tf, "magic": cfg.magic,
            "bars_decided": len(bars), "bars_expected": len(expected), "missed": len(missed),
            "missed_list": [_iso(m[0]) for m in missed][:50],
            "latency_sec": _q(lat), "late_over_5min": int(sum(x > 300 for x in lat)),
            "mismatch_bars": len(mismatch), "mismatch_kinds": dict(kinds), "mismatch_list": mismatch[:30],
            "actions": dict(actions), "order_results": dict(results), "journal_kinds": dict(other),
            "floor_skip": dict(floor_skip), "state": state}


def _q(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    a = np.asarray(v)
    return {"n": len(a), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def report(r: dict) -> str:
    """結果の Markdown。"""
    lines = [f"# fxsim trade の再生 {r['start']} .. {r['end']} UTC", "",
             f"- 相場: {r['source']} {r['symbol']} の 1 分足 / 故障: {r['faults']}(シード {r['seed']})",
             f"- 設定: {', '.join(r['configs'])}",
             f"- 実時間 {r['wall_sec']:.0f} 秒、CPU {r['cpu_sec']:.0f} 秒 / 記録: `{r['out_dir']}`", "",
             "## 故障の予定", "",
             f"- 端末が落ちる {len(r['plan']['terminal'])} 回、橋が切れる {len(r['plan']['bridge'])} 回、"
             f"PC のスリープ {len(r['plan']['suspend'])} 回、常駐の再起動 "
             f"{sum(len(v) for v in r['plan']['restart'].values())} 回"
             f"(実際に再起動 {sum(r['restarts'].values())} 回)", "",
             "## 常駐ごと", "",
             "| 常駐 | 足 | 判断した足 / 確定した足 | 判断しなかった足 | 遅れ 中央 / 95% / 最大(秒) | 5 分超の遅れ | "
             "食い違い(足) | 注文の結果 | 判断の CPU 平均 / 最大(ms) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for t in r["traders"]:
        lat = t["latency_sec"]
        cpu = r["decide_cpu"].get(t["name"], {})
        lat_s = f"{lat['median']:.0f} / {lat['p95']:.0f} / {lat['max']:.0f}" if lat.get("n") else "-"
        cpu_s = f"{cpu['mean_ms']:.0f} / {cpu['max_ms']:.0f}" if cpu.get("n") else "-"
        lines.append(f"| {t['name']}({t['behavior']}) | {t['timeframe']} 分 | "
                     f"{t['bars_decided']} / {t['bars_expected']} | "
                     f"{t['missed']} | {lat_s} | {t['late_over_5min']} | {t['mismatch_bars']} | "
                     f"{t['order_results']} | {cpu_s} |")
    lines += ["", "## 常駐ごとの中身", ""]
    for t in r["traders"]:
        lines += [f"### {t['name']}", "",
                  f"- 出した注文の種類: {t['actions']}",
                  f"- 床で外した新規: {t['floor_skip'] or 'なし'}",
                  f"- 食い違いの種類: {t['mismatch_kinds'] or 'なし'}",
                  f"- 記録の種類: {t['journal_kinds']}",
                  f"- 判断しなかった足(最初の 50): {', '.join(t['missed_list']) or 'なし'}",
                  f"- 食い違い(最初の 30): {t['mismatch_list'] or 'なし'}", ""]
    lines += ["## 口座", ""]
    for k, term in enumerate(r["terminals"]):
        lines += [f"### 口座 {k + 1}", "",
                  f"- 残高 {term['balance']:,.0f} 円、有効証拠金 {term['equity']:,.0f} 円、"
                  f"建玉 {term['open_positions']}",
                  f"- 約定(magic ごと。in = 新規、out_client = 常駐の決済、out_sl = サーバーの逆指値、"
                  f"out_stop_out = ストップアウト): "
                  f"{term['deals_by_magic']}",
                  f"- サーバー側の出来事: {term['events']}",
                  f"- 端末の関数の呼び出し: {term['calls']}", ""]
    lines += ["## ログの警告・エラー", "", f"- レベル: {r['log']['levels']}", ""]
    for k, v in r["log"]["warnings_and_errors"].items():
        lines.append(f"- {v} 回: {k}")
    return "\n".join(lines) + "\n"


__all__ = ["Feed", "ModuleView", "ReplayOptions", "ReplayTerminal", "Scheduler", "VirtualClock", "report",
           "run_replay"]
