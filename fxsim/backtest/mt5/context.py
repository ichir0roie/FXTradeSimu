"""EA(本命の戦略)から見える MT5 の口座と取引の窓口。名前は MQL5 の CTrade・PositionGet*・AccountInfo* に寄せる。

呼ばれる時点: 新しい足の最初のティック(足 i+1 の始値)。on_bar(i, ctx) の i は確定した足で、ctx.bid / ctx.ask は
足 i+1 の始値の気配。成行はその気配でその場で約定し、結果(TradeResult)がすぐ返る。
待機注文・損切り・利確は、その後の足の中の経路で発動する(tester.py)。

MQL5 との対応(主なもの):
    ctx.buy(0.1, sl=.., tp=..)              CTrade::Buy
    ctx.sell_limit(0.1, price, ..)          CTrade::SellLimit(BuyStop / SellStop / BuyStopLimit ...)
    ctx.position_close(ticket[, volume])    CTrade::PositionClose / PositionClosePartial
    ctx.position_close_by(ticket, by)       CTrade::PositionCloseBy
    ctx.position_modify(ticket, sl, tp)     CTrade::PositionModify
    ctx.order_modify / ctx.order_delete     CTrade::OrderModify / OrderDelete
    ctx.positions(magic) / ctx.orders()     PositionsTotal + PositionGet* / OrdersTotal + OrderGet*
    ctx.account()                           AccountInfoDouble(ACCOUNT_BALANCE / EQUITY / MARGIN / MARGIN_FREE / LEVEL)
    ctx.symbol()                            SymbolInfo*(POINT / DIGITS / VOLUME_* / TRADE_STOPS_LEVEL / SWAP_*)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...data import Bars
from .orders import Mt5Deal, TradeResult
from .spec import (
    ORDER_TIME_GTC,
    ORDER_TYPE_BUY_LIMIT,
    ORDER_TYPE_BUY_STOP,
    ORDER_TYPE_BUY_STOP_LIMIT,
    ORDER_TYPE_SELL_LIMIT,
    ORDER_TYPE_SELL_STOP,
    ORDER_TYPE_SELL_STOP_LIMIT,
)


@dataclass(frozen=True)
class SymbolInfo:
    name: str
    digits: int
    point: float
    pip: float
    contract_size: float
    volume_min: float
    volume_step: float
    volume_max: float
    stops_level: int
    freeze_level: int
    commission_per_lot: float        # 片道、円
    bid: float
    ask: float
    swap_long_per_lot_day: float     # 円(受け取りが正)。今の金利差とブローカーの上乗せから
    swap_short_per_lot_day: float

    @property
    def spread(self) -> float:
        """スプレッド(価格)。"""
        return self.ask - self.bid

    @property
    def spread_points(self) -> int:
        return int(round((self.ask - self.bid) / self.point))


@dataclass(frozen=True)
class AccountInfo:
    mode: str
    currency: str
    leverage: float
    balance: float
    equity: float
    margin: float
    margin_free: float
    margin_level: float              # %(建玉が無ければ inf)
    margin_call_level: float
    stop_out_level: float


@dataclass(frozen=True)
class PositionInfo:
    ticket: int
    type: int                        # 0 = POSITION_TYPE_BUY / 1 = POSITION_TYPE_SELL
    side: int
    volume: float
    price_open: float
    sl: float | None
    tp: float | None
    profit: float                    # 評価損益(円、スワップ・手数料を除く)
    swap: float
    time: int
    magic: int
    comment: str
    bars_held: int                   # 建ててから確定した足の数


@dataclass(frozen=True)
class OrderInfo:
    ticket: int
    type: int
    volume: float
    price: float
    stoplimit: float | None
    sl: float | None
    tp: float | None
    time_setup: int
    expiration: int | None
    magic: int
    comment: str


class Mt5Context:
    def __init__(self, tester):
        self._t = tester

    # ---- 時刻と相場 ----
    @property
    def i(self) -> int:
        """確定した足の番号(on_bar の i と同じ)。"""
        return self._t.i - 1

    @property
    def bars(self) -> Bars:
        return self._t.bars

    @property
    def rng(self) -> np.random.Generator:
        return self._t.strategy_rng

    @property
    def time(self) -> int:
        """今の時刻(ns、新しい足の始まり)。"""
        return self._t._now()

    @property
    def bid(self) -> float:
        return self._t._bid()

    @property
    def ask(self) -> float:
        return self._t._ask()

    @property
    def mid(self) -> float:
        return self._t._mid()

    @property
    def pip(self) -> float:
        return self._t.pip

    @property
    def point(self) -> float:
        return self._t.sym.point

    @property
    def can_enter(self) -> bool:
        """新規注文を出してよい時間か(動作ケースの取引時間)。決済はいつでもできる。"""
        return self._t._entry_allowed()

    def symbol(self) -> SymbolInfo:
        t, s = self._t, self._t.sym
        return SymbolInfo(s.name, s.digits, s.point, s.pip, s.contract_size, s.volume_min, s.volume_step, s.volume_max,
                          s.stops_level, s.freeze_level, s.commission_per_lot, t._bid(), t._ask(),
                          t.swap_per_lot_day(1), t.swap_per_lot_day(-1))

    def account(self) -> AccountInfo:
        t, a = self._t, self._t.acct
        margin = t.margin()
        equity = t.balance + t.floating()
        return AccountInfo(a.mode, a.currency, a.leverage, t.balance, equity, margin, equity - margin,
                           equity / margin * 100.0 if margin > 0 else float("inf"), a.margin_call_level,
                           a.stop_out_level)

    @property
    def equity(self) -> float:
        return self._t.balance + self._t.floating()

    @property
    def balance(self) -> float:
        return self._t.balance

    @property
    def initial_balance(self) -> float:
        return float(self._t.cfg.initial_balance)

    # ---- 建玉・注文・履歴 ----
    def positions(self, magic: int | None = None) -> list[PositionInfo]:
        t = self._t
        bid, ask, cs = t._bid(), t._ask(), t.sym.contract_size
        return [PositionInfo(p.ticket, p.type, p.side, p.volume, p.price_open, p.sl, p.tp, p.profit(bid, ask, cs),
                             p.swap, p.time, p.magic, p.comment, max(0, self.i - p.open_idx + 1))
                for p in t.positions.values() if magic is None or p.magic == magic]

    def orders(self, magic: int | None = None) -> list[OrderInfo]:
        return [OrderInfo(o.ticket, o.type, o.volume, o.price, o.stoplimit, o.sl, o.tp, o.time_setup, o.expiration,
                          o.magic, o.comment)
                for o in self._t.orders.values() if magic is None or o.magic == magic]

    def net_volume(self, magic: int | None = None) -> float:
        """建玉の数量の合計(買い − 売り、ロット)。"""
        return sum(p.side * p.volume for p in self.positions(magic))

    @property
    def deals(self) -> list[Mt5Deal]:
        """約定の履歴(HistoryDeal*)。決済の損益は entry が OUT / INOUT / OUT_BY の行の profit。"""
        return self._t.deals

    @property
    def trades(self):
        """決済が済んだ取引(1 ペアのエンジンと同じ Trade の形)。"""
        return self._t.trades

    @property
    def last_result(self) -> TradeResult | None:
        return self._t.results[-1] if self._t.results else None

    # ---- 数量と価格 ----
    def normalize_volume(self, volume: float) -> float:
        """刻みに切り捨て、上限で抑える。最小に満たなければ 0。"""
        return self._t.sym.normalize_volume(volume)

    def normalize_price(self, price: float) -> float:
        return round(float(price), self._t.sym.digits)

    def lots_for_units(self, units: float) -> float:
        return self.normalize_volume(units / self._t.sym.contract_size)

    def margin_required(self, side: int, volume: float) -> float:
        """この注文を足したときに増える必要証拠金(円)。両建ての扱いは口座の設定どおり。"""
        return self._t.margin(extra_side=side, extra_volume=volume) - self._t.margin()

    def swap_per_lot_day(self, side: int) -> float:
        return self._t.swap_per_lot_day(side)

    # ---- 成行 ----
    def _magic(self, magic: int | None) -> int:
        return int(getattr(self._t.strategy, "magic", 0) if magic is None else magic)

    def buy(self, volume: float, sl: float | None = None, tp: float | None = None, *, deviation: int | None = None,
            magic: int | None = None, comment: str = "") -> TradeResult:
        return self._t.request_market(1, volume, sl=sl, tp=tp, deviation=deviation, magic=self._magic(magic),
                                      comment=comment)

    def sell(self, volume: float, sl: float | None = None, tp: float | None = None, *, deviation: int | None = None,
             magic: int | None = None, comment: str = "") -> TradeResult:
        return self._t.request_market(-1, volume, sl=sl, tp=tp, deviation=deviation, magic=self._magic(magic),
                                      comment=comment)

    def position_close(self, ticket: int, volume: float | None = None, *, deviation: int | None = None,
                       reason: str = "signal") -> TradeResult:
        """reason は取引の記録に残す理由(分析用。MQL5 には無い)。"""
        return self._t.request_close(ticket, volume, deviation=deviation, reason=reason)

    def position_close_by(self, ticket: int, by: int) -> TradeResult:
        return self._t.request_close_by(ticket, by)

    def position_modify(self, ticket: int, sl: float | None, tp: float | None) -> TradeResult:
        return self._t.request_modify(ticket, sl, tp)

    def close_all(self, magic: int | None = None) -> list[TradeResult]:
        """建玉を全部決済し、待機注文を全部消す(magic を指せばその EA の分だけ)。"""
        out = [self.position_close(p.ticket) for p in self.positions(magic)]
        out += [self.order_delete(o.ticket) for o in self.orders(magic)]
        return out

    # ---- 待機注文 ----
    def _pending(self, otype, volume, price, sl, tp, stoplimit, type_time, expiration, magic, comment):
        return self._t.request_pending(otype, volume, price, sl=sl, tp=tp, stoplimit=stoplimit, type_time=type_time,
                                       expiration=expiration, magic=self._magic(magic), comment=comment)

    def buy_limit(self, volume, price, sl=None, tp=None, *, type_time=ORDER_TIME_GTC, expiration=None, magic=None,
                  comment="") -> TradeResult:
        return self._pending(ORDER_TYPE_BUY_LIMIT, volume, price, sl, tp, None, type_time, expiration, magic, comment)

    def sell_limit(self, volume, price, sl=None, tp=None, *, type_time=ORDER_TIME_GTC, expiration=None, magic=None,
                   comment="") -> TradeResult:
        return self._pending(ORDER_TYPE_SELL_LIMIT, volume, price, sl, tp, None, type_time, expiration, magic, comment)

    def buy_stop(self, volume, price, sl=None, tp=None, *, type_time=ORDER_TIME_GTC, expiration=None, magic=None,
                 comment="") -> TradeResult:
        return self._pending(ORDER_TYPE_BUY_STOP, volume, price, sl, tp, None, type_time, expiration, magic, comment)

    def sell_stop(self, volume, price, sl=None, tp=None, *, type_time=ORDER_TIME_GTC, expiration=None, magic=None,
                  comment="") -> TradeResult:
        return self._pending(ORDER_TYPE_SELL_STOP, volume, price, sl, tp, None, type_time, expiration, magic, comment)

    def buy_stop_limit(self, volume, price, stoplimit, sl=None, tp=None, *, type_time=ORDER_TIME_GTC,
                       expiration=None, magic=None, comment="") -> TradeResult:
        return self._pending(ORDER_TYPE_BUY_STOP_LIMIT, volume, price, sl, tp, stoplimit, type_time, expiration,
                             magic, comment)

    def sell_stop_limit(self, volume, price, stoplimit, sl=None, tp=None, *, type_time=ORDER_TIME_GTC,
                        expiration=None, magic=None, comment="") -> TradeResult:
        return self._pending(ORDER_TYPE_SELL_STOP_LIMIT, volume, price, sl, tp, stoplimit, type_time, expiration,
                             magic, comment)

    def order_modify(self, ticket: int, price: float, sl=None, tp=None, *, stoplimit=None, type_time=None,
                     expiration=None) -> TradeResult:
        return self._t.request_order_modify(ticket, price, sl, tp, stoplimit=stoplimit, type_time=type_time,
                                            expiration=expiration)

    def order_delete(self, ticket: int) -> TradeResult:
        return self._t.request_order_delete(ticket)
