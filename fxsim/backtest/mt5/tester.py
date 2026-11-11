"""MT5 のストラテジーテスターを真似たバックテスト(1 銘柄)。

1 ペアのエンジン(backtest/engine.py)との違い:
    - 口座は両建て(hedging、建玉ごとにチケット)かネット(netting、銘柄ごとに 1 つ)。数量はロット
    - 注文は成行・指値・逆指値・ストップリミット。損切り・利確は価格(絶対値)で、建玉ごとに持つ。
      待機注文には期限を付けられる
    - EA(戦略)は「新しい足の最初のティック」で呼ばれる。on_bar(i) の i は確定した足で、そのときの気配は足 i+1 の始値。
      成行はその気配でその場で約定し、結果(TradeResult)がすぐ返る(MT5 の OnTick で新しい足を判定する書き方と同じ)
    - 足の中の発動(損切り・利確・待機注文・ストップアウト)は、経路の上で起きた順に 1 つずつ処理する
    - ストップアウト: 証拠金維持率(有効証拠金 / 必要証拠金 x 100)が stop_out_level を割ったら、
      損失の一番大きい建玉から 1 つずつ決済し、水準を戻すまで続ける。マージンコールの水準は数えるだけ
    - 証拠金が足りない注文は約定しない(10019)。1 ペアのエンジンのように量を勝手に減らさない
    - 損切り・利確・待機注文は、正しい側に stops_level(ポイント)以上離して置く(違反は 10015 / 10016)
足の中の判定(テスターのモデル、BrokerConfig.mt5.tester_model。spec.Mt5Options の説明と docs/MT5.md):
    every_tick  基準解像度(1 分)の経路の点と、点の間をブラウン橋とみなした「間で触れる確率」(1 ペアのエンジンと同じ)
    m1_points   経路の点だけ(点の間では触れない)
    m1_ohlc     経路の点で判定し、待機注文・損切り・利確は窓でも注文の価格で約定する(MT5 のテスターの 1 分足 OHLC)
    open_prices 足の始値・高値・安値・終値の 4 点で判定し、注文の価格で約定する(MT5 のテスターの「始値のみ」)
経路を持たない足(日足の再生など)は、1 つ前の終値 → 始値 → 安値 → 高値 → 終値(陰線なら高値が先)の点で判定する。
約定の価格(BrokerConfig のブローカーの振る舞いはそのまま使う):
    逆指値・損切り・ストップアウト  ふつうの 1 歩で越えたら水準の価格、飛び(窓)で越えたら飛んだ先の価格。
                                      不利な滑り(stop 倍)
    指値・利確                        水準の価格。飛びで越えたら、positive_slippage なら飛んだ先(有利)の価格
    成行                              気配 + 滑り。ラストルック(last_look_prob)で拒否されることがある
                                      (market 実行は 10006、instant 実行は 10004。
                                      instant は deviation を超える滑りも 10004)
ストップアウト(BrokerConfig.mt5.stop_out): largest_first は損失の一番大きい建玉から 1 つずつ、水準を戻すまで
(実際の業者)。all は全部の建玉と待機注文を一度に閉じる(MT5 のテスター)。
費用・スワップ・破綻の数え方・成績の形(trades / equity_curve / stats)は 1 ペアのエンジンと同じ。

足の中の判定は「符号をそろえた価格 q が水準 lvl 以下に触れたか」の 1 つの形にまとめる(q = sgn x 気配):
    買いの損切り   q = bid,  lvl = sl        売りの損切り   q = −ask, lvl = −sl
    買いの利確     q = −bid, lvl = −tp       売りの利確     q = ask,  lvl = tp
    買いの指値     q = ask,  lvl = price     売りの指値     q = −bid, lvl = −price
    買いの逆指値   q = −ask, lvl = −price    売りの逆指値   q = bid,  lvl = price
この形では、客に不利な向きは常に q の下向き。
"""
from __future__ import annotations

import itertools

import numpy as np

from ...calendar import NS_DAY, NS_HOUR, NS_MIN, trading_day
from ...data import Bars
from ..broker import BrokerConfig
from ..engine import Backtester, EngineConfig, ProgressFn, Trade, _first_touch
from ..schedule import TradingSchedule
from .orders import LIMIT_TYPES, STOP_LIMIT_TYPES, Mt5Deal, Mt5Order, Mt5Position, TradeResult, order_side
from .spec import (
    DEAL_ENTRY_IN,
    DEAL_ENTRY_INOUT,
    DEAL_ENTRY_OUT,
    DEAL_ENTRY_OUT_BY,
    DEAL_REASON_CLIENT,
    DEAL_REASON_EXPERT,
    DEAL_REASON_SL,
    DEAL_REASON_SO,
    DEAL_REASON_TP,
    ORDER_TIME_DAY,
    ORDER_TIME_GTC,
    ORDER_TIME_SPECIFIED,
    ORDER_TIME_SPECIFIED_DAY,
    ORDER_TYPE_BUY_LIMIT,
    ORDER_TYPE_BUY_STOP,
    ORDER_TYPE_BUY_STOP_LIMIT,
    ORDER_TYPE_SELL_LIMIT,
    TRADE_RETCODE_DONE,
    TRADE_RETCODE_FROZEN,
    TRADE_RETCODE_INVALID,
    TRADE_RETCODE_INVALID_EXPIRATION,
    TRADE_RETCODE_INVALID_PRICE,
    TRADE_RETCODE_INVALID_STOPS,
    TRADE_RETCODE_INVALID_VOLUME,
    TRADE_RETCODE_LIMIT_POSITIONS,
    TRADE_RETCODE_MARKET_CLOSED,
    TRADE_RETCODE_NO_CHANGES,
    TRADE_RETCODE_NO_MONEY,
    TRADE_RETCODE_PLACED,
    TRADE_RETCODE_POSITION_CLOSED,
    TRADE_RETCODE_REJECT,
    TRADE_RETCODE_REQUOTE,
    specs,
)

# 同じ点で複数起きたときの順番(小さいほど先)
_PRIO = {"so": 0, "sl": 1, "hunt": 1, "tp": 2, "order": 3}


class _Ticks:
    """足 i の中の気配の並び(先頭は 1 つ前の点)。"""

    def __init__(self, mid, bid, ask, gap: float, sig2: float, bridge: bool, step_ns: int,
                 opens: np.ndarray | None = None):
        self.mid, self.bid, self.ask = mid, bid, ask
        self.gap, self.sig2, self.bridge, self.step_ns = gap, sig2, bridge, step_ns
        self.opens = opens          # 実データの OHLC の点(real_pair)なら、点ごとに「足の始値の点か」

    def __len__(self) -> int:
        return len(self.mid)


class Mt5Tester(Backtester):
    def __init__(self, bars: Bars, strategy, broker: BrokerConfig | None = None,
                 config: EngineConfig | None = None, pip: float = 0.01, seed: int = 0,
                 schedule: TradingSchedule | None = None, rate_diff: np.ndarray | None = None,
                 symbol: str = "USDJPY"):
        super().__init__(bars, strategy, broker, config, pip=pip, seed=seed, schedule=schedule, rate_diff=rate_diff)
        self.sym, self.acct, self.opt = specs(self.broker, pip, symbol)
        self.positions: dict[int, Mt5Position] = {}
        self.orders: dict[int, Mt5Order] = {}
        self.deals: list[Mt5Deal] = []
        self.history_orders: list[tuple[Mt5Order, str]] = []     # (注文, filled / canceled / expired / rejected)
        self.results: list[TradeResult] = []
        self._ticket = itertools.count(1)
        self._hunt_draw: dict[tuple[int, int], float] = {}
        self._below_mc = False
        self.stats.update({"stop_out": 0, "margin_call": 0, "requote": 0, "no_money": 0, "invalid": 0,
                           "expired": 0, "triggered": 0})
        self.ticks: _Ticks | None = None
        self.k = 0                  # 足の中の今の点(ticks の位置)
        self.margin_curve = np.full(len(bars), np.nan)

    # ======================================================================
    # 気配
    # ======================================================================
    def _make_ticks(self, i: int) -> _Ticks:
        model = self.opt.tester_model
        path = self.bars.path(i, prev=True) if model != "open_prices" and self.broker.intrabar == "path" else None
        if path is not None and len(path[0]) >= 2:
            mid, spread = path
            half = (spread + 2 * self._widen(spread)) / 2
            opens = self.bars.open_steps(i)
            if opens is not None:          # 実データの OHLC の点: 足の中は連続、飛びは始値の点だけ
                gap, sig2 = 0.0, 0.0
            elif len(mid) > 10:
                med = max(float(np.median(np.abs(np.diff(mid)))), 1e-12)
                gap, sig2 = self.broker.gap_k * med, (med / 0.6745) ** 2
            else:
                gap, sig2 = 0.0, 0.0
            step = self.bars.timeframe * NS_MIN // max(1, len(mid) - 1)
            return _Ticks(mid, mid - half, mid + half, gap, sig2,
                          model == "every_tick" and self.broker.touch == "bridge" and sig2 > 0, step, opens)
        prev = self.c[i - 1] if i > 0 else self.o[i]
        o, c = self.o[i], self.c[i]
        h, lo = float(self.bars.high[i]), float(self.bars.low[i])
        mid = np.array([prev, o, lo, h, c] if c >= o else [prev, o, h, lo, c])
        half = np.full(len(mid), self.sm[i] / 2)
        if i > 0:
            half[0] = (self.ac[i - 1] - self.bc[i - 1]) / 2
        half[1] = (self.ao[i] - self.bo[i]) / 2
        return _Ticks(mid, mid - half, mid + half, 0.0, 0.0, False, 0)

    def _bid(self) -> float:
        return float(self.ticks.bid[self.k])

    def _ask(self) -> float:
        return float(self.ticks.ask[self.k])

    def _mid(self) -> float:
        return float(self.ticks.mid[self.k])

    def _now(self) -> int:
        """今の点の時刻(ns)。経路の点は足の始まりから 1 歩ずつ、経路の無い足は足の始まり。"""
        return int(self.t[self.i]) + max(self.k - 1, 0) * (self.ticks.step_ns if self.ticks is not None else 0)

    # ======================================================================
    # 証拠金
    # ======================================================================
    def _hedge_volume(self, long_v: float, short_v: float) -> float:
        m = self.acct.hedged_margin
        if m == "full":
            return long_v + short_v
        if m == "larger_leg":
            return max(long_v, short_v)
        return abs(long_v - short_v) + 2 * self.acct.margin_hedged_rate * min(long_v, short_v)

    def _legs(self, extra_side: int = 0, extra_volume: float = 0.0) -> tuple[float, float]:
        long_v = sum(p.volume for p in self.positions.values() if p.side > 0)
        short_v = sum(p.volume for p in self.positions.values() if p.side < 0)
        if self.acct.mode == "netting":
            net = long_v - short_v + extra_side * extra_volume
            return max(net, 0.0), max(-net, 0.0)
        if extra_side > 0:
            long_v += extra_volume
        elif extra_side < 0:
            short_v += extra_volume
        return long_v, short_v

    def margin(self, price: float | None = None, extra_side: int = 0, extra_volume: float = 0.0) -> float:
        """必要証拠金(円)。USDJPY の円口座: ロット x 1 ロットの通貨量 x 価格 / レバレッジ。"""
        price = self._mid() if price is None else price
        return self._hedge_volume(*self._legs(extra_side, extra_volume)) * self.sym.contract_size * price \
            / self.acct.leverage

    def floating(self) -> float:
        bid, ask = self._bid(), self._ask()
        return sum(p.profit(bid, ask, self.sym.contract_size) for p in self.positions.values())

    def margin_level(self) -> float:
        m = self.margin()
        return (self.balance + self.floating()) / m * 100.0 if m > 0 else float("inf")

    def _money_ok(self, side: int, volume: float) -> bool:
        need = self.margin(extra_side=side, extra_volume=volume) - self.margin()
        free = self.balance + self.floating() - self.margin()
        return need <= 0 or need <= free

    # ======================================================================
    # 約定と記録
    # ======================================================================
    def _deal(self, *, order: int, position: int, side: int, entry: int, volume: float, price: float, mid: float,
              profit: float, commission: float, swap: float, reason: int, magic: int, comment: str) -> Mt5Deal:
        d = Mt5Deal(next(self._ticket), order, position, self._now(), 0 if side > 0 else 1, entry, volume, price,
                    profit, commission, swap, reason, magic, comment, mid, self.i)
        self.deals.append(d)
        return d

    def _commission_lots(self, volume: float) -> float:
        fee = self.sym.commission_per_lot * volume
        self.balance -= fee
        self.stats["commission_total"] += fee
        return fee

    def _open_position(self, side: int, volume: float, price: float, mid: float, slip: float, *, sl, tp, magic,
                       comment, order: int, reason: int) -> Mt5Position:
        fee = self._commission_lots(volume)
        units = self.sym.units(volume)
        self.stats["slippage_total"] += slip * units
        p = Mt5Position(next(self._ticket), side, volume, price, sl, tp, self._now(), magic, comment, mid, self.i,
                        0.0, fee, slip * units, self.k)
        self.positions[p.ticket] = p
        self._deal(order=order, position=p.ticket, side=side, entry=DEAL_ENTRY_IN, volume=volume, price=price, mid=mid,
                   profit=0.0, commission=fee, swap=0.0, reason=reason, magic=magic, comment=comment)
        return p

    def _close_position(self, p: Mt5Position, volume: float, price: float, mid: float, slip: float, reason: str, *,
                        order: int = 0, deal_reason: int = DEAL_REASON_EXPERT, entry: int = DEAL_ENTRY_OUT,
                        commission: bool = True) -> None:
        """建玉 p のうち volume ロットを price で決済する。スワップ・手数料・滑りは数量で按分して、その取引に付ける。"""
        volume = min(volume, p.volume)
        frac = volume / p.volume if p.volume > 0 else 1.0
        units = self.sym.units(volume)
        fee = self._commission_lots(volume) if commission else 0.0
        self.stats["slippage_total"] += slip * units
        swap, comm, slp = p.swap * frac, p.commission * frac, p.slippage * frac
        profit = (price - p.price_open) * p.side * units
        self.balance += profit
        pips = (price - p.price_open) * p.side / self.pip
        gross = (mid - p.open_mid) * p.side / self.pip
        self.trades.append(Trade(p.side, units, int(self.t[p.open_idx]), int(self.t[self.i]), p.price_open, price,
                                 profit + swap - comm - fee, pips, gross, gross - pips, swap, reason,
                                 self.i - p.open_idx, comm + fee, slp + slip * units, self.sym.name))
        self._deal(order=order, position=p.ticket, side=-p.side, entry=entry, volume=volume, price=price, mid=mid,
                   profit=profit, commission=fee, swap=swap, reason=deal_reason, magic=p.magic, comment=p.comment)
        p.volume = round(p.volume - volume, 8)
        p.swap -= swap
        p.commission -= comm
        p.slippage -= slp
        if p.volume <= 1e-9:
            del self.positions[p.ticket]
        if self.balance < 0 and self.broker.zero_cut and not self.positions:
            self.stats["zero_cut"] += -self.balance          # ゼロカット: マイナス残高はブローカー持ち
            self.balance = 0.0

    def _fill_side(self, side: int, volume: float, price: float, mid: float, slip: float, *, sl, tp, magic, comment,
                   order: int, reason: int, position: int = 0, close_reason: str = "signal") -> None:
        """約定を口座に反映する。hedging なら新しい建玉(position を指せばその建玉の決済)、
        netting なら銘柄の 1 つの建玉に足す・減らす・ドテンする(取得単価は数量の加重平均)。"""
        if position:
            p = self.positions.get(position)
            if p is not None:
                self._close_position(p, volume, price, mid, slip, close_reason, order=order, deal_reason=reason)
            return
        if self.acct.mode == "hedging" or not self.positions:
            self._open_position(side, volume, price, mid, slip, sl=sl, tp=tp, magic=magic, comment=comment,
                                order=order, reason=reason)
            return
        p = next(iter(self.positions.values()))
        if p.side == side:
            fee = self._commission_lots(volume)
            units = self.sym.units(volume)
            total = p.volume + volume
            p.price_open = (p.price_open * p.volume + price * volume) / total
            p.open_mid = (p.open_mid * p.volume + mid * volume) / total
            p.volume = round(total, 8)
            p.commission += fee
            p.slippage += slip * units
            self.stats["slippage_total"] += slip * units
            if sl is not None:
                p.sl = sl
            if tp is not None:
                p.tp = tp
            self._deal(order=order, position=p.ticket, side=side, entry=DEAL_ENTRY_IN, volume=volume, price=price,
                       mid=mid, profit=0.0, commission=fee, swap=0.0, reason=reason, magic=magic, comment=comment)
            return
        rest = round(volume - p.volume, 8)
        close_v = min(volume, p.volume)
        self._close_position(p, close_v, price, mid, slip, "reverse" if rest > 0 else close_reason, order=order,
                             deal_reason=reason, entry=DEAL_ENTRY_INOUT if rest > 0 else DEAL_ENTRY_OUT)
        if rest > 1e-9:
            self._open_position(side, rest, price, mid, slip, sl=sl, tp=tp, magic=magic, comment=comment,
                                order=order, reason=reason)

    # ======================================================================
    # 取引の要求(Mt5Context から呼ぶ)
    # ======================================================================
    def _result(self, retcode: int, **kw) -> TradeResult:
        r = TradeResult(retcode, bid=self._bid(), ask=self._ask(), **kw)
        if retcode not in (TRADE_RETCODE_DONE, TRADE_RETCODE_PLACED, TRADE_RETCODE_NO_CHANGES) \
                and not r.extra.get("counted"):
            key = {TRADE_RETCODE_REQUOTE: "requote", TRADE_RETCODE_NO_MONEY: "no_money"}.get(retcode, "invalid")
            self.stats[key] += 1
            if retcode == TRADE_RETCODE_NO_MONEY:
                self.stats["blocked"] += 1
        self.results.append(r)
        return r

    def _log(self, *items) -> None:
        """先読み検査用の判断の記録(確定した足の番号から始める)。"""
        self.decisions.append((self.i - 1, *[round(x, 6) if isinstance(x, float) else x for x in items]))

    def _stops_ok(self, side: int, ref_close: float, sl, tp) -> bool:
        """損切り・利確が正しい側にあり、stops_level 以上離れているか。ref_close は建玉を閉じる側の価格。"""
        dist = max(self.sym.stops_level * self.sym.point, 1e-12)
        if sl is not None and (ref_close - sl) * side < dist:
            return False
        return tp is None or (tp - ref_close) * side >= dist

    def _norm(self, x: float | None) -> float | None:
        return None if x is None else round(float(x), self.sym.digits)

    def _entry_allowed(self) -> bool:
        return bool(self.can_enter[max(self.i - 1, 0)])

    def request_market(self, side: int, volume: float, *, sl=None, tp=None, deviation: int | None = None,
                       magic: int = 0, comment: str = "", position: int = 0, reason: str = "signal") -> TradeResult:
        """成行(TRADE_ACTION_DEAL)。position を指せば、その建玉の決済(hedging)。
        reason は決済したときの取引の記録(Trade.reason)に残す理由(MQL5 には無い。分析用)。"""
        sl, tp = self._norm(sl), self._norm(tp)
        volume = round(float(volume), 8)
        self._log("deal", side, volume, position, sl, tp)
        if not self.sym.valid_volume(volume):
            return self._result(TRADE_RETCODE_INVALID_VOLUME)
        if position and position not in self.positions:
            return self._result(TRADE_RETCODE_POSITION_CLOSED)
        reducing = bool(position) or (self.acct.mode == "netting"
                                      and any(p.side == -side for p in self.positions.values()))
        if not reducing:
            if not self._entry_allowed():
                return self._result(TRADE_RETCODE_MARKET_CLOSED)
            if self.acct.mode == "hedging" and len(self.positions) + len(self.orders) >= self.acct.max_positions:
                return self._result(TRADE_RETCODE_LIMIT_POSITIONS)
            if not self._stops_ok(side, self._bid() if side > 0 else self._ask(), sl, tp):
                return self._result(TRADE_RETCODE_INVALID_STOPS)
            if self._last_look_rejects(self.i, side):         # stats["rejected"] はそちらで数える
                code = TRADE_RETCODE_REQUOTE if self.opt.execution == "instant" else TRADE_RETCODE_REJECT
                return self._result(code, comment="last look", extra={"counted": True})
        slip = self._slip(self.i)
        if self.opt.execution == "instant":
            dev = (self.opt.deviation_points if deviation is None else deviation) * self.sym.point
            if abs(slip) > dev + 1e-12:
                return self._result(TRADE_RETCODE_REQUOTE, comment="deviation")
        price = self._ask() + slip if side > 0 else self._bid() - slip
        if not reducing and not self._money_ok(side, volume):
            return self._result(TRADE_RETCODE_NO_MONEY)
        before = len(self.deals)
        self._fill_side(side, volume, price, self._mid(), slip, sl=sl, tp=tp, magic=magic, comment=comment,
                        order=next(self._ticket), reason=DEAL_REASON_EXPERT, position=position, close_reason=reason)
        return self._result(TRADE_RETCODE_DONE, deal=self.deals[-1].ticket if len(self.deals) > before else 0,
                            volume=volume, price=price)

    def request_close(self, ticket: int, volume: float | None = None, *, deviation: int | None = None,
                      reason: str = "signal") -> TradeResult:
        """建玉の決済(全部、または volume ロットだけ)。"""
        p = self.positions.get(ticket)
        if p is None:
            self._log("close", ticket, volume)
            return self._result(TRADE_RETCODE_POSITION_CLOSED)
        v = p.volume if volume is None else min(float(volume), p.volume)
        if self.acct.mode == "netting":
            return self.request_market(-p.side, v, deviation=deviation, magic=p.magic, reason=reason)
        return self.request_market(-p.side, v, deviation=deviation, magic=p.magic, position=ticket, reason=reason)

    def request_close_by(self, ticket: int, by: int) -> TradeResult:
        """両建ての相殺(TRADE_ACTION_CLOSE_BY)。少ない方の数量だけ、両方を by の建値で閉じる(売買の費用なし)。"""
        self._log("close_by", ticket, by)
        a, b = self.positions.get(ticket), self.positions.get(by)
        if self.acct.mode != "hedging" or a is None or b is None or a.side == b.side:
            return self._result(TRADE_RETCODE_INVALID)
        v = min(a.volume, b.volume)
        px, mid = b.price_open, b.open_mid
        self._close_position(a, v, px, mid, 0.0, "close_by", entry=DEAL_ENTRY_OUT_BY, commission=False)
        self._close_position(b, v, px, mid, 0.0, "close_by", entry=DEAL_ENTRY_OUT_BY, commission=False)
        return self._result(TRADE_RETCODE_DONE, volume=v, price=px)

    def request_modify(self, ticket: int, sl=None, tp=None) -> TradeResult:
        """建玉の損切り・利確を変える(TRADE_ACTION_SLTP)。None は「置かない」。"""
        sl, tp = self._norm(sl), self._norm(tp)
        self._log("sltp", ticket, sl, tp)
        p = self.positions.get(ticket)
        if p is None:
            return self._result(TRADE_RETCODE_POSITION_CLOSED)
        if p.sl == sl and p.tp == tp:
            return self._result(TRADE_RETCODE_NO_CHANGES)
        ref = self._bid() if p.side > 0 else self._ask()
        if not self._stops_ok(p.side, ref, sl, tp):
            return self._result(TRADE_RETCODE_INVALID_STOPS)
        frz = self.sym.freeze_level * self.sym.point
        if frz > 0 and any(x is not None and abs(ref - x) <= frz for x in (p.sl, p.tp)):
            return self._result(TRADE_RETCODE_FROZEN)
        p.sl, p.tp = sl, tp
        return self._result(TRADE_RETCODE_DONE)

    def _pending_price_ok(self, otype: int, price: float) -> bool:
        dist = max(self.sym.stops_level * self.sym.point, 1e-12)
        bid, ask = self._bid(), self._ask()
        if otype == ORDER_TYPE_BUY_LIMIT:
            return ask - price >= dist
        if otype == ORDER_TYPE_SELL_LIMIT:
            return price - bid >= dist
        if otype in (ORDER_TYPE_BUY_STOP, ORDER_TYPE_BUY_STOP_LIMIT):
            return price - ask >= dist
        return bid - price >= dist                      # 売りの逆指値・ストップリミット

    def _day_end(self, t_ns: int) -> int:
        return int((trading_day(t_ns) + 1) * NS_DAY - 2 * NS_HOUR)

    def _expiry(self, type_time: int, expiration: int | None) -> int | None | bool:
        """期限の時刻(ns)。None = 期限なし。False = 不正。"""
        if type_time == ORDER_TIME_GTC:
            return None
        if type_time == ORDER_TIME_DAY:
            return self._day_end(self._now())
        if expiration is None or expiration <= self._now():
            return False
        if type_time == ORDER_TIME_SPECIFIED:
            return int(expiration)
        if type_time == ORDER_TIME_SPECIFIED_DAY:
            return self._day_end(int(expiration))
        return False

    def request_pending(self, otype: int, volume: float, price: float, *, sl=None, tp=None, stoplimit=None,
                        type_time: int = ORDER_TIME_GTC, expiration: int | None = None, magic: int = 0,
                        comment: str = "") -> TradeResult:
        """待機注文を置く(TRADE_ACTION_PENDING)。"""
        price, sl, tp, stoplimit = self._norm(price), self._norm(sl), self._norm(tp), self._norm(stoplimit)
        volume = round(float(volume), 8)
        self._log("pending", otype, volume, price, sl, tp, stoplimit, type_time)
        if not self.sym.valid_volume(volume):
            return self._result(TRADE_RETCODE_INVALID_VOLUME)
        if not self._entry_allowed():
            return self._result(TRADE_RETCODE_MARKET_CLOSED)
        if len(self.positions) + len(self.orders) >= self.acct.max_positions:
            return self._result(TRADE_RETCODE_LIMIT_POSITIONS)
        if not self._pending_price_ok(otype, price):
            return self._result(TRADE_RETCODE_INVALID_PRICE)
        side = order_side(otype)
        if otype in STOP_LIMIT_TYPES and (stoplimit is None or (stoplimit - price) * side > 0):
            return self._result(TRADE_RETCODE_INVALID_PRICE, comment="stoplimit")
        fill_ref = stoplimit if otype in STOP_LIMIT_TYPES else price
        if not self._stops_ok(side, fill_ref - side * (self._ask() - self._bid()), sl, tp):
            return self._result(TRADE_RETCODE_INVALID_STOPS)
        exp = self._expiry(type_time, expiration)
        if exp is False:
            return self._result(TRADE_RETCODE_INVALID_EXPIRATION)
        o = Mt5Order(next(self._ticket), otype, volume, price, sl, tp, self._now(), type_time, exp, stoplimit,
                     magic, comment, 0, self.i, self.k)
        self.orders[o.ticket] = o
        return self._result(TRADE_RETCODE_PLACED, order=o.ticket, volume=volume, price=price)

    def request_order_modify(self, ticket: int, price: float, sl=None, tp=None, *, stoplimit=None,
                             type_time: int | None = None, expiration: int | None = None) -> TradeResult:
        price, sl, tp = self._norm(price), self._norm(sl), self._norm(tp)
        self._log("order_modify", ticket, price, sl, tp)
        o = self.orders.get(ticket)
        if o is None:
            return self._result(TRADE_RETCODE_INVALID, comment="no such order")
        frz = self.sym.freeze_level * self.sym.point
        ref = self._ask() if o.side > 0 else self._bid()
        if frz > 0 and abs(ref - o.price) <= frz:
            return self._result(TRADE_RETCODE_FROZEN)
        if not self._pending_price_ok(o.type, price):
            return self._result(TRADE_RETCODE_INVALID_PRICE)
        fill_ref = (self._norm(stoplimit) if stoplimit is not None else (o.stoplimit or price))
        if not self._stops_ok(o.side, fill_ref - o.side * (self._ask() - self._bid()), sl, tp):
            return self._result(TRADE_RETCODE_INVALID_STOPS)
        if type_time is not None:
            exp = self._expiry(type_time, expiration)
            if exp is False:
                return self._result(TRADE_RETCODE_INVALID_EXPIRATION)
            o.type_time, o.expiration = type_time, exp
        o.price, o.sl, o.tp = price, sl, tp
        if stoplimit is not None:
            o.stoplimit = self._norm(stoplimit)
        o.placed_idx, o.placed_at = self.i, self.k
        return self._result(TRADE_RETCODE_DONE, order=ticket)

    def request_order_delete(self, ticket: int) -> TradeResult:
        self._log("order_delete", ticket)
        o = self.orders.pop(ticket, None)
        if o is None:
            return self._result(TRADE_RETCODE_INVALID, comment="no such order")
        self.history_orders.append((o, "canceled"))
        return self._result(TRADE_RETCODE_DONE, order=ticket)

    # ======================================================================
    # 足の中の発動
    # ======================================================================
    def _touch(self, q: np.ndarray, level: float, start: int) -> tuple[int, bool]:
        """q[start:] の先頭を「1 つ前の点」として、level 以下に触れた最初の点(ticks の位置)と、点で越えたか。"""
        tk = self.ticks
        seg = q[start:]
        if len(seg) < 2:
            return -1, False
        u = self.touch_rng.random(len(seg) - 1) if tk.bridge else None
        j, cross = _first_touch(seg, level, tk.sig2, tk.gap, u)
        return (start + j, cross) if j > 0 else (-1, False)

    def _quote(self, sgn: int, use_ask: bool, stop: int) -> np.ndarray:
        tk = self.ticks
        return sgn * (tk.ask[: stop + 1] if use_ask else tk.bid[: stop + 1])

    def _candidates(self, k0: int, stop: int) -> list[tuple]:
        """点 k0 より後(stop まで)に起きる発動の候補 (点, 優先, 種類, 対象, 越えたか, 水準, q, 符号)。"""
        out = []
        for p in list(self.positions.values()):
            start = max(k0, p.active_at if p.open_idx == self.i else 0)
            if start >= stop:
                continue
            if p.sl is not None:                # 買いは bid が下がって届く / 売りは ask が上がって届く
                sgn = 1 if p.side > 0 else -1
                q = self._quote(sgn, p.side < 0, stop)
                lvl = sgn * p.sl
                j, cross = self._touch(q, lvl, start)
                if j > 0:
                    out.append((j, _PRIO["sl"], "sl", p, cross, lvl, q, sgn))
                elif self.broker.hunt_prob > 0:
                    near = np.flatnonzero(q[start + 1:] - lvl <= self.broker.hunt_pips * self.pip)
                    key = (p.ticket, self.i)
                    if key not in self._hunt_draw:
                        self._hunt_draw[key] = float(self.rng.random())
                    if len(near) and self._hunt_draw[key] < self.broker.hunt_prob:
                        out.append((start + 1 + int(near[0]), _PRIO["hunt"], "hunt", p, False, lvl, q, sgn))
            if p.tp is not None:                # 買いは bid が上がって届く / 売りは ask が下がって届く
                sgn = -1 if p.side > 0 else 1
                q = self._quote(sgn, p.side < 0, stop)
                lvl = sgn * p.tp
                j, cross = self._touch(q, lvl, start)
                if j > 0:
                    out.append((j, _PRIO["tp"], "tp", p, cross, lvl, q, sgn))
        for o in list(self.orders.values()):
            start = max(k0, o.placed_at if o.placed_idx == self.i else 0)
            if start >= stop:
                continue
            limit = o.type in LIMIT_TYPES
            # 指値: 買いは ask が下がって、売りは bid が上がって届く
            # 逆指値: 買いは ask が上がって、売りは bid が下がって届く
            sgn = (1 if o.side > 0 else -1) if limit else (-1 if o.side > 0 else 1)
            q = self._quote(sgn, o.side > 0, stop)
            lvl = sgn * o.price
            j, cross = self._touch(q, lvl, start)
            if j > 0:
                out.append((j, _PRIO["order"], "order", o, cross, lvl, q, sgn))
        if self.positions:
            j = self._stop_out_point(k0, stop)
            if j > 0:
                out.append((j, _PRIO["so"], "so", None, True, 0.0, None, 1))
        return out

    def _stop_out_point(self, k0: int, stop: int) -> int:
        """点 k0 より後で、証拠金維持率が stop_out_level 以下になる最初の点。"""
        tk = self.ticks
        hv = self._hedge_volume(*self._legs())
        if hv <= 0 or stop <= k0:
            return -1
        cs = self.sym.contract_size
        bid, ask, mid = tk.bid[k0 + 1: stop + 1], tk.ask[k0 + 1: stop + 1], tk.mid[k0 + 1: stop + 1]
        eq = np.full(len(mid), self.balance)
        for p in self.positions.values():
            eq = eq + ((bid if p.side > 0 else ask) - p.price_open) * p.side * p.volume * cs
        level = eq / (hv * cs * mid / self.acct.leverage) * 100.0
        hit = np.flatnonzero(level <= self.acct.stop_out_level)
        return k0 + 1 + int(hit[0]) if len(hit) else -1

    def _level_fill(self, q: np.ndarray, sgn: int, j: int, cross: bool, lvl: float,
                    stop_like: bool) -> tuple[float, float, float]:
        """q で水準 lvl に触れた点 j の、約定価格(実際の価格)・その瞬間の mid・滑り(価格、不利を正)。

        逆指値の型(損切り・逆指値): ふつうの 1 歩なら水準、飛びなら飛んだ先。どちらも不利な滑り(stop 倍)を足す。
        指値の型(利確・指値): 水準。飛びで positive_slippage なら飛んだ先(有利)。
        経路の無い足(4 点)は、始値の点(j = 1)で越えたときだけ飛び(窓)とみなす。
        """
        tk = self.ticks
        if tk.opens is not None:
            jumped = cross and (q[j - 1] <= lvl or bool(tk.opens[j]))
        else:
            jumped = cross and (q[j - 1] <= lvl or (tk.gap > 0 and q[j - 1] - q[j] > tk.gap)
                                or (tk.gap == 0 and j == 1))
        if self.opt.order_price_fills:          # MT5 のテスターの OHLC 系: 窓でも注文の価格
            jumped = False
        slip = 0.0
        if stop_like:
            slip = self._slip(self.i, stop=True)
            fill_q = (min(q[j], lvl) if jumped else lvl) - slip       # 不利な向きは q の下向き
        else:
            fill_q = q[j] if (self.broker.positive_slippage and jumped and q[j] < lvl) else lvl
        quote_j = q[j] * sgn
        at = float(tk.mid[j]) if jumped else float((lvl if stop_like else fill_q) * sgn + (tk.mid[j] - quote_j))
        return float(fill_q * sgn), at, float(slip)

    def _execute(self, ev: tuple) -> None:
        j, _, kind, obj, cross, lvl, q, sgn = ev
        self.k = j
        if kind == "so":                        # 損失の一番大きい建玉から 1 つ決済する(all なら全部)
            self.stats["stop_out"] += 1
            self.stats["losscut"] += 1
            bid, ask = self._bid(), self._ask()
            if self.acct.stop_out == "all":
                for p in list(self.positions.values()):
                    slip = self._slip(self.i, stop=True)
                    px = bid - slip if p.side > 0 else ask + slip
                    self._close_position(p, p.volume, px, self._mid(), slip, "losscut", deal_reason=DEAL_REASON_SO)
                for o in list(self.orders.values()):
                    del self.orders[o.ticket]
                    self.history_orders.append((o, "canceled"))
                return
            worst = min(self.positions.values(), key=lambda p: p.profit(bid, ask, self.sym.contract_size))
            slip = self._slip(self.i, stop=True)
            px = bid - slip if worst.side > 0 else ask + slip
            self._close_position(worst, worst.volume, px, self._mid(), slip, "losscut", deal_reason=DEAL_REASON_SO)
            return
        if kind == "hunt":                      # ストップ狩り: 届いていないのに損切りの価格で刺される
            p: Mt5Position = obj
            self.stats["hunted"] += 1
            slip = self._slip(self.i, stop=True)
            px = p.sl - p.side * slip
            quote = self.ticks.bid[j] if p.side > 0 else self.ticks.ask[j]
            self._close_position(p, p.volume, px, float(p.sl + self.ticks.mid[j] - quote), slip, "hunted",
                                 deal_reason=DEAL_REASON_SL)
            return
        if kind in ("sl", "tp"):
            p = obj
            fill, at, slip = self._level_fill(q, sgn, j, cross, lvl, stop_like=(kind == "sl"))
            self._close_position(p, p.volume, fill, at, slip, kind,
                                 deal_reason=DEAL_REASON_SL if kind == "sl" else DEAL_REASON_TP)
            return
        o: Mt5Order = obj
        self.stats["triggered"] += 1
        if o.type in STOP_LIMIT_TYPES:          # 発動したら、stoplimit の指値に置き換える(同じチケット)
            o.type = ORDER_TYPE_BUY_LIMIT if o.side > 0 else ORDER_TYPE_SELL_LIMIT
            o.price, o.stoplimit = o.stoplimit, None
            o.placed_idx, o.placed_at = self.i, j
            return
        del self.orders[o.ticket]
        fill, at, slip = self._level_fill(q, sgn, j, cross, lvl, stop_like=o.type not in LIMIT_TYPES)
        if not self._money_ok(o.side, o.volume):
            self.stats["no_money"] += 1
            self.stats["blocked"] += 1
            self.history_orders.append((o, "rejected"))
            return
        self.history_orders.append((o, "filled"))
        self._fill_side(o.side, o.volume, fill, at, slip, sl=o.sl, tp=o.tp, magic=o.magic, comment=o.comment,
                        order=o.ticket, reason=DEAL_REASON_EXPERT)

    def _process(self, k0: int, stop: int) -> None:
        """点 k0 より後、stop までに起きる発動を、起きた順に 1 つずつ処理する。"""
        for _ in range(100_000):
            cands = self._candidates(k0, stop)
            if not cands:
                break
            ev = min(cands, key=lambda e: (e[0], e[1]))
            self._execute(ev)
            k0 = ev[0] - 1                      # 同じ点で起きるほかの発動も拾う(処理した対象は消えている)
        else:
            raise RuntimeError("too many intrabar events")
        self.k = stop

    def _expire(self) -> None:
        now = int(self.t[self.i])
        for o in [o for o in self.orders.values() if o.expiration is not None and o.expiration <= now]:
            del self.orders[o.ticket]
            self.history_orders.append((o, "expired"))
            self.stats["expired"] += 1

    # ======================================================================
    # スワップ・評価・破綻
    # ======================================================================
    def _swap_amount(self, side: int, volume: float, i: int, days: float) -> float:
        b = self.broker
        pts = self.sym.swap_points_long if side > 0 else self.sym.swap_points_short
        units = self.sym.units(volume)
        if pts is not None:
            return pts * self.sym.point * units * days
        if b.swap_mode == "rates":
            return units * self.c[max(i - 1, 0)] * (side * self.rate_diff[i] - b.swap_markup) * days / 365
        return (b.swap_long_per_10k if side > 0 else b.swap_short_per_10k) * days * units / 10_000

    def _swap(self, i: int) -> None:
        days = self.swap_days[i]
        if not days:
            return
        for p in self.positions.values():
            amt = self._swap_amount(p.side, p.volume, i, days)
            p.swap += amt
            self.balance += amt
            self.stats["swap_total"] += amt

    def swap_per_lot_day(self, side: int) -> float:
        """今の 1 ロット 1 日あたりのスワップ(円、受け取りが正)。費用の見積もり用。"""
        return self._swap_amount(side, 1.0, self.i, 1)

    def _mark_close(self, i: int) -> None:
        cs = self.sym.contract_size
        fl = sum(p.profit(self.bc[i], self.ac[i], cs) for p in self.positions.values())
        self.equity = self.balance + fl
        m = self._hedge_volume(*self._legs()) * cs * self.c[i] / self.acct.leverage
        self.margin_curve[i] = m
        below = m > 0 and self.equity / m * 100.0 <= self.acct.margin_call_level
        if below and not self._below_mc:
            self.stats["margin_call"] += 1
        self._below_mc = below

    def close_all(self, reason: str) -> None:
        for p in list(self.positions.values()):
            px = self._bid() if p.side > 0 else self._ask()
            self._close_position(p, p.volume, px, self._mid(), 0.0, reason, deal_reason=DEAL_REASON_CLIENT)
        for o in list(self.orders.values()):
            del self.orders[o.ticket]
            self.history_orders.append((o, "canceled"))

    def _check_ruin_mt5(self, i: int) -> None:
        self.min_equity = min(self.min_equity, self.equity)
        if not self.ruined and (self.equity <= 0 or self.equity <= self.cfg.initial_balance * self.cfg.ruin_frac):
            self.ruined = True
            self.ruin_idx = i
        if self.equity <= 0 and not self.positions:
            self.halted = True
        elif self.ruined and self.cfg.stop_at_ruin and not self.halted:
            self.close_all("ruin")
            self.equity = self.balance
            self.halted = True

    # ======================================================================
    # 実行
    # ======================================================================
    def run(self, progress: ProgressFn | None = None, every: int = 500) -> Mt5Tester:
        from .context import Mt5Context

        n = len(self.bars)
        self.strategy.prepare(self.bars, self.strategy_rng)
        ctx = Mt5Context(self)
        warmup = int(getattr(self.strategy, "warmup", 0))
        on_init = getattr(self.strategy, "on_init", None)
        if on_init is not None:
            on_init(ctx)
        for i in range(n):
            self.i = i
            if self.halted:
                self.equity_curve[i] = self.equity
                self.balance_curve[i] = self.balance
                continue
            self._swap(i)
            self._expire()
            self.ticks = self._make_ticks(i)
            last = len(self.ticks) - 1
            self.k = 0
            self._process(0, 1)                   # 始値のティック(窓)で起きる発動
            d = i - 1                             # 確定した足
            if d >= 0 and not self.halted:
                self.k = 1
                if self.force_close[d]:
                    self.close_all("schedule")
                elif d >= warmup and self.decide[d]:
                    self.strategy.on_bar(d, ctx)
            if last > 1:
                self._process(1, last)
            self.k = last
            self._mark_close(i)
            self._check_ruin_mt5(i)
            self.equity_curve[i] = self.equity
            self.balance_curve[i] = self.balance
            if progress is not None and (i % every == every - 1):
                progress(self, i)
        if self.ticks is not None:
            self.i = n - 1
            self.k = len(self.ticks) - 1
            self.close_all("end")
        self.equity = self.balance
        self.equity_curve[n - 1] = self.equity
        self.balance_curve[n - 1] = self.balance
        if progress is not None:
            progress(self, n - 1)
        return self


__all__ = ["Mt5Tester"]
