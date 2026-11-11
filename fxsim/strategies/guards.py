"""本命の戦略(strategies/main/)に共通の対策。

E012 の 4 枚の壁のうち、避け方が分かっている 1〜3 と、過剰な損失を止める仕組み。

    壁 1 費用        cost_ok: 見込みの値幅が、往復の費用(スプレッド + 滑り + 手数料 + 持つ間のスワップの支払い)の
                     cost_multiple 倍に届かなければ入らない。
                     TurnoverBudget: 新規の回数を 1 年あたり max_trades_per_year に抑える
    壁 2 スワップ    費用の見積もりに、持つ予定の日数分のスワップの支払いを入れる(受け取りは割り引かない = 当てにしない)
    壁 3 量          size_lots: 損切りまでの損失 = 有効証拠金 x risk_per_trade(固定比率)。負けた後に量を増やさない
                     (有効証拠金が減れば量も減る)。建玉の総額は有効証拠金 x max_leverage まで、証拠金の余力の 8 割まで
    狩り・窓         損切りは足の終値で判定し(close_stop)、サーバーには非常用の逆指値だけを
                     遠く(emergency_stop_mult 倍)に置く。
                     ストップ狩りと逆指値の滑り(malicious で 2.5 倍)を避け、窓・急変の大損だけを防ぐ
    止める仕組み     DrawdownBreaker: 有効証拠金が山から max_drawdown 下げたら新規を止める
                     (cooldown 本で再開。0 なら止めたまま)
    時間             ロールオーバー(22:00 UTC)の前後 avoid_rollover_hours 時間は新規を出さない(スプレッドが広がる)。
                     flat_before_weekend なら、金曜の最後の判断で手仕舞う(週末の窓を持ち越さない)
    口座の床         EquityFloor: 月初の有効証拠金 x protect を床にする(資産の 9 割を守る)。新規は、口座の全建玉が
                     サーバーの逆指値まで窓で滑って飛んだ最悪の場合(逆指値までの損失 x gap_mult)でも
                     有効証拠金が床を割らない量まで(floor_lots)。床を割ったら全部手仕舞い、その月は新規を出さない。
                     逆指値の無い建玉が口座にあれば、最悪の損失が分からないので新規を出さない
どれも「負けを減らす」ための仕組みで、優位性は作らない(E012)。勝つには、戦略の側に費用を越える情報が要る。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from ..calendar import NS_DAY, NS_HOUR, weekday


@dataclass
class GuardConfig:
    risk_per_trade: float = 0.005        # 損切り(終値の判定)までの損失 / 有効証拠金
    max_leverage: float = 3.0            # 建玉の総額 / 有効証拠金 の上限(口座のレバレッジとは別の、自分の上限)
    max_trades_per_year: float = 24.0    # 新規の回数の上限(直近 365 日)
    min_hold_bars: int = 0               # 建ててからこの本数は、損切り以外で手仕舞わない
    cost_multiple: float = 3.0           # 見込みの値幅 ≥ 往復の費用の見積もり x これ でなければ入らない(0 = 見ない)
    slippage_pips_est: float = 0.5       # 費用の見積もりに使う片道の滑り(pip)
    hold_days_est: float = 20.0          # スワップの見積もりに使う、持つ予定の日数
    close_stop: bool = True              # 損切りを足の終値で判定する(False ならサーバーの逆指値)
    emergency_stop_mult: float = 3.0     # close_stop のとき、損切りの幅のこの倍に非常用の逆指値を置く(0 = 置かない)
    trail: bool = False                  # 損切りを、有利に動いた分だけ追いかける(終値で判定)
    max_drawdown: float = 0.25           # 有効証拠金が山からこれだけ下げたら新規を止める(0 = 止めない)
    dd_cooldown_bars: int = 0            # 止めてからこの本数で再開し、山を今の有効証拠金に付け直す(0 = 止めたまま)
    avoid_rollover_hours: float = 1.0    # ロールオーバーの前後この時間は新規を出さない(日足以上の戦略では効かない)
    flat_before_weekend: bool = False    # 金曜の最後の判断で手仕舞う
    protect: float = 0.9                 # 口座の床 = 月初の有効証拠金 x これ(0 = 床を置かない)
    gap_mult: float = 1.5                # 逆指値が窓で滑る分の見込み(最悪の損失 = 逆指値までの損失 x これ)

    @classmethod
    def from_dict(cls, d: dict | None) -> GuardConfig:
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown guard settings {sorted(unknown)} (known: {sorted(known)})")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


# ---- 壁 3: 量 -------------------------------------------------------------------

def size_lots(ctx, stop_dist: float, g: GuardConfig, side: int = 1) -> float:
    """損切りまでの損失が 有効証拠金 x risk_per_trade になるロット(USDJPY の円口座)。上限と証拠金の余力で抑える。"""
    if stop_dist <= 0 or not np.isfinite(stop_dist):
        return 0.0
    sym, acct = ctx.symbol(), ctx.account()
    equity = acct.equity
    if equity <= 0:
        return 0.0
    units = equity * g.risk_per_trade / stop_dist
    price = sym.ask if side > 0 else sym.bid
    units = min(units, equity * g.max_leverage / price)
    lots = ctx.normalize_volume(units / sym.contract_size)
    while lots > 0 and ctx.margin_required(side, lots) > 0.8 * max(acct.margin_free, 0.0):
        lots = ctx.normalize_volume(lots - sym.volume_step)
    return lots


# ---- 壁 1・2: 費用 ----------------------------------------------------------------

def round_trip_cost(ctx, side: int, g: GuardConfig, hold_days: float | None = None) -> float:
    """往復の費用の見積もり(価格の単位、1 通貨あたり)。

    スプレッド + 往復の滑り + 往復の手数料 + 持つ間のスワップの支払い。
    スワップの受け取りは当てにしない(0 として数える)。"""
    sym = ctx.symbol()
    days = g.hold_days_est if hold_days is None else hold_days
    spread = sym.ask - sym.bid
    slip = 2 * g.slippage_pips_est * sym.pip
    commission = 2 * sym.commission_per_lot / sym.contract_size
    swap = (sym.swap_long_per_lot_day if side > 0 else sym.swap_short_per_lot_day) / sym.contract_size
    return spread + slip + commission + max(0.0, -swap) * days


def cost_ok(ctx, side: int, expected_move: float, g: GuardConfig, hold_days: float | None = None) -> bool:
    if g.cost_multiple <= 0:
        return True
    return bool(np.isfinite(expected_move)) and expected_move >= g.cost_multiple * round_trip_cost(ctx, side, g,
                                                                                                   hold_days)


class TurnoverBudget:
    """新規の回数を直近 365 日で max_trades_per_year 回までにする(約定の履歴から数える)。"""

    def __init__(self, g: GuardConfig, magic: int):
        self.g, self.magic = g, magic

    def allows(self, ctx) -> bool:
        if self.g.max_trades_per_year <= 0:
            return True
        since = ctx.time - 365 * NS_DAY
        n = sum(1 for d in ctx.deals if d.entry == 0 and d.magic == self.magic and d.time >= since)
        return n < self.g.max_trades_per_year


# ---- 口座の床(資産の 9 割を守る) --------------------------------------------------

def month_key(t_ns: int) -> int:
    """UTC の年月(年 x 12 + 月 − 1)。床を付け直す区切り。"""
    m = np.datetime64(int(t_ns), "ns").astype("datetime64[M]").astype(np.int64)
    return int(m)


@dataclass
class EquityFloor:
    """月初の有効証拠金 x protect を床にする。月が変わって最初に見た有効証拠金で付け直し、止めていれば再開する。

    バックテストでは戦略が持ち、常駐(fxsim trade)では口座ごとのファイルに残して、同じ口座の常駐どうしで共有する。"""
    protect: float
    key: int | None = None              # 床を付けた月(month_key)
    floor: float = 0.0
    halted: bool = False                # 床を割って止めている(その月は新規を出さない)
    trips: int = 0

    def update(self, t_ns: int, equity: float) -> None:
        k = month_key(t_ns)
        if k != self.key:
            self.key, self.floor, self.halted = k, equity * self.protect, False

    def check(self, equity: float) -> bool:
        """床を割っていれば止めて True(呼んだ側が全部手仕舞う)。止めている間は False(手仕舞いは済んでいる)。"""
        if self.protect <= 0 or self.halted or equity > self.floor:
            return False
        self.halted = True
        self.trips += 1
        return True

    def room(self, equity: float, worst_open: float | None) -> float:
        """新規の建玉の最悪の損失に使える額。床を割る・止めている・最悪が分からないなら 0。"""
        if self.protect <= 0:
            return float("inf")
        if self.halted or worst_open is None:
            return 0.0
        return max(0.0, equity - worst_open - self.floor)


def worst_loss(side: int, units: float, close_price: float, sl: float | None, gap_mult: float) -> float | None:
    """1 つの建玉が、今の決済値段(買いは bid、売りは ask)から逆指値まで飛んで、さらに窓で滑ったときの損失
    (決済通貨。USDJPY の円口座なら円)。逆指値が無ければ None。利の乗った側の逆指値なら 0。"""
    if sl is None:
        return None
    return max(0.0, (close_price - sl) * side) * units * gap_mult


def open_worst_loss(ctx, g: GuardConfig) -> float | None:
    """MT5 のテスターの口座の全建玉(magic を問わない)の最悪の損失の合計。逆指値の無い建玉があれば None。"""
    sym = ctx.symbol()
    total = 0.0
    for p in ctx.positions(None):
        w = worst_loss(p.side, p.volume * sym.contract_size, sym.bid if p.side > 0 else sym.ask, p.sl, g.gap_mult)
        if w is None:
            return None
        total += w
    return total


def floor_lots(lots: float, room: float, per_lot_worst: float, volume_step: float) -> float:
    """最悪の損失が room に収まるロット(volume_step で切り捨て)。"""
    if room == float("inf"):
        return lots
    if per_lot_worst <= 0 or not np.isfinite(per_lot_worst):
        return 0.0
    cap = np.floor(room / per_lot_worst / volume_step + 1e-9) * volume_step
    return float(round(min(lots, cap), 8))


# ---- 止める仕組み・時間 ----------------------------------------------------------

class DrawdownBreaker:
    def __init__(self, g: GuardConfig):
        self.g = g
        self.peak: float | None = None
        self.stopped_at: int | None = None
        self.trips = 0

    def allows(self, ctx, i: int) -> bool:
        if self.g.max_drawdown <= 0:
            return True
        eq = ctx.equity
        if self.stopped_at is not None:
            if self.g.dd_cooldown_bars and i - self.stopped_at >= self.g.dd_cooldown_bars:
                self.stopped_at, self.peak = None, eq
            else:
                return False
        self.peak = eq if self.peak is None else max(self.peak, eq)
        if eq < self.peak * (1 - self.g.max_drawdown):
            self.stopped_at = i
            self.trips += 1
            return False
        return True


def near_rollover(t_ns: int, hours: float) -> bool:
    """t が 22:00 UTC(ロールオーバー)の前後 hours 時間以内か。"""
    if hours <= 0:
        return False
    h = ((t_ns % NS_DAY) / NS_HOUR - 22.0) % 24.0
    return min(h, 24.0 - h) < hours


def last_decision_of_week(bar_t: np.ndarray, i: int) -> bool:
    """足 i の次の足が週をまたぐ(金曜の最後の足)か。"""
    if i + 1 >= len(bar_t):
        return False
    return int(weekday(np.asarray([bar_t[i + 1]]))[0]) < int(weekday(np.asarray([bar_t[i]]))[0]) \
        or (bar_t[i + 1] - bar_t[i]) > 2 * NS_DAY


__all__ = ["DrawdownBreaker", "EquityFloor", "GuardConfig", "TurnoverBudget", "cost_ok", "floor_lots",
           "last_decision_of_week", "month_key", "near_rollover", "open_worst_loss", "round_trip_cost", "size_lots",
           "worst_loss"]
