"""MT5 の注文・建玉・約定(取引履歴)と、取引の結果。名前と意味は MQL5 にそろえる。"""
from __future__ import annotations

from dataclasses import dataclass, field

from .spec import (
    ORDER_TYPE_BUY,
    ORDER_TYPE_BUY_LIMIT,
    ORDER_TYPE_BUY_STOP,
    ORDER_TYPE_BUY_STOP_LIMIT,
    ORDER_TYPE_NAMES,
    ORDER_TYPE_SELL_LIMIT,
    ORDER_TYPE_SELL_STOP,
    ORDER_TYPE_SELL_STOP_LIMIT,
    RETCODE_NAMES,
    TRADE_RETCODE_DONE,
    TRADE_RETCODE_PLACED,
)

LIMIT_TYPES = (ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_SELL_LIMIT)
STOP_TYPES = (ORDER_TYPE_BUY_STOP, ORDER_TYPE_SELL_STOP)
STOP_LIMIT_TYPES = (ORDER_TYPE_BUY_STOP_LIMIT, ORDER_TYPE_SELL_STOP_LIMIT)


def order_side(order_type: int) -> int:
    """注文の種類の向き(+1 買い / -1 売り)。"""
    return 1 if order_type % 2 == ORDER_TYPE_BUY else -1


@dataclass
class Mt5Position:
    """建玉(POSITION_*)。数量はロット、価格は約定値(買いは ask、売りは bid)。"""
    ticket: int
    side: int                 # +1 = POSITION_TYPE_BUY / -1 = POSITION_TYPE_SELL
    volume: float
    price_open: float
    sl: float | None
    tp: float | None
    time: int                 # 建てた時刻(ns)
    magic: int = 0
    comment: str = ""
    # ---- シミュレーションの記録(MQL5 には無い) ----
    open_mid: float = 0.0     # 素の損益の基準(建てた瞬間の mid)
    open_idx: int = 0
    swap: float = 0.0         # 円(付いた分の合計。残高にはその都度入れている)
    commission: float = 0.0   # 円(建てたときの手数料のうち、まだ決済の取引に付けていない分)
    slippage: float = 0.0     # 円(不利な向きを正)
    active_at: int = 0        # 建てた足の中で、損切り・利確を見始める経路の位置(建てた点より後だけ見る)

    @property
    def type(self) -> int:
        return 0 if self.side > 0 else 1

    def profit(self, bid: float, ask: float, contract_size: float) -> float:
        """評価損益(円。USDJPY の円口座なので換算なし)。買いは bid、売りは ask で評価する。"""
        px = bid if self.side > 0 else ask
        return (px - self.price_open) * self.side * self.volume * contract_size


@dataclass
class Mt5Order:
    """待機注文(ORDER_*)。price は発動の価格、stoplimit はストップリミットが発動した後の指値。"""
    ticket: int
    type: int
    volume: float
    price: float
    sl: float | None
    tp: float | None
    time_setup: int
    type_time: int = 0
    expiration: int | None = None   # ns(ORDER_TIME_SPECIFIED のとき)
    stoplimit: float | None = None
    magic: int = 0
    comment: str = ""
    position: int = 0               # netting / hedging で、この注文が閉じる建玉(0 = 新規)
    placed_idx: int = 0             # 置いた足(その足の置いた点より後から有効)
    placed_at: int = 0              # 置いた足の中の経路の位置

    @property
    def side(self) -> int:
        return order_side(self.type)

    @property
    def type_name(self) -> str:
        return ORDER_TYPE_NAMES[self.type]


@dataclass
class Mt5Deal:
    """約定(DEAL_*)。entry は IN(新規)/ OUT(決済)/ INOUT(ネットでの反転)/ OUT_BY(相殺)。
    profit は決済の分の損益(円、手数料・スワップを除く)。"""
    ticket: int
    order: int
    position: int
    time: int
    type: int                 # 0 買い / 1 売り
    entry: int
    volume: float
    price: float
    profit: float
    commission: float
    swap: float
    reason: int
    magic: int = 0
    comment: str = ""
    mid: float = 0.0          # 約定した瞬間の mid(素の損益の計算用)
    idx: int = 0


@dataclass
class TradeResult:
    """OrderSend の結果(MqlTradeResult)。retcode 10009 = 約定、10008 = 待機注文を置いた。"""
    retcode: int
    deal: int = 0
    order: int = 0
    volume: float = 0.0
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    comment: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.retcode in (TRADE_RETCODE_DONE, TRADE_RETCODE_PLACED)

    @property
    def name(self) -> str:
        return RETCODE_NAMES.get(self.retcode, str(self.retcode))
