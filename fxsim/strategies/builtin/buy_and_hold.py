"""S005 buy_and_hold: 最初に買って持ち続ける。ドリフトとスワップだけを見る基準。"""
from dataclasses import dataclass

from .. import STRATEGIES
from ..base import Strategy


@STRATEGIES.register("buy_and_hold", version="1", description="買って持ち続ける(ドリフトとスワップの基準)")
class BuyAndHold(Strategy):
    name = "buy_and_hold"

    @dataclass
    class Params:
        timeframe: int = 1440

    def on_bar(self, i, ctx):
        if ctx.side == 0:
            ctx.buy()
