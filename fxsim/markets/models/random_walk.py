"""M001 random_walk: 悪意の無い基準。

一定ボラの正規ランダムウォーク + 一定スプレッド。旧 RandomWorkMarket.py の後継。
ほかのモデルの「悪意」は、この基準と比べて測る。
"""
from .. import MARKETS
from ..base import MarketModel
from ..components import ConstantSpread


@MARKETS.register("random_walk", version="1", description="一定ボラの正規ランダムウォーク(悪意なしの基準)",
                  tags=("baseline",))
def build(params: dict) -> MarketModel:
    return MarketModel("random_walk", "1", [ConstantSpread()]).with_params(params)
