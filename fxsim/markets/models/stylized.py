"""M002 stylized: 実際の為替に見られる統計的な性質(stylized facts)を持つ相場。

時間帯の季節性、ボラの塊、裾の厚さ、局面の入れ替わり、指標、週明けの窓、
状況で広がるスプレッド。経路を見て参加者を狙う「悪意」はまだ入れない。
"""
from .. import MARKETS
from ..base import MarketModel
from ..components import (
    DynamicSpread,
    FatTails,
    NewsShocks,
    RangeMeanReversion,
    RegimeSwitching,
    SessionSeasonality,
    StochasticVolatility,
    WeekendGap,
)


def stylized_components():
    return [
        SessionSeasonality(),
        StochasticVolatility(),
        NewsShocks(),
        RegimeSwitching(),
        FatTails(),
        WeekendGap(),
        RangeMeanReversion(),
        DynamicSpread(),
    ]


@MARKETS.register("stylized", version="1", description="季節性・ボラの塊・裾の厚さ・局面・指標・窓・可変スプレッド",
                  tags=("realistic",))
def build(params: dict) -> MarketModel:
    return MarketModel("stylized", "1", stylized_components()).with_params(params)
