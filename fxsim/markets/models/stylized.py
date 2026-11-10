"""M002 stylized: 実際の為替に見られる統計的な性質(stylized facts)を持つ相場。

時間帯の季節性、ボラの塊、裾の厚さ、局面の入れ替わり、指標、週明けの窓、
状況で広がるスプレッド。経路を見て参加者を狙う「悪意」はまだ入れない。

v2(M005): 薄商いの期間(年末年始・GW・感謝祭)と、異常な値動き(フラッシュクラッシュ・為替介入・ショック)を足した。
v1 を再現するには params に {"holidays": None, "extreme": None}。
"""
from .. import MARKETS
from ..base import MarketModel
from ..components import (
    DynamicSpread,
    ExtremeEvents,
    FatTails,
    HolidayLiquidity,
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
        HolidayLiquidity(),
        NewsShocks(),
        RegimeSwitching(),
        FatTails(),
        WeekendGap(),
        RangeMeanReversion(),
        ExtremeEvents(),
        DynamicSpread(),
    ]


@MARKETS.register("stylized", version="2",
                  description="季節性・ボラの塊・裾の厚さ・局面・指標・窓・薄商い・異常な値動き・可変スプレッド",
                  tags=("realistic",))
def build(params: dict) -> MarketModel:
    return MarketModel("stylized", "2", stylized_components()).with_params(params)
