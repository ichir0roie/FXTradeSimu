"""市場モデルの部品。新しい悪意はここにクラスを足し、models/ のモデルに組み込む。"""
from .malice import CrowdSqueeze, FalseBreakout, StopHunt
from .regime import RangeMeanReversion, RegimeSwitching
from .shocks import FatTails, NewsShocks, WeekendGap
from .spread import ConstantSpread, DynamicSpread
from .volatility import SessionSeasonality, StochasticVolatility

__all__ = [
    "ConstantSpread", "CrowdSqueeze", "DynamicSpread", "FalseBreakout", "FatTails", "NewsShocks",
    "RangeMeanReversion", "RegimeSwitching", "SessionSeasonality", "StochasticVolatility", "StopHunt",
    "WeekendGap",
]
