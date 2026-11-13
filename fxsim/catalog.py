"""組み込みの市場モデルと戦略をすべて登録する。使う側は ``import fxsim.catalog`` するだけでよい。"""
from .markets import MARKETS
from .markets import models as _market_models  # noqa: F401  (登録の副作用)
from .strategies import STRATEGIES
from .strategies import builtin as _builtin_strategies  # noqa: F401  (登録の副作用)
from .strategies import main as _main_strategies  # noqa: F401  (登録の副作用。本命の戦略)

__all__ = ["MARKETS", "STRATEGIES"]
