"""組み込みの市場モデルと戦略をすべて登録する。使う側は ``import fxsim.catalog`` するだけでよい。"""
from .markets import MARKETS  # noqa: F401
from .markets import models as _market_models  # noqa: F401

try:
    from .strategies import STRATEGIES  # noqa: F401
    from .strategies import builtin as _builtin_strategies  # noqa: F401
except ImportError:  # pragma: no cover
    pass
