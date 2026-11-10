"""組み込みの戦略。ファイル 1 つ = 戦略 1 つ。追加したらここで import する。"""
from . import (  # noqa: F401
    bollinger_reversion,
    buy_and_hold,
    direction_null,
    donchian_breakout,
    ma_cross,
    random_entry,
    simple_follower,
    tsmom,
)
