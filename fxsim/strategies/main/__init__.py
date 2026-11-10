"""本命の戦略(MT5 の EA として回す)。土台は base.MainStrategy、共通の対策は strategies/guards.py。

追加したらここで import する。基本検証の戦略(手法の一覧)は strategies/builtin/。
"""
from . import trend_d1  # noqa: F401
