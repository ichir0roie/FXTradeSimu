"""戦略。``STRATEGIES.create(name, params)`` で Strategy を得る。"""
from ..registry import Registry

STRATEGIES = Registry("strategy")

from .base import Strategy  # noqa: E402

__all__ = ["STRATEGIES", "Strategy"]
