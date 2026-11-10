"""先読み(未来のデータを使った判断)の検査。

足の先頭 k 本だけで走らせた結果と、全部で走らせた結果を比べ、
k-1 本目までの判断(注文)が一致しなければ、prepare か on_bar が未来を見ている。
bars が dict(ペア -> Bars)なら複数ペアの戦略として PortfolioBacktester で回す。
"""
from __future__ import annotations

from ..data import Bars
from .engine import Backtester
from .portfolio import PortfolioBacktester


def _run(make_strategy, bars: Bars | dict[str, Bars], seed: int):
    s = make_strategy()
    if isinstance(bars, dict):
        return PortfolioBacktester(bars, s, seed=seed).run()
    if getattr(s, "engine", "classic") == "mt5":
        from .mt5 import Mt5Tester

        return Mt5Tester(bars, s, seed=seed).run()
    return Backtester(bars, s, seed=seed).run()


def _head(bars: Bars | dict[str, Bars], k: int):
    return {p: b.slice(k) for p, b in bars.items()} if isinstance(bars, dict) else bars.slice(k)


def check_lookahead(make_strategy, bars: Bars | dict[str, Bars], cut_points=(0.3, 0.6),
                    seed: int = 0) -> list[str]:
    full = _run(make_strategy, bars, seed)
    n = len(next(iter(bars.values()))) if isinstance(bars, dict) else len(bars)
    problems = []
    for frac in cut_points:
        k = int(n * frac)
        part = _run(make_strategy, _head(bars, k), seed)
        a = [d for d in full.decisions if d[0] < k - 1]
        b = [d for d in part.decisions if d[0] < k - 1]
        if a != b:
            # 最初に食い違った判断。片方が短いだけなら、長い方の次の判断
            first = next((x for x, y in zip(a, b, strict=False) if x != y), None)
            first = first or (a[len(b):len(b) + 1] or b[len(a):len(a) + 1])
            problems.append(f"cut={k}: decisions differ (first mismatch near {first})")
    return problems
