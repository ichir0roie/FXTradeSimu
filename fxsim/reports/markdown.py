"""Markdown の表と数値の書式。"""
from __future__ import annotations


def num(x, nd: int = 2, pct: bool = False) -> str:
    """数値の書式。None は "-"、1 万以上は桁区切りの整数、pct=True は百分率。"""
    if x is None:
        return "-"
    if pct:
        return f"{x * 100:.{nd}f}%"
    if abs(x) >= 10000:
        return f"{x:,.0f}"
    return f"{x:.{nd}f}"


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)
