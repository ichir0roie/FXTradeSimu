"""バックテスト結果の指標。"""
from __future__ import annotations

import numpy as np

from ..calendar import NS_DAY, trading_day
from .engine import Backtester


def daily_equity(bt: Backtester) -> tuple[np.ndarray, np.ndarray]:
    """取引日ごとの最後の有効証拠金。(取引日の開始時刻 ns, 証拠金)。"""
    days = trading_day(bt.bars.t)
    last = np.flatnonzero(np.r_[days[1:] != days[:-1], True])
    return days[last] * NS_DAY - 2 * 3_600_000_000_000, bt.equity_curve[last]


def yearly_pnl(day_t: np.ndarray, eq: np.ndarray, eq0: float) -> dict[str, float]:
    """暦年(UTC)ごとの損益(円)。年末の有効証拠金 − 前年末(最初の年は初期資金)。"""
    years = day_t.astype("datetime64[ns]").astype("datetime64[Y]").astype(int) + 1970
    out = {}
    prev = eq0
    for y in np.unique(years):
        last = eq[np.flatnonzero(years == y)[-1]]
        out[str(int(y))] = float(last - prev)
        prev = last
    return out


def compute_metrics(bt: Backtester) -> dict:
    day_t, eq = daily_equity(bt)
    eq0 = bt.cfg.initial_balance
    years = max(len(eq) / 260.0, 1e-9)
    final = float(eq[-1])
    rets = np.diff(np.r_[eq0, eq]) / np.maximum(np.r_[eq0, eq][:-1], 1e-9)
    sd = rets.std()
    sharpe = float(rets.mean() / sd * np.sqrt(260)) if sd > 0 else 0.0
    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std() * np.sqrt(260)) if len(downside) > 1 and downside.std() > 0 else 0.0
    peak = np.maximum.accumulate(np.r_[eq0, eq])
    dd = (np.r_[eq0, eq] - peak) / peak
    max_dd = float(-dd.min())
    # 最長のドローダウン期間(日)
    under = dd < 0
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    cagr = float((max(final, 0) / eq0) ** (1 / years) - 1) if final > 0 else -1.0
    tr = bt.trades
    n = len(tr)
    pips = np.array([t.pips for t in tr]) if n else np.zeros(0)
    gross = np.array([t.gross_pips for t in tr]) if n else np.zeros(0)
    cost = np.array([t.cost_pips for t in tr]) if n else np.zeros(0)
    pnl = np.array([t.pnl for t in tr]) if n else np.zeros(0)
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    held = np.array([t.bars_held for t in tr]) if n else np.zeros(0)
    reasons: dict[str, int] = {}
    for t in tr:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1
    # 素の優位性(mid どうし)のシャープ: トレード単位の gross pip を年率に
    g_sharpe = float(gross.mean() / gross.std() * np.sqrt(n / years)) if n > 1 and gross.std() > 0 else 0.0
    exposure = float(np.mean([t.bars_held for t in tr]) * n / len(bt.bars)) if n else 0.0
    return {
        "final_equity": final,
        "total_return": final / eq0 - 1,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "max_dd_days": int(longest),
        "calmar": float(cagr / max_dd) if max_dd > 0 else 0.0,
        "n_trades": n,
        "trades_per_year": n / years,
        "win_rate": float((pnl > 0).mean()) if n else 0.0,
        "profit_factor": float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0),
        "avg_pips": float(pips.mean()) if n else 0.0,
        "net_pips_per_year": float(pips.sum() / years),
        "gross_pips_per_year": float(gross.sum() / years),
        "cost_pips_per_year": float(cost.sum() / years),
        "gross_sharpe": g_sharpe,
        "swap_total": float(bt.stats["swap_total"]),
        "avg_bars_held": float(held.mean()) if n else 0.0,
        "exposure": min(exposure, 1.0),
        "hunted": int(bt.stats["hunted"]),
        "losscut": int(bt.stats["losscut"]),
        "ruined": bool(bt.ruined),
        "exit_reasons": reasons,
        "years": years,
        "yearly_pnl": yearly_pnl(day_t, eq, eq0),
        "costs": cost_breakdown(bt),
        "rejected": int(bt.stats.get("rejected", 0)),
        "zero_cut": float(bt.stats.get("zero_cut", 0.0)),
    }


def cost_breakdown(bt: Backtester) -> dict:
    """費用の内訳(円)。spread + slippage + commission − swap が、素の損益と実際の損益の差。"""
    pip = bt.pip
    spread_and_slip = sum(t.cost_pips * pip * t.units for t in bt.trades)
    slippage = sum(t.slippage for t in bt.trades)
    commission = sum(t.commission for t in bt.trades)
    swap = sum(t.swap for t in bt.trades)
    gross = sum(t.gross_pips * pip * t.units for t in bt.trades)
    return {
        "gross_pnl": float(gross),
        "spread": float(spread_and_slip - slippage),
        "slippage": float(slippage),
        "commission": float(commission),
        "swap": float(swap),
        "total": float(spread_and_slip + commission - swap),
    }
