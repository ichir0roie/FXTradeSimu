"""MT5 のテスターのレポートと同じ定義の指標(metrics["mt5"] に入る)。

定義は MetaTrader 5 のヘルプ(テスターのレポート)に合わせる。取引 1 回 = 決済の約定 1 つ(一部決済も 1 回)で、
損益はスワップと手数料を含む(Trade.pnl)。
    Net Profit = Gross Profit + Gross Loss、Profit Factor = Gross Profit / |Gross Loss|、Expected Payoff = Net / 回数
    Recovery Factor = Net / 有効証拠金の最大ドローダウン(金額)
    Balance / Equity Drawdown: Absolute = 初期 − 最小、Maximal = 山から次の谷までの最大(金額と、その時の %)、
                               Relative = 山からの下げの割合の最大(%)
    AHPR = 取引ごとの残高の比(後 / 前)の平均、GHPR = (最終 / 初期) ^ (1 / 回数)
    Z-Score = (N (R − 0.5) − P) / √(P (P − N) / (N − 1))、P = 2 W L、R = 勝ち負けの連の数
    LR Correlation / LR Standard Error = 取引ごとの残高と、その最小二乗の直線との相関 / 残差の標準誤差
    Margin Level = テスト中の証拠金維持率の最小(%、足の終わりで測る)
残高と有効証拠金のドローダウンは、足の終わりの値で測る(MT5 はティックごと。足の中の谷の分だけ小さく出る)。
"""
from __future__ import annotations

import math

import numpy as np


def _drawdown(curve: np.ndarray, start: float) -> dict:
    x = np.r_[start, curve[np.isfinite(curve)]]
    peak = np.maximum.accumulate(x)
    dd = peak - x
    rel = np.where(peak > 0, dd / peak, 0.0)
    k = int(np.argmax(dd)) if len(dd) else 0
    return {"absolute": float(max(0.0, start - x.min())), "maximal": float(dd.max()) if len(dd) else 0.0,
            "maximal_pct": float(rel[k] * 100) if len(dd) else 0.0,
            "relative_pct": float(rel.max() * 100) if len(rel) else 0.0}


def _runs(wins: np.ndarray, pnl: np.ndarray) -> dict:
    best = {"wins": (0, 0.0), "losses": (0, 0.0)}
    cur_n, cur_sum, cur_w = 0, 0.0, None
    runs_w, runs_l = [], []
    for w, p in zip(wins, pnl, strict=True):
        if w == cur_w:
            cur_n, cur_sum = cur_n + 1, cur_sum + p
        else:
            if cur_w is not None:
                (runs_w if cur_w else runs_l).append(cur_n)
            cur_n, cur_sum, cur_w = 1, p, w
        key = "wins" if w else "losses"
        if cur_n > best[key][0]:
            best[key] = (cur_n, cur_sum)
    if cur_w is not None:
        (runs_w if cur_w else runs_l).append(cur_n)
    return {"max_consecutive_wins": best["wins"][0], "max_consecutive_wins_money": float(best["wins"][1]),
            "max_consecutive_losses": best["losses"][0], "max_consecutive_losses_money": float(best["losses"][1]),
            "avg_consecutive_wins": float(np.mean(runs_w)) if runs_w else 0.0,
            "avg_consecutive_losses": float(np.mean(runs_l)) if runs_l else 0.0,
            "n_runs": len(runs_w) + len(runs_l)}


def mt5_report(bt) -> dict:
    eq0 = float(bt.cfg.initial_balance)
    pnl = np.array([t.pnl for t in bt.trades], dtype=float)
    n = len(pnl)
    gp, gl = float(pnl[pnl > 0].sum()), float(pnl[pnl < 0].sum())
    net = gp + gl
    eq_dd = _drawdown(bt.equity_curve, eq0)
    bal_dd = _drawdown(bt.balance_curve, eq0)
    out = {
        "total_net_profit": net, "gross_profit": gp, "gross_loss": gl,
        "profit_factor": gp / -gl if gl < 0 else (float("inf") if gp > 0 else 0.0),
        "expected_payoff": net / n if n else 0.0,
        "recovery_factor": net / eq_dd["maximal"] if eq_dd["maximal"] > 0 else 0.0,
        "balance_drawdown": bal_dd, "equity_drawdown": eq_dd,
        "total_trades": n,
        "short_trades": int(sum(1 for t in bt.trades if t.side < 0)),
        "long_trades": int(sum(1 for t in bt.trades if t.side > 0)),
        "short_won_pct": _won([t for t in bt.trades if t.side < 0]),
        "long_won_pct": _won([t for t in bt.trades if t.side > 0]),
        "largest_profit": float(pnl.max()) if n else 0.0, "largest_loss": float(pnl.min()) if n else 0.0,
        "average_profit": float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0,
        "average_loss": float(pnl[pnl < 0].mean()) if (pnl < 0).any() else 0.0,
        "deals": len(bt.deals),
        "margin_level_min_pct": _min_margin_level(bt),
        "stop_outs": int(bt.stats.get("stop_out", 0)), "margin_calls": int(bt.stats.get("margin_call", 0)),
        "requotes": int(bt.stats.get("requote", 0)), "rejected": int(bt.stats.get("rejected", 0)),
        "no_money": int(bt.stats.get("no_money", 0)), "invalid_requests": int(bt.stats.get("invalid", 0)),
        "orders_triggered": int(bt.stats.get("triggered", 0)), "orders_expired": int(bt.stats.get("expired", 0)),
    }
    if n:
        bal = eq0 + np.cumsum(pnl)
        before = np.r_[eq0, bal[:-1]]
        hpr = np.where(before > 0, bal / np.where(before > 0, before, 1.0), 0.0)
        out["ahpr"] = float(hpr.mean())
        out["ghpr"] = float((max(bal[-1], 0.0) / eq0) ** (1.0 / n)) if eq0 > 0 else 0.0
        wins = pnl > 0
        r = _runs(wins, pnl)
        out.update(r)
        w, lo = int(wins.sum()), int((~wins).sum())
        p = 2.0 * w * lo
        out["z_score"] = (float((n * (r["n_runs"] - 0.5) - p) / math.sqrt(p * (p - n) / (n - 1)))
                          if n > 1 and p * (p - n) > 0 else 0.0)
        if n > 2:
            x = np.arange(n, dtype=float)
            a, b = np.polyfit(x, bal, 1)
            resid = bal - (a * x + b)
            out["lr_correlation"] = float(np.corrcoef(x, bal)[0, 1]) if bal.std() > 0 else 0.0
            out["lr_standard_error"] = float(math.sqrt((resid ** 2).sum() / (n - 2)))
    return out


def _won(trades) -> float:
    return float(100.0 * sum(1 for t in trades if t.pnl > 0) / len(trades)) if trades else 0.0


def _min_margin_level(bt) -> float | None:
    m = getattr(bt, "margin_curve", None)
    if m is None:
        return None
    ok = np.isfinite(m) & (m > 0)
    if not ok.any():
        return None
    return float((bt.equity_curve[ok] / m[ok] * 100).min())
