"""市場モデルの評価: 実際の為替らしさ(stylized facts)と、悪意の量。

TARGETS は USDJPY を念頭に置いた「だいたいこの範囲」という目安で、厳密な実測値ではない。
範囲の根拠を更新したら docs/MARKET_MALICE.md も直すこと。
悪意の指標(false_breakout_rate など)には目標を置かず、random_walk との差で読む。

尖度は数本の極端な足(フラッシュクラッシュ・介入)で何倍にも変わる(3 年で 4% の急落が 1 回あると 20 → 70)。
採点には両側 0.05% を切り詰めた尖度(kurt_h1_trim)を使い、素の尖度(kurt_h1)と極端な動きの量
(max_h1_move_pct、big_h1_moves_per_year)は記録だけする。
"""
from __future__ import annotations

import numpy as np

from ..data import Bars, MarketData

# fact 名 -> (下限, 上限, 説明)
TARGETS: dict[str, tuple[float, float, str]] = {
    "annual_vol": (0.06, 0.13, "日次リターンの年率ボラ"),
    "kurt_h1_trim": (3.0, 30.0, "1 時間足リターンの超過尖度(裾の厚さ。両側 0.05% を切り詰め)"),
    "kurt_d1": (0.5, 10.0, "日足リターンの超過尖度"),
    "acf1_h1": (-0.05, 0.03, "1 時間足リターンの 1 次自己相関(ほぼ 0)"),
    "acf_abs_h1_lag1": (0.15, 0.55, "|1 時間足リターン| の 1 次自己相関(ボラの塊)"),
    "acf_abs_h1_lag24": (0.08, 0.50, "|1 時間足リターン| の 24 次自己相関(日内の季節性 + 持続)"),
    "vr_h1_d1": (0.80, 1.15, "分散比 Var(日足)/(24 Var(1 時間足))。1 ならランダムウォーク"),
    "daily_range_pct": (0.4, 1.1, "日足の平均値幅(価格に対する %)"),
    "spread_median_pips": (0.2, 0.6, "スプレッドの中央値(pip)"),
    "spread_p99_pips": (1.5, 15.0, "スプレッドの 99 パーセンタイル(pip)"),
}


def _ret(b: Bars) -> np.ndarray:
    return np.diff(np.log(b.close))


def _kurt(x: np.ndarray) -> float:
    x = x - x.mean()
    v = np.mean(x * x)
    return float(np.mean(x ** 4) / (v * v) - 3.0) if v > 0 else 0.0


def _kurt_trim(x: np.ndarray, q: float = 5e-4) -> float:
    """両側 q の分位点で切り詰めた超過尖度。数本の極端な足に振り回されない。"""
    if len(x) < 100:
        return _kurt(x)
    lo, hi = np.quantile(x, [q, 1 - q])
    return _kurt(np.clip(x, lo, hi))


def _acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    d = np.dot(x, x)
    return float(np.dot(x[:-lag], x[lag:]) / d) if d > 0 and len(x) > lag else 0.0


def breakout_stats(b: Bars, lookback: int = 24, horizon: int = 6) -> dict:
    """直近 lookback 本の高値/安値を終値で抜けたあと、horizon 本以内に終値でレンジ内へ戻った割合など。"""
    import pandas as pd

    hi = pd.Series(b.high).rolling(lookback).max().shift(1).to_numpy()
    lo = pd.Series(b.low).rolling(lookback).min().shift(1).to_numpy()
    c = b.close
    n = len(c)
    up = np.flatnonzero(c > hi)
    dn = np.flatnonzero(c < lo)
    fails = total = 0
    for idx, sign in ((up, 1), (dn, -1)):
        for i in idx:
            if i + horizon >= n:
                continue
            lvl = hi[i] if sign > 0 else lo[i]
            fut = c[i + 1:i + 1 + horizon]
            total += 1
            fails += bool(np.any(fut < lvl) if sign > 0 else np.any(fut > lvl))
    # ヒゲで抜けて終値で戻した足の割合(ストップ狩りの痕跡)
    pierce_up = b.high > hi
    pierce_dn = b.low < lo
    rej = (pierce_up & (c < hi)) | (pierce_dn & (c > lo))
    pierced = pierce_up | pierce_dn
    return {
        "false_breakout_rate": fails / total if total else float("nan"),
        "breakouts_per_day": total / max(1, n / 24),
        "wick_rejection_rate": float(rej.sum() / pierced.sum()) if pierced.sum() else float("nan"),
    }


def compute_facts(md: MarketData) -> dict:
    """統計的性質。基準解像度が 1 時間より粗い(日足の再生など)ときは日足の項目だけ。"""
    d1 = md.bars(1440)
    if md.step_minutes > 60:
        rd = _ret(d1)
        return {
            "annual_vol": float(np.std(rd) * np.sqrt(260)),
            "kurt_d1": _kurt(rd),
            "acf1_d1": _acf(rd, 1),
            "daily_range_pct": float(np.mean((d1.high - d1.low) / d1.close) * 100),
            "spread_median_pips": float(np.median(md.spread / md.pip)),
        }
    h1 = md.bars(60)
    rh = _ret(h1)
    rd = _ret(d1)
    pip = md.pip
    sp = md.spread / pip
    facts = {
        "annual_vol": float(np.std(rd) * np.sqrt(260)),
        "kurt_h1": _kurt(rh),
        "kurt_h1_trim": _kurt_trim(rh),
        "kurt_d1": _kurt(rd),
        "acf1_h1": _acf(rh, 1),
        "acf_abs_h1_lag1": _acf(np.abs(rh), 1),
        "acf_abs_h1_lag24": _acf(np.abs(rh), 24),
        "vr_h1_d1": float(np.var(rd) / (24 * np.var(rh))) if np.var(rh) > 0 else float("nan"),
        "daily_range_pips": float(np.mean(d1.high - d1.low) / pip),
        "daily_range_pct": float(np.mean((d1.high - d1.low) / d1.close) * 100),
        "spread_median_pips": float(np.median(sp)),
        "spread_mean_pips": float(np.mean(sp)),
        "spread_p99_pips": float(np.percentile(sp, 99)),
        "max_abs_h1_move_pips": float(np.max(np.abs(np.diff(h1.close))) / pip),
        "max_h1_move_pct": float(np.max(np.abs(rh)) * 100),
        "big_h1_moves_per_year": float(np.sum(np.abs(rh) >= 0.01) / max(1e-9, len(rh) / (260 * 24))),
        "max_spread_pips": float(np.max(sp)),
    }
    if md.step_minutes <= 5:
        # 5 分足の勢い(M006)。Var(1 時間足) / (12 Var(5 分足))。1 ならランダムウォーク、1 より大きいと数十分の勢い。
        # 悪意を数十分かけた直線で入れると 1.05 前後になり、5 分足の抜けに乗る規則が素で勝っていた。記録だけで採点しない
        r5 = _ret(md.bars(5))
        facts["vr_m5_h1"] = float(np.var(rh) / (12 * np.var(r5))) if np.var(r5) > 0 else float("nan")
    facts.update(breakout_stats(h1))
    counts: dict[str, int] = {}
    for e in md.events:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    days = len(d1)
    for k, v in counts.items():
        facts[f"events_per_day_{k}"] = v / max(1, days)
    return facts


def score_facts(facts: dict) -> dict:
    """各 fact が目安の範囲に入るか。penalty は範囲からの外れ具合(範囲の幅で正規化)。

    realism_score = 100 * exp(-penalty の合計)。全部範囲内なら 100。
    """
    rows = {}
    total = 0.0
    for k, (lo, hi, desc) in TARGETS.items():
        if k not in facts:
            continue   # 解像度が粗くて測れない項目は採点しない
        v = facts.get(k)
        if v is None or not np.isfinite(v):
            pen = 1.0
        elif v < lo:
            pen = (lo - v) / (hi - lo)
        elif v > hi:
            pen = (v - hi) / (hi - lo)
        else:
            pen = 0.0
        total += pen
        rows[k] = {"value": v, "lo": lo, "hi": hi, "ok": pen == 0.0, "penalty": pen, "desc": desc}
    return {"realism_score": float(100 * np.exp(-total)), "n_fail": sum(not r["ok"] for r in rows.values()),
            "checks": rows}
