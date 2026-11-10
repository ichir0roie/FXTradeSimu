"""試した数の補正(過剰適合の測り方)。

Deflated Sharpe Ratio(DSR、Bailey と López de Prado 2014)
    N 個の戦略(または設定)を試して一番良かったものは、どれにも優位性が無くても良いシャープになる。
    中身の無い N 個の中の最大のシャープの期待値 SR0 を基準に、観測したシャープ SR がそれを超えている確率を、
    標本の長さ T・歪度・尖度で補正して出す。0.95 以上なら「試した数を考えても偶然では説明しにくい」。
        SR0 = √V x ((1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N e)))     γ はオイラーの定数
        DSR = Φ((SR − SR0) √(T − 1) / √(1 − 歪度 x SR + (尖度 − 1) / 4 x SR²))
    V は試した戦略どうしのシャープの分散。分からなければ、中身が無いときの推定量の分散 1 / T を使う。
    SR は 1 期間(ここでは 1 日)あたり(年率ではない)。
PBO(バックテストの過剰適合の確率、Bailey ら 2017 の CSCV)
    (期間 x 戦略)の成績の表を S 個の塊に分け、半分を「選ぶ期間」、残りを「確かめる期間」にする組み合わせすべてで、
    選ぶ期間の一番を、確かめる期間で順位付けする。その順位が中央より下になる割合が PBO。
    0.5 なら「選んだ一番」は確かめる期間では平均並み(選び方に情報が無い)、0 に近いほど選び方が効いている。
    1 本の歴史で測った PBO は大きくばらつく(中身の無い戦略の集まりでも 0.07〜0.65 になる。全期間で偶然強かった戦略は
    どちらの半分でも上に来やすいため)。並べ替えた歴史の複数の経路で平均して読む。
"""
from __future__ import annotations

import itertools
import math

import numpy as np

EULER = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """標準正規の分位点(Acklam の近似、相対誤差 1e-9 程度)。"""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    lo = 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - lo:
        return -_norm_ppf(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """中身の無い n_trials 個のシャープ(分散 sr_var)の最大値の期待値。"""
    if n_trials <= 1:
        return 0.0
    n = float(n_trials)
    return math.sqrt(sr_var) * ((1 - EULER) * _norm_ppf(1 - 1 / n) + EULER * _norm_ppf(1 - 1 / (n * math.e)))


def deflated_sharpe(sr: float, n_obs: int, n_trials: int, skew: float = 0.0, kurt: float = 3.0,
                    sr_var: float | None = None) -> dict:
    """sr は 1 期間あたりのシャープ。kurt は尖度(正規 = 3)。"""
    if n_obs < 3:
        return {"sr0": float("nan"), "dsr": float("nan"), "psr": float("nan")}
    v = sr_var if sr_var is not None else 1.0 / n_obs
    sr0 = expected_max_sharpe(n_trials, v)
    den = 1 - skew * sr + (kurt - 1) / 4 * sr * sr
    den = math.sqrt(max(den, 1e-12))
    z = (sr - sr0) * math.sqrt(n_obs - 1) / den
    psr = _norm_cdf(sr * math.sqrt(n_obs - 1) / den)          # 試した数を考えない(基準 0)
    return {"sr0": sr0, "dsr": _norm_cdf(z), "psr": psr}


def pbo(perf: np.ndarray, n_blocks: int = 10) -> dict:
    """perf は(期間 x 戦略)の成績(損益など、大きいほど良い)。塊の数は偶数。"""
    perf = np.asarray(perf, dtype=float)
    t, n = perf.shape
    s = int(n_blocks) - int(n_blocks) % 2
    s = min(s, t - t % 2)
    if n < 2 or s < 2:
        return {"pbo": float("nan"), "n_combos": 0, "logit_median": float("nan"), "n_strategies": n, "n_blocks": s}
    blocks = np.array_split(np.arange(t), s)
    logits = []
    for is_idx in itertools.combinations(range(s), s // 2):
        rows_is = np.concatenate([blocks[k] for k in is_idx])
        rows_os = np.concatenate([blocks[k] for k in range(s) if k not in is_idx])
        best = int(np.argmax(perf[rows_is].mean(axis=0)))
        os_mean = perf[rows_os].mean(axis=0)
        rank = (os_mean < os_mean[best]).sum() + 0.5 * ((os_mean == os_mean[best]).sum() - 1) + 1   # 1..n
        w = rank / (n + 1)
        logits.append(math.log(w / (1 - w)))
    lg = np.array(logits)
    return {"pbo": float(np.mean(lg <= 0)), "n_combos": len(lg), "logit_median": float(np.median(lg)),
            "n_strategies": n, "n_blocks": s}


def run_overfit(con, run_id: int, *, market: str | None = None, broker: str | None = None, n_blocks: int = 10,
                exclude_null: bool = True) -> dict:
    """1 つの実験の中の戦略の集まりについて、PBO(年ごとの損益で CSCV)と、戦略ごとの Deflated Sharpe。

    market は市場のラベル(省略で最初のもの)、broker はブローカーのラベル(省略で最初のもの)。
    同じ市場に経路が複数(シード)あれば、経路ごとに PBO を出して平均する。試した数 N は集まりの戦略の数。
    """
    import json

    from .storage import queries

    rows = queries.jobs_for_overfit(con, run_id)
    if exclude_null:
        rows = [r for r in rows if "null" not in r["strategy_label"].lower()]
    if not rows:
        raise ValueError(f"run {run_id} has no finished jobs")
    market = market or rows[0]["market_label"]
    broker = broker or rows[0]["broker"]
    rows = [r for r in rows if r["market_label"] == market and r["broker"] == broker]
    names = sorted({r["strategy_label"] for r in rows})
    by_path: dict[int, dict[str, dict]] = {}
    for r in rows:
        by_path.setdefault(r["path_id"], {})[r["strategy_label"]] = json.loads(r["metrics_json"] or "{}")
    pbos, dsr_rows = [], {n: [] for n in names}
    years_used = 0
    for metrics in by_path.values():
        if set(metrics) != set(names):
            continue
        years = sorted(set.intersection(*[set(m.get("yearly_pnl", {})) for m in metrics.values()]))
        perf = np.array([[metrics[n]["yearly_pnl"][y] for n in names] for y in years])
        years_used = len(years)
        pbos.append(pbo(perf, n_blocks))
        for n in names:
            m = metrics[n]
            if "n_days" in m:
                d = deflated_sharpe(m["sharpe"] / math.sqrt(260), m["n_days"], len(names), m["daily_skew"],
                                    m["daily_kurt"])
                dsr_rows[n].append((m["sharpe"], d["dsr"]))
    per = []
    for n in names:
        if dsr_rows[n]:
            a = np.array([(x[0], x[1]) for x in dsr_rows[n]])
            per.append({"strategy": n, "sharpe": float(a[:, 0].mean()), "dsr": float(a[:, 1].mean())})
    return {"run_id": run_id, "market": market, "broker": broker, "n_strategies": len(names), "n_paths": len(pbos),
            "years": years_used, "pbo": float(np.mean([p["pbo"] for p in pbos])) if pbos else float("nan"),
            "logit_median": float(np.mean([p["logit_median"] for p in pbos])) if pbos else float("nan"),
            "n_blocks": pbos[0]["n_blocks"] if pbos else 0, "strategies": sorted(per, key=lambda x: -x["sharpe"]),
            "sr0_annual": expected_max_sharpe(len(names), 1.0 / max(1, years_used * 260)) * math.sqrt(260)}


def format_overfit(o: dict) -> str:
    lines = [f"## 試した数の補正(run #{o['run_id']}、{o['market']} / {o['broker']})", "",
             f"- 戦略 {o['n_strategies']} 本 x 経路 {o['n_paths']} 本、{o['years']} 年を"
             f" {o['n_blocks']} 個の塊に分けた CSCV",
             f"- PBO(選んだ一番が、確かめる期間で中央より下になる割合): {o['pbo']:.2f}"
             "(0.5 = 選び方に情報が無い、0 に近いほど選び方が効く)",
             f"- 中身の無い {o['n_strategies']} 本の最大の年率シャープの期待値: {o['sr0_annual']:.2f}", "",
             "| 戦略 | 年率シャープ | Deflated Sharpe |", "|---|---|---|"]
    for s in o["strategies"]:
        lines.append(f"| {s['strategy']} | {s['sharpe']:.2f} | {s['dsr']:.2f} |")
    return "\n".join(lines)
