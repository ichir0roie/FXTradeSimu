"""E059: 株価指数(S&P500 など)の毎月の積立を、1 倍・NISA(S050)と 2 倍型・課税口座(S051)で比べる。

    python -m fxsim.research.leveraged_index
    python -m fxsim.research.leveraged_index --borrow 0.02 0.04 --years 20 --monthly 30000 --df 4

FX ではなく株価指数の試算。実データは使わず、指数の対数リターンを乱数で作る(正規分布、--df で裾の重い t 分布)。
毎日組み直す L 倍の投資信託の 1 か月の対数リターンは、連続で組み直す近似で
    L * mu - (L - 1) * r - fee - L^2 * sigma^2 / 2  +  L * sigma * Z
(mu は指数の算術の期待リターン、r は借りた分の金利 = 先物に入っている米ドルの短期金利、fee は信託報酬)。
二つの案には同じ乱数を使う(共通乱数)。課税口座は最後に売るときに利益 x 20.315% を払う。
NISA は生涯の枠(買った額で 1,800 万円)と年の枠(つみたて + 成長で 360 万円)を越えた分を、同じ中身の課税口座に回す。
2 倍型は成長投資枠の対象外なので、S051 は全部を課税口座で持つ。
外しているもの: 円とドルの為替、信託報酬以外の費用(売買の差・先物のずれ)、税の制度の変更。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

TAX = 0.20315
NISA_YEAR = 3_600_000
NISA_LIFE = 18_000_000


@dataclass(frozen=True)
class IndexPlan:
    pid: str
    name: str
    leverage: float
    fee: float          # 年率の信託報酬
    nisa: bool          # True なら NISA の枠に入れる(越えた分は課税口座)


PLANS = [IndexPlan("S050", "1 倍・NISA", 1.0, 0.001, True),
         IndexPlan("S051", "2 倍型・課税口座", 2.0, 0.009, False)]


def shocks(rng: np.random.Generator, paths: int, months: int, df: float | None = None) -> np.ndarray:
    """分散 1 の月次の乱数。df を与えると t 分布(裾が重い)を分散 1 にそろえて使う。"""
    if df is None:
        return rng.standard_normal((paths, months))
    if df <= 2:
        raise ValueError("df は 2 より大きくする(分散が有限になるように)")
    return rng.standard_t(df, (paths, months)) / np.sqrt(df / (df - 2))


def fund_log_returns(plan: IndexPlan, z: np.ndarray, mu: float, sigma: float, borrow: float) -> np.ndarray:
    """L 倍の投資信託の月次の対数リターン(信託報酬と借りた分の金利を引いた後)。"""
    dt = 1 / 12
    lev = plan.leverage
    drift = (lev * mu - (lev - 1) * borrow - plan.fee - lev * lev * sigma ** 2 / 2) * dt
    return drift + lev * sigma * np.sqrt(dt) * z


def simulate(plan: IndexPlan, z: np.ndarray, mu: float, sigma: float, borrow: float,
             monthly: float) -> dict[str, np.ndarray | float]:
    """毎月 monthly 円を積み立て、最後に売った後の手取りを経路ごとに返す。

    返り値: after_tax(手取り)、pre_tax(売る前の評価額)、paid(払った額)、tax(払った税)、
    nav_mdd(基準価額の最大 DD。積立の入金に左右されない、中身そのものの下げ)。
    """
    paths, months = z.shape
    growth = np.exp(fund_log_returns(plan, z, mu, sigma, borrow))
    nisa = np.zeros(paths)
    taxable = np.zeros(paths)
    basis = 0.0                      # 課税口座の取得額(全経路で同じ)
    nisa_paid = 0.0
    year_paid = 0.0
    nav = np.ones(paths)
    peak = np.ones(paths)
    mdd = np.zeros(paths)
    for m in range(months):
        if m % 12 == 0:
            year_paid = 0.0
        to_nisa = 0.0
        if plan.nisa:
            to_nisa = max(0.0, min(monthly, NISA_YEAR - year_paid, NISA_LIFE - nisa_paid))
        nisa_paid += to_nisa
        year_paid += to_nisa
        basis += monthly - to_nisa
        nisa = (nisa + to_nisa) * growth[:, m]
        taxable = (taxable + monthly - to_nisa) * growth[:, m]
        nav = nav * growth[:, m]
        peak = np.maximum(peak, nav)
        mdd = np.maximum(mdd, 1 - nav / peak)
    tax = np.maximum(taxable - basis, 0.0) * TAX
    pre = nisa + taxable
    return {"after_tax": pre - tax, "pre_tax": pre, "paid": monthly * months, "tax": tax, "nav_mdd": mdd}


def compare(mu: float, sigma: float, borrow: float, years: int, monthly: float, paths: int,
            seed: int, df: float | None = None) -> dict[str, dict]:
    z = shocks(np.random.default_rng(seed), paths, years * 12, df)
    return {p.pid: simulate(p, z, mu, sigma, borrow, monthly) for p in PLANS}


def table(res: dict[str, dict]) -> str:
    man = 1e4
    rows = ["| 案 | 手取り 下から 10% | 中央値 | 上から 10% | 平均 | 元本割れ | 税の中央値 "
            "| 基準価額の最大 DD 中央値 / 悪い側 10% |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in PLANS:
        r = res[p.pid]
        v = r["after_tax"]
        q10, q50, q90 = np.percentile(v, [10, 50, 90]) / man
        dd50, dd90 = np.percentile(r["nav_mdd"], [50, 90]) * 100
        below = (v < r["paid"]).mean() * 100
        tax = np.median(r["tax"]) / man
        rows.append(f"| {p.pid} {p.name} | {q10:,.0f} 万円 | {q50:,.0f} 万円 | {q90:,.0f} 万円 "
                    f"| {v.mean() / man:,.0f} 万円 | {below:.0f}% | {tax:,.0f} 万円 | {dd50:.0f}% / {dd90:.0f}% |")
    a, b = res["S051"]["after_tax"], res["S050"]["after_tax"]
    rows.append("")
    rows.append(f"S051 の手取りが S050 を上回る経路: {(a > b).mean() * 100:.0f}%。"
                f"払った額 {res['S050']['paid'] / man:,.0f} 万円")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mu", type=float, default=0.09, help="指数の算術の期待リターン(年率)")
    ap.add_argument("--sigma", type=float, default=0.18, help="指数のボラティリティ(年率)")
    ap.add_argument("--borrow", type=float, nargs="+", default=[0.02, 0.04], help="借りた分の金利(年率)")
    ap.add_argument("--years", type=int, default=20)
    ap.add_argument("--monthly", type=float, default=30_000)
    ap.add_argument("--paths", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--df", type=float, default=None, help="t 分布の自由度(裾を重くする。省くと正規分布)")
    a = ap.parse_args()
    tail = "正規分布" if a.df is None else f"t 分布(自由度 {a.df:g})"
    for r in a.borrow:
        print(f"\n### 借りる金利 {r:.1%}、指数 {a.mu:.0%} / ボラ {a.sigma:.0%}、{tail}、"
              f"{a.years} 年 x 毎月 {a.monthly:,.0f} 円、{a.paths:,} 経路\n")
        print(table(compare(a.mu, a.sigma, r, a.years, a.monthly, a.paths, a.seed, a.df)))


if __name__ == "__main__":
    main()
