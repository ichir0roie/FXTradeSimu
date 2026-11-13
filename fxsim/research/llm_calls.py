"""E066: LLM に日付だけを渡して日々の向きを決めさせ、実際の日足で採点する(未来が漏れない期間だけ)。

    python -m fxsim.research.llm_calls docs/results/E066-llm-calls.csv docs/results/E066-llm-calls-2.csv

LLM は学習した期間の相場を知っているので、過去の日付で試すと未来が漏れる(E061)。ここでは LLM の知識の締め切り
(2026 年 6 月)より後の 2026-07-01〜2026-11-12 の平日だけを使う。LLM には道具を使わせず、日付と知っていることだけで
向き(+1 / −1 / 0)を決めさせた。CSV の 1 列目は date、残りの列はペアの名前(ファイルごとに別の LLM の呼び出し)。
採点: 相場の DB の日足(その日 22:00 UTC 前後で終わる足)の始値 → 終値の bp に向きを掛ける。費用はその足の記録のスプレッド(往復 1 回分)。
帰無: 同じ数の +1 / −1 / 0 をペアの中で日付の間で並べ替えた版と、向きを全部でたらめにした版(各 10,000 通り)。
見分けられる最小の当たり率: 取引の数 n から、50% + 2 x 0.5 / √n(日々の判断が独立なときの目安。同じ向きを続けると実際はもっと甘い)。
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone

import numpy as np

from ..storage import prices
from .intraday_edge import SOURCE


def daily_returns(pair: str) -> dict[str, tuple[float, float]]:
    """日付(その足が終わる日)→(始値 → 終値の bp、記録のスプレッドの bp)。"""
    con = prices.open_market_db()
    d = prices.read_rates(con, SOURCE, pair, 1440, start=int(datetime(2026, 6, 25, tzinfo=timezone.utc).timestamp()))
    con.close()
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    out = {}
    for t, o, c, sp in zip(d["time"], d["open"], d["close"], d["spread"], strict=True):
        day = datetime.fromtimestamp(int(t) + 3 * 3600, timezone.utc).date().isoformat()   # 21/22 時 UTC 始まりの足は翌日の取引日
        out[day] = ((c - o) / o * 1e4, sp * pip / 10 / o * 1e4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("calls", nargs="+")
    ap.add_argument("--perms", type=int, default=10_000)
    args = ap.parse_args()
    sig, ret, cost, who = [], [], [], []
    first_date, last_date = "9999", "0000"
    for path in args.calls:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for p in [c for c in rows[0] if c != "date"]:
            rets = daily_returns(p)
            for r in rows:
                if r["date"] in rets:
                    rr, cc = rets[r["date"]]
                    sig.append(int(r[p]))
                    ret.append(rr)
                    cost.append(cc)
                    who.append(p)
                    first_date, last_date = min(first_date, r["date"]), max(last_date, r["date"])
    s, x, c, who = np.array(sig), np.array(ret), np.array(cost), np.array(who)
    pairs = list(dict.fromkeys(who.tolist()))
    traded = s != 0
    gross = s * x
    net = gross - np.where(traded, c, 0.0)
    hit = (np.sign(x[traded]) == s[traded]).mean()
    n = int(traded.sum())
    print(f"\n## E066 LLM の日々の向き({first_date}〜{last_date}、{len(pairs)} ペア、呼び出し {len(args.calls)} 回)\n")
    print(f"- 判断の数 {len(s)}、取引した数 {n}(買い {int((s > 0).sum())}・売り {int((s < 0).sum())}・見送り {int((s == 0).sum())})")
    print(f"- 当たりの割合 {hit:.1%}(独立なら見分けられる最小は {0.5 + 1 / np.sqrt(n):.1%})、1 取引の素の bp {gross[traded].mean():+.2f}、"
          f"費用込み {net[traded].mean():+.2f}、合計 {net.sum():+.0f} bp")
    flips = {p: int((np.diff(s[who == p][s[who == p] != 0]) != 0).sum()) for p in pairs}
    print(f"- 向きを変えた回数(見送りを除く、全ペアの合計){sum(flips.values())} 回")
    rng = np.random.default_rng(1)
    idx = {p: np.flatnonzero(who == p) for p in pairs}
    perm = np.empty(args.perms)
    for b in range(args.perms):
        perm[b] = sum((rng.permutation(s[ii]) * x[ii]).sum() for ii in idx.values())
    rand = np.array([(rng.choice([-1, 1], size=len(s)) * np.abs(s) * x).sum() for _ in range(args.perms)])
    tot = gross.sum()
    print(f"- 素の合計 {tot:+.0f} bp。日付の並べ替え(ペアごとに同じ買い・売りの数)より上の割合 {(perm < tot).mean():.1%}"
          f"(片側 p {(perm >= tot).mean():.3f})、向きでたらめより上の割合 {(rand < tot).mean():.1%}(片側 p {(rand >= tot).mean():.3f})")
    print("\n| ペア | 取引 | 当たり | 素の bp / 取引 | 費用込み bp / 取引 | 買い / 売り | 向きを変えた回数 | その期間のペアの動き(日々の合計 bp) |")
    print("|---|---:|---:|---:|---:|---|---:|---:|")
    for p in pairs:
        m = who == p
        mt = m & traded
        print(f"| {p} | {mt.sum()} | {(np.sign(x[mt]) == s[mt]).mean():.0%} | {gross[mt].mean():+.2f} | {net[mt].mean():+.2f} "
              f"| {int((s[m] > 0).sum())} / {int((s[m] < 0).sum())} | {flips[p]} | {x[m].sum():+.0f} |")


if __name__ == "__main__":
    main()
