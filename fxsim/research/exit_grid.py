"""E050: 損切り・利確・入り方(成行 / 指値)の組み合わせで、押し目(S048)は費用を越えるか。

    python -m fxsim.research.exit_grid            # 全部の格子(約数分)
    python -m fxsim.research.exit_grid --top 15
    python -m fxsim.research.exit_grid --null-seeds 20   # でたらめな入りを 20 通り回して、押し目をその分布と比べる

信号は日足(相場の DB)、約定は 1 時間足の bid の高安で 1 時間ごとに追う(2010-10〜2026-11、主要 10 ペア)。
    入り    成行(判断の 2 時間後の足の始値)/ 指値(日足の終値から ATR x k だけ有利な値。24 時間有効、届かなければ取り消し)
    損切り  建値から ATR x s 逆(逆指値。届いた足で、その値か始値の悪い方 + 滑り 0.3 pip)
    利確    建値から ATR x p 有利(指値。その値で約定)
    信号の出 日足の RSI(2) が 50 を越えた次の判断で成行(あり / なし)。どちらも 10 営業日で成行の手仕舞い
約定の決まり(bid の足と、その足の記録のスプレッドから)
    買いは ask(bid + スプレッド)、売りは bid。成行は滑り 0.1 pip。指値はロールオーバーの前後の足では約定させない
    同じ足で損切りと利確の両方に届いたら損切りを先にする。指値で入った足では利確を見ない(どちらも不利な側に倒す)
対照: 同じ回数・同じ持ち方ででたらめな日に、でたらめな向きで入る(random)。20 日の高値抜け(breakout)。
発見 = 2010-10〜2018-10、確認 = 2018-11〜2026-11。格子から選ぶのは発見だけで行い、確認で確かめる。
単位は 1 取引の bp(建値比 0.01%)。t は全ペアの損益を月ごとに足してから数える。
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

from ..storage import prices
from ..strategies.indicators import atr, highest, lowest, rsi, sma
from .intraday_edge import BAD_HOURS, SOURCE

PAIRS = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD", "EURJPY", "GBPJPY", "AUDJPY"]
ENTRIES = [0.0, 0.25, 0.5, 1.0]                 # 0 = 成行、ほかは指値の深さ(ATR x)
STOPS = [0.5, 1.0, 1.5, 2.0, 3.0, 0.0]          # 0 = 置かない
TAKES = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 0.0]    # 0 = 置かない
SIGNAL_EXIT = [True, False]
MAX_DAYS = 10
FILL_DELAY = 2 * 3600
SPLIT = int(np.datetime64("2018-11-01").astype("datetime64[s]").astype(np.int64))
SLIP_MKT, SLIP_STOP = 0.1, 0.3
FILL_MARGIN = 0.0        # 指値は、この pip だけ余分に抜けたときだけ約定させる(約定の仮定の厳しさ)


def load(pair: str) -> dict:
    con = prices.open_market_db()
    d = prices.read_rates(con, SOURCE, pair, 1440)
    h = prices.read_rates(con, SOURCE, pair, 60)
    con.close()
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    return {"d": d, "h": h, "pip": pip, "sp": h["spread"] * pip / 10,
            "bad": np.isin((h["time"] // 3600) % 24, BAD_HOURS)}


def day_signals(kind: str, d: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """日足の入りの向き(+1 / −1 / 0)と、向きごとの「信号の出」(その日の終値で出たい)。"""
    c, hi, lo = d["close"], d["high"], d["low"]
    n = len(c)
    if kind == "breakout":
        up, dn = np.r_[np.nan, highest(hi, 20)[:-1]], np.r_[np.nan, lowest(lo, 20)[:-1]]
        e = np.where(c > up, 1, np.where(c < dn, -1, 0))
        xl, xs = c < np.r_[np.nan, lowest(lo, 10)[:-1]], c > np.r_[np.nan, highest(hi, 10)[:-1]]
        return e, np.where(np.arange(n)[:, None] >= 0, np.column_stack([xl, xs]), False)
    ma, r = sma(c, 200), rsi(c, 2)
    ok = np.isfinite(ma) & np.isfinite(r)
    e = np.where(ok & (c > ma) & (r < 10), 1, np.where(ok & (c < ma) & (r > 90), -1, 0))
    out = np.column_stack([r > 50, r < 50])
    if kind == "random":                      # 同じ回数を、でたらめな日・でたらめな向きに
        rng = np.random.default_rng(seed)
        k = int((e != 0).sum())
        e2 = np.zeros(n, dtype=int)
        e2[rng.choice(np.arange(200, n), size=k, replace=False)] = rng.choice([-1, 1], size=k)
        e = e2
        out = np.column_stack([rng.random(n) < 0.4, rng.random(n) < 0.4])   # 出の頻度だけ似せる
    return e, out


def simulate(pair: str, kind: str, seed: int = 0) -> dict:
    x = load(pair)
    d, h, pip = x["d"], x["h"], x["pip"]
    dt, c = d["time"], d["close"]
    a = atr(d["high"], d["low"], c, 20)
    vr = a / atr(d["high"], d["low"], c, 250)             # 入る日のボラの比(E051)
    e, sig_out = day_signals(kind, d, seed=1000 * seed + PAIRS.index(pair) + 1)
    ht, ho, hh, hl, sp, bad = h["time"], h["open"], h["high"], h["low"], x["sp"], x["bad"]
    n = len(dt)
    # 日足 i の終値での判断が約定する 1 時間足(次の日足の始まりの 2 時間後)
    j_of = np.searchsorted(ht, np.r_[dt[1:], dt[-1] + 86400] + FILL_DELAY)
    first = int(np.searchsorted(dt, ht[0])) + 1
    res = {}
    for k_in, s_mult, p_mult, use_sig in product(ENTRIES, STOPS, TAKES, SIGNAL_EXIT):
        tr = []
        i = max(first, 200)
        while i < n - 2:
            side = int(e[i])
            if side == 0 or not np.isfinite(a[i]):
                i += 1
                continue
            j0 = int(j_of[i])
            if j0 >= len(ht) - 1:
                break
            # 出たい日(信号か日数)
            last = min(i + MAX_DAYS, n - 2)
            x_day = last
            if use_sig:
                hit = np.flatnonzero(sig_out[i + 1:last + 1, 0 if side > 0 else 1])
                if len(hit):
                    x_day = i + 1 + int(hit[0])
            j_end = min(int(j_of[x_day]), len(ht) - 1)
            # 入り
            if k_in == 0:
                entry = ho[j0] + sp[j0] + SLIP_MKT * pip if side > 0 else ho[j0] - SLIP_MKT * pip
                je = j0
            else:
                lim = c[i] - side * k_in * a[i]
                j1 = min(int(j_of[i + 1]), j_end)
                seg = slice(j0, j1)
                fm = FILL_MARGIN * pip
                reach = ((hl[seg] + sp[seg] <= lim - fm) if side > 0 else (hh[seg] >= lim + fm)) & ~bad[seg]
                if not reach.any():
                    i += 1
                    continue
                je = j0 + int(np.argmax(reach))
                entry = min(lim, ho[je] + sp[je]) if side > 0 else max(lim, ho[je])
            if je >= j_end:
                i += 1
                continue
            stop = entry - side * s_mult * a[i] if s_mult > 0 else None
            take = entry + side * p_mult * a[i] if p_mult > 0 else None
            seg = slice(je if k_in > 0 else je, j_end)
            m = j_end - seg.start
            js = jt = m
            if stop is not None:
                hit = (hl[seg] <= stop) if side > 0 else (hh[seg] + sp[seg] >= stop)
                if hit.any():
                    js = int(np.argmax(hit))
            if take is not None:
                hit = ((hh[seg] >= take + FILL_MARGIN * pip) if side > 0
                       else (hl[seg] + sp[seg] <= take - FILL_MARGIN * pip)) & ~bad[seg]
                if k_in > 0:
                    hit[0] = False                       # 指値で入った足では利確を見ない
                if hit.any():
                    jt = int(np.argmax(hit))
            if js <= jt and js < m:                      # 損切り(同じ足なら損切りが先)
                j = seg.start + js
                px = (min(stop, ho[j]) - SLIP_STOP * pip if side > 0
                      else max(stop, ho[j] + sp[j]) + SLIP_STOP * pip)
                why = 1
            elif jt < m:
                j = seg.start + jt
                px = max(take, ho[j]) if side > 0 else min(take, ho[j] + sp[j])
                why = 2
            else:
                j = j_end
                px = ho[j] - SLIP_MKT * pip if side > 0 else ho[j] + sp[j] + SLIP_MKT * pip
                why = 0
            tr.append((side * (px - entry) / entry * 1e4, ht[j], why, (ht[j] - ht[je]) / 86400, vr[i], a[i] / c[i]))
            i = int(np.searchsorted(dt, ht[j], side="right"))      # 出た日の終値から次の判断
        res[(k_in, s_mult, p_mult, use_sig)] = np.array(tr).reshape(-1, 6)
    return res


def month_t(v, t) -> float:
    if len(v) < 10:
        return 0.0
    mon = (np.asarray(t) // (86400 * 30.4375)).astype(int)
    s = np.bincount(mon - mon.min(), weights=v)[np.unique(mon - mon.min())]
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 2 and s.std() > 0 else 0.0


def summarize(tr: np.ndarray) -> dict:
    v, t = tr[:, 0], tr[:, 1]
    a, b = t < SPLIT, t >= SPLIT
    win = v > 0
    return {"n": len(v), "per_year": len(v) / 16.1, "win": win.mean(), "avg": v.mean(),
            "payoff": (v[win].mean() / -v[~win].mean()) if win.any() and (~win).any() else np.nan,
            "year": v.sum() / 16.1 / 100, "t": month_t(v, t),
            "a": v[a].mean() if a.any() else 0.0, "ta": month_t(v[a], t[a]),
            "b": v[b].mean() if b.any() else 0.0, "tb": month_t(v[b], t[b]),
            "days": tr[:, 3].mean(), "sl": (tr[:, 2] == 1).mean(), "tp": (tr[:, 2] == 2).mean()}


def label(key) -> str:
    k, s, p, sig = key
    return (f"{'成行' if k == 0 else f'指値 {k}'} | {'なし' if s == 0 else s} | {'なし' if p == 0 else p} "
            f"| {'あり' if sig else 'なし'}")


HEAD = ("| 入り | 損切り(ATR x) | 利確(ATR x) | 信号の出 | 取引 / 年 | 持つ日数 | 勝率 | 利益 / 損失の比 | 損切り・利確で出た割合 "
        "| 1 取引(bp) | 年率 % | t(月) | 発見(bp)/ t | 確認(bp)/ t |\n|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")


def row(key, s) -> str:
    return (f"| {label(key)} | {s['per_year']:.0f} | {s['days']:.1f} | {s['win']:.0%} | {s['payoff']:.2f} "
            f"| {s['sl']:.0%}・{s['tp']:.0%} | {s['avg']:+.2f} | {s['year']:+.2f} | {s['t']:+.2f} "
            f"| {s['a']:+.2f} / {s['ta']:+.2f} | {s['b']:+.2f} / {s['tb']:+.2f} |")


def merge(jobs, outs, pick) -> dict:
    merged = {}
    for job, res in zip(jobs, outs, strict=True):
        if pick(job):
            for key, tr in res.items():
                merged.setdefault(key, []).append(tr)
    return {key: summarize(np.vstack(v)) for key, v in merged.items()}


def against_null(seeds: int, workers: int, top: int) -> None:
    """押し目の格子を、でたらめな入り(同じ回数・同じ格子)の seeds 通りの分布と比べる。"""
    jobs = [(p, "pullback", 0) for p in PAIRS] + [(p, "random", s) for s in range(1, seeds + 1) for p in PAIRS]
    with ProcessPoolExecutor(workers) as ex:
        outs = list(ex.map(simulate, *zip(*jobs, strict=True)))
    real = merge(jobs, outs, lambda j: j[1] == "pullback")
    null = [merge(jobs, outs, lambda j, s=s: j[1] == "random" and j[2] == s) for s in range(1, seeds + 1)]
    keys = list(real)
    nv = np.array([[n[k]["avg"] for k in keys] for n in null])            # seeds x 格子
    mu, sd = nv.mean(axis=0), nv.std(axis=0, ddof=1)
    z = (np.array([real[k]["avg"] for k in keys]) - mu) / sd
    # でたらめ 1 本ずつを、残りの分布で同じように z にしたときの、格子の中の最大
    zmax = []
    for i in range(seeds):
        rest = np.delete(nv, i, axis=0)
        zmax.append(float(((nv[i] - rest.mean(axis=0)) / rest.std(axis=0, ddof=1)).max()))
    print(f"\n## 押し目とでたらめな入り {seeds} 通り(同じ格子 {len(keys)} 通り、1 取引の bp、全期間)")
    print(f"\n- でたらめな入りの格子全体の平均 {mu.mean():+.2f} bp、1 通りごとの標準偏差(運の振れ)の平均 {sd.mean():.2f} bp")
    print(f"- 押し目が、でたらめの平均 + 2 標準偏差を越えた格子: {int((z >= 2).sum())} / {len(keys)}")
    print(f"- 押し目の z の最大 {z.max():+.2f}。でたらめ 1 通りが格子の中で出す z の最大: 中央 {np.median(zmax):+.2f}、"
          f"95% 点 {np.quantile(zmax, 0.95):+.2f}、最大 {max(zmax):+.2f}")
    print("\n入り方ごとの平均(損切り・利確・信号の出の全部の平均、1 取引の bp)\n")
    print("| 入り | 押し目 | でたらめの平均 | 差 | でたらめの振れ(標準偏差) |\n|---|---:|---:|---:|---:|")
    for k_in in ENTRIES:
        m = np.array([k[0] == k_in for k in keys])
        r = np.mean([real[k]["avg"] for k, ok in zip(keys, m, strict=True) if ok])
        per_seed = nv[:, m].mean(axis=1)
        print(f"| {'成行' if k_in == 0 else f'指値 {k_in}'} | {r:+.2f} | {per_seed.mean():+.2f} | {r - per_seed.mean():+.2f} "
              f"| {per_seed.std(ddof=1):.2f} |")
    print(f"\nz の上位 {top}\n")
    print("| 入り | 損切り | 利確 | 信号の出 | 押し目(bp) | でたらめの平均 | でたらめの振れ | z | 発見 / 確認(bp) | 勝率 | 取引 / 年 |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for i in np.argsort(-z)[:top]:
        k, s = keys[i], real[keys[i]]
        print(f"| {label(k)} | {s['avg']:+.2f} | {mu[i]:+.2f} | {sd[i]:.2f} | {z[i]:+.2f} | {s['a']:+.2f} / {s['b']:+.2f} "
              f"| {s['win']:.0%} | {s['per_year']:.0f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--null-seeds", type=int, default=0)
    args = ap.parse_args()
    if args.null_seeds:
        against_null(args.null_seeds, args.workers, args.top)
        return
    kinds = ["pullback", "random", "breakout"]
    jobs = [(p, k, 1) for k in kinds for p in PAIRS]
    with ProcessPoolExecutor(args.workers) as ex:
        outs = list(ex.map(simulate, *zip(*jobs, strict=True)))
    for kind in kinds:
        stats = merge(jobs, outs, lambda j, kind=kind: j[1] == kind)
        base = (0.0, 3.0, 0.0, True) if kind != "breakout" else (0.0, 2.0, 0.0, True)
        pos_a = sum(s["a"] > 0 for s in stats.values())
        pos_both = sum(s["a"] > 0 and s["b"] > 0 for s in stats.values())
        sig_both = sum(s["ta"] >= 2 and s["tb"] >= 2 for s in stats.values())
        print(f"\n## {kind}: 格子 {len(stats)} 通り。発見でプラス {pos_a}、発見・確認の両方でプラス {pos_both}、"
              f"両方で t ≥ 2 は {sig_both}")
        print("\n基準(今の形に近い: 成行・損切り ATR x 3・利確なし・信号の出あり)\n\n" + HEAD)
        print(row(base, stats[base]))
        print(f"\n発見の t の上位 {args.top}(確認は選ぶのに使っていない)\n\n" + HEAD)
        for key, s in sorted(stats.items(), key=lambda kv: -kv[1]["ta"])[:args.top]:
            print(row(key, s))
        if kind == "pullback":
            print("\n1 つずつ変えたときの平均(ほかの軸の全部の平均、1 取引の bp: 発見 / 確認)")
            for name, idx, vals in (("入り", 0, ENTRIES), ("損切り", 1, STOPS), ("利確", 2, TAKES), ("信号の出", 3, SIGNAL_EXIT)):
                cells = []
                for val in vals:
                    sel = [s for key, s in stats.items() if key[idx] == val]
                    cells.append(f"{val}: {np.mean([s['a'] for s in sel]):+.2f} / {np.mean([s['b'] for s in sel]):+.2f}"
                                 f"(年率 {np.mean([s['year'] for s in sel]):+.2f}%)")
                print(f"- {name}: " + "、".join(cells))


if __name__ == "__main__":
    main()
