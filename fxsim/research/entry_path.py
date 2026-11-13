"""E042: スワップ狙いのブレイクアウト(B049)の、入った直後の値動き。ダマシは損切りの幅と確認待ちのどちらで避けるか。

    python -m fxsim.research.entry_path

B049 の入りの条件(受け取る向き・金利差 2% 以上・キャリー・トゥ・リスク 0.4 以上・終値が 10 日の高値抜け)と
出の条件(終値が反対側の 5 日のチャネルを割る・金利差が下限を割る)を、ペアごとに独立に当てる(証拠金の上限・口座の床・
スプレッドの門は入れない)。建値は抜けた足の終値。損益は値動きだけで、単位は建てた足の ATR(20 日)か建値に対する %。

    A  入った後 k 日の値動き(予測性能)。同じ下限を満たす全部の足(抜けたかどうかを問わない)の平均を引いて見る
    B  確認待ち: k 日後に建値より上(確認できた)/ 下(できなかった)で分け、そこから出の決まりまでの残りの損益を比べる
    C  損切りの幅: 建値から ATR x s の逆指値(足の中で触れたら約定。窓は始値)を足して、s を変える
    D  最後に勝った取引が、途中でどこまで沈んだか(損切りを置ける近さ)

相場は verify と同じ 30 年の実データ 1 本と、並べ替え 10 本(250 日ブロック、金利差も日付と一緒)。
t は、実データは入った月でまとめた月単位、並べ替えはパターン単位(10 本)。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fxsim.cases.loader import load_cases
from fxsim.research.carry_timing_null import market
from fxsim.strategies import STRATEGIES

BEHAVIOR = "B049"
HORIZONS = [1, 2, 3, 5, 10, 15, 20]
CONFIRM = [1, 2, 3, 5]
STOPS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf]
COST = 0.0003          # 1 取引の費用の概算(建玉比。OANDA・約定 2 時で 3〜4.5 pip、E017)
BOOT_SEEDS = list(range(1, 11))


def tstat(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 and x.std(ddof=1) > 0 else 0.0


def cluster_t(v, month) -> float:
    """入った月ごとに足してから数える t(同じ時期のペアは一緒に動くので、取引を独立とみなさない)。"""
    v, month = np.asarray(v, float), np.asarray(month)
    if len(v) < 3:
        return 0.0
    u, inv = np.unique(month, return_inverse=True)
    s = np.bincount(inv, weights=v, minlength=len(u))
    return float(s.sum() / np.sqrt((s ** 2).sum())) if (s ** 2).sum() > 0 else 0.0


def load(kind: str, seed: int):
    b = load_cases().behaviors[BEHAVIOR]
    md = market(kind, seed)
    s = STRATEGIES.create(b.strategy, b.strategy_params())
    bars = {p: leg.bars(1440) for p, leg in md.meta["legs"].items()}
    s.prepare(bars, np.random.default_rng(0))
    return s, bars


def walk(s, bars, pair: str, stop: float):
    """1 ペアを最初から最後まで。戻り値は取引の一覧 (入りの足, 出の足, 出の値, 損切りか)。"""
    p = s.p
    c, hi, lo, op = s.c[pair], bars[pair].high, bars[pair].low, bars[pair].open
    side, carry, c2r, a = s.side[pair], s.carry[pair], s.c2r[pair], s.atr[pair]
    n = len(c)
    out = []
    i = s.warmup
    while i < n - 1:
        d = int(side[i])
        ok = d != 0 and carry[i] >= p.min_carry and np.isfinite(c2r[i]) and c2r[i] >= p.c2r_min and np.isfinite(a[i])
        if not (ok and (c[i] > s.hi_in[pair][i] if d > 0 else c[i] < s.lo_in[pair][i])):
            i += 1
            continue
        level = c[i] - d * stop * a[i] if np.isfinite(stop) else None
        j = i + 1
        while j < n:
            if level is not None and (lo[j] <= level if d > 0 else hi[j] >= level):
                px = min(level, op[j]) if d > 0 else max(level, op[j])
                out.append((i, j, px, True))
                break
            want = int(side[j]) if carry[j] >= p.min_carry else 0
            if want != d or (c[j] < s.lo_out[pair][j] if d > 0 else c[j] > s.hi_out[pair][j]) or j == n - 1:
                out.append((i, j, c[j], False))
                break
            j += 1
        i = j + 1 if j < n else n
    return out


def study(spec):
    kind, seed = spec
    s, bars = load(kind, seed)
    p = s.p
    day = s.t / 86400e9                      # 足の時刻は ns
    month = (day // 30).astype(int)
    res = {"kind": kind, "seed": seed, "A": {}, "B": {}, "C": {}, "D": None}
    base = {pair: walk(s, bars, pair, np.inf) for pair in s.pairs}
    # A: 入った後 k 日
    for k in HORIZONS:
        ent, exc, mon, flo = [], [], [], []
        for pair in s.pairs:
            c, d, a = s.c[pair], s.side[pair], s.atr[pair]
            n = len(c)
            ok = (d != 0) & (s.carry[pair] >= p.min_carry) & np.nan_to_num(s.c2r[pair] >= p.c2r_min) & np.isfinite(a)
            idx = np.flatnonzero(ok[:n - k])
            idx = idx[idx >= s.warmup]
            f = (c[idx + k] - c[idx]) * d[idx] / a[idx]
            m0 = f.mean() if len(f) else 0.0
            flo.append(f)
            for (i, _, _, _) in base[pair]:
                if i + k < n:
                    r = (c[i + k] - c[i]) * d[i] / a[i]
                    ent.append(r)
                    exc.append(r - m0)
                    mon.append(month[i])
        ent, exc = np.array(ent), np.array(exc)
        res["A"][k] = {"n": len(ent), "mean": float(ent.mean()), "win": float((ent > 0).mean()),
                       "floor": float(np.concatenate(flo).mean()), "excess": float(exc.mean()),
                       "t": cluster_t(exc, mon)}
    # B: 確認待ち
    full = []
    for pair in s.pairs:
        c, a = s.c[pair], s.atr[pair]
        for (i, j, px, _) in base[pair]:
            full.append((pair, i, j, (px - c[i]) * s.side[pair][i] / a[i]))
    res["B"][0] = {"n": len(full), "mean": float(np.mean([x[3] for x in full])),
                   "win": float(np.mean([x[3] > 0 for x in full])),
                   "t": cluster_t([x[3] for x in full], [month[x[1]] for x in full])}
    for k in CONFIRM:
        g = {"up": [], "down": [], "gone": 0, "up_m": [], "down_m": [], "early": []}
        for pair, i, j, tot in full:
            c, a, d = s.c[pair], s.atr[pair], s.side[pair][i]
            if j <= i + k:
                g["gone"] += 1
                g["early"].append(tot)
                continue
            first = (c[i + k] - c[i]) * d / a[i]
            key = "up" if first > 0 else "down"
            g[key].append(tot - first)
            g[key + "_m"].append(month[i])
        res["B"][k] = {"gone": g["gone"], "early": float(np.mean(g["early"])) if g["early"] else 0.0,
                       "n_up": len(g["up"]), "up": float(np.mean(g["up"])) if g["up"] else 0.0,
                       "up_win": float(np.mean(np.array(g["up"]) > 0)) if g["up"] else 0.0,
                       "t_up": cluster_t(g["up"], g["up_m"]),
                       "n_down": len(g["down"]), "down": float(np.mean(g["down"])) if g["down"] else 0.0,
                       "down_win": float(np.mean(np.array(g["down"]) > 0)) if g["down"] else 0.0,
                       "t_down": cluster_t(g["down"], g["down_m"]),
                       # 確認できたときだけ k 日後に入る型の、信号 1 回あたりの損益(入らなかった信号は 0)
                       "wait_per_signal": float(np.sum(g["up"]) / len(full))}
    # C: 損切りの幅
    years = (day[-1] - day[s.warmup]) / 365.25
    for st in STOPS:
        pct, swp, stopped, noise, n, wins, mons = 0.0, 0.0, 0, 0, 0, 0, []
        per = []
        for pair in s.pairs:
            c = s.c[pair]
            tr = base[pair] if not np.isfinite(st) else walk(s, bars, pair, st)
            ends = {i: (j, px) for (i, j, px, _) in base[pair]}
            for (i, j, px, hit) in tr:
                d = s.side[pair][i]
                r = (px - c[i]) * d / c[i]
                sw = float(np.sum(s.carry[pair][i:j] * np.diff(day[i:j + 1])) / 365.0)
                pct += r
                swp += sw
                n += 1
                wins += r > 0
                per.append(r + sw - COST)
                mons.append(month[i])
                if hit:
                    stopped += 1
                    if i in ends and (ends[i][1] - c[i]) * d > 0:
                        noise += 1
        res["C"][st] = {"n": n, "per_year": n / years, "win": wins / max(1, n), "stopped": stopped / max(1, n),
                        "noise": noise / max(1, stopped), "move": pct / years * 100, "swap": swp / years * 100,
                        "net": (pct + swp - COST * n) / years * 100, "t": cluster_t(per, mons)}
    # D: 最後に勝った取引の沈み
    mae_w, mae_l = [], []
    for pair in s.pairs:
        c, lo, hi, a = s.c[pair], bars[pair].low, bars[pair].high, s.atr[pair]
        for (i, j, px, _) in base[pair]:
            d = s.side[pair][i]
            worst = (c[i] - lo[i + 1:j + 1].min()) if d > 0 else (hi[i + 1:j + 1].max() - c[i])
            (mae_w if (px - c[i]) * d > 0 else mae_l).append(max(0.0, worst) / a[i])
    res["D"] = {"win": np.percentile(mae_w, [25, 50, 75, 90, 95]).tolist(), "n_win": len(mae_w),
                "lose": np.percentile(mae_l, [25, 50, 75, 90, 95]).tolist(), "n_lose": len(mae_l)}
    return res


def main() -> None:
    with ProcessPoolExecutor(6) as ex:
        rows = list(ex.map(study, [("real", 1)] + [("boot", x) for x in BOOT_SEEDS]))
    real, boot = rows[0], rows[1:]

    def v(sec, key, field) -> np.ndarray:
        """並べ替え 10 本の値。"""
        return np.array([r[sec][key][field] for r in boot], float)

    print("## A 入った後 k 日の値動き(建てた足の ATR 単位、スワップと費用は別)\n")
    print("| k 日後 | 実データ: 取引 | 平均 | 勝率 | 下限を満たす全部の足の平均 | 差 | 差の t(月) | "
          "並べ替え 10 本: 平均 | 下限の足 | 差 | 差の t(パターン) |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in HORIZONS:
        a = real["A"][k]
        print(f"| {k} | {a['n']} | {a['mean']:+.3f} | {a['win']:.0%} | {a['floor']:+.3f} | {a['excess']:+.3f} | "
              f"{a['t']:+.2f} | {v('A', k, 'mean').mean():+.3f} | {v('A', k, 'floor').mean():+.3f} | "
              f"{v('A', k, 'excess').mean():+.3f} | {tstat(v('A', k, 'excess')):+.2f} |")

    print("\n## B 確認待ち(k 日後の終値が建値より有利か不利かで分けた、そこから出の決まりまでの残りの損益。ATR 単位)\n")
    b0 = real["B"][0]
    now = v("B", 0, "mean")
    print(f"すぐ入る: 実データ {b0['n']} 取引・平均 {b0['mean']:+.3f}・勝率 {b0['win']:.0%}・t {b0['t']:+.2f} / "
          f"並べ替え 平均 {now.mean():+.3f}・t {tstat(now):+.2f}\n")
    print("| k 日後 | 実データ: もう出ていた(その損益) | 有利: 数・残り・勝率・t | 不利: 数・残り・勝率・t | "
          "確認して入る型(信号 1 回あたり) | 並べ替え: 有利の残り | 不利の残り | 差の t | "
          "確認して入る型 − すぐ入る / t |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in CONFIRM:
        b = real["B"][k]
        up, down, wait = v("B", k, "up"), v("B", k, "down"), v("B", k, "wait_per_signal") - now
        print(f"| {k} | {b['gone']}({b['early']:+.2f}) | {b['n_up']}・{b['up']:+.3f}・{b['up_win']:.0%}・"
              f"{b['t_up']:+.2f} | {b['n_down']}・{b['down']:+.3f}・{b['down_win']:.0%}・{b['t_down']:+.2f} | "
              f"{b['wait_per_signal']:+.3f} | {up.mean():+.3f} | {down.mean():+.3f} | {tstat(up - down):+.2f} | "
              f"{wait.mean():+.3f} / {tstat(wait):+.2f} |")

    print("\n## C 損切りの幅(建値から ATR x s。10 ペアの合計、建玉比の年率 %。費用は 1 取引 0.03% の概算)\n")
    print("| s | 実データ: 取引 / 年 | 勝率 | 損切りで終わった | うち損切りが無ければ勝っていた | 値動き | スワップ | "
          "差し引き | t(月) | 並べ替え: 取引 / 年 | 損切り | 無ければ勝ち | 差し引き | 損切りなしとの差 / t |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for st in STOPS:
        c = real["C"][st]
        name = "なし" if not np.isfinite(st) else f"{st:g}"
        diff = v("C", st, "net") - v("C", np.inf, "net")
        print(f"| {name} | {c['per_year']:.1f} | {c['win']:.0%} | {c['stopped']:.0%} | {c['noise']:.0%} | "
              f"{c['move']:+.2f} | {c['swap']:+.2f} | {c['net']:+.2f} | {c['t']:+.2f} | "
              f"{v('C', st, 'per_year').mean():.1f} | {v('C', st, 'stopped').mean():.0%} | "
              f"{v('C', st, 'noise').mean():.0%} | {v('C', st, 'net').mean():+.2f} | "
              f"{diff.mean():+.2f} / {tstat(diff):+.2f} |")

    print("\n## D 途中でいちばん沈んだ深さ(足の安値で、ATR 単位。損切りなし)\n")
    print("| | 取引 | 25% | 中央 | 75% | 90% | 95% |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for key, name in (("win", "最後に勝った取引"), ("lose", "最後に負けた取引")):
        d = real["D"]
        print(f"| 実データ・{name} | {d['n_' + key]} | " + " | ".join(f"{x:.2f}" for x in d[key]) + " |")
        q = np.mean([r["D"][key] for r in boot], axis=0)
        n = np.mean([r["D"]["n_" + key] for r in boot])
        print(f"| 並べ替え・{name} | {n:.0f} | " + " | ".join(f"{x:.2f}" for x in q) + " |")


if __name__ == "__main__":
    main()
