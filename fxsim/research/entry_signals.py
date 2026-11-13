"""E043: 入った直後の伸び(E042)は、どの信号・どの向きで出るか。数日〜10 日で手仕舞う型は成り立つか。

    python -m fxsim.research.entry_signals

E042 と同じ相場(30 年の実データ 1 本 + 並べ替え 10 本)・同じ単位(建てた足の ATR)で、信号と向きを変えて測る。

    1  信号 x 向き: 信号が点いた足(直前 5 本は点いていない)の後 1・3・5・10 日の値動き。そのペア・その区分の全部の足の
       平均を引く。区分は 下限あり(受け取る向き・金利差 2% 以上・c2r 0.4 以上)/
       下限未満(受け取る向きだが下限に届かない)/ 払う向き
    2  重ねがけ: 下限ありの 10 日抜けを、同じ向きを指している状態の数(EMA10 > EMA30・終値 > SMA50・63 日の値動き・
       MACD > シグナル)で分ける
    3  手仕舞い: 下限ありで、入りと出の決まりを変える(日数で出る・信号が落ちたら出る・その両方)

信号(向き d に読み替える)
    B10 / B20  終値が直近 10 / 20 日の高値を抜ける
    X          EMA10 が EMA30 を上に抜ける
    MACD       MACD がシグナルを上に抜ける
    RSI        RSI(14) が 30 を割る(押し目買い。逆張りの対照)
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fxsim.research.entry_path import BOOT_SEEDS, COST, cluster_t, load, tstat
from fxsim.strategies.indicators import ema, highest, lowest, macd, rsi, sma

HORIZONS = [1, 3, 5, 10]
SIGNALS = ["B10", "B20", "X", "MACD", "RSI"]
REGIMES = ["下限あり", "下限未満", "払う向き"]
FRESH = 5
EXITS = [("B10", "ch5", 0), ("B10", "none", 3), ("B10", "none", 5), ("B10", "none", 10), ("B10", "ch5", 10),
         ("B10", "ema", 10), ("B10", "ema", 0), ("any", "ch5", 10), ("any", "ema", 10), ("X", "ema", 0),
         ("X", "ema", 10)]


def prepare(s, bars, pair: str) -> dict:
    """向き +1 で読んだ信号と状態(売りは符号を返して使う)。"""
    b = bars[pair]
    c = b.close
    e10, e30 = ema(c, 10), ema(c, 30)
    m, sig = macd(c)
    r = rsi(c, 14)
    mom = np.r_[np.full(63, np.nan), c[63:] - c[:-63]]
    up = {"B10": c > np.r_[np.nan, highest(b.high, 10)[:-1]], "B20": c > np.r_[np.nan, highest(b.high, 20)[:-1]],
          "X": e10 > e30, "MACD": m > sig, "RSI": r < 30}
    dn = {"B10": c < np.r_[np.nan, lowest(b.low, 10)[:-1]], "B20": c < np.r_[np.nan, lowest(b.low, 20)[:-1]],
          "X": e10 < e30, "MACD": m < sig, "RSI": r > 70}
    state = {"ema": np.sign(e10 - e30), "sma": np.sign(c - sma(c, 50)), "mom": np.sign(mom), "macd": np.sign(m - sig)}
    return {"up": up, "dn": dn, "state": state}


def fresh(x: np.ndarray) -> np.ndarray:
    """点いた足のうち、直前 FRESH 本に点いていないもの。"""
    x = np.nan_to_num(x).astype(bool)
    k = np.convolve(x.astype(int), np.ones(FRESH + 1, dtype=int))[:len(x)]
    return x & (k == 1)


def regimes(s, pair: str) -> dict:
    p = s.p
    d = s.side[pair]
    fin = np.isfinite(s.atr[pair])
    floor = (d != 0) & (s.carry[pair] >= p.min_carry) & np.nan_to_num(s.c2r[pair] >= p.c2r_min) & fin
    recv = (d != 0) & (s.carry[pair] > 0) & fin
    return {"下限あり": (floor, d), "下限未満": (recv & ~floor, d), "払う向き": ((d != 0) & fin, -d)}


def trigger(ind: dict, name: str, d: np.ndarray) -> np.ndarray:
    names = ["B10", "X", "MACD"] if name == "any" else [name]
    out = np.zeros(len(d), dtype=bool)
    for nm in names:
        out |= np.where(d > 0, fresh(ind["up"][nm]), fresh(ind["dn"][nm]))
    return out


def study(spec):
    kind, seed = spec
    s, bars = load(kind, seed)
    day = s.t / 86400e9
    month = (day // 30).astype(int)
    years = (day[-1] - day[s.warmup + 63]) / 365.25
    w0 = s.warmup + 63
    ind = {pair: prepare(s, bars, pair) for pair in s.pairs}
    res = {"S": {}, "N": {}, "X": {}}
    # 1: 信号 x 向き
    for reg in REGIMES:
        for name in SIGNALS:
            acc = {k: ([], []) for k in HORIZONS}
            for pair in s.pairs:
                ok, d = regimes(s, pair)[reg]
                c, a = s.c[pair], s.atr[pair]
                n = len(c)
                ev = np.flatnonzero(trigger(ind[pair], name, d) & ok)
                ev = ev[ev >= w0]
                for k in HORIZONS:
                    idx = np.flatnonzero(ok[:n - k])
                    idx = idx[idx >= w0]
                    base = ((c[idx + k] - c[idx]) * d[idx] / a[idx]).mean() if len(idx) else 0.0
                    e = ev[ev + k < n]
                    acc[k][0].extend(((c[e + k] - c[e]) * d[e] / a[e] - base).tolist())
                    acc[k][1].extend(month[e].tolist())
            res["S"][(reg, name)] = {k: {"n": len(v[0]), "mean": float(np.mean(v[0])) if v[0] else 0.0,
                                         "win": float(np.mean(np.array(v[0]) > 0)) if v[0] else 0.0,
                                         "t": cluster_t(v[0], v[1])} for k, v in acc.items()}
    # 2: 重ねがけ(下限ありの B10 を、同じ向きの状態の数で分ける)
    groups = {"0〜1": ([], []), "2": ([], []), "3": ([], []), "4": ([], [])}
    for pair in s.pairs:
        ok, d = regimes(s, pair)["下限あり"]
        c, a = s.c[pair], s.atr[pair]
        ev = np.flatnonzero(trigger(ind[pair], "B10", d) & ok)
        ev = ev[(ev >= w0) & (ev + 10 < len(c))]
        st = ind[pair]["state"]
        for i in ev:
            cnt = sum(int(st[key][i] == d[i]) for key in st)
            g = groups["0〜1" if cnt <= 1 else str(cnt)]
            g[0].append([(c[i + k] - c[i]) * d[i] / a[i] for k in HORIZONS])
            g[1].append(month[i])
    for key, (v, mon) in groups.items():
        v = np.array(v) if v else np.zeros((0, len(HORIZONS)))
        res["N"][key] = {"n": len(v), "mean": v.mean(axis=0).tolist() if len(v) else [0.0] * len(HORIZONS),
                         "t": [cluster_t(v[:, q], mon) for q in range(len(HORIZONS))] if len(v) else [0.0] * 4}
    # 3: 手仕舞い(下限あり)
    for entry, rule, limit in EXITS:
        atrs, nets, mons, move, swap, held = [], [], [], 0.0, 0.0, 0
        for pair in s.pairs:
            ok, d = regimes(s, pair)["下限あり"]
            c, a = s.c[pair], s.atr[pair]
            n = len(c)
            trig = trigger(ind[pair], entry, d) & ok
            st = ind[pair]["state"]["ema"]
            i = w0
            while i < n - 1:
                if not trig[i]:
                    i += 1
                    continue
                di = d[i]
                j = i + 1
                while j < n - 1:
                    want = int(s.side[pair][j]) if s.carry[pair][j] >= s.p.min_carry else 0
                    out = want != di or (limit and j - i >= limit)
                    if rule == "ch5":
                        out |= (c[j] < s.lo_out[pair][j]) if di > 0 else (c[j] > s.hi_out[pair][j])
                    elif rule == "ema":
                        out |= st[j] != di
                    if out:
                        break
                    j += 1
                r = (c[j] - c[i]) * di
                sw = float(np.sum(s.carry[pair][i:j] * np.diff(day[i:j + 1])) / 365.0)
                atrs.append(r / a[i])
                nets.append(r / c[i] + sw - COST)
                mons.append(month[i])
                move += r / c[i]
                swap += sw
                held += j - i
                i = j + 1
        k = max(1, len(atrs))
        res["X"][(entry, rule, limit)] = {
            "per_year": len(atrs) / years, "atr": float(np.mean(atrs)) if atrs else 0.0,
            "win": float(np.mean(np.array(atrs) > 0)) if atrs else 0.0, "days": held / k,
            "move": move / years * 100, "swap": swap / years * 100, "net": float(np.sum(nets)) / years * 100,
            "per_day": float(np.sum(nets)) / max(1, held) * 1e4, "t": cluster_t(nets, mons)}
    return res


def main() -> None:
    with ProcessPoolExecutor(6) as ex:
        rows = list(ex.map(study, [("real", 1)] + [("boot", x) for x in BOOT_SEEDS]))
    real, boot = rows[0], rows[1:]

    print("## 1 信号 x 向き: 点いた後の値動き(ATR 単位、その区分の全部の足の平均を引いた差)\n")
    print("| 区分 | 信号 | 実データ: 数 | 1 日 | 3 日 | 5 日 | 10 日 | t(月)1 日 / 3 日 / 10 日 | 勝率 3 日 | "
          "並べ替え: 1 日 | 3 日 | 10 日 | t(パターン)1 日 / 3 日 / 10 日 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for reg in REGIMES:
        for name in SIGNALS:
            a = real["S"][(reg, name)]
            bv = {k: np.array([r["S"][(reg, name)][k]["mean"] for r in boot]) for k in HORIZONS}
            print(f"| {reg} | {name} | {a[1]['n']} | " + " | ".join(f"{a[k]['mean']:+.3f}" for k in HORIZONS)
                  + f" | {a[1]['t']:+.2f} / {a[3]['t']:+.2f} / {a[10]['t']:+.2f} | {a[3]['win']:.0%} | "
                  + " | ".join(f"{bv[k].mean():+.3f}" for k in (1, 3, 10))
                  + f" | {tstat(bv[1]):+.2f} / {tstat(bv[3]):+.2f} / {tstat(bv[10]):+.2f} |")

    print("\n## 2 重ねがけ: 下限ありの 10 日抜けを、同じ向きを指す状態の数で分ける(ATR 単位、差し引き前)\n")
    print("| 同じ向きの状態の数(4 つ中) | 実データ: 数 | 1 日 | 3 日 | 5 日 | 10 日 | t(月)3 日 / 10 日 | "
          "並べ替え: 数 | 3 日 | 10 日 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in ("0〜1", "2", "3", "4"):
        a = real["N"][key]
        print(f"| {key} | {a['n']} | " + " | ".join(f"{x:+.3f}" for x in a["mean"])
              + f" | {a['t'][1]:+.2f} / {a['t'][3]:+.2f} | {np.mean([r['N'][key]['n'] for r in boot]):.0f} | "
              f"{np.mean([r['N'][key]['mean'][1] for r in boot]):+.3f} | "
              f"{np.mean([r['N'][key]['mean'][3] for r in boot]):+.3f} |")

    print("\n## 3 手仕舞い(下限あり。10 ペアの合計、建玉比の年率 %。費用は 1 取引 0.03% の概算)\n")
    print("| 入り | 出 | 実データ: 取引 / 年 | 持つ日数 | 勝率 | 1 取引(ATR) | 値動き | スワップ | 差し引き | "
          "持った 1 日あたり(bp) | t(月) | 並べ替え: 取引 / 年 | 差し引き | 1 日あたり(bp) | 最初の行との差 / t |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    label = {"ch5": "5 日のチャネル", "ema": "EMA10 < EMA30", "none": ""}
    names = {"any": "B10・X・MACD のどれか"}
    ref = np.array([r["X"][EXITS[0]]["net"] for r in boot])
    for key in EXITS:
        entry, rule, limit = key
        x = real["X"][key]
        out = "・".join(v for v in (label[rule], f"{limit} 日" if limit else "") if v)
        net = np.array([r["X"][key]["net"] for r in boot])
        print(f"| {names.get(entry, entry)} | {out} | {x['per_year']:.1f} | {x['days']:.1f} | {x['win']:.0%} | "
              f"{x['atr']:+.3f} | {x['move']:+.2f} | {x['swap']:+.2f} | {x['net']:+.2f} | {x['per_day']:+.2f} | "
              f"{x['t']:+.2f} | {np.mean([r['X'][key]['per_year'] for r in boot]):.1f} | {net.mean():+.2f} | "
              f"{np.mean([r['X'][key]['per_day'] for r in boot]):+.2f} | "
              f"{(net - ref).mean():+.2f} / {tstat(net - ref):+.2f} |")


if __name__ == "__main__":
    main()
