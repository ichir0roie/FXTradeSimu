"""E063: 発表の種類を問わず、「足が平常の何倍も跳ねた」直後に入る売買は費用を越えるか。

    python -m fxsim.research.spike_follow

E062 は米雇用統計の日時だけを使った。ほかの発表(CPI・FOMC・各国の中銀・要人発言)の日時は手元に無いので、
暦の代わりに、足の終値で分かる「跳ねた足」を使う: 1 時間足 i の(高値 − 安値)が、その前の 480 本の中央値の k 倍以上で、
実体(終値 − 始値)が幅の半分以上ある足。次の新しい跳ねは 24 本あける。ロールオーバーの足(20〜22 時 UTC)は使わない。
入りは全部、足 i の終値の後(i + 1 の始値か、それ以降の指値)なので、後から分かる情報は使っていない。

形(m = 足 i の終値 − 始値、r = 前の 480 本の幅の中央値)
    cont   跳ねた向きに成行。出口 1h / 4h / 24h / calm(足の幅が中央値以下に戻った最初の足の終値、最長 120 本)
    fade   逆向きに成行。出口は同じ
    cont_tp / fade_tp   成行で入り、r の 1 倍で利確、逆指値なし。負けは calm か 5 日(利用者の最初の案の「跳ねたら利確、
           負けは落ち着いたら閉じる」を、跳ねた後に入る形にしたもの)
    pull   跳ねた向きに、m の半分戻した値で指値(4 時間有効)。出口 24h / stop_o(足 i の始値で損切り、最長 24h)
    pull_against  跳ねた先の m の 1/4 先に、逆向きの指値(4 時間)。出口 24h / stop(m の 1 倍先で損切り、最長 24h)
対照: 跳ねていない足(幅が中央値の 0.8〜1.2 倍で、実体が幅の半分以上)を同じ数だけでたらめに選び、同じ決まりを当てる。
約定は E050・E062 と同じく不利な側に倒す(event_nfp.Path)。
主要 10 ペアの 2010-10〜2018-10 を発見、2018-11〜を確認、クロス 18 本(全期間)を初見として分けて出す。
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .event_nfp import Path, load
from .vol_band_fade import CROSSES, PAIRS

SPLIT = int(np.datetime64("2018-11-01").astype("datetime64[s]").astype(np.int64))
KS = [3.0, 5.0]
GAP = 24
CONFIGS = ([(f, ex) for f in ("cont", "fade") for ex in ("1h", "4h", "24h", "calm")]
           + [(f, ex) for f in ("cont_tp", "fade_tp") for ex in ("calm", "5d")]
           + [("pull", "24h"), ("pull", "stop_o"), ("pull_against", "24h"), ("pull_against", "stop")])
HOURS = {"1h": 1, "4h": 4, "24h": 24, "calm": 120, "5d": 120}


def pick(cand: np.ndarray, n: int) -> list[int]:
    out, last = [], -10**9
    for i in cand:
        if i - last >= GAP and 480 <= i < n - 200:
            out.append(int(i))
            last = i
    return out


def simulate(pair: str) -> dict:
    x = load(pair)
    n = len(x["t"])
    rng_ = x["h"] - x["l"]
    mv = x["c"] - x["o"]
    ratio = rng_ / x["med"]
    body = np.abs(mv) >= 0.5 * rng_
    ok = np.isfinite(ratio) & body & ~x["bad"]
    res: dict = {}
    rng = np.random.default_rng((PAIRS + CROSSES).index(pair) + 1)
    for k in KS:
        spikes = pick(np.flatnonzero(ok & (ratio >= k)), n)
        normal = np.flatnonzero(ok & (ratio >= 0.8) & (ratio <= 1.2))
        ctrl = pick(np.sort(rng.choice(normal, size=min(len(normal), len(spikes) * 3), replace=False)), n)[:len(spikes)]
        for kind, events in (("spike", spikes), ("ctrl", ctrl)):
            for i in events:
                p = Path(x, 0.0, i + 1)
                r, m, P = x["med"][i], mv[i], x["c"][i]
                sgn = int(np.sign(m))
                j0 = p.next_good(i + 1)
                if j0 >= n - 130:
                    continue
                bp = 1e4 / P
                for form, ex in CONFIGS:
                    key = (k, kind, form, ex)
                    if form in ("cont", "fade", "cont_tp", "fade_tp"):
                        side = sgn if form.startswith("cont") else -sgn
                        e = p.market_in(j0, side)
                        take = e + side * r if form.endswith("_tp") else None
                        j, px = p.run_exit(j0, side, e, j0 + HOURS[ex], take, None, True, calm=(ex == "calm"))
                    elif form == "pull":
                        side = sgn
                        f = p.limit_fill(j0, j0 + 4, side, P - sgn * 0.5 * abs(m))
                        if f is None:
                            continue
                        j1, e = f
                        stop = x["o"][i] if ex == "stop_o" else None
                        j, px = p.run_exit(j1, side, e, j1 + 24, None, stop, False)
                    else:
                        side = -sgn
                        f = p.limit_fill(j0, j0 + 4, side, P + sgn * 0.25 * abs(m))
                        if f is None:
                            continue
                        j1, e = f
                        stop = P + sgn * 1.25 * abs(m) if ex == "stop" else None
                        j, px = p.run_exit(j1, side, e, j1 + 24, None, stop, False)
                    res.setdefault(key, []).append((int(x["t"][i]), side * (px - e) * bp))
    return {"res": res, "years": (x["t"][-1] - x["t"][0]) / (365.25 * 86400)}


def month_t(v: np.ndarray, tt: np.ndarray) -> float:
    if len(v) < 10:
        return 0.0
    mon = (tt // (86400 * 30.4375)).astype(int)
    s = np.bincount(mon - mon.min(), weights=v)[np.unique(mon - mon.min())]
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 2 and s.std() > 0 else 0.0


def stats(rows: list) -> dict:
    if not rows:
        return {"n": 0, "avg": 0.0, "t": 0.0, "win": 0.0, "worst": 0.0}
    a = np.array(rows, dtype=float)
    tt, v = a[:, 0], a[:, 1]
    return {"n": len(v), "avg": float(v.mean()), "t": month_t(v, tt), "win": float((v > 0).mean()), "worst": float(v.min())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2 - 2))
    args = ap.parse_args()
    pairs = PAIRS + CROSSES
    with ProcessPoolExecutor(args.workers) as ex:
        outs = dict(zip(pairs, ex.map(simulate, pairs), strict=True))
    yrs_major = sum(outs[p]["years"] for p in PAIRS)
    print("\n## E063 跳ねた足(幅が前の 480 本の中央値の k 倍以上)の後に入る。費用込みの bp / 取引\n")
    for k in KS:
        print(f"\n### k = {k}\n")
        n_sp = sum(len(outs[p]["res"].get((k, "spike", "cont", "1h"), [])) for p in PAIRS)
        print(f"主要 10 ペアの跳ねた足: {n_sp} 本(1 ペア年 {n_sp / yrs_major:.1f} 本)\n")
        print("| 形 | 出口 | 主要: 取引 / bp / t | 勝率 | 最悪 bp | 発見 bp / t | 確認 bp / t | 初見のクロス: 取引 / bp / t | 対照(跳ねていない足): bp / t |")
        print("|---|---|---|---:|---:|---|---|---|---|")
        for form, exn in CONFIGS:
            key = (k, "spike", form, exn)
            rows = [r for p in PAIRS for r in outs[p]["res"].get(key, [])]
            s = stats(rows)
            a = stats([r for r in rows if r[0] < SPLIT])
            b = stats([r for r in rows if r[0] >= SPLIT])
            c = stats([r for p in CROSSES for r in outs[p]["res"].get(key, [])])
            ct = stats([r for p in PAIRS for r in outs[p]["res"].get((k, "ctrl", form, exn), [])])
            print(f"| {form} | {exn} | {s['n']} / {s['avg']:+.2f} / {s['t']:+.2f} | {s['win']:.0%} | {s['worst']:+.0f} "
                  f"| {a['avg']:+.2f} / {a['t']:+.2f} | {b['avg']:+.2f} / {b['t']:+.2f} | {c['n']} / {c['avg']:+.2f} / {c['t']:+.2f} "
                  f"| {ct['avg']:+.2f} / {ct['t']:+.2f} |")


if __name__ == "__main__":
    main()
