"""E068: 研究で使っている費用(1 取引 1.3 bp 前後)は本当か。どこを下げれば勝つ形が出るか。

    python -m fxsim.research.cost_reality fills     # デモ口座の常駐(B056)の成行の約定を、同じ分の 1 分足の bid と比べる
    python -m fxsim.research.cost_reality spreads   # ペア x 年ごとの記録のスプレッド(pip と bp)
    python -m fxsim.research.cost_reality cheap     # 4 時間足の順張り(E060 の裏返し)を、費用の安いペアで、滑りあり / 0、2 つの期間で

費用の中身(研究の約定、vol_band_fade): スプレッド(その足の記録 = 足の中の最小)+ 成行の滑り 0.1 pip + 逆指値の滑り 0.3 pip。
fills: 常駐の記録(data/trading/demo-usdjpy-b056/journal.jsonl)の成行の新規の約定の値段から、同じ分の 1 分足の bid(始値と終値を
    注文の秒で按分)を引く。買い(ask で約定)なら実際に払ったスプレッド、売り(bid で約定)なら 0 になるはず(測り方の確かめ)。
cheap: 4 時間足(区切り UTC 0 時と 2 時の 2 通りを合わせる)、帯の外を逆指値で抜けたら乗り、3 本で出る(q 0.9・0.95)。
    2020 年以降の結果を見て選んだ 5 ペアなので、2010-10〜2019-12 を「選ぶのに使っていない期間」として並べる。
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import datetime as dt
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..paths import DATA_DIR
from ..storage import prices
from . import vol_band_fade as vb
from .intraday_edge import BAD_HOURS, SOURCE

CHEAP = ["USDJPY", "EURJPY", "GBPJPY", "EURUSD", "GBPUSD"]
CFGS = [("break", 0.9, "time", 3, 100), ("break", 0.95, "time", 3, 100)]


def ts(s: str) -> int:
    return int(np.datetime64(s).astype("datetime64[s]").astype(np.int64))


def fills(journal: str = "demo-usdjpy-b056") -> None:
    con = prices.open_market_db()
    m1 = prices.read_rates(con, SOURCE, "USDJPY", 1)
    con.close()
    idx = {int(t): k for k, t in enumerate(m1["time"])}
    rows = []
    with open(DATA_DIR / "trading" / journal / "journal.jsonl") as f:
        lines = f.readlines()
    for line in lines:
        d = json.loads(line)
        for res in d.get("results") or []:
            for r in res:
                if not r.get("ok") or r["request"]["action"] != 1:
                    continue
                minute = int(d["ts"]) // 60 * 60
                k = idx.get(minute)
                if k is None:
                    continue
                side = 1 if r["request"]["type"] == 0 else -1
                o, c = m1["open"][k], m1["close"][k]
                bid = o + (c - o) * (d["ts"] - minute) / 60
                rows.append((side, (r["price"] - bid) / 0.01, (r["price"] - r["request"]["price"]) / 0.01, m1["spread"][k] / 10, d["ts"]))
    a = np.array(rows)
    t0 = dt.datetime.fromtimestamp(a[:, 4].min(), dt.timezone.utc)
    t1 = dt.datetime.fromtimestamp(a[:, 4].max(), dt.timezone.utc)
    print(f"\n## デモ口座の成行の新規(USDJPY、{len(a)} 本、{t0:%Y-%m-%d %H:%M}〜{t1:%Y-%m-%d %H:%M} UTC)\n")
    print("| 向き | 本数 | 約定値 − 同じ分の bid(pip、平均 / 中央 / 標準偏差) | 注文の値 → 約定の値の滑り(pip) | 1 分足の記録のスプレッド(pip) |\n|---|---:|---|---:|---:|")
    for s, lab in ((1, "買い(ask で約定)"), (-1, "売り(bid で約定)")):
        m = a[:, 0] == s
        print(f"| {lab} | {m.sum()} | {a[m,1].mean():+.2f} / {np.median(a[m,1]):+.2f} / {a[m,1].std():.2f} | {a[m,2].mean():+.2f} | {a[m,3].mean():.2f} |")


def spreads() -> None:
    con = prices.open_market_db()
    print("\n## 記録のスプレッド(1 時間足、ロールオーバーの時間を除く中央値)\n")
    print("| ペア | pip: 2011 / 2015 / 2020 / 2025〜 | 2025〜 の bp | 往復の目安 bp(+ 滑り 0.2 pip) |\n|---|---|---:|---:|")
    for p in vb.PAIRS + vb.CROSSES:
        h = prices.read_rates(con, SOURCE, p, 60)
        t = h["time"]
        ok = ~np.isin((t // 3600) % 24, BAD_HOURS)
        yr = np.array([dt.datetime.fromtimestamp(int(x), dt.timezone.utc).year for x in t])
        sp = h["spread"] / 10
        pip = 0.01 if p.endswith("JPY") else 0.0001

        def med(y0: int, y1: int, ok=ok, yr=yr, sp=sp) -> float:
            m = ok & (yr >= y0) & (yr <= y1)
            return float(np.median(sp[m])) if m.any() else float("nan")

        m = ok & (yr >= 2025)
        bp = np.median(sp[m] * pip / h["close"][m] * 1e4)
        bpc = np.median((sp[m] + 0.2) * pip / h["close"][m] * 1e4)
        print(f"| {p} | {med(2011, 2011):.1f} / {med(2015, 2015):.1f} / {med(2020, 2020):.1f} / {med(2025, 2026):.1f} | {bp:.2f} | {bpc:.2f} |")
    con.close()


SLIP_DEFAULT = (vb.SLIP_MKT, vb.SLIP_STOP)


def _job(a):
    # ワーカーは使い回されるので、滑りは仕事ごとに必ず設定し直す(前の仕事の 0 が漏れないように)
    vb.SLIP_MKT, vb.SLIP_STOP = (0.0, 0.0) if a[-1] else SLIP_DEFAULT
    return vb.simulate(*a[:-1])


def _month_t(v: np.ndarray, tt: np.ndarray) -> float:
    mon = (tt // (86400 * 30.4375)).astype(int)
    s = np.bincount(mon - mon.min(), weights=v)[np.unique(mon - mon.min())]
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s))))


def cheap(workers: int) -> None:
    periods = {"2010-10〜2019-12(選ぶのに使っていない)": (60, ts("2010-10-01"), ts("2019-12-31")), "2020-01〜": (60, ts("2020-01-01"), None)}
    print("\n## 4 時間足の順張り(帯の外を逆指値で抜けたら乗る、3 本で出る)、費用の安い 5 ペア、区切り 2 通り\n")
    print("| 期間 | ペア | 形 | 素 bp | 費用込み(滑りあり)| 費用込み(滑り 0)| t(滑りあり / 滑り 0) | 取引 |\n|---|---|---|---:|---:|---:|---|---:|")
    for pl, src in periods.items():
        jobs = [(p, "H4", off, g, 0, 0.0, CFGS, src, s0) for p in CHEAP for off in (0, 2)
                for g, s0 in ((False, False), (True, False), (False, True))]
        with ProcessPoolExecutor(workers) as ex:
            outs = list(ex.map(_job, jobs))
        for p in CHEAP + ["5 ペア合計"]:
            for cfg in CFGS:
                def get(g: bool, s0: bool, p=p, cfg=cfg, jobs=jobs, outs=outs) -> np.ndarray:
                    return np.vstack([o["res"][cfg] for j, o in zip(jobs, outs, strict=True)
                                      if (p.startswith("5") or j[0] == p) and j[3] == g and j[-1] == s0])
                net, gross, net0 = get(False, False), get(True, False), get(False, True)
                print(f"| {pl} | {p} | q {cfg[1]} | {gross[:,0].mean():+.2f} | {net[:,0].mean():+.2f} | {net0[:,0].mean():+.2f} "
                      f"| {_month_t(net[:,0], net[:,1]):+.2f} / {_month_t(net0[:,0], net0[:,1]):+.2f} | {len(net)} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["fills", "spreads", "cheap"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    {"fills": fills, "spreads": spreads, "cheap": lambda: cheap(args.workers)}[args.what]()


if __name__ == "__main__":
    main()
