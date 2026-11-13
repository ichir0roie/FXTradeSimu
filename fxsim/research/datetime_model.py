"""E064: 日時だけを入力にしたモデル(「日時で AI に判断させる」)は、翌 1 時間の向きを費用より当てられるか。

    python -m fxsim.research.datetime_model

LLM に日付を渡す形は、学習した未来が漏れるので使わない(E061)。ここでは、その 1 時間足の日時から事前に分かる特徴だけで、
その 1 時間の値動き(始値 → 終値、bid)を予測する。
    特徴: UTC の時、ニューヨークとロンドンの現地時刻の時(夏時間を反映)、曜日、日付、月末までの営業日数、五十日、月
    モデル: lgbm(LightGBM の回帰、ペアごと)/ cells((ニューヨークの時 x 曜日)ごとの過去の平均。数の少ないマスは 0 に縮める)
ウォークフォワード: 2013〜2026 年の各年を、その年より前の全部(2010-10〜)で学習して予測する。
売買: 予測の絶対値が費用(学習期間のスプレッドの中央値 + 往復の滑り 0.2 pip、bp)を越える足だけ、予測の向きに始値で入り終値で出る。
ロールオーバーの時間(20〜22 時 UTC)は学習にも売買にも使わない(記録のスプレッドの作り物、E015)。
「向きだけ」は全部の足で予測の向きに張ったときの素の損益(当てる力そのもの)。
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..storage import prices
from .intraday_edge import BAD_HOURS, SOURCE
from .vol_band_fade import PAIRS

YEARS = list(range(2013, 2027))
SLIP = 0.1


def features(t: np.ndarray) -> pd.DataFrame:
    ts = pd.to_datetime(t, unit="s", utc=True)
    ny = ts.tz_convert(ZoneInfo("America/New_York"))
    ldn = ts.tz_convert(ZoneInfo("Europe/London"))
    day = ts.normalize()
    month_end = (day + pd.offsets.BMonthEnd(0)).normalize()
    bdays = np.busday_count(day.values.astype("datetime64[D]"), month_end.values.astype("datetime64[D]"))
    dom = ts.day.to_numpy()
    last_bday = bdays == 0
    gotobi = np.isin(dom, [5, 10, 15, 20, 25]) | last_bday
    return pd.DataFrame({"hour": ts.hour.to_numpy(), "ny_hour": ny.hour.to_numpy(), "ldn_hour": ldn.hour.to_numpy(),
                         "weekday": ts.day_of_week.to_numpy(), "dom": dom, "bdays_to_end": bdays, "gotobi": gotobi.astype(int),
                         "month": ts.month.to_numpy()})


def simulate(pair: str) -> dict:
    import lightgbm as lgb

    con = prices.open_market_db()
    h = prices.read_rates(con, SOURCE, pair, 60)
    con.close()
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    t, o, c = h["time"], h["open"], h["close"]
    ok = ~np.isin((t // 3600) % 24, BAD_HOURS)
    ret = (c - o) / o * 1e4
    cost_all = (h["spread"] * pip / 10 + 2 * SLIP * pip) / o * 1e4
    X = features(t)
    year = pd.to_datetime(t, unit="s", utc=True).year.to_numpy()
    out = {m: [] for m in ("lgbm", "cells")}
    for y in YEARS:
        tr = ok & (year < y)
        te = ok & (year == y)
        if tr.sum() < 5000 or te.sum() == 0:
            continue
        cost = float(np.median(cost_all[tr]))
        # lgbm
        params = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 300,
                  "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "seed": 1, "verbose": -1,
                  "deterministic": True, "num_threads": 1}
        model = lgb.train(params, lgb.Dataset(X[tr].to_numpy(dtype=float), ret[tr]), num_boost_round=200)
        pred_l = model.predict(X[te].to_numpy(dtype=float))
        # cells
        df = pd.DataFrame({"k": X["ny_hour"][tr].to_numpy() * 7 + X["weekday"][tr].to_numpy(), "r": ret[tr]})
        g = df.groupby("k")["r"].agg(["sum", "count"])
        shrink = g["sum"] / (g["count"] + 200)                  # 数の少ないマスは 0 に縮める
        kt = X["ny_hour"][te].to_numpy() * 7 + X["weekday"][te].to_numpy()
        pred_c = pd.Series(kt).map(shrink).fillna(0.0).to_numpy()
        for name, pred in (("lgbm", pred_l), ("cells", pred_c)):
            side = np.sign(pred)
            gross_all = side * ret[te]
            trade = np.abs(pred) > cost
            net = (side * ret[te] - cost_all[te])[trade]
            out[name].append({"year": y, "t": t[te], "dir": gross_all, "trade_t": t[te][trade], "net": net,
                              "gross": (side * ret[te])[trade], "n_test": int(te.sum())})
    return out


def month_t(v: np.ndarray, tt: np.ndarray) -> float:
    if len(v) < 10:
        return 0.0
    mon = (tt // (86400 * 30.4375)).astype(int)
    s = np.bincount(mon - mon.min(), weights=v)[np.unique(mon - mon.min())]
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 2 and s.std() > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2 - 2))
    args = ap.parse_args()
    with ProcessPoolExecutor(args.workers) as ex:
        outs = dict(zip(PAIRS, ex.map(simulate, PAIRS), strict=True))
    print("\n## E064 日時だけのモデル(ウォークフォワード 2013〜2026、主要 10 ペア、1 時間足、ロールオーバーの時間を除く)\n")
    print("| モデル | ペア | 向きだけ: 素の bp / 足 | t(月) | 売買した足 / 年 | 素の bp / 取引 | 費用込み bp / 取引 | t(月) | プラスの年 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in ("lgbm", "cells"):
        allrows = {"dir": [], "t": [], "net": [], "tt": [], "gross": []}
        for p in PAIRS:
            rs = outs[p][m]
            d = np.concatenate([r["dir"] for r in rs])
            tt = np.concatenate([r["t"] for r in rs])
            net = np.concatenate([r["net"] for r in rs])
            gr = np.concatenate([r["gross"] for r in rs])
            ntt = np.concatenate([r["trade_t"] for r in rs])
            yrs = len(rs)
            pos_y = sum(r["net"].sum() > 0 for r in rs if len(r["net"]))
            print(f"| {m} | {p} | {d.mean():+.3f} | {month_t(d, tt):+.2f} | {len(net) / yrs:.0f} | {gr.mean() if len(gr) else 0:+.2f} "
                  f"| {net.mean() if len(net) else 0:+.2f} | {month_t(net, ntt):+.2f} | {pos_y}/{yrs} |")
            for k, v in (("dir", d), ("t", tt), ("net", net), ("tt", ntt), ("gross", gr)):
                allrows[k].append(v)
        d, tt = np.concatenate(allrows["dir"]), np.concatenate(allrows["t"])
        net, ntt, gr = np.concatenate(allrows["net"]), np.concatenate(allrows["tt"]), np.concatenate(allrows["gross"])
        print(f"| {m} | 10 ペア | {d.mean():+.3f} | {month_t(d, tt):+.2f} | {len(net) / 14 / 10:.0f} | {gr.mean() if len(gr) else 0:+.2f} "
              f"| {net.mean() if len(net) else 0:+.2f} | {month_t(net, ntt):+.2f} | |")


if __name__ == "__main__":
    main()
