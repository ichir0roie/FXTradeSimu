"""FRED から複数通貨ペアの日足と短期金利を取ってきて、data/real/ に書く(`python -m fxsim fetch-fred`)。

為替(日次、NY 正午の買いレート、H.10)
    対ドルの 7 系列から対円のペアを作る。EURJPY = EURUSD x USDJPY など。全ペアの値がそろう日だけ残す
    (ユーロの始まった 1999-01-04 から)。終値だけ(始値・高値・安値は無い)。
短期金利(月次、OECD の主要経済指標)
    3 か月物の銀行間金利(IR3TIB01)を使う。その系列が始まる前の月は翌日物・コール金利(IRSTCI01)で埋め、
    系列が終わった後(公表の遅れ)は最後の値のまま。
    月 M の値は月の平均で、月末まで分からないので、翌月 1 日から有効にする(判断に未来の値を使わないため)。
どちらもパブリックドメイン(FRED。元は Federal Reserve Board H.10 と OECD MEI)。
"""
from __future__ import annotations

import io
import urllib.request

import pandas as pd

from ..paths import resolve

FX_OUT = "data/real/fx_jpy_crosses_d1_fred.csv"
RATES_OUT = "data/real/short_rates_monthly_fred.csv"
PAIRS = ("usdjpy", "eurjpy", "gbpjpy", "audjpy", "nzdjpy", "cadjpy", "chfjpy")

# 対ドルの系列: (FRED の ID, 1 単位あたりのドルなら True / 1 ドルあたりの外貨なら False)
_USD_SERIES = {"jpy": ("DEXJPUS", False), "eur": ("DEXUSEU", True), "gbp": ("DEXUSUK", True),
               "aud": ("DEXUSAL", True), "nzd": ("DEXUSNZ", True), "cad": ("DEXCAUS", False),
               "chf": ("DEXSZUS", False)}
# 金利: 通貨 -> OECD の国コード
_RATE_COUNTRY = {"usd": "US", "jpy": "JP", "eur": "EZ", "gbp": "GB", "aud": "AU", "nzd": "NZ", "cad": "CA",
                 "chf": "CH"}


def _fred(series: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    with urllib.request.urlopen(url, timeout=60) as r:          # noqa: S310 (固定の https の URL)
        df = pd.read_csv(io.BytesIO(r.read()), na_values=["."])
    df.columns = ["date", "value"]
    return pd.Series(df["value"].to_numpy(float), index=pd.to_datetime(df["date"]), name=series).dropna()


def build_crosses(usd: dict[str, pd.Series]) -> pd.DataFrame:
    """対ドルの系列(通貨 -> 1 ドルあたりの外貨、または 1 外貨あたりのドル)から対円のペアを作る。"""
    usd_per = {}
    for ccy, (_, per_unit) in _USD_SERIES.items():
        s = usd[ccy]
        usd_per[ccy] = s if per_unit else 1.0 / s           # 1 外貨あたりのドル
    df = pd.DataFrame(usd_per).dropna()
    out = pd.DataFrame(index=df.index)
    jpy_per_usd = 1.0 / df["jpy"]
    out["usdjpy"] = jpy_per_usd
    for ccy in ("eur", "gbp", "aud", "nzd", "cad", "chf"):
        out[f"{ccy}jpy"] = df[ccy] * jpy_per_usd
    return out.round(4)


def build_rates(primary: dict[str, pd.Series], fallback: dict[str, pd.Series]) -> pd.DataFrame:
    cols = {}
    for ccy in _RATE_COUNTRY:
        p, f = primary[ccy], fallback[ccy]
        # 代わりの系列は、元の系列が始まる前だけに使う(途中で切り替えると、系列の違いの分だけ段差ができる)
        cols[ccy] = p.combine_first(f[f.index < p.index[0]])
    df = pd.DataFrame(cols).sort_index().ffill()
    df.index = df.index + pd.offsets.MonthBegin(1)          # 月 M の平均は翌月 1 日から有効
    return df.dropna().round(4)


def fetch(log=print) -> tuple[str, str]:
    usd = {ccy: _fred(sid) for ccy, (sid, _) in _USD_SERIES.items()}
    fx = build_crosses(usd)
    fx.index.name = "time"
    fx.to_csv(resolve(FX_OUT), date_format="%Y-%m-%d")
    log(f"{FX_OUT}: {len(fx)} 日 {fx.index[0].date()}〜{fx.index[-1].date()}")
    primary = {c: _fred(f"IR3TIB01{k}M156N") for c, k in _RATE_COUNTRY.items()}
    fallback = {c: _fred(f"IRSTCI01{k}M156N") for c, k in _RATE_COUNTRY.items()}
    rates = build_rates(primary, fallback)
    rates.index.name = "date"
    rates.to_csv(resolve(RATES_OUT), date_format="%Y-%m-%d")
    log(f"{RATES_OUT}: {len(rates)} か月 {rates.index[0].date()}〜{rates.index[-1].date()}")
    return FX_OUT, RATES_OUT
