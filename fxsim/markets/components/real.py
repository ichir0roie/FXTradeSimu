"""実際の相場を元にする部品。

RealReplay   実データをそのまま再生する(シードで変わらない)
RealSkeleton 実データの足(既定は日足)を「骨格」にし、1 分足の時刻列を用意する。
             mode="replay"    骨格 = 実データそのまま
             mode="bootstrap" 骨格 = 実データのリターンをブロック単位で並べ替えたもの(別の相場)
AnchorBridge 合成した経路を骨格に合わせる。骨格の終値を必ず通り、OHLC があれば
             各足の高値・安値にちょうど届くよう、はみ出しを伸縮する。
             骨格の足の中の動き(季節性・ボラの塊・指標・悪意)はシードごとに変わる。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ...calendar import NS_DAY, NS_MIN, server_hour, trading_times
from ...data.real import DB_PREFIX, RealSeries, load_ohlc
from ..base import BaseParams, Component, GenContext, Stage

DEFAULT_DATASET = "data/real/usdjpy_d1_fred.csv"


@dataclass
class _SourceParams:
    path: str = DEFAULT_DATASET
    format: str = "auto"
    start: str | None = None      # 期間。未指定なら実験の years だけ末尾から取る
    end: str | None = None


def _series(p, years: float | None, days: float | None) -> RealSeries:
    s = load_ohlc(p.path, p.format)
    if days is not None and years is None:
        years = days / 365.25
    w = s.window(p.start, p.end, None if p.start else years)
    if len(w) < 3:
        raise ValueError(f"real data window too short: {len(w)} rows ({p.path} {p.start}..{p.end})")
    return w


def _realized_vol(close: np.ndarray, step_minutes: int) -> float:
    r = np.diff(np.log(close))
    per_year = 260 * 1440 / step_minutes
    return float(np.std(r) * np.sqrt(per_year)) if len(r) > 1 else 0.09


class RealReplay(Component):
    """実データをそのままの解像度で再生する。

    足 1 本を「始値 → (高値 → 安値) → 終値」の点で表す(足の区切りの 1 秒後〜終わりの 1 秒前)。
    戦略の注文は次の足の始値で約定するので、終値しか無い日足でも、始値 = 前日の終値で約定する。
    v1 は終値の 1 点だけだったため、注文が翌日の終値で約定し、日足の戦略が 1 日遅れていた。
    """

    key = "source"
    stage = Stage.SOURCE
    deterministic = True

    @dataclass
    class Params(_SourceParams):
        pass

    def prepare_times(self, base: BaseParams, years, days, rng):
        s = _series(self.p, years, days)
        self._s = s
        sec = 1_000_000_000
        start = s.times - s.step_minutes * NS_MIN          # 足の始まり(times は足の終わり)
        if s.has_ohlc:
            offsets = [(start, 1), (start, 2), (start, 3), (s.times, -1)]
            prices = [s.open, s.high, s.low, s.close]
        else:
            offsets = [(start, 1), (s.times, -1)]
            prices = [s.open, s.close]                    # 終値だけのデータの始値は前日の終値
        times = np.stack([t + k * sec for t, k in offsets], axis=1).ravel()
        self._price = np.stack(prices, axis=1).ravel()
        b = replace(base, p0=float(s.close[0]), step_minutes=s.step_minutes,
                    annual_vol=_realized_vol(s.close, s.step_minutes),
                    start=str(np.datetime64(int(s.times[0]), "ns"))[:10])
        return times, b

    def apply(self, ctx: GenContext) -> None:
        s = self._s
        ctx.logp = np.log(self._price)
        r = np.abs(np.diff(ctx.logp, prepend=ctx.logp[0]))
        ctx.vol = np.maximum(r, 1e-12)
        ctx.meta["real_source"] = s.source
        ctx.meta["real_has_ohlc"] = s.has_ohlc


def _db_symbol(path: str) -> str | None:
    """db:[<出どころ>/]<銘柄>/<時間足> の銘柄。DB でなければ None。"""
    if not str(path).startswith(DB_PREFIX):
        return None
    return str(path)[len(DB_PREFIX):].split("/")[-2].upper()


def _last_close(symbol: str, t_ns: int) -> float | None:
    """相場の DB の日足(無ければ 1 時間足)で、時刻 t 以前の最後の終値(無ければ最初の終値)。"""
    for tf in ("D1", "H1"):
        try:
            s = load_ohlc(f"{DB_PREFIX}{symbol}/{tf}")
        except ValueError:
            continue
        k = int(np.searchsorted(s.times, t_ns, "right")) - 1
        return float(s.close[max(k, 0)])
    return None


def jpy_per_unit(ccy: str, t_ns: int) -> float:
    """通貨 ccy の 1 単位の円の値(相場の DB の USDJPY と USD<ccy> / <ccy>USD から)。"""
    ccy = ccy.upper()
    if ccy == "JPY":
        return 1.0
    usdjpy = _last_close("USDJPY", t_ns)
    if usdjpy is None:
        raise ValueError("converting to JPY needs USDJPY in the market DB (fetch-mt5)")
    if ccy == "USD":
        return usdjpy
    direct = _last_close(f"{ccy}JPY", t_ns)
    if direct is not None:
        return direct
    inv = _last_close(f"USD{ccy}", t_ns)
    if inv is not None:
        return usdjpy / inv
    via = _last_close(f"{ccy}USD", t_ns)
    if via is not None:
        return usdjpy * via
    raise ValueError(f"no rate in the market DB to convert {ccy} to JPY")


def _hourly_spread(symbol: str) -> tuple[np.ndarray, str] | None:
    """相場の DB の 1 分足の、サーバー時刻の時(足の始まり)ごとのスプレッドの平均(価格)と、サーバー時刻の決まり。

    サーバー時刻で分けるのは、ロールオーバー(NY の 17 時)が夏も冬もサーバーの 0 時になるため
    (UTC で分けると夏と冬が混ざる)。
    1 分足が無ければ None。
    """
    try:
        m1 = load_ohlc(f"{DB_PREFIX}{symbol}/M1")
    except ValueError:
        return None
    if m1.spread is None or len(m1) < 24 * 60 or not m1.server_tz:
        return None
    h = server_hour(m1.times - NS_MIN, m1.server_tz)
    out = np.zeros(24)
    for k in range(24):
        v = m1.spread[h == k]
        out[k] = float(v.mean()) if len(v) else 0.0
    return out, m1.server_tz


class PairReplay(RealReplay):
    """相場の DB(MT5 の足)の 1 ペアをそのまま再生し、円の口座で回せるようにする(real_pair、R012)。

    - 決済通貨が円でないペア(EURUSD など)は、期間の最初の時点の「決済通貨 1 単位の円の値」K を価格に掛ける。
      値動きの比率はそのままで、損益・証拠金・量・pip(= 元の pip x K)が円にそろう(円への換算は期間の中で固定)
    - 足の中の順番は MT5 のテスター(1 分足 OHLC)と同じ: 陽線は 始値 → 安値 → 高値 → 終値、
      陰線は 始値 → 高値 → 安値 → 終値。足の中は連続して動いたとみなす(Bars.points_per_bar)
      (RealReplay は常に高値が先で、損切りと利確が同じ足に入ったときの順番が偏る)
    - 足ごとの記録のスプレッドを meta に置く(RecordedSpread が使う)。同じ銘柄の 1 分足が相場の DB にあれば、
      その UTC の時ごとの平均(1 分ごとの最小の平均)も置く(1 時間足・日足の記録は足の中の最小で、ふだんの幅より狭いため)
    - スワップはペアの金利差(8 通貨の短期金利の月次)。meta["swap_pair"] で backtest と戦略に伝える
    """

    @dataclass
    class Params(_SourceParams):
        to_jpy: bool = True          # 決済通貨を円に換算する(価格に K を掛ける)
        # 日足の始値の点のスプレッドに使うサーバーの時。0 = MT5 の日足の始まり(ロールオーバー)。
        # 2 なら「判断は日足の確定、約定は 2 時間後」の近似(値段は始値のまま、スプレッドだけ 2 時台の平均)
        daily_fill_hour: int = 0

    def prepare_times(self, base: BaseParams, years, days, rng):
        s = _series(self.p, years, days)
        symbol = _db_symbol(self.p.path)
        if symbol is None or len(symbol) != 6:
            raise ValueError(f"real_pair needs a db:<symbol>/<timeframe> source, got {self.p.path!r}")
        native_pip = 0.01 if symbol.endswith("JPY") else 0.0001
        k = jpy_per_unit(symbol[3:], int(s.times[0])) if self.p.to_jpy else 1.0
        self._s, self._symbol, self._k = s, symbol, k
        sec = 1_000_000_000
        start = s.times - s.step_minutes * NS_MIN
        up = s.close >= s.open
        first = np.where(up, s.low, s.high)
        second = np.where(up, s.high, s.low)
        times = np.stack([start + sec, start + 2 * sec, start + 3 * sec, s.times - sec], axis=1).ravel()
        self._price = np.stack([s.open, first, second, s.close], axis=1).ravel() * k
        spread = s.spread if s.spread is not None else np.full(len(s), np.nan)
        self._spread = np.repeat(spread * k, 4)
        self._profile = None
        found = _hourly_spread(symbol)
        if found is not None:
            # 始値・終値の点はその時刻のサーバーの時の平均。高値・安値の点は足が覆う時間の平均
            # (日足の高安はロールオーバーの時刻には付かない。点の時刻は並びのための仮のもの)。
            # MT5 の日足はサーバーの 0 時(ロールオーバー)に始まる
            prof, rule = found
            if s.step_minutes >= 1440:
                h_open = np.full(len(s), int(self.p.daily_fill_hour) % 24, dtype=np.int64)
                h_close = np.full(len(s), 23)
            else:
                h_open = server_hour(start, rule)
                h_close = server_hour(s.times - sec, rule)
            n_h = max(1, s.step_minutes // 60)
            cover = np.stack([prof[(h_open + d) % 24] for d in range(n_h)], axis=1).mean(axis=1)
            self._profile = np.stack([prof[h_open], cover, cover, prof[h_close]], axis=1).ravel() * k
        b = replace(base, p0=float(s.close[0] * k), pip=native_pip * k, step_minutes=s.step_minutes,
                    annual_vol=_realized_vol(s.close, s.step_minutes),
                    start=str(np.datetime64(int(s.times[0]), "ns"))[:10])
        return times, b

    def apply(self, ctx: GenContext) -> None:
        super().apply(ctx)
        ctx.meta["_recorded_spread"] = self._spread
        if self._profile is not None:
            ctx.meta["_spread_profile"] = self._profile
        ctx.meta["_bar_points"] = 4
        ctx.meta["ohlc_points"] = 4           # 点は実データの OHLC(backtest は足の中を連続とみなす)
        ctx.meta["swap_pair"] = self._symbol.lower()
        ctx.meta["quote_to_jpy"] = self._k


class RealSkeleton(Component):
    """実データを骨格にした 1 分足の時刻列を用意し、骨格(終値・高値・安値)を meta に置く。

    骨格の時間足は実データの解像度(日足なら日足)。
    mode:
      replay     骨格 = 実データそのまま
      bootstrap  骨格 = 実データのリターンを block_days 日のブロック単位で並べ替えたもの(標本の平均ドリフトは残る)
      placebo    骨格 = 実データの日次リターンの符号だけをでたらめにしたもの。大きさ・ボラの塊・裾の厚さは同じで、
                 方向の情報だけが無い。ここで勝つ戦略は「合成の作り方の癖」で勝っている(偽薬の対照)
    """

    key = "source"
    stage = Stage.SOURCE

    @dataclass
    class Params(_SourceParams):
        mode: str = "replay"          # replay | bootstrap | placebo
        block_days: float = 20.0
        # bootstrap: 日次リターンの平均を引いてから並べ替える(この期間の上げ下げ = ドリフトを消す)。
        # 買いに偏った戦略が円安に乗っただけで勝つのを切り分ける参考の組(verify の real_bootstrap_demeaned)
        demean: bool = False

    def prepare_times(self, base: BaseParams, years, days, rng):
        s = _series(self.p, years, days)
        if s.step_minutes < base.step_minutes:
            raise ValueError("skeleton must be coarser than the base step")
        lc = np.log(s.close)
        lo_ = np.log(s.open)
        lh = np.log(np.maximum(s.high, np.maximum(s.open, s.close)))
        ll = np.log(np.minimum(s.low, np.minimum(s.open, s.close)))
        if self.p.mode == "bootstrap":
            lc, lo_, lh, ll = _block_bootstrap(lc, lo_, lh, ll, s, self.p.block_days, rng, self.p.demean)
        elif self.p.mode == "placebo":
            lc, lo_, lh, ll = _placebo(lc, lo_, lh, ll, rng)
        elif self.p.mode != "replay":
            raise ValueError(f"unknown skeleton mode {self.p.mode}")
        start_ns = int(s.times[0]) - s.step_minutes * NS_MIN
        span_days = (int(s.times[-1]) - start_ns) / NS_DAY
        start = str(np.datetime64(start_ns, "ns"))
        times = trading_times(start, days=span_days + 1e-9, step_minutes=base.step_minutes)
        times = times[times < int(s.times[-1])]
        # 骨格の各足の終わり = その時刻より前の最後のステップ
        idx = np.searchsorted(times, s.times, side="left") - 1
        ok = idx >= 0
        self._anchor = {
            "idx": idx[ok], "close": lc[ok], "open": lo_[ok], "high": lh[ok], "low": ll[ok],
            "has_ohlc": s.has_ohlc,
        }
        b = replace(base, p0=float(np.exp(lc[0])),
                    annual_vol=_realized_vol(np.exp(lc), s.step_minutes),
                    start=start[:10])
        return times, b

    def apply(self, ctx: GenContext) -> None:
        ctx.meta["anchor"] = self._anchor
        ctx.meta["real_source"] = f"{self.p.path} ({self.p.mode})"


def _block_bootstrap(lc, lo_, lh, ll, s: RealSeries, block_days: float, rng, demean: bool = False):
    """終値のリターンと、足の形(始値・高値・安値の終値からの相対位置)を、ブロックごとに引き直す。
    demean なら終値のリターンの平均を引いてから並べ替える(ドリフトを消す)。"""
    n = len(lc)
    r = np.diff(lc)
    if demean:
        r = r - r.mean()
    shape_o = (lo_ - np.r_[lc[0], lc[:-1]])[1:]     # 前の終値からの始値の窓
    shape_h = (lh - lc)[1:]
    shape_l = (ll - lc)[1:]
    bars_per_block = max(1, int(round(block_days * 1440 / s.step_minutes)))
    m = len(r)
    picks = []
    while len(picks) < m:
        st = int(rng.integers(0, max(1, m - bars_per_block)))
        picks.extend(range(st, min(m, st + bars_per_block)))
    picks = np.asarray(picks[:m])
    nr = r[picks]
    c = np.r_[lc[0], lc[0] + np.cumsum(nr)]
    o = np.r_[lo_[0], c[:-1] + shape_o[picks]]
    h = np.r_[lh[0], c[1:] + shape_h[picks]]
    lw = np.r_[ll[0], c[1:] + shape_l[picks]]
    h = np.maximum(h, np.maximum(o, c))
    lw = np.minimum(lw, np.minimum(o, c))
    assert len(c) == n
    return c, o, h, lw


def _placebo(lc, lo_, lh, ll, rng):
    """実データの日次リターンの符号だけをでたらめにした骨格(大きさ・ボラの塊・裾の厚さは実データのまま)。
    足の形(始値の窓、高値・安値の終値からの距離)も実データのものを使う。"""
    r = np.diff(lc)
    nr = np.abs(r) * np.where(rng.random(len(r)) < 0.5, -1.0, 1.0)
    c = np.r_[lc[0], lc[0] + np.cumsum(nr)]
    o = np.r_[lo_[0], c[:-1] + (lo_[1:] - lc[:-1])]
    h = np.maximum(c + (lh - lc), np.maximum(o, c))
    lw = np.minimum(c + (ll - lc), np.minimum(o, c))
    return c, o, h, lw


class AnchorBridge(Component):
    """経路を骨格に合わせる。

    0. 振れ幅の合わせ方(calibrate_vol)
       - "posterior"(既定、v4 / R008): 骨格の足ごとに、足の中の分散を g² に合わせる。g² は
         「直前の局所ボラ² を中心にした対数正規の事前分布(幅 vol_spread)」と「その足のリターン r ~ N(0, g²)」から、
         r を見た事後分布で引く(_posterior_goal)。g が分かっても r は N(0, g²) なので、足の中の経路は自分の過去だけを
         見ればマルチンゲールに近くなる(荒れ方から |r| が漏れない)
       - "local"(v2〜v3): 目標は実データの局所ボラと、その足自身のリターンの混ぜ合わせ(day_weight)。
         荒れ方から |r| が漏れ、静かな日は戻り・荒れた日は走りになる。day_weight をどう選んでも、
         1 時間足の RSI・ストキャス・ボリンジャーのどれかに年数百 pip の素の優位性が残っていた(R008)
       - "global"(v1): 経路全体で 1 つの倍率だけ合わせる
       - "none": 合わせない
    1. 骨格の各足の終わりで、終値との差を足し込む(ブラウン橋の補正)。
       clock="variance"(v2)は差を「動いた量」(実現分散の累積)に比例して配る。
       clock="time"(v1)は時刻に比例して配るため、荒れた時間帯の動きが静かな時間帯に戻され、
       日中の平均回帰(1 時間足の逆張りが勝つ穴)を作っていた
    2. OHLC があれば、各足の中で max(始値, 終値) より上のはみ出しを高値に、
       min(始値, 終値) より下のはみ出しを安値に、ちょうど届くよう伸縮する
       (はみ出しが無い足は、ランダムな位置に小さな山を作って届かせる)
    """

    key = "anchor"
    stage = Stage.ANCHOR

    @dataclass
    class Params:
        fit_high_low: bool = True
        calibrate_vol: str = "posterior"  # posterior | local | global | none
        vol_window: int = 21             # 実データのボラを測る足の本数(local は前後合わせて、posterior は直前)
        day_weight: float = 0.6          # local のとき、目標の分散に混ぜる「その足自身のリターン²」の重み
        # posterior のとき、事前分布 ln g² の分散(偽薬で単純な規則の素の優位性が 0 に近くなる値、R008)
        vol_spread: float = 0.7
        # posterior のとき、事前分布の中心 = (1 − w) x 直前 vol_window 本の r² の平均
        #                                   + w x 直前 prior_long_window 本の平均。
        # 直前 21 本だけだと、ボラの平均回帰と 21 本の推定のぶれで中心が偏る(静かな時期の後の足は実データの方が
        # 大きく動き、終値への補正が一方向のドリフトになって日中にトレンドが出る。荒れた時期の後は戻る)。
        # 偽薬でも 1 時間足のブレイクアウトが素で +1.2 pip / 取引(t 6)勝っていた(R009)。0 なら v4 と同じ
        prior_long_window: int = 250
        prior_long_weight: float = 0.25             # 0 / 0.25 / 0.5 / 0.75 を比べて決めた(R009)
        # 終値とのずれを配る時計: expected(事前の分散 vol² に比例、v4)| variance(動いた分 dx² に比例、v2〜v3)
        # | time(時刻に比例、v1)
        clock: str = "expected"

    def apply(self, ctx: GenContext) -> None:
        a = ctx.meta.get("anchor")
        if a is None:
            return
        idx, target = a["idx"], a["close"]
        x = ctx.logp
        mode = self.p.calibrate_vol
        if mode is True:                 # v1 の設定ファイル(真偽値)との互換
            mode = "global"
        dx = np.diff(x, prepend=x[0])
        ev = ctx.vol * ctx.vol                     # 1 歩の事前の分散(季節性・確率ボラ・指標の後の跳ね上がり込み)
        if mode == "posterior" and len(idx) > 2:
            goal = _posterior_goal(target, self.p.vol_window, self.p.vol_spread, ctx.rng,
                                   self.p.prior_long_window, self.p.prior_long_weight)
            # 足ごとの分散の見込み(事前の分散 + その足のジャンプ²)を g² に合わせる。実現した dx² で合わせると、
            # 足の中の大きな 1 歩の後が静かになる(未来の大きさを見た伸縮)ので使わない
            sc = _scale_to_goal(ev + ctx.jumps * ctx.jumps, idx, goal, squared=True)
            dx = dx * sc
            ev = ev * sc * sc
            x = x[0] + np.cumsum(dx)
        elif mode == "local" and len(idx) > 2:
            dx = dx * _local_scale(dx, idx, target, self.p.vol_window, self.p.day_weight)
            x = x[0] + np.cumsum(dx)
        elif mode == "global" and len(idx) > 10:
            vx = np.var(np.diff(x[idx]))
            vt = np.var(np.diff(target))
            if vx > 0:
                k = float(np.sqrt(vt / vx))
                x = x[0] + (x - x[0]) * k
                dx = dx * k
                ctx.meta["intraday_scale"] = k
        elif mode not in ("posterior", "local", "global", "none", False):
            raise ValueError(f"anchor.calibrate_vol must be posterior / local / global / none, got {mode!r}")
        err = target - x[idx]
        if self.p.clock == "expected":
            # 事前の分散の時計。実現した dx² に比例して配ると、ジャンプのような大きな 1 歩にその日の残り(未来)を
            # 含んだ補正が大きく乗り、大きく動いた後に戻る癖ができていた(R008)
            clock = np.cumsum(ev)
        elif self.p.clock == "variance":
            clock = np.cumsum(dx * dx)       # 実現分散の時計: 動いた分だけ補正を配る
        elif self.p.clock == "time":
            clock = np.arange(ctx.n, dtype=float)
        else:
            raise ValueError(f"anchor.clock must be expected / variance / time, got {self.p.clock!r}")
        x = x + np.interp(clock, clock[idx], err)
        if a["has_ohlc"] and self.p.fit_high_low and len(idx) > 1:
            x = _fit_envelope(x, idx, a["high"], a["low"], ctx.rng)
        ctx.logp = x


def _local_scale(dx: np.ndarray, idx: np.ndarray, target: np.ndarray, window: int, day_weight: float) -> np.ndarray:
    """骨格の足ごとの伸縮率。足の中の実現分散を目標に合わせる。

    目標の分散 = (1 - day_weight) x 局所ボラ² + day_weight x その足のリターン²
    (局所ボラ = 前後 window 本のリターンの二乗平均)。平均すると局所ボラ² に等しい。
    day_weight = 0 だと静かに終わった日も足の中が荒れ、終値へ引き戻されて日中の平均回帰が生まれる。
    1 に寄せすぎると逆に勢いが生まれる。既定の 0.6 は、符号をでたらめにした偽薬(R006)で
    単純な規則の素の優位性がほぼ 0 になる値(docs/proposals/markets/R005-bridge-artifact.md)。
    足の中の動きの形(時間帯・指標・短期のボラの塊)は残る。
    """
    r = np.diff(target)
    w = max(1, int(window))
    local2 = np.convolve(r * r, np.ones(w) / w, mode="same")
    g = np.sqrt((1 - day_weight) * local2 + day_weight * r * r)
    return _scale_to_goal(dx, idx, np.r_[g[0], g])                       # 最初の足は 2 本目の値で代用


_Z_GRID = np.linspace(-4.0, 4.0, 81)


def _posterior_goal(target: np.ndarray, window: int, spread: float, rng, long_window: int = 250,
                    long_weight: float = 0.0) -> np.ndarray:
    """骨格の足ごとの、足の中の振れ幅 g(_scale_to_goal の目標)。

    L² = 直前 window 本のリターン² の平均(その足と、それより後は使わない)。long_weight > 0 なら、直前 long_window 本の
    平均と (1 − w) : w で混ぜる(R009。直前 21 本だけの見込みは偏る)。事前分布 ln g² ~ N(ln L² − s²/2, s²)
    (s² = spread、E[g²] = L²)と r ~ N(0, g²) から、その足のリターン r を見た事後分布で g² を引く(z の格子 ±4、81 点)。
    L を前後の窓で測ると、荒れる時期の直前の足で g だけが先回りして大きくなり、日中の戻りを作っていた(R008)。
    """
    r = np.diff(target)
    if len(r) == 0:
        return np.zeros(1)
    w = max(2, int(window))
    r2 = r * r
    csum = np.r_[0.0, np.cumsum(r2)]
    k = np.arange(len(r))
    lo = np.maximum(k - w, 0)
    L2 = np.where(k > 0, (csum[k] - csum[lo]) / np.maximum(k - lo, 1), np.nan)
    early = k < min(w, len(r))                                          # 最初の w 本は先頭 w 本の平均で代用
    L2[early] = np.mean(r2[:min(w, len(r))])
    pos = L2[L2 > 0]
    L2 = np.where(L2 > 0, L2, np.median(pos) if len(pos) else 1e-12)
    if long_weight > 0:
        wl = max(w, int(long_window))
        lo_l = np.maximum(k - wl, 0)
        LL = np.where(k > 0, (csum[k] - csum[lo_l]) / np.maximum(k - lo_l, 1), np.nan)
        LL[early] = L2[early]
        LL = np.where(LL > 0, LL, L2)
        L2 = (1 - long_weight) * L2 + long_weight * LL
    s2 = max(float(spread), 1e-9)
    v = L2[:, None] * np.exp(np.sqrt(s2) * _Z_GRID[None, :] - s2 / 2)  # 候補の g²(足 x 格子)
    logw = -0.5 * _Z_GRID[None, :] ** 2 - 0.5 * np.log(v) - r2[:, None] / (2 * v)
    logw -= logw.max(axis=1, keepdims=True)
    cdf = np.cumsum(np.exp(logw), axis=1)
    u = rng.random(len(r))[:, None] * cdf[:, -1:]
    j = np.minimum((cdf < u).sum(axis=1), len(_Z_GRID) - 1)
    g = np.sqrt(v[np.arange(len(r)), j])
    return np.r_[g[0], g]                                                # 最初の足は 2 本目の値で代用


def _scale_to_goal(dx: np.ndarray, idx: np.ndarray, goal: np.ndarray, squared: bool = False) -> np.ndarray:
    """足 k の中の分散が goal[k]² になる伸縮率(ステップごと)。squared なら dx は 1 歩ごとの分散そのもの。"""
    starts = np.clip(np.r_[0, idx[:-1] + 1], 0, len(dx) - 1)             # 足 k の中身は starts[k]..idx[k]
    rv = np.add.reduceat(dx if squared else dx * dx, starts)[:len(idx)]
    f = np.where(rv > 0, goal / np.sqrt(np.maximum(rv, 1e-300)), 1.0)
    lens = np.diff(np.r_[starts, idx[-1] + 1])
    scale = np.full(len(dx), f[-1])
    scale[:idx[-1] + 1] = np.repeat(f, lens)
    return scale


def _fit_envelope(x, idx, high, low, rng):
    x = x.copy()
    for k in range(1, len(idx)):
        a, b = int(idx[k - 1]) + 1, int(idx[k]) + 1
        if b - a < 3:
            continue
        seg = x[a:b]
        o, c = x[a - 1], seg[-1]
        top, bot = max(o, c), min(o, c)
        H, L = max(high[k], top), min(low[k], bot)
        seg = seg.copy()
        inner = seg[1:-1]
        mx, mn = inner.max(), inner.min()
        if mx > top:
            up = inner > top
            inner[up] = top + (inner[up] - top) * (H - top) / (mx - top)
        elif top < H:
            j = int(rng.integers(0, len(inner)))
            inner[j] = H
        if mn < bot:
            dn = inner < bot
            inner[dn] = bot + (inner[dn] - bot) * (L - bot) / (mn - bot)
        elif bot > L:
            j = int(rng.integers(0, len(inner)))
            inner[j] = L
        seg[1:-1] = inner
        x[a:b] = seg
    return x
