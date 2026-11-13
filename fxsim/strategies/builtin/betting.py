"""S030 bet: カジノの賭け方(マネーマネジメント)を FX でそのまま使う(1 時間足)。

利確と損切りを同じ幅(既定 ±20 pip)に置いた「ほぼ五分五分の賭け」を繰り返し、賭け金(数量)を staking.py の
賭け方で変える。向きは coin(コイン投げ)か trend(100 本 EMA の向き)。
1 回ごとの期待値は費用の分だけマイナスで、賭け方はそれを変えない。変わるのは損益の分布の形:
マーチンゲールは小さな勝ちが続いた後にまれに大きく負け(打ち止め max_mult か口座の限界で)、
パーリーは小さな負けが続いた後にまれに大きく勝つ。

このファイルは例外的に、同じ仕組みを賭け方ごとの名前でも登録する(martingale / paroli / dalembert / fibonacci /
labouchere / oscars_grind / kelly)。中身は bet の system を変えただけ。
"""
from dataclasses import dataclass

import numpy as np

from .. import STRATEGIES
from ..base import Strategy
from ..indicators import ema
from ..staking import STAKING_SYSTEMS, Staker


class Bet(Strategy):
    name = "bet"

    @dataclass
    class Params:
        timeframe: int = 60
        system: str = "flat"
        direction: str = "coin"        # coin | trend
        tp_pips: float = 20.0
        sl_pips: float = 20.0
        max_mult: float = 64.0
        trend_n: int = 100
        kelly_fraction: float = 0.5

    def prepare(self, bars, rng):
        p = self.p
        if p.system not in STAKING_SYSTEMS:
            raise ValueError(f"unknown system {p.system!r}. known: {STAKING_SYSTEMS}")
        if p.direction not in ("coin", "trend"):
            raise ValueError("direction must be coin or trend")
        self.staker = Staker(system=p.system, max_mult=p.max_mult, kelly_fraction=p.kelly_fraction)
        self.flip = rng.random(len(bars)) < 0.5            # 足ごとのコイン(prepare の rng で再現できる)
        self.trend = np.sign(bars.close - ema(bars.close, p.trend_n))
        self.warmup = p.trend_n if p.direction == "trend" else 0
        self.seen = 0
        self.unit = None

    def _learn(self, ctx) -> None:
        trades = ctx.trades
        for t in trades[self.seen:]:
            risk = self.p.sl_pips * ctx.pip * t.units
            self.staker.update(t.pnl / risk if risk > 0 else float(np.sign(t.pnl)))
        self.seen = len(trades)

    def _units(self, ctx) -> float:
        base = ctx.base_units
        if self.p.system == "kelly":
            f = self.staker.kelly_f() * self.p.kelly_fraction
            if f <= 0:
                return 1000.0                               # 推定できるまで(または優位性が無い間)は最小の量で様子を見る
            return max(1000.0, ctx.equity * f / (self.p.sl_pips * ctx.pip))
        return base * self.staker.mult

    def on_bar(self, i, ctx):
        self._learn(ctx)
        if ctx.side != 0:
            return
        if self.p.direction == "coin":
            side = 1 if self.flip[i] else -1
        else:
            side = int(self.trend[i])
            if side == 0:
                return
        pip = ctx.pip
        (ctx.buy if side > 0 else ctx.sell)(sl=self.p.sl_pips * pip, tp=self.p.tp_pips * pip, units=self._units(ctx))


STRATEGIES.register("bet", version="1", description="利確=損切りの賭けを繰り返し、賭け方(system)で数量を変える")(Bet)


def _alias(sys_name: str, description: str) -> type:
    """賭け方 sys_name を既定にした bet を、その名前で登録する。"""

    @dataclass
    class Params(Bet.Params):
        system: str = sys_name

    cls = type(f"Bet_{sys_name}", (Bet,), {"name": sys_name, "Params": Params})
    STRATEGIES.register(sys_name, version="1", description=description)(cls)
    return cls


Martingale = _alias("martingale", "負けたら倍賭け、勝ったら 1 単位に戻す(打ち止め 64 倍)")
Paroli = _alias("paroli", "勝ったら倍賭け、3 連勝か負けで 1 単位に戻す(逆マーチンゲール)")
DAlembert = _alias("dalembert", "負けたら +1 単位、勝ったら −1 単位(ダランベール)")
Fibonacci = _alias("fibonacci", "負けたらフィボナッチ数列を 1 つ進め、勝ったら 2 つ戻す")
Labouchere = _alias("labouchere", "数列 1-2-3 の両端の和を賭け、勝ったら消し、負けたら末尾に足す(ラブシェール)")
OscarsGrind = _alias("oscars_grind", "勝ったら 1 単位足し、1 周 +1 単位で戻す(オスカーズ・グラインド)")
Kelly = _alias("kelly", "過去の勝率と損益比から推定したケリー基準の半分を賭ける")
