"""市場の悪意。経路を見て「多くの参加者が困る方向」に動かす部品。

いずれも POST_PATH 段で、出来上がった経路に対して働く。
詳しい考え方は docs/MARKET_MALICE.md。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..base import Component, GenContext, Stage
from ..blocks import apply_block_increments, block_view, rolling_prev_max, rolling_prev_min


class CrowdSqueeze(Component):
    """群衆の逆を突く圧力(ペイン・トレード)。

    群衆の構成 w は -1(逆張り勢が多い)〜 +1(順張り勢が多い)をゆっくり行き来する。
    群衆のポジション = w * tanh(z)、z は価格と EMA の乖離をボラで割ったもの。
    価格は群衆のポジションと逆へ drift する。
      - 順張りが多い時期: 伸びた分が戻される(トレンドフォローが負ける)
      - 逆張りが多い時期: 伸びた方向にさらに走る(逆張りが焼かれる)
    どちらが多いかは外から見えないので、単純な規則はどちらも安定して勝てない。
    """

    key = "crowd_squeeze"
    stage = Stage.POST_PATH

    @dataclass
    class Params:
        strength: float = 0.06          # 群衆が振り切れたとき、1 ブロックの典型的な値幅の何倍を逆へ動かすか
        ema_hours: float = 24.0         # 群衆が見ている移動平均
        mix_halflife_days: float = 8.0  # 群衆の構成が入れ替わる速さ
        mix_scale: float = 1.2          # w = tanh(u) の u の標準偏差
        block_minutes: int = 15

    def apply(self, ctx: GenContext) -> None:
        bs = max(1, self.p.block_minutes // ctx.base.step_minutes)
        _, ends, _, _, close = block_view(ctx, bs)
        nb = len(close)
        blocks_per_hour = 60 / self.p.block_minutes
        a_ema = 1 - 0.5 ** (1 / (self.p.ema_hours * blocks_per_hour))
        a_sc = 1 - 0.5 ** (1 / (24 * blocks_per_hour))
        a_u = 0.5 ** (1 / (self.p.mix_halflife_days * 24 * blocks_per_hour))
        su = self.p.mix_scale * np.sqrt(1 - a_u * a_u)
        noise = (ctx.rng.standard_normal(nb) * su).tolist()
        sigma_block = float(np.median(ctx.vol)) * np.sqrt(bs)
        k = self.p.strength
        n_ema = self.p.ema_hours * blocks_per_hour
        d = np.zeros(nb)
        w_trace = np.zeros(nb)
        off = 0.0
        cl = close.tolist()
        p_prev = cl[0]
        ema = cl[0]
        scale = sigma_block
        u = ctx.rng.standard_normal() * self.p.mix_scale
        tanh = np.tanh
        for b in range(nb):
            p = cl[b] + off
            scale += a_sc * (abs(p - p_prev) - scale)
            p_prev = p
            ema += a_ema * (p - ema)
            z = (p - ema) / (scale * np.sqrt(n_ema) * 0.5 + 1e-12)
            u = a_u * u + noise[b]
            w = tanh(u)
            w_trace[b] = w
            step = -k * w * tanh(z) * scale
            d[b] = step
            off += step
        apply_block_increments(ctx, d, bs)
        # 後段(ダマシ・ストップ狩り)が「いまの群衆」を読めるよう、ステップ単位に展開して残す
        ctx.meta["crowd_mix"] = np.repeat(w_trace, bs)[:ctx.n]


class StopHunt(Component):
    """ストップ狩り。

    逆指値が溜まりやすい水準(キリ番、直近の高値・安値)に価格が近づくと、
    一定の確率でそこを overshoot だけ突き抜ける。結末は 2 通り:
      - 戻り(revert): ヒゲで元の経路へ戻り、さらに reversal_pips だけ逆へずれる
        (「損切りさせられた直後に、思っていた方向へ動く」。飛び乗った順張りも焼かれる)
      - 連鎖(cascade): 損切りが損切りを呼び、cascade_pips だけ突き抜けた方向へ走ったまま
        (ヒゲを逆張りで売った/買った人が焼かれる)
    戻りの確率は revert_prob + crowd_gain * w。w は crowd_squeeze が残す群衆の構成
    (+1 順張り勢が多い / -1 逆張り勢が多い)で、無ければ 0。
    つまり順張り勢が多ければ戻し、逆張り勢が多ければ走らせる。どちらに賭けても当てにならない。
    """

    key = "stop_hunt"
    stage = Stage.POST_PATH

    @dataclass
    class Params:
        prob: float = 0.08              # 候補ブロックごとの発生確率
        trigger_pips: float = 8.0       # 水準までこの距離以内なら候補
        round_pips: float = 50.0        # キリ番の間隔(xx.00 / xx.50)
        swing_hours: float = 24.0       # 直近高値・安値を見る期間
        overshoot_pips: tuple = (2.0, 10.0)
        rise_minutes: tuple = (2, 8)
        decay_minutes: tuple = (8, 30)
        reversal_pips: tuple = (0.0, 12.0)
        revert_prob: float = 0.65
        crowd_gain: float = 0.4
        cascade_pips: tuple = (0.0, 6.0)      # 水準までの距離 + overshoot に足す分
        cooldown_hours: float = 4.0
        block_minutes: int = 15

    def apply(self, ctx: GenContext) -> None:
        p = self.p
        bs = max(1, p.block_minutes // ctx.base.step_minutes)
        starts, ends, hi, lo, close = block_view(ctx, bs)
        nb = len(close)
        lb = max(1, int(p.swing_hours * 60 / p.block_minutes))
        swing_hi = rolling_prev_max(hi, lb)
        swing_lo = rolling_prev_min(lo, lb)
        price = np.exp(close)
        step = p.round_pips * ctx.base.pip
        round_up = np.ceil(price / step) * step
        round_dn = np.floor(price / step) * step
        lvl_up = np.where(np.exp(swing_hi) > price, np.minimum(np.exp(swing_hi), round_up), round_up)
        lvl_dn = np.where(np.exp(swing_lo) < price, np.maximum(np.exp(swing_lo), round_dn), round_dn)
        dist_up = (lvl_up - price) / ctx.base.pip
        dist_dn = (price - lvl_dn) / ctx.base.pip
        rng = ctx.rng
        cand = np.flatnonzero(((dist_up < p.trigger_pips) | (dist_dn < p.trigger_pips))
                              & (rng.random(nb) < p.prob))
        cooldown = int(p.cooldown_hours * 60 / p.block_minutes)
        last = -10 ** 9
        mix = ctx.meta.get("crowd_mix")
        add = np.zeros(ctx.n)
        persist = np.zeros(ctx.n + 1)   # 持続するずれは差分で積み、最後に 1 回だけ累積する
        mask = np.zeros(ctx.n, dtype=bool)
        sm = ctx.base.step_minutes
        for b in cand.tolist():
            if b - last < cooldown or b + 1 >= nb:
                continue
            last = b
            up = dist_up[b] <= dist_dn[b]
            direction = 1.0 if up else -1.0
            level = lvl_up[b] if up else lvl_dn[b]
            over = rng.uniform(*p.overshoot_pips)
            target = level + direction * over * ctx.base.pip
            amp = abs(np.log(target) - close[b])
            rise = max(1, int(rng.uniform(*p.rise_minutes) / sm))
            decay = max(1.0, rng.uniform(*p.decay_minutes) / sm)
            s0 = int(ends[b]) + 1
            w = float(mix[s0 - 1]) if mix is not None else 0.0
            revert = rng.random() < np.clip(p.revert_prob + p.crowd_gain * w, 0.0, 1.0)
            if revert:
                final = -ctx.pips_to_log(rng.uniform(*p.reversal_pips))   # ヒゲと逆へ
            else:
                final = amp + ctx.pips_to_log(rng.uniform(*p.cascade_pips))  # 突き抜けた方向へ
            L = min(ctx.n - s0, rise + int(decay * 5))
            if rise >= L:
                continue
            k = np.empty(L)
            k[:rise] = amp * np.sqrt(np.arange(1, rise + 1) / rise)
            tt = np.arange(L - rise)
            e = np.exp(-tt / decay)
            k[rise:] = amp * e + final * (1 - e)
            add[s0:s0 + L] += direction * k
            mask[s0:s0 + rise + int(decay)] = True
            if s0 + L < ctx.n:
                persist[s0 + L] += direction * final   # ずれは持続する
            ctx.event(s0, "stop_hunt" if revert else "stop_cascade", direction=int(direction), level=float(level),
                      overshoot_pips=float(over), final_pips=float(ctx.log_to_pips(final)))
        ctx.logp = ctx.logp + add + np.cumsum(persist[:ctx.n])
        ctx.meta["hunt_mask"] = mask


class FalseBreakout(Component):
    """ダマシと走り。直近レンジを抜けたとき、ダマシ(押し戻し)か本物(走り)かのどちらかになる。

    - ダマシ: 抜けた幅 + extra_pips を、delay 後に revert の時間をかけて打ち消す(持続的なずれ)。
      確率は局面ごとの基準(prob_range / prob_trend / prob_other)+ crowd_gain * w。
    - 走り: 確率 follow_prob - crowd_gain * w で、follow_pips だけ同じ方向へ伸びる。
    w は crowd_squeeze が残す群衆の構成(+1 順張り勢 / -1 逆張り勢)。
    順張り勢が多いときは抜けを罠にし、逆張り勢が多いときは抜けを走らせる。
    ダマシだけだと日足が平均回帰しすぎて(分散比が 1 を大きく割る)逆張りが楽に勝つので、
    走りと釣り合わせて、ブレイクアウトを「当たり外れの大きい賭け」にする。
    """

    key = "false_breakout"
    stage = Stage.POST_PATH

    @dataclass
    class Params:
        lookback_hours: float = 24.0
        prob_range: float = 0.45
        prob_trend: float = 0.2
        prob_other: float = 0.35
        delay_blocks: tuple = (1, 4)
        revert_blocks: tuple = (3, 12)
        extra_pips: tuple = (3.0, 20.0)
        follow_prob: float = 0.2
        follow_pips: tuple = (5.0, 25.0)
        crowd_gain: float = 0.35
        cooldown_hours: float = 6.0
        block_minutes: int = 15

    def apply(self, ctx: GenContext) -> None:
        p = self.p
        bs = max(1, p.block_minutes // ctx.base.step_minutes)
        _, ends, hi, lo, close = block_view(ctx, bs)
        nb = len(close)
        lb = max(2, int(p.lookback_hours * 60 / p.block_minutes))
        regime = ctx.meta.get("regime")
        reg_b = regime[ends].tolist() if regime is not None else [-1] * nb
        rng = ctx.rng
        u = rng.random(nb).tolist()
        mix = ctx.meta.get("crowd_mix")
        mix_b = mix[ends].tolist() if mix is not None else [0.0] * nb
        pending = np.zeros(nb + 64)   # ブロックごとに加える log 変化
        off = 0.0
        dq_hi: deque = deque()
        dq_lo: deque = deque()
        hi_l, lo_l, cl = hi.tolist(), lo.tolist(), close.tolist()
        cooldown = int(p.cooldown_hours * 60 / p.block_minutes)
        last = -10 ** 9
        pip_log = float(ctx.pips_to_log(1.0))
        for b in range(nb):
            off += pending[b]
            h = hi_l[b] + off
            l = lo_l[b] + off
            c = cl[b] + off
            if b >= lb and b - last >= cooldown and b + 20 < nb:
                prior_hi = dq_hi[0][1]
                prior_lo = dq_lo[0][1]
                direction = 1 if c > prior_hi else (-1 if c < prior_lo else 0)
                if direction:
                    r = reg_b[b]
                    w = mix_b[b]
                    prob = p.prob_range if r == 0 else (p.prob_trend if r in (1, 2) else p.prob_other)
                    prob = min(max(prob + p.crowd_gain * w, 0.0), 1.0)
                    run = min(max(p.follow_prob - p.crowd_gain * w, 0.0), 1.0 - prob)
                    if u[b] < prob:
                        last = b
                        excess = (c - prior_hi) if direction > 0 else (prior_lo - c)
                        total = excess + rng.uniform(*p.extra_pips) * pip_log
                        delay = int(rng.integers(p.delay_blocks[0], p.delay_blocks[1] + 1))
                        rev = int(rng.integers(p.revert_blocks[0], p.revert_blocks[1] + 1))
                        pending[b + delay:b + delay + rev] += -direction * total / rev
                        ctx.event(int(ends[b]) + 1, "false_breakout", direction=direction,
                                  pips=float(total / pip_log))
                    elif u[b] < prob + run:
                        last = b
                        total = rng.uniform(*p.follow_pips) * pip_log
                        rev = int(rng.integers(p.revert_blocks[0], p.revert_blocks[1] + 1))
                        pending[b + 1:b + 1 + rev] += direction * total / rev
                        ctx.event(int(ends[b]) + 1, "breakout_run", direction=direction,
                                  pips=float(total / pip_log))
            # 窓を更新(単調デック)
            while dq_hi and dq_hi[-1][1] <= h:
                dq_hi.pop()
            dq_hi.append((b, h))
            while dq_lo and dq_lo[-1][1] >= l:
                dq_lo.pop()
            dq_lo.append((b, l))
            while dq_hi[0][0] <= b - lb:
                dq_hi.popleft()
            while dq_lo[0][0] <= b - lb:
                dq_lo.popleft()
        # pending[b] はブロック b の中で効かせる(apply_block_increments は d[k] をブロック k+1 に置く)
        apply_block_increments(ctx, pending[1:nb + 1], bs)

