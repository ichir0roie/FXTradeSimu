"""シミュレーションの判断を実口座の注文に直して出す。

突き合わせの決まり(plan):
    1. 実口座の建玉がシミュレーションと食い違うときは、シミュレーションの手仕舞いに合わせる
       - シミュレーションが持っていない(損切り・利確・ロスカットで閉じた)/ 向きが違う → 実口座を決済する
       - 逆(実口座だけ損切りに掛かった、起動したときにシミュレーションがすでに持っていた)は、新しく建て直さない。
         次にシミュレーションが注文を出すまで待つ(途中から乗ると、戦略が見ていない値段で建てることになる)
    2. 最後の確定足の注文を、Backtester の _fill と同じ決まりで実口座に流す
       - enter: 同じ向きを持っていれば何もしない。逆向きなら決済してから建てる。損切り・利確は約定値からの距離
       - close / add / reduce: 同じ向きを持っているときだけ
    3. 注文が無く、シミュレーションの同じ建玉(建てた足が同じ)で損切り・利確が動いていたら、実口座も同じ価格に動かす
       (トレーリング。すでに越えた価格なら、その場で決済する)
数量はシミュレーションの数量(通貨)x volume_scale をロットに直し、volume_step で切り捨てる。

口座の床(protect、資産の 9 割を守る。guards.EquityFloor と同じ決まりを実口座の有効証拠金で):
    - 床 = 月初の有効証拠金 x protect。同じ口座の常駐どうしで data/trading/account-<login>.json を共有する
    - floor_check_seconds ごとと、注文を出す直前に有効証拠金を比べる。割ったら、この常駐の建玉を全部決済し、
      その月は新規(open / add)を出さない。手動やほかの EA の建玉には触らない
    - 新規の量は、口座の全建玉(銘柄・magic を問わない)とこの新規が、サーバーの逆指値まで飛んで窓で滑っても
      (逆指値までの損失 x gap_mult、MT5 の order_calc_profit で口座の通貨に換算)床を割らない量まで減らす。
      逆指値の無い建玉が口座にあれば、最悪の損失が分からないので新規を出さない
    - シミュレーションの中の床は切る(シミュレーションの有効証拠金は実口座と別物なので、床は実口座だけで見る)
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..backtest.broker import BrokerConfig
from ..backtest.engine import EngineConfig
from ..calendar import NS_MIN
from ..cases.loader import load_cases
from ..cases.model import BehaviorCase
from ..strategies import STRATEGIES
from ..strategies.guards import EquityFloor, floor_lots
from ..strategies.main.base import MainStrategy
from .config import TraderConfig
from .feed import pip_from_digits, rates_to_bars
from .mt5 import LivePosition, MT5Client, MT5Error, OrderResult, SymbolSpec
from .signals import SimSnapshot, simulate

log = logging.getLogger(__name__)

# 少し待てば通りうる失敗(リクオート・価格の変化・タイムアウト・要求が多すぎる)は、その場でやり直す
RETRY_CODES = {10004, 10012, 10020, 10021, 10024}


@dataclass
class Action:
    kind: str                       # close_all | open | add | reduce | modify
    side: int = 0
    lots: float = 0.0
    sl_dist: float | None = None
    tp_dist: float | None = None
    sl_from_average: bool = False
    sl: float | None = None         # modify: 絶対価格
    tp: float | None = None
    reason: str = ""


@dataclass
class TraderState:
    anchor: int | None = None               # 起点の足の時刻(サーバー時刻、秒)
    last_forming: int | None = None         # 最後に処理したときの作りかけの足の時刻(サーバー時刻、秒)
    last_bar_t: int | None = None           # 最後に処理した確定足(UTC ns)
    utc_offset_sec: int | None = None       # 最後に分かったサーバー時刻 − UTC
    sim_position_t: int | None = None       # 前回のシミュレーションの建玉(建てた足の時刻)と損切り・利確
    sim_sl: float | None = None
    sim_tp: float | None = None
    updated_at: float | None = None

    @classmethod
    def load(cls, path: Path) -> TraderState:
        if not path.exists():
            return cls()
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        tmp.replace(path)


def net_side(real: list[LivePosition]) -> int:
    """実口座の建玉の向き。買いと売りが混ざっていれば 0(全部決済の対象)。"""
    sides = {p.side for p in real}
    return sides.pop() if len(sides) == 1 else 0


def plan(snap: SimSnapshot, real: list[LivePosition], prev: TraderState,
         to_lots: Callable[[float], float]) -> list[Action]:
    """シミュレーションの判断(snap)と実口座の建玉(real)から、出す注文を決める。副作用なし。"""
    actions: list[Action] = []
    side = net_side(real)
    sim = snap.position
    if real and side == 0:
        actions.append(Action("close_all", reason="mixed sides"))
    elif side and (sim is None or sim.side != side):
        actions.append(Action("close_all", reason="sim flat" if sim is None else "sim reversed"))
        side = 0
    for o in snap.orders:
        if o.kind == "close":
            if side:
                actions.append(Action("close_all", reason="signal"))
                side = 0
        elif o.kind == "enter":
            if side == o.side:
                continue
            if side:
                actions.append(Action("close_all", reason="reverse"))
            actions.append(Action("open", o.side, to_lots(o.units), o.sl_dist, o.tp_dist, reason="signal"))
            side = o.side
        elif o.kind == "add" and side == o.side:
            actions.append(Action("add", o.side, to_lots(o.units), o.sl_dist, o.tp_dist, o.sl_from_average,
                                  reason="signal"))
        elif o.kind == "reduce" and side == o.side:
            actions.append(Action("reduce", o.side, to_lots(o.units), reason="signal"))
    if (sim is not None and side == sim.side and not any(a.kind in ("open", "add") for a in actions)
            and prev.sim_position_t == snap.position_t and (sim.sl, sim.tp) != (prev.sim_sl, prev.sim_tp)):
        actions.append(Action("modify", side, sl=sim.sl, tp=sim.tp, reason="sim moved stops"))
    return actions


def lots_for(units: float, spec: SymbolSpec, scale: float) -> float:
    """通貨の数量をロットにする(volume_step で切り捨て。最小に届かなければ 0)。"""
    raw = units * scale / spec.contract_size
    steps = math.floor(raw / spec.volume_step + 1e-9)
    lots = round(steps * spec.volume_step, 8)
    return lots if lots >= spec.volume_min else 0.0


class LiveTrader:
    def __init__(self, cfg: TraderConfig, client: MT5Client, behavior: BehaviorCase | None = None,
                 strategy_factory: Callable[[], object] | None = None, sleep: Callable[[float], None] | None = None,
                 expected_login: int | None = None):
        self.cfg = cfg
        self.expected_login = expected_login
        self.client = client
        self.behavior = behavior or load_cases().behaviors[cfg.behavior]
        params = self.behavior.strategy_params()
        if strategy_factory is None and isinstance(STRATEGIES.create(self.behavior.strategy, params), MainStrategy):
            # 床は実口座の有効証拠金で見る(下の check_floor)。シミュレーションの中では切る
            params = {**params, "guard": {**(params.get("guard") or {}), "protect": 0.0}}
        self.make_strategy = strategy_factory or (lambda: STRATEGIES.create(self.behavior.strategy, params))
        probe = self.make_strategy()
        if getattr(probe, "multi", False):
            raise ValueError(f"{self.behavior.id}: multi-pair strategies are not supported by fxsim trade")
        self.timeframe = int(probe.timeframe)
        b = cfg.sim_broker
        overrides = dict(b.get("overrides") or {})
        if getattr(probe, "engine", "classic") == "mt5" and "mt5" not in overrides:
            # 本命の戦略: 口座の決まり(ロット・証拠金・両建て)は動作ケースのものを使う。約定は素直なまま
            overrides["mt5"] = self.behavior.broker_spec["overrides"].get("mt5", {})
        self.sim_broker = BrokerConfig.preset(b.get("preset", "fair"), overrides)
        self.engine_cfg = EngineConfig(**self.behavior.engine)
        self.schedule = self.behavior.trading_schedule()
        self.sleep = sleep if sleep is not None else time.sleep
        d = cfg.state_dir if not cfg.dry_run else cfg.state_dir.with_name(cfg.state_dir.name + ".dry")
        self.state_path = d / "state.json"
        self.journal_path = d / "journal.jsonl"
        self.state = TraderState.load(self.state_path)
        self.poll_interval = cfg.poll_interval(self.timeframe)
        self._floor_checked = 0.0
        self._login: int | None = None

    # ---- 記録 ----
    def journal(self, kind: str, **data) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "kind": kind, **data}
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    # ---- 起動 ----
    def startup(self) -> dict:
        acct = self.client.account()
        kind = self.client.account_kind()
        if kind != self.cfg.account:
            raise MT5Error(f"account {acct['login']} is a {kind} account "
                           f"but the config says account: {self.cfg.account}")
        if kind == "real" and not self.cfg.allow_real:
            raise MT5Error(f"account {acct['login']} is a real account (set allow_real: true to trade it)")
        if self.expected_login is not None and int(acct["login"]) != self.expected_login:
            # 1 つの端末は 1 つの口座にしかつながらない。ほかの常駐が別の口座でログインし直した、など
            raise MT5Error(f"terminal is logged in to {acct['login']}, "
                           f"not {self.expected_login} from .env.{self.cfg.account}")
        self._login = int(acct["login"])
        spec = self.client.symbol(self.cfg.symbol)
        if not self.client.trade_allowed() and not self.cfg.dry_run:
            log.warning("algo trading is disabled in the terminal; orders will be rejected (10027)")
        info = {"account": acct, "symbol": asdict(spec), "behavior": self.behavior.definition(),
                "timeframe": self.timeframe, "dry_run": self.cfg.dry_run}
        self.journal("startup", **info)
        log.info("account %s (%s, %s) balance %.2f; trading %s %s with %s (%s)%s", acct["login"], acct["server"],
                 acct["currency"], acct["balance"], self.cfg.symbol, self.timeframe, self.behavior.id,
                 self.behavior.strategy, " [dry-run]" if self.cfg.dry_run else "")
        return info

    # ---- 口座の床 ----
    @property
    def floor_path(self) -> Path:
        login = self._login if self._login is not None else int(self.client.account()["login"])
        suffix = ".dry" if self.cfg.dry_run else ""
        return self.cfg.state_dir.parent / f"account-{login}{suffix}.json"

    def load_floor(self) -> EquityFloor:
        f = EquityFloor(self.cfg.protect)
        if self.floor_path.exists():
            with open(self.floor_path, encoding="utf-8") as fh:
                d = json.load(fh)
            f.key, f.floor, f.halted, f.trips = d.get("key"), d.get("floor", 0.0), d.get("halted", False), \
                d.get("trips", 0)
        return f

    def save_floor(self, f: EquityFloor) -> None:
        path = self.floor_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{self.cfg.name}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"key": f.key, "floor": f.floor, "halted": f.halted, "trips": f.trips, "protect": f.protect,
                       "updated_at": time.time(), "by": self.cfg.name}, fh, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def check_floor(self) -> tuple[EquityFloor, float]:
        """有効証拠金を床と比べる。割ったら止めて、この常駐の建玉を全部決済する。(床, 有効証拠金)を返す。"""
        self._floor_checked = time.time()
        f = self.load_floor()
        equity = float(self.client.account()["equity"])
        if self.cfg.protect <= 0:
            return f, equity
        f.update(time.time_ns(), equity)
        tripped = f.check(equity)
        self.save_floor(f)
        if tripped:
            log.error("equity %.0f fell to the floor %.0f (%.0f%% of the month's start): closing and halting "
                      "new entries for this month", equity, f.floor, f.protect * 100)
            results = [] if self.cfg.dry_run else self.execute(Action("close_all", reason="floor"))
            self.journal("floor", equity=equity, floor=f.floor, results=[asdict(r) for r in results],
                         dry_run=self.cfg.dry_run)
        return f, equity

    def account_worst_loss(self, skip_magic: int | None = None) -> float | None:
        """口座の全建玉が、今の決済値段からサーバーの逆指値まで飛んで窓で滑ったときの損失の合計(口座の通貨)。
        逆指値の無い建玉があれば None。skip_magic の建玉は数えない(この足で決済するもの)。"""
        total, ticks = 0.0, {}
        for sym, p, magic in self.client.account_positions():
            if skip_magic is not None and magic == skip_magic:
                continue
            if p.sl is None:
                return None
            tick = ticks.get(sym) or ticks.setdefault(sym, self.client.tick(sym))
            close = tick.bid if p.side > 0 else tick.ask
            if (close - p.sl) * p.side <= 0:
                continue
            total += -self.client.profit(sym, p.side, p.volume, close, p.sl) * self.cfg.gap_mult
        return total

    def apply_floor(self, actions: list[Action]) -> list[Action]:
        """新規(open / add)の量を、口座の床を割らない量まで減らす。止めているなら新規を外す。"""
        if self.cfg.protect <= 0 or not any(a.kind in ("open", "add") for a in actions):
            return actions
        f, equity = self.check_floor()
        closes_own = any(a.kind == "close_all" for a in actions)
        worst = self.account_worst_loss(self.cfg.magic if closes_own else None)
        room = f.room(equity, worst)
        spec = self.client.symbol(self.cfg.symbol)
        tick = self.client.tick(self.cfg.symbol)
        out = []
        for a in actions:
            if a.kind not in ("open", "add"):
                out.append(a)
                continue
            px = tick.ask if a.side > 0 else tick.bid
            if a.sl_dist:
                per_lot = -self.client.profit(self.cfg.symbol, a.side, 1.0, px, px - a.side * a.sl_dist) \
                    * self.cfg.gap_mult
            else:
                per_lot = float("inf")
            lots = floor_lots(a.lots, room, per_lot, spec.volume_step)
            if lots < spec.volume_min:
                why = ("halted for the month" if f.halted else "no stop on some position" if worst is None
                       else "no stop on the order" if not a.sl_dist else f"room {room:.0f}")
                log.warning("%s %+d %.2f lots skipped by the equity floor (%s)", a.kind, a.side, a.lots, why)
                self.journal("floor_skip", action=asdict(a), room=room, floor=f.floor, equity=equity, reason=why)
                continue
            if lots < a.lots:
                log.info("%s %+d cut from %.2f to %.2f lots by the equity floor", a.kind, a.side, a.lots, lots)
                a = Action(**{**asdict(a), "lots": lots, "reason": a.reason + " (floor)"})
            room -= lots * per_lot
            out.append(a)
        return out

    # ---- 足 ----
    def utc_offset(self) -> int:
        v = self.cfg.server_utc_offset_hours
        if v != "auto":
            return int(round(float(v) * 3600))
        off = self.client.server_offset(self.cfg.symbol)
        if off is not None:
            if off != self.state.utc_offset_sec:
                log.info("server time offset: %+.1f h", off / 3600)
            self.state.utc_offset_sec = off
            return off
        if self.state.utc_offset_sec is not None:
            return self.state.utc_offset_sec
        raise MT5Error("cannot infer the server time offset while the market is closed; "
                       "set server_utc_offset_hours in the config")

    def fetch_rates(self, forming: int):
        tf_sec = self.timeframe * 60
        cfg = self.cfg
        if self.state.anchor is None:
            count = cfg.history_bars + 1
        else:
            count = (forming - self.state.anchor) // tf_sec + 2      # 週末の分だけ多めに取り、起点より前を捨てる
        count = int(min(max(count, 2), cfg.max_history_bars))
        rates = self.client.rates(cfg.symbol, self.timeframe, count)
        first = int(rates["time"][0])
        if self.state.anchor is None or first > self.state.anchor:
            if self.state.anchor is not None:
                log.warning("history no longer reaches the anchor; moving it forward (strategy state may change)")
            self.state.anchor = first
        return rates[rates["time"] >= self.state.anchor]

    def new_bar(self) -> int | None:
        """作りかけの足が前回から変わっていれば、その時刻(サーバー時刻、秒)。"""
        forming = int(self.client.rates(self.cfg.symbol, self.timeframe, 1)["time"][-1])
        return None if forming == self.state.last_forming else forming

    def decide(self, forming: int) -> tuple[SimSnapshot, list[LivePosition], list[Action], dict]:
        spec = self.client.symbol(self.cfg.symbol)
        rates = self.fetch_rates(forming)
        bars = rates_to_bars(rates, self.timeframe, spec.point, self.utc_offset())
        pip = self.cfg.pip or pip_from_digits(spec.digits)
        snap = simulate(bars, self.make_strategy(), self.sim_broker, self.engine_cfg, self.schedule, pip)
        real = self.client.positions(self.cfg.symbol, self.cfg.magic)
        actions = plan(snap, real, self.state, lambda u: lots_for(u, spec, self.cfg.volume_scale))
        close_t = int(bars.t[-2]) + self.timeframe * NS_MIN
        meta = {"bars": len(bars), "pip": pip,
                "bar_close_utc": str(np.datetime64(close_t, "ns").astype("datetime64[s]"))}
        return snap, real, actions, meta

    def poll(self) -> bool:
        """新しい足が確定していれば判断して注文を出す。処理したら True。
        足を待つ間も floor_check_seconds ごとに口座の床を見る。"""
        if self.cfg.protect > 0 and time.time() - self._floor_checked >= self.cfg.floor_check_seconds:
            self.check_floor()
        forming = self.new_bar()
        if forming is None:
            return False
        snap, real, actions, meta = self.decide(forming)
        actions = self.apply_floor(actions)
        # 注文を出す前に「処理済み」を書く。途中で落ちても、同じ足の注文を二重に出さない
        s = self.state
        s.last_forming, s.last_bar_t = forming, snap.bar_t
        s.sim_position_t = snap.position_t
        s.sim_sl = snap.position.sl if snap.position else None
        s.sim_tp = snap.position.tp if snap.position else None
        s.updated_at = time.time()
        s.save(self.state_path)
        results = [] if self.cfg.dry_run else [self.execute(a) for a in actions]
        self.journal("bar", **meta, sim={"position": asdict(snap.position) if snap.position else None,
                                         "orders": [asdict(o) for o in snap.orders], "equity": snap.equity,
                                         "halted": snap.halted, "trades": snap.n_trades},
                     real=[asdict(p) for p in real], actions=[asdict(a) for a in actions],
                     results=[[asdict(r) for r in rs] for rs in results], dry_run=self.cfg.dry_run)
        if actions:
            log.info("bar %s: %s", meta["bar_close_utc"],
                     "; ".join(f"{a.kind} {a.side:+d} {a.lots or ''} ({a.reason})" for a in actions))
            for rs in results:
                for r in rs:
                    if not r.ok:
                        log.error("order failed: %s %s %s", r.retcode, r.comment, r.request)
        else:
            log.debug("bar %s: no action", meta["bar_close_utc"])
        return True

    # ---- 注文 ----
    def _send(self, fn: Callable[[], OrderResult]) -> OrderResult:
        r = fn()
        for k in range(3):
            if r.ok or r.retcode not in RETRY_CODES:
                break
            self.sleep(0.5 * (k + 1))
            r = fn()
        return r

    def execute(self, a: Action) -> list[OrderResult]:
        cfg, c, sym = self.cfg, self.client, self.cfg.symbol
        kw = {"magic": cfg.magic, "deviation": cfg.deviation_points, "comment": cfg.comment}
        real = c.positions(sym, cfg.magic)
        if a.kind == "close_all":
            return [self._send(lambda p=p: c.market(sym, -p.side, p.volume, position=p.ticket, **kw)) for p in real]
        if a.kind in ("open", "add"):
            held = sum(p.volume for p in real)
            if a.lots <= 0:
                log.warning("%s skipped: size below the symbol's minimum volume", a.kind)
                return []
            if held + a.lots > cfg.max_lots + 1e-9:
                log.error("%s %.2f lots skipped: would exceed max_lots %.2f (holding %.2f)",
                          a.kind, a.lots, cfg.max_lots, held)
                return [OrderResult(False, -1, "max_lots", request=asdict(a))]
        if a.kind == "open":
            tick = c.tick(sym)
            px = tick.ask if a.side > 0 else tick.bid
            sl = px - a.side * a.sl_dist if a.sl_dist else None
            tp = px + a.side * a.tp_dist if a.tp_dist else None
            return [self._send(lambda: c.market(sym, a.side, a.lots, sl=sl, tp=tp, **kw))]
        if a.kind == "add":
            r = self._send(lambda: c.market(sym, a.side, a.lots, **kw))
            out = [r]
            if not r.ok or (a.sl_dist is None and a.tp_dist is None):
                return out
            ps = c.positions(sym, cfg.magic)
            vol = sum(p.volume for p in ps)
            avg = sum(p.price_open * p.volume for p in ps) / vol
            base = avg if a.sl_from_average else (r.price or avg)
            for p in ps:
                sl = base - a.side * a.sl_dist if a.sl_dist is not None else p.sl
                tp = avg + a.side * a.tp_dist if a.tp_dist is not None else p.tp
                out.append(self._send(lambda p=p, sl=sl, tp=tp: c.modify(sym, p.ticket, sl, tp, magic=cfg.magic)))
            return out
        if a.kind == "reduce":
            left, out = a.lots, []
            step = c.symbol(sym).volume_step
            for p in sorted(real, key=lambda p: p.time):
                if left < step - 1e-9:
                    break
                v = round(min(p.volume, left), 8)
                out.append(self._send(lambda p=p, v=v: c.market(sym, -p.side, v, position=p.ticket, **kw)))
                left -= v
            return out
        if a.kind == "modify":
            tick = c.tick(sym)
            bid_ask = tick.bid if a.side > 0 else tick.ask     # 決済する側の値段
            q = bid_ask * a.side
            if (a.sl is not None and q <= a.sl * a.side) or (a.tp is not None and q >= a.tp * a.side):
                log.info("stops already crossed (sl %s tp %s at %s); closing", a.sl, a.tp, bid_ask)
                return self.execute(Action("close_all", reason="stop crossed"))
            return [self._send(lambda p=p: c.modify(sym, p.ticket, a.sl, a.tp, magic=cfg.magic)) for p in real]
        raise ValueError(f"unknown action {a.kind}")
