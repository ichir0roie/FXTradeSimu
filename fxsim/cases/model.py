"""動作ケースとテストケースの型。書式は docs/CASES.md。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..backtest.broker import BROKER_PRESETS
from ..backtest.schedule import TradingSchedule
from ..markets import MARKETS
from ..strategies import STRATEGIES

# fxsim verify の既定(動作ケースの verify で上書きできる)。E009 と同じ物差し
VERIFY_DEFAULTS = {"years": 30.0, "seeds": "1-10", "block_days": 250.0}


def parse_seeds(seeds) -> list[int]:
    """[1, 2] / "1-10" / 3 のどれでも受ける。"""
    if isinstance(seeds, str):
        a, b = seeds.split("-")
        return list(range(int(a), int(b) + 1))
    if isinstance(seeds, int):
        return [seeds]
    return [int(s) for s in seeds]


@dataclass
class BehaviorCase:
    """どう取引するか: 手法・判断に使う足・判断の間隔・取引時間・ブローカー・数量。"""
    id: str
    strategy: str
    name: str = ""
    params: dict = field(default_factory=dict)
    timeframe: int | None = None
    schedule: dict = field(default_factory=dict)
    broker: str | dict = "fair"
    engine: dict = field(default_factory=dict)
    # fxsim verify の既定(years / seeds / block_days)。コマンドで指定すればそちらが優先
    verify: dict = field(default_factory=dict)
    path: str = ""

    @property
    def broker_spec(self) -> dict:
        b = self.broker if isinstance(self.broker, dict) else {"preset": self.broker}
        return {"preset": b["preset"], "label": b.get("label", b["preset"]), "overrides": b.get("overrides", {})}

    def strategy_params(self) -> dict:
        p = dict(self.params)
        if self.timeframe:
            p["timeframe"] = int(self.timeframe)
        return p

    def trading_schedule(self) -> TradingSchedule:
        return TradingSchedule.from_dict(self.schedule)

    def is_multi(self) -> bool:
        """複数ペアの戦略か(相場は basket_*、帰無は basket_direction_null)。"""
        return STRATEGIES.create(self.strategy, self.strategy_params()).multi

    def pair_market(self) -> str | None:
        """複数ペアの戦略が回る相場の種類(real_pairs なら主要 10 ペアの口座。None なら basket_*)。"""
        return getattr(STRATEGIES.create(self.strategy, self.strategy_params()), "pair_market", None)

    def null_strategy(self) -> str:
        """検証の帰無(方向でたらめ版)の戦略の名前。複数ペアは basket_、本命(MT5 のテスター)は main_ の版。"""
        s = STRATEGIES.create(self.strategy, self.strategy_params())
        if getattr(s, "engine", "classic") == "mt5":
            return "main_direction_null"
        return "basket_direction_null" if s.multi else "direction_null"

    def verify_null(self, engine: dict | None = None) -> str:
        """検証(verify・pair-eval)の帰無の戦略の名前。複数ペアの戦略を swap_side_only の口座で回すときは、
        向きに自由が無いので時期だけでたらめ版(basket_timing_null、E032)。それ以外は null_strategy()。"""
        eng = {**self.engine, **(engine or {})}
        if eng.get("swap_side_only") and self.is_multi() and self.null_strategy() == "basket_direction_null":
            return "basket_timing_null"
        return self.null_strategy()

    def verify_settings(self, **given) -> dict:
        """検証の設定。コマンドで指定した値(None 以外)> 動作ケースの verify > 既定。"""
        out = {**VERIFY_DEFAULTS, **self.verify}
        out["seeds"] = parse_seeds(out["seeds"])
        out.update({k: v for k, v in given.items() if v is not None})
        return out

    def validate(self) -> None:
        unknown = set(self.verify) - set(VERIFY_DEFAULTS)
        if unknown:
            raise ValueError(f"{self.id}: unknown verify keys {sorted(unknown)} (known: {sorted(VERIFY_DEFAULTS)})")
        STRATEGIES.create(self.strategy, self.strategy_params())
        self.trading_schedule()
        if self.broker_spec["preset"] not in BROKER_PRESETS:
            raise ValueError(f"{self.id}: unknown broker preset {self.broker_spec['preset']}")

    def definition(self) -> dict:
        """記録に残す定義(ハッシュの元)。"""
        return {"id": self.id, "name": self.name, "strategy": self.strategy, "params": self.params,
                "timeframe": self.timeframe, "schedule": self.trading_schedule().to_dict(),
                "broker": self.broker, "engine": self.engine, "verify": self.verify}


@dataclass
class Pattern:
    """相場パターン: 市場モデル(とパラメータ)とシードの組。"""
    market: str
    label: str | None = None
    params: dict = field(default_factory=dict)
    seeds: list[int] = field(default_factory=lambda: [1])

    @classmethod
    def parse(cls, x: dict) -> Pattern:
        return cls(x["market"], x.get("label"), x.get("params") or {}, parse_seeds(x.get("seeds", [1])))

    @property
    def display_label(self) -> str:
        return self.label or self.market


@dataclass
class TestCase:
    """ある動作ケースを、どの相場パターンで、どれだけの期間試すか。"""
    __test__ = False   # pytest に集めさせない

    id: str
    behavior: str
    patterns: list[Pattern]
    name: str = ""
    period: dict = field(default_factory=lambda: {"years": 3})
    detail_seeds: list[int] | None = None
    workers: int = 0
    verify: bool = False      # True: 方向でたらめ版(direction_null)を対で回し、合否を判定して記録する
    null: bool = False        # True: 方向でたらめ版を対で回すだけ(判定は呼んだ側。fxsim pair-eval)
    broker: dict | None = None  # 動作ケースのブローカーの差し替え(fxsim pair-eval --broker)
    engine: dict | None = None  # 動作ケースのエンジンの設定への上書き(fxsim pair-eval --swap-side)
    path: str = ""

    def __post_init__(self):
        self.patterns = [p if isinstance(p, Pattern) else Pattern.parse(p) for p in self.patterns]

    def validate(self, behaviors: dict[str, BehaviorCase]) -> None:
        if self.behavior not in behaviors:
            raise ValueError(f"{self.id}: unknown behavior {self.behavior}")
        for p in self.patterns:
            MARKETS.get(p.market)
        labels = [p.display_label for p in self.patterns]
        if len(labels) != len(set(labels)):
            raise ValueError(f"{self.id}: pattern labels must be unique (set label): {labels}")
        if "years" not in self.period and not ("start" in self.period and "end" in self.period):
            raise ValueError(f"{self.id}: period needs years or start/end")

    def years(self) -> float:
        if "years" in self.period:
            return float(self.period["years"])
        a = date.fromisoformat(str(self.period["start"]))
        b = date.fromisoformat(str(self.period["end"]))
        return (b - a).days / 365.25

    def period_label(self) -> str:
        if "years" in self.period:
            return f"{self.period['years']} 年"
        return f"{self.period['start']}〜{self.period['end']}"

    def definition(self) -> dict:
        d = {"id": self.id, "name": self.name, "behavior": self.behavior, "period": self.period,
             "patterns": [p.__dict__ for p in self.patterns], "detail_seeds": self.detail_seeds,
             "verify": self.verify}
        if self.null:
            d["null"] = True
        if self.broker:
            d["broker"] = self.broker
        if self.engine:
            d["engine"] = self.engine
        return d
