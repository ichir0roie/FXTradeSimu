"""市場モデルの骨組み。

市場モデル = 基本パラメータ + 部品(Component)の並び。
部品は段階(Stage)ごとに順番に呼ばれ、共有の GenContext の配列を書き換える。

    SOURCE      (任意)時刻列と基準を外から与える。実データの再生・骨格など
    VOLATILITY  ボラティリティ倍率・スケジュールされたイベント
    DRIFT       ドリフト(レジームなど)
    INNOVATION  ショックの分布(正規 / t 分布)
    RETURNS     ジャンプ(窓開け・指標)
    --- ここで log 価格の経路を組み立てる ---
    POST_PATH   経路を見て反応する悪意(ストップ狩り・ダマシ・逆張り圧力)
    ANCHOR      経路を実データの骨格(日足の終値・高値・安値)に合わせる
    SPREAD      スプレッド
    FINAL       最後の調整

新しい案を試すときは、部品を 1 つ書き足すか、既存部品の組み合わせ・パラメータを変えた
モデルを 1 ファイル追加する。部品は params で個別に上書き・無効化できる(アブレーション用)。
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import IntEnum
from typing import Any, ClassVar

import numpy as np

from ..calendar import hour_of_day, steps_per_year, trading_times
from ..data import MarketData
from ..registry import params_hash


class Stage(IntEnum):
    SOURCE = 5
    VOLATILITY = 10
    DRIFT = 20
    INNOVATION = 30
    RETURNS = 40
    POST_PATH = 60
    ANCHOR = 65
    SPREAD = 70
    FINAL = 90


@dataclass
class BaseParams:
    p0: float = 110.0             # 初値(USDJPY 風)
    pip: float = 0.01
    annual_vol: float = 0.09      # 年率ボラ(部品が倍率を掛ける前の基準)
    step_minutes: int = 1
    start: str = "2015-01-05"


@dataclass
class GenContext:
    rng: np.random.Generator
    times: np.ndarray
    base: BaseParams
    vol: np.ndarray                     # 1 ステップの log リターンの標準偏差
    drift: np.ndarray                   # 1 ステップの log ドリフト
    jumps: np.ndarray                   # 1 ステップに加える log ジャンプ
    shocks: np.ndarray | None = None    # 標準化ショック(未設定なら正規乱数)
    logp: np.ndarray | None = None      # 経路(POST_PATH 以降で有効)
    spread: np.ndarray | None = None    # 価格単位
    events: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.times)

    @property
    def steps_per_year(self) -> float:
        return steps_per_year(self.base.step_minutes)

    @property
    def hours(self) -> np.ndarray:
        if "_hours" not in self.meta:
            self.meta["_hours"] = hour_of_day(self.times)
        return self.meta["_hours"]

    def pips_to_log(self, pips: float | np.ndarray) -> float | np.ndarray:
        """pip 数を初値基準の log 変化に直す。"""
        return np.asarray(pips) * self.base.pip / self.base.p0

    def log_to_pips(self, x):
        return np.asarray(x) * self.base.p0 / self.base.pip

    def event(self, idx: int, kind: str, **detail) -> None:
        self.events.append({"idx": int(idx), "kind": kind, **detail})


class Component:
    """市場モデルの部品。サブクラスは key / stage / Params / apply を定義する。"""

    key: ClassVar[str] = "component"
    stage: ClassVar[Stage] = Stage.FINAL
    deterministic: ClassVar[bool] = False   # SOURCE 部品で True なら、シードで結果が変わらない(実データの再生)

    @dataclass
    class Params:
        pass

    def __init__(self, **params: Any):
        self.p = self.Params(**params)

    def apply(self, ctx: GenContext) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    # SOURCE 段の部品だけが実装する: 時刻列と(必要なら書き換えた)基本パラメータを返す
    def prepare_times(self, base: BaseParams, years: float | None, days: float | None,
                      rng: np.random.Generator) -> tuple[np.ndarray, BaseParams]:  # pragma: no cover
        raise NotImplementedError

    def params_dict(self) -> dict:
        return asdict(self.p) if is_dataclass(self.p) else {}

    def with_overrides(self, overrides: dict) -> Component:
        names = {f.name for f in fields(self.p)}
        unknown = set(overrides) - names
        if unknown:
            raise KeyError(f"{self.key}: unknown params {sorted(unknown)} (known: {sorted(names)})")
        merged = {**self.params_dict(), **overrides}
        return type(self)(**merged)


class MarketModel:
    def __init__(self, name: str, version: str, components: list[Component], base: BaseParams | None = None,
                 description: str = ""):
        keys = [c.key for c in components]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate component keys in {name}: {keys}")
        self.name = name
        self.version = version
        self.components = components
        self.base = base or BaseParams()
        self.description = description

    # ---- 構成 -------------------------------------------------------------
    def with_params(self, params: dict | None) -> MarketModel:
        """{"base": {...}, "<部品key>": {...} | None | False} で上書きした複製を返す。

        None / False / {"enabled": False} はその部品を外す(アブレーション)。
        """
        if not params:
            return self
        params = copy.deepcopy(params)
        base = self.base
        if "base" in params:
            base = BaseParams(**{**asdict(self.base), **params.pop("base")})
        by_key = {c.key: c for c in self.components}
        unknown = set(params) - set(by_key)
        if unknown:
            raise KeyError(f"{self.name}: unknown components {sorted(unknown)} (known: {sorted(by_key)})")
        comps = []
        for c in self.components:
            if c.key not in params:
                comps.append(c)
                continue
            ov = params[c.key]
            if ov is None or ov is False or (isinstance(ov, dict) and ov.get("enabled") is False):
                continue
            comps.append(c.with_overrides({k: v for k, v in ov.items() if k != "enabled"}))
        return MarketModel(self.name, self.version, comps, base, self.description)

    def info(self) -> dict:
        params = {"base": asdict(self.base), **{c.key: c.params_dict() for c in self.components}}
        return {
            "name": self.name,
            "version": self.version,
            "components": [c.key for c in self.components],
            "params": params,
            "params_hash": params_hash(params),
        }

    @property
    def source(self) -> Component | None:
        return next((c for c in self.components if c.stage == Stage.SOURCE), None)

    @property
    def deterministic(self) -> bool:
        """シードで結果が変わらないモデル(実データの再生)。実験ではシードを 1 つだけ回す。"""
        src = self.source
        return bool(src is not None and src.deterministic)

    # ---- 生成 -------------------------------------------------------------
    def generate(self, *, years: float | None = None, days: float | None = None, seed: int = 0) -> MarketData:
        rng = np.random.default_rng(seed)
        b = self.base
        src = self.source
        if src is not None:
            times, b = src.prepare_times(b, years, days, rng)
        else:
            times = trading_times(b.start, days=days, years=years, step_minutes=b.step_minutes)
        n = len(times)
        sigma = b.annual_vol / np.sqrt(steps_per_year(b.step_minutes))
        ctx = GenContext(
            rng=rng,
            times=times,
            base=b,
            vol=np.full(n, sigma),
            drift=np.zeros(n),
            jumps=np.zeros(n),
        )
        for comp in sorted(self.components, key=lambda c: int(c.stage)):  # sorted は安定
            if comp.stage >= Stage.POST_PATH and ctx.logp is None:
                _build_path(ctx)
            comp.apply(ctx)
        if ctx.logp is None:
            _build_path(ctx)
        if ctx.spread is None:
            ctx.spread = np.full(n, 0.3 * b.pip)
        ctx.events.sort(key=lambda e: e["idx"])
        meta = {k: v for k, v in ctx.meta.items() if not k.startswith("_")}
        info = self.info()
        info["base_effective"] = asdict(b)
        return MarketData(
            times=times,
            mid=np.exp(ctx.logp),
            spread=ctx.spread,
            pip=b.pip,
            step_minutes=b.step_minutes,
            events=ctx.events,
            meta=meta,
            model_info=info,
        )


def _build_path(ctx: GenContext) -> None:
    if ctx.shocks is None:
        ctx.shocks = ctx.rng.standard_normal(ctx.n)
    r = ctx.drift + ctx.vol * ctx.shocks + ctx.jumps
    ctx.logp = np.log(ctx.base.p0) + np.cumsum(r)
