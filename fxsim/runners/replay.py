"""再生のランナー: Mt5Runner の端末を、相場の DB の 1 分足を再生する偽の端末に替える(fxsim trade-replay)。"""
from __future__ import annotations

from pathlib import Path

from .mt5 import Mt5Runner


class ReplayRunner(Mt5Runner):
    """設定(configs/trading/*.yaml)の常駐を、1 分足の再生の上で 24 時間動かす。複数の設定は同じ口座で一緒に動く
    (Mt5Runner と同じ決まりで、端末と時計だけを替える。故障も入れられる)。"""

    def __init__(self, configs: list[str | Path], opts=None, market_db: str | Path | None = None):
        from ..trading.replay import ReplayOptions
        paths = [str(c) for c in configs]
        if not paths:
            raise ValueError("give at least one trading config")
        super().__init__(paths[0])
        self.paths = paths
        self.opts = opts or ReplayOptions()
        self.market_db = market_db

    def run(self) -> dict:
        from ..trading.replay import run_replay
        return run_replay(self.paths, self.opts, market_db=self.market_db)


__all__ = ["ReplayRunner"]
