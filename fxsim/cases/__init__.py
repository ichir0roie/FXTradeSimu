"""動作ケースとテストケース(書式は docs/CASES.md)。

    model.py   BehaviorCase / TestCase / Pattern
    loader.py  cases/behaviors と cases/tests の YAML を読んで検査する
    runner.py  テストケースを実験に直して回し、case_defs / case_runs に記録する

``python -m fxsim cases run --all`` で全テストケースを回す。定義とコードの版が前回と同じなら飛ばす。
"""
from .loader import CaseBook, load_cases
from .model import BehaviorCase, Pattern, TestCase
from .runner import run_cases, summarize_run, to_experiment

__all__ = ["BehaviorCase", "CaseBook", "Pattern", "TestCase", "load_cases", "run_cases", "summarize_run",
           "to_experiment"]
