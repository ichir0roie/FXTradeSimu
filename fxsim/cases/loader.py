"""cases/behaviors と cases/tests の YAML を読み、検査する。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..paths import CASES_DIR
from .model import BehaviorCase, TestCase


@dataclass
class CaseBook:
    behaviors: dict[str, BehaviorCase]
    tests: dict[str, TestCase]


def _read_dir(d: Path) -> list[tuple[dict, Path]]:
    out = []
    for p in sorted(d.glob("*.y*ml")):
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data:
            out.append((data, p))
    return out


def load_cases(root: Path | str | None = None) -> CaseBook:
    root = Path(root or CASES_DIR)
    behaviors: dict[str, BehaviorCase] = {}
    for d, p in _read_dir(root / "behaviors"):
        b = BehaviorCase(**d, path=str(p.relative_to(root.parent)))
        if b.id in behaviors:
            raise ValueError(f"duplicate behavior id {b.id} ({p})")
        b.validate()
        behaviors[b.id] = b
    tests: dict[str, TestCase] = {}
    for d, p in _read_dir(root / "tests"):
        t = TestCase(**d, path=str(p.relative_to(root.parent)))
        if t.id in tests:
            raise ValueError(f"duplicate test id {t.id} ({p})")
        t.validate(behaviors)
        tests[t.id] = t
    return CaseBook(behaviors, tests)
