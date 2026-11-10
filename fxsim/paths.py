"""リポジトリ内の決まった場所。ほかのモジュールは場所をここからだけ取る。"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
REAL_DATA_DIR = DATA_DIR / "real"
DEFAULT_DB = DATA_DIR / "fxsim.sqlite"
CASES_DIR = REPO_ROOT / "cases"
CONFIGS_DIR = REPO_ROOT / "configs"


def resolve(path: str | Path) -> Path:
    """リポジトリ直下からの相対パスを絶対パスにする(絶対パスはそのまま)。"""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
