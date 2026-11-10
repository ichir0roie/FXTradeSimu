"""名前 -> ファクトリの登録簿。市場モデルと戦略で共用する。

案(モデル/戦略)を増やすときは、新しいファイルで ``@REGISTRY.register(...)`` するだけでよい。
登録した名前は YAML の実験設定や CLI からそのまま指定できる。
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Entry:
    name: str
    version: str
    description: str
    factory: Callable[..., Any]
    tags: tuple[str, ...] = field(default_factory=tuple)


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, Entry] = {}

    def register(self, name: str, *, version: str = "1", description: str = "", tags: tuple[str, ...] = ()):
        def deco(factory):
            if name in self._entries:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._entries[name] = Entry(name, version, description, factory, tuple(tags))
            return factory

        return deco

    def get(self, name: str) -> Entry:
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(f"unknown {self.kind} '{name}'. known: {sorted(self._entries)}") from None

    def create(self, name: str, params: dict | None = None):
        """クラスなら factory(**params)、関数なら factory(params) で作る。"""
        f = self.get(name).factory
        return f(**(params or {})) if isinstance(f, type) else f(params or {})

    def names(self) -> list[str]:
        return sorted(self._entries)

    def entries(self) -> list[Entry]:
        return [self._entries[k] for k in self.names()]


def params_hash(obj: Any) -> str:
    """パラメータ辞書の短いハッシュ。同じ設定の結果を束ねるのに使う。"""
    blob = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]
