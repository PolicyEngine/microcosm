# ruff: noqa: F401
from __future__ import annotations

import copy
import hashlib
import json
from importlib import metadata
from importlib.resources import files
from pathlib import Path

import pytest

from microcosm.build.uk_runtime.take_up_contract import (
    load_uk_take_up_contract,
    uk_take_up_contract_identity,
)


def _resource() -> dict:
    return json.loads(
        files("microcosm.build.uk").joinpath("take_up_contract.json").read_text()
    )


def _reload_with(monkeypatch, mutated: dict) -> None:
    load_uk_take_up_contract.cache_clear()
    payload = json.dumps(mutated)

    class _FakePath:
        def read_text(self, *args, **kwargs):
            return payload

    monkeypatch.setattr(
        "microcosm.build.uk_runtime.take_up_contract._contract_path",
        lambda: _FakePath(),
    )


__all__ = [name for name in globals() if not name.startswith("__")]
