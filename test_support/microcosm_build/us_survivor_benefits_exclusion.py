"""Evidence contract for the irreducible survivor-benefits exclusion."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.us_runtime.release_input_coverage import (
    load_release_input_coverage_manifest,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

ROOT = _TEST_PATHS.repository


def _entry() -> dict[str, object]:
    payload = json.loads(
        files("microcosm.build.us").joinpath("ecps_parity_known_gaps.json").read_text()
    )
    return payload["known_gaps"]["survivor_benefits"]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [name for name in globals() if not name.startswith("__")]
