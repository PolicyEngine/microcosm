"""Evidence contract for irreducible Early Head Start source unavailability."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
import inspect
import json
import os
from hashlib import sha256
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import h5py
import pandas as pd
import pytest

from microcosm.build.us_runtime.asec_pool import load_asec_h5_tables
from microcosm.build.us_runtime.release_input_coverage import (
    load_release_input_coverage_manifest,
)
from microcosm.build.us_runtime.sipp_head_start import (
    SIPP_2023_HEAD_START_DONOR_REVISION,
    SIPP_2023_HEAD_START_DONOR_SHA256,
    SIPP_2023_HEAD_START_DONOR_SIZE_BYTES,
)
from microcosm.build.us_runtime.take_up_contract import load_take_up_contract
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

ROOT = _TEST_PATHS.repository
_RUN_LARGE_SOURCE_AUDITS = os.environ.get("POPULACE_RUN_LARGE_SOURCE_AUDITS") == "1"


def _entry() -> dict[str, object]:
    payload = json.loads(
        files("microcosm.build.us").joinpath("ecps_parity_known_gaps.json").read_text()
    )
    return payload["known_gaps"]["takes_up_early_head_start_if_eligible"]


def _build_summary() -> dict[str, object]:
    return json.loads(
        (ROOT / "experiments/build_j_recert/base_j.summary.json").read_text()
    )


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_recorded_artifact(recorded_path: str, filename: str) -> Path | None:
    """Resolve a Build-J input without silently substituting another artifact."""

    candidates = (
        Path(recorded_path).expanduser(),
        Path.home()
        / "PolicyEngine"
        / ("policyengine-" + "us-data")
        / ("policyengine_" + "us_data")
        / "storage"
        / filename,
    )
    return next((path for path in candidates if path.is_file()), None)


def _sipp_snapshot() -> Path:
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / ("models--policyengine--policyengine-" + "us-data")
        / "snapshots"
        / SIPP_2023_HEAD_START_DONOR_REVISION
        / "pu2023.csv"
    )


__all__ = [name for name in globals() if not name.startswith("__")]
