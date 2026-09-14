"""The former national CLI is an alias for the canonical full build."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_driver_module():
    path = (
        Path(__file__).resolve().parents[3] / "tools/calibrate_uk_national_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("calibrate_uk_national_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("selector", [[], ["--target-geographies", "country"]])
def test_old_command_delegates_without_inventing_a_target_filter(monkeypatch, selector):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "microcosm.build.uk_runtime.full_build_cli",
        SimpleNamespace(main=lambda args: calls.append(args) or 7),
    )
    arguments = [
        "--input-h5",
        "bound-spine.h5",
        "--ladder",
        "ladder.zip",
        "--ledger-facts",
        "facts.jsonl",
        *selector,
    ]
    assert _load_driver_module().main(arguments) == 7
    assert calls == [arguments]


@pytest.mark.parametrize(
    "flag",
    [
        "--staging-h5",
        "--diagnostics-json",
        "--build-record-json",
        "--terminal-gate-json",
        "--allow-unpinned-feed",
    ],
)
def test_retired_national_output_controls_explain_migration(flag):
    with pytest.raises(
        SystemExit, match="independent UK national calibration driver is retired"
    ):
        _load_driver_module().main([flag, "old-output"])


def test_old_command_cannot_invoke_a_second_solver():
    driver = _load_driver_module()
    assert not hasattr(driver, "run_uk_calibration")
    assert not hasattr(driver, "UKMeasureResolver")
    assert not hasattr(driver, "_parse_args")
