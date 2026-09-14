"""The older geography-only builder delegates to the single full build."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _driver():
    path = Path(__file__).resolve().parents[3] / "tools/build_uk_rowwise_dataset.py"
    spec = importlib.util.spec_from_file_location("build_uk_rowwise_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("selector", [[], ["--target-geographies", "country"]])
def test_geography_command_preserves_explicit_scope_only(monkeypatch, selector):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "microcosm.build.uk_runtime.full_build_cli",
        SimpleNamespace(main=lambda arguments: calls.append(arguments) or 0),
    )
    arguments = [
        "--input-h5",
        "bound-spine.h5",
        "--ladder",
        "ladder.npz",
        "--ledger-facts",
        "facts.jsonl",
        "--out",
        "out",
        "--n-clones",
        "2",
        *selector,
    ]
    assert _driver().main(arguments) == 0
    assert calls == [arguments]


@pytest.mark.parametrize(
    "flag",
    [
        "--crosswalk",
        "--candidate-clone-counts",
        "--allow-cross-region-assignment",
        "--dataset-filename",
    ],
)
def test_removed_geography_workflow_options_explain_migration(flag):
    with pytest.raises(
        SystemExit, match="independent UK geography-only build driver is retired"
    ):
        _driver().main([flag, "old-value"])


def test_command_contains_no_independent_cloning_export_path():
    driver = _driver()
    assert not hasattr(driver, "_main_impl")
    assert not hasattr(driver, "clone_uk_dataset_with_ladder_geography")
    assert not hasattr(driver, "_load_or_build_crosswalk")
