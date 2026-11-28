"""The local UK firm build writes a valid table and a reproducible receipt.

A thinned 2024-25 build on the paper-parity fixtures, with a test-only rules
double, runs twice into different directories: the parquet round-trips, the
receipt's hashes match the files, and the two builds agree exactly without a
committed golden hash. The tool is loaded from its path, refuses to run
without ``--rules`` and writes the same receipt as the library call.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_build import (
    UK_FIRM_RECEIPT_FILE,
    UK_FIRM_TABLE_FILE,
    build_uk_firm_dataset,
)
from microcosm.build.uk_runtime.firm_generation import UKFirmGenerationConfig
from microcosm.build.uk_runtime.firm_rules import (
    UK_FIRM_RULE_PERIOD_NOTE,
    UKFirmRuleOutputs,
    rule_period_for_vintage,
)
from microcosm.build.uk_runtime.firm_table import (
    UK_FIRM_TABLE_COLUMNS,
    validate_uk_firm_table,
)
from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
FIXTURES = PATHS.tests / "fixtures" / "uk"
PROCESSED = FIXTURES / "firm_2024_25"
TOOL = PATHS.repository / "tools" / "build_uk_firms.py"
SEED = 42
ITERATIONS = 20
WINDOW_FRACTION = 0.02
TAIL_FRACTION = 0.005


class _StandardRateDouble:
    """Test-only rules double, 0.20 x (turnover - inputs); not UK VAT law."""

    def evaluate(self, firms: pd.DataFrame, *, period: int) -> UKFirmRuleOutputs:
        del period
        liability = 0.20 * (
            firms["firm_turnover"].to_numpy(dtype=np.float64)
            - firms["firm_input_costs"].to_numpy(dtype=np.float64)
        )
        return UKFirmRuleOutputs(
            firm_id=firms["firm_id"].to_numpy(dtype=np.int64),
            vat_liability_if_registered=liability,
            provenance={"rules": "standard-rate test double"},
        )


def _config(rules: _StandardRateDouble) -> UKFirmGenerationConfig:
    return UKFirmGenerationConfig(
        seed=SEED,
        n_iterations=ITERATIONS,
        sample_window_fraction=WINDOW_FRACTION,
        sample_tail_fraction=TAIL_FRACTION,
        firm_rules=rules,
    ).with_vintage("2024-25")


def _build(output_dir: Path) -> dict:
    rules = _StandardRateDouble()
    return build_uk_firm_dataset(
        processed_dir=PROCESSED,
        output_dir=output_dir,
        rules=rules,
        config=_config(rules),
    )


@pytest.fixture(scope="module")
def builds(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, dict]:
    return (
        _build(tmp_path_factory.mktemp("first") / "out"),
        _build(tmp_path_factory.mktemp("second") / "out"),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@cache
def _load_tool():
    # Registered before it runs, as the stock cell tool's test does.
    if "build_uk_firms" in sys.modules:
        return sys.modules["build_uk_firms"]
    spec = importlib.util.spec_from_file_location("build_uk_firms", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parquet_round_trips_and_validates(builds) -> None:
    built, _ = builds
    table_path = built["table_path"]

    assert table_path.name == UK_FIRM_TABLE_FILE
    written = pd.read_parquet(table_path)
    pd.testing.assert_frame_equal(written, built["table"])
    validate_uk_firm_table(written)
    receipt = built["receipt"]
    assert receipt["output"]["rows"] == len(written)
    assert receipt["output"]["columns"] == list(written.columns)
    assert receipt["output"]["weighted_firms"] == pytest.approx(
        float(written["firm_weight"].sum())
    )


def test_receipt_hashes_match_the_files(builds) -> None:
    built, _ = builds
    receipt = json.loads(built["receipt_path"].read_text(encoding="utf-8"))

    assert built["receipt_path"].name == UK_FIRM_RECEIPT_FILE
    assert receipt == built["receipt"]
    assert receipt["output"]["sha256"] == _sha256(built["table_path"])
    for role, directory in (("processed", PROCESSED),):
        expected = {path.name: _sha256(path) for path in directory.glob("*.csv")}
        assert expected
        assert receipt["inputs"][role] == expected


def test_receipt_records_the_run(builds) -> None:
    built, _ = builds
    receipt = built["receipt"]

    assert receipt["schema_version"] == 1
    assert receipt["data_vintage"] == "2024-25"
    assert receipt["seed"] == SEED
    assert receipt["rule_period"] == rule_period_for_vintage("2024-25")
    assert receipt["rule_period_note"] == UK_FIRM_RULE_PERIOD_NOTE
    assert receipt["data_years"]["ons_frame_turnover"] == "2023-24"
    assert receipt["data_years"]["hmrc_vat_statistics"] == "2024-25"
    assert receipt["config"]["n_iterations"] == ITERATIONS
    assert "firm_rules" not in receipt["config"]
    assert "vat_rule_evaluator" not in receipt["config"]
    assert receipt["rules_provenance"] == [{"rules": "standard-rate test double"}]
    assert 0.0 < receipt["validation"]["overall"] <= 1.0
    assert receipt["target_diagnostics"]
    assert receipt["output"]["columns"] == list(UK_FIRM_TABLE_COLUMNS)
    assert set(receipt["inputs"]) == {"processed"}


def _comparable(receipt: dict) -> dict:
    """A receipt without the parquet hash, floats rounded to 9 significant digits.

    Same-seed builds agree exactly on the machines tested so far; the
    comparison tolerates last-digit float differences so a platform's
    floating-point reductions cannot fail it.
    """

    def _round(value):
        if isinstance(value, float):
            return float(f"{value:.9g}")
        if isinstance(value, dict):
            return {key: _round(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_round(item) for item in value]
        return value

    trimmed = json.loads(json.dumps(receipt))
    trimmed["output"].pop("sha256")
    return _round(trimmed)


def test_two_builds_agree(builds) -> None:
    first, second = builds

    pd.testing.assert_frame_equal(first["table"], second["table"])
    assert _comparable(first["receipt"]) == _comparable(second["receipt"])


def test_receipt_holds_no_absolute_path(builds, tmp_path_factory) -> None:
    for built in builds:
        text = built["receipt_path"].read_text(encoding="utf-8")
        assert str(tmp_path_factory.getbasetemp()) not in text
        assert str(built["table_path"].parent) not in text
        assert str(PATHS.repository) not in text


def test_tool_needs_rules(tmp_path: Path) -> None:
    tool = _load_tool()
    with pytest.raises(SystemExit) as raised:
        tool.main(
            [
                "--processed-dir",
                str(PROCESSED),
                "--out-dir",
                str(tmp_path),
            ]
        )
    assert raised.value.code != 0
    assert not any(tmp_path.iterdir())


def test_tool_needs_an_artifact_for_axiom(tmp_path: Path) -> None:
    tool = _load_tool()
    with pytest.raises(SystemExit) as raised:
        tool.main(
            [
                "--processed-dir",
                str(PROCESSED),
                "--out-dir",
                str(tmp_path),
                "--rules",
                "axiom",
            ]
        )
    assert raised.value.code != 0


def test_tool_writes_the_library_build(
    builds, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The tool's flags reach the build: thinned to the test's fractions
    after ``--fast`` is seen, it writes the library build's receipt."""

    tool = _load_tool()
    real_config = tool._config
    seen: list[UKFirmGenerationConfig] = []

    def thinned_config(args, rules):
        config = real_config(args, rules)
        seen.append(config)
        return dataclasses.replace(
            config,
            sample_window_fraction=WINDOW_FRACTION,
            sample_tail_fraction=TAIL_FRACTION,
        )

    monkeypatch.setattr(tool, "_rules", lambda args: _StandardRateDouble())
    monkeypatch.setattr(tool, "_config", thinned_config)
    out = tmp_path / "tool"
    code = tool.main(
        [
            "--processed-dir",
            str(PROCESSED),
            "--out-dir",
            str(out),
            "--rules",
            "policyengine-uk",
            "--fast",
            "--seed",
            str(SEED),
            "--n-iterations",
            str(ITERATIONS),
        ]
    )

    assert code == 0
    assert str(out / UK_FIRM_TABLE_FILE) in capsys.readouterr().out
    (config,) = seen
    fast = UKFirmGenerationConfig.fast()
    assert config.sample_window_fraction == fast.sample_window_fraction
    assert config.sample_tail_fraction == fast.sample_tail_fraction
    assert config.n_iterations == ITERATIONS
    assert config.data_vintage == "2024-25"
    receipt = json.loads((out / UK_FIRM_RECEIPT_FILE).read_text(encoding="utf-8"))
    assert _comparable(receipt) == _comparable(builds[0]["receipt"])
