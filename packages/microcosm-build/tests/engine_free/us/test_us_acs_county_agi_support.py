"""The pre-R2 check for counties whose SOI AGI row would be empty (microcosm#922).

``tools/check_us_acs_county_agi_support.py`` lists counties that have
households but none with a nonzero AGI: R2's ``attach_surface`` refuses such
a county after the engine pass. These tests stub the engine pass (the
household AGI), so no country engine is imported; the engine-tier test
``engine_workflow/us/test_us_acs_county_agi_support_engine.py`` runs it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


@pytest.fixture(scope="module")
def check():
    path = _TEST_PATHS.repository / "tools" / "check_us_acs_county_agi_support.py"
    spec = importlib.util.spec_from_file_location("check_us_county_agi", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frame(person_columns: dict[str, list] | None = None) -> Frame:
    """Six households in three counties of state 01, one person each.

    01001 and 01003 hold two donor households each; 01005 holds two ACS
    households, the second a group-quarters record.
    """

    n = 6
    ids = np.arange(1, n + 1)
    person = pd.DataFrame(
        {
            "person_id": ids,
            **{f"person_{entity}_id": ids for entity in US_SCHEMA.group_entities},
            "age": np.full(n, 40.0),
            **(person_columns or {}),
        }
    )
    tables = {"person": person}
    for entity in US_SCHEMA.group_entities:
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    household = tables["household"]
    household["state_fips"] = np.full(n, 1)
    household["county_fips"] = [1001, 1001, 1003, 1003, 1005, 1005]
    household["county_fips_2023"] = household["county_fips"]
    household["household_spine"] = pd.Series(
        ["asec_puf"] * 4 + ["acs_2024_1yr"] * 2, dtype=object
    )
    household["TYPEHUGQ"] = [np.nan] * 4 + [1.0, 3.0]
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(n), WeightKind.CALIBRATED)},
    )


def test_report_lists_counties_with_households_but_no_nonzero_agi(check) -> None:
    households = _frame().table("household")
    nonzero = np.asarray([True, True, False, False, False, True])

    report = check.county_agi_report(
        households,
        nonzero,
        county_column="county_fips_2023",
        county_populations={1001: 10.0, 1003: 5.0, 1005: 5.0, 1007: 3.0},
    )

    assert report["at_risk"] == ["01003"]
    assert report["at_risk_detail"]["01003"] == {
        "households": 2,
        "supporting_households": 2,
        "households_with_agi_income": 0,
    }
    # 1007 has ladder population but no household: minimum support's case.
    assert report["no_household"] == ["01007"]
    # A group-quarters record counts for AGI but not for support.
    assert report["by_county"]["01005"]["supporting_households"] == 1
    assert report["fewer_than_min_support_with_agi_income"] == ["01001", "01005"]
    assert report["counties_outside_ladder"] == []


_NEVER_DEFAULT = frozenset(
    {
        ("person", "takes_up_ssi_if_eligible"),
        ("person", "is_blind"),
        ("spm_unit", "takes_up_snap_if_eligible"),
    }
)


def test_null_runtime_columns_outside_the_agi_are_held_back(check) -> None:
    frame = _frame(
        {
            "takes_up_ssi_if_eligible": [True, np.nan, True, True, False, True],
            "is_blind": [False] * 6,
        }
    )

    held_frame, held = check.hold_back_null_runtime_columns(
        frame, never_default_filled=_NEVER_DEFAULT, agi_inputs={"is_blind", "age"}
    )

    assert held == {"person": ["takes_up_ssi_if_eligible"]}
    assert "takes_up_ssi_if_eligible" not in held_frame.table("person").columns
    assert "is_blind" in held_frame.table("person").columns
    assert held_frame.n("household") == frame.n("household")


def test_a_null_never_default_agi_input_is_refused(check) -> None:
    frame = _frame({"is_blind": [False, np.nan, False, False, False, False]})

    with pytest.raises(SystemExit, match=r"person\.is_blind"):
        check.hold_back_null_runtime_columns(
            frame, never_default_filled=_NEVER_DEFAULT, agi_inputs={"is_blind"}
        )


def test_a_frame_without_null_runtime_columns_is_returned_as_is(check) -> None:
    frame = _frame({"takes_up_ssi_if_eligible": [True] * 6})

    held_frame, held = check.hold_back_null_runtime_columns(
        frame, never_default_filled=_NEVER_DEFAULT, agi_inputs={"age"}
    )

    assert held == {}
    assert held_frame is frame


@dataclass(frozen=True)
class _Closure:
    input_leaves: tuple[str, ...]


def test_agi_engine_inputs_are_the_static_closure_of_agi(check) -> None:
    class _Index:
        def variable_dependency_closure(self, name):
            assert name == "adjusted_gross_income"
            return _Closure(("age", "employment_income_before_lsr"))

    assert check.agi_engine_inputs(_Index()) == {"age", "employment_income_before_lsr"}


def _stub_release(frame: Frame) -> SimpleNamespace:
    return SimpleNamespace(_load_staging_frame=lambda path: frame)


def test_command_exits_2_and_reports_an_at_risk_county(
    check, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging.h5"
    staging.write_bytes(b"")
    (tmp_path / "staging.summary.json").write_text("{}")
    out = tmp_path / "report.json"
    monkeypatch.setattr(check, "_release_tool", lambda: _stub_release(_frame()))
    calls = []

    def agi(frame, summary_path, *, release, hh_chunk):
        calls.append((summary_path, hh_chunk))
        # 01003's households have an AGI of 0; a negative AGI is stored.
        return np.asarray([100.0, 0.0, 0.0, 0.0, -5.0, 0.0]), {"stub": True}

    monkeypatch.setattr(check, "household_agi", agi)

    status = check.main(["--h5", str(staging), "--out", str(out)])

    report = json.loads(out.read_text())
    assert status == check.EXIT_AT_RISK
    assert report["at_risk"] == ["01003"]
    assert report["households_with_agi_income"] == 2
    assert report["agi_engine"] == {"stub": True}
    assert report["minimum_support"] is None
    # The summary path defaults as R2's does: the H5 with .summary.json.
    assert calls == [(tmp_path / "staging.summary.json", check.DEFAULT_HH_CHUNK)]

    def healthy(frame, summary_path, *, release, hh_chunk):
        return np.asarray([1.0, 0.0, 0.0, 2.0, 0.0, 3.0]), {}

    monkeypatch.setattr(check, "household_agi", healthy)
    assert check.main(["--h5", str(staging), "--county-column", "county_fips"]) == 0


def test_an_agi_below_float32_resolution_is_stored_as_zero(
    check, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging.h5"
    staging.write_bytes(b"")
    summary = tmp_path / "other.summary.json"
    summary.write_text("{}")
    monkeypatch.setattr(check, "_release_tool", lambda: _stub_release(_frame()))
    monkeypatch.setattr(
        check,
        "household_agi",
        lambda frame, path, *, release, hh_chunk: (
            np.asarray([1.0, 0.0, 1e-50, 0.0, 1.0, 0.0]),
            {},
        ),
    )

    status = check.main(["--h5", str(staging), "--staging-summary", str(summary)])

    assert status == check.EXIT_AT_RISK


def test_command_refuses_without_a_staging_summary(
    check, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging.h5"
    staging.write_bytes(b"")
    monkeypatch.setattr(check, "_release_tool", lambda: _stub_release(_frame()))

    with pytest.raises(SystemExit, match="Staging summary not found"):
        check.main(["--h5", str(staging)])
