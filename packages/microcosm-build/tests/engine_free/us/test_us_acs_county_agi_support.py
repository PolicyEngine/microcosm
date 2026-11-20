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
    # 1007 has ladder population but no household after the minimum-support
    # moves (the report's input), so R2's check_support refuses it.
    assert report["no_household"] == ["01007"]
    assert report["unsupported_counties"] == ["01007"]
    # A group-quarters record counts for AGI but not for support.
    assert report["by_county"]["01005"]["supporting_households"] == 1
    assert report["fewer_than_min_support_with_agi_income"] == ["01001", "01005"]
    assert report["counties_outside_ladder"] == []
    assert check.exit_status(report) == check.EXIT_R2_REFUSES


def test_report_lists_the_counties_r2_refuses_before_its_engine_pass(check) -> None:
    households = _frame().table("household").copy()
    # 01005 keeps only its group-quarters record, and one household names
    # 01009, a county outside the ladder.
    households["county_fips_2023"] = [1001, 1001, 1003, 1009, 1003, 1005]
    nonzero = np.ones(len(households), dtype=bool)

    report = check.county_agi_report(
        households,
        nonzero,
        county_column="county_fips_2023",
        county_populations={1001: 10.0, 1003: 5.0, 1005: 5.0},
    )

    assert report["at_risk"] == []
    assert report["no_household"] == []
    assert report["counties_outside_ladder"] == ["01009"]
    assert report["unsupported_counties"] == ["01005"]
    assert check.exit_status(report) == check.EXIT_R2_REFUSES
    # Without the ladder neither list is checked.
    unladdered = check.county_agi_report(
        households, nonzero, county_column="county_fips_2023"
    )
    assert unladdered["counties_outside_ladder"] is None
    assert unladdered["unsupported_counties"] is None
    assert check.exit_status(unladdered) == 0


def test_exit_status_3_takes_precedence_over_2(check) -> None:
    at_risk = {"at_risk": ["01003"], "counties_outside_ladder": []}
    assert check.exit_status({**at_risk, "unsupported_counties": []}) == 2
    assert check.exit_status({**at_risk, "unsupported_counties": ["01005"]}) == 3
    assert (
        check.exit_status(
            {
                **at_risk,
                "counties_outside_ladder": ["01009"],
                "unsupported_counties": [],
            }
        )
        == 3
    )
    assert check.exit_status({"at_risk": [], "unsupported_counties": None}) == 0


def test_ladder_support_moves_no_household_when_a_county_is_outside_the_ladder(
    check, monkeypatch: pytest.MonkeyPatch
) -> None:
    import microcosm.build.us_runtime.puma_ladder as puma_ladder

    households = _frame().table("household")
    surface = check.county_surface
    planned = []
    monkeypatch.setattr(puma_ladder, "load_us_puma_ladder", lambda path: "ladder")
    monkeypatch.setattr(surface, "county_puma_overlaps", lambda path, column: {})

    def plan(households, populations, overlaps, *, county_column, per_county):
        planned.append(sorted(populations))
        return [], {"rewritten_columns": [county_column], "moves": []}

    monkeypatch.setattr(surface, "plan_minimum_support", plan)
    for populations, expected in (
        # 01005 is outside this ladder: R2 refuses the run before any move.
        ({1001: 10.0, 1003: 5.0, 1007: 3.0}, []),
        # Every county is in the ladder and 01007 has no household: R2 plans
        # its moves.
        ({1001: 10.0, 1003: 5.0, 1005: 5.0, 1007: 3.0}, [[1001, 1003, 1005, 1007]]),
    ):
        planned.clear()
        monkeypatch.setattr(
            surface,
            "with_county_populations",
            lambda ladder, known, geographies, *, county_column, _p=populations: {
                "county": _p
            },
        )

        _moved, ladder_populations, support = check.ladder_support(
            households,
            Path("ladder.npz"),
            county_column="county_fips_2023",
            county_min_support=3,
        )

        assert ladder_populations == populations
        assert planned == expected
        assert (support is None) is (not expected)


_NEVER_DEFAULT = frozenset(
    {
        ("person", "takes_up_ssi_if_eligible"),
        ("person", "is_blind"),
        ("spm_unit", "takes_up_snap_if_eligible"),
    }
)


#: The columns of ``_NEVER_DEFAULT`` that an R2 stage fills on ACS rows;
#: ``is_blind`` stands in for one that staging must fill.
_STAGE_FILLED = frozenset(
    {
        ("person", "takes_up_ssi_if_eligible"),
        ("spm_unit", "takes_up_snap_if_eligible"),
    }
)
#: One person per household: persons 5 and 6 are ACS rows.
_PERSON_SPINE = ["asec_puf"] * 4 + ["acs_2024_1yr"] * 2


def _hold_back(check, frame, agi_inputs):
    return check.hold_back_null_runtime_columns(
        frame,
        never_default_filled=_NEVER_DEFAULT,
        stage_filled=_STAGE_FILLED,
        agi_inputs=agi_inputs,
    )


def test_null_stage_filled_columns_on_acs_rows_outside_the_agi_are_held_back(
    check,
) -> None:
    frame = _frame(
        {
            "person_spine": _PERSON_SPINE,
            "takes_up_ssi_if_eligible": [True, True, True, True, np.nan, True],
            "is_blind": [False] * 6,
        }
    )

    held_frame, held = _hold_back(check, frame, {"is_blind", "age"})

    assert held == {"person": ["takes_up_ssi_if_eligible"]}
    assert "takes_up_ssi_if_eligible" not in held_frame.table("person").columns
    assert "is_blind" in held_frame.table("person").columns
    assert held_frame.n("household") == frame.n("household")


@pytest.mark.parametrize(
    ("columns", "agi_inputs", "message"),
    [
        # An R2 stage fills it on the ACS rows, but the AGI reads it.
        (
            {"takes_up_ssi_if_eligible": [True] * 4 + [np.nan, True]},
            {"takes_up_ssi_if_eligible"},
            r"person\.takes_up_ssi_if_eligible: the AGI reads it",
        ),
        # R2's stages fill the ACS rows only; this null is on a donor row.
        (
            {"takes_up_ssi_if_eligible": [np.nan] + [True] * 4 + [np.nan]},
            {"age"},
            r"person\.takes_up_ssi_if_eligible: null on 1 non-ACS row",
        ),
        # No R2 stage fills it, so staging must, on any spine.
        (
            {"is_blind": [False] * 4 + [np.nan, False]},
            {"age"},
            r"person\.is_blind: no R2 stage fills it",
        ),
    ],
    ids=["agi-input", "donor-row", "staging-filled"],
)
def test_never_default_nulls_r2_refuses_are_refused(
    check, columns, agi_inputs, message
) -> None:
    frame = _frame({"person_spine": _PERSON_SPINE, **columns})

    with pytest.raises(SystemExit, match=message):
        _hold_back(check, frame, agi_inputs)


def test_a_null_stage_filled_column_without_a_spine_tag_is_refused(check) -> None:
    frame = _frame({"takes_up_ssi_if_eligible": [True] * 4 + [np.nan, True]})

    with pytest.raises(SystemExit, match="no person_spine column"):
        _hold_back(check, frame, {"age"})


def test_a_frame_without_null_runtime_columns_is_returned_as_is(check) -> None:
    frame = _frame({"takes_up_ssi_if_eligible": [True] * 6})

    held_frame, held = _hold_back(check, frame, {"age"})

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


@dataclass(frozen=True)
class _EdgeClosure:
    input_leaves: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]


def test_agi_engine_inputs_stop_at_a_deduction_the_year_does_not_list(check) -> None:
    """Past an unread deduction the walk reads nothing; a leaf it reaches by
    another path stays."""

    class _Index:
        def variable_dependency_closure(self, name):
            assert name == "adjusted_gross_income"
            return _EdgeClosure(
                input_leaves=("age", "employment_income_before_lsr", "takes_up_x"),
                edges=(
                    ("adjusted_gross_income", "irs_gross_income"),
                    ("adjusted_gross_income", "charitable_contributions_ald"),
                    ("irs_gross_income", "employment_income_before_lsr"),
                    ("charitable_contributions_ald", "tax_unit_itemizes"),
                    ("tax_unit_itemizes", "takes_up_x"),
                    ("tax_unit_itemizes", "age"),
                    ("irs_gross_income", "age"),
                ),
            )

    unread = ("charitable_contributions_ald",)
    assert check.agi_engine_inputs(_Index(), unread=unread) == {
        "age",
        "employment_income_before_lsr",
    }
    assert check.agi_engine_inputs(_Index()) == {
        "age",
        "employment_income_before_lsr",
        "takes_up_x",
    }


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
    assert report["exit_status"] == check.EXIT_AT_RISK
    assert report["at_risk"] == ["01003"]
    # Without --puma-ladder, R2's support refusals are not checked.
    assert report["unsupported_counties"] is None
    assert report["households_with_agi_income"] == 2
    assert report["agi_engine"] == {"stub": True}
    assert report["minimum_support"] is None
    # The summary path defaults as R2's does: the H5 with .summary.json.
    assert calls == [(tmp_path / "staging.summary.json", check.DEFAULT_HH_CHUNK)]

    def healthy(frame, summary_path, *, release, hh_chunk):
        return np.asarray([1.0, 0.0, 0.0, 2.0, 0.0, 3.0]), {}

    monkeypatch.setattr(check, "household_agi", healthy)
    assert check.main(["--h5", str(staging), "--county-column", "county_fips"]) == 0


def test_command_exits_3_when_r2_refuses_before_its_engine_pass(
    check, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging.h5"
    staging.write_bytes(b"")
    (tmp_path / "staging.summary.json").write_text("{}")
    out = tmp_path / "report.json"
    monkeypatch.setattr(check, "_release_tool", lambda: _stub_release(_frame()))
    # 01003's households have an AGI of 0, so it is also at risk.
    monkeypatch.setattr(
        check,
        "household_agi",
        lambda frame, path, *, release, hh_chunk: (
            np.asarray([1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
            {},
        ),
    )
    calls = []

    def ladder(households, path, *, county_column, county_min_support):
        calls.append((path, county_column, county_min_support))
        # 01005 is not in this ladder.
        return households, {1001: 10.0, 1003: 5.0}, None

    monkeypatch.setattr(check, "ladder_support", ladder)
    ladder_path = tmp_path / "ladder.npz"

    status = check.main(
        ["--h5", str(staging), "--puma-ladder", str(ladder_path), "--out", str(out)]
    )

    report = json.loads(out.read_text())
    assert status == check.EXIT_R2_REFUSES
    assert report["exit_status"] == check.EXIT_R2_REFUSES
    assert report["at_risk"] == ["01003"]
    assert report["counties_outside_ladder"] == ["01005"]
    assert report["unsupported_counties"] == []
    assert calls == [
        (
            ladder_path,
            check.county_surface.COUNTY_COLUMN_2023,
            check.county_surface.DEFAULT_COUNTY_MIN_SUPPORT,
        )
    ]


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
