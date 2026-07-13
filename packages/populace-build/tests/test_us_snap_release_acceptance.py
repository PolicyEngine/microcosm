"""SNAP release acceptance tests."""

from __future__ import annotations

import pandas as pd

from populace.build.us_runtime.snap_release_acceptance import (
    SNAP_STATE_FIPS,
    assemble_us_snap_release_acceptance,
    load_snap_fy2022_participation_reference,
    us_snap_artifact_caseload_gate,
    us_snap_core_release_gate,
    us_snap_county_coverage_gate,
    us_snap_state_target_fit_gate,
)


def _target_rows(*, relative_error: float = 0.0) -> list[dict]:
    rows = []
    for role in ("snap_households", "snap_total"):
        for state in sorted(SNAP_STATE_FIPS):
            target = 100.0
            rows.append(
                {
                    "name": f"{role}.{state}",
                    "target_name": f"{role}.{state}",
                    "target": target,
                    "final_estimate": target * (1.0 + relative_error),
                    "metadata": {"target_role": role, "state_fips": state},
                }
            )
    return rows


def _snap_take_up_diagnostics() -> dict:
    return {
        "states_without_targets": [],
        "saturated_states": ["06"],
        "states": [
            {
                "state_fips": state,
                "target": 100.0,
                "eligible_weight": 200.0,
                "anchored_eligible_weight": 20.0,
                "caseload_weight": 100.0,
                "saturated": state == "06",
                "anchored_not_taking_up_count": 0,
                "max_unit_weight": 1.0,
            }
            for state in sorted(SNAP_STATE_FIPS)
        ],
    }


def _write_households(path, *, collapse_alaska: bool = False) -> None:
    rows = [
        {"state_fips": state, "county_fips": f"{state}001"}
        for state in sorted(SNAP_STATE_FIPS)
    ]
    rows.append(
        {
            "state_fips": "02",
            "county_fips": "02001" if collapse_alaska else "02003",
        }
    )
    pd.DataFrame(rows).to_hdf(path, key="household", format="table")


def test_fy2022_participation_reference_covers_every_state() -> None:
    reference = load_snap_fy2022_participation_reference()
    assert set(reference["state_rates"]) == SNAP_STATE_FIPS
    assert reference["national_rate"] == 0.88
    assert reference["state_rates"]["06"] == 0.81
    assert reference["state_rates"]["11"] == 1.0


def test_core_release_gate_requires_explicit_passing_gate() -> None:
    assert us_snap_core_release_gate(
        {"gates": {"calibration": {"passed": True}}}
    ).passed
    failed = us_snap_core_release_gate({"gates": {"calibration": {"passed": False}}})
    assert not failed.passed


def test_state_target_fit_gate_requires_all_states_within_tolerance() -> None:
    diagnostics = {"targets": _target_rows(relative_error=0.05)}
    assert us_snap_state_target_fit_gate(
        diagnostics,
        target_role="snap_households",
        label="households",
    ).passed

    diagnostics["targets"][0]["final_estimate"] = 130.0
    failed = us_snap_state_target_fit_gate(
        diagnostics,
        target_role="snap_households",
        label="households",
    )
    assert not failed.passed
    assert failed.details["states_outside_tolerance"] == ["01"]


def test_state_target_fit_gate_rejects_missing_state() -> None:
    rows = _target_rows()
    rows = [
        row
        for row in rows
        if not (
            row["metadata"]["target_role"] == "snap_total"
            and row["metadata"]["state_fips"] == "56"
        )
    ]
    failed = us_snap_state_target_fit_gate(
        {"targets": rows},
        target_role="snap_total",
        label="benefits",
    )
    assert not failed.passed
    assert any("56" in failure for failure in failed.failures)


def test_county_gate_checks_alaska_and_state_prefixes(tmp_path) -> None:
    good = tmp_path / "good.h5"
    _write_households(good)
    result = us_snap_county_coverage_gate(good)
    assert result.passed
    assert result.details["alaska_unique_county_fips"] == 2

    collapsed = tmp_path / "collapsed.h5"
    _write_households(collapsed, collapse_alaska=True)
    failed = us_snap_county_coverage_gate(collapsed)
    assert not failed.passed
    assert any("Alaska" in failure for failure in failed.failures)


def test_assembled_acceptance_keeps_participation_advisory(tmp_path) -> None:
    h5 = tmp_path / "candidate.h5"
    _write_households(h5)
    participation = {
        "required": False,
        "passed_within_advisory_tolerance": False,
        "states_outside_advisory_tolerance": ["06"],
    }
    payload = assemble_us_snap_release_acceptance(
        release_id="populace-us-test",
        build_manifest={
            "build_id": "populace-us-test",
            "gates": {"calibration": {"passed": True}},
        },
        snap_state_take_up=_snap_take_up_diagnostics(),
        calibration_diagnostics={"targets": _target_rows()},
        h5_path=h5,
        participation_validation=participation,
        simulated_caseloads={state: 100.0 for state in sorted(SNAP_STATE_FIPS)},
    )
    assert payload["passed"]
    assert payload["required_failures"] == []
    assert payload["california_saturation"]["saturated"] is True
    assert payload["fy2022_eligible_person_participation"] is participation


def test_artifact_caseload_gate_passes_when_simulated_matches_targets(tmp_path) -> None:
    diagnostics = {"targets": _target_rows()}
    result = us_snap_artifact_caseload_gate(
        tmp_path / "unused.h5",
        diagnostics,
        simulated_caseloads={state: 100.0 for state in sorted(SNAP_STATE_FIPS)},
    )
    assert result.passed
    assert result.details["states_outside_tolerance"] == []


def test_artifact_caseload_gate_fails_when_artifact_diverges_from_calibration(
    tmp_path,
) -> None:
    """The populace#419 class: calibrated fit green, shipped artifact 5x off."""
    diagnostics = {"targets": _target_rows(relative_error=0.0)}
    simulated = {state: 100.0 for state in sorted(SNAP_STATE_FIPS)}
    simulated["02"] = 663.0  # AK-style divergence
    result = us_snap_artifact_caseload_gate(
        tmp_path / "unused.h5",
        diagnostics,
        simulated_caseloads=simulated,
    )
    assert not result.passed
    assert result.details["states_outside_tolerance"] == ["02"]
    assert any("state 02" in failure for failure in result.failures)


def test_artifact_caseload_gate_fails_when_release_lacks_caseload_targets(
    tmp_path,
) -> None:
    diagnostics = {"targets": []}
    result = us_snap_artifact_caseload_gate(
        tmp_path / "unused.h5",
        diagnostics,
        simulated_caseloads={},
    )
    assert not result.passed
    assert any("predates" in failure for failure in result.failures)
