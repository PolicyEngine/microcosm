"""Post-build SNAP acceptance checks for a certified US release candidate.

The release builder already fails closed while constructing the SNAP take-up
surface.  This module checks the persisted candidate and its diagnostics as a
separate acceptance step:

* every state household-caseload and benefit target is present and fitted;
* the stage-time state take-up gate still reports a complete, anchor-preserving
  surface, with California saturation made explicit;
* county FIPS coverage is complete and internally consistent, including a
  non-collapsed Alaska borough/census-area distribution; and
* modeled eligible-person participation is compared with USDA FNS FY2022 state
  estimates as an advisory validator only.

The participation comparison deliberately does not seed take-up and does not
enter the release pass/fail verdict.  It has a different denominator and
vintage from the FY2024 average-monthly household targets used by calibration.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd

from populace.build.gates import GateResult
from populace.build.us_runtime.snap_state_take_up import us_snap_state_take_up_gate

SNAP_PARTICIPATION_REFERENCE_RESOURCE = "snap_fy2022_participation_rates.json"
SNAP_STATE_FIPS = frozenset(
    {
        "01",
        "02",
        "04",
        "05",
        "06",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
        "40",
        "41",
        "42",
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "50",
        "51",
        "53",
        "54",
        "55",
        "56",
    }
)

__all__ = [
    "SNAP_PARTICIPATION_REFERENCE_RESOURCE",
    "SNAP_STATE_FIPS",
    "assemble_us_snap_release_acceptance",
    "load_snap_fy2022_participation_reference",
    "us_snap_core_release_gate",
    "us_snap_county_coverage_gate",
    "us_snap_participation_validation",
    "us_snap_state_target_fit_gate",
]


def load_snap_fy2022_participation_reference() -> dict[str, Any]:
    """Load and validate the packaged FNS FY2022 eligible-person rates."""
    payload = json.loads(
        files("populace.build.us")
        .joinpath(SNAP_PARTICIPATION_REFERENCE_RESOURCE)
        .read_text()
    )
    rates = payload.get("state_rates")
    if not isinstance(rates, dict):
        raise ValueError("SNAP participation reference needs a state_rates mapping.")
    normalized = {str(state).zfill(2): float(rate) for state, rate in rates.items()}
    if set(normalized) != SNAP_STATE_FIPS:
        missing = sorted(SNAP_STATE_FIPS - set(normalized))
        unexpected = sorted(set(normalized) - SNAP_STATE_FIPS)
        raise ValueError(
            "SNAP participation reference must cover 50 states plus DC: "
            f"missing={missing}, unexpected={unexpected}."
        )
    invalid = {
        state: rate for state, rate in normalized.items() if not 0.0 < rate <= 1.0
    }
    if invalid:
        raise ValueError(f"SNAP participation reference has invalid rates: {invalid}.")
    payload["state_rates"] = normalized
    return payload


def _gate_payload(result: GateResult, *, required: bool = True) -> dict[str, Any]:
    return {
        "required": required,
        "passed": bool(result.passed),
        "failures": list(result.failures),
        "details": result.details,
    }


def us_snap_core_release_gate(build_manifest: dict[str, Any]) -> GateResult:
    """Require every explicit top-level release gate to have passed."""
    gates = build_manifest.get("gates")
    failures: list[str] = []
    checked: list[str] = []
    if not isinstance(gates, dict):
        failures.append("build_manifest.json has no gates object.")
        gates = {}
    for name, payload in sorted(gates.items()):
        if not isinstance(payload, dict) or "passed" not in payload:
            continue
        checked.append(name)
        if not bool(payload["passed"]):
            failures.append(f"build gate {name!r} did not pass.")
    if not checked:
        failures.append("build_manifest.json records no explicit passed release gates.")
    return GateResult(
        name="snap_core_release",
        passed=not failures,
        failures=tuple(failures),
        details={"checked_gates": checked},
    )


def us_snap_state_target_fit_gate(
    calibration_diagnostics: dict[str, Any],
    *,
    target_role: str,
    label: str,
    relative_tolerance: float = 0.10,
) -> GateResult:
    """Require 51 state target rows and bound every absolute relative miss."""
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must be in (0, 1).")
    targets = calibration_diagnostics.get("targets")
    if not isinstance(targets, list):
        targets = []
    selected = [
        row
        for row in targets
        if isinstance(row, dict)
        and isinstance(row.get("metadata"), dict)
        and row["metadata"].get("target_role") == target_role
        and row["metadata"].get("state_fips") is not None
    ]
    states = [str(row["metadata"]["state_fips"]).zfill(2) for row in selected]
    counts = Counter(states)
    duplicates = sorted(state for state, count in counts.items() if count != 1)
    missing = sorted(SNAP_STATE_FIPS - set(states))
    unexpected = sorted(set(states) - SNAP_STATE_FIPS)
    failures: list[str] = []
    if missing:
        failures.append(f"{label}: missing state target rows {missing}.")
    if unexpected:
        failures.append(f"{label}: unexpected state target rows {unexpected}.")
    if duplicates:
        failures.append(f"{label}: duplicate state target rows {duplicates}.")

    rows: list[dict[str, Any]] = []
    ordered = sorted(zip(states, selected, strict=True), key=lambda item: item[0])
    for state, row in ordered:
        target = float(row.get("target", 0.0))
        estimate = float(row.get("final_estimate", math.nan))
        relative_error = (
            (estimate - target) / target
            if target > 0 and math.isfinite(estimate)
            else math.nan
        )
        within = math.isfinite(relative_error) and (
            abs(relative_error) <= relative_tolerance
        )
        rows.append(
            {
                "state_fips": state,
                "target_name": row.get("target_name", row.get("name")),
                "target": target,
                "final_estimate": estimate if math.isfinite(estimate) else None,
                "relative_error": (
                    relative_error if math.isfinite(relative_error) else None
                ),
                "within_tolerance": within,
            }
        )
        if not within:
            rendered_error = (
                f"{relative_error:+.1%}"
                if math.isfinite(relative_error)
                else "not finite"
            )
            failures.append(
                f"{label}: state {state} relative miss {rendered_error} exceeds "
                f"{relative_tolerance:.1%}."
            )

    finite_errors = [
        abs(row["relative_error"]) for row in rows if row["relative_error"] is not None
    ]
    return GateResult(
        name=f"snap_{target_role}_state_fit",
        passed=not failures,
        failures=tuple(failures),
        details={
            "label": label,
            "target_role": target_role,
            "relative_tolerance": relative_tolerance,
            "state_rows": len(rows),
            "max_absolute_relative_error": (
                max(finite_errors) if finite_errors else None
            ),
            "states_outside_tolerance": [
                row["state_fips"] for row in rows if not row["within_tolerance"]
            ],
            "rows": rows,
        },
    )


def _normalized_fips(series: pd.Series, *, width: int) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(width)


def us_snap_county_coverage_gate(h5_path: Path | str) -> GateResult:
    """Require complete state-consistent county FIPS and non-collapsed Alaska."""
    h5_path = Path(h5_path)
    household = pd.read_hdf(
        h5_path,
        "household",
        columns=["state_fips", "county_fips"],
    )
    missing_state = int(household["state_fips"].isna().sum())
    missing_county = int(household["county_fips"].isna().sum())
    state = _normalized_fips(household["state_fips"], width=2)
    county = _normalized_fips(household["county_fips"], width=5)
    invalid_state = int((state.str.fullmatch(r"\d{2}") != True).sum())  # noqa: E712
    invalid_county = int((county.str.fullmatch(r"\d{5}") != True).sum())  # noqa: E712
    prefix_mismatch = int((county.str[:2] != state).fillna(True).sum())
    represented_states = set(state.dropna().astype(str))
    missing_states = sorted(SNAP_STATE_FIPS - represented_states)
    alaska = county[state == "02"].dropna().astype(str)
    alaska_counties = sorted(alaska.unique().tolist())
    failures: list[str] = []
    if missing_state or missing_county:
        failures.append(
            "household geography has missing values: "
            f"state_fips={missing_state}, county_fips={missing_county}."
        )
    if invalid_state or invalid_county:
        failures.append(
            "household geography has invalid FIPS formatting: "
            f"state_fips={invalid_state}, county_fips={invalid_county}."
        )
    if prefix_mismatch:
        failures.append(
            f"{prefix_mismatch} household county FIPS values disagree with state_fips."
        )
    if missing_states:
        failures.append(f"household geography is missing states {missing_states}.")
    if len(alaska_counties) < 2:
        failures.append(
            "Alaska county/borough geography collapsed to fewer than two FIPS values."
        )
    return GateResult(
        name="snap_county_coverage",
        passed=not failures,
        failures=tuple(failures),
        details={
            "household_records": int(len(household)),
            "missing_state_fips": missing_state,
            "missing_county_fips": missing_county,
            "invalid_state_fips": invalid_state,
            "invalid_county_fips": invalid_county,
            "state_county_prefix_mismatches": prefix_mismatch,
            "represented_state_count": len(represented_states & SNAP_STATE_FIPS),
            "alaska_household_records": int(len(alaska)),
            "alaska_unique_county_fips": len(alaska_counties),
            "alaska_county_fips": alaska_counties,
            "geography_semantics": (
                "Calibrated allocation from the Populace geography ladder, not "
                "measured household location."
            ),
        },
    )


def _spm_unit_state_fips(h5_path: Path) -> pd.Series:
    spm_unit = pd.read_hdf(h5_path, "spm_unit", columns=["spm_unit_id"])
    person = pd.read_hdf(
        h5_path,
        "person",
        columns=["person_spm_unit_id", "person_household_id"],
    )
    household = pd.read_hdf(
        h5_path,
        "household",
        columns=["household_id", "state_fips"],
    )
    household_counts = person.groupby("person_spm_unit_id")[
        "person_household_id"
    ].nunique()
    if (household_counts != 1).any():
        examples = household_counts[household_counts != 1].index[:5].tolist()
        raise ValueError(
            "SNAP participation validation requires each SPM unit to occupy one "
            f"household; invalid SPM unit ids include {examples}."
        )
    spm_to_household = person.drop_duplicates("person_spm_unit_id").set_index(
        "person_spm_unit_id"
    )["person_household_id"]
    household_state = household.set_index("household_id")["state_fips"]
    aligned = spm_unit["spm_unit_id"].map(spm_to_household).map(household_state)
    if aligned.isna().any():
        raise ValueError(
            "SNAP participation validation could not map every SPM unit to a state."
        )
    return _normalized_fips(aligned, width=2)


def us_snap_participation_validation(
    h5_path: Path | str,
    *,
    period: int = 2024,
    tolerance_percentage_points: float = 10.0,
) -> dict[str, Any]:
    """Compare post-calibration modeled participation with FNS FY2022 rates.

    PolicyEngine ``MicroSeries`` objects retain the release's calibrated
    weights through multiplication, filtering, and ``sum``; no weight vector
    is loaded or applied manually here.
    """
    if not 0.0 < tolerance_percentage_points <= 100.0:
        raise ValueError("tolerance_percentage_points must be in (0, 100].")
    from policyengine_us import Microsimulation
    from policyengine_us.data import USSingleYearDataset

    h5_path = Path(h5_path)
    reference = load_snap_fy2022_participation_reference()
    dataset = USSingleYearDataset(file_path=str(h5_path))
    simulation = Microsimulation(dataset=dataset)
    eligible = simulation.calc("is_snap_eligible", period=period, map_to="spm_unit")
    takes_up = simulation.calc(
        "takes_up_snap_if_eligible", period=period, map_to="spm_unit"
    )
    snap_unit_size = simulation.calc("snap_unit_size", period=period, map_to="spm_unit")
    eligible_people = eligible * snap_unit_size
    participating_people = eligible_people * takes_up
    state_fips = _spm_unit_state_fips(h5_path)
    if len(state_fips) != len(eligible_people):
        raise ValueError(
            "SNAP participation validation state and simulation rows do not align: "
            f"{len(state_fips)} states, {len(eligible_people)} simulation rows."
        )
    state_fips.index = eligible_people.index

    rows: list[dict[str, Any]] = []
    target_participating_total = 0.0
    for state, target_rate in sorted(reference["state_rates"].items()):
        mask = state_fips == state
        eligible_total = float(eligible_people[mask].sum())
        participating_total = float(participating_people[mask].sum())
        modeled_rate = (
            participating_total / eligible_total if eligible_total > 0 else None
        )
        difference_pp = (
            100.0 * (modeled_rate - target_rate) if modeled_rate is not None else None
        )
        within = difference_pp is not None and (
            abs(difference_pp) <= tolerance_percentage_points
        )
        target_participating_total += eligible_total * target_rate
        rows.append(
            {
                "state_fips": state,
                "fns_fy2022_rate": target_rate,
                "modeled_rate": modeled_rate,
                "difference_percentage_points": difference_pp,
                "within_tolerance": within,
                "fns_rate_capped_at_100": target_rate == 1.0,
                "modeled_eligible_people": eligible_total,
                "modeled_participating_people": participating_total,
            }
        )

    eligible_national = float(eligible_people.sum())
    participating_national = float(participating_people.sum())
    modeled_national_rate = (
        participating_national / eligible_national if eligible_national > 0 else None
    )
    state_mix_reference_rate = (
        target_participating_total / eligible_national
        if eligible_national > 0
        else None
    )
    misses = [row for row in rows if not row["within_tolerance"]]
    finite_differences = [
        abs(row["difference_percentage_points"])
        for row in rows
        if row["difference_percentage_points"] is not None
    ]
    return {
        "required": False,
        "passed_within_advisory_tolerance": not misses,
        "advisory_tolerance_percentage_points": tolerance_percentage_points,
        "reason_not_release_blocking": (
            "FNS FY2022 estimates measure eligible people and carry sampling/model "
            "uncertainty; release calibration uses FY2024 average-monthly households."
        ),
        "source": {
            key: reference[key]
            for key in (
                "title",
                "publisher",
                "landing_page",
                "report",
                "table",
                "fiscal_year",
                "measure",
                "national_rate",
                "source_precision",
                "caveats",
            )
        },
        "national": {
            "modeled_eligible_people": eligible_national,
            "modeled_participating_people": participating_national,
            "modeled_rate": modeled_national_rate,
            "published_fns_rate": reference["national_rate"],
            "state_mix_weighted_fns_rate": state_mix_reference_rate,
        },
        "states_outside_advisory_tolerance": [row["state_fips"] for row in misses],
        "max_absolute_difference_percentage_points": (
            max(finite_differences) if finite_differences else None
        ),
        "rows": rows,
    }


def assemble_us_snap_release_acceptance(
    *,
    release_id: str,
    build_manifest: dict[str, Any],
    snap_state_take_up: dict[str, Any],
    calibration_diagnostics: dict[str, Any],
    h5_path: Path | str,
    participation_validation: dict[str, Any],
    target_relative_tolerance: float = 0.10,
) -> dict[str, Any]:
    """Assemble the final required-gate verdict and advisory validator."""
    checks = {
        "core_release": _gate_payload(us_snap_core_release_gate(build_manifest)),
        "state_take_up": _gate_payload(us_snap_state_take_up_gate(snap_state_take_up)),
        "state_household_caseload_fit": _gate_payload(
            us_snap_state_target_fit_gate(
                calibration_diagnostics,
                target_role="snap_households",
                label="FNS FY2024 average-monthly SNAP households",
                relative_tolerance=target_relative_tolerance,
            )
        ),
        "state_benefit_fit": _gate_payload(
            us_snap_state_target_fit_gate(
                calibration_diagnostics,
                target_role="snap_total",
                label="FNS FY2024 SNAP benefits",
                relative_tolerance=target_relative_tolerance,
            )
        ),
        "county_coverage": _gate_payload(us_snap_county_coverage_gate(h5_path)),
    }
    required_failures = [
        name
        for name, check in checks.items()
        if check["required"] and not check["passed"]
    ]
    saturated_states = snap_state_take_up.get("saturated_states", [])
    return {
        "schema_version": 1,
        "classification": "release_acceptance",
        "release_id": release_id,
        "passed": not required_failures,
        "required_failures": required_failures,
        "checks": checks,
        "california_saturation": {
            "saturated": "06" in saturated_states,
            "saturated_states": saturated_states,
            "interpretation": (
                "Saturation means the FNS household count meets or exceeds modeled "
                "eligible SPM-unit weight; it is an eligibility-undercount diagnostic, "
                "not evidence of literal universal participation."
            ),
        },
        "fy2022_eligible_person_participation": participation_validation,
    }
