"""Synthetic and restricted full-artifact diagnostics for QBI simulation v3."""

from __future__ import annotations

import hashlib
import json
import os
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from populace.build.us_runtime import qbi_inputs as qbi_inputs_module
from populace.build.us_runtime.qbi_inputs import US_QBI_OUTPUT_COLUMNS
from populace.build.us_runtime.qbi_simulation import (
    QBI_SIMULATION_V2,
    QBI_SIMULATION_V3,
    QbiSimulationInputs,
    load_qbi_simulation_assumptions,
    qbi_simulation_summary,
    simulate_qbi_v3_wage_capital,
    with_qbi_simulation_from_puf_arrays,
)

_PUF_2024_PATH = os.environ.get("POPULACE_PUF_2024_H5")
requires_puf_2024 = pytest.mark.skipif(
    not _PUF_2024_PATH or not Path(_PUF_2024_PATH).is_file(),
    reason="set POPULACE_PUF_2024_H5 to the restricted pinned artifact",
)

_FORM_NAMES = (
    "sole_proprietorship",
    "partnership",
    "s_corporation",
)
_EXPECTED_V2_W2_NONZERO_SHARE = 0.000995818675737822
_EXPECTED_V3_W2_NONZERO_SHARE = 0.020430544809711643
_EXPECTED_V2_TO_V3_W2_NONZERO_SHARE_DELTA = 0.01943472613397382
_ZERO_EMPLOYEE_ABSOLUTE_TOLERANCE = 0.02


def _v3_assumptions_payload() -> dict[str, object]:
    resource = files("populace.build.us").joinpath("qbi_assumptions_v3.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested_mapping(
    value: object,
    *,
    label: str,
) -> dict[str, object]:
    assert isinstance(value, dict), f"{label} must be a JSON object"
    return value


def _persisted_replay_contract() -> tuple[
    dict[str, float],
    float,
    tuple[float, float],
]:
    payload = _v3_assumptions_payload()
    employer_presence = _nested_mapping(
        payload["employer_presence"],
        label="employer_presence",
    )
    calibration = _nested_mapping(
        employer_presence["calibration"],
        label="employer_presence.calibration",
    )
    raw_targets = _nested_mapping(
        calibration["target_zero_employee_share_by_form"],
        label=("employer_presence.calibration.target_zero_employee_share_by_form"),
    )
    targets = {form: float(raw_targets[form]) for form in _FORM_NAMES}
    overall_target = float(calibration["overall_zero_employee_target"])

    w2 = _nested_mapping(payload["w2"], label="w2")
    plausibility_band = _nested_mapping(
        w2["plausibility_band"],
        label="w2.plausibility_band",
    )
    wage_band = (
        float(plausibility_band["lower_dollars"]),
        float(plausibility_band["upper_dollars"]),
    )
    return targets, overall_target, wage_band


def _synthetic_arrays(n: int = 4_096) -> dict[str, np.ndarray]:
    row = np.arange(n, dtype=np.float64)
    form_pattern = row.astype(np.int64) % 3
    is_sole = form_pattern == 0
    is_passthrough = form_pattern == 1
    return {
        "self_employment_income": np.where(
            is_sole,
            15_000.0 + (row % 80) * 5_000.0,
            0.0,
        ),
        "farm_operations_income": np.zeros(n, dtype=np.float64),
        "farm_rent_income": np.zeros(n, dtype=np.float64),
        "rental_income": np.zeros(n, dtype=np.float64),
        "estate_income": np.zeros(n, dtype=np.float64),
        "partnership_s_corp_income": np.where(
            is_passthrough,
            20_000.0 + (row % 90) * 6_000.0,
            0.0,
        ),
        "non_qualified_dividend_income": np.where(
            (row.astype(np.int64) % 5) == 0,
            1_000.0 + row,
            0.0,
        ),
    }


def _restricted_arrays_and_weights() -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
]:
    h5py = pytest.importorskip("h5py")
    assert _PUF_2024_PATH is not None
    keys = (
        "tax_unit_id",
        "household_weight",
        "person_tax_unit_id",
        "self_employment_income",
        "farm_rent_income",
        "rental_income",
        "estate_income",
        "partnership_s_corp_income",
        "non_qualified_dividend_income",
    )
    with h5py.File(_PUF_2024_PATH) as artifact:
        arrays = {key: artifact[key][:] for key in keys}

    tax_unit_ids = np.asarray(arrays["tax_unit_id"])
    person_tax_unit_ids = np.asarray(arrays["person_tax_unit_id"])
    assert np.all(tax_unit_ids[:-1] <= tax_unit_ids[1:])
    tax_unit_positions = np.searchsorted(tax_unit_ids, person_tax_unit_ids)
    assert np.all(tax_unit_ids[tax_unit_positions] == person_tax_unit_ids)
    person_weights = np.asarray(
        arrays["household_weight"],
        dtype=np.float64,
    )[tax_unit_positions]
    return arrays, person_weights


def _weighted_share(
    mask: np.ndarray,
    weights: np.ndarray,
    *,
    universe: np.ndarray | None = None,
) -> float:
    denominator = weights if universe is None else weights[universe]
    total_weight = float(denominator.sum())
    assert total_weight > 0.0
    return float(weights[mask].sum()) / total_weight


def test_v3_synthetic_diagnostics_match_the_public_fifteen_outputs() -> None:
    arrays = _synthetic_arrays()
    inputs = QbiSimulationInputs.from_puf_arrays(arrays)
    assumptions = load_qbi_simulation_assumptions(QBI_SIMULATION_V3)

    first = simulate_qbi_v3_wage_capital(
        inputs,
        assumptions=assumptions,
    )
    second = simulate_qbi_v3_wage_capital(
        inputs,
        assumptions=assumptions,
    )
    public = with_qbi_simulation_from_puf_arrays(
        arrays,
        qbi_simulation_version=QBI_SIMULATION_V3,
        assumptions=assumptions,
    )

    for name in (
        "w2_wages",
        "ubia",
        "positive_qbi",
        "legal_form",
        "has_employees",
        "receipts",
    ):
        np.testing.assert_array_equal(
            getattr(first, name),
            getattr(second, name),
        )
    assert set(first.legal_form) == {"none", *_FORM_NAMES}
    assert set(first.legal_form[first.positive_qbi]) == set(_FORM_NAMES)
    assert not np.any(first.w2_wages[~first.has_employees])
    assert not np.any(first.receipts[~first.positive_qbi])
    assert set(US_QBI_OUTPUT_COLUMNS) <= set(public)
    np.testing.assert_array_equal(
        first.w2_wages,
        public["w2_wages_from_qualified_business"],
    )
    np.testing.assert_array_equal(
        first.ubia,
        public["unadjusted_basis_qualified_property"],
    )


def test_v3_persisted_replay_contract_names_every_form() -> None:
    payload = _v3_assumptions_payload()
    employer_presence = _nested_mapping(
        payload["employer_presence"],
        label="employer_presence",
    )
    calibration = _nested_mapping(
        employer_presence["calibration"],
        label="employer_presence.calibration",
    )
    targets, overall_target, wage_band = _persisted_replay_contract()

    shifts = _nested_mapping(
        calibration["log_odds_shift_by_form"],
        label="employer_presence.calibration.log_odds_shift_by_form",
    )
    expected = _nested_mapping(
        calibration["expected_zero_employee_share_by_form"],
        label=("employer_presence.calibration.expected_zero_employee_share_by_form"),
    )
    assert set(targets) == set(_FORM_NAMES)
    assert set(shifts) == set(_FORM_NAMES)
    assert set(expected) == set(_FORM_NAMES)
    assert all(0.0 < target < 1.0 for target in targets.values())
    assert all(np.isfinite(float(shifts[form])) for form in _FORM_NAMES)
    assert all(0.0 < float(expected[form]) < 1.0 for form in _FORM_NAMES)
    assert 0.0 < overall_target < 1.0
    assert 0.0 < wage_band[0] <= wage_band[1]


@requires_puf_2024
def test_v3_full_artifact_replay_matches_persisted_diagnostics(
    record_property,
) -> None:
    assert _PUF_2024_PATH is not None
    artifact_path = Path(_PUF_2024_PATH)
    calibration = _nested_mapping(
        _nested_mapping(
            _v3_assumptions_payload()["employer_presence"],
            label="employer_presence",
        )["calibration"],
        label="employer_presence.calibration",
    )
    replay = _nested_mapping(
        calibration["replay"],
        label="employer_presence.calibration.replay",
    )
    assert artifact_path.name == replay["artifact_filename"]
    assert artifact_path.stat().st_size == replay["artifact_bytes"]
    assert _sha256(artifact_path) == replay["artifact_sha256"]

    arrays, person_weights = _restricted_arrays_and_weights()
    inputs = QbiSimulationInputs.from_puf_arrays(arrays)
    assumptions = load_qbi_simulation_assumptions(QBI_SIMULATION_V3)
    diagnostic = simulate_qbi_v3_wage_capital(
        inputs,
        assumptions=assumptions,
    )
    v3_public = with_qbi_simulation_from_puf_arrays(
        arrays,
        qbi_simulation_version=QBI_SIMULATION_V3,
        assumptions=assumptions,
    )
    v2_public = with_qbi_simulation_from_puf_arrays(
        arrays,
        qbi_simulation_version=QBI_SIMULATION_V2,
    )

    targets, overall_target, wage_band = _persisted_replay_contract()
    for form, target in targets.items():
        form_mask = diagnostic.positive_qbi & (diagnostic.legal_form == form)
        assert np.any(form_mask)
        realized = _weighted_share(
            form_mask & ~diagnostic.has_employees,
            person_weights,
            universe=form_mask,
        )
        record_property(f"qbi_v3_{form}_zero_employee_share", realized)
        assert realized == pytest.approx(
            target,
            abs=_ZERO_EMPLOYEE_ABSOLUTE_TOLERANCE,
        )

    realized_overall = _weighted_share(
        diagnostic.positive_qbi & ~diagnostic.has_employees,
        person_weights,
        universe=diagnostic.positive_qbi,
    )
    record_property("qbi_v3_overall_zero_employee_share", realized_overall)
    assert realized_overall == pytest.approx(
        overall_target,
        abs=_ZERO_EMPLOYEE_ABSOLUTE_TOLERANCE,
    )

    np.testing.assert_array_equal(
        diagnostic.w2_wages,
        v3_public["w2_wages_from_qualified_business"],
    )
    np.testing.assert_array_equal(
        diagnostic.ubia,
        v3_public["unadjusted_basis_qualified_property"],
    )
    total_person_weight = float(person_weights.sum())
    w2_aggregate = float(np.sum(diagnostic.w2_wages * person_weights))
    record_property("qbi_v3_w2_aggregate_dollars", w2_aggregate)
    assert wage_band[0] <= w2_aggregate <= wage_band[1]
    public_summary = qbi_simulation_summary(
        v3_public,
        weights=person_weights,
    )
    assert float(
        public_summary["w2_wages_from_qualified_business"]["weighted_mean"]
    ) * total_person_weight == pytest.approx(w2_aggregate, rel=1e-12)

    v2_w2 = np.asarray(
        v2_public["w2_wages_from_qualified_business"],
        dtype=np.float64,
    )
    v3_w2 = diagnostic.w2_wages
    v2_w2_share = _weighted_share(v2_w2 != 0.0, person_weights)
    v3_w2_share = _weighted_share(v3_w2 != 0.0, person_weights)
    w2_share_delta = v3_w2_share - v2_w2_share
    record_property("qbi_v2_w2_nonzero_share", v2_w2_share)
    record_property("qbi_v3_w2_nonzero_share", v3_w2_share)
    record_property("qbi_v2_to_v3_w2_nonzero_share_delta", w2_share_delta)
    assert v2_w2_share == pytest.approx(
        _EXPECTED_V2_W2_NONZERO_SHARE,
        abs=1e-15,
    )
    assert v3_w2_share == pytest.approx(
        _EXPECTED_V3_W2_NONZERO_SHARE,
        abs=1e-15,
    )
    assert w2_share_delta == pytest.approx(
        _EXPECTED_V2_TO_V3_W2_NONZERO_SHARE_DELTA,
        abs=1e-15,
    )
    w2_signal_band = qbi_inputs_module._NUMERIC_NONZERO_SHARE_BANDS[
        "w2_wages_from_qualified_business"
    ]
    assert w2_signal_band == (0.001, 0.35)
    assert w2_signal_band[0] <= v3_w2_share <= w2_signal_band[1]

    v2_reit_ptp = np.asarray(
        v2_public["qualified_reit_and_ptp_income"],
        dtype=np.float64,
    )
    v3_reit_ptp = np.asarray(
        v3_public["qualified_reit_and_ptp_income"],
        dtype=np.float64,
    )
    assert v3_reit_ptp.tobytes() == v2_reit_ptp.tobytes()
    reit_ptp_aggregate = float(np.sum(v3_reit_ptp * person_weights))
    anchor = assumptions.reit_ptp_anchor
    assert anchor.replay_factor_band is not None
    assert anchor.published_income_dollars is not None
    low_factor, high_factor = anchor.replay_factor_band
    assert (
        low_factor * anchor.published_income_dollars
        <= reit_ptp_aggregate
        <= high_factor * anchor.published_income_dollars
    )
