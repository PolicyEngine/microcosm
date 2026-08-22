"""Assignment and replay contracts for passive partnership/S-corp income."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import qbi_passive_passthrough as passive_module
from microcosm.build.us_runtime.qbi_inputs import (
    US_QBI_BOOLEAN_OUTPUT_COLUMNS,
    US_QBI_OUTPUT_COLUMNS,
    with_us_qbi_input_reconciliation,
)
from microcosm.build.us_runtime.qbi_passive_passthrough import (
    QBI_PASSIVE_EVIDENCE_RESOURCE,
    US_QBI_PASSIVE_PASSTHROUGH_OUTPUT_COLUMN,
    assign_passive_partnership_s_corp_income,
    calibrate_qbi_passive_log_odds_shift,
    load_qbi_passive_passthrough_assumptions,
    validate_qbi_passive_passthrough_assumptions,
    with_us_qbi_passive_passthrough_assignment,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

ROOT = Path(__file__).resolve().parents[3]
ASSUMPTIONS_BUILDER_PATH = (
    ROOT / "tools/build_us_qbi_passive_passthrough_assumptions.py"
)
PUF_REPLAY_ENVIRONMENT = "POPU" + "LACE_PUF_2024_H5"
EXPECTED_REPLAY_SHA256 = (
    "8182579ddfecaf5e5b872e2307b88f03e8e8def993171b648f701a19a847f37b"
)
EXPECTED_REPLAY_BYTES = 241_045_964
EXPECTED_REPLAY_TAX_UNIT_ROWS = 207_692
EXPECTED_REPLAY_PERSON_ROWS = 484_015
PROVISIONAL_TARGET = 54_628_492_000.0
QBI_PASSIVE_ASSUMPTIONS_RESOURCE = "qbi_passive_passthrough_assumptions_v1.json"
_TEST_LOCAL_QUANTILE_NAMES = ("q05", "q25", "q50", "q75", "q95")
_TEST_LOCAL_SCHEDULE_E_CUTS = np.array(
    [0.0, 25_000.0, 100_000.0, 250_000.0, 1_000_000.0]
)


def _load_assumptions_builder():
    spec = importlib.util.spec_from_file_location(
        "qbi_passive_passthrough_assumptions_builder",
        ASSUMPTIONS_BUILDER_PATH,
    )
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def _test_local_resource_payload(name: str) -> dict[str, object]:
    payload = json.loads(files("microcosm.build.us").joinpath(name).read_bytes())
    assert isinstance(payload, dict)
    return payload


def _test_local_evidence_model(
    evidence: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    methodology = evidence["methodology"]
    assert isinstance(methodology, dict)
    expected_band_order = [
        "nonpositive",
        "0_to_25k",
        "25k_to_100k",
        "100k_to_250k",
        "250k_to_1m",
        "over_1m",
    ]
    assert [band["id"] for band in methodology["income_bands"]] == expected_band_order
    probabilities = np.asarray(
        methodology["quantile_probabilities"],
        dtype=np.float64,
    )
    assert probabilities.tolist() == [0.05, 0.25, 0.5, 0.75, 0.95]
    cells = evidence["cells"]
    assert isinstance(cells, list)
    assert [cell["income_band"] for cell in cells] == expected_band_order
    prevalence = np.asarray(
        [cell["holding_prevalence"]["estimate"] for cell in cells],
        dtype=np.float64,
    )
    quantiles = np.asarray(
        [
            [
                cell["conditional_share"]["selected_quantiles"][name]
                for name in _TEST_LOCAL_QUANTILE_NAMES
            ]
            for cell in cells
        ],
        dtype=np.float64,
    )
    probability_knots = np.concatenate(([0.0], probabilities, [1.0]))
    flat_extended_quantiles = np.concatenate(
        (quantiles[:, :1], quantiles, quantiles[:, -1:]),
        axis=1,
    )
    share_means = np.sum(
        np.diff(probability_knots)
        * (flat_extended_quantiles[:, :-1] + flat_extended_quantiles[:, 1:])
        / 2.0,
        axis=1,
    )
    return prevalence, probabilities, quantiles, share_means


def _test_local_shifted_probability(
    prevalence: np.ndarray,
    shift: float,
) -> np.ndarray:
    odds = prevalence / (1.0 - prevalence)
    shifted_odds = odds * math.exp(shift)
    return shifted_odds / (1.0 + shifted_odds)


def _test_local_expected_aggregate(
    partnership_s_corp_income: object,
    schedule_e_income: object,
    person_weights: object,
    *,
    evidence: dict[str, object],
    log_odds_shift: float,
) -> float:
    passthrough = np.maximum(
        np.asarray(partnership_s_corp_income, dtype=np.float64),
        0.0,
    )
    schedule_e = np.asarray(schedule_e_income, dtype=np.float64)
    weights = np.asarray(person_weights, dtype=np.float64)
    assert passthrough.ndim == schedule_e.ndim == weights.ndim == 1
    assert len(passthrough) == len(schedule_e) == len(weights)
    assert np.isfinite(passthrough).all()
    assert np.isfinite(schedule_e).all()
    assert np.isfinite(weights).all() and np.all(weights >= 0.0)

    band = np.searchsorted(
        _TEST_LOCAL_SCHEDULE_E_CUTS,
        schedule_e,
        side="left",
    )
    prevalence, _probabilities, _quantiles, share_means = _test_local_evidence_model(
        evidence
    )
    chance = _test_local_shifted_probability(prevalence, log_odds_shift)
    return float(
        np.sum(
            weights * passthrough * share_means[band] * chance[band],
            dtype=np.float64,
        )
    )


def _test_local_bisect_shift(
    partnership_s_corp_income: object,
    schedule_e_income: object,
    person_weights: object,
    *,
    evidence: dict[str, object],
    target: float,
    solver: dict[str, object],
) -> tuple[float, float]:
    lower = float(solver["lower_bound"])
    upper = float(solver["upper_bound"])
    iterations = int(solver["iterations"])

    def expected(shift: float) -> float:
        return _test_local_expected_aggregate(
            partnership_s_corp_income,
            schedule_e_income,
            person_weights,
            evidence=evidence,
            log_odds_shift=shift,
        )

    if not expected(lower) <= target <= expected(upper):
        raise ValueError("Test-local passive target is outside the solver bracket.")
    for _iteration in range(iterations):
        candidate = lower + (upper - lower) / 2.0
        if expected(candidate) >= target:
            upper = candidate
        else:
            lower = candidate
    shift = lower + (upper - lower) / 2.0
    return shift, expected(shift)


def _test_local_bisection_tolerance(
    solver: dict[str, object],
    *,
    reference_shift: float,
) -> float:
    span = float(solver["upper_bound"]) - float(solver["lower_bound"])
    nominal_final_bracket = math.ldexp(span, -int(solver["iterations"]))
    # The documented 128 halvings outrun binary64. Around mathematical zero,
    # exp(shift) and the shifted odds operate at unit scale, so one ULP of that
    # scale is the effective floor; nonzero roots use their own larger scale.
    binary64_floor = math.ulp(max(1.0, abs(reference_shift)))
    return max(nominal_final_bracket, binary64_floor)


def _test_local_seeded_assignment(
    partnership_s_corp_income: object,
    schedule_e_income: object,
    *,
    seed: int,
    evidence: dict[str, object],
    assumptions: dict[str, object],
) -> np.ndarray:
    passthrough = np.maximum(
        np.asarray(partnership_s_corp_income, dtype=np.float64),
        0.0,
    )
    schedule_e = np.asarray(schedule_e_income, dtype=np.float64)
    band = np.searchsorted(
        _TEST_LOCAL_SCHEDULE_E_CUTS,
        schedule_e,
        side="left",
    )
    prevalence, probabilities, quantiles, _share_means = _test_local_evidence_model(
        evidence
    )
    calibration = assumptions["calibration"]
    streams = assumptions["random_streams"]
    assert isinstance(calibration, dict) and isinstance(streams, dict)
    chance = _test_local_shifted_probability(
        prevalence,
        float(calibration["log_odds_shift"]),
    )
    presence = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                [seed, streams["family_entropy"], streams["presence_family"]]
            )
        )
    ).random(len(passthrough))
    share_draw = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                [seed, streams["family_entropy"], streams["share_family"]]
            )
        )
    ).random(len(passthrough))
    sampled_share = np.asarray(
        [
            np.interp(share_draw[row], probabilities, quantiles[band[row]])
            for row in range(len(passthrough))
        ],
        dtype=np.float64,
    )
    return np.where(
        (passthrough > 0.0) & (presence < chance[band]),
        passthrough * sampled_share,
        0.0,
    )


def _constant_evidence(
    *,
    prevalence: float,
    shares: list[float] | None = None,
    quantiles: tuple[float, float, float, float, float] | None = None,
) -> dict[str, object]:
    evidence = deepcopy(_test_local_resource_payload(QBI_PASSIVE_EVIDENCE_RESOURCE))
    if shares is None:
        shares = [0.5] * len(evidence["cells"])
    if len(shares) != len(evidence["cells"]):
        raise AssertionError("The fixture requires one share per income band.")
    names = ("q05", "q25", "q50", "q75", "q95")
    for cell, share in zip(evidence["cells"], shares, strict=True):
        cell["holding_prevalence"]["estimate"] = prevalence
        values = quantiles or (share,) * len(names)
        cell["conditional_share"]["selected_mean"] = float(np.mean(values))
        cell["conditional_share"]["selected_quantiles"] = dict(
            zip(names, values, strict=True)
        )
    return evidence


def _assumptions_with_shift(shift: float) -> dict[str, object]:
    assumptions = deepcopy(load_qbi_passive_passthrough_assumptions())
    assumptions["calibration"]["log_odds_shift"] = shift
    validate_qbi_passive_passthrough_assumptions(assumptions)
    return assumptions


def _frame(person: pd.DataFrame) -> Frame:
    person = person.copy(deep=True).reset_index(drop=True)
    count = len(person)
    ids = np.arange(1, count + 1, dtype=np.int64)
    person.insert(0, "person_id", ids)
    for entity in ("household", "tax_unit", "spm_unit", "family", "marital_unit"):
        person[f"person_{entity}_id"] = ids
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(count), WeightKind.DESIGN)},
    )


def _qbi_frame() -> Frame:
    count = 64
    positions = np.arange(count)
    columns: dict[str, np.ndarray] = {}
    for offset, name in enumerate(US_QBI_OUTPUT_COLUMNS):
        if name in US_QBI_BOOLEAN_OUTPUT_COLUMNS:
            columns[name] = (positions + offset) % 3 == 0
        else:
            columns[name] = positions.astype(np.float64) + offset + 0.125
    columns.update(
        {
            "partnership_income": np.where(positions % 3 == 0, 10_000.0, 0.0),
            "s_corp_income": np.where(positions % 5 == 0, 20_000.0, 0.0),
            "rental_income": positions.astype(np.float64) * 100.0,
            "estate_income": positions.astype(np.float64) * 10.0,
            "self_employment_income_before_lsr": (
                positions.astype(np.float64) + 1_000.0
            ),
        }
    )
    return _frame(pd.DataFrame(columns))


def test_packaged_assumptions_are_strict_and_pin_evidence_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assumptions = load_qbi_passive_passthrough_assumptions()
    evidence_bytes = (
        files("microcosm.build.us").joinpath(QBI_PASSIVE_EVIDENCE_RESOURCE).read_bytes()
    )

    assert (
        assumptions["evidence"]["sha256"] == hashlib.sha256(evidence_bytes).hexdigest()
    )
    assert assumptions["calibration"]["provisional_target"]["amount"] == (
        PROVISIONAL_TARGET
    )
    assert assumptions["random_streams"] == {
        "bit_generator": "PCG64",
        "root_seed_default": 0,
        "family_entropy": 4722,
        "presence_family": 0,
        "share_family": 1,
        "draw_policy": "full_length_before_support_masks",
    }

    extra_key = deepcopy(assumptions)
    extra_key["unreviewed"] = True
    with pytest.raises(ValueError, match="keys must be exactly"):
        validate_qbi_passive_passthrough_assumptions(extra_key)

    drifted_stream = deepcopy(assumptions)
    drifted_stream["random_streams"]["family_entropy"] += 1
    with pytest.raises(ValueError, match="random-stream contract drifted"):
        validate_qbi_passive_passthrough_assumptions(drifted_stream)

    out_of_bounds_shift = deepcopy(assumptions)
    out_of_bounds_shift["calibration"]["log_odds_shift"] = 31.0
    with pytest.raises(ValueError, match="outside its solver bounds"):
        validate_qbi_passive_passthrough_assumptions(out_of_bounds_shift)

    monkeypatch.setattr(passive_module, "_resource_sha256", lambda _name: "0" * 64)
    with pytest.raises(ValueError, match="evidence digest does not match"):
        load_qbi_passive_passthrough_assumptions()


def test_synthetic_income_bands_select_the_reviewed_share_cells() -> None:
    evidence = _constant_evidence(
        prevalence=1.0,
        shares=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    assumptions = _assumptions_with_shift(30.0)
    passthrough = np.full(11, 100.0)
    schedule_e = np.array(
        [
            -1.0,
            0.0,
            1.0,
            25_000.0,
            25_000.01,
            100_000.0,
            100_000.01,
            250_000.0,
            250_000.01,
            1_000_000.0,
            1_000_000.01,
        ]
    )

    assigned = assign_passive_partnership_s_corp_income(
        passthrough,
        schedule_e,
        evidence=evidence,
        assumptions=assumptions,
    )

    np.testing.assert_allclose(
        assigned,
        [0.0, 0.0, 20.0, 20.0, 40.0, 40.0, 60.0, 60.0, 80.0, 80.0, 100.0],
    )
    assert np.all(assigned >= 0.0)
    assert np.all(assigned <= passthrough)


@pytest.mark.parametrize(
    ("forms", "eligible"),
    [
        (
            [
                "partnership",
                "partnership_or_llc",
                "s_corporation",
                "sole_proprietorship",
                "",
            ],
            [True, True, True, False, False],
        ),
        ([1, 2, 3, 0, -1], [True, True, False, False, False]),
    ],
)
def test_latent_entity_form_only_routes_eligible_records(
    forms: list[object], eligible: list[bool]
) -> None:
    assigned = assign_passive_partnership_s_corp_income(
        np.full(5, 100.0),
        np.full(5, 50_000.0),
        latent_entity_form=forms,
        evidence=_constant_evidence(prevalence=1.0, shares=[1.0] * 6),
        assumptions=_assumptions_with_shift(30.0),
    )

    np.testing.assert_array_equal(assigned, np.asarray(eligible) * 100.0)


@pytest.mark.parametrize(
    ("passthrough", "schedule_e", "forms", "message"),
    [
        ([1.0, np.nan], [1.0, 1.0], None, "must be finite"),
        ([1.0, 1.0], [1.0, np.inf], None, "must be finite"),
        ([1.0, 1.0], [1.0, 1.0], [1.0, np.nan], "codes must be finite"),
    ],
)
def test_assignment_rejects_nonfinite_inputs(
    passthrough: list[float],
    schedule_e: list[float],
    forms: list[float] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        assign_passive_partnership_s_corp_income(
            passthrough,
            schedule_e,
            latent_entity_form=forms,
        )


def test_assignment_uses_independent_full_length_rng_families() -> None:
    count = 2_048
    seed = 19
    evidence = _constant_evidence(
        prevalence=0.5,
        quantiles=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    assumptions = _assumptions_with_shift(0.0)
    passthrough = np.ones(count)
    schedule_e = np.full(count, 50_000.0)
    assigned = assign_passive_partnership_s_corp_income(
        passthrough,
        schedule_e,
        seed=seed,
        evidence=evidence,
        assumptions=assumptions,
    )

    streams = assumptions["random_streams"]
    presence = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                [seed, streams["family_entropy"], streams["presence_family"]]
            )
        )
    ).random(count)
    share = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                [seed, streams["family_entropy"], streams["share_family"]]
            )
        )
    ).random(count)
    expected = np.where(
        presence < 0.5,
        np.interp(share, (0.05, 0.25, 0.5, 0.75, 0.95), (0, 0.25, 0.5, 0.75, 1)),
        0.0,
    )
    np.testing.assert_array_equal(assigned, expected)
    np.testing.assert_array_equal(
        assigned,
        assign_passive_partnership_s_corp_income(
            passthrough,
            schedule_e,
            seed=seed,
            evidence=evidence,
            assumptions=assumptions,
        ),
    )
    assert not np.array_equal(
        assigned,
        assign_passive_partnership_s_corp_income(
            passthrough,
            schedule_e,
            seed=seed + 1,
            evidence=evidence,
            assumptions=assumptions,
        ),
    )

    support = np.ones(count, dtype=bool)
    support[::4] = False
    masked_income = passthrough.copy()
    masked_income[~support] = 0.0
    support_masked = assign_passive_partnership_s_corp_income(
        masked_income,
        schedule_e,
        seed=seed,
        evidence=evidence,
        assumptions=assumptions,
    )
    form_masked = assign_passive_partnership_s_corp_income(
        passthrough,
        schedule_e,
        seed=seed,
        latent_entity_form=np.where(support, "partnership", "ineligible"),
        evidence=evidence,
        assumptions=assumptions,
    )
    np.testing.assert_array_equal(support_masked[support], assigned[support])
    np.testing.assert_array_equal(form_masked[support], assigned[support])
    assert np.count_nonzero(support_masked[~support]) == 0
    assert np.count_nonzero(form_masked[~support]) == 0


def test_current_qbi_pipeline_preserves_leaf_bytes_when_passive_realizes() -> None:
    frame = _qbi_frame()
    before = with_us_qbi_input_reconciliation(frame).table("person")
    qbi_bytes = {
        name: (before[name].dtype.str, before[name].to_numpy(copy=True).tobytes())
        for name in US_QBI_OUTPUT_COLUMNS
    }

    after = with_us_qbi_input_reconciliation(
        with_us_qbi_passive_passthrough_assignment(frame, seed=13)
    ).table("person")

    assert US_QBI_PASSIVE_PASSTHROUGH_OUTPUT_COLUMN not in before
    assert US_QBI_PASSIVE_PASSTHROUGH_OUTPUT_COLUMN in after
    for name, (dtype, raw) in qbi_bytes.items():
        assert after[name].dtype.str == dtype
        assert after[name].to_numpy(copy=False).tobytes() == raw
    passthrough = after["partnership_income"] + after["s_corp_income"]
    passive = after[US_QBI_PASSIVE_PASSTHROUGH_OUTPUT_COLUMN]
    assert np.flatnonzero(passive.to_numpy()).tolist() == [25, 36]
    assert passive.between(0.0, passthrough.clip(lower=0.0)).all()


def test_test_local_calibration_matches_hand_answer_and_production_solver() -> None:
    evidence = _constant_evidence(prevalence=0.5, shares=[0.5] * 6)
    assumptions = _test_local_resource_payload(QBI_PASSIVE_ASSUMPTIONS_RESOURCE)
    calibration = assumptions["calibration"]
    assert isinstance(calibration, dict)
    solver = calibration["solver"]
    assert isinstance(solver, dict)
    passthrough = np.array([100.0, 200.0])
    schedule_e = np.array([1.0, 1.0])
    weights = np.array([1.0, 2.0])
    hand_computed_shift = math.log(3.0)
    hand_computed_target = (100.0 * 1.0 + 200.0 * 2.0) * 0.5 * 0.75
    aggregate_tolerance = math.ulp(hand_computed_target)

    assert hand_computed_target == 187.5
    assert (
        abs(
            _test_local_expected_aggregate(
                passthrough,
                schedule_e,
                weights,
                evidence=evidence,
                log_odds_shift=hand_computed_shift,
            )
            - hand_computed_target
        )
        <= aggregate_tolerance
    )
    local_shift, local_achieved = _test_local_bisect_shift(
        passthrough,
        schedule_e,
        weights,
        evidence=evidence,
        target=hand_computed_target,
        solver=solver,
    )
    shift_tolerance = _test_local_bisection_tolerance(
        solver,
        reference_shift=hand_computed_shift,
    )

    production_shift, production_achieved = calibrate_qbi_passive_log_odds_shift(
        passthrough,
        schedule_e,
        weights,
        evidence=evidence,
        target=hand_computed_target,
    )

    assert abs(local_shift - hand_computed_shift) <= shift_tolerance
    assert abs(local_achieved - hand_computed_target) <= aggregate_tolerance
    assert abs(production_shift - local_shift) <= shift_tolerance
    assert abs(production_achieved - hand_computed_target) <= aggregate_tolerance


def test_assumptions_builder_is_deterministic_on_a_synthetic_replay(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    builder = _load_assumptions_builder()
    evidence = _constant_evidence(prevalence=0.5, shares=[0.5] * 6)
    evidence_path = tmp_path / QBI_PASSIVE_EVIDENCE_RESOURCE
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    count = 5_000
    passthrough_value = PROVISIONAL_TARGET / (count * 0.25)
    replay_path = tmp_path / "synthetic_replay.h5"
    ids = np.arange(1, count + 1, dtype=np.int64)
    with h5py.File(replay_path, mode="w") as replay:
        replay["tax_unit_id"] = ids
        replay["household_weight"] = np.ones(count)
        replay["person_tax_unit_id"] = ids
        replay["partnership_s_corp_income"] = np.full(count, passthrough_value)
        replay["rental_income"] = np.zeros(count)
        replay["estate_income"] = np.zeros(count)

    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    common = ["--puf-h5", str(replay_path), "--evidence", str(evidence_path)]
    assert builder.main([*common, "--output", str(first_output)]) == 0
    assert builder.main([*common, "--output", str(second_output)]) == 0

    assert first_output.read_bytes() == second_output.read_bytes()
    payload = json.loads(first_output.read_text(encoding="utf-8"))
    validate_qbi_passive_passthrough_assumptions(payload)
    assert payload["calibration"]["log_odds_shift"] == pytest.approx(0.0, abs=1e-12)
    assert payload["calibration"]["expected_aggregate"] == pytest.approx(
        PROVISIONAL_TARGET, abs=1.0
    )
    assert payload["calibration"]["seeded_replay"]["relative_error"] <= 0.05


@pytest.mark.skipif(
    not os.environ.get(PUF_REPLAY_ENVIRONMENT),
    reason=f"set {PUF_REPLAY_ENVIRONMENT} to run the restricted replay",
)
def test_restricted_replay_independently_resolves_persisted_shift_and_pins_artifact() -> (
    None
):
    builder = _load_assumptions_builder()
    replay_path = Path(os.environ[PUF_REPLAY_ENVIRONMENT]).expanduser()
    passthrough, schedule_e, weights, artifact = builder.read_replay_artifact(
        replay_path
    )
    assumptions = _test_local_resource_payload(QBI_PASSIVE_ASSUMPTIONS_RESOURCE)
    committed_artifact = assumptions["calibration"]["replay_artifact"]

    assert artifact == {
        "filename": "puf_2024.h5",
        "sha256": EXPECTED_REPLAY_SHA256,
        "bytes": EXPECTED_REPLAY_BYTES,
        "tax_unit_rows": EXPECTED_REPLAY_TAX_UNIT_ROWS,
    }
    assert committed_artifact["sha256"] == EXPECTED_REPLAY_SHA256
    assert committed_artifact["bytes"] == EXPECTED_REPLAY_BYTES
    assert committed_artifact["tax_unit_rows"] == EXPECTED_REPLAY_TAX_UNIT_ROWS
    assert committed_artifact["person_rows"] == EXPECTED_REPLAY_PERSON_ROWS

    evidence = _test_local_resource_payload(QBI_PASSIVE_EVIDENCE_RESOURCE)
    bounds = evidence["external_anchor"]["passive_passthrough_bounds"]
    target = (bounds["lower"]["amount"] + bounds["upper"]["amount"]) / 2.0
    persisted_calibration = assumptions["calibration"]
    solver = persisted_calibration["solver"]
    solved_shift, solved_expected = _test_local_bisect_shift(
        passthrough,
        schedule_e,
        weights,
        evidence=evidence,
        target=target,
        solver=solver,
    )
    shift_tolerance = _test_local_bisection_tolerance(
        solver,
        reference_shift=float(persisted_calibration["log_odds_shift"]),
    )

    assert target == PROVISIONAL_TARGET
    assert (
        abs(solved_shift - persisted_calibration["log_odds_shift"]) <= shift_tolerance
    )
    assert abs(solved_expected - target) <= 1e-3
    assert abs(solved_expected - persisted_calibration["expected_aggregate"]) <= 1e-3

    assigned = _test_local_seeded_assignment(
        passthrough,
        schedule_e,
        seed=persisted_calibration["seeded_replay"]["seed"],
        evidence=evidence,
        assumptions=assumptions,
    )
    achieved = float(np.dot(weights, assigned))

    assert abs(achieved - persisted_calibration["seeded_replay"]["aggregate"]) <= 1e-3
    assert (
        np.count_nonzero(assigned > 0.0)
        == (persisted_calibration["seeded_replay"]["positive_assigned_rows"])
    )
    assert abs(achieved / target - 1.0) == pytest.approx(
        persisted_calibration["seeded_replay"]["relative_error"],
        rel=0.0,
        abs=1e-15,
    )
