"""Reporter-anchored, Bernoulli-at-documented-prior SSI take-up tests."""

from __future__ import annotations

import copy
import importlib.util
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime.fiscal_targets import (
    SSA_SSI_AGE_BAND_RECIPIENTS_TARGET_ROLE,
)
from populace.build.us_runtime.ssi_take_up import (
    SSI_TAKE_UP_ARCHIVED_DERIVATION_URL,
    SSI_TAKE_UP_ARCHIVED_RANDOMNESS_URL,
    SSI_TAKE_UP_ARCHIVED_TARGETS_URL,
    SSI_TAKE_UP_SSA_SOURCE_URL,
    US_SSI_TAKE_UP_AGE_TARGETS,
    US_SSI_TAKE_UP_ANCHOR,
    US_SSI_TAKE_UP_BAND_DELIVERY_RELATIVE_TOLERANCE,
    US_SSI_TAKE_UP_ENFORCED_BAND_KEYS,
    US_SSI_TAKE_UP_OUTPUT_COLUMNS,
    US_SSI_TAKE_UP_PRIOR_BASIS_CURRENT_FRAME,
    US_SSI_TAKE_UP_PRIOR_BASIS_RELEASE_ARTIFACT,
    US_SSI_TAKE_UP_STAGE_NAME,
    SSITakeUpBandPriorBasis,
    SSITakeUpPriorBasis,
    _band_prior,
    _stable_source_draw,
    ssi_take_up_prior_basis_from_artifact,
    ssi_take_up_prior_basis_from_diagnostics,
    us_ssi_take_up_delivery_gate,
    us_ssi_take_up_diagnostics,
    us_ssi_take_up_gate,
    us_ssi_take_up_reporter_source_ids,
    us_ssi_take_up_stage_spec,
    with_us_ssi_take_up,
    write_us_ssi_take_up_diagnostics,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

_OUTPUT = US_SSI_TAKE_UP_OUTPUT_COLUMNS[0]
_TARGETS = {target.key: 50.0 for target in US_SSI_TAKE_UP_AGE_TARGETS}
_AGES = {"under_18": 12.0, "18_64": 40.0, "65_plus": 72.0}
# Fixture arithmetic per band: six dual-channel candidates weigh 20.0 each
# and the PUF-only candidate weighs 10.0 (capacity 130.0); the sole anchored
# candidate is source 0 (reporter floor 20.0).
_BAND_CAPACITY = 130.0
_REPORTER_FLOOR = 20.0
_UNSATURATED_PRIOR = (50.0 - _REPORTER_FLOOR) / (_BAND_CAPACITY - _REPORTER_FLOOR)
_ANCHORED_SOURCE_NUMBERS = {"0", "6"}


def _expected_bernoulli_flag(source_id: str, prior: float, *, seed: int = 17) -> bool:
    """The selection law: anchors unconditionally, else draw below prior."""

    if source_id.split(":")[1] in _ANCHORED_SOURCE_NUMBERS:
        return True
    return _stable_source_draw(source_id, seed=seed) < prior


_policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not _policyengine_us_installed,
    reason="requires the policyengine-us [us] extra",
)


def _frame(*, stale_output: bool = False) -> tuple[Frame, np.ndarray]:
    """Build three age bands with ASEC/PUF clones and PUF-only support."""

    rows: list[dict[str, object]] = []
    potential: list[float] = []
    person_id = 0
    for band, age in _AGES.items():
        for source_number in range(9):
            if source_number <= 5 or source_number == 8:
                channels = ("asec", "puf_tax_detail")
            elif source_number == 6:
                channels = ("asec",)
            else:
                channels = ("puf_tax_detail",)
            candidate = source_number <= 5 or source_number == 7
            reporter = source_number in {0, 6}
            for channel in channels:
                rows.append(
                    {
                        "person_id": person_id,
                        "person_household_id": person_id,
                        "person_tax_unit_id": person_id,
                        "person_spm_unit_id": person_id,
                        "person_family_id": person_id,
                        "person_marital_unit_id": person_id,
                        "age": age,
                        # PUF copies can carry SSI_VAL, but only direct ASEC
                        # rows are independent reporter anchors.
                        US_SSI_TAKE_UP_ANCHOR: 1_200.0 if reporter else 0.0,
                        "person_source_id": f"{band}:{source_number}",
                        "person_support_channel": channel,
                        _OUTPUT: bool((person_id % 2) == 0) if stale_output else False,
                    }
                )
                potential.append(100.0 if candidate else 0.0)
                person_id += 1

    person = pd.DataFrame(rows)
    ids = person["person_id"].to_numpy()
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
    }
    frame = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.full(len(person), 10.0),
                kind=WeightKind.DESIGN,
            )
        },
    )
    return frame, np.asarray(potential, dtype=np.float64)


def _replace_person(frame: Frame, person: pd.DataFrame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def _assigned(
    *,
    seed: int = 17,
    targets: dict[str, float] | None = None,
    stale_output: bool = False,
) -> tuple[Frame, Frame, np.ndarray, dict[str, object]]:
    frame, potential = _frame(stale_output=stale_output)
    result, diagnostics = with_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=seed,
        targets=targets or _TARGETS,
    )
    return frame, result, potential, diagnostics


def test_stage_contract_pins_archived_method_and_band_structure() -> None:
    spec = us_ssi_take_up_stage_spec()
    assert spec.stage == US_SSI_TAKE_UP_STAGE_NAME
    assert spec.source == SSI_TAKE_UP_SSA_SOURCE_URL
    assert spec.outputs == (_OUTPUT,)
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "assign_binary_from_rate",
    ]
    assignment = dict(spec.operations[1].parameters)
    assert assignment["rate_target_role"] == SSA_SSI_AGE_BAND_RECIPIENTS_TARGET_ROLE
    assert "uncapped_ssi > 0" in assignment["rate_derivation"]
    # populace#507/#508: the capacity basis may be a prior release's
    # delivered-weight diagnostics artifact, and the declared derivation
    # must say so.
    assert "delivered-weight" in assignment["rate_derivation"]
    # Recipient counts live in the ledger and bind via the calibration
    # registry (populace#469/#470) — the stage may never hardcode them.
    assert all("target_values" not in artifact for artifact in spec.artifacts)
    assert "42ed5d45" in SSI_TAKE_UP_ARCHIVED_DERIVATION_URL
    assert "cps.py#L650-L657" in SSI_TAKE_UP_ARCHIVED_DERIVATION_URL
    assert "takeup.py#L10-L35" in SSI_TAKE_UP_ARCHIVED_RANDOMNESS_URL
    assert "ssi_targets.py#L41-L74" in SSI_TAKE_UP_ARCHIVED_TARGETS_URL
    assert [
        (band.key, band.minimum_age, band.maximum_age)
        for band in US_SSI_TAKE_UP_AGE_TARGETS
    ] == [("under_18", None, 17), ("18_64", 18, 64), ("65_plus", 65, None)]


def test_assignment_preserves_asec_reporters_and_fans_source_decisions() -> None:
    _, result, _, diagnostics = _assigned()
    person = result.table("person")
    flag = person[_OUTPUT].to_numpy(dtype=bool)
    direct_reporter = (
        person["person_support_channel"].eq("asec")
        & person[US_SSI_TAKE_UP_ANCHOR].gt(0)
    ).to_numpy()
    assert flag[direct_reporter].all()
    assert diagnostics["reporter_anchor_lost_count"] == 0
    assert diagnostics["source_identity_mismatch_count"] == 0
    assert person.groupby("person_source_id")[_OUTPUT].nunique().max() == 1


def test_puf_only_ssi_value_is_not_promoted_to_reporter_anchor() -> None:
    frame, potential = _frame()
    baseline, baseline_diagnostics = with_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=17,
        targets={key: 20.0 for key in _TARGETS},
    )
    person = frame.table("person").copy()
    puf_only = person["person_source_id"].str.endswith(":7")
    person.loc[puf_only, US_SSI_TAKE_UP_ANCHOR] = 9_999.0
    result, diagnostics = with_us_ssi_take_up(
        _replace_person(frame, person),
        uncapped_ssi=potential,
        seed=17,
        targets={key: 20.0 for key in _TARGETS},
    )
    np.testing.assert_array_equal(
        result.table("person")[_OUTPUT], baseline.table("person")[_OUTPUT]
    )
    assert diagnostics["age_bands"] == baseline_diagnostics["age_bands"]


def test_reporter_lineage_survives_when_l0_keeps_only_the_puf_clone() -> None:
    full, potential = _frame()
    reporter_source_ids = us_ssi_take_up_reporter_source_ids(full)
    person = full.table("person")
    dropped = person["person_source_id"].eq("under_18:0") & person[
        "person_support_channel"
    ].eq("asec")
    sparse = full.select(~dropped.to_numpy())
    sparse_potential = potential[~dropped.to_numpy()]
    result, diagnostics = with_us_ssi_take_up(
        sparse,
        uncapped_ssi=sparse_potential,
        seed=17,
        targets=_TARGETS,
        reporter_source_ids=reporter_source_ids,
    )
    surviving_clone = result.table("person")["person_source_id"].eq("under_18:0")
    assert result.table("person").loc[surviving_clone, _OUTPUT].all()
    child = diagnostics["age_bands"][0]
    assert child["reporter_source_identity_count"] == 2


def test_non_candidate_reporter_remains_anchored_but_not_in_recipient_count() -> None:
    _, result, _, diagnostics = _assigned()
    person = result.table("person")
    noncandidate_reporter = person["person_source_id"].str.endswith(":6")
    assert person.loc[noncandidate_reporter, _OUTPUT].all()
    for band in diagnostics["age_bands"]:
        assert band["reporter_source_identity_count"] == 2
        assert band["reporter_candidate_floor"] == 20.0


def test_selection_is_anchors_union_of_seeded_draws_below_the_band_prior() -> None:
    _, result, _, diagnostics = _assigned()
    priors = {
        band["age_band"]: band["assignment_prior"] for band in diagnostics["age_bands"]
    }
    by_source = result.table("person").groupby("person_source_id")[_OUTPUT].first()
    for source_id, flagged in by_source.items():
        prior = priors[source_id.split(":")[0]]
        assert bool(flagged) == _expected_bernoulli_flag(source_id, prior)
    for band in diagnostics["age_bands"]:
        assert not band["saturated"]
        assert band["assignment_prior"] == pytest.approx(_UNSATURATED_PRIOR)
        # At assignment time the stored prior and the current-weight
        # recomputation coincide by construction.
        assert band["prior_recomputed_from_current_weights"] == pytest.approx(
            band["assignment_prior"]
        )
        assert band["assignment_prior_status"] == "floor_aware"
        assert band["prior_status_recomputed_from_current_weights"] == "floor_aware"
        assert band["drawable_candidate_capacity"] == pytest.approx(
            _BAND_CAPACITY - _REPORTER_FLOOR
        )
        assert band["prior_basis_drawable_candidate_capacity"] == pytest.approx(
            _BAND_CAPACITY - _REPORTER_FLOOR
        )
        assert band["empty_band"] is False
    assert diagnostics["bernoulli_law_violation_count"] == 0
    # Schema 4 adds floor-aware arithmetic plus explicit edge-regime fields;
    # pin the literal so reverting only the constant cannot pass.
    assert diagnostics["schema_version"] == 4
    assert diagnostics["prior_weight_basis"] == {
        "kind": US_SSI_TAKE_UP_PRIOR_BASIS_CURRENT_FRAME,
        "source_sha256": None,
        "source_schema_version": None,
    }
    for band in diagnostics["age_bands"]:
        # Without an artifact basis the prior basis IS this frame's
        # capacity/floor, so the documented arithmetic is auditable in place.
        assert band["prior_basis_candidate_capacity"] == pytest.approx(_BAND_CAPACITY)
        assert band["prior_basis_reporter_candidate_floor"] == pytest.approx(
            _REPORTER_FLOOR
        )
    assert us_ssi_take_up_gate(diagnostics, targets=_TARGETS).passed


def test_saturated_band_prior_falls_back_to_the_observed_reporter_rate() -> None:
    targets = {"under_18": 1_000.0, "18_64": 50.0, "65_plus": 50.0}
    _, result, _, diagnostics = _assigned(targets=targets)
    child, adult, aged = diagnostics["age_bands"]
    assert child["saturated"]
    assert child["assignment_prior"] == pytest.approx(_REPORTER_FLOOR / _BAND_CAPACITY)
    assert child["target_shortfall"] > 0
    assert not adult["saturated"]
    assert adult["assignment_prior"] == pytest.approx(_UNSATURATED_PRIOR)
    assert not aged["saturated"]
    by_source = result.table("person").groupby("person_source_id")[_OUTPUT].first()
    for source_id, flagged in by_source.items():
        if not source_id.startswith("under_18:"):
            continue
        assert bool(flagged) == _expected_bernoulli_flag(
            source_id, _REPORTER_FLOOR / _BAND_CAPACITY
        )
    assert us_ssi_take_up_gate(diagnostics, targets=targets).passed


def test_every_band_saturated_stays_nonconstant_and_passes_assignment_gate() -> None:
    """Universal saturation must not flag the whole pool.

    Build M's first sparse run died here: the restored disability battery put
    SSI candidates in every age band, every band's candidate capacity fell
    short of its SSA target, and a raw target/capacity prior degenerates to
    Bernoulli(1.0) — a constant, signal-free flag. The prior therefore falls
    back to the observed reporter rate (reporter mass over capacity) and the
    pool keeps signal. Candidates are no longer force-selected to chase the
    count (populace#469), so the assignment-integrity gate accepts the
    documented fallback. The separate delivery gate still rejects saturated
    support in enforced adult bands; only under-18 remains fenced.
    """

    targets = {"under_18": 1e6, "18_64": 1e6, "65_plus": 1e6}
    _, result, potential, diagnostics = _assigned(targets=targets)
    person = result.table("person")
    flag = person[_OUTPUT].to_numpy(dtype=bool)
    anchored = (
        person["person_source_id"]
        .str.split(":")
        .str[1]
        .isin(_ANCHORED_SOURCE_NUMBERS)
        .to_numpy()
    )
    assert flag[anchored].all()
    assert not flag[potential > 0].all()
    assert not flag.all()
    fallback = _REPORTER_FLOOR / _BAND_CAPACITY
    by_source = person.groupby("person_source_id")[_OUTPUT].first()
    for source_id, flagged in by_source.items():
        assert bool(flagged) == _expected_bernoulli_flag(source_id, fallback)
    for band in diagnostics["age_bands"]:
        assert band["saturated"]
        assert band["assignment_prior"] == pytest.approx(fallback)
        assert band["target_shortfall"] > 0
    assert us_ssi_take_up_gate(diagnostics, targets=targets).passed


def test_reporter_floor_above_target_never_drops_an_anchor() -> None:
    targets = {key: 5.0 for key in _TARGETS}
    _, result, _, diagnostics = _assigned(targets=targets)
    person = result.table("person")
    direct_reporter = person["person_support_channel"].eq("asec") & person[
        US_SSI_TAKE_UP_ANCHOR
    ].gt(0)
    assert person.loc[direct_reporter, _OUTPUT].all()
    assert diagnostics["reporter_anchor_lost_count"] == 0
    for band in diagnostics["age_bands"]:
        assert band["anchor_excess"] == pytest.approx(_REPORTER_FLOOR - 5.0)
        assert band["assignment_prior"] == 0.0
        assert band["assignment_prior_status"] == "reporter_floor_exceeds_target"
        assert (
            band["prior_status_recomputed_from_current_weights"]
            == "reporter_floor_exceeds_target"
        )
        assert band["target_shortfall"] == 0.0
        assert band["selected_recipient_weight"] == pytest.approx(_REPORTER_FLOOR)
        assert band["selected_recipient_weight"] >= band["reporter_candidate_floor"]
    assert us_ssi_take_up_gate(diagnostics, targets=targets).passed


def test_reporter_floor_equal_target_is_explicit_and_exact() -> None:
    targets = {key: _REPORTER_FLOOR for key in _TARGETS}
    _, result, _, diagnostics = _assigned(targets=targets)
    person = result.table("person")
    expected = (
        person["person_source_id"]
        .str.split(":")
        .str[1]
        .isin(_ANCHORED_SOURCE_NUMBERS)
        .to_numpy()
    )
    np.testing.assert_array_equal(person[_OUTPUT].to_numpy(dtype=bool), expected)
    for band in diagnostics["age_bands"]:
        assert band["assignment_prior"] == 0.0
        assert band["assignment_prior_status"] == "reporter_floor_meets_target"
        assert band["anchor_excess"] == 0.0
        assert band["target_shortfall"] == 0.0
        assert band["selected_recipient_weight"] == pytest.approx(_REPORTER_FLOOR)
    assert _band_prior(20.0, 20.0, 20.0) == 0.0
    assert us_ssi_take_up_gate(diagnostics, targets=targets).passed
    assert us_ssi_take_up_delivery_gate(diagnostics, targets=targets).passed


def test_reporter_floor_equal_target_and_capacity_passes_delivery_gate() -> None:
    frame, potential = _frame()
    person = frame.table("person")
    adult = person["person_source_id"].str.startswith("18_64:")
    anchored_candidate = person["person_source_id"].eq("18_64:0")
    potential[(adult & ~anchored_candidate).to_numpy()] = 0.0
    targets = {key: _REPORTER_FLOOR for key in _TARGETS}

    _, diagnostics = with_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=17,
        targets=targets,
    )
    row = diagnostics["age_bands"][1]
    assert row["candidate_capacity"] == pytest.approx(_REPORTER_FLOOR)
    assert row["reporter_candidate_floor"] == pytest.approx(_REPORTER_FLOOR)
    assert row["selected_recipient_weight"] == pytest.approx(_REPORTER_FLOOR)
    assert row["assignment_prior"] == 0.0
    assert row["assignment_prior_status"] == "reporter_floor_meets_target"

    delivery = us_ssi_take_up_delivery_gate(diagnostics, targets=targets)
    assert delivery.passed
    assert delivery.details["prior_weight_basis_retry_applicable"] is False


def test_delivery_gate_fails_reporter_floor_excess_inside_tolerance() -> None:
    """A small impossible excess is a support defect, not tolerated draw noise."""

    targets = {"under_18": 20.0, "18_64": 19.5, "65_plus": 20.0}
    _, _, _, diagnostics = _assigned(targets=targets)
    adult = diagnostics["age_bands"][1]
    assert adult["assignment_prior_status"] == "reporter_floor_exceeds_target"
    assert (adult["selected_recipient_weight"] - adult["target"]) / adult[
        "target"
    ] < US_SSI_TAKE_UP_BAND_DELIVERY_RELATIVE_TOLERANCE

    gate = us_ssi_take_up_delivery_gate(diagnostics, targets=targets)
    assert not gate.passed
    assert any(
        "18_64" in failure
        and "Forced reporters alone over-deliver" in failure
        and "reporter identification/support" in failure
        for failure in gate.failures
    )
    assert not any(
        "18_64" in failure and "--ssi-take-up-prior-weight-basis" in failure
        for failure in gate.failures
    )
    assert gate.details["prior_weight_basis_retry_applicable"] is False


def test_zero_candidate_capacity_is_named_and_fails_the_gate() -> None:
    frame, potential = _frame()
    child = frame.table("person")["person_source_id"].str.startswith("under_18:")
    potential[child.to_numpy()] = 0.0
    _, diagnostics = with_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
    )
    row = diagnostics["age_bands"][0]
    assert row["empty_band"] is False
    assert row["source_identity_count"] > 0
    assert row["candidate_capacity"] == 0.0
    assert row["drawable_candidate_capacity"] == 0.0
    assert row["assignment_prior"] == 0.0
    assert row["assignment_prior_status"] == "zero_candidate_capacity"
    assert row["prior_status_recomputed_from_current_weights"] == (
        "zero_candidate_capacity"
    )
    assert row["target_shortfall"] == pytest.approx(_TARGETS["under_18"])
    gate = us_ssi_take_up_gate(diagnostics, targets=_TARGETS)
    assert not gate.passed
    assert any("zero candidate capacity" in failure for failure in gate.failures)


def test_empty_age_band_is_named_and_fails_the_gate() -> None:
    frame, potential = _frame()
    keep = ~frame.table("person")["person_source_id"].str.startswith("under_18:")
    sparse = frame.select(keep.to_numpy())
    _, diagnostics = with_us_ssi_take_up(
        sparse,
        uncapped_ssi=potential[keep.to_numpy()],
        seed=17,
        targets=_TARGETS,
    )
    row = diagnostics["age_bands"][0]
    assert row["empty_band"] is True
    assert row["source_identity_count"] == 0
    assert row["candidate_source_identity_count"] == 0
    assert row["candidate_capacity"] == 0.0
    assert row["assignment_prior_status"] == "zero_candidate_capacity"
    gate = us_ssi_take_up_gate(diagnostics, targets=_TARGETS)
    assert not gate.passed
    assert any("zero candidate capacity" in failure for failure in gate.failures)


def test_no_drawable_candidate_capacity_is_named() -> None:
    frame, potential = _frame()
    person = frame.table("person")
    adult = person["person_source_id"].str.startswith("18_64:")
    anchored_candidate = person["person_source_id"].eq("18_64:0")
    potential[(adult & ~anchored_candidate).to_numpy()] = 0.0
    result, diagnostics = with_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
    )
    row = diagnostics["age_bands"][1]
    assert row["candidate_capacity"] == pytest.approx(_REPORTER_FLOOR)
    assert row["reporter_candidate_floor"] == pytest.approx(_REPORTER_FLOOR)
    assert row["drawable_candidate_capacity"] == 0.0
    assert row["assignment_prior"] == 0.0
    assert row["assignment_prior_status"] == "no_drawable_candidate_capacity"
    assert row["saturated"] is True
    assert row["target_shortfall"] == pytest.approx(30.0)
    adult_flags = (
        result.table("person").loc[adult].groupby("person_source_id")[_OUTPUT].first()
    )
    assert set(adult_flags.loc[adult_flags].index) == {"18_64:0", "18_64:6"}
    assert us_ssi_take_up_gate(diagnostics, targets=_TARGETS).passed
    delivered = copy.deepcopy(diagnostics)
    delivered["age_bands"][2]["selected_recipient_weight"] = 50.0
    delivery = us_ssi_take_up_delivery_gate(delivered, targets=_TARGETS)
    assert not delivery.passed
    assert any(
        "18_64" in failure and "no drawable candidate mass" in failure
        for failure in delivery.failures
    )
    assert delivery.details["prior_weight_basis_retry_applicable"] is False


def test_target_equal_capacity_uses_explicit_saturated_fallback() -> None:
    targets = {"under_18": _BAND_CAPACITY, "18_64": 50.0, "65_plus": 50.0}
    _, _, _, diagnostics = _assigned(targets=targets)
    child = diagnostics["age_bands"][0]
    assert child["saturated"] is True
    assert child["assignment_prior_status"] == "saturated_reporter_rate"
    assert child["assignment_prior"] == pytest.approx(_REPORTER_FLOOR / _BAND_CAPACITY)
    assert us_ssi_take_up_gate(diagnostics, targets=targets).passed


def test_assignment_is_deterministic_source_keyed_and_seed_sensitive() -> None:
    _, first, _, first_diagnostics = _assigned(seed=17)
    _, repeat, _, repeat_diagnostics = _assigned(seed=17)
    _, alternative, _, _ = _assigned(seed=18)
    np.testing.assert_array_equal(
        first.table("person")[_OUTPUT], repeat.table("person")[_OUTPUT]
    )
    assert first_diagnostics == repeat_diagnostics
    assert not np.array_equal(
        first.table("person")[_OUTPUT], alternative.table("person")[_OUTPUT]
    )


def test_fixed_seed_selections_are_monotone_in_floor_aware_prior() -> None:
    low_targets = {key: 35.0 for key in _TARGETS}
    high_targets = {key: 80.0 for key in _TARGETS}
    _, low, _, low_diagnostics = _assigned(seed=17, targets=low_targets)
    _, high, _, high_diagnostics = _assigned(seed=17, targets=high_targets)

    low_selected = set(
        low.table("person")
        .groupby("person_source_id")[_OUTPUT]
        .first()
        .loc[lambda flag: flag]
        .index
    )
    high_selected = set(
        high.table("person")
        .groupby("person_source_id")[_OUTPUT]
        .first()
        .loc[lambda flag: flag]
        .index
    )
    assert low_selected <= high_selected
    low_priors = {
        row["age_band"]: row["assignment_prior"] for row in low_diagnostics["age_bands"]
    }
    high_priors = {
        row["age_band"]: row["assignment_prior"]
        for row in high_diagnostics["age_bands"]
    }
    assert all(low_priors[key] < high_priors[key] for key in low_priors)


def test_stale_output_is_healed_and_exact_rerun_returns_same_frame() -> None:
    _, result, potential, diagnostics = _assigned(stale_output=True)
    again, again_diagnostics = with_us_ssi_take_up(
        result,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
    )
    assert again is result
    assert again_diagnostics == diagnostics
    assert us_ssi_take_up_gate(again_diagnostics, targets=_TARGETS).passed


def test_matching_numeric_output_is_rewritten_to_canonical_boolean() -> None:
    _, result, potential, _ = _assigned()
    person = result.table("person").copy()
    person[_OUTPUT] = person[_OUTPUT].astype(np.float64)
    numeric = _replace_person(result, person)
    healed, diagnostics = with_us_ssi_take_up(
        numeric,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
    )
    assert healed is not numeric
    assert pd.api.types.is_bool_dtype(healed.table("person")[_OUTPUT])
    assert us_ssi_take_up_gate(diagnostics, targets=_TARGETS).passed


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_column", "missing person source"),
        ("unknown_channel", "exact ASEC/PUF"),
        ("missing_channel", "complete support provenance"),
        ("cross_age_band", "cross SSA age bands"),
        ("blank_source", "nonblank"),
        ("duplicate_channel", "at most one row"),
    ],
)
def test_source_provenance_failures_are_rejected(mutation: str, message: str) -> None:
    frame, potential = _frame()
    person = frame.table("person").copy()
    if mutation == "missing_column":
        person = person.drop(columns=["person_source_id"])
    elif mutation == "unknown_channel":
        person.loc[person.index[0], "person_support_channel"] = "synthetic"
    elif mutation == "missing_channel":
        person.loc[person.index[0], "person_support_channel"] = None
    elif mutation == "cross_age_band":
        adult = person["person_source_id"].eq("18_64:7")
        person.loc[adult, "person_source_id"] = "under_18:6"
    elif mutation == "blank_source":
        person.loc[person.index[0], "person_source_id"] = "   "
    else:
        person.loc[person.index[1], "person_support_channel"] = "asec"
    with pytest.raises(ValueError, match=message):
        with_us_ssi_take_up(
            _replace_person(frame, person),
            uncapped_ssi=potential,
            seed=17,
            targets=_TARGETS,
        )


@pytest.mark.parametrize(
    ("field", "value", "failure_fragment"),
    [
        ("schema_version", 999, "schema version"),
        ("target_source", "https://example.org", "target source"),
        ("candidate_definition", "is_ssi_eligible", "candidate definition"),
        ("unique_count", 1, "constant"),
        ("reporter_anchor_lost_count", 1, "reporter anchors"),
        ("source_identity_mismatch_count", 1, "source identity"),
        ("bernoulli_law_violation_count", 2, "Bernoulli law"),
        ("prior_weight_basis", {"kind": "handwave"}, "prior weight basis"),
    ],
)
def test_gate_rejects_tampered_top_level_diagnostics(
    field: str, value: object, failure_fragment: str
) -> None:
    _, _, _, diagnostics = _assigned()
    tampered = copy.deepcopy(diagnostics)
    tampered[field] = value
    gate = us_ssi_take_up_gate(tampered, targets=_TARGETS)
    assert not gate.passed
    assert any(failure_fragment in failure for failure in gate.failures)


def test_gate_rejects_hidden_saturation_and_corrupted_band_arithmetic() -> None:
    _, _, _, diagnostics = _assigned()
    hidden = copy.deepcopy(diagnostics)
    hidden["age_bands"][0]["saturated"] = True
    hidden_gate = us_ssi_take_up_gate(hidden, targets=_TARGETS)
    assert not hidden_gate.passed
    assert any("saturation status" in failure for failure in hidden_gate.failures)

    escaped = copy.deepcopy(diagnostics)
    escaped["age_bands"][0]["selected_recipient_weight"] = 0.0
    escaped_gate = us_ssi_take_up_gate(escaped, targets=_TARGETS)
    assert not escaped_gate.passed
    assert any("envelope" in failure for failure in escaped_gate.failures)

    miscomputed = copy.deepcopy(diagnostics)
    miscomputed["age_bands"][0]["prior_recomputed_from_current_weights"] = 0.9
    miscomputed_gate = us_ssi_take_up_gate(miscomputed, targets=_TARGETS)
    assert not miscomputed_gate.passed
    assert any("recomputed prior" in failure for failure in miscomputed_gate.failures)

    invalid_probability = copy.deepcopy(diagnostics)
    invalid_probability["age_bands"][0]["assignment_prior"] = 1.5
    invalid_gate = us_ssi_take_up_gate(invalid_probability, targets=_TARGETS)
    assert not invalid_gate.passed
    assert any("outside [0, 1]" in failure for failure in invalid_gate.failures)

    hidden_status = copy.deepcopy(diagnostics)
    hidden_status["age_bands"][0]["assignment_prior_status"] = (
        "reporter_floor_exceeds_target"
    )
    status_gate = us_ssi_take_up_gate(hidden_status, targets=_TARGETS)
    assert not status_gate.passed
    assert any("assignment prior status" in failure for failure in status_gate.failures)

    invalid_count = copy.deepcopy(diagnostics)
    invalid_count["age_bands"][0]["source_identity_count"] = None
    count_gate = us_ssi_take_up_gate(invalid_count, targets=_TARGETS)
    assert not count_gate.passed
    assert any("source identity count" in failure for failure in count_gate.failures)


def test_gate_rejects_duplicate_age_band_diagnostics() -> None:
    _, _, _, diagnostics = _assigned()
    duplicated = copy.deepcopy(diagnostics)
    duplicated["age_bands"].append(copy.deepcopy(duplicated["age_bands"][0]))
    gate = us_ssi_take_up_gate(duplicated, targets=_TARGETS)
    assert not gate.passed
    assert any("exactly one diagnostic row" in failure for failure in gate.failures)


def test_existing_assignment_diagnostics_do_not_reassign_flags() -> None:
    _, result, potential, stage_diagnostics = _assigned()
    stage_priors = {
        band["age_band"]: band["assignment_prior"]
        for band in stage_diagnostics["age_bands"]
    }
    original = result.table("person")[_OUTPUT].to_numpy(dtype=bool).copy()
    candidate_selected = original & (potential > 0)
    drifted_weights = np.where(candidate_selected, 100.0, 1.0)
    reweighted = Frame(
        {entity: result.table(entity).copy() for entity in result.entities},
        result.schema,
        {
            "household": Weights(
                drifted_weights,
                WeightKind.CALIBRATED,
            )
        },
        result.strata,
        mass_log=result.mass_log,
    )
    diagnostics = us_ssi_take_up_diagnostics(
        reweighted,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
        assignment_priors=stage_priors,
        prior_basis=ssi_take_up_prior_basis_from_diagnostics(stage_diagnostics),
    )
    np.testing.assert_array_equal(reweighted.table("person")[_OUTPUT], original)
    # Weight drift pushes the measured recipient mass far off target, and the
    # gate still passes: the count miss is calibration's residual
    # (populace#469/#470), reported in the scorecard, never a module failure.
    # The published assignment prior stays the one that generated the frozen
    # flags — never recomputed from the drifted weights — while the
    # current-weight recomputation is reported separately and the flags
    # re-verify against the seeded law exactly.
    assert diagnostics["bernoulli_law_violation_count"] == 0
    for band in diagnostics["age_bands"]:
        assert band["selected_recipient_weight"] > band["target"]
        assert band["assignment_prior"] == pytest.approx(stage_priors[band["age_band"]])
        assert band["prior_recomputed_from_current_weights"] != pytest.approx(
            band["assignment_prior"]
        )
    assert us_ssi_take_up_gate(diagnostics, targets=_TARGETS).passed

    recomputed, _ = with_us_ssi_take_up(
        reweighted,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
    )
    assert not np.array_equal(recomputed.table("person")[_OUTPUT], original)


def test_gate_rejects_persisted_flags_that_break_the_bernoulli_law() -> None:
    _, result, potential, stage_diagnostics = _assigned()
    stage_priors = {
        band["age_band"]: band["assignment_prior"]
        for band in stage_diagnostics["age_bands"]
    }
    person = result.table("person").copy()
    # A non-anchored, non-candidate source with both support clones: flipping
    # its flag keeps source-identity consistency and every anchor intact,
    # so only the seeded-law recheck can catch the corruption.
    flipped = person["person_source_id"].eq("18_64:8")
    assert flipped.sum() == 2
    person.loc[flipped, _OUTPUT] = ~person.loc[flipped, _OUTPUT].astype(bool)
    corrupted = _replace_person(result, person)
    diagnostics = us_ssi_take_up_diagnostics(
        corrupted,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
        assignment_priors=stage_priors,
        prior_basis=ssi_take_up_prior_basis_from_diagnostics(stage_diagnostics),
    )
    assert diagnostics["bernoulli_law_violation_count"] == 1
    assert diagnostics["reporter_anchor_lost_count"] == 0
    assert diagnostics["source_identity_mismatch_count"] == 0
    gate = us_ssi_take_up_gate(diagnostics, targets=_TARGETS)
    assert not gate.passed
    assert any("Bernoulli law" in failure for failure in gate.failures)


# --- Delivered-weight prior basis + hard delivery gate (populace#507/#508) ---

_BASIS_SHA = "ab" * 32


def _artifact_basis(
    *,
    capacities: dict[str, float],
    floors: dict[str, float] | None = None,
) -> SSITakeUpPriorBasis:
    resolved_floors = floors or {key: _REPORTER_FLOOR for key in capacities}
    return SSITakeUpPriorBasis(
        kind=US_SSI_TAKE_UP_PRIOR_BASIS_RELEASE_ARTIFACT,
        bands=tuple(
            SSITakeUpBandPriorBasis(
                key=band.key,
                candidate_capacity=float(capacities[band.key]),
                reporter_candidate_floor=float(resolved_floors[band.key]),
            )
            for band in US_SSI_TAKE_UP_AGE_TARGETS
        ),
        source_sha256=_BASIS_SHA,
        source_schema_version=2,
    )


def test_enforced_bands_are_the_adult_bands_pending_child_disability_stage() -> None:
    """The under-18 band is honestly fenced until populace#453/#509 lands.

    Build N's under-18 candidate capacity was 177,582 against the 1,001,922
    ledger target — no seeding basis can truthfully reconcile that band, and
    treating saturation as success would defeat the delivery gate. Flipping
    this roster is a deliberate act for the child-disability lane, not a
    side effect.
    """

    assert US_SSI_TAKE_UP_ENFORCED_BAND_KEYS == ("18_64", "65_plus")
    assert 0.0 < US_SSI_TAKE_UP_BAND_DELIVERY_RELATIVE_TOLERANCE < 0.1


def test_release_artifact_loader_basis_drives_floor_aware_band_priors() -> None:
    capacities = {"under_18": 200.0, "18_64": 100.0, "65_plus": 500.0}
    _, _, _, payload = _assigned()
    for row in payload["age_bands"]:
        key = row["age_band"]
        row["candidate_capacity"] = capacities[key]
        row["reporter_candidate_floor"] = _REPORTER_FLOOR
    basis = ssi_take_up_prior_basis_from_artifact(
        payload,
        targets=_TARGETS,
        source_sha256=_BASIS_SHA,
    )
    frame, potential = _frame()
    result, diagnostics = with_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
        prior_basis=basis,
    )
    expected = {
        key: (50.0 - _REPORTER_FLOOR) / (value - _REPORTER_FLOOR)
        for key, value in capacities.items()
    }
    by_source = result.table("person").groupby("person_source_id")[_OUTPUT].first()
    for source_id, flagged in by_source.items():
        prior = expected[source_id.split(":")[0]]
        assert bool(flagged) == _expected_bernoulli_flag(source_id, prior)
    assert diagnostics["prior_weight_basis"] == {
        "kind": US_SSI_TAKE_UP_PRIOR_BASIS_RELEASE_ARTIFACT,
        "source_sha256": _BASIS_SHA,
        "source_schema_version": 4,
    }
    for band in diagnostics["age_bands"]:
        key = band["age_band"]
        assert band["assignment_prior"] == pytest.approx(expected[key])
        assert band["prior_basis_candidate_capacity"] == pytest.approx(capacities[key])
        assert band["prior_basis_reporter_candidate_floor"] == pytest.approx(
            _REPORTER_FLOOR
        )
        # The current-weight recomputation still measures THIS frame, so the
        # release diagnostics never misdocument where the priors came from.
        assert band["prior_recomputed_from_current_weights"] == pytest.approx(
            _UNSATURATED_PRIOR
        )
    assert diagnostics["bernoulli_law_violation_count"] == 0
    assert us_ssi_take_up_gate(diagnostics, targets=_TARGETS).passed


def test_saturated_artifact_basis_falls_back_to_the_basis_reporter_rate() -> None:
    basis = _artifact_basis(
        capacities={"under_18": 40.0, "18_64": 100.0, "65_plus": 100.0}
    )
    frame, potential = _frame()
    result, diagnostics = with_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
        prior_basis=basis,
    )
    # target 50 over basis capacity 40 saturates, so the prior falls back to
    # the basis reporter rate 20/40 — never Bernoulli(1.0).
    child = diagnostics["age_bands"][0]
    assert child["age_band"] == "under_18"
    assert child["assignment_prior"] == pytest.approx(0.5)
    by_source = result.table("person").groupby("person_source_id")[_OUTPUT].first()
    for source_id, flagged in by_source.items():
        if source_id.startswith("under_18:"):
            assert bool(flagged) == _expected_bernoulli_flag(source_id, 0.5)
    assert us_ssi_take_up_gate(diagnostics, targets=_TARGETS).passed


def test_attempt_5_floor_aware_prior_hits_expected_target() -> None:
    """Pin the exact attempt-5 inputs that exposed the forced-floor defect."""

    target = 3_905_779.0
    capacity = 4_840_938.516496049
    floor = 2_169_544.5101818806
    prior = _band_prior(target, capacity, floor)

    assert prior == pytest.approx(0.6499358, rel=0.0, abs=5e-8)
    assert prior == pytest.approx(0.6499357585269396, rel=0.0, abs=1e-15)
    expected_selected = floor + prior * (capacity - floor)
    assert expected_selected == pytest.approx(target, rel=0.0, abs=1e-6)


def test_attempt_5_seeded_delivery_removes_floor_blind_overshoot() -> None:
    """The same one-shot draws land inside the gate only with the fixed law."""

    target = 3_905_779.0
    capacity = 4_840_938.516496049
    floor = 2_169_544.5101818806
    prior = _band_prior(target, capacity, floor)
    old_prior = target / capacity

    old_expected = floor + old_prior * (capacity - floor)
    old_expected_relative_error = (old_expected - target) / target
    assert old_expected == pytest.approx(4_324_885.7885, rel=0.0, abs=1e-4)
    assert old_expected_relative_error == pytest.approx(
        0.1073043,
        rel=0.0,
        abs=1e-7,
    )

    draws = np.asarray(
        [
            _stable_source_draw(
                f"attempt5:nonreporter:{source_number}",
                seed=17,
            )
            for source_number in range(1_000)
        ]
    )
    nonreporter_weight = (capacity - floor) / len(draws)
    fixed_selected = floor + float(np.count_nonzero(draws < prior)) * (
        nonreporter_weight
    )
    old_selected = floor + float(np.count_nonzero(draws < old_prior)) * (
        nonreporter_weight
    )
    tolerance = US_SSI_TAKE_UP_BAND_DELIVERY_RELATIVE_TOLERANCE
    assert abs(fixed_selected - target) <= tolerance * target
    assert abs(old_selected - target) > tolerance * target

    targets = {"under_18": 50.0, "18_64": target, "65_plus": 50.0}

    def delivery_payload(
        adult_selected: float, adult_assignment_prior: float
    ) -> dict[str, object]:
        # The recomputation always applies the fixed law to the current
        # weights, so an artifact whose flags were drawn under the old
        # floor-blind prior records assignment_prior ≠ recomputation — the
        # exact condition under which a basis retry genuinely changes the
        # thresholds (PR #554 review finding 1).
        return {
            "schema_version": 4,
            "age_bands": [
                {
                    "age_band": "under_18",
                    "selected_recipient_weight": 50.0,
                    "candidate_capacity": 100.0,
                    "reporter_candidate_floor": 0.0,
                    "empty_band": False,
                    "assignment_prior": 0.5,
                    "prior_recomputed_from_current_weights": 0.5,
                    "prior_status_recomputed_from_current_weights": "floor_aware",
                },
                {
                    "age_band": "18_64",
                    "selected_recipient_weight": adult_selected,
                    "candidate_capacity": capacity,
                    "reporter_candidate_floor": floor,
                    "empty_band": False,
                    "assignment_prior": adult_assignment_prior,
                    "prior_recomputed_from_current_weights": prior,
                    "prior_status_recomputed_from_current_weights": "floor_aware",
                },
                {
                    "age_band": "65_plus",
                    "selected_recipient_weight": 50.0,
                    "candidate_capacity": 100.0,
                    "reporter_candidate_floor": 0.0,
                    "empty_band": False,
                    "assignment_prior": 0.5,
                    "prior_recomputed_from_current_weights": 0.5,
                    "prior_status_recomputed_from_current_weights": "floor_aware",
                },
            ],
        }

    assert us_ssi_take_up_delivery_gate(
        delivery_payload(fixed_selected, prior),
        targets=targets,
    ).passed
    old_gate = us_ssi_take_up_delivery_gate(
        delivery_payload(old_selected, old_prior),
        targets=targets,
    )
    assert not old_gate.passed
    assert old_gate.details["prior_weight_basis_retry_applicable"] is True


@pytest.mark.parametrize(
    ("capacity", "floor", "message"),
    [
        (-1.0, 0.0, "candidate capacity"),
        (40.0, 41.0, "negative drawable candidate capacity"),
    ],
)
def test_band_prior_rejects_invalid_or_negative_drawable_capacity(
    capacity: float,
    floor: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _band_prior(50.0, capacity, floor)


def test_prior_basis_round_trips_through_diagnostics() -> None:
    _, result, potential, stage_diagnostics = _assigned()
    recovered = ssi_take_up_prior_basis_from_diagnostics(stage_diagnostics)
    assert recovered.kind == US_SSI_TAKE_UP_PRIOR_BASIS_CURRENT_FRAME
    assert recovered.source_sha256 is None
    for band in recovered.bands:
        assert band.candidate_capacity == pytest.approx(_BAND_CAPACITY)
        assert band.reporter_candidate_floor == pytest.approx(_REPORTER_FLOOR)
    stage_priors = {
        band["age_band"]: band["assignment_prior"]
        for band in stage_diagnostics["age_bands"]
    }
    final = us_ssi_take_up_diagnostics(
        result,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
        assignment_priors=stage_priors,
        prior_basis=recovered,
    )
    assert final["prior_weight_basis"] == stage_diagnostics["prior_weight_basis"]
    assert ssi_take_up_prior_basis_from_diagnostics(final) == recovered

    artifact = _artifact_basis(
        capacities={"under_18": 200.0, "18_64": 100.0, "65_plus": 500.0}
    )
    frame, potential = _frame()
    _, artifact_diagnostics = with_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
        prior_basis=artifact,
    )
    assert ssi_take_up_prior_basis_from_diagnostics(artifact_diagnostics) == artifact


def test_prior_basis_loader_accepts_schema_4_3_and_2_artifacts() -> None:
    _, _, _, diagnostics = _assigned()
    payload4 = copy.deepcopy(diagnostics)
    basis4 = ssi_take_up_prior_basis_from_artifact(
        payload4, targets=_TARGETS, source_sha256=_BASIS_SHA
    )
    assert basis4.kind == US_SSI_TAKE_UP_PRIOR_BASIS_RELEASE_ARTIFACT
    assert basis4.source_sha256 == _BASIS_SHA
    assert basis4.source_schema_version == 4
    for band in basis4.bands:
        assert band.candidate_capacity == pytest.approx(_BAND_CAPACITY)
        assert band.reporter_candidate_floor == pytest.approx(_REPORTER_FLOOR)

    # Schema-3 artifacts carry the same usable capacity/floor measurements;
    # only the prior they recorded was floor-blind.
    payload3 = copy.deepcopy(diagnostics)
    payload3["schema_version"] = 3
    basis3 = ssi_take_up_prior_basis_from_artifact(
        payload3, targets=_TARGETS, source_sha256=_BASIS_SHA
    )
    assert basis3.kind == US_SSI_TAKE_UP_PRIOR_BASIS_RELEASE_ARTIFACT
    assert basis3.source_sha256 == _BASIS_SHA
    assert basis3.source_schema_version == 3
    # Per-band capacity/floor must survive the schema-3 load byte-for-byte
    # (PR #554 review finding 2: a mutation adding 1.0 to every loaded
    # schema-3 capacity passed when only kind/hash/version were pinned).
    assert [
        (band.key, band.candidate_capacity, band.reporter_candidate_floor)
        for band in basis3.bands
    ] == [
        (band.key, band.candidate_capacity, band.reporter_candidate_floor)
        for band in basis4.bands
    ]
    # Build N's certified artifact predates the basis fields (schema 2); the
    # chain must be able to start from it (populace#507 Build O attempt 2).
    payload2 = copy.deepcopy(diagnostics)
    payload2["schema_version"] = 2
    payload2.pop("prior_weight_basis")
    for band in payload2["age_bands"]:
        band.pop("prior_basis_candidate_capacity")
        band.pop("prior_basis_reporter_candidate_floor")
    basis2 = ssi_take_up_prior_basis_from_artifact(
        payload2, targets=_TARGETS, source_sha256=_BASIS_SHA
    )
    assert basis2.source_schema_version == 2
    assert [
        (band.key, band.candidate_capacity, band.reporter_candidate_floor)
        for band in basis2.bands
    ] == [
        (band.key, band.candidate_capacity, band.reporter_candidate_floor)
        for band in basis4.bands
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unknown_schema", "schema version"),
        ("wrong_table", "target table"),
        ("wrong_period", "target period"),
        ("target_contract_drift", "target contract"),
        ("missing_band", "age band"),
        ("duplicate_band", "age band"),
        ("zero_enforced_capacity", "candidate capacity"),
        ("floor_above_capacity", "reporter floor"),
        ("nonfinite_capacity", "candidate capacity"),
        ("blank_sha", "sha256"),
    ],
)
def test_prior_basis_loader_rejects_invalid_artifacts(
    mutation: str, message: str
) -> None:
    _, _, _, diagnostics = _assigned()
    payload = copy.deepcopy(diagnostics)
    sha = _BASIS_SHA
    if mutation == "unknown_schema":
        payload["schema_version"] = 1
    elif mutation == "wrong_table":
        payload["target_table"] = "another_table"
    elif mutation == "wrong_period":
        payload["target_period"] = "2019-12"
    elif mutation == "target_contract_drift":
        payload["age_bands"][2]["target"] = 999.0
    elif mutation == "missing_band":
        payload["age_bands"] = payload["age_bands"][:2]
    elif mutation == "duplicate_band":
        payload["age_bands"].append(copy.deepcopy(payload["age_bands"][0]))
    elif mutation == "zero_enforced_capacity":
        payload["age_bands"][2]["candidate_capacity"] = 0.0
    elif mutation == "floor_above_capacity":
        payload["age_bands"][1]["reporter_candidate_floor"] = 1_000.0
    elif mutation == "nonfinite_capacity":
        payload["age_bands"][1]["candidate_capacity"] = float("inf")
    else:
        sha = "   "
    with pytest.raises(ValueError, match=message):
        ssi_take_up_prior_basis_from_artifact(
            payload, targets=_TARGETS, source_sha256=sha
        )


def _delivered(diagnostics: dict[str, object], **selected: float) -> dict[str, object]:
    delivered = copy.deepcopy(diagnostics)
    for band in delivered["age_bands"]:
        key = str(band["age_band"])
        if key in selected:
            band["selected_recipient_weight"] = float(selected[key])
    return delivered


def test_delivery_gate_passes_within_tolerance_and_reports_the_fenced_band() -> None:
    _, _, _, diagnostics = _assigned()
    delivered = _delivered(
        diagnostics,
        under_18=5.0,  # 90% miss — fenced, must not fail (populace#453/#509)
        **{"18_64": 52.4, "65_plus": 47.6},  # within the 5% envelope
    )
    gate = us_ssi_take_up_delivery_gate(delivered, targets=_TARGETS)
    assert gate.passed
    assert gate.details["enforced_band_keys"] == list(US_SSI_TAKE_UP_ENFORCED_BAND_KEYS)
    assert gate.details["relative_tolerance"] == pytest.approx(
        US_SSI_TAKE_UP_BAND_DELIVERY_RELATIVE_TOLERANCE
    )
    fenced = gate.details["fenced_bands"]
    assert [row["age_band"] for row in fenced] == ["under_18"]
    assert "#453" in fenced[0]["fence"] and "#509" in fenced[0]["fence"]
    assert fenced[0]["selected_recipient_weight"] == pytest.approx(5.0)


def test_delivery_gate_does_not_prescribe_a_no_op_retry() -> None:
    """PR #554 review finding 1: when the release-weight prior recomputation
    matches the assignment prior for every missing band, a basis retry
    regenerates identical flags under the same seed. The gate must say so
    instead of prescribing an endless no-op ladder."""
    _, _, _, diagnostics = _assigned()
    # The fixture's basis IS the current frame, so both priors agree exactly.
    delivered = _delivered(
        diagnostics, **{"18_64": 50.0, "65_plus": 30.0}
    )  # 40% aged miss, realization-only
    gate = us_ssi_take_up_delivery_gate(delivered, targets=_TARGETS)
    assert not gate.passed
    assert gate.details["prior_weight_basis_retry_applicable"] is False
    assert any(
        "65_plus" in failure
        and "already true of the release weights" in failure
        and "reproduce identical flags" in failure
        for failure in gate.failures
    )
    assert not any(
        "is not true of the release weights" in failure for failure in gate.failures
    )
    assert not any("under_18" in failure for failure in gate.failures)


def test_delivery_gate_names_the_remedy_when_the_basis_would_change() -> None:
    _, _, _, diagnostics = _assigned()
    delivered = _delivered(
        diagnostics, **{"18_64": 50.0, "65_plus": 30.0}
    )  # 40% aged miss
    # Simulate post-calibration weight drift: the release-weight prior
    # recomputation now disagrees with the assignment prior, so a basis
    # retry would genuinely change the enforced band's thresholds.
    for band in delivered["age_bands"]:
        if band["age_band"] == "65_plus":
            band["prior_recomputed_from_current_weights"] = (
                float(band["assignment_prior"]) + 0.05
            )
    gate = us_ssi_take_up_delivery_gate(delivered, targets=_TARGETS)
    assert not gate.passed
    assert gate.details["prior_weight_basis_retry_applicable"] is True
    assert any(
        "65_plus" in failure
        and "is not true of the release weights" in failure
        and "--ssi-take-up-prior-weight-basis" in failure
        for failure in gate.failures
    )
    assert not any("under_18" in failure for failure in gate.failures)


def test_delivery_gate_names_saturated_enforced_support_inside_tolerance() -> None:
    _, _, _, diagnostics = _assigned()
    delivered = _delivered(
        diagnostics,
        **{"18_64": 49.5, "65_plus": 50.0},
    )
    adult = delivered["age_bands"][1]
    adult["candidate_capacity"] = 49.5
    adult["reporter_candidate_floor"] = 20.0
    adult["prior_status_recomputed_from_current_weights"] = "saturated_reporter_rate"

    gate = us_ssi_take_up_delivery_gate(delivered, targets=_TARGETS)
    assert not gate.passed
    assert any(
        "18_64" in failure
        and "insufficient candidate support" in failure
        and "eligibility/support" in failure
        for failure in gate.failures
    )
    assert not any(
        "18_64" in failure and "--ssi-take-up-prior-weight-basis" in failure
        for failure in gate.failures
    )
    assert gate.details["prior_weight_basis_retry_applicable"] is False


def test_mixed_structural_and_ordinary_misses_are_not_retryable() -> None:
    _, _, _, diagnostics = _assigned()
    delivered = _delivered(
        diagnostics,
        **{"18_64": 51.0, "65_plus": 30.0},
    )
    adult = delivered["age_bands"][1]
    adult["reporter_candidate_floor"] = 51.0
    adult["prior_status_recomputed_from_current_weights"] = (
        "reporter_floor_exceeds_target"
    )

    gate = us_ssi_take_up_delivery_gate(delivered, targets=_TARGETS)
    assert not gate.passed
    assert any(
        "Forced reporters alone over-deliver" in failure for failure in gate.failures
    )
    assert any(
        "65_plus" in failure
        and "co-occurs with a non-retryable" in failure
        and "--ssi-take-up-prior-weight-basis" not in failure
        for failure in gate.failures
    )
    assert gate.details["ordinary_delivery_miss_present"] is True
    assert gate.details["structural_support_failure_present"] is True
    assert gate.details["prior_weight_basis_retry_applicable"] is False


def test_delivery_gate_boundary_sits_at_the_documented_tolerance() -> None:
    _, _, _, diagnostics = _assigned()
    tolerance = US_SSI_TAKE_UP_BAND_DELIVERY_RELATIVE_TOLERANCE
    inside = _delivered(
        diagnostics,
        **{"18_64": 50.0 * (1.0 + tolerance) - 0.01, "65_plus": 50.0},
    )
    assert us_ssi_take_up_delivery_gate(inside, targets=_TARGETS).passed
    outside = _delivered(
        diagnostics,
        **{"18_64": 50.0 * (1.0 + tolerance) + 0.01, "65_plus": 50.0},
    )
    assert not us_ssi_take_up_delivery_gate(outside, targets=_TARGETS).passed


def test_delivery_gate_rejects_malformed_diagnostics() -> None:
    _, _, _, diagnostics = _assigned()
    truncated = copy.deepcopy(diagnostics)
    truncated["age_bands"] = truncated["age_bands"][:1]
    gate = us_ssi_take_up_delivery_gate(truncated, targets=_TARGETS)
    assert not gate.passed
    assert gate.details["invalid_diagnostics_failure_present"] is True
    assert gate.details["prior_weight_basis_retry_applicable"] is False

    wrong_schema = copy.deepcopy(diagnostics)
    wrong_schema["schema_version"] = 2
    gate = us_ssi_take_up_delivery_gate(wrong_schema, targets=_TARGETS)
    assert not gate.passed
    assert gate.details["invalid_diagnostics_failure_present"] is True
    assert gate.details["prior_weight_basis_retry_applicable"] is False


def test_gate_rejects_basis_arithmetic_drift() -> None:
    """The prior/basis link is weight-free, so the gate can audit it exactly."""

    _, _, _, diagnostics = _assigned()
    drifted = copy.deepcopy(diagnostics)
    drifted["age_bands"][0]["prior_basis_candidate_capacity"] = 1_000.0
    gate = us_ssi_take_up_gate(drifted, targets=_TARGETS)
    assert not gate.passed
    assert any("prior basis" in failure for failure in gate.failures)


def test_writer_emits_strict_json_and_refuses_nan(tmp_path: Path) -> None:
    _, _, _, diagnostics = _assigned()
    path = write_us_ssi_take_up_diagnostics(diagnostics, tmp_path / "ssi.json")
    assert json.loads(path.read_text()) == diagnostics
    invalid = copy.deepcopy(diagnostics)
    invalid["weighted_flag_true_share"] = np.nan
    with pytest.raises(ValueError, match="Out of range float"):
        write_us_ssi_take_up_diagnostics(invalid, tmp_path / "invalid.json")


@requires_us
def test_policyengine_us_1_764_6_take_up_flag_controls_positive_ssi() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "1.764.6"
    variable = CountryTaxBenefitSystem().variables[_OUTPUT]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert variable.value_type is bool
    assert variable.default_value is True

    def situation(takes_up: bool) -> dict[str, object]:
        return {
            "people": {
                "adult": {
                    "age": {"2024": 40},
                    "meets_ssi_disability_criteria": {"2024": True},
                    _OUTPUT: {"2024": takes_up},
                }
            },
            "tax_units": {
                "unit": {
                    "members": ["adult"],
                    "filing_status": {"2024": "SINGLE"},
                }
            },
            "families": {"family": {"members": ["adult"]}},
            "spm_units": {"spm": {"members": ["adult"]}},
            "households": {
                "household": {
                    "members": ["adult"],
                    "state_code": {"2024": "CA"},
                }
            },
            "marital_units": {"marital": {"members": ["adult"]}},
        }

    active = Simulation(situation=situation(True))
    neutralized = Simulation(situation=situation(False))
    period = "2024-12"
    assert active.calculate("uncapped_ssi", period)[0] == pytest.approx(943.0)
    assert neutralized.calculate("uncapped_ssi", period)[0] == pytest.approx(943.0)
    assert active.calculate("ssi", period)[0] == pytest.approx(943.0)
    assert neutralized.calculate("ssi", period)[0] == 0.0
