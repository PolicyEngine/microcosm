"""Policy-independent SSI asset propensity and source-level seeding contracts."""

from __future__ import annotations

import copy
import importlib.util

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.ssi_asset_gradient import (
    solve_ssi_asset_intercept,
    ssi_asset_propensity,
)
from microcosm.build.us_runtime.ssi_take_up import (
    _stable_source_draw,
    reseed_us_ssi_take_up,
    ssi_take_up_prior_basis_from_artifact,
    ssi_take_up_prior_basis_from_diagnostics,
    us_ssi_take_up_diagnostics,
    us_ssi_take_up_gate,
    us_ssi_take_up_reporter_source_ids,
    us_ssi_take_up_source_liquid_assets,
    with_us_ssi_take_up,
)
from microcosm.frame import Frame, WeightKind, Weights
from test_support.microcosm_build.us_ssi_take_up import (
    _ANCHORED_SOURCE_NUMBERS,
    _BAND_CAPACITY,
    _BASIS_SHA,
    _OUTPUT,
    _REPORTER_FLOOR,
    _TARGETS,
    _ZERO_ASSET_SLOPES,
    _artifact_basis,
    _frame,
    _replace_person,
)
from test_support.paths import paths_for

_SLOPES = {"under_18": -0.2, "18_64": -0.35, "65_plus": -0.5}
_ASSETS_BY_SOURCE = np.asarray(
    [20_000, 0, 100, 500, 1_000, 2_000, 100_000, 5_000, 20_000]
)


def _asset_frame() -> tuple[Frame, np.ndarray]:
    frame, potential = _frame()
    person = frame.table("person").copy()
    number = person["person_source_id"].str.rsplit(":").str[-1].astype(int)
    assets = _ASSETS_BY_SOURCE[number]
    # Three independent input columns must contribute to the person's sum.
    person["bank_account_assets"] = assets / 2
    person["stock_assets"] = assets / 4
    person["bond_assets"] = assets / 4
    return _replace_person(frame, person), potential


def _gradient_assignment(
    *,
    asset_scale: float = 1.0,
    slopes: dict[str, float] | None = None,
) -> tuple[Frame, np.ndarray, dict[str, object]]:
    frame, potential = _asset_frame()
    person = frame.table("person").copy()
    for column in ("bank_account_assets", "stock_assets", "bond_assets"):
        person[column] *= asset_scale
    frame = _replace_person(frame, person)
    result, diagnostics = with_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
        asset_slopes=_SLOPES if slopes is None else slopes,
    )
    return result, potential, diagnostics


@settings(max_examples=80, deadline=None)
@given(
    assets=st.lists(
        st.floats(0, 1e9, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
    ),
    intercept=st.floats(-20, 20, allow_nan=False, allow_infinity=False),
    slope=st.floats(-5, 0, allow_nan=False, allow_infinity=False),
)
def test_propensity_is_monotone_for_nonpositive_slopes(
    assets: list[float],
    intercept: float,
    slope: float,
) -> None:
    probability = ssi_asset_propensity(np.sort(assets), intercept, slope)
    assert np.all(np.diff(probability) <= 0)


def test_finite_propensity_stays_strictly_inside_unit_interval() -> None:
    for intercept in (-1_000.0, -30.0, 0.0, 30.0, 1_000.0):
        for slope in (-100.0, -1.0, 0.0, 1.0, 100.0):
            probability = ssi_asset_propensity(
                np.asarray([0.0, 1.0, 2_000.0, 20_000.0, 1e200]),
                intercept,
                slope,
            )
            assert np.all(np.isfinite(probability))
            assert np.all((probability > 0) & (probability < 1))


@settings(max_examples=80, deadline=None)
@given(
    pairs=st.lists(
        st.tuples(
            st.floats(0, 1e6, allow_nan=False, allow_infinity=False),
            st.floats(0.1, 1e4, allow_nan=False, allow_infinity=False),
        ),
        min_size=2,
        max_size=20,
    ),
    fraction=st.floats(0.001, 0.999, allow_nan=False, allow_infinity=False),
    slope=st.floats(-3, 3, allow_nan=False, allow_infinity=False),
)
def test_intercept_solve_reproduces_weighted_target(
    pairs: list[tuple[float, float]],
    fraction: float,
    slope: float,
) -> None:
    assets, weights = np.asarray(pairs).T
    target = fraction * weights.sum()
    intercept = solve_ssi_asset_intercept(assets, weights, target, slope)
    expected = np.dot(weights, ssi_asset_propensity(assets, intercept, slope))
    assert expected == pytest.approx(target, rel=1e-11, abs=1e-8)


def test_zero_weight_assets_do_not_move_intercept() -> None:
    assets = np.asarray([0.0, 100.0, 2_000.0])
    weights = np.asarray([2.0, 3.0, 5.0])
    original = solve_ssi_asset_intercept(assets, weights, 4.0, -0.4)
    appended = solve_ssi_asset_intercept(
        np.append(assets, [0.0, 1e200]),
        np.append(weights, [0.0, 0.0]),
        4.0,
        -0.4,
    )
    assert appended == pytest.approx(original, abs=1e-12)


@pytest.mark.parametrize(
    ("asset_scale", "slopes"),
    [(1.0, _SLOPES), (1e100, {key: -5.0 for key in _TARGETS})],
    ids=("observed_asset_range", "extreme_intercept_precision"),
)
def test_every_person_obeys_own_asset_law_and_reporters_stay_true(
    asset_scale: float,
    slopes: dict[str, float],
) -> None:
    result, potential, diagnostics = _gradient_assignment(
        asset_scale=asset_scale, slopes=slopes
    )
    person = result.table("person")
    by_band = {row["age_band"]: row for row in diagnostics["age_bands"]}
    for position, row in person.iterrows():
        source_id = row["person_source_id"]
        key, number = source_id.split(":")
        band = by_band[key]
        own_assets = sum(
            float(row[column])
            for column in (
                "bank_account_assets",
                "stock_assets",
                "bond_assets",
            )
        )
        probability = ssi_asset_propensity(
            np.asarray([own_assets]),
            band["assignment_intercept"],
            slopes[key],
        )[0]
        anchor = number in _ANCHORED_SOURCE_NUMBERS
        assert bool(row[_OUTPUT]) == (
            anchor or _stable_source_draw(source_id, seed=17) < probability
        )
        if anchor:
            assert row[_OUTPUT]
        # This includes source 8: no current candidate benefit and high assets.
        if number == "8":
            assert potential[position] == 0
            assert own_assets == 20_000 * asset_scale
    for band in by_band.values():
        assert band["expected_recipient_weight_on_prior_basis"] == pytest.approx(
            50.0, abs=1e-8
        )
    assert diagnostics["reporter_anchor_lost_count"] == 0
    assert diagnostics["bernoulli_law_violation_count"] == 0
    assert us_ssi_take_up_gate(diagnostics, targets=_TARGETS).passed


def test_source_draws_survive_person_row_reordering() -> None:
    frame, potential = _asset_frame()
    result, _ = with_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
        asset_slopes=_SLOPES,
    )
    permutation = np.random.default_rng(83).permutation(len(potential))
    reordered = _replace_person(
        frame, frame.table("person").iloc[permutation].reset_index(drop=True)
    )
    alternate, _ = with_us_ssi_take_up(
        reordered,
        uncapped_ssi=potential[permutation],
        seed=17,
        targets=_TARGETS,
        asset_slopes=_SLOPES,
    )
    pd.testing.assert_series_equal(
        result.table("person").set_index("person_id")[_OUTPUT].sort_index(),
        alternate.table("person").set_index("person_id")[_OUTPUT].sort_index(),
    )


def test_weight_split_support_clones_preserve_original_flags() -> None:
    frame, potential = _asset_frame()
    result, diagnostics = with_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
        asset_slopes=_SLOPES,
    )
    person = frame.table("person").copy()
    person["person_support_clone_index"] = np.where(
        person["person_support_channel"].eq("asec"), 0, 1
    )
    original_row = person.index[
        person["person_source_id"].eq("65_plus:5")
        & person["person_support_channel"].eq("puf_tax_detail")
    ][0]
    household = frame.table("household").copy()
    household_id = int(household["household_id"].max()) + 1
    clone = person.loc[[original_row]].copy()
    clone["person_id"] = int(person["person_id"].max()) + 1
    clone["person_household_id"] = household_id
    clone["person_support_clone_index"] = 2
    person = pd.concat([person, clone], ignore_index=True)
    weights = frame.weights_for("household").values.copy()
    weights[original_row] -= 4.0
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    tables["household"] = pd.concat(
        [household, pd.DataFrame({"household_id": [household_id]})], ignore_index=True
    )
    cloned_frame = Frame(
        tables,
        frame.schema,
        {"household": Weights(np.append(weights, 4.0), WeightKind.DESIGN)},
    )
    cloned, cloned_diagnostics = with_us_ssi_take_up(
        cloned_frame,
        uncapped_ssi=np.append(potential, potential[original_row]),
        seed=17,
        targets=_TARGETS,
        asset_slopes=_SLOPES,
    )
    np.testing.assert_array_equal(
        cloned.table("person")[_OUTPUT].iloc[:-1], result.table("person")[_OUTPUT]
    )
    assert (
        cloned.table("person").groupby("person_source_id")[_OUTPUT].nunique().max() == 1
    )
    assert cloned_diagnostics["age_bands"] == diagnostics["age_bands"]


def test_zero_slope_flags_match_frozen_schema4_implementation_exactly() -> None:
    reference_path = (
        paths_for("microcosm-build").tests
        / "fixtures"
        / "ssi_take_up_schema4_reference.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ssi_schema4_reference", reference_path
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    frame, potential = _asset_frame()
    for target in (5.0, 20.0, 50.0, 130.0, 1_000.0):
        targets = {key: target for key in _TARGETS}
        prior = reference._band_prior(target, _BAND_CAPACITY, _REPORTER_FLOOR)
        for seed in (0, 17, 18, 900):
            result, diagnostics = with_us_ssi_take_up(
                frame,
                uncapped_ssi=potential,
                seed=seed,
                targets=targets,
                asset_slopes=_ZERO_ASSET_SLOPES,
            )
            expected = [
                source_id.split(":")[1] in _ANCHORED_SOURCE_NUMBERS
                or reference._stable_source_draw(source_id, seed=seed) < prior
                for source_id in result.table("person")["person_source_id"]
            ]
            np.testing.assert_array_equal(result.table("person")[_OUTPUT], expected)
            assert all(
                row["assignment_prior"] == prior for row in diagnostics["age_bands"]
            )


def test_nonzero_slopes_reject_scalar_only_legacy_prior_basis() -> None:
    frame, potential = _asset_frame()
    basis = _artifact_basis(capacities={key: 130.0 for key in _TARGETS})
    with pytest.raises(ValueError, match="asset"):
        with_us_ssi_take_up(
            frame,
            uncapped_ssi=potential,
            seed=17,
            targets=_TARGETS,
            prior_basis=basis,
            asset_slopes=_SLOPES,
        )


def test_schema5_delivered_basis_retains_asset_distribution_and_target_mass() -> None:
    result, potential, stage = _gradient_assignment()
    priors = {row["age_band"]: row["assignment_prior"] for row in stage["age_bands"]}
    # Reweight as an already-released frame; the intercept stays frozen in
    # measurement and is solved once on the delivered basis when reseeded.
    weights = result.weights_for("household").values * np.linspace(
        0.6, 1.8, len(potential)
    )
    reweighted = Frame(
        {entity: result.table(entity).copy() for entity in result.entities},
        result.schema,
        {"household": Weights(weights, WeightKind.CALIBRATED)},
    )
    final = us_ssi_take_up_diagnostics(
        reweighted,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
        assignment_priors=priors,
        prior_basis=ssi_take_up_prior_basis_from_diagnostics(stage),
        asset_slopes=_SLOPES,
    )
    basis = ssi_take_up_prior_basis_from_artifact(
        final, targets=_TARGETS, source_sha256=_BASIS_SHA
    )
    for band in basis.bands:
        row = next(row for row in final["age_bands"] if row["age_band"] == band.key)
        assert band.nonanchor_asset_distribution == tuple(
            map(tuple, row["candidate_nonanchor_asset_distribution"])
        )
        assert sum(
            weight for _, weight in band.nonanchor_asset_distribution
        ) == pytest.approx(band.candidate_capacity - band.reporter_candidate_floor)
    reseeded, diagnostics = with_us_ssi_take_up(
        reweighted,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
        prior_basis=basis,
        asset_slopes=_SLOPES,
    )
    assert (
        reseeded.table("person").groupby("person_source_id")[_OUTPUT].nunique().max()
        == 1
    )
    assert all(
        row["expected_recipient_weight_on_prior_basis"] == pytest.approx(50.0, abs=1e-8)
        for row in diagnostics["age_bands"]
    )
    assert us_ssi_take_up_gate(diagnostics, targets=_TARGETS).passed


def test_gate_rejects_gradient_coefficient_corruption() -> None:
    _, _, diagnostics = _gradient_assignment()
    for field in ("assignment_intercept", "asset_slope"):
        corrupted = copy.deepcopy(diagnostics)
        corrupted["age_bands"][1][field] += 0.1
        assert not us_ssi_take_up_gate(corrupted, targets=_TARGETS).passed, field


def test_gate_rejects_candidate_asset_mass_corruption() -> None:
    _, _, diagnostics = _gradient_assignment()
    corrupted = copy.deepcopy(diagnostics)
    band = corrupted["age_bands"][1]
    distribution = list(band["prior_basis_nonanchor_asset_distribution"])
    asset, weight = distribution[0]
    distribution[0] = (asset, weight + 1.0)
    band["prior_basis_nonanchor_asset_distribution"] = distribution
    assert not us_ssi_take_up_gate(corrupted, targets=_TARGETS).passed


def test_physical_asec_assets_own_source_despite_support_reimputation() -> None:
    frame, potential = _asset_frame()
    baseline, baseline_diagnostics = with_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
        asset_slopes=_SLOPES,
    )
    person = frame.table("person").copy()
    donor = person["person_support_channel"].eq("puf_tax_detail") & ~person[
        "person_source_id"
    ].str.endswith(":7")
    person.loc[donor, "bank_account_assets"] += 1e6
    permutation = np.concatenate([np.flatnonzero(donor), np.flatnonzero(~donor)])
    result, diagnostics = with_us_ssi_take_up(
        _replace_person(frame, person.iloc[permutation].reset_index(drop=True)),
        uncapped_ssi=potential[permutation],
        seed=17,
        targets=_TARGETS,
        asset_slopes=_SLOPES,
    )
    pd.testing.assert_series_equal(
        result.table("person").set_index("person_id")[_OUTPUT].sort_index(),
        baseline.table("person").set_index("person_id")[_OUTPUT].sort_index(),
    )
    assert diagnostics["age_bands"] == baseline_diagnostics["age_bands"]
    assert diagnostics["asset_age_bands"] == baseline_diagnostics["asset_age_bands"]
    assert (
        diagnostics["asset_gradient"]["support_asset_disagreement_source_count"] == 21
    )
    assert diagnostics["asset_gradient"]["source_without_physical_asec_row_count"] == 3


def test_asec_absent_source_assets_must_be_unambiguous() -> None:
    frame, potential = _asset_frame()
    person = frame.table("person").copy()
    person["person_support_clone_index"] = np.where(
        person["person_support_channel"].eq("asec"), 0, 1
    )
    original = person["person_source_id"].eq("under_18:8") & person[
        "person_support_channel"
    ].eq("asec")
    person.loc[original, "person_support_channel"] = "puf_tax_detail"
    person.loc[original, "person_support_clone_index"] = 2
    person.loc[original, "bank_account_assets"] += 1.0
    with pytest.raises(ValueError, match="without a physical ASEC"):
        with_us_ssi_take_up(
            _replace_person(frame, person),
            uncapped_ssi=potential,
            seed=17,
            targets=_TARGETS,
            asset_slopes=_SLOPES,
        )


def test_captured_source_assets_preserve_frozen_law_after_owner_pruning() -> None:
    frame, potential = _asset_frame()
    person = frame.table("person").copy()
    donor = person["person_support_channel"].eq("puf_tax_detail") & ~person[
        "person_source_id"
    ].str.endswith(":7")
    person.loc[donor, "bank_account_assets"] += 1e8
    frame = _replace_person(frame, person)
    owned_assets = us_ssi_take_up_source_liquid_assets(frame)
    assert owned_assets == {
        f"{key}:{number}": float(asset)
        for key in _TARGETS
        for number, asset in enumerate(_ASSETS_BY_SOURCE)
    }
    reporter_ids = us_ssi_take_up_reporter_source_ids(frame)
    assigned, initial = with_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
        asset_slopes=_SLOPES,
        source_liquid_assets=owned_assets,
    )
    person = assigned.table("person")
    keep = person["person_support_channel"].eq("puf_tax_detail") | person[
        "person_source_id"
    ].isin(reporter_ids)
    # Retain unselected ASEC rows so the existing channel nonconstancy gate
    # remains meaningful; selected nonreporters lose their physical owner.
    keep |= ~person[_OUTPUT]
    # Source 8 disappears entirely; extra captured IDs remain valid.
    keep &= ~person["person_source_id"].str.endswith(":8")
    retained = person.loc[keep].reset_index(drop=True)
    tables = {"person": retained}
    weights = {}
    for entity in assigned.entities:
        if entity == "person":
            continue
        table = assigned.table(entity)
        group_keep = table[f"{entity}_id"].isin(retained[f"person_{entity}_id"])
        tables[entity] = table.loc[group_keep].reset_index(drop=True)
        if entity in assigned.weighted_entities:
            original_weight = assigned.weights_for(entity)
            weights[entity] = Weights(
                values=original_weight.values[group_keep],
                kind=original_weight.kind,
            )
    pruned = Frame(
        tables,
        assigned.schema,
        weights,
        assigned.strata.loc[keep].reset_index(drop=True),
    )
    arguments = {
        "uncapped_ssi": potential[keep],
        "seed": 17,
        "targets": _TARGETS,
        "assignment_priors": {
            row["age_band"]: row["assignment_prior"] for row in initial["age_bands"]
        },
        "prior_basis": ssi_take_up_prior_basis_from_diagnostics(initial),
        "reporter_source_ids": reporter_ids,
        "asset_slopes": _SLOPES,
    }
    preserved = us_ssi_take_up_diagnostics(
        pruned, source_liquid_assets=owned_assets, **arguments
    )
    assert preserved["bernoulli_law_violation_count"] == 0
    assert preserved["reporter_anchor_lost_count"] == 0
    assert us_ssi_take_up_gate(preserved, targets=_TARGETS).passed
    reinterpreted = us_ssi_take_up_diagnostics(pruned, **arguments)
    assert reinterpreted["bernoulli_law_violation_count"] > 0
    assert not us_ssi_take_up_gate(reinterpreted, targets=_TARGETS).passed


@pytest.mark.parametrize("invalid", ["missing", "negative", "nan", "physical_mismatch"])
def test_captured_source_asset_map_requires_valid_coverage_and_owner_match(
    invalid: str,
) -> None:
    frame, potential = _asset_frame()
    owned_assets = us_ssi_take_up_source_liquid_assets(frame)
    # A PUF-only source isolates coverage/finite guards from the independent
    # physical-owner agreement guard.
    key = "18_64:1" if invalid == "physical_mismatch" else "18_64:7"
    if invalid == "missing":
        del owned_assets[key]
    elif invalid == "negative":
        owned_assets[key] = -1.0
    elif invalid == "nan":
        owned_assets[key] = float("nan")
    else:
        owned_assets[key] += 1.0
    with pytest.raises(ValueError, match="frozen"):
        with_us_ssi_take_up(
            frame,
            uncapped_ssi=potential,
            seed=17,
            targets=_TARGETS,
            asset_slopes=_SLOPES,
            source_liquid_assets=owned_assets,
        )


def test_missing_liquid_asset_inputs_are_rejected() -> None:
    frame, potential = _asset_frame()
    person = frame.table("person").drop(columns=["stock_assets"])
    with pytest.raises(ValueError, match="asset"):
        with_us_ssi_take_up(
            _replace_person(frame, person),
            uncapped_ssi=potential,
            seed=17,
            targets=_TARGETS,
            asset_slopes=_SLOPES,
        )


def test_asset_age_diagnostics_reconcile_all_source_people_and_weights() -> None:
    result, _, diagnostics = _gradient_assignment()
    rows = diagnostics["asset_age_bands"]
    assert len(rows) == 15
    for key in _TARGETS:
        band = [row for row in rows if row["age_band"] == key]
        assert [row["source_identity_count"] for row in band] == [4, 2, 0, 2, 1]
        assert [row["person_weight"] for row in band] == [80.0, 30.0, 0.0, 40.0, 10.0]
        assert (
            sum(row["flag_true_source_identity_count"] for row in band)
            == result.table("person")
            .loc[result.table("person")["person_source_id"].str.startswith(key + ":")]
            .groupby("person_source_id")[_OUTPUT]
            .first()
            .sum()
        )
        probability = [
            row["mean_propensity"] for row in band if row["mean_propensity"] is not None
        ]
        assert np.all(np.diff(probability) <= 0)
        assert band[-1]["reporter_source_identity_count"] == 1
        assert band[-1]["expected_flag_true_weight"] == 10.0
    assert "#644" in diagnostics["asset_gradient"]["candidate_basis_interaction"]


def test_public_reseed_uses_existing_weights_and_candidate_probe() -> None:
    frame, potential = _asset_frame()
    old_flags = frame.table("person")[_OUTPUT].copy()
    expected, expected_diagnostics = with_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
        asset_slopes=_SLOPES,
    )
    result, diagnostics = reseed_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
        asset_slopes=_SLOPES,
    )
    assert diagnostics == expected_diagnostics
    pd.testing.assert_frame_equal(result.table("person"), expected.table("person"))
    pd.testing.assert_series_equal(frame.table("person")[_OUTPUT], old_flags)
    np.testing.assert_array_equal(
        result.weights_for("household").values, frame.weights_for("household").values
    )
