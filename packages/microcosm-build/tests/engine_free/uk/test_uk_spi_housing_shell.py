"""The SPI housing shell imputes housing from the recipient's own incomes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.council_tax_constants import (
    UK_COUNCIL_TAX_BANDS,
    UK_COUNCIL_TAX_BANDS_BY_REGION,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.spi_housing_shell import (
    HOUSING_AMOUNT_COLUMNS,
    HOUSING_CATEGORICAL_COLUMNS,
    RENTED_TENURES,
    UK_SPI_HOUSING_SHELL_REWRITES,
    UKSPIHousingShellStageTransform,
    _assert_invariants,
    _assert_stage_parameters,
    impute_spi_housing_shell,
    restrict_bands_to_nation,
)
from microcosm.frame import WeightKind
from test_support.microcosm_build.uk_spi_housing_shell import housing_shell_tables


def _stage():
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()["spi_housing_shell"]


def test_packaged_stage_matches_the_module_contract() -> None:
    _assert_stage_parameters(_stage())
    assert _stage().rewrites == UK_SPI_HOUSING_SHELL_REWRITES


def test_frs_rows_are_untouched_and_the_invariants_hold() -> None:
    person, _, household, weights = housing_shell_tables()
    person_out, household_out, receipt = impute_spi_housing_shell(
        person, household, weights, seed=0, n_estimators=20
    )
    _assert_invariants(person, household, person_out, household_out)
    frs = household["household_support_channel"] == "frs"
    pd.testing.assert_frame_equal(household.loc[frs], household_out.loc[frs])
    assert receipt["training_households"] == int(frs.sum())
    assert receipt["recipient_households"] == int((~frs).sum())


def test_spi_housing_follows_the_recipients_own_income() -> None:
    person, _, household, weights = housing_shell_tables()
    _, household_out, _ = impute_spi_housing_shell(
        person, household, weights, seed=0, n_estimators=20
    )
    spi = household_out[household_out["household_support_channel"] == "spi"]
    # SPI copies carry high incomes, so the planted relation puts them in
    # owned, detached housing; the parent's shell would not.
    owned = spi["tenure_type"].str.startswith("OWNED").mean()
    detached = (spi["accommodation_type"] == "HOUSE_DETACHED").mean()
    assert owned > 0.85
    assert detached > 0.85


def test_structural_rules_hold_on_recipients() -> None:
    person, _, household, weights = housing_shell_tables()
    person_out, household_out, _ = impute_spi_housing_shell(
        person, household, weights, seed=0, n_estimators=20
    )
    spi = household_out[household_out["household_support_channel"] == "spi"]
    mortgage = spi["mortgage_interest_repayment"] + spi["mortgage_capital_repayment"]
    assert (mortgage[spi["tenure_type"] != "OWNED_WITH_MORTGAGE"] == 0).all()
    assert (spi.loc[spi["region"] != "NORTHERN_IRELAND", "domestic_rates"] == 0).all()
    assert (spi.loc[spi["region"] == "NORTHERN_IRELAND", "council_tax"] == 0).all()
    assert set(spi["num_bedrooms"].unique()) <= {2, 4}
    for column in HOUSING_CATEGORICAL_COLUMNS:
        assert set(spi[column].astype(str)) <= set(household[column].astype(str))
    spi_people = person_out[person_out["person_household_id"].isin(spi["household_id"])]
    tenure = spi.set_index("household_id")["tenure_type"]
    hb = spi_people[spi_people["housing_benefit_reported"] > 0]
    assert hb["person_household_id"].map(tenure).isin(RENTED_TENURES).all()
    assert hb["is_household_head"].all()


def test_draws_are_keyed_by_household_identity() -> None:
    person, _, household, weights = housing_shell_tables()
    _, first, _ = impute_spi_housing_shell(
        person, household, weights, seed=0, n_estimators=20
    )
    order = np.random.default_rng(1).permutation(len(household))
    _, second, _ = impute_spi_housing_shell(
        person,
        household.iloc[order].reset_index(drop=True),
        weights[order],
        seed=0,
        n_estimators=20,
    )
    columns = [*HOUSING_CATEGORICAL_COLUMNS, *HOUSING_AMOUNT_COLUMNS]
    left = first.set_index("household_id")[columns].sort_index()
    right = second.set_index("household_id")[columns].sort_index()
    pd.testing.assert_frame_equal(left, right)


def test_stage_transform_runs_on_a_frame_and_records_its_receipt() -> None:
    person, benunit, household, weights = housing_shell_tables()
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=weights,
    )
    transform = UKSPIHousingShellStageTransform(stage=_stage(), n_estimators=20)
    result = transform(frame)
    assert transform.output_columns() == ()
    evidence = transform.checkpoint_metadata()["evidence"]
    assert evidence["stage"] == "spi_housing_shell"
    assert "rule_firings" in evidence
    np.testing.assert_array_equal(result.weights_for("household").values, weights)


def test_a_frame_without_frs_training_rows_is_refused() -> None:
    person, _, household, weights = housing_shell_tables()
    household = household.assign(household_support_channel="spi")
    with pytest.raises(ValueError, match="FRS training rows"):
        impute_spi_housing_shell(person, household, weights, seed=0, n_estimators=20)


def test_a_band_inside_its_nation_keeps_its_draw() -> None:
    classes = np.asarray(UK_COUNCIL_TAX_BANDS)
    scores = np.zeros((3, len(classes)))
    choice = np.array([7, 8, 3])  # H in England, I in Wales, D in Northern Ireland
    regions = np.array(["LONDON", "WALES", "NORTHERN_IRELAND"])
    corrected, redrawn = restrict_bands_to_nation(
        choice, scores, classes, regions, np.full(3, 0.5)
    )
    np.testing.assert_array_equal(corrected, choice)
    assert sum(redrawn.values()) == 0
    assert set(redrawn) == {
        "council_tax_band:redrawn:ENGLAND",
        "council_tax_band:redrawn:WALES",
        "council_tax_band:redrawn:SCOTLAND",
        "council_tax_band:redrawn:NORTHERN_IRELAND",
    }


def test_a_band_outside_its_nation_is_redrawn_from_the_nations_bands() -> None:
    classes = np.asarray(UK_COUNCIL_TAX_BANDS)
    regions = np.array(["LONDON", "SCOTLAND", "NORTHERN_IRELAND", "NORTHERN_IRELAND"])
    scores = np.zeros((4, len(classes)))
    choice = np.array([8, 8, 0, 3])  # I, I, A, D
    corrected, redrawn = restrict_bands_to_nation(
        choice, scores, classes, regions, np.array([0.99, 0.0, 0.5, 0.5])
    )
    bands = classes[corrected]
    for band, region in zip(bands, regions, strict=True):
        assert band in UK_COUNCIL_TAX_BANDS_BY_REGION[region]
    assert list(bands[2:]) == ["D", "D"]
    assert corrected[3] == choice[3]
    assert redrawn["council_tax_band:redrawn:ENGLAND"] == 1
    assert redrawn["council_tax_band:redrawn:SCOTLAND"] == 1
    assert redrawn["council_tax_band:redrawn:NORTHERN_IRELAND"] == 1


def test_redrawn_bands_follow_the_classifier_over_the_nations_bands() -> None:
    # Every English row drew band I; the redraw must spread them over A-H in
    # proportion to the classifier's own probabilities.
    n = 40_000
    classes = np.asarray(UK_COUNCIL_TAX_BANDS)
    weights = np.array([1, 2, 3, 4, 3, 2, 1, 1, 3], dtype=float)
    scores = np.tile(np.log(weights), (n, 1))
    uniforms = stable_identity_uniforms(np.arange(n), seed=0, salt="test")
    corrected, _ = restrict_bands_to_nation(
        np.full(n, 8), scores, classes, np.full(n, "LONDON"), uniforms
    )
    observed = np.bincount(corrected, minlength=len(classes)) / n
    expected = np.append(weights[:8] / weights[:8].sum(), 0.0)
    np.testing.assert_allclose(observed, expected, atol=0.01)


def test_bands_the_softmax_rounded_to_zero_are_still_drawn() -> None:
    # On licensed data a few English rows score band I so far above A-H that
    # the softmax gives A-H probability exactly zero; the conditional draw on
    # the raw scores still spreads them over A-H.
    classes = np.asarray(UK_COUNCIL_TAX_BANDS)
    scores = np.tile(np.append(np.linspace(-1000.0, -999.0, 8), 0.0), (4, 1))
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    assert (probabilities[:, :8] == 0.0).all()
    corrected, redrawn = restrict_bands_to_nation(
        np.full(4, 8),
        scores,
        classes,
        np.full(4, "NORTH_WEST"),
        np.array([0.0, 0.3, 0.6, 0.99]),
    )
    assert set(classes[corrected]) <= set("ABCDEFGH")
    assert redrawn["council_tax_band:redrawn:ENGLAND"] == 4


def test_class_scores_soften_into_the_classifiers_probabilities() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    from microcosm.build.uk_runtime.spi_housing_shell import class_scores

    rng = np.random.default_rng(0)
    features = rng.normal(size=(300, 3))
    for labels in (
        np.where(features[:, 0] > 0, "A", "B"),
        np.select([features[:, 0] > 0.5, features[:, 0] < -0.5], ["A", "C"], "B"),
    ):
        classifier = HistGradientBoostingClassifier(max_iter=20).fit(features, labels)
        scores = class_scores(classifier, features)
        softmax = np.exp(scores - scores.max(axis=1, keepdims=True))
        softmax /= softmax.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(
            softmax, classifier.predict_proba(features), atol=1e-9
        )


def test_a_nation_the_classifier_cannot_serve_is_refused() -> None:
    classes = np.array(["B", "F"])
    with pytest.raises(ValueError, match="no council-tax band the classifier"):
        restrict_bands_to_nation(
            np.array([0]),
            np.array([[0.5, 0.5]]),
            classes,
            np.array(["NORTHERN_IRELAND"]),
            np.array([0.5]),
        )
    with pytest.raises(ValueError, match="need a nation for regions"):
        restrict_bands_to_nation(
            np.array([0]),
            np.array([[0.5, 0.5]]),
            classes,
            np.array(["ATLANTIS"]),
            np.array([0.5]),
        )


def _band_i_everywhere():
    """FRS training rows that hold band I in every region, so SPI rows draw it."""

    person, benunit, household, weights = housing_shell_tables()
    frs = (household["household_support_channel"] == "frs").to_numpy()
    plant = frs & (np.arange(len(household)) % 3 == 0)
    household.loc[plant, "council_tax_band"] = "I"
    return person, benunit, household, weights


def test_the_stage_keeps_every_spi_band_inside_its_nation() -> None:
    person, _, household, weights = _band_i_everywhere()
    _, household_out, receipt = impute_spi_housing_shell(
        person, household, weights, seed=0, n_estimators=20
    )
    spi = household_out[household_out["household_support_channel"] == "spi"]
    allowed = spi["region"].map(UK_COUNCIL_TAX_BANDS_BY_REGION)
    assert all(
        band in bands
        for band, bands in zip(spi["council_tax_band"], allowed, strict=True)
    )
    firings = receipt["rule_firings"]
    assert firings["council_tax_band:redrawn:ENGLAND"] > 0
    frs = household["household_support_channel"] == "frs"
    pd.testing.assert_frame_equal(household.loc[frs], household_out.loc[frs])


def test_without_the_restriction_spi_rows_leave_their_nation(monkeypatch) -> None:
    # Mutation check: the planted case does reach out-of-nation bands when the
    # restriction is removed, so the test above can fail.
    import microcosm.build.uk_runtime.spi_housing_shell as shell

    monkeypatch.setattr(
        shell,
        "restrict_bands_to_nation",
        lambda choice, *_args: (choice, {}),
    )
    person, _, household, weights = _band_i_everywhere()
    _, household_out, _ = impute_spi_housing_shell(
        person, household, weights, seed=0, n_estimators=20
    )
    spi = household_out[household_out["household_support_channel"] == "spi"]
    allowed = spi["region"].map(UK_COUNCIL_TAX_BANDS_BY_REGION)
    assert not all(
        band in bands
        for band, bands in zip(spi["council_tax_band"], allowed, strict=True)
    )
