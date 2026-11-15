"""The SPI housing shell imputes housing from the recipient's own incomes.

The tables are test_support's invented FRS and SPI households
(``uk_spi_housing_shell``). The council tax of the Great Britain rows the SPI
support channel copied is priced on the toy conditioned supports at an
invented Band D (``uk_conditioned_support_fixtures`` and
``uk_council_tax_fixtures``), and each bill expected here is worked again
from the toys, the invented Band D and the statute. Every amount is invented.
"""

from __future__ import annotations

import ast
import functools
import inspect
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import microcosm.build.uk_runtime.spi_housing_shell as housing_shell
from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceOperationSpec, SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.atomic_area_conditioning import (
    IDENTITY_COLUMN,
    uk_conditioned_assignment_definition,
)
from microcosm.build.uk_runtime.council_tax_band_d import (
    UKCouncilTaxBandDError,
    uk_statutory_council_tax,
)
from microcosm.build.uk_runtime.council_tax_constants import (
    UK_COUNCIL_TAX_BANDS,
    UK_COUNCIL_TAX_BANDS_BY_REGION,
    UK_NATION_BY_REGION,
)
from microcosm.build.uk_runtime.household_lineage import GEOGRAPHIC_FOLD_EVIDENCE
from microcosm.build.uk_runtime.local_authority_input import (
    UK_LOCAL_AUTHORITY_CODE_COLUMN,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.spi_housing_council_tax import (
    UK_SPI_HOUSING_COUNCIL_TAX_ARTIFACTS,
    UK_SPI_HOUSING_COUNCIL_TAX_CHECKS,
    UK_SPI_HOUSING_COUNCIL_TAX_EVIDENCE,
    UKSPIHousingCouncilTaxInputs,
    uk_spi_housing_council_tax_derive_parameters,
    uk_spi_housing_council_tax_evidence,
    uk_spi_housing_council_tax_inputs,
    uk_spi_housing_council_tax_verify_parameters,
)
from microcosm.build.uk_runtime.spi_housing_shell import (
    HOUSING_AMOUNT_COLUMNS,
    HOUSING_CATEGORICAL_COLUMNS,
    RENTED_TENURES,
    UK_SPI_HOUSING_SHELL_OPERATION,
    UK_SPI_HOUSING_SHELL_REWRITES,
    UK_SPI_HOUSING_SHELL_STAGE_NAME,
    FittedHousingModel,
    UKSPIHousingShellStageTransform,
    _assert_invariants,
    _assert_stage_parameters,
    apply_structural_rules,
    apply_support_rules,
    cap_council_tax_benefit,
    fit_housing_model,
    household_housing_predictors,
    household_housing_targets,
    impute_spi_housing_shell,
    restrict_bands_to_nation,
)
from microcosm.frame import WeightKind
from test_support.microcosm_build.uk_conditioned_support_fixtures import (
    toy_conditioned_support_payloads,
    toy_conditioned_supports,
)
from test_support.microcosm_build.uk_council_tax_fixtures import (
    invented_band_d,
    toy_area_draw_definition,
)
from test_support.microcosm_build.uk_geographic_clones import with_geographic_clones
from test_support.microcosm_build.uk_spi_housing_shell import housing_shell_tables
from test_support.paths import paths_for

_NORTHERN_IRELAND = "NORTHERN_IRELAND"
_SPI = "spi"
#: The chained QRF's targets, in chain order, written out here rather than
#: read from the module: ``council_tax`` stays the chain's first target on
#: every row, so every later amount keeps today's chain (microcosm#931).
_CHAIN = (
    "council_tax",
    "council_tax_reported",
    "rent",
    "mortgage_interest_repayment",
    "mortgage_capital_repayment",
    "structural_insurance_payments",
    "housing_service_charges",
    "water_and_sewerage_charges",
    "domestic_rates",
    "subrent",
    "housing_benefit_head_unit",
    "council_tax_benefit_hrp",
)


def _stage():
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()["spi_housing_shell"]


@functools.cache
def _supports() -> dict:
    return toy_conditioned_supports()


@functools.cache
def _payloads() -> dict:
    return toy_conditioned_support_payloads()


@functools.cache
def _definition() -> dict:
    return toy_area_draw_definition()


@functools.cache
def _band_d():
    return invented_band_d()


def _inputs(household: pd.DataFrame) -> UKSPIHousingCouncilTaxInputs:
    """The SPI-channel rows' council-tax inputs, built from ``household``
    itself on the toy supports at the invented Band D."""

    return uk_spi_housing_council_tax_inputs(
        household, definition=_definition(), supports=_supports(), band_d=_band_d()
    )


def _transform(stage=None) -> UKSPIHousingShellStageTransform:
    """The stage transform on the toy supports' bytes at the invented Band D."""

    return UKSPIHousingShellStageTransform(
        stage=_stage() if stage is None else stage,
        supports=_payloads(),
        n_estimators=20,
        band_d=_band_d(),
    )


def _bits(values: object) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64).view(np.int64)


def _reference(region: str) -> Fraction:
    """The invented Band D of the region's toy support areas, by households:
    the expected Band D of a copy's L3 cell, worked again from the toys."""

    band_d = _band_d().band_d
    areas = [
        (int(households), band_d[str(code)])
        for support in _supports().values()
        for frs_region, code, households in zip(
            support.arrays["frs_region"],
            support.arrays[UK_LOCAL_AUTHORITY_CODE_COLUMN],
            support.arrays["households"],
            strict=True,
        )
        if str(frs_region) == region and int(households) > 0
    ]
    return sum((count * value for count, value in areas), Fraction(0)) / sum(
        count for count, _ in areas
    )


def _statutes(households: pd.DataFrame) -> np.ndarray:
    """Each Great Britain row's 2024-25 statute bill at its region's
    reference, for its drawn band, from the statute itself."""

    return np.asarray(
        [
            float(
                uk_statutory_council_tax(
                    _reference(str(region)),
                    UK_NATION_BY_REGION[str(region)],
                    str(band),
                    single_adult=bool(float(adults) == 1.0),
                )
            )
            for region, band, adults in zip(
                households["region"],
                households["council_tax_band"],
                households["council_tax_single_adult_raw"],
                strict=True,
            )
        ],
        dtype=np.float64,
    )


def _leaves(value: object) -> Iterator[object]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


#: The digits a refusal may print: the Band D's fiscal year and build
#: period, the issue, and dtype names, as the localisation's tests declare
#: them. Copied: under pytest's importlib mode one test module cannot import
#: another.
_DECLARED_DIGITS = ("microcosm#931", "2024-25", "2024", "float64", "int64")


def _assert_names_no_household(message: str, household: pd.DataFrame) -> None:
    text = message
    for token in _DECLARED_DIGITS:
        text = text.replace(token, "")
    assert not re.search(r"[0-9]", text), message
    for column in (IDENTITY_COLUMN, "atomic_area_code", UK_LOCAL_AUTHORITY_CODE_COLUMN):
        if column not in household.columns:
            continue
        values = {
            value for value in household[column] if isinstance(value, str) and value
        }
        assert not any(value in message for value in values), column


def _northern_ireland_billed_tables():
    """The shell's tables with invented council tax on Northern Ireland's FRS
    rows, and an invented council tax reduction on every third FRS reference
    person in both nations, large on an even household and small on an odd
    one, so that some drawn reductions exceed their bill and some do not."""

    person, benunit, household, weights = housing_shell_tables()
    ids = household["household_id"].to_numpy()
    frs = (household["household_support_channel"] == "frs").to_numpy()
    billed = frs & (household["region"] == _NORTHERN_IRELAND).to_numpy()
    household.loc[billed, "council_tax"] = 600.0 + 25.0 * (ids[billed] % 7)
    household.loc[billed, "council_tax_reported"] = household.loc[billed, "council_tax"]
    reduced = (
        person["is_household_head"]
        & person["person_household_id"].isin(ids[frs & (ids % 3 == 0)])
    ).to_numpy()
    person.loc[reduced, "council_tax_benefit_reported"] = np.where(
        person.loc[reduced, "person_household_id"].to_numpy() % 2 == 0, 3_500.0, 250.0
    )
    return person, benunit, household, weights


_TABLES = {
    "invented_tables": housing_shell_tables,
    "northern_ireland_billed": _northern_ireland_billed_tables,
}


@dataclass(frozen=True)
class _TodaysPath:
    """The shell's fit and draw before the statute, as today's stage made
    them on a table without geographic clones: the FRS rows train at their
    weights at the stage's seed. ``supported`` is the draw after the support
    rules, ``capped`` after ``apply_structural_rules``, with ``fired`` its
    firings and the band redraws after them; each is on the SPI-channel rows'
    household ids in table order."""

    recipient_ids: list
    model: FittedHousingModel
    raw: pd.DataFrame
    supported: pd.DataFrame
    capped: pd.DataFrame
    fired: dict[str, int]


def _todays_path(
    person: pd.DataFrame, household: pd.DataFrame, weights: np.ndarray
) -> _TodaysPath:
    predictors = household_housing_predictors(person, household)
    targets = household_housing_targets(person, household)
    channel = household["household_support_channel"].astype(str).to_numpy()
    by_id = pd.Series(weights, index=household["household_id"])
    training = household.loc[(channel == "frs") & (weights > 0.0), "household_id"]
    recipients = household.loc[channel == _SPI, "household_id"]
    model = fit_housing_model(
        predictors.loc[training],
        targets.loc[training],
        by_id.loc[training].to_numpy(),
        seed=0,
        n_estimators=20,
    )
    redrawn: dict[str, int] = {}
    raw = model.draw(predictors.loc[recipients], firings=redrawn)
    supported, _ = apply_support_rules(
        raw, predictors["region"], model.structural_zeros
    )
    capped, fired = apply_structural_rules(
        raw, predictors["region"], model.structural_zeros
    )
    return _TodaysPath(
        recipient_ids=recipients.tolist(),
        model=model,
        raw=raw,
        supported=supported,
        capped=capped,
        fired={**fired, **redrawn},
    )


def test_packaged_stage_matches_the_module_contract() -> None:
    _assert_stage_parameters(_stage())
    assert _stage().rewrites == UK_SPI_HOUSING_SHELL_REWRITES
    # The Scottish verify operation, the imputation verbatim and the derive
    # of the Great Britain bill, in that order, and the two Band D artifacts.
    assert [(op.kind, dict(op.parameters)) for op in _stage().operations] == [
        (
            "verify_vendored_fact_resource",
            uk_spi_housing_council_tax_verify_parameters(),
        ),
        ("impute_spi_housing_shell", UK_SPI_HOUSING_SHELL_OPERATION),
        ("derive", uk_spi_housing_council_tax_derive_parameters()),
    ]
    assert (
        tuple(
            (artifact["role"], artifact["resource"]) for artifact in _stage().artifacts
        )
        == UK_SPI_HOUSING_COUNCIL_TAX_ARTIFACTS
    )


def test_frs_rows_are_untouched_and_the_invariants_hold() -> None:
    person, _, household, weights = housing_shell_tables()
    person_out, household_out, receipt = impute_spi_housing_shell(
        person,
        household,
        weights,
        council_tax_inputs=_inputs(household),
        seed=0,
        n_estimators=20,
    )
    _assert_invariants(person, household, person_out, household_out)
    frs = household["household_support_channel"] == "frs"
    pd.testing.assert_frame_equal(household.loc[frs], household_out.loc[frs])
    assert receipt["training_households"] == int(frs.sum())
    assert receipt["recipient_households"] == int((~frs).sum())


def test_spi_housing_follows_the_recipients_own_income() -> None:
    person, _, household, weights = housing_shell_tables()
    _, household_out, _ = impute_spi_housing_shell(
        person,
        household,
        weights,
        council_tax_inputs=_inputs(household),
        seed=0,
        n_estimators=20,
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
        person,
        household,
        weights,
        council_tax_inputs=_inputs(household),
        seed=0,
        n_estimators=20,
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
        person,
        household,
        weights,
        council_tax_inputs=_inputs(household),
        seed=0,
        n_estimators=20,
    )
    order = np.random.default_rng(1).permutation(len(household))
    permuted = household.iloc[order].reset_index(drop=True)
    _, second, _ = impute_spi_housing_shell(
        person,
        permuted,
        weights[order],
        council_tax_inputs=_inputs(permuted),
        seed=0,
        n_estimators=20,
    )
    columns = [*HOUSING_CATEGORICAL_COLUMNS, *HOUSING_AMOUNT_COLUMNS]
    left = first.set_index("household_id")[columns].sort_index()
    right = second.set_index("household_id")[columns].sort_index()
    pd.testing.assert_frame_equal(left, right)


def test_stage_transform_runs_on_a_frame_and_records_its_receipt(monkeypatch) -> None:
    person, benunit, household, weights = housing_shell_tables()
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=weights,
    )
    transform = _transform()
    calls = []
    real = housing_shell._assert_invariants

    def counted(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(housing_shell, "_assert_invariants", counted)
    result = transform(frame)
    # The transform checks the housing invariants once.
    assert len(calls) == 1
    assert transform.output_columns() == ()
    evidence = transform.checkpoint_metadata()["evidence"]
    assert evidence["stage"] == "spi_housing_shell"
    assert "rule_firings" in evidence
    np.testing.assert_array_equal(result.weights_for("household").values, weights)
    # The bill's block, as the frame's own household table gives it, and each
    # Great Britain copy's bill at the statute of its reference.
    assert evidence[UK_SPI_HOUSING_COUNCIL_TAX_EVIDENCE] == (
        uk_spi_housing_council_tax_evidence(
            _inputs(frame.table("household")),
            checks=dict.fromkeys(UK_SPI_HOUSING_COUNCIL_TAX_CHECKS, True),
        )
    )
    out = result.table("household")
    gb = (out["household_support_channel"] == _SPI) & (
        out["region"] != _NORTHERN_IRELAND
    )
    assert gb.any()
    assert np.array_equal(_bits(out.loc[gb, "council_tax"]), _bits(_statutes(out[gb])))
    # The area draw's definition at seed 0 on the supports given, decoded.
    assert transform.definition == uk_conditioned_assignment_definition(
        _payloads(), seed=0
    )
    assert {
        system: support.sha256 for system, support in transform.decoded_supports.items()
    } == {system: support.sha256 for system, support in _supports().items()}


def test_a_frame_without_frs_training_rows_is_refused() -> None:
    person, _, household, weights = housing_shell_tables()
    household = household.assign(
        household_support_channel="spi", household_support_clone_index=1
    )
    with pytest.raises(ValueError, match="FRS training rows"):
        impute_spi_housing_shell(
            person,
            household,
            weights,
            council_tax_inputs=_inputs(household),
            seed=0,
            n_estimators=20,
        )


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
        person,
        household,
        weights,
        council_tax_inputs=_inputs(household),
        seed=0,
        n_estimators=20,
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
    # restriction is removed, so the test above can fail. Band I then reaches
    # English rows, whose bill the statute refuses to price, with the bill
    # helper's own message (microcosm#931).
    monkeypatch.setattr(
        housing_shell,
        "restrict_bands_to_nation",
        lambda choice, *_args: (choice, {}),
    )
    person, _, household, weights = _band_i_everywhere()
    with pytest.raises(
        ValueError, match="some row's band is not one of them"
    ) as caught:
        impute_spi_housing_shell(
            person,
            household,
            weights,
            council_tax_inputs=_inputs(household),
            seed=0,
            n_estimators=20,
        )
    _assert_names_no_household(str(caught.value), household)


def test_geographic_clones_train_the_model_as_their_survey_household() -> None:
    """Two geographic clones of every household at half its weight fit the
    model each survey household fits alone, so every SPI copy's first clone
    draws the housing it draws without clones (microcosm#931)."""

    person, benunit, household, weights = housing_shell_tables()
    pool = with_geographic_clones(
        uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            household_weights=weights,
            time_period="2024",
        )
    )
    _, alone, alone_receipt = impute_spi_housing_shell(
        person,
        household,
        weights,
        council_tax_inputs=_inputs(household),
        seed=0,
        n_estimators=20,
    )
    _, cloned, receipt = impute_spi_housing_shell(
        pool.table("person"),
        pool.table("household"),
        pool.weights_for("household").values,
        council_tax_inputs=_inputs(pool.table("household")),
        seed=0,
        n_estimators=20,
    )
    columns = [*HOUSING_CATEGORICAL_COLUMNS, *HOUSING_AMOUNT_COLUMNS]
    first = cloned.loc[cloned["household_clone_index"] == 0]
    pd.testing.assert_frame_equal(
        first.set_index("household_id")[columns],
        alone.set_index("household_id")[columns],
    )
    assert receipt["training_households"] == alone_receipt["training_households"]
    assert {
        column: shares["training"]
        for column, shares in receipt["category_shares"].items()
    } == {
        column: shares["training"]
        for column, shares in alone_receipt["category_shares"].items()
    }


def test_the_models_fold_receipt_is_absent_at_one_clone_and_rounded_at_three() -> None:
    """The stage records how its model's geographic fold met the FRS
    households only on more than one clone, so a one-clone frame's receipt
    keeps its bytes; the counts are true source households, rounded
    (microcosm#931)."""

    person, benunit, household, weights = housing_shell_tables()
    _, _, alone = impute_spi_housing_shell(
        person,
        household,
        weights,
        council_tax_inputs=_inputs(household),
        seed=0,
        n_estimators=5,
    )
    assert GEOGRAPHIC_FOLD_EVIDENCE not in alone
    pool = with_geographic_clones(
        uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            household_weights=weights,
            time_period="2024",
        ),
        n_clones=3,
    )
    _, _, receipt = impute_spi_housing_shell(
        pool.table("person"),
        pool.table("household"),
        pool.weights_for("household").values,
        council_tax_inputs=_inputs(pool.table("household")),
        seed=0,
        n_estimators=5,
    )
    # The 600 FRS households train the model; their clones agree.
    assert receipt[GEOGRAPHIC_FOLD_EVIDENCE] == {
        "clone_count": 3,
        "records": 600,
        "varied_records": 0,
        "varied_share": 0,
    }
    # The bill's block is the same at every clone count.
    block = receipt[UK_SPI_HOUSING_COUNCIL_TAX_EVIDENCE]
    assert block == alone[UK_SPI_HOUSING_COUNCIL_TAX_EVIDENCE]


# --- Council tax on the SPI-channel rows (microcosm#931) -----------------------------


def _heads(person_out: pd.DataFrame, ids: list) -> pd.Series:
    """Each household's reference person's reported council tax reduction,
    on ``ids`` in their order."""

    heads = person_out.loc[person_out["is_household_head"].astype(bool)]
    return heads.set_index("person_household_id")["council_tax_benefit_reported"].loc[
        ids
    ]


@pytest.mark.parametrize("tables", sorted(_TABLES))
def test_only_great_britain_copies_bill_path_moves(tables) -> None:
    """Every output is today's fit, draw and structural rules but Great
    Britain copies' bill, reduction and rebate; Northern Ireland's rows are
    today's bit for bit (María's ruling of 2026-11-14), with zero and with
    positive training bills."""

    person, _, household, weights = _TABLES[tables]()
    today = _todays_path(person, household, weights)
    # Today's path is the module's own fit and draw, so the chain it fits is
    # pinned here: council_tax stays its first target on every row.
    assert today.model.amount_targets == _CHAIN
    assert today.model.amount_model.targets == list(_CHAIN)
    person_out, household_out, receipt = impute_spi_housing_shell(
        person,
        household,
        weights,
        council_tax_inputs=_inputs(household),
        seed=0,
        n_estimators=20,
    )
    ids = today.recipient_ids
    out = household_out.set_index("household_id").loc[ids]
    assert list(today.capped.index) == ids
    ni = (out["region"] == _NORTHERN_IRELAND).to_numpy()
    assert ni.any() and (~ni).any()
    # Every band and category, and every chained amount but the bill, the
    # chain's first target.
    for column in HOUSING_CATEGORICAL_COLUMNS:
        assert out[column].astype(str).tolist() == (
            today.capped[column].astype(str).tolist()
        ), column
    assert HOUSING_AMOUNT_COLUMNS[0] == "council_tax"
    for column in HOUSING_AMOUNT_COLUMNS[1:]:
        assert np.array_equal(_bits(out[column]), _bits(today.capped[column])), column
    heads = person_out.loc[person_out["is_household_head"].astype(bool)].set_index(
        "person_household_id"
    )
    assert np.array_equal(
        _bits(heads.loc[ids, "housing_benefit_reported"]),
        _bits(today.capped["housing_benefit_head_unit"]),
    )
    bill = out["council_tax"].to_numpy(dtype=np.float64)
    reduction = _heads(person_out, ids).to_numpy(dtype=np.float64)
    rebate = out["council_tax_rebate"].to_numpy(dtype=np.float64)
    # Northern Ireland: the chained bill, today's cap and rebate.
    assert np.array_equal(_bits(bill[ni]), _bits(today.capped["council_tax"][ni]))
    expected_ni = today.capped["council_tax_benefit_hrp"].to_numpy()[ni]
    assert np.array_equal(_bits(reduction[ni]), _bits(expected_ni))
    assert np.array_equal(_bits(rebate[ni]), _bits(expected_ni))
    # Great Britain: the statute at the reference, the drawn reduction capped
    # at it, the rebate equal to it.
    statute = _statutes(out.loc[~ni])
    assert np.array_equal(_bits(bill[~ni]), _bits(statute))
    drawn = today.supported["council_tax_benefit_hrp"].to_numpy()
    expected_gb = np.minimum(drawn[~ni], statute)
    assert np.array_equal(_bits(reduction[~ni]), _bits(expected_gb))
    assert np.array_equal(_bits(rebate[~ni]), _bits(expected_gb))
    assert not np.array_equal(_bits(bill[~ni]), _bits(today.capped["council_tax"][~ni]))
    # The cases each table plants.
    chained_ni = today.capped["council_tax"].to_numpy()[ni]
    if tables == "invented_tables":
        assert (chained_ni == 0.0).all() and (drawn == 0.0).all()
    else:
        assert (chained_ni > 0.0).any()
        assert ((expected_ni > 0.0) & (expected_ni < chained_ni)).any()
        assert (drawn[~ni] > statute).any()
        assert ((drawn[~ni] > 0.0) & (drawn[~ni] < statute)).any()
    # FRS rows as given.
    frs = (household["household_support_channel"] == "frs").to_numpy()
    pd.testing.assert_frame_equal(household.loc[frs], household_out.loc[frs])
    # The receipt: today's keys and forms, the cap's firing against the bill
    # written, and the bill's block.
    assert list(receipt) == [
        "training_households",
        "recipient_households",
        "structural_zero_rules",
        "rule_firings",
        "category_shares",
        UK_SPI_HOUSING_COUNCIL_TAX_EVIDENCE,
    ]
    assert receipt["training_households"] == int(frs.sum())
    assert receipt["recipient_households"] == len(ids)
    by_id = pd.Series(weights, index=household["household_id"])
    for column in HOUSING_CATEGORICAL_COLUMNS:
        assert receipt["category_shares"][column]["recipients"] == (
            housing_shell._weighted_shares(today.capped[column], by_id.loc[ids])
        ), column
    assert receipt["structural_zero_rules"] == {
        target: {key: list(levels) for key, levels in rules.items()}
        for target, rules in today.model.structural_zeros.items()
    }
    cap = "council_tax_benefit_hrp:cap"
    assert list(receipt["rule_firings"]) == list(today.fired)
    assert {k: v for k, v in receipt["rule_firings"].items() if k != cap} == {
        k: v for k, v in today.fired.items() if k != cap
    }
    assert receipt["rule_firings"][cap] == int((drawn > bill).sum())
    assert all(type(value) is int for value in receipt["rule_firings"].values())


def test_the_great_britain_bill_is_written_after_the_learned_zeros() -> None:
    """A tenure whose training rows hold no council tax gets the learned zero
    on the chained draw, counted as today, and the statute then prices its
    Great Britain rows; its Northern Ireland rows keep the zero."""

    person, _, household, weights = housing_shell_tables()
    outright = (household["household_support_channel"] == "frs") & (
        household["tenure_type"] == "OWNED_OUTRIGHT"
    )
    household.loc[outright, ["council_tax", "council_tax_reported"]] = 0.0
    today = _todays_path(person, household, weights)
    assert (
        "OWNED_OUTRIGHT" in today.model.structural_zeros["council_tax"]["tenure_type"]
    )
    _, household_out, receipt = impute_spi_housing_shell(
        person,
        household,
        weights,
        council_tax_inputs=_inputs(household),
        seed=0,
        n_estimators=20,
    )
    out = household_out.set_index("household_id").loc[today.recipient_ids]
    owned = (out["tenure_type"] == "OWNED_OUTRIGHT").to_numpy()
    ni = (out["region"] == _NORTHERN_IRELAND).to_numpy()
    assert (owned & ~ni).any() and (owned & ni).any()
    assert (today.supported["council_tax"].to_numpy()[owned] == 0.0).all()
    gb = owned & ~ni
    assert np.array_equal(
        _bits(out.loc[gb, "council_tax"]), _bits(_statutes(out.loc[gb]))
    )
    assert (out.loc[gb, "council_tax"] > 0.0).all()
    assert (out.loc[owned & ni, "council_tax"] == 0.0).all()
    # The zero rule's firings are today's: the statute is written after it.
    assert receipt["rule_firings"]["council_tax"] == today.fired["council_tax"]


def test_the_capped_reduction_is_the_reference_persons_and_the_rebate() -> None:
    person, _, household, weights = _northern_ireland_billed_tables()
    today = _todays_path(person, household, weights)
    person_out, household_out, _ = impute_spi_housing_shell(
        person,
        household,
        weights,
        council_tax_inputs=_inputs(household),
        seed=0,
        n_estimators=20,
    )
    ids = today.recipient_ids
    out = household_out.set_index("household_id").loc[ids]
    bill = out["council_tax"].to_numpy(dtype=np.float64)
    drawn = today.supported["council_tax_benefit_hrp"].to_numpy()
    capped = np.minimum(drawn, bill)
    assert np.array_equal(_bits(_heads(person_out, ids)), _bits(capped))
    assert np.array_equal(_bits(out["council_tax_rebate"]), _bits(capped))
    people = person_out.loc[person_out["person_household_id"].isin(ids)]
    others = people.loc[~people["is_household_head"].astype(bool)]
    assert len(others) and (others["council_tax_benefit_reported"] == 0.0).all()
    # The cap binds on some row, and some reduction sits below its bill.
    assert (drawn > bill).any()
    assert ((capped > 0.0) & (capped < bill)).any()


def test_apply_structural_rules_is_the_support_rules_then_the_cap() -> None:
    person, _, household, weights = _northern_ireland_billed_tables()
    today = _todays_path(person, household, weights)
    region = household_housing_predictors(person, household)["region"]
    zeros = today.model.structural_zeros
    given = today.raw.copy()
    capped, fired = apply_structural_rules(today.raw, region, zeros)
    supported, support_fired = apply_support_rules(today.raw, region, zeros)
    composed, cap_fired = cap_council_tax_benefit(supported)
    pd.testing.assert_frame_equal(capped, composed, check_exact=True)
    assert fired == {**support_fired, **cap_fired}
    # Today's firing keys, in today's order: the cap's last.
    assert list(fired) == [
        *zeros,
        "mortgage_interest_repayment:not_mortgaged",
        "mortgage_capital_repayment:not_mortgaged",
        "housing_benefit_head_unit:not_rented",
        "council_tax_benefit_hrp:cap",
    ]
    # Neither changes what it is given; the support rules leave the benefit
    # uncapped, and the cap changes nothing else: the minimum of the two, the
    # rows above the bill counted.
    pd.testing.assert_frame_equal(today.raw, given, check_exact=True)
    benefit = supported["council_tax_benefit_hrp"].to_numpy()
    bill = supported["council_tax"].to_numpy()
    assert (benefit > bill).any()
    assert cap_fired == {"council_tax_benefit_hrp:cap": int((benefit > bill).sum())}
    assert np.array_equal(
        _bits(composed["council_tax_benefit_hrp"]), _bits(np.minimum(benefit, bill))
    )
    pd.testing.assert_frame_equal(
        composed.drop(columns="council_tax_benefit_hrp"),
        supported.drop(columns="council_tax_benefit_hrp"),
        check_exact=True,
    )


def test_the_receipt_carries_the_council_tax_block_with_no_number_leaf() -> None:
    person, _, household, weights = housing_shell_tables()
    inputs = _inputs(household)
    _, _, receipt = impute_spi_housing_shell(
        person, household, weights, council_tax_inputs=inputs, seed=0, n_estimators=5
    )
    block = receipt[UK_SPI_HOUSING_COUNCIL_TAX_EVIDENCE]
    assert block == uk_spi_housing_council_tax_evidence(
        inputs, checks=dict.fromkeys(UK_SPI_HOUSING_COUNCIL_TAX_CHECKS, True)
    )
    leaves = list(_leaves(block))
    assert leaves and all(leaf is None or type(leaf) in (str, bool) for leaf in leaves)
    assert json.loads(json.dumps(block)) == block


def _one_check_fails(check: str):
    real = housing_shell.uk_spi_housing_council_tax_checks

    def checks(*args, **kwargs) -> dict[str, bool]:
        result = real(*args, **kwargs)
        assert all(result.values())
        return {**result, check: False}

    return checks


@pytest.mark.parametrize("check", UK_SPI_HOUSING_COUNCIL_TAX_CHECKS)
def test_a_failed_council_tax_check_refuses_with_its_fixed_message(
    check, monkeypatch
) -> None:
    assert set(housing_shell._COUNCIL_TAX_CHECK_REFUSALS) == set(
        UK_SPI_HOUSING_COUNCIL_TAX_CHECKS
    )
    monkeypatch.setattr(
        housing_shell, "uk_spi_housing_council_tax_checks", _one_check_fails(check)
    )
    messages = set()
    for tables in _TABLES.values():
        person, _, household, weights = tables()
        with pytest.raises(ValueError, match="failed its check") as caught:
            impute_spi_housing_shell(
                person,
                household,
                weights,
                council_tax_inputs=_inputs(household),
                seed=0,
                n_estimators=5,
            )
        message = str(caught.value)
        _assert_names_no_household(message, household)
        messages.add(message)
    # One fixed message for the check, whatever the table.
    assert messages == {
        f"Housing shell's council tax failed its check {check!r}: "
        f"{housing_shell._COUNCIL_TAX_CHECK_REFUSALS[check]}."
    }


def test_a_chained_great_britain_bill_fails_the_statute_check(monkeypatch) -> None:
    """Mutation check: a shell that kept the chained bill in Great Britain is
    refused by the statute check, with its fixed message."""

    monkeypatch.setattr(
        housing_shell,
        "uk_spi_council_tax_bills",
        lambda chained, **_kwargs: chained.to_numpy(dtype=np.float64),
    )
    person, _, household, weights = housing_shell_tables()
    with pytest.raises(
        ValueError, match="failed its check 'bill_at_statute_of_reference'"
    ) as caught:
        impute_spi_housing_shell(
            person,
            household,
            weights,
            council_tax_inputs=_inputs(household),
            seed=0,
            n_estimators=5,
        )
    _assert_names_no_household(str(caught.value), household)


def test_council_tax_inputs_are_required_and_are_the_spi_rows_in_table_order() -> None:
    parameter = inspect.signature(impute_spi_housing_shell).parameters[
        "council_tax_inputs"
    ]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    person, _, household, weights = housing_shell_tables()
    with pytest.raises(TypeError, match="required keyword-only argument"):
        impute_spi_housing_shell(person, household, weights, seed=0, n_estimators=5)
    # The shell's own refusal, before the inputs reach the bill's module.
    with pytest.raises(
        TypeError,
        match="Housing shell prices council tax with council_tax_inputs built by",
    ) as caught:
        impute_spi_housing_shell(
            person,
            household,
            weights,
            council_tax_inputs=object(),
            seed=0,
            n_estimators=5,
        )
    _assert_names_no_household(str(caught.value), household)
    spi = (household["household_support_channel"] == _SPI).to_numpy()
    # The same rows in another order, and one row fewer.
    others = (
        household.iloc[::-1].reset_index(drop=True),
        household.drop(index=household.index[spi][-1:]),
    )
    for other in others:
        with pytest.raises(
            ValueError, match="household_id values in table order, and these do not"
        ) as caught:
            impute_spi_housing_shell(
                person,
                household,
                weights,
                council_tax_inputs=_inputs(other),
                seed=0,
                n_estimators=5,
            )
        _assert_names_no_household(str(caught.value), household)
    # The same rows, whose first Northern Ireland copy reads a Great Britain
    # region: its bill's nation would not be its own.
    moved = household.copy()
    northern_ireland = spi & (moved["region"] == _NORTHERN_IRELAND).to_numpy()
    assert northern_ireland.any()
    moved.loc[moved.index[northern_ireland][:1], "region"] = "LONDON"
    with pytest.raises(
        ValueError, match="region values in table order, and these do not"
    ) as caught:
        impute_spi_housing_shell(
            person,
            household,
            weights,
            council_tax_inputs=_inputs(moved),
            seed=0,
            n_estimators=5,
        )
    _assert_names_no_household(str(caught.value), household)


def test_the_transform_checks_scotlands_band_d_period_before_it_reads_a_row(
    monkeypatch,
) -> None:
    import microcosm.build.uk_runtime.national_frame as national_frame

    person, benunit, household, weights = housing_shell_tables()
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2025",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=weights,
    )
    transform = _transform()

    def unread(*_args, **_kwargs):
        raise AssertionError("the transform read the frame")

    for module, name in (
        (national_frame, "validate_uk_national_frame"),
        (housing_shell, "uk_spi_housing_council_tax_inputs"),
        (housing_shell, "impute_spi_housing_shell"),
    ):
        monkeypatch.setattr(module, name, unread)
    with pytest.raises(UKCouncilTaxBandDError, match="maps Scotland's Band D"):
        transform(frame)


def _operations(stage, *kinds: str) -> tuple[SourceOperationSpec, ...]:
    by_kind = {operation.kind: operation for operation in stage.operations}
    return tuple(by_kind[kind] for kind in kinds)


def _edited_formula(stage):
    verify, impute, derive = stage.operations
    edited = SourceOperationSpec(
        kind=derive.kind, parameters={**derive.parameters, "formula": "edited"}
    )
    return replace(stage, operations=(verify, impute, edited))


_DRIFTS = {
    "another_stage": (
        lambda stage: replace(stage, stage="private_rent"),
        "Housing shell received stage",
    ),
    "without_the_verify_operation": (
        lambda stage: replace(
            stage,
            operations=_operations(stage, "impute_spi_housing_shell", "derive"),
        ),
        "operation order drifted",
    ),
    "derive_before_impute": (
        lambda stage: replace(
            stage,
            operations=_operations(
                stage,
                "verify_vendored_fact_resource",
                "derive",
                "impute_spi_housing_shell",
            ),
        ),
        "operation order drifted",
    ),
    "an_edited_formula": (_edited_formula, "derive declaration drifted"),
    "an_output": (
        lambda stage: replace(stage, outputs=("council_tax",)),
        "must declare no new outputs",
    ),
    "a_missing_artifact": (
        lambda stage: replace(stage, artifacts=stage.artifacts[:1]),
        "reads the artifacts",
    ),
    "an_artifact_without_its_flag": (
        lambda stage: replace(
            stage,
            artifacts=(
                {**stage.artifacts[0], "runtime_sha256_required": False},
                *stage.artifacts[1:],
            ),
        ),
        "reads the artifacts",
    ),
    "another_artifact": (
        lambda stage: replace(
            stage,
            artifacts=(*stage.artifacts, {**stage.artifacts[0], "role": "invented"}),
        ),
        "reads the artifacts",
    ),
    "another_artifact_without_the_flag": (
        lambda stage: replace(
            stage,
            artifacts=(
                *stage.artifacts,
                {
                    **stage.artifacts[0],
                    "role": "invented",
                    "runtime_sha256_required": False,
                },
            ),
        ),
        "reads the artifacts",
    ),
}


@pytest.mark.parametrize("drift", sorted(_DRIFTS))
def test_a_drifted_stage_is_refused_before_the_transform_runs(drift) -> None:
    change, text = _DRIFTS[drift]
    stage = change(_stage())
    with pytest.raises(ValueError, match=text):
        _assert_stage_parameters(stage)
    with pytest.raises(ValueError, match=text):
        _transform(stage)


def test_the_shell_imports_no_council_tax_stage_module() -> None:
    """The council-tax localisation stays the one step that localises the
    bill: the shell imports the bill's module, not the localisation's or
    stage 5's."""

    tree = ast.parse(inspect.getsource(housing_shell))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "microcosm.build.uk_runtime.spi_housing_council_tax" in imported
    assert not {name.rsplit(".", 1)[-1] for name in imported} & {
        "spine_council_tax",
        "frs_council_tax",
    }


def test_committed_smoke_fixture_uses_current_housing_shell_contract() -> None:
    # The integration smoke driver and the H2 registry rebuild this frozen
    # stage descriptor, separately from the packaged country spec checked
    # above.
    path = (
        paths_for("microcosm-graph").tests
        / "fixtures/parity/uk_spine/sources/fixture.json"
    )
    descriptor = json.loads(path.read_text())
    stage = SourceStageSpec.from_mapping(
        descriptor["stages"][UK_SPI_HOUSING_SHELL_STAGE_NAME]
    )
    _assert_stage_parameters(stage)


def test_the_fixture_builders_build_the_shell_on_the_toy_supports() -> None:
    """The H2 registry (graph_kernels) and the fixture tool's legacy oracle
    each build the housing shell on the fixture's toy supports: the tool's
    oracle runs only when the H2 fixture regenerates, so this holds its entry
    engine-free (microcosm#931)."""

    repository = paths_for("microcosm-build").repository
    builders = {
        "packages/microcosm-build/src/microcosm/build/uk_runtime/graph_kernels.py": (
            "_fixture_implementations"
        ),
        "tools/graph_uk_spine_fixture.py": "_build_implementations",
    }
    for relative, name in builders.items():
        tree = ast.parse((repository / relative).read_text(encoding="utf-8"))
        (builder,) = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        (call,) = [
            node
            for node in ast.walk(builder)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "UKSPIHousingShellStageTransform"
        ]
        given = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
        assert given.get("supports") == "supports", name
