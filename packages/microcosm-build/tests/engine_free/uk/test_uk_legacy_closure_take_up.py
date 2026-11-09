"""Universal Credit claims at the legacy closure, ported from uk-data#534
(microcosm#1095)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime import legacy_closure_take_up as lc
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind

_CLAIMED = "Claimed UC"
_CLOSED = "Did not claim UC - Legacy benefit closed"
_IN_PROGRESS = "In Progress"


def _row(legacy_benefit: str, move_status: str, value: float) -> dict:
    return {
        "concept": "dwp.mtuc_households_sent_migration_notice",
        "dimensions": {"legacy_benefit": legacy_benefit, "move_status": move_status},
        "value": value,
    }


def _rows() -> list[dict]:
    rows = []
    for label, claimed, closed, in_progress in (
        ("Child Tax Credit only", 600.0, 400.0, 50.0),
        ("Child Tax Credit and Housing Benefit", 900.0, 100.0, 5.0),
        ("Employment Support Allowance only", 950.0, 50.0, 0.0),
        ("all", 2_450.0, 550.0, 55.0),
    ):
        rows.append(_row(label, _CLAIMED, claimed))
        rows.append(_row(label, _CLOSED, closed))
        rows.append(_row(label, _IN_PROGRESS, in_progress))
    return rows


def test_rates_are_claimed_over_claimed_and_closed() -> None:
    rates = lc.move_to_uc_claim_rates_from_rows(_rows())
    # Households still in progress are left out (uk-data#534).
    assert rates == {
        "all": pytest.approx(2_450 / 3_000),
        "child_tax_credit": pytest.approx(0.6),
        "child_tax_credit+housing_benefit": pytest.approx(0.9),
        "esa_income": pytest.approx(0.95),
    }


def test_combination_keys_follow_the_engine_order() -> None:
    rows = [
        row
        for row in _rows()
        if row["dimensions"]["legacy_benefit"] != "Child Tax Credit and Housing Benefit"
    ]
    rows.append(_row("Housing Benefit and Child Tax Credit", _CLAIMED, 9.0))
    rows.append(_row("Housing Benefit and Child Tax Credit", _CLOSED, 1.0))
    rates = lc.move_to_uc_claim_rates_from_rows(rows)
    assert rates["child_tax_credit+housing_benefit"] == pytest.approx(0.9)


def test_rates_refuse_unknown_labels_repeats_and_a_missing_all_row() -> None:
    with pytest.raises(ValueError, match="Unknown legacy benefit"):
        lc.move_to_uc_claim_rates_from_rows(
            [*_rows(), _row("Pension Credit only", _CLAIMED, 1.0)]
        )
    with pytest.raises(ValueError, match="repeat"):
        lc.move_to_uc_claim_rates_from_rows([*_rows(), _row("all", _CLAIMED, 1.0)])
    no_all = [row for row in _rows() if row["dimensions"]["legacy_benefit"] != "all"]
    with pytest.raises(ValueError, match="no all-households row"):
        lc.move_to_uc_claim_rates_from_rows(no_all)
    no_closed = [
        row
        for row in _rows()
        if row["dimensions"]
        != {"legacy_benefit": "Child Tax Credit only", "move_status": _CLOSED}
    ]
    with pytest.raises(ValueError, match="lack the claimed and closed counts"):
        lc.move_to_uc_claim_rates_from_rows(no_closed)


def test_probabilities_hit_the_rate_at_the_weighted_share_already_claiming() -> None:
    combination = np.array(["child_tax_credit"] * 4 + ["jsa_income"] * 2, dtype=object)
    drawn = np.ones(6, dtype=bool)
    claims_anyway = np.array([True, False, False, False, False, False])
    # The claimant carries half the cohort's weight: weighted p = 0.5, where
    # the unweighted share would be 0.25.
    weights = np.array([3.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    probability, receipt = lc.legacy_closure_claim_probabilities(
        combination=combination,
        drawn=drawn,
        claims_anyway=claims_anyway,
        weights=weights,
        rates={"child_tax_credit": 0.8, "all": 0.6},
        great_britain=np.ones(6, dtype=bool),
        source_ids=np.arange(6),
    )
    np.testing.assert_allclose(probability[1:4], (0.8 - 0.5) / (1 - 0.5))
    # A combination DWP does not report takes the all-households rate.
    np.testing.assert_allclose(probability[4:], 0.6)
    assert receipt["child_tax_credit"]["rate_source"] == "combination"
    assert receipt["jsa_income"]["rate_source"] == "all_households"
    # Four sources carry the tax credit cohort and two the JSA one.
    assert receipt["child_tax_credit"]["support"] == "3-9"
    assert receipt["jsa_income"]["support"] == "<3"


def test_probabilities_clip_where_the_claimants_exceed_the_rate() -> None:
    probability, _ = lc.legacy_closure_claim_probabilities(
        combination=np.array(["esa_income"] * 3, dtype=object),
        drawn=np.ones(3, dtype=bool),
        claims_anyway=np.array([True, True, False]),
        weights=np.ones(3),
        rates={"esa_income": 0.5, "all": 0.5},
        great_britain=np.ones(3, dtype=bool),
        source_ids=np.arange(3),
    )
    np.testing.assert_array_equal(probability, 0.0)


def test_a_cohort_counts_only_its_sources_of_positive_weight() -> None:
    """The fourth review of #1305: one carrier rule across the receipts.

    Twenty-four sources, five of them weighted: the cohort rests on five, so
    its support reads "3-9", not "10-29", and its flag, with one side of p on
    two weighted sources, reads "<3".
    """

    n = 24
    weights = np.zeros(n)
    weights[:5] = 1.0
    _, receipt = lc.legacy_closure_claim_probabilities(
        combination=np.array(["esa_income"] * n, dtype=object),
        drawn=np.ones(n, dtype=bool),
        claims_anyway=np.arange(n) % 2 == 0,
        weights=weights,
        rates={"esa_income": 0.5, "all": 0.5},
        great_britain=np.ones(n, dtype=bool),
        source_ids=np.arange(n),
    )
    assert receipt["esa_income"]["support"] == "3-9"
    assert receipt["esa_income"]["rate_unreachable"] == "<3"


_REPORTED = (
    *(f"{name}_reported" for name, _ in lc.LEGACY_BENEFITS),
    "universal_credit_reported",
)


def _frame(
    reported: dict[str, np.ndarray],
    would_claim_uc: np.ndarray,
    weights: np.ndarray,
    regions=None,
    *,
    source_ids=None,
    lineage: bool = True,
    time_period: str = "2024",
):
    """One person per benefit unit per household; ``source_ids`` makes support
    copies (each household's own id by default), ``lineage=False`` drops them."""

    n = len(would_claim_uc)
    ids = np.arange(1, n + 1, dtype="int64")
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_benunit_id": ids,
            "age": np.full(n, 40, dtype="int64"),
            # One person per unit, so each is its claimant.
            "is_claimant_or_partner": np.ones(n, dtype=bool),
            **{
                column: reported.get(column, np.zeros(n)).astype(float)
                for column in _REPORTED
            },
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": ids,
            "would_claim_uc": would_claim_uc,
            # A context cell every graph node receives.
            "frs_benunit_capital": np.zeros(n),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": ids,
            "region": pd.array(
                ["NORTH_WEST"] * n if regions is None else list(regions),
                dtype="string",
            ),
        }
    )
    if lineage:
        household["household_source_id"] = (
            ids if source_ids is None else np.asarray(source_ids, dtype="int64")
        )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=time_period,
        weight_kind=WeightKind.DESIGN,
        household_weights=weights,
    )


_RATES = {"child_tax_credit": 0.6, "child_tax_credit+housing_benefit": 0.9, "all": 0.8}


def test_units_outside_the_draw_keep_the_engine_default() -> None:
    n = 4
    frame = _frame(
        {
            # Unit 1 reports UC beside tax credits; unit 2 would claim anyway;
            # unit 3 reports no legacy benefit; unit 4 reports Child Tax Credit.
            "child_tax_credit_reported": np.array([500.0, 500.0, 0.0, 500.0]),
            "universal_credit_reported": np.array([800.0, 0.0, 0.0, 0.0]),
        },
        would_claim_uc=np.array([False, True, False, False]),
        weights=np.ones(n),
    )
    result = lc.assign_uc_claim_at_legacy_closure(frame, rates=_RATES)
    flag = result.frame.table("benunit")[lc.LEGACY_CLOSURE_OUTPUT].to_numpy()
    assert flag[:3].all()
    assert result.evidence()["combinations"] == {
        "child_tax_credit": {
            "rate": 0.6,
            "rate_source": "combination",
            "rate_unreachable": "<3",
            "support": "<3",
        }
    }


def test_only_the_claimant_and_partner_place_a_unit_in_the_draw() -> None:
    """The engine closes an award through its owners (pe-uk#2549), so a
    dependant's own award neither puts the family in the draw nor keeps it
    out as a Universal Credit unit."""

    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 5, dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2, 2], dtype="int64"),
            "person_benunit_id": np.asarray([1, 1, 2, 2], dtype="int64"),
            "age": np.asarray([45, 17, 40, 18], dtype="int64"),
            "is_claimant_or_partner": [True, False, True, False],
            **{column: np.zeros(4) for column in _REPORTED},
        }
    )
    # Unit 1: only its dependant reports income-related ESA, in their own
    # right. Unit 2: its claimant reports Child Tax Credit, and a member who
    # is neither claimant nor partner reports Universal Credit.
    person["esa_income_reported"] = [0.0, 300.0, 0.0, 0.0]
    person["child_tax_credit_reported"] = [0.0, 0.0, 500.0, 0.0]
    person["universal_credit_reported"] = [0.0, 0.0, 0.0, 800.0]
    benunit = pd.DataFrame(
        {
            "benunit_id": np.asarray([1, 2], dtype="int64"),
            "would_claim_uc": [False, False],
            "frs_benunit_capital": np.zeros(2),
        }
    )
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=pd.DataFrame(
            {
                "household_id": np.asarray([1, 2], dtype="int64"),
                "region": pd.array(["NORTH_WEST", "NORTH_WEST"], dtype="string"),
                "household_source_id": np.asarray([1, 2], dtype="int64"),
            }
        ),
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
        household_weights=np.ones(2),
    )

    assert list(lc.legacy_benefit_combinations(person, benunit)) == [
        "",
        "child_tax_credit",
    ]
    result = lc.assign_uc_claim_at_legacy_closure(frame, rates=_RATES)
    # Unit 1 is outside the draw and keeps the engine's default; unit 2 is
    # drawn on its claimant's Child Tax Credit alone.
    assert result.frame.table("benunit")[lc.LEGACY_CLOSURE_OUTPUT].to_numpy()[0]
    assert set(result.evidence()["combinations"]) == {"child_tax_credit"}

    without = person.drop(columns="is_claimant_or_partner")
    with pytest.raises(ValueError, match="is_claimant_or_partner"):
        lc.legacy_benefit_combinations(without, benunit)


def test_the_weighted_share_claiming_matches_dwps_rate() -> None:
    rng = np.random.default_rng(11)
    n = 6_000
    tax_credit = rng.random(n) < 0.7
    housing = tax_credit & (rng.random(n) < 0.3)
    frame = _frame(
        {
            "child_tax_credit_reported": np.where(tax_credit, 400.0, 0.0),
            "housing_benefit_reported": np.where(housing, 300.0, 0.0),
        },
        would_claim_uc=rng.random(n) < 0.2,
        weights=rng.uniform(0.5, 3.0, n),
    )
    result = lc.assign_uc_claim_at_legacy_closure(frame, rates=_RATES)
    benunit = result.frame.table("benunit")
    flag = benunit[lc.LEGACY_CLOSURE_OUTPUT].to_numpy()
    weights = np.asarray(result.frame.weights_for("household").values)
    for mask, rate in (
        (tax_credit & ~housing, 0.6),
        (housing, 0.9),
    ):
        share = weights[mask & flag].sum() / weights[mask].sum()
        assert share == pytest.approx(rate, abs=0.03)
    assert result.evidence()["combinations"]["child_tax_credit"]["support"] == "30+"
    assert result.evidence()["p_basis"] == "spine_design_weights_great_britain"
    # Both sides of the share claiming rest on thousands of sources: it is given.
    share = result.evidence()["great_britain_weighted_share_claiming_at_closure"]
    assert isinstance(share, float)


def test_draws_are_keyed_by_benunit_identity() -> None:
    rng = np.random.default_rng(5)
    n = 300
    reported = {"child_tax_credit_reported": np.full(n, 400.0)}
    full = lc.assign_uc_claim_at_legacy_closure(
        _frame(reported, would_claim_uc=np.zeros(n, dtype=bool), weights=np.ones(n)),
        rates=_RATES,
    )
    again = lc.assign_uc_claim_at_legacy_closure(
        _frame(reported, would_claim_uc=np.zeros(n, dtype=bool), weights=np.ones(n)),
        rates=_RATES,
    )
    column = lc.LEGACY_CLOSURE_OUTPUT
    pd.testing.assert_series_equal(
        full.frame.table("benunit")[column], again.frame.table("benunit")[column]
    )
    # A unit's uniform reads its own id, so the stream does not depend on
    # which other units the frame carries.
    ids = np.arange(1, n + 1)
    subset = ids[rng.random(n) < 0.5]
    uniforms = stable_identity_uniforms(ids, seed=0, salt=column)
    np.testing.assert_array_equal(
        uniforms[subset - 1],
        stable_identity_uniforms(subset, seed=0, salt=column),
    )


def test_the_stage_refuses_a_stored_output_and_keeps_the_weights() -> None:
    n = 3
    frame = _frame(
        {"esa_income_reported": np.full(n, 100.0)},
        would_claim_uc=np.zeros(n, dtype=bool),
        weights=np.full(n, 2.0),
    )
    result = lc.assign_uc_claim_at_legacy_closure(frame, rates=_RATES)
    # A flag only: the weights and the mass log pass through, as in the other
    # benefit-unit take-up redraws.
    assert result.frame.mass_log == frame.mass_log
    np.testing.assert_array_equal(
        result.frame.weights_for("household").values,
        frame.weights_for("household").values,
    )
    with pytest.raises(ValueError, match="already stored"):
        lc.assign_uc_claim_at_legacy_closure(result.frame, rates=_RATES)


def test_the_packaged_rates_are_dwps_claimed_over_resolved_shares() -> None:
    rates = lc.load_move_to_uc_claim_rates()
    # Every combination DWP reports, and the all-households row.
    assert "all" in rates
    assert len(rates) == 20
    assert all(0.0 < rate <= 1.0 for rate in rates.values())
    assert set(rates) - {"all"} <= {
        "+".join(names)
        for size in range(1, 4)
        for names in itertools.combinations(
            [name for name, _ in lc.LEGACY_BENEFITS], size
        )
    }


def test_the_packaged_stage_carries_the_reviewed_operations_in_order() -> None:
    from microcosm.build.country_spec import load_country_spec

    stages = {stage.stage: stage for stage in load_country_spec("uk").sources.stages}
    stage = stages[lc.LEGACY_CLOSURE_STAGE_NAME]
    # Closed world: the transform refuses any drift from the code's constants.
    lc.UKLegacyClosureTakeUpStageTransform(stage=stage)
    order = list(stages)
    position = order.index(lc.LEGACY_CLOSURE_STAGE_NAME)
    # After the last writers of would_claim_uc (uc_take_up, microcosm#1334) and
    # the UC receipts, before the Pension Credit redraw and any CGT clone.
    assert order.index("uc_take_up") == position - 1
    assert order.index("uc_capital_coherence") == position - 2
    assert position < order.index("pension_credit_take_up")
    assert position < order.index("cgt_support_split")


def test_p_is_measured_over_great_britain_and_applied_in_northern_ireland() -> None:
    # Four Great Britain units, one claiming anyway (p = 0.25), and two Northern
    # Irish units claiming anyway, which a UK-wide p would count (0.5).
    combination = np.array(["child_tax_credit"] * 6, dtype=object)
    great_britain = np.array([True, True, True, True, False, False])
    probability, _ = lc.legacy_closure_claim_probabilities(
        combination=combination,
        drawn=np.ones(6, dtype=bool),
        claims_anyway=np.array([True, False, False, False, True, True]),
        weights=np.ones(6),
        rates={"child_tax_credit": 0.8, "all": 0.8},
        great_britain=great_britain,
        source_ids=np.arange(6),
    )
    np.testing.assert_allclose(probability, (0.8 - 0.25) / (1 - 0.25))


def test_an_unreachable_rate_is_flagged_only_on_three_or_more_sources_a_side() -> None:
    # Twenty of thirty units claim anyway: p (two thirds) exceeds DWP's rate,
    # so none is drawn, and both sides of p rest on three or more sources.
    kwargs = dict(
        combination=np.array(["esa_income"] * 30, dtype=object),
        drawn=np.ones(30, dtype=bool),
        claims_anyway=np.arange(30) < 20,
        weights=np.ones(30),
        rates={"esa_income": 0.5, "all": 0.5},
        great_britain=np.ones(30, dtype=bool),
    )
    probability, receipt = lc.legacy_closure_claim_probabilities(
        **kwargs, source_ids=np.arange(30)
    )
    np.testing.assert_array_equal(probability, 0.0)
    assert receipt["esa_income"]["rate_unreachable"] is True
    assert receipt["esa_income"]["support"] == "30+"
    # Thirty support copies of two households: the cell rests on fewer
    # than three sources, so neither the flag nor the support is given.
    _, receipt = lc.legacy_closure_claim_probabilities(
        **kwargs, source_ids=np.arange(30) % 2
    )
    assert receipt["esa_income"]["rate_unreachable"] == "<3"
    assert receipt["esa_income"]["support"] == "<3"
    # Thirty sources, one of them claiming anyway: the flag would say on which
    # side of the rate that one unit's weight share lies, so it is withheld.
    _, receipt = lc.legacy_closure_claim_probabilities(
        **dict(kwargs, claims_anyway=np.arange(30) < 1), source_ids=np.arange(30)
    )
    assert receipt["esa_income"]["rate_unreachable"] == "<3"


def test_a_combination_without_great_britain_units_has_no_flag() -> None:
    _, receipt = lc.legacy_closure_claim_probabilities(
        combination=np.array(["esa_income"] * 12, dtype=object),
        drawn=np.ones(12, dtype=bool),
        claims_anyway=np.zeros(12, dtype=bool),
        weights=np.ones(12),
        rates={"esa_income": 0.5, "all": 0.5},
        great_britain=np.zeros(12, dtype=bool),
        source_ids=np.arange(12),
    )
    # Northern Ireland only: there is no p, so no flag, not "<3".
    assert receipt["esa_income"]["rate_unreachable"] is None


def test_the_share_claiming_hides_a_small_side() -> None:
    # Forty Great Britain units on forty sources, two claiming anyway at a
    # rate they already exceed, so only those two claim: the share times the
    # drawn weight would give their weight back, so it is withheld.
    n = 40
    frame = _frame(
        {"child_tax_credit_reported": np.full(n, 400.0)},
        would_claim_uc=np.arange(n) < 2,
        weights=np.ones(n),
    )
    evidence = lc.assign_uc_claim_at_legacy_closure(
        frame, rates={"child_tax_credit": 0.01, "all": 0.8}
    ).evidence()
    assert evidence["great_britain_drawn_benunit_weight"] == pytest.approx(40.0)
    assert evidence["great_britain_weighted_share_claiming_at_closure"] == "<3"
    # The flag is withheld too: one side of p rests on two sources.
    assert evidence["combinations"]["child_tax_credit"]["rate_unreachable"] == "<3"


def test_the_receipt_aggregates_cover_great_britain_only() -> None:
    n = 40
    regions = ["NORTH_WEST"] * 30 + ["NORTHERN_IRELAND"] * 10
    frame = _frame(
        {"child_tax_credit_reported": np.full(n, 400.0)},
        would_claim_uc=np.zeros(n, dtype=bool),
        weights=np.full(n, 2.0),
        regions=regions,
    )
    result = lc.assign_uc_claim_at_legacy_closure(frame, rates=_RATES)
    # Thirty Great Britain units at weight 2; Northern Ireland's are left out.
    assert result.evidence()["great_britain_drawn_benunit_weight"] == pytest.approx(
        60.0
    )


def _graph_evidence(frame) -> dict:
    """The stage's receipt from the production node, run as a graph build runs it.

    The executor hands the kernel only the cells the node declares, so a
    column the stage reads but the node does not declare is missing here.
    """

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.graph import uk_spine_graph
    from microcosm.build.uk_runtime.graph_kernels import UKStageKernel
    from microcosm.graph.executor import _project_context
    from microcosm.graph.population import Population

    node = uk_spine_graph(source_mode="split").node(lc.LEGACY_CLOSURE_STAGE_NAME)
    context = _project_context(
        node,
        Population.from_frame(frame, "legacy-closure-input"),
        key="0" * 64,
        sources={},
        tolerances={},
        numerics={},
    )
    stage = load_country_spec("uk").sources.stage_map()[lc.LEGACY_CLOSURE_STAGE_NAME]
    transform = lc.UKLegacyClosureTakeUpStageTransform(stage=stage, rates=_RATES)
    UKStageKernel(lc.LEGACY_CLOSURE_STAGE_NAME, transform).run(context)
    return transform.checkpoint_metadata()["evidence"]


def _copies_of_two_households():
    # Twelve support copies of two source households, eleven claiming
    # anyway: p exceeds DWP's rate on twelve rows but on two sources.
    n = 12
    return _frame(
        {"esa_income_reported": np.full(n, 100.0)},
        would_claim_uc=np.arange(n) < 11,
        weights=np.ones(n),
        source_ids=np.arange(n) % 2,
    )


def test_the_stage_bands_support_copies_on_their_sources() -> None:
    # UKDS disclosure control: neither the flag nor a band above "<3" for a
    # cell two source households carry, however many rows copy them.
    combinations = lc.assign_uc_claim_at_legacy_closure(
        _copies_of_two_households(), rates=_RATES
    ).evidence()["combinations"]
    assert combinations["esa_income"]["rate_unreachable"] == "<3"
    assert combinations["esa_income"]["support"] == "<3"


def test_the_graph_node_bands_support_copies_on_their_sources() -> None:
    # The same cell through the executor's input projection: the node declares
    # household_source_id, so a graph build counts sources too.
    combinations = _graph_evidence(_copies_of_two_households())["combinations"]
    assert combinations["esa_income"]["rate_unreachable"] == "<3"
    assert combinations["esa_income"]["support"] == "<3"


def test_the_stage_refuses_a_frame_without_lineage() -> None:
    frame = _frame(
        {"esa_income_reported": np.full(3, 100.0)},
        would_claim_uc=np.zeros(3, dtype=bool),
        weights=np.ones(3),
        lineage=False,
    )
    with pytest.raises(ValueError, match="household_source_id"):
        lc.assign_uc_claim_at_legacy_closure(frame, rates=_RATES)


def test_the_stage_measures_p_over_great_britain() -> None:
    # Twenty Great Britain units, ten claiming anyway (p = 0.5, below the rate
    # of 0.6), and twenty Northern Irish units claiming anyway, which a
    # UK-wide p would count (0.75, above it): only the Great Britain basis
    # leaves the rate reachable. Each side of p rests on ten sources.
    n = 40
    frame = _frame(
        {"child_tax_credit_reported": np.full(n, 400.0)},
        would_claim_uc=np.r_[np.arange(20) < 10, np.ones(20, dtype=bool)],
        weights=np.ones(n),
        regions=["NORTH_WEST"] * 20 + ["NORTHERN_IRELAND"] * 20,
    )
    combinations = lc.assign_uc_claim_at_legacy_closure(frame, rates=_RATES).evidence()[
        "combinations"
    ]
    assert combinations["child_tax_credit"]["rate_unreachable"] is False


@pytest.mark.parametrize("period", ["2023", "2030", "2024-25"])
def test_the_transform_refuses_a_frame_outside_the_mapped_period(period) -> None:
    from microcosm.build.country_spec import load_country_spec

    stage = load_country_spec("uk").sources.stage_map()[lc.LEGACY_CLOSURE_STAGE_NAME]
    frame = _frame(
        {"esa_income_reported": np.full(3, 100.0)},
        would_claim_uc=np.zeros(3, dtype=bool),
        weights=np.ones(3),
        time_period=period,
    )
    # DWP's cumulative rates are mapped to build period 2024 only: an earlier
    # or a later year, or a fiscal-year label, is refused by name.
    with pytest.raises(ValueError, match="build period 2024"):
        lc.UKLegacyClosureTakeUpStageTransform(stage=stage, rates=_RATES)(frame)


def test_the_stage_refuses_a_null_lineage() -> None:
    frame = _frame(
        {"esa_income_reported": np.full(3, 100.0)},
        would_claim_uc=np.zeros(3, dtype=bool),
        weights=np.ones(3),
    )
    frame.table("household")["household_source_id"] = pd.array(
        [1, None, 3], dtype="Int64"
    )
    # A null would drop out of a distinct count or count as one source.
    with pytest.raises(ValueError, match="null"):
        lc.assign_uc_claim_at_legacy_closure(frame, rates=_RATES)
