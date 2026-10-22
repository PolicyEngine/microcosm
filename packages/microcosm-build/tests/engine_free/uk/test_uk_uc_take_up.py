"""Universal Credit take-up by award band and family type (microcosm#1334)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.frs_take_up import (
    UK_TAKE_UP_SIGNAL_OUTPUTS,
    UKTakeUpPopulationPolicy,
)
from microcosm.build.uk_runtime.measure_simulation import (
    _uc_calibration_composition,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract
from microcosm.build.uk_runtime.uc_deduction_attributes import (
    UC_DEDUCTION_OUTPUT_COLUMNS,
)
from microcosm.build.uk_runtime.uc_take_up import (
    UC_AWARD_BAND_LABELS,
    UC_FAMILY_TYPES,
    UC_TAKE_UP_ENTITLEMENT_VARIABLES,
    UC_TAKE_UP_MONTHS,
    UC_TAKE_UP_RESOURCE,
    UC_TAKE_UP_STAGE_NAME,
    UKUCTakeUpStageTransform,
    _assert_stage_parameters,
    assign_award_band_take_up,
    load_uc_award_band_targets,
    redraw_uc_take_up,
    uc_award_band_targets_from_rows,
    uc_award_bands,
    uc_family_types,
)
from microcosm.frame import WeightKind

_POLICY = UKTakeUpPopulationPolicy(
    adult_age=18, state_pension_age=66, instant="2024-01-01", source="test"
)


def _targets(**cells: float) -> dict[tuple[str, int], float]:
    """Every cell zero but the ones named ``FAMILY_band``."""

    targets = {
        (name, index): 0.0
        for name, _ in UC_FAMILY_TYPES
        for index in range(len(UC_AWARD_BAND_LABELS))
    }
    for key, value in cells.items():
        name, index = key.rsplit("_", 1)
        targets[(name, int(index))] = value
    return targets


def _assign(**overrides):
    count = len(overrides["award"])
    arguments = {
        "family": np.full(count, "SINGLE", dtype=object),
        "population": np.ones(count, dtype=bool),
        "reporter": np.zeros(count, dtype=bool),
        "moved": np.zeros(count, dtype=bool),
        "moved_claim": np.zeros(count),
        "previous": np.zeros(count, dtype=bool),
        "weights": np.ones(count),
        "draws": np.linspace(0.01, 0.99, count),
        "great_britain": np.ones(count, dtype=bool),
        "source_ids": np.arange(count),
        "minimum_pool_sources": 1,
        **overrides,
    }
    return assign_award_band_take_up(**arguments)


def test_award_bands_follow_the_registry_pence_edges() -> None:
    # (1,200 i, 1,200 (i + 1)] a year in whole pence, -1 for no award, and one
    # open band above GBP 30,000 (DWP's "GBP 2,500.01 or over").
    awards = np.asarray([0.0, 0.004, 0.006, 1200.0, 1200.01, 30_000.0, 30_000.01])
    assert uc_award_bands(awards).tolist() == [-1, -1, 0, 0, 1, 24, 25]
    assert UC_AWARD_BAND_LABELS[0] == "£0.01 to £100.00"
    assert UC_AWARD_BAND_LABELS[24] == "£2400.01 to £2500.00"
    assert UC_AWARD_BAND_LABELS[25] == "£2500.01 or over"


def _rows(value_for=lambda family, band, month: 10.0):
    rows = []
    for month in UC_TAKE_UP_MONTHS:
        for _, family in (*UC_FAMILY_TYPES, (None, "Unknown or missing family type")):
            for band in (*UC_AWARD_BAND_LABELS, "No payment"):
                rows.append(
                    {
                        "concept": "dwp.uc_households",
                        "measure_id": "benefit_units",
                        "dimensions": {
                            "family_type": family,
                            "monthly_award_amount_bands": band,
                        },
                        "period": {"type": "month", "value": month},
                        "value": value_for(family, band, month),
                    }
                )
    return rows


def test_targets_are_the_mean_of_the_nine_monthly_counts() -> None:
    months = {month: index for index, month in enumerate(UC_TAKE_UP_MONTHS)}
    targets = uc_award_band_targets_from_rows(
        _rows(lambda family, band, month: 9.0 * months[month])
    )
    # The unknown family type and the households with no payment are no cell.
    assert len(targets) == 4 * 26
    assert targets[("SINGLE", 0)] == pytest.approx(36.0)
    assert targets[("COUPLE_WITH_CHILDREN", 25)] == pytest.approx(36.0)

    short = [row for row in _rows() if row["period"]["value"] != "2025-12"]
    with pytest.raises(ValueError, match="every family type and award band"):
        uc_award_band_targets_from_rows(short)
    repeated = [*_rows(), _rows()[0]]
    with pytest.raises(ValueError, match="repeat"):
        uc_award_band_targets_from_rows(repeated)


def test_committed_targets_are_the_registry_rows_values() -> None:
    targets = load_uc_award_band_targets()
    assert len(targets) == 4 * 26
    # The registry's dwp/uc_payment_dist rows average the same months: their
    # 2025 values for two cells, as published facts (Stat-Xplore, GB).
    assert targets[("SINGLE", 0)] == pytest.approx(805_314 / 9)
    assert targets[("COUPLE_NO_CHILDREN", 0)] == pytest.approx(7_506.333, abs=1e-3)


def test_family_types_are_the_calibration_measures() -> None:
    # Single; lone parent with an under-20 child who is no UC child; couple
    # without children; couple with a UC child; a single parent's 20-year-old
    # non-dependent who is not a child for DWP.
    person = pd.DataFrame(
        {
            "person_id": range(9),
            "person_benunit_id": [1, 2, 2, 3, 3, 4, 4, 4, 5],
            "age": [30, 40, 19, 35, 36, 30, 31, 5, 45],
            "is_benunit_head": [1, 1, 0, 1, 0, 1, 0, 0, 1],
            "is_parent": [0, 1, 0, 0, 0, 1, 1, 0, 0],
        }
    )
    benunit = pd.DataFrame(
        {"benunit_id": [1, 2, 3, 4, 5], "dependent_children": [0, 1, 0, 1, 0]}
    )
    qualifying = np.asarray([0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=bool)
    frame = SimpleNamespace(
        table=lambda name: {"person": person, "benunit": benunit}[name]
    )
    simulation = SimpleNamespace(calculate=lambda variable, year: qualifying)
    measured, _ = _uc_calibration_composition(frame, simulation, 2025)

    families = uc_family_types(person, benunit, qualifying)
    assert families.tolist() == measured.tolist()
    assert families.tolist() == [
        "SINGLE",
        "LONE_PARENT",
        "COUPLE_NO_CHILDREN",
        "COUPLE_WITH_CHILDREN",
        "SINGLE",
    ]


def test_a_pool_realizes_dwp_count_with_reporters_first() -> None:
    # Ten entitled singles in band 0: two reporters of weight 2, eight
    # non-reporters of weight 1. DWP counts 7, so the drawn units must carry 3.
    award = np.full(10, 600.0)
    reporter = np.asarray([True, True] + [False] * 8)
    weights = np.asarray([2.0, 2.0] + [1.0] * 8)
    draws = np.asarray([0.9, 0.9, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
    would_claim, receipt = _assign(
        award=award,
        reporter=reporter,
        weights=weights,
        draws=draws,
        targets=_targets(SINGLE_0=7.0),
    )

    assert would_claim[reporter].all()
    # The three drawn units with the lowest uniforms claim, the rest do not.
    assert would_claim[2:].tolist() == [True, True, True] + [False] * 5
    assert float(weights[would_claim].sum()) == pytest.approx(7.0)
    single = receipt["families"]["SINGLE"]
    assert single["caseload_after"] == pytest.approx(7.0)
    assert single["reachable_pools_caseload_after"] == pytest.approx(7.0)
    assert single["reachable_pools_dwp_caseload"] == pytest.approx(7.0)
    assert receipt["reporters_not_claiming"] == 0
    assert receipt["changed_outside_draw"] == 0


def test_the_draw_keeps_earlier_claimants_when_the_rate_rises() -> None:
    # The early single-rate draw claimed the two lowest uniforms; a pool that
    # needs five claims keeps them and adds the next three.
    award = np.full(8, 2_000.0)
    draws = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    previous = draws < 0.25
    would_claim, _ = _assign(
        award=award, draws=draws, previous=previous, targets=_targets(SINGLE_1=5.0)
    )
    assert would_claim.tolist() == [True] * 5 + [False] * 3
    assert (would_claim | ~previous).all()


def test_a_pool_nearer_zero_than_its_first_unit_draws_no_claimant() -> None:
    # Two entitled non-reporters of weight 100 and a DWP count of 10: no
    # claimant (0) is nearer the count than one (100), so neither claims
    # (Vahid's review of #1344, item 15).
    award = np.full(2, 600.0)
    would_claim, _ = _assign(
        award=award,
        weights=np.asarray([100.0, 100.0]),
        draws=np.asarray([0.2, 0.7]),
        targets=_targets(SINGLE_0=10.0),
    )
    assert not would_claim.any()
    # One unit's weight past half the count, and the first claimant is nearer.
    would_claim, _ = _assign(
        award=award,
        weights=np.asarray([100.0, 100.0]),
        draws=np.asarray([0.2, 0.7]),
        targets=_targets(SINGLE_0=60.0),
    )
    assert would_claim.tolist() == [True, False]


def test_clipped_pools_are_flagged_unreachable() -> None:
    # Band 0: reporters alone exceed DWP's count, so no drawn unit claims.
    # Band 2: DWP's count exceeds every entitled unit, so all of them claim.
    award = np.asarray([100.0] * 4 + [3_000.0] * 4)
    reporter = np.asarray([True, True, True, False, False, False, False, False])
    would_claim, receipt = _assign(
        award=award,
        reporter=reporter,
        previous=np.asarray([False, False, False, True, False, False, False, False]),
        targets=_targets(SINGLE_0=2.0, SINGLE_2=10.0),
        source_ids=np.arange(8),
        minimum_pool_sources=1,
    )
    assert would_claim.tolist() == [True, True, True, False, True, True, True, True]
    flags = {
        tuple(pool["bands"]): pool["target_unreachable"]
        for pool in receipt["pools"]
        if pool["family_type"] == "SINGLE"
    }
    assert flags[(UC_AWARD_BAND_LABELS[0],)] == "<10"
    pools = receipt["pools"]
    assert all("rate" not in pool for pool in pools)
    # With ten or more sources behind a pool the flag says which side it is on.
    many = np.asarray([100.0] * 12 + [3_000.0] * 12)
    _, receipt = _assign(
        award=many,
        reporter=np.asarray([True] * 10 + [False] * 14),
        targets=_targets(SINGLE_0=2.0, SINGLE_2=50.0),
        minimum_pool_sources=1,
    )

    # Band 1 holds no unit, so it pools with band 2, and the empty bands above
    # join that pool.
    def flag(label):
        return next(
            pool["target_unreachable"]
            for pool in receipt["pools"]
            if pool["family_type"] == "SINGLE" and label in pool["bands"]
        )

    assert flag(UC_AWARD_BAND_LABELS[0]) == "undrawn_claims_reach_target"
    assert flag(UC_AWARD_BAND_LABELS[2]) == "target_exceeds_entitled"
    single = receipt["families"]["SINGLE"]
    # No pool is reachable, so the gate's reachable totals are empty.
    assert single["reachable_pools_dwp_caseload"] == 0.0


def test_undrawn_units_keep_their_flag_and_count_at_their_expected_claim() -> None:
    # Twice over: a unit with no award (keeps its early claim); one outside
    # the age population (keeps no claim); one moved off its legacy benefits,
    # with an expected claim of 0.5 at the closure (kept, counted at 0.5); and
    # four drawn units. DWP counts 5, so four of the eight drawn units claim.
    award = np.tile([0.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0], 2)
    draws = np.concatenate(
        [
            [0.9, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8],
            [0.9, 0.1, 0.1, 0.201, 0.401, 0.601, 0.801],
        ]
    )
    would_claim, receipt = _assign(
        award=award,
        population=np.tile([True, False, True, True, True, True, True], 2),
        moved=np.tile([False, False, True, False, False, False, False], 2),
        moved_claim=np.tile([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], 2),
        previous=np.tile([True, False, False, True, True, True, True], 2),
        draws=draws,
        targets=_targets(SINGLE_0=5.0),
    )
    assert would_claim.tolist() == [True, False, False, True, True, False, False] * 2
    single = receipt["families"]["SINGLE"]
    # Before: the moved units at 0.5 each and eight drawn claimants.
    assert single["caseload_before"] == pytest.approx(9.0)
    assert single["caseload_after"] == pytest.approx(5.0)
    assert receipt["changed_outside_draw"] == 0
    assert receipt["moved_units"] == "<10"


def test_rates_are_solved_over_great_britain_and_drawn_in_northern_ireland() -> None:
    # Four Great Britain units and four heavy Northern Ireland units in one
    # pool. DWP counts 2 in Great Britain: its two lowest uniforms claim, and
    # Northern Ireland's units claim below the same uniform.
    award = np.full(8, 900.0)
    gb = np.asarray([True] * 4 + [False] * 4)
    draws = np.asarray([0.1, 0.3, 0.5, 0.7, 0.05, 0.25, 0.35, 0.9])
    would_claim, receipt = _assign(
        award=award,
        great_britain=gb,
        weights=np.asarray([1.0] * 4 + [50.0] * 4),
        draws=draws,
        targets=_targets(SINGLE_0=2.0),
    )
    assert would_claim[gb].tolist() == [True, True, False, False]
    assert would_claim[~gb].tolist() == [True, True, False, False]


def test_bands_pool_until_the_drawn_units_rest_on_enough_sources() -> None:
    # Bands 0 and 1 each hold one source household's drawn unit, band 3 two,
    # band 5 one: with three sources a pool, bands 0 to 3 pool, and band 5's
    # short pool joins them.
    award = np.asarray([600.0, 1_800.0, 4_000.0, 4_100.0, 6_500.0])
    _, receipt = _assign(
        award=award,
        source_ids=np.asarray([1, 2, 3, 4, 5]),
        targets=_targets(SINGLE_0=1.0),
        minimum_pool_sources=3,
    )
    singles = [
        pool["bands"] for pool in receipt["pools"] if pool["family_type"] == "SINGLE"
    ]
    assert singles == [list(UC_AWARD_BAND_LABELS)]
    _, receipt = _assign(
        award=award,
        source_ids=np.asarray([1, 2, 3, 4, 5]),
        targets=_targets(SINGLE_0=1.0),
        minimum_pool_sources=2,
    )
    singles = [
        pool["bands"] for pool in receipt["pools"] if pool["family_type"] == "SINGLE"
    ]
    assert singles[0] == list(UC_AWARD_BAND_LABELS[:2])
    assert singles[1] == list(UC_AWARD_BAND_LABELS[2:])
    # Copies of one source household count once.
    _, receipt = _assign(
        award=award,
        source_ids=np.asarray([1, 1, 1, 1, 1]),
        targets=_targets(SINGLE_0=1.0),
        minimum_pool_sources=2,
    )
    assert [
        pool["bands"] for pool in receipt["pools"] if pool["family_type"] == "SINGLE"
    ] == [list(UC_AWARD_BAND_LABELS)]


def test_the_receipt_bands_counts_and_carries_no_rate() -> None:
    award = np.full(40, 1_500.0)
    _, receipt = _assign(
        award=award,
        weights=np.full(40, 3.0),
        targets=_targets(SINGLE_1=60.0),
    )
    # Counts of records are banded or rounded, never exact (UKDS CD137 cl. 8).
    assert receipt["drawn_units"] == 40
    assert receipt["drawn_units_northern_ireland"] == 0
    pool = next(
        pool
        for pool in receipt["pools"]
        if pool["family_type"] == "SINGLE" and pool["support"] != 0
    )
    assert pool["support"] == "30+"
    assert pool["drawn_support"] == "30+"
    assert set(pool) == {
        "family_type",
        "bands",
        "dwp_caseload",
        "support",
        "drawn_support",
        "target_unreachable",
    }
    _, small = _assign(award=np.full(5, 1_500.0), targets=_targets(SINGLE_1=2.0))
    assert small["drawn_units"] == "<10"
    assert small["families"]["SINGLE"]["caseload_after"] == "<10"


def test_inputs_must_align_and_name_known_cells() -> None:
    with pytest.raises(ValueError, match="must align"):
        _assign(award=np.ones(3), draws=np.ones(2), targets=_targets())
    with pytest.raises(ValueError, match="unknown"):
        _assign(
            award=np.ones(2),
            family=np.asarray(["SINGLE", "TRIPLE"], dtype=object),
            targets=_targets(),
        )
    with pytest.raises(ValueError, match="missing cells"):
        _assign(award=np.ones(2), targets={("SINGLE", 0): 1.0})


class _StubEngine:
    country = "uk"

    def __init__(self, awards: dict[int, float]) -> None:
        self.awards = awards
        self.calls: list[tuple[tuple[str, ...], str, str | None]] = []
        self.claims_seen: list[bool] | None = None
        self.deduction_columns_seen: bool | None = None

    def materialize(self, frame, variables, period, *, dataset_period=None):
        self.calls.append(
            (
                tuple(variables),
                str(period),
                None if dataset_period is None else str(dataset_period),
            )
        )
        benunit = frame.table("benunit")
        self.claims_seen = benunit["would_claim_uc"].tolist()
        self.deduction_columns_seen = set(UC_DEDUCTION_OUTPUT_COLUMNS) <= set(
            benunit.columns
        )
        ids = benunit["benunit_id"].to_numpy()
        person = frame.table("person")
        values = {
            "universal_credit": np.asarray([self.awards[int(i)] for i in ids]),
            "legacy_benefits_closed": np.zeros(len(ids), dtype=bool),
            "is_child_or_qualifying_young_person_for_universal_credit": (
                person["age"].to_numpy() < 16
            ),
        }
        return {name: values[name] for name in variables}


_LEGACY_REPORTS = (
    "child_tax_credit_reported",
    "working_tax_credit_reported",
    "housing_benefit_reported",
    "esa_income_reported",
    "income_support_reported",
    "jsa_income_reported",
)


def _frame(time_period: str = "2024"):
    # Units 1 to 3 in Wales, unit 4 in Northern Ireland; unit 1 reports UC,
    # unit 3 has no award.
    person = pd.DataFrame(
        {
            "person_id": [11, 21, 31, 41],
            "person_benunit_id": [1, 2, 3, 4],
            "person_household_id": [1, 2, 3, 4],
            "age": [30, 40, 50, 35],
            "is_uc_claimant": [True] * 4,
            "is_benunit_head": [True] * 4,
            "is_parent": [False] * 4,
            "universal_credit_reported": [1_000.0, 0.0, 0.0, 0.0],
            **{column: [0.0] * 4 for column in _LEGACY_REPORTS},
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": [1, 2, 3, 4],
            "dependent_children": [0, 0, 0, 0],
            "would_claim_uc": [True, True, True, False],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "region": ["WALES", "WALES", "WALES", "NORTHERN_IRELAND"],
            "household_source_id": [101, 102, 103, 104],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.asarray([10.0, 10.0, 10.0, 10.0]),
        weight_kind=WeightKind.IMPORTANCE,
        time_period=time_period,
    )


def _redraw(frame, engine):
    return redraw_uc_take_up(
        frame,
        engine=engine,
        population_policy=_POLICY,
        targets=_targets(SINGLE_0=10.0),
        closure_rates={"all": 0.8},
        minimum_pool_sources=1,
    )


def test_redraw_reads_the_paid_award_once_and_rewrites_only_would_claim_uc() -> None:
    engine = _StubEngine({1: 900.0, 2: 900.0, 3: 0.0, 4: 900.0})
    frame = _frame()

    result = _redraw(frame, engine)

    # One read at the calibration year on the frame stamped with its survey
    # year, every unit claiming and carrying its deduction attributes, so the
    # award is the one the engine pays a claimant.
    assert engine.calls == [(UC_TAKE_UP_ENTITLEMENT_VARIABLES, "2025", "2024")]
    assert engine.claims_seen == [True, True, True, True]
    assert engine.deduction_columns_seen is True
    after = result.frame.table("benunit")
    # The reporter's 10 meets DWP's 10, so the drawn unit 2 stops claiming and
    # Northern Ireland's unit 4 draws at the same rate; unit 3, with no award,
    # keeps its claim.
    assert after["would_claim_uc"].tolist() == [True, False, True, False]
    for entity in ("person", "household"):
        pd.testing.assert_frame_equal(
            result.frame.table(entity), frame.table(entity), check_exact=True
        )
    pd.testing.assert_frame_equal(
        after.drop(columns=["would_claim_uc"]),
        frame.table("benunit").drop(columns=["would_claim_uc"]),
    )
    evidence = result.evidence()
    assert evidence["stage"] == UC_TAKE_UP_STAGE_NAME
    assert evidence["entitlement_year"] == 2025
    assert evidence["solve_scope"] == "great_britain"
    assert evidence["population_state_pension_age"] == 66


def test_redraw_refuses_a_frame_outside_the_declared_dataset_year() -> None:
    engine = _StubEngine({1: 900.0, 2: 900.0, 3: 0.0, 4: 900.0})
    with pytest.raises(ValueError, match="survey_year 2024 values"):
        _redraw(_frame("2025"), engine)
    assert engine.calls == []


def _stage() -> SourceStageSpec:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()[UC_TAKE_UP_STAGE_NAME]


def test_committed_manifest_declares_the_stage_and_its_rewrite() -> None:
    stage = _stage()

    assert stage.grain == "benunit"
    assert stage.outputs == ()
    assert stage.rewrites == ("would_claim_uc",)
    _assert_stage_parameters(stage)
    assert [artifact["resource"] for artifact in stage.artifacts] == [
        UC_TAKE_UP_RESOURCE
    ]
    transform = UKUCTakeUpStageTransform(
        stage=stage,
        engine=_StubEngine({1: 900.0, 2: 900.0, 3: 0.0, 4: 900.0}),
        population_policy=_POLICY,
        targets=_targets(SINGLE_0=10.0),
        closure_rates={"all": 0.8},
    )
    transform(_frame())
    assert transform.checkpoint_metadata()["evidence"]["stage"] == UC_TAKE_UP_STAGE_NAME
    # The stage runs after the last writer of the capital and UC reports and
    # before the Move to UC draw reads would_claim_uc.
    order = [stage.stage for stage in load_country_spec("uk").sources.stages]
    position = order.index(UC_TAKE_UP_STAGE_NAME)
    assert order[position - 1] == "uc_capital_coherence"
    assert order[position + 1] == "uc_legacy_closure_take_up"


def test_stage_parameters_refuse_drift() -> None:
    stage = _stage()
    operations = []
    for operation in stage.operations:
        payload = {"kind": operation.kind, **dict(operation.parameters)}
        if operation.kind == "assign_award_band_take_up_residual":
            payload["minimum_pool_sources"] = 10
        operations.append(payload)
    drifted = SourceStageSpec.from_mapping({**stage.__dict__, "operations": operations})
    with pytest.raises(ValueError, match="declaration drifted"):
        _assert_stage_parameters(drifted)
    with pytest.raises(ValueError, match="expected 'uc_take_up'"):
        _assert_stage_parameters(replace(stage, stage="frs_take_up"))


def test_terminal_signal_gate_leaves_would_claim_uc_to_the_stage_gate() -> None:
    assert all(output != "would_claim_uc" for _, output, _ in UK_TAKE_UP_SIGNAL_OUTPUTS)


def test_contract_records_the_u8_ruling_and_keeps_the_early_rate() -> None:
    # María's ruling of 2026-10-21 lifts U8 for entitled units, which uc_take_up
    # draws by award band and family type; 0.55 stays the early draw's rate,
    # which a unit with no award keeps (microcosm#1334).
    contract = load_uk_take_up_contract()
    entry = contract.entry("universal_credit")
    assert contract.rate("universal_credit") == 0.55
    assert entry.source["status"] == "frozen_by_adjudication"
    freeze = entry.source["freeze"]
    assert freeze["approved_by"] == "juaristi22"
    assert freeze["approved_on"] == "2026-10-21"
    assert "microcosm#1334" in freeze["adjudication"]
    assert "uc_take_up" in freeze["decision"]
