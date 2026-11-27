"""Child Benefit claims and opt-outs after the SPI chain (microcosm#1063)."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime import stage_health
from microcosm.build.uk_runtime.child_benefit_take_up import (
    CHILD_BENEFIT_ENGINE_VARIABLES,
    CHILD_BENEFIT_STATISTICS_RESOURCE,
    CHILD_BENEFIT_TAKE_UP_STAGE_NAME,
    UKChildBenefitChargeThresholds,
    UKChildBenefitStatistics,
    UKChildBenefitTakeUpStageTransform,
    _assert_stage_parameters,
    assign_child_benefit_claims,
    assign_child_benefit_opt_outs,
    assign_spi_child_benefit_reports,
    child_benefit_copy_report_cells,
    child_benefit_family_key_ages,
    child_benefit_take_up_operation_parameters,
    load_child_benefit_statistics,
    redraw_child_benefit_take_up,
    solve_child_benefit_claim_rates,
)
from microcosm.build.uk_runtime.frs_take_up import UK_TAKE_UP_SIGNAL_OUTPUTS
from microcosm.build.uk_runtime.geographic_clones import ladder_clone_index_column
from microcosm.build.uk_runtime.ledger_fact_vendoring import load_vendor_selections
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind
from test_support.microcosm_build.uk_child_benefit_take_up import (
    STUB_ENTITLEMENT_PER_CHILD,
    _copy_frame,
    _frame,
    _statistics,
    _StubEngine,
)
from test_support.paths import paths_for

THRESHOLDS = UKChildBenefitChargeThresholds(
    phase_out_start=60_000.0, phase_out_end=80_000.0
)


def _families(count: int = 30_000, *, reporter_share: float = 0.3, seed: int = 5):
    """Families of one or two children aged 0 to 2, one row per child."""

    rng = np.random.default_rng(seed)
    eldest = rng.integers(0, 3, size=count)
    second = rng.random(count) < 0.5
    younger = np.where(second, rng.integers(0, 3, size=count), -1)
    younger = np.where(younger > eldest, eldest, younger)
    reporter = rng.random(count) < reporter_share
    units = np.r_[np.arange(count), np.flatnonzero(second)]
    ages = np.r_[eldest, younger[second]]
    return eldest, reporter, units, ages, rng.random(count)


def test_solved_rates_reproduce_the_published_rate_at_every_age() -> None:
    # Given families whose children's ages differ from the family's key age,
    # when the rates are solved from the eldest age down, then the expected
    # claimed share of the children of every age is the published one.
    eldest, reporter, units, ages, _ = _families()
    rates = {0: 0.7, 1: 0.8, 2: 0.9}
    weights = np.ones(len(units))
    residual, rows = solve_child_benefit_claim_rates(
        child_ages=ages,
        child_key_ages=eldest[units],
        child_reporter=reporter[units],
        child_weights=weights,
        claim_rates=rates,
    )
    assert [row["age"] for row in rows] == [0, 1, 2]
    assert not any(row["clipped"] for row in rows)
    for age, rate in rates.items():
        of_age = ages == age
        claimed = reporter[units][of_age].sum() + sum(
            residual[key] * (of_age & ~reporter[units] & (eldest[units] == key)).sum()
            for key in residual
        )
        assert claimed / of_age.sum() == pytest.approx(rate, abs=1e-12)
    # The eldest age has no older sibling: its rate is the plain residual.
    top = ages == 2
    assert residual[2] == pytest.approx(
        (0.9 * top.sum() - reporter[units][top].sum()) / (top & ~reporter[units]).sum()
    )


def test_solved_rates_are_weighted() -> None:
    # One age, a reporter child of weight 1 and a non-reporter of weight 3:
    # r = (0.5 * 4 - 1) / 3.
    residual, rows = solve_child_benefit_claim_rates(
        child_ages=np.asarray([0, 0]),
        child_key_ages=np.asarray([0, 0]),
        child_reporter=np.asarray([True, False]),
        child_weights=np.asarray([1.0, 3.0]),
        claim_rates={0: 0.5},
    )
    assert residual == {0: pytest.approx(1 / 3)}
    assert rows[0]["eligible_children"] == 4.0 and rows[0]["reporter_children"] == 1.0


def test_a_rate_the_draw_cannot_reach_is_clipped_and_receipted() -> None:
    # Reporters alone exceed the published rate at age 0; at age 1 every
    # family claiming still falls short of it.
    residual, rows = solve_child_benefit_claim_rates(
        child_ages=np.asarray([0, 0, 1, 1]),
        child_key_ages=np.asarray([0, 0, 1, 1]),
        child_reporter=np.asarray([True, False, False, False]),
        child_weights=np.asarray([9.0, 1.0, 1.0, 1.0]),
        claim_rates={0: 0.5, 1: 1.0},
    )
    assert residual == {0: 0.0, 1: 1.0}
    by_age = {row["age"]: row for row in rows}
    assert by_age[0]["clipped"] is True
    assert by_age[0]["residual_rate_unclipped"] == pytest.approx(-4.0)
    assert by_age[1]["clipped"] is False
    with pytest.raises(ValueError, match=r"aged \[3\]"):
        solve_child_benefit_claim_rates(
            child_ages=np.asarray([3]),
            child_key_ages=np.asarray([3]),
            child_reporter=np.asarray([False]),
            child_weights=np.asarray([1.0]),
            claim_rates={0: 0.5},
        )


def test_family_key_prefers_the_eldest_child_under_16() -> None:
    # A young person with younger siblings keys the family on the eldest
    # sibling under 16; young people alone key it on the eldest of them.
    key = child_benefit_family_key_ages(
        child_units=np.asarray([0, 0, 1, 1, 2]),
        child_ages=np.asarray([17, 13, 19, 16, 5]),
        unit_count=4,
    )
    assert key.tolist() == [13, 19, 5, -1]


def test_a_young_person_with_younger_siblings_claims_at_their_rate() -> None:
    # Families: (17, 13), (13), (17), none reporting. Keyed on the eldest
    # child under 16, the first family claims at the 13-year-olds' rate, so
    # r_13 = 0.9 and r_17 = (0.5 * 2 - 0.9) / 1. Keyed on the eldest child it
    # would claim at r_17 = 0.5 and r_13 would need (0.9 * 2 - 0.5) / 1 > 1.
    residual, rows = solve_child_benefit_claim_rates(
        child_ages=np.asarray([17, 13, 13, 17]),
        child_key_ages=np.asarray([13, 13, 13, 17]),
        child_reporter=np.zeros(4, dtype=bool),
        child_weights=np.ones(4),
        claim_rates={13: 0.9, 17: 0.5},
    )
    assert residual == {13: pytest.approx(0.9), 17: pytest.approx(0.1)}
    assert not any(row["clipped"] for row in rows)


def test_a_child_under_16_keyed_below_its_age_is_refused() -> None:
    with pytest.raises(ValueError, match="not a young person"):
        solve_child_benefit_claim_rates(
            child_ages=np.asarray([10, 8]),
            child_key_ages=np.asarray([8, 8]),
            child_reporter=np.zeros(2, dtype=bool),
            child_weights=np.ones(2),
            claim_rates={8: 0.9, 10: 0.9},
        )


def _young_person_families(with_child_share: float, *, count: int = 60_000):
    """Families of a young person (16 to 19), with a younger child or not."""

    rng = np.random.default_rng(11)
    young = rng.integers(16, 20, size=count)
    with_child = rng.random(count) < with_child_share
    child = rng.integers(10, 16, size=count)
    units = np.r_[np.arange(count), np.flatnonzero(with_child)]
    ages = np.r_[young, child[with_child]]
    eldest = np.full(count, -1)
    np.maximum.at(eldest, units, ages)
    rates = {age: 0.95 for age in range(10, 16)} | {
        16: 0.9,
        17: 0.75,
        18: 0.7,
        19: 0.5,
    }
    return assign_child_benefit_claims(
        child_units=units,
        child_ages=ages,
        eldest=eldest,
        reporter=rng.random(count) < 0.2,
        weights=np.ones(count),
        draws=rng.random(count),
        statistics=replace(_statistics(), claim_rates=rates),
    )


def test_claims_realise_the_published_rates_for_mixed_age_families() -> None:
    # The young people's lower rates and the younger children's higher ones
    # both hold: a young person with a younger sibling claims with them, and
    # the families of young people only make up each young age's share.
    _, receipt = _young_person_families(0.3)
    assert receipt["clipped_ages"] == []
    for row in receipt["ages"]:
        assert row["realized_rate"] == pytest.approx(row["published_rate"], abs=0.02)


def test_young_people_living_with_younger_siblings_can_be_over_claimed() -> None:
    # Claims stay all or none inside a family: when most 19-year-olds have a
    # younger sibling, their siblings' claims alone exceed the published 50%,
    # so the young-people-only rate clips at zero and the receipt says so.
    _, receipt = _young_person_families(0.9)
    by_age = {row["age"]: row for row in receipt["ages"]}
    assert 19 in receipt["clipped_ages"]
    assert by_age[19]["residual_rate_unclipped"] < 0.0
    assert by_age[19]["realized_rate"] > by_age[19]["published_rate"]
    assert all(age >= 16 for age in receipt["clipped_ages"])


def test_claims_keep_reporters_and_realise_the_published_rates() -> None:
    eldest, reporter, units, ages, draws = _families()
    # Every sixth unit has no eligible child.
    childless = np.arange(len(eldest)) % 6 == 0
    keep = ~childless[units]
    eldest = np.where(childless, -1, eldest)
    claims, receipt = assign_child_benefit_claims(
        child_units=units[keep],
        child_ages=ages[keep],
        eldest=eldest,
        reporter=reporter,
        weights=np.ones(len(eldest)),
        draws=draws,
        statistics=_statistics(),
    )
    assert claims[reporter & ~childless].all()
    # A unit without an eligible child is not a claiming family, reporter or not.
    assert not claims[childless].any()
    assert receipt["eligible_family_units"] == int((~childless).sum())
    for row in receipt["ages"]:
        assert row["realized_rate"] == pytest.approx(row["published_rate"], abs=0.02)
    assert receipt["realized_rate"] == pytest.approx(receipt["target_rate"], abs=0.01)
    assert receipt["clipped_ages"] == []
    assert receipt["published_all_ages_rate"] == 0.8


def test_the_receipt_counts_child_rows_among_clone_zero_children() -> None:
    """Each age's eligible_child_rows, and the receipt's, count the children
    whose rows are geographic clone 0's (microcosm#931). Three invented
    families: the first has children aged 2 and 0, the second one aged 1 and
    the third one aged 2; at two clones each clone's rows weigh half their
    source's."""

    units = np.asarray([0, 0, 1, 2])
    ages = np.asarray([2, 0, 1, 2])
    eldest = np.asarray([2, 1, 2])

    def receipt(clones: int, clone_zero: np.ndarray | None) -> dict:
        return assign_child_benefit_claims(
            child_units=np.concatenate([units + 3 * clone for clone in range(clones)]),
            child_ages=np.tile(ages, clones),
            eldest=np.tile(eldest, clones),
            reporter=np.zeros(3 * clones, dtype=bool),
            weights=np.full(3 * clones, 10.0 / clones),
            draws=np.zeros(3 * clones),
            statistics=_statistics(),
            child_clone_zero=clone_zero,
        )[1]

    one = receipt(1, None)
    two = receipt(2, np.repeat([True, False], 4))
    assert [row["eligible_child_rows"] for row in one["ages"]] == [1, 1, 2]
    assert one["eligible_child_rows"] == 4
    # Every figure but the count of family units, which counts every clone's,
    # is the one-clone receipt's: the row counts are clone 0's, and the
    # weighted figures over both clones are one clone's at its full weight.
    assert two == {**one, "eligible_family_units": 6}
    # Without clone 0's flags every child row counts, as on a one-clone frame.
    every = receipt(2, None)
    assert [row["eligible_child_rows"] for row in every["ages"]] == [2, 2, 4]
    assert every["eligible_child_rows"] == 8
    with pytest.raises(ValueError, match="one flag per eligible child"):
        receipt(2, np.ones(4, dtype=bool))


def _cloned_frame(clones: int):
    """``clones`` geographic clones of the invented frame: each clone's ids
    moved by a stride, its households given their clone index, and each
    household at ``1 / clones`` of its source's weight."""

    frame = _frame()
    person = frame.table("person")
    benunit = frame.table("benunit")
    household = frame.table("household")
    clone_index = ladder_clone_index_column("household")
    return uk_national_frame(
        person=pd.concat(
            [
                person.assign(
                    person_id=person["person_id"] + 1_000 * clone,
                    person_benunit_id=person["person_benunit_id"] + 100 * clone,
                    person_household_id=person["person_household_id"] + 100 * clone,
                )
                for clone in range(clones)
            ],
            ignore_index=True,
        ),
        benunit=pd.concat(
            [
                benunit.assign(benunit_id=benunit["benunit_id"] + 100 * clone)
                for clone in range(clones)
            ],
            ignore_index=True,
        ),
        household=pd.concat(
            [
                household.assign(
                    household_id=household["household_id"] + 100 * clone,
                    **{clone_index: np.int64(clone)},
                )
                for clone in range(clones)
            ],
            ignore_index=True,
        ),
        household_weights=np.full(len(household) * clones, 10.0 / clones),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def test_the_stage_counts_child_rows_in_clone_zero_at_every_clone_count() -> None:
    """The stage reads each child's clone from its household's clone index
    (microcosm#931): at four clones its receipt counts the child rows a
    one-clone frame has, at each age and overall, and a one-clone frame that
    carries the index gives the receipt of one without it."""

    def claims(frame) -> dict:
        return redraw_child_benefit_take_up(
            frame,
            engine=_StubEngine(),
            statistics=_statistics(),
            thresholds=THRESHOLDS,
        ).claims

    def rows(receipt: dict) -> list[tuple[int, int, float]]:
        return [
            (row["age"], row["eligible_child_rows"], row["eligible_children"])
            for row in receipt["ages"]
        ]

    one = claims(_frame())
    four = claims(_cloned_frame(4))
    assert rows(four) == rows(one) == [(0, 1, 10.0), (1, 1, 10.0), (2, 2, 20.0)]
    assert four["eligible_child_rows"] == one["eligible_child_rows"] == 4
    assert four["eligible_family_units"] == 4 * one["eligible_family_units"]
    assert claims(_cloned_frame(1)) == one


def _packaged_child_benefit_gate_parameters() -> dict:
    gates = json.loads(files("microcosm.build.uk").joinpath("gates.json").read_text())
    return dict(
        next(
            gate
            for gate in gates["gates"]
            if gate["id"] == "uk_stage_child_benefit_take_up"
        )["parameters"]
    )


def _thin_age_evidence(clones: int, *, count_every_clone: bool = False) -> dict:
    """An invented receipt at ``clones`` geographic clones: 900 source
    families with one child aged 5 and 150 with one aged 19, at published
    rates of 0.9 and 0.5. Every clone of a family weighs ``1 / clones`` of it
    and takes its source's draw, so the realised rates are the same at every
    clone count: 819 of the 900 and 63 of the 150 claim, 0.91 and 0.42. No
    family reports, and the opted-out share is at its target. The frame has
    no SPI copies, so its copy reports (microcosm#1508) hold no share, and
    their one cell rests on the 1,050 source families at every clone count."""

    sources = 1_050
    ages = np.r_[np.full(900, 5), np.full(150, 19)]
    draws = np.r_[np.zeros(819), np.full(81, 0.95), np.zeros(63), np.full(87, 0.75)]
    rows = sources * clones
    child_ages = np.tile(ages, clones)
    _, claims = assign_child_benefit_claims(
        child_units=np.arange(rows),
        child_ages=child_ages,
        eldest=child_ages,
        reporter=np.zeros(rows, dtype=bool),
        weights=np.full(rows, 15.0 / clones),
        draws=np.tile(draws, clones),
        statistics=_statistics({5: 0.9, 19: 0.5}),
        child_clone_zero=(None if count_every_clone else np.arange(rows) < sources),
    )
    return {
        "stage": CHILD_BENEFIT_TAKE_UP_STAGE_NAME,
        "copy_reports": {
            "minimum_source_households": 30,
            "cells": [
                {
                    "children": ["1"],
                    "income_bands": ["below_charge_start"],
                    "frs_source_households": sources,
                }
            ],
            "copy_rate_expected": None,
            "copy_rate_after": None,
            "frs_rows_unchanged": True,
        },
        "claims": claims,
        "opt_outs": {
            "target_share": 0.1,
            "realized_share": 0.1,
            "pool_exhausted": False,
        },
    }


def test_the_stage_gate_gives_one_verdict_at_one_and_fifteen_clones() -> None:
    """The packaged gate measures an age from 200 child rows, which the stage
    counts in clone 0 (microcosm#931). At one clone the age of 150 rows is too
    thin to measure and the gate passes on the other age; at fifteen it is as
    thin, and the gate gives the same verdict. Counted over every clone, the
    age would hold 2,250 rows at fifteen clones and be measured, its
    deviation of 0.08 past the 0.05 floor, which three standard errors over
    2,250 rows at 0.5 (about 0.032) do not reach."""

    parameters = _packaged_child_benefit_gate_parameters()

    def verdict(evidence: dict):
        return stage_health.uk_stage_health_gate(
            evidence=evidence,
            stage=CHILD_BENEFIT_TAKE_UP_STAGE_NAME,
            check="child_benefit_take_up",
            parameters=parameters,
        )

    one = verdict(_thin_age_evidence(1))
    fifteen = verdict(_thin_age_evidence(15))
    assert one.passed, one.failures
    assert (fifteen.passed, fifteen.failures) == (one.passed, one.failures)
    assert one.details["ages_measured"] == 1
    # The details are the one-clone details, but for the count of family
    # units, a floor of one that counts every clone's units.
    assert fifteen.details == {
        **one.details,
        "eligible_family_units": 15 * one.details["eligible_family_units"],
    }
    every = verdict(_thin_age_evidence(15, count_every_clone=True))
    assert not every.passed
    assert every.details["ages_measured"] == 2
    assert every.failures == (
        "child_benefit_take_up: at age 19 0.4200 of eligible children are "
        "claimed for against the published 0.5000 (tolerance 0.0500: the "
        "larger of 0.05 and 3 standard errors of a share at the published rate "
        "over that age's child rows).",
    )


def test_opt_outs_take_the_published_share_from_fully_charged_families_first() -> None:
    count = 20_000
    rng = np.random.default_rng(3)
    claims = rng.random(count) < 0.9
    reporter = claims & (rng.random(count) < 0.4)
    income = rng.choice([30_000.0, 70_000.0, 90_000.0], size=count, p=[0.6, 0.2, 0.2])
    weights = np.ones(count)
    arguments = {
        "claims": claims,
        "reporter": reporter,
        "highest_income": income,
        "children": np.full(count, 2.0),
        "weights": weights,
        "draws": rng.random(count),
        "thresholds": THRESHOLDS,
    }
    opt_out, receipt = assign_child_benefit_opt_outs(
        statistics=_statistics(), **arguments
    )
    # Only claiming, non-reporting families at or above full withdrawal.
    assert (claims & ~reporter & (income >= 80_000.0))[opt_out].all()
    assert receipt["target_share"] == pytest.approx(0.1)
    assert receipt["realized_share"] == pytest.approx(0.1, abs=0.01)
    assert receipt["pool_exhausted"] is False
    assert [pool["pool"] for pool in receipt["pools"]] == ["fully_charged", "taper"]
    assert receipt["pools"][1]["rate"] == 0.0
    assert receipt["children_per_opted_out_family"] == 2.0
    assert receipt["published_children_per_opted_out_family"] == 1.5

    # A target heavier than the fully charged pool spills into the taper, and
    # one heavier than both pools is recorded as a shortfall, never drawn from
    # families outside the charge.
    spill = UKChildBenefitStatistics(
        **{**_statistics().__dict__, "families_opted_out": 150.0}
    )
    opt_out, receipt = assign_child_benefit_opt_outs(statistics=spill, **arguments)
    assert receipt["pools"][0]["rate"] == 1.0
    assert 0.0 < receipt["pools"][1]["rate"] < 1.0
    assert receipt["pool_exhausted"] is False
    short = UKChildBenefitStatistics(
        **{**_statistics().__dict__, "families_opted_out": 400.0}
    )
    opt_out, receipt = assign_child_benefit_opt_outs(statistics=short, **arguments)
    assert receipt["pool_exhausted"] is True and receipt["pool_shortfall_families"] > 0
    assert (opt_out == (claims & ~reporter & (income > 60_000.0))).all()


def test_statistics_and_thresholds_refuse_unusable_values() -> None:
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        _statistics({0: 1.2})
    with pytest.raises(ValueError, match="positive part of the registered"):
        UKChildBenefitStatistics(
            **{**_statistics().__dict__, "families_opted_out": 2_000.0}
        )
    with pytest.raises(ValueError, match="0 < start < end"):
        UKChildBenefitChargeThresholds(phase_out_start=80_000.0, phase_out_end=60_000.0)


def test_redraw_reads_the_engine_once_and_rewrites_only_the_two_flags() -> None:
    engine = _StubEngine()
    frame = _frame()
    # Every family claims and half of the claiming families opt out, so the
    # one fully charged family is certain to.
    statistics = UKChildBenefitStatistics(
        **{
            **_statistics({0: 1.0, 1: 1.0, 2: 1.0}).__dict__,
            "families_opted_out": 500.0,
        }
    )

    result = redraw_child_benefit_take_up(
        frame, engine=engine, statistics=statistics, thresholds=THRESHOLDS
    )

    assert engine.calls == [(CHILD_BENEFIT_ENGINE_VARIABLES, "2024")]
    after = result.frame.table("benunit").set_index("benunit_id")
    # The reporter is paid; the fully charged family claims and opts out, so
    # it is not; the low-income family is paid; the unit without a child
    # keeps its early draw and is never opted out.
    assert after["would_claim_child_benefit"].to_dict() == {
        1: True,
        2: False,
        3: True,
        4: True,
    }
    assert after["child_benefit_opts_out"].to_dict() == {
        1: False,
        2: True,
        3: False,
        4: False,
    }
    for entity in ("person", "household"):
        pd.testing.assert_frame_equal(
            result.frame.table(entity), frame.table(entity), check_exact=True
        )
    for name in CHILD_BENEFIT_ENGINE_VARIABLES:
        assert name not in result.frame.table("person")
        assert name not in result.frame.table("benunit")
    evidence = result.evidence()
    assert evidence["claims"]["eligible_family_units"] == 3
    assert evidence["claims"]["eligible_children"] == 40.0
    assert evidence["opt_outs"]["opted_out_families"] == 10.0
    assert evidence["opt_outs"]["opted_out_children"] == 20.0
    assert evidence["in_payment"]["families"] == 20.0
    assert evidence["in_payment"]["children"] == 20.0
    assert evidence["reporters_without_eligible_child"] == {
        "units": 1,
        "weighted_units": 10.0,
    }
    assert evidence["changed_units"] == {
        "would_claim_child_benefit": 1,
        "child_benefit_opts_out": 4,
        "child_benefit_reported": 0,
    }
    # A frame without SPI copies has nothing to redraw.
    assert evidence["copy_reports"]["copy_family_units"] == 0
    assert evidence["copy_reports"]["copy_rate_after"] is None
    assert evidence["copy_reports"]["frs_rows_unchanged"] is True


def _stage() -> SourceStageSpec:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()[CHILD_BENEFIT_TAKE_UP_STAGE_NAME]


def test_committed_manifest_declares_the_stage_its_resource_and_rewrites() -> None:
    stage = _stage()

    assert stage.grain == "benunit"
    assert stage.outputs == ()
    assert stage.rewrites == (
        "would_claim_child_benefit",
        "child_benefit_opts_out",
        "child_benefit_reported",
    )
    assert [artifact["resource"] for artifact in stage.artifacts] == [
        CHILD_BENEFIT_STATISTICS_RESOURCE
    ]
    _assert_stage_parameters(stage)
    assert {
        operation.kind: dict(operation.parameters) for operation in stage.operations
    } == child_benefit_take_up_operation_parameters()
    transform = UKChildBenefitTakeUpStageTransform(
        stage=stage,
        engine=_StubEngine(),
        statistics=_statistics({0: 1.0, 1: 1.0, 2: 1.0}),
        thresholds=THRESHOLDS,
    )
    transform(_frame())
    assert transform.checkpoint_metadata()["evidence"]["stage"] == (
        CHILD_BENEFIT_TAKE_UP_STAGE_NAME
    )
    # The stage runs straight after the Pension Credit redraw.
    roster = [entry.stage for entry in load_country_spec("uk").sources.stages]
    assert roster[roster.index("pension_credit_take_up") + 1] == stage.stage


def test_stage_parameters_refuse_drift() -> None:
    stage = _stage()
    operations = []
    for operation in stage.operations:
        payload = {"kind": operation.kind, **dict(operation.parameters)}
        if operation.kind == "assign_opt_out_by_charge_income":
            payload["pools"] = ["taper", "fully_charged"]
        operations.append(payload)
    drifted = SourceStageSpec.from_mapping({**stage.__dict__, "operations": operations})
    with pytest.raises(ValueError, match=r"operation\(s\) \['assign_opt_out"):
        _assert_stage_parameters(drifted)
    with pytest.raises(ValueError, match="expected 'child_benefit_take_up'"):
        _assert_stage_parameters(replace(stage, stage="frs_take_up"))
    with pytest.raises(ValueError, match="exactly the rewrites"):
        _assert_stage_parameters(replace(stage, rewrites=("child_benefit_opts_out",)))


def test_vendored_statistics_are_the_august_2025_release() -> None:
    statistics = load_child_benefit_statistics()

    # Table 15, May 2025: single years 0 to 19 and the all-ages rate.
    assert sorted(statistics.claim_rates) == list(range(20))
    assert statistics.all_ages_claim_rate == pytest.approx(0.866)
    assert statistics.claim_rates[0] == pytest.approx(0.688)
    assert statistics.claim_rates[15] == pytest.approx(0.966)
    assert statistics.claim_rates[19] == pytest.approx(0.497)
    # August 2025 caseload: registered families are those in payment plus
    # those opted out.
    assert statistics.families_registered == 7_552_330
    assert statistics.families_opted_out == 684_635
    assert statistics.families_in_payment == 6_867_695
    assert (
        statistics.families_in_payment + statistics.families_opted_out
        == statistics.families_registered
    )
    assert statistics.children_in_payment == 11_730_730
    assert statistics.children_opted_out == 999_680
    assert statistics.opt_out_share == pytest.approx(684_635 / 7_552_330)
    entry = next(
        resource
        for resource in load_vendor_selections()["resources"]
        if resource["resource"] == CHILD_BENEFIT_STATISTICS_RESOURCE
    )
    assert entry["consumers"] == ["uk_runtime.child_benefit_take_up"]


def test_terminal_signal_gate_leaves_the_child_benefit_flags_to_the_stage_gate() -> (
    None
):
    outputs = {output for _, output, _ in UK_TAKE_UP_SIGNAL_OUTPUTS}
    assert not outputs & {"would_claim_child_benefit", "child_benefit_opts_out"}


@pytest.mark.parametrize("supports_opt_out", [False, True])
def test_export_preserves_registered_claims_only_with_opt_out_aware_engine(
    supports_opt_out: bool,
) -> None:
    # UK #2140 reads opt-out separately. Older models pay the would-claim
    # flag directly, so their export must keep the legacy in-payment flag.
    statistics = replace(
        _statistics({0: 1.0, 1: 1.0, 2: 1.0}), families_opted_out=500.0
    )
    result = redraw_child_benefit_take_up(
        _frame(),
        engine=_StubEngine(),
        statistics=statistics,
        thresholds=replace(
            THRESHOLDS,
            supports_opt_out=supports_opt_out,
            opt_out_charge_share=1.0 if supports_opt_out else None,
        ),
    )
    after = result.frame.table("benunit").set_index("benunit_id")
    assert after["would_claim_child_benefit"].to_dict() == {
        1: True,
        2: supports_opt_out,
        3: True,
        4: True,
    }
    assert after["child_benefit_opts_out"].to_dict() == {
        1: False,
        2: True,
        3: False,
        4: False,
    }
    # Draw audit quantities do not depend on the engine/export encoding.
    assert result.claims["eligible_family_units"] == 3
    assert result.opt_outs["opted_out_families"] == 10.0
    assert result.in_payment["families"] == 20.0
    assert result.in_payment["children"] == 20.0
    assert result.changed_units["would_claim_child_benefit"] == 1 + supports_opt_out
    assert result.evidence()["claim_export"] == {
        "encoding": "registered_claims" if supports_opt_out else "legacy_payment",
        "supports_opt_out": supports_opt_out,
        "engine_parameters_source": "caller",
    }


@pytest.mark.parametrize("supports_opt_out", [False, True])
def test_redraw_keeps_nonclaimants_false_under_either_export_contract(
    supports_opt_out: bool,
) -> None:
    # Only the reporting family claims; independent early opt-out flags on
    # nonclaimants must not turn them into registered claims.
    result = redraw_child_benefit_take_up(
        _frame(),
        engine=_StubEngine(),
        statistics=_statistics({0: 0.0, 1: 0.0, 2: 0.0}),
        thresholds=replace(
            THRESHOLDS,
            supports_opt_out=supports_opt_out,
            opt_out_charge_share=1.0 if supports_opt_out else None,
        ),
    )
    after = result.frame.table("benunit").set_index("benunit_id")
    assert after.loc[2:3, "would_claim_child_benefit"].tolist() == [False, False]
    assert after.loc[2:3, "child_benefit_opts_out"].tolist() == [False, False]


def test_committed_smoke_fixture_uses_current_child_benefit_contract() -> None:
    # The integration smoke driver reconstructs this frozen stage descriptor,
    # separately from the packaged country spec checked above.
    path = (
        paths_for("microcosm-graph").tests
        / "fixtures/parity/uk_spine/sources/fixture.json"
    )
    descriptor = json.loads(path.read_text())
    stage = SourceStageSpec.from_mapping(
        descriptor["stages"][CHILD_BENEFIT_TAKE_UP_STAGE_NAME]
    )
    _assert_stage_parameters(stage)


@pytest.mark.parametrize("supports_opt_out", [False, True])
def test_new_contract_reports_shortfall_without_paid_taper_spillover(
    supports_opt_out: bool,
) -> None:
    # Weighted claims total 40; the 50% target is 20. Only 2 fully charged
    # families are eligible under share=1, while 30 sit in the paid taper.
    thresholds = replace(
        THRESHOLDS,
        supports_opt_out=supports_opt_out,
        opt_out_charge_share=1.0 if supports_opt_out else None,
    )
    opt_out, receipt = assign_child_benefit_opt_outs(
        claims=np.asarray([True, True, True, True, False, False]),
        reporter=np.asarray([False, False, True, False, False, False]),
        highest_income=np.asarray([80_000, 70_000, 95_000, 60_000, 95_000, 95_000]),
        children=np.asarray([2, 1, 1, 1, 1, 0]),
        weights=np.asarray([2.0, 30.0, 5.0, 3.0, 7.0, 11.0]),
        draws=np.zeros(6),
        statistics=replace(_statistics(), families_opted_out=500.0),
        thresholds=thresholds,
    )
    assert opt_out.tolist() == [True, not supports_opt_out, False, False, False, False]
    assert receipt["target_families"] == 20.0
    assert receipt["pool_shortfall_families"] == (18.0 if supports_opt_out else 0.0)
    assert receipt["pool_exhausted"] is supports_opt_out
    assert receipt["opted_out_children"] == (4.0 if supports_opt_out else 34.0)


@pytest.mark.parametrize(
    ("share", "expected"),
    [
        (1.0, [False, False, False, False, False, True, True]),
        (0.5, [False, False, False, False, True, True, True]),
        (0.0, [False, False, True, True, True, True, True]),
        (1.1, [False, False, False, False, False, False, False]),
    ],
)
def test_new_opt_out_pool_matches_positive_charge_share_boundaries(share, expected):
    income = np.asarray([59_999, 60_000, 60_001, 69_999, 70_000, 80_000, 90_000])
    opt_out, receipt = assign_child_benefit_opt_outs(
        claims=np.ones(7, dtype=bool),
        reporter=np.zeros(7, dtype=bool),
        highest_income=income,
        children=np.ones(7),
        weights=np.ones(7),
        draws=np.zeros(7),
        statistics=replace(_statistics(), families_opted_out=990.0),
        thresholds=replace(
            THRESHOLDS, supports_opt_out=True, opt_out_charge_share=share
        ),
    )
    assert opt_out.tolist() == expected
    assert receipt["opt_out_charge_share"] == share
    assert receipt["pool_shortfall_families"] == pytest.approx(6.93 - sum(expected))


@pytest.mark.parametrize("share", [None, float("nan"), float("inf"), "1", True])
def test_opt_out_capability_requires_a_finite_numeric_share(share):
    with pytest.raises(ValueError, match="opt_out_charge_share.*finite numeric"):
        replace(THRESHOLDS, supports_opt_out=True, opt_out_charge_share=share)


@pytest.mark.parametrize("end", [60_000.0, 50_000.0])
@pytest.mark.parametrize("supports_opt_out", [False, True])
def test_both_contracts_reject_nonpositive_taper_width(end, supports_opt_out):
    with pytest.raises(ValueError, match="0 < start < end"):
        replace(THRESHOLDS, phase_out_end=end, supports_opt_out=supports_opt_out)


def test_new_contract_preserves_draws_when_full_charge_pool_is_sufficient():
    arguments = dict(
        claims=np.ones(4, dtype=bool),
        reporter=np.zeros(4, dtype=bool),
        highest_income=np.asarray([80_000, 90_000, 100_000, 70_000]),
        children=np.ones(4),
        weights=np.full(4, 10.0),
        draws=np.asarray([0.05, 0.4, 0.8, 0.0]),
        statistics=_statistics(),
    )
    legacy, legacy_receipt = assign_child_benefit_opt_outs(
        **arguments, thresholds=THRESHOLDS
    )
    modern, modern_receipt = assign_child_benefit_opt_outs(
        **arguments,
        thresholds=replace(THRESHOLDS, supports_opt_out=True, opt_out_charge_share=1.0),
    )
    assert legacy.tolist() == modern.tolist() == [True, False, False, False]
    assert legacy_receipt["pools"][0] == modern_receipt["pools"][0]
    # The new contract removes the paid taper candidate even when its draw
    # rate would be zero; the unchanged full-charge pool selects identically.
    assert legacy_receipt["pools"][1]["units"] == 1
    assert modern_receipt["pools"][1]["units"] == 0
    assert legacy_receipt["pools"][1]["rate"] == modern_receipt["pools"][1]["rate"] == 0
    assert modern_receipt["pool_shortfall_families"] == 0.0
    assert modern_receipt["pool_exhausted"] is False


# The SPI copies' reports (microcosm#1508).


def _copy_result(frame=None):
    return redraw_child_benefit_take_up(
        _copy_frame() if frame is None else frame,
        engine=_StubEngine(),
        statistics=_statistics({5: 0.9}),
        thresholds=THRESHOLDS,
        minimum_source_households=1,
    )


def _reports(result) -> dict[int, float]:
    person = result.frame.table("person").set_index("person_id")
    return person["child_benefit_reported"].to_dict()


def _rebuilt(frame, *, person=None, benunit=None, household=None):
    return uk_national_frame(
        person=frame.table("person") if person is None else person,
        benunit=frame.table("benunit") if benunit is None else benunit,
        household=frame.table("household") if household is None else household,
        household_weights=frame.weights_for("household").values,
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def test_copy_reports_follow_their_cells_and_twins_and_keep_frs_rows() -> None:
    frame = _copy_frame()
    result = _copy_result(frame)
    before = frame.table("person").set_index("person_id")["child_benefit_reported"]
    after = _reports(result)

    assert {key: after[key] for key in before.index if key < 1_000} == {
        key: value for key, value in before.items() if key < 1_000
    }
    # Below the charge start two of the four FRS families report; in the
    # fully charged cell none does. Copy 101's twin reports in its own cell,
    # so it keeps the twin's report and amount.
    assert (after[1011], after[1012]) == (1_300.0, 0.0)
    # Copy 102's twin does not report in its cell: the forests' report goes.
    assert after[1021] == 0.0
    # Copy 105's twin reports, but the copy is fully charged.
    assert after[1051] == 0.0
    # Copy 106 has no eligible child and takes its twin's (absent) report.
    assert after[1061] == 0.0
    receipt = result.copy_reports
    assert [cell["children"] for cell in receipt["cells"]] == [["1", "2", "3+"]] * 2
    assert [cell["income_bands"] for cell in receipt["cells"]] == [
        ["below_charge_start"],
        ["taper", "fully_charged"],
    ]
    assert receipt["cells"][0]["frs_source_households"] == 4
    assert receipt["cells"][0]["frs_rate"] == 0.5
    # One FRS household carries the fully charged cell: its figures are
    # withheld, and so is every count of one or two copies.
    assert receipt["cells"][1]["frs_source_households"] is None
    assert receipt["cells"][1]["frs_rate"] is None
    assert receipt["cells"][1]["copy_rate_after"] is None
    assert receipt["copy_family_units"] == 3
    assert receipt["copy_rate_before"] == pytest.approx(2 / 3)
    assert receipt["copy_rate_expected"] == pytest.approx(1 / 3)
    assert receipt["copy_rate_after"] == pytest.approx(1 / 3)
    assert receipt["transitions"] == {
        "promoted": None,
        "demoted": None,
        "held_reporter": 0,
        "held_nonreporter": 0,
    }
    assert receipt["copies_without_eligible_child"] == {
        "units": None,
        "reporting_before": None,
        "reporting_after": 0,
    }
    assert receipt["frs_rows_unchanged"] is True
    assert result.changed_units["child_benefit_reported"] == 4
    # The claim draw reads the redrawn reports: copy 101 claims as a reporter.
    claims = result.frame.table("benunit").set_index("benunit_id")
    assert bool(claims.loc[101, "would_claim_child_benefit"])


def test_a_copy_of_a_non_reporting_twin_can_take_up_a_report() -> None:
    frame = _copy_frame()
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    # Every FRS family below the charge start reports, and copy 105 now copies
    # the fully charged unit 5, which does not, into that cell.
    person.loc[person["person_id"].isin([31, 41]), "child_benefit_reported"] = 1_000.0
    benunit.loc[benunit["benunit_id"] == 105, "benunit_source_id"] = 5
    copy = person["person_benunit_id"] == 105
    person.loc[copy, "person_source_id"] = [51, 52]
    person.loc[person["person_id"] == 1051, "stub_income"] = 30_000.0
    household.loc[household["household_id"] == 105, "source_household_id"] = 5
    result = _copy_result(
        _rebuilt(frame, person=person, benunit=benunit, household=household)
    )

    after = _reports(result)
    # A new report carries the engine's entitlement on the benefit unit head.
    assert (after[1051], after[1052]) == (STUB_ENTITLEMENT_PER_CHILD, 0.0)
    # Copies of reporting twins in the twins' cell take the twins' amounts.
    assert (after[1011], after[1021]) == (1_300.0, 1_000.0)
    assert result.copy_reports["copy_rate_after"] == 1.0


def test_copy_report_redraw_refuses_broken_lineage() -> None:
    frame = _copy_frame()
    cases = []
    benunit = frame.table("benunit").copy()
    benunit.loc[benunit["benunit_id"] == 101, "benunit_source_id"] = 999
    cases.append((_rebuilt(frame, benunit=benunit), "names no benefit unit"))
    benunit = frame.table("benunit").copy()
    benunit.loc[benunit["benunit_id"] == 102, "benunit_source_id"] = 101
    cases.append((_rebuilt(frame, benunit=benunit), "must be an FRS benefit unit"))
    person = frame.table("person").copy()
    person.loc[person["person_id"] == 1012, "person_source_id"] = 22
    cases.append((_rebuilt(frame, person=person), "copy's twin benefit unit"))
    benunit = frame.table("benunit").copy()
    benunit.loc[benunit["benunit_id"] == 106, "benunit_support_channel"] = "cgt"
    cases.append((_rebuilt(frame, benunit=benunit), "only the FRS and SPI"))
    person = frame.table("person").drop(columns=["person_source_id"])
    cases.append((_rebuilt(frame, person=person), "person columns missing"))
    for broken, message in cases:
        with pytest.raises(ValueError, match=message):
            _copy_result(broken)


def test_copy_report_cells_pool_bands_then_children_groups() -> None:
    # One-child families: three households below the charge start, one in
    # the taper, two fully charged. Two-child families: one below the start
    # and one fully charged. No larger family; one unit without a child.
    children = np.array([1, 1, 1, 1, 1, 1, 2, 2, 0])
    income = np.array([10, 20, 30, 70, 90, 95, 10, 90, 10]) * 1_000.0
    arguments = {
        "family": children > 0,
        "children": children,
        "highest_income": income,
        "base": np.ones(9, dtype=bool),
        "source_households": np.arange(9),
        "thresholds": THRESHOLDS,
    }

    cell, cells = child_benefit_copy_report_cells(
        **arguments, minimum_source_households=2
    )
    assert [(row["children"], row["income_bands"]) for row in cells] == [
        (["1"], ["below_charge_start"]),
        (["1"], ["taper", "fully_charged"]),
        (["2", "3+"], ["below_charge_start", "taper", "fully_charged"]),
    ]
    assert cell.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, -1]
    assert [row["frs_source_households"] for row in cells] == [3, 3, 2]
    # With three households needed, the two-child group cannot stand alone.
    _, cells = child_benefit_copy_report_cells(**arguments, minimum_source_households=3)
    assert [(row["children"], row["income_bands"]) for row in cells] == [
        (["1", "2", "3+"], ["below_charge_start"]),
        (["1", "2", "3+"], ["taper", "fully_charged"]),
    ]
    # Support counts each FRS household once: six one-child units from one
    # household cannot stand alone, and merge with the two-child group.
    _, cells = child_benefit_copy_report_cells(
        **{**arguments, "source_households": np.r_[np.zeros(6), np.arange(1, 4)]},
        minimum_source_households=2,
    )
    assert [(row["children"], row["income_bands"]) for row in cells] == [
        (["1", "2", "3+"], ["below_charge_start"]),
        (["1", "2", "3+"], ["taper", "fully_charged"]),
    ]
    assert [row["frs_source_households"] for row in cells] == [2, 2]
    # FRS families alone carry support; copies in a cell add none.
    _, cells = child_benefit_copy_report_cells(
        **{**arguments, "base": np.r_[np.ones(6, dtype=bool), np.zeros(3, dtype=bool)]},
        minimum_source_households=3,
    )
    assert [(row["children"], row["income_bands"]) for row in cells] == [
        (["1", "2", "3+"], ["below_charge_start"]),
        (["1", "2", "3+"], ["taper", "fully_charged"]),
    ]


def test_coupled_copy_reports_keep_same_cell_twins_and_reach_the_cell_rate() -> None:
    rng = np.random.default_rng(3)
    count = 40_000
    rates = np.array([0.6, 0.3, 0.9])
    twin_reports = rng.random(count) < rates[0]
    copy_cell = rng.integers(0, 3, size=count)
    drawn = assign_spi_child_benefit_reports(
        copy=np.r_[np.zeros(count, dtype=bool), np.ones(count, dtype=bool)],
        family=np.ones(2 * count, dtype=bool),
        twin=np.r_[np.full(count, -1), np.arange(count)],
        cell=np.r_[np.zeros(count, dtype=np.int64), copy_cell],
        rates=rates,
        # The copies' own earlier reports play no part.
        reporter=np.r_[twin_reports, rng.random(count) < 0.5],
        draws=rng.random(2 * count),
    )

    assert not drawn[:count].any()
    copies = drawn[count:]
    same = copy_cell == 0
    assert np.array_equal(copies[same], twin_reports[same])
    for index, rate in enumerate(rates):
        assert copies[copy_cell == index].mean() == pytest.approx(rate, abs=0.015)
    # Into a lower-rate cell only a reporting twin's copy reports, with
    # probability p_copy / p_twin; into a higher one every such copy does.
    low, high = copy_cell == 1, copy_cell == 2
    assert not copies[low & ~twin_reports].any()
    assert copies[low & twin_reports].mean() == pytest.approx(0.3 / 0.6, abs=0.02)
    assert copies[high & twin_reports].all()
    with pytest.raises(ValueError, match="needs an FRS twin"):
        assign_spi_child_benefit_reports(
            copy=np.array([True]),
            family=np.array([True]),
            twin=np.array([-1]),
            cell=np.array([0]),
            rates=rates,
            reporter=np.array([False]),
            draws=np.array([0.5]),
        )


@pytest.mark.parametrize("twin_reports", [True, False])
def test_a_copy_whose_twin_is_in_no_cell_draws_at_its_own_rate(
    twin_reports: bool,
) -> None:
    # A twin with no eligible child has no rate to couple to (Vahid's review
    # of microcosm#1718): its copy reports when u < p_copy, not always or never.
    u = np.array([0.1, 0.29, 0.31, 0.9])
    drawn = assign_spi_child_benefit_reports(
        copy=np.r_[np.zeros(4, dtype=bool), np.ones(4, dtype=bool)],
        family=np.r_[np.zeros(4, dtype=bool), np.ones(4, dtype=bool)],
        twin=np.r_[np.full(4, -1), np.arange(4)],
        cell=np.r_[np.full(4, -1), np.zeros(4, dtype=np.int64)],
        rates=np.array([0.3]),
        reporter=np.r_[np.full(4, twin_reports), np.zeros(4, dtype=bool)],
        draws=np.r_[np.zeros(4), u],
    )

    assert drawn.tolist() == [False] * 4 + [True, True, False, False]
