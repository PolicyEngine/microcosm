"""Child Benefit claims and opt-outs after the SPI chain (microcosm#1063)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceStageSpec
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
    child_benefit_take_up_operation_parameters,
    load_child_benefit_statistics,
    redraw_child_benefit_take_up,
    solve_child_benefit_claim_rates,
)
from microcosm.build.uk_runtime.frs_take_up import UK_TAKE_UP_SIGNAL_OUTPUTS
from microcosm.build.uk_runtime.ledger_fact_vendoring import load_vendor_selections
from test_support.microcosm_build.uk_child_benefit_take_up import (
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
    }


def _stage() -> SourceStageSpec:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()[CHILD_BENEFIT_TAKE_UP_STAGE_NAME]


def test_committed_manifest_declares_the_stage_its_resource_and_rewrites() -> None:
    stage = _stage()

    assert stage.grain == "benunit"
    assert stage.outputs == ()
    assert stage.rewrites == ("would_claim_child_benefit", "child_benefit_opts_out")
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
