"""Pension Credit take-up by component after the SPI chain (microcosm#1069 R9)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.frs_take_up import UK_TAKE_UP_SIGNAL_OUTPUTS
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.pension_credit_take_up import (
    PENSION_CREDIT_ENTITLEMENT_VARIABLES,
    PENSION_CREDIT_MIXED_AGE_SAVING,
    PENSION_CREDIT_TAKE_UP_STAGE_NAME,
    UKPensionCreditTakeUpStageTransform,
    _assert_stage_parameters,
    assign_component_take_up,
    redraw_pension_credit_take_up,
)
from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract
from microcosm.frame import WeightKind

RATES = {
    "pension_credit_guarantee_credit": 0.69,
    "pension_credit_savings_credit_only": 0.37,
}


def _stage() -> SourceStageSpec:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()[PENSION_CREDIT_TAKE_UP_STAGE_NAME]


def _population(count: int = 30_000, *, gc_reporter_share: float = 0.1):
    rng = np.random.default_rng(5)
    band = np.repeat(["gc", "sc", "none"], count // 3)
    guarantee = np.where(band == "gc", 50.0, 0.0)
    savings = np.where(band == "sc", 20.0, 0.0)
    eligible = band != "none"
    reporter = np.where(
        band == "gc",
        rng.random(len(band)) < gc_reporter_share,
        rng.random(len(band)) < 0.05,
    )
    return guarantee, savings, eligible, reporter, rng.random(len(band)), band


def test_each_component_band_claims_at_its_rate_with_reporters_first() -> None:
    guarantee, savings, eligible, reporter, draws, band = _population()
    weights = np.ones(len(band))

    would_claim, receipts = assign_component_take_up(
        guarantee=guarantee,
        savings=savings,
        eligible=eligible,
        reporter=reporter,
        weights=weights,
        draws=draws,
        rates=RATES,
    )

    assert would_claim[reporter].all()
    assert not would_claim[(band == "none") & ~reporter].any()
    by_band = {receipt["band"]: receipt for receipt in receipts}
    for name, mask, rate in (
        ("guarantee_credit", band == "gc", 0.69),
        ("savings_credit_only", band == "sc", 0.37),
    ):
        receipt = by_band[name]
        entitled, reporting = mask.sum(), (mask & reporter).sum()
        assert receipt["residual_rate"] == pytest.approx(
            (rate * entitled - reporting) / (entitled - reporting)
        )
        assert receipt["realized_take_up"] == pytest.approx(rate, abs=0.03)
        assert receipt["reporters_exceed_rate"] is False


def test_reporters_above_the_rate_leave_no_residual_draw() -> None:
    guarantee, savings, eligible, reporter, draws, band = _population(
        gc_reporter_share=0.8
    )

    would_claim, receipts = assign_component_take_up(
        guarantee=guarantee,
        savings=savings,
        eligible=eligible,
        reporter=reporter,
        weights=np.ones(len(band)),
        draws=draws,
        rates=RATES,
    )

    gc = receipts[0]
    assert gc["band"] == "guarantee_credit"
    assert gc["residual_rate"] == 0.0
    assert gc["drawn_units"] == 0
    assert gc["reporters_exceed_rate"] is True
    gc_mask = band == "gc"
    assert (would_claim[gc_mask] == reporter[gc_mask]).all()


def test_the_residual_is_weighted() -> None:
    # Two entitled Guarantee Credit units: a reporter of weight 1 and a
    # non-reporter of weight 3, so r = (0.5 * 4 - 1) / (4 - 1) = 1 / 3.
    _, receipts = assign_component_take_up(
        guarantee=np.asarray([10.0, 10.0]),
        savings=np.zeros(2),
        eligible=np.asarray([True, True]),
        reporter=np.asarray([True, False]),
        weights=np.asarray([1.0, 3.0]),
        draws=np.asarray([0.9, 0.9]),
        rates={**RATES, "pension_credit_guarantee_credit": 0.5},
    )
    assert receipts[0]["residual_rate"] == pytest.approx(1 / 3)


class _StubEngine:
    country = "uk"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def materialize(self, frame, variables, period):
        self.calls.append((tuple(variables), str(period)))
        benunit = frame.table("benunit")
        ids = benunit["benunit_id"].to_numpy()
        # 1: Guarantee Credit; 2: Savings Credit only; 3: not entitled.
        # Unit 2 is a mixed-age couple keeping the SI 2019/37 saving.
        return {
            PENSION_CREDIT_MIXED_AGE_SAVING: ids == 2,
            "guarantee_credit": np.where(ids == 1, 40.0, 0.0),
            "savings_credit": np.where(ids == 2, 15.0, 0.0),
            "is_pension_credit_eligible": ids != 3,
        }


def _frame():
    person = pd.DataFrame(
        {
            "person_id": [11, 21, 31],
            "person_benunit_id": [1, 2, 3],
            "person_household_id": [1, 2, 3],
            "age": [80, 78, 70],
            "pension_credit_reported": [0.0, 0.0, 500.0],
        }
    )
    benunit = pd.DataFrame(
        {"benunit_id": [1, 2, 3], "would_claim_pc": [True, True, False]}
    )
    household = pd.DataFrame({"household_id": [1, 2, 3], "region": ["WALES"] * 3})
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.asarray([10.0, 10.0, 10.0]),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def test_redraw_reads_engine_entitlement_once_and_rewrites_only_would_claim_pc() -> (
    None
):
    engine = _StubEngine()
    frame = _frame()

    result = redraw_pension_credit_take_up(
        frame, engine=engine, contract=load_uk_take_up_contract()
    )

    # One materialization at the frame's survey-year period, the saving first.
    assert engine.calls == [
        (
            (PENSION_CREDIT_MIXED_AGE_SAVING, *PENSION_CREDIT_ENTITLEMENT_VARIABLES),
            "2024",
        )
    ]
    after = result.frame.table("benunit")
    # Unit 3 reports Pension Credit the engine finds it is not entitled to: it
    # keeps claiming, and the receipt counts it.
    assert bool(after.loc[after["benunit_id"] == 3, "would_claim_pc"].iloc[0])
    assert result.reporters_without_entitlement == {
        "units": 1,
        "weighted_units": 10.0,
    }
    for entity in ("person", "household"):
        pd.testing.assert_frame_equal(
            result.frame.table(entity), frame.table(entity), check_exact=True
        )
    pd.testing.assert_frame_equal(
        after.drop(columns=["would_claim_pc", PENSION_CREDIT_MIXED_AGE_SAVING]),
        frame.table("benunit").drop(columns=["would_claim_pc"]),
    )
    # The saving is stored as the engine's survey-year default; the
    # entitlement it fed stays consumed.
    assert after[PENSION_CREDIT_MIXED_AGE_SAVING].tolist() == [False, True, False]
    assert after[PENSION_CREDIT_MIXED_AGE_SAVING].dtype == bool
    for name in PENSION_CREDIT_ENTITLEMENT_VARIABLES:
        assert name not in after
    evidence = result.evidence()
    assert [band["band"] for band in evidence["bands"]] == [
        "guarantee_credit",
        "savings_credit_only",
    ]
    assert evidence["bands"][0]["entitled_units"] == 1


def test_committed_manifest_declares_the_stage_and_its_rewrite() -> None:
    stage = _stage()

    assert stage.grain == "benunit"
    assert stage.outputs == (PENSION_CREDIT_MIXED_AGE_SAVING,)
    assert stage.rewrites == ("would_claim_pc",)
    _assert_stage_parameters(stage)
    transform = UKPensionCreditTakeUpStageTransform(stage=stage, engine=_StubEngine())
    transform(_frame())
    assert transform.checkpoint_metadata()["evidence"]["stage"] == (
        PENSION_CREDIT_TAKE_UP_STAGE_NAME
    )


def test_stage_parameters_refuse_a_drifted_band() -> None:
    stage = _stage()
    operations = []
    for operation in stage.operations:
        payload = {"kind": operation.kind, **dict(operation.parameters)}
        if operation.kind == "assign_component_take_up_residual":
            bands = [dict(band) for band in payload["bands"]]
            bands[1]["rate_key"] = "pension_credit"
            payload["bands"] = bands
        operations.append(payload)
    drifted = SourceStageSpec.from_mapping({**stage.__dict__, "operations": operations})
    with pytest.raises(ValueError, match="parameters drifted"):
        _assert_stage_parameters(drifted)
    with pytest.raises(ValueError, match="expected 'pension_credit_take_up'"):
        _assert_stage_parameters(replace(stage, stage="frs_take_up"))


def test_contract_carries_the_fye2024_component_rates() -> None:
    contract = load_uk_take_up_contract()

    assert contract.rate("pension_credit_guarantee_credit") == 0.69
    assert contract.rate("pension_credit_savings_credit_only") == 0.37


def test_terminal_signal_gate_leaves_would_claim_pc_to_the_stage_gate() -> None:
    assert all(output != "would_claim_pc" for _, output, _ in UK_TAKE_UP_SIGNAL_OUTPUTS)
