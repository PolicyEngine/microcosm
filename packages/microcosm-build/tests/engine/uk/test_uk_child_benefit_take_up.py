"""Round-trip the registered-claim export through the installed UK model."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from policyengine_uk import Simulation
from policyengine_uk.data.dataset_schema import UKMultiYearDataset, UKSingleYearDataset
from policyengine_uk.utils.scenario import Scenario

from microcosm.build.uk_runtime.child_benefit_take_up import (
    redraw_child_benefit_take_up,
    uk_child_benefit_charge_thresholds,
)
from test_support.microcosm_build.uk_child_benefit_take_up import (
    _frame,
    _statistics,
    _StubEngine,
    _taper_frame,
)


@pytest.mark.parametrize("supports_opt_out", [False, True])
def test_installed_parameter_boundary_selects_export_capability(
    monkeypatch,
    supports_opt_out: bool,
) -> None:
    import policyengine_core.parameters

    parameters = _parameter_tree(supports_opt_out, 0.5)
    monkeypatch.setattr(
        policyengine_core.parameters, "ParameterNode", lambda **kwargs: parameters
    )
    thresholds = uk_child_benefit_charge_thresholds(2026)
    assert thresholds.supports_opt_out is supports_opt_out
    assert thresholds.opt_out_charge_share == (0.5 if supports_opt_out else None)
    assert "policyengine-uk " in thresholds.source


def _abolish_child_benefit_charge(simulation):
    simulation.tax_benefit_system.neutralize_variable("CB_HITC")


@pytest.mark.parametrize(
    "reform",
    [
        Scenario(simulation_modifier=_abolish_child_benefit_charge),
        Scenario.from_reform(
            {
                "gov.hmrc.income_tax.charges.CB_HITC.phase_out_end": {
                    "2026": float("inf")
                }
            }
        ),
    ],
)
def test_registered_claim_export_round_trips_baseline_and_charge_relief(reform):
    thresholds = uk_child_benefit_charge_thresholds(2026)
    # The reporter and fully charged two-child family claim; the family
    # whose eldest child is age zero does not claim. All opt-outs come from
    # the fully charged family. This isolates the export/payment contract.
    statistics = replace(
        _statistics({0: 0.0, 1: 1.0, 2: 1.0}), families_opted_out=500.0
    )
    result = redraw_child_benefit_take_up(
        _frame(), engine=_StubEngine(), statistics=statistics, thresholds=thresholds
    )
    benunit = result.frame.table("benunit")
    assert benunit["would_claim_child_benefit"].tolist() == [
        True,
        thresholds.supports_opt_out,
        False,
        True,
    ]
    assert benunit["child_benefit_opts_out"].tolist() == [False, True, False, False]
    dataset = UKSingleYearDataset(
        person=result.frame.table("person").rename(
            columns={"stub_income": "adjusted_net_income"}
        ),
        benunit=benunit,
        household=result.frame.table("household"),
        fiscal_year=2026,
    )
    simulation = Simulation(
        dataset=UKMultiYearDataset(datasets=[dataset]), scenario=reform
    )
    baseline = np.asarray(simulation.baseline.calculate("child_benefit", 2026))
    payment = np.asarray(simulation.calculate("child_benefit", 2026))
    entitlement = np.asarray(simulation.calculate("child_benefit_entitlement", 2026))
    assert baseline[0] == pytest.approx(entitlement[0])
    assert baseline[1:].tolist() == [0, 0, 0]
    assert payment[0] == pytest.approx(entitlement[0])
    assert payment[1] == pytest.approx(
        entitlement[1] if thresholds.supports_opt_out else 0
    )
    assert payment[2:].tolist() == [0, 0]
    assert np.asarray(simulation.calculate("CB_HITC", 2026)).sum() == 0
    assert result.evidence()["claim_export"]["encoding"] == (
        "registered_claims" if thresholds.supports_opt_out else "legacy_payment"
    )


@pytest.mark.parametrize("share", [1.0, 0.5, 0.0, 1.1])
def test_taper_and_full_charge_draws_agree_with_baseline_model_payments(share):
    thresholds = uk_child_benefit_charge_thresholds(2026)
    if thresholds.supports_opt_out:
        thresholds = replace(thresholds, opt_out_charge_share=share)
    frame = _taper_frame()
    result = redraw_child_benefit_take_up(
        frame,
        engine=_StubEngine(),
        statistics=replace(
            _statistics({0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0}),
            families_opted_out=900.0,
        ),
        thresholds=thresholds,
    )
    # Weight 10 reporter; weight 2 full-charge family with two children;
    # weight 20 half-charge family; childless reporter; genuine nonclaimant.
    full_opt_out = not thresholds.supports_opt_out or share <= 1
    taper_opt_out = not thresholds.supports_opt_out or share <= 0.5
    paid = np.asarray([True, not full_opt_out, not taper_opt_out, False, False])
    benunit = result.frame.table("benunit")
    assert benunit["child_benefit_opts_out"].tolist() == [
        False,
        full_opt_out,
        taper_opt_out,
        False,
        False,
    ]
    assert benunit["would_claim_child_benefit"].tolist() == [
        True,
        thresholds.supports_opt_out or not full_opt_out,
        thresholds.supports_opt_out or not taper_opt_out,
        True,
        False,
    ]
    dataset = UKSingleYearDataset(
        person=result.frame.table("person").rename(
            columns={"stub_income": "adjusted_net_income"}
        ),
        benunit=benunit,
        household=result.frame.table("household"),
        fiscal_year=2026,
    )
    scenario = Scenario(
        parameter_changes={
            "gov.hmrc.child_benefit.opt_out_charge_share": {"2026": share}
        }
        if thresholds.supports_opt_out
        else {}
    )
    simulation = Simulation(
        dataset=UKMultiYearDataset(datasets=[dataset]), scenario=scenario
    )
    payment = np.asarray(simulation.calculate("child_benefit", 2026))
    entitlement = np.asarray(simulation.calculate("child_benefit_entitlement", 2026))
    weights = frame.weights_for("household").values
    children = np.asarray([1, 2, 1, 0, 1])
    assert payment.tolist() == pytest.approx((entitlement * paid).tolist())
    assert result.in_payment["families"] == float(weights[payment > 0].sum())
    assert result.in_payment["children"] == float(
        (weights * children)[payment > 0].sum()
    )
    assert float(weights @ payment) == pytest.approx(
        float(weights @ (entitlement * paid))
    )
    assert result.opt_outs["pool_shortfall_families"] == pytest.approx(
        28.8 - 2 * full_opt_out - 20 * taper_opt_out
    )


def _parameter_tree(supports_opt_out, share):
    children = (
        {"opt_out_charge_share": lambda instant: share} if supports_opt_out else {}
    )
    return SimpleNamespace(
        gov=SimpleNamespace(
            hmrc=SimpleNamespace(
                child_benefit=SimpleNamespace(children=children, **children),
                income_tax=SimpleNamespace(
                    charges=SimpleNamespace(
                        CB_HITC=SimpleNamespace(
                            phase_out_start=lambda instant: 60_000.0,
                            phase_out_end=lambda instant: 80_000.0,
                        )
                    )
                ),
            )
        )
    )


@pytest.mark.parametrize("share", [None, float("nan"), float("inf"), "1", True])
def test_present_invalid_share_is_not_treated_as_legacy_capability(monkeypatch, share):
    import policyengine_core.parameters

    monkeypatch.setattr(
        policyengine_core.parameters,
        "ParameterNode",
        lambda **kwargs: _parameter_tree(True, share),
    )
    with pytest.raises(ValueError, match="opt_out_charge_share.*finite numeric"):
        uk_child_benefit_charge_thresholds(2026)
