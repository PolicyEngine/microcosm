"""The acceptance gates: each failure mode named, each pass earned.

The aggregate-vs-admin tests replay the two real incidents (net-STCG sign
flip, investment-interest blow-up) as the cases the gate must catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from populace.build import (
    GateReport,
    GateResult,
    aggregate_admin_gate,
    parity_gate,
    per_family_fit_gate,
    relative_error_loss,
    support_gate,
)
from populace.calibrate import TargetSpec


class TestGateResultInvariants:
    def test_pass_with_failures_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot pass with failures"):
            GateResult(name="x", passed=True, failures=("oops",))

    def test_fail_without_reason_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot fail without naming"):
            GateResult(name="x", passed=False)

    def test_report_aggregates(self) -> None:
        report = GateReport(
            (
                GateResult(name="a", passed=True),
                GateResult(name="b", passed=False, failures=("broke",)),
            )
        )
        assert not report.passed
        assert report.failures == ("[b] broke",)
        manifest = report.to_manifest()
        assert manifest["passed"] is False
        assert manifest["gates"]["b"]["failures"] == ["broke"]


class TestParityGate:
    def test_gap_fails_with_the_variable_named(self) -> None:
        result = parity_gate(
            {"snap": 0.1, "net_worth": 0.0},
            {"snap": 0.12, "net_worth": 0.9},
        )
        assert not result.passed
        assert "net_worth" in result.failures[0]

    def test_full_parity_passes(self) -> None:
        result = parity_gate(
            {"snap": 0.1, "net_worth": 0.5},
            {"snap": 0.12, "net_worth": 0.9},
        )
        assert result.passed
        assert result.details["gaps"] == 0

    def test_candidate_extras_are_not_failures(self) -> None:
        result = parity_gate({"snap": 0.1, "tips": 0.004}, {"snap": 0.12})
        assert result.passed

    def test_known_gap_is_exempt_and_recorded(self) -> None:
        result = parity_gate(
            {"snap": 0.1}, {"snap": 0.1, "rare_var": 0.01}, known_gaps=["rare_var"]
        )
        assert result.passed
        assert result.details["exempted"] == ["rare_var"]


class TestSupportGate:
    def test_within_donor_support_passes(self) -> None:
        result = support_gate(
            {"stcg": np.asarray([-50.0, 0.0, 120.0])},
            {"stcg": (-100.0, 200.0)},
        )
        assert result.passed

    def test_escape_fails_with_ranges_shown(self) -> None:
        result = support_gate(
            {"stcg": np.asarray([-500.0, 0.0])}, {"stcg": (-100.0, 200.0)}
        )
        assert not result.passed
        assert "outside donor support" in result.failures[0]

    def test_missing_range_declaration_fails(self) -> None:
        result = support_gate({"stcg": np.asarray([1.0])}, {})
        assert not result.passed
        assert "no donor range declared" in result.failures[0]


class TestAggregateAdminGate:
    def _stcg_anchor(self) -> TargetSpec:
        return TargetSpec(
            name="puf/net_short_term_capital_gains",
            entity="household",
            value=-76.8e9,
            signed=True,
            source="IRS SOI (PUF P22250, uprated)",
            family="irs_soi",
        )

    def test_sign_flip_is_caught(self) -> None:
        # The v2 incident: calibration drove net STCG to -$3.9T... and a
        # positive build would be just as wrong. Flip vs the signed anchor:
        result = aggregate_admin_gate(
            {"puf/net_short_term_capital_gains": +3.9e12},
            [self._stcg_anchor()],
        )
        assert not result.passed
        assert "sign flip" in result.failures[0]

    def test_order_of_magnitude_blowup_is_caught(self) -> None:
        # The investment-interest incident: $33.5T against tens of billions.
        anchor = TargetSpec(
            name="soi/investment_interest_expense",
            entity="household",
            value=24.0e9,
            source="IRS SOI Table 1.4",
            family="irs_soi",
        )
        result = aggregate_admin_gate(
            {"soi/investment_interest_expense": 33.5e12}, [anchor]
        )
        assert not result.passed
        assert "relative miss" in result.failures[0]

    def test_within_tolerance_passes(self) -> None:
        result = aggregate_admin_gate(
            {"puf/net_short_term_capital_gains": -80e9},
            [self._stcg_anchor()],
        )
        assert result.passed

    def test_declared_but_unmeasured_anchor_fails(self) -> None:
        result = aggregate_admin_gate({}, [self._stcg_anchor()])
        assert not result.passed
        assert "did not measure" in result.failures[0]

    def test_spec_tolerance_overrides_default(self) -> None:
        anchor = TargetSpec(
            name="census/population",
            entity="household",
            value=334e6,
            tolerance=1e6,  # absolute
            source="Census",
            family="census",
        )
        tight_miss = aggregate_admin_gate({"census/population": 340e6}, [anchor])
        assert not tight_miss.passed  # 6M off against a 1M tolerance


class TestPerFamilyFitGate:
    def test_collapsed_family_cannot_hide(self) -> None:
        names = [f"good/t{i}" for i in range(20)] + [f"bad/t{i}" for i in range(6)]
        errors = [0.01] * 20 + [0.5] * 6
        result = per_family_fit_gate(names, errors)
        assert not result.passed
        assert result.failures and "bad" in result.failures[0]
        # the global average would have looked fine:
        global_share = np.mean([abs(e) <= 0.10 for e in errors])
        assert global_share > 0.7

    def test_small_families_report_but_do_not_gate(self) -> None:
        names = (
            ["good/t0"] * 0
            + [f"good/t{i}" for i in range(10)]
            + [
                "tiny/t0",
                "tiny/t1",
            ]
        )
        errors = [0.01] * 10 + [0.9, 0.9]
        result = per_family_fit_gate(names, errors, min_family_size=5)
        assert result.passed
        assert result.details["family_within_shares"]["tiny"] == 0.0

    def test_misalignment_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must align"):
            per_family_fit_gate(["a/b"], [0.1, 0.2])


class TestRelativeErrorLoss:
    def test_matches_the_calibrator_formula(self) -> None:
        est = np.asarray([110.0, 90.0])
        tgt = np.asarray([100.0, 100.0])
        expected = (((est - tgt) / (tgt + 1.0)) ** 2).mean()
        assert relative_error_loss(est, tgt) == pytest.approx(expected)

    def test_shape_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must align"):
            relative_error_loss(np.zeros(2), np.zeros(3))


class TestZeroValuedAnchor:
    def test_zero_anchor_gates_on_absolute_scale(self) -> None:
        anchor = TargetSpec(
            name="soi/net_operating_loss_carryforward_count",
            entity="household",
            value=0.0,
            source="IRS SOI (no filers expected)",
            family="irs_soi",
        )
        assert aggregate_admin_gate({anchor.name: 0.0}, [anchor]).passed
        assert aggregate_admin_gate({anchor.name: 0.3}, [anchor]).passed
        assert not aggregate_admin_gate({anchor.name: 0.8}, [anchor]).passed
