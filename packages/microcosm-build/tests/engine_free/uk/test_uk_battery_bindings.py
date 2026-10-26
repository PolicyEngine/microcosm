"""The UK consumer half of the gate battery (microcosm#611 increment 1).

Fixtures are synthetic throughout: no UKDS unit records. The schema-3
aggregator retired in #654; these tests pin the battery-side behavior that
survived the differential receipt.
"""

from __future__ import annotations

import base64
from datetime import date, datetime
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from microcosm.build import load_country_spec
from microcosm.build.country_spec import GatesManifest
from microcosm.build.gate_battery import (
    BlockingMode,
    EvidenceContext,
    GateBatteryBlockedError,
    GateBatteryRun,
    GateStatus,
    _evaluate_gate,
    evaluate_phase,
    gate_signing_key_env,
    validate_gate_parameters,
)
from microcosm.build.gates import FitWeightRecord, GateResult
from microcosm.build.uk_runtime.battery_bindings import (
    UK_GATE_REGISTRY,
    UKGateBinding,
    _ledger_compile_parity_registry,
)
from microcosm.build.uk_runtime.national_frame import (
    _uk_gate_surface,
    uk_household_weight_kind,
    uk_national_frame,
)
from microcosm.build.uk_runtime.release_input_coverage import (
    UKReleaseInputColumn,
    UKReleaseInputCoverageManifest,
    load_uk_release_input_coverage_manifest,
    uk_release_input_coverage_gate,
)
from microcosm.build.uk_runtime.terminal_gates import (
    UKInputMassReference,
)
from microcosm.build.uk_runtime.weighted_integrity import UKReviewedExclusion
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import engine_tables
from tools.generate_uk_local_target_references import SUPPORT_FLOOR_DEFERRED_AREA_IDS

KEY = base64.b64encode(b"\x07" * 32).decode("ascii")
#: The shared exclusion-expiry clock, fixed inside the committed registers'
#: validity window (latest approval 2026-09-23, the two v21b deferrals on the
#: target-fit register from the PolicyEngine/chronicle#280 lane; earliest expiry
#: 2026-10-21, the same entries) so the suite never drifts across an approval
#: or expiry boundary. Move it forward when a register gains a later approval.
# The committed registers are evaluated as of this date; the microcosm#1063
# c9 input-mass entries take force on 2026-10-02, the
# microcosm#1095 UC payment-band measure exclusions on 2026-10-07 (its West
# Midlands target-fit deferral is retired with microcosm#1106), and the
# microcosm#1106 property_wealth input-mass entry on 2026-10-11. The earliest
# expiry is 2026-10-15.
# On or after the newest committed receipt's approval (2026-10-14).
CLOCK = date(2026, 10, 14)

VALIDATE_REFERENCE = (
    "microcosm.build.uk_runtime.weighted_integrity."
    "_validate_input_mass_reference_for_descriptor"
)


@pytest.fixture(autouse=True)
def signing_env(monkeypatch) -> None:
    monkeypatch.setenv(gate_signing_key_env("uk"), KEY)


@pytest.fixture(scope="module")
def uk_gates():
    return load_country_spec("uk").gates


def test_declared_export_surface_preserves_claimants_and_matches_runtime(uk_gates):
    from microcosm.build.uk_runtime.terminal_gates import (
        UK_ALLOWED_EXTRA_EXPORT_COLUMNS,
    )

    entry = next(entry for entry in uk_gates.gates if entry.id == "uk_export_surface")
    declared = set(entry.parameters["allowed_extra_columns"])
    assert "person.is_uc_claimant" in declared
    assert declared == set(UK_ALLOWED_EXTRA_EXPORT_COLUMNS)
    battery = _run_battery(
        _tables(),
        parity=_parity(candidate_columns={"person.age", "person.is_uc_claimant"}),
    )
    outcome = next(o for o in battery.outcomes if o.entry.id == "uk_export_surface")
    assert outcome.status is GateStatus.PASSED


def test_spine_export_surface_is_reference_or_allow_listed(uk_gates):
    """Every column the spine declares leaves through the release boundary as
    an enhanced-FRS input, an allow-listed extra or a reviewed drop.

    The export-surface gate runs only at release-cut certification, on a built
    file; war_widows_pension_reported (#1222) reached main without an
    allow-list entry and blocked the first certification after it. This runs
    the same gate on the spine's declared surface through the production
    export path, against the checked-in enhanced-FRS reference.
    """

    from microcosm.build.uk_runtime.graph import uk_spine_endpoint, uk_spine_graph
    from microcosm.build.uk_runtime.national_frame import (
        uk_national_frame,
        uk_release_export_frame,
    )
    from microcosm.build.uk_runtime.parity_reference import (
        load_efrs_parity_reference,
    )
    from microcosm.build.uk_runtime.terminal_gates import (
        uk_export_candidate_columns,
        uk_export_surface_gate,
    )

    columns = {
        "person": ["person_id", "person_benunit_id", "person_household_id"],
        "benunit": ["benunit_id"],
        "household": ["household_id"],
    }
    for cells in uk_spine_endpoint(uk_spine_graph()).inputs:
        columns[cells.entity] = list(
            dict.fromkeys([*columns[cells.entity], *cells.columns])
        )
    tables = {
        entity: pd.DataFrame({column: np.array([1]) for column in names})
        for entity, names in columns.items()
    }
    frame = uk_national_frame(**tables, time_period="2024", household_weights=[1.0])
    candidate = uk_export_candidate_columns(uk_release_export_frame(frame))
    reference = {
        f"{entity}.{name}"
        for name, entity in load_efrs_parity_reference().input_entities.items()
    }
    entry = next(entry for entry in uk_gates.gates if entry.id == "uk_export_surface")
    result = uk_export_surface_gate(candidate, reference, **dict(entry.parameters))
    assert result.passed, result.failures


def _tables(*, n: int = 4, weights=None):
    if weights is None:
        weights = np.ones(n, dtype=float)
    household_ids = np.arange(1, n + 1, dtype=np.int64)
    # One person per benefit unit and household, so each person is the
    # household reference person and the unit holds the household's property
    # at 90% of its value: other property for UC, and that with land for
    # Pension Credit (microcosm#1095). The units that report Universal Credit
    # hold none here.
    claimant = np.arange(n) % 3 != 2
    other_property = 1_000.0 * (np.arange(n) % 2)
    non_residential = 500.0 * (np.arange(n) % 2)
    land = 250.0 * (np.arange(n) % 2)
    # Debt secured on the last unit's property (microcosm#1095): the shares
    # net it class by class, the non-residential debt above 90% of its
    # property's value flooring that class at zero.
    last = np.arange(n) == n - 1
    other_property_debt = np.where(last, 300.0, 0.0)
    non_residential_debt = np.where(last, 600.0, 0.0)
    property_share = np.maximum(0.9 * other_property - other_property_debt, 0.0) + (
        np.maximum(0.9 * non_residential - non_residential_debt, 0.0)
    )
    carrier = np.arange(n, dtype=float)
    person = pd.DataFrame(
        {
            "person_id": np.arange(101, 101 + n, dtype=np.int64),
            "person_household_id": household_ids,
            "person_benunit_id": np.arange(201, 201 + n, dtype=np.int64),
            "employment_income": np.arange(1, n + 1, dtype=float),
            "universal_credit_reported": np.asarray([10.0, 0.0] * n)[:n],
            # Distinct sub-exempt gains: not degenerate, and none crosses the
            # frozen 3,000 under the fake projection's pinned growth path.
            "capital_gains": np.linspace(500.0, 2_000.0, n),
            "age": 30 + 10 * np.arange(n),
            "is_uc_claimant": claimant,
            "is_household_head": True,
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": np.arange(201, 201 + n, dtype=np.int64),
            "would_claim_uc": np.asarray([True, False] * n)[:n],
            "frs_benunit_capital": carrier,
            "uc_reported_capital": carrier + property_share,
            "pension_credit_reported_capital": carrier + 0.9 * land + property_share,
        }
    )
    household = pd.DataFrame(
        {
            "household_id": household_ids,
            "household_weight": np.asarray(weights, dtype=float),
            "household_is_spi_synthetic": np.arange(n) % 2 == 1,
            "household_is_capital_gains_clone": np.arange(n) % 4 >= 2,
            "owned_land": land,
            "other_residential_property_value": other_property,
            "non_residential_property_value": non_residential,
            "other_residential_property_secured_debt": other_property_debt,
            "non_residential_property_secured_debt": non_residential_debt,
        }
    )
    return person, benunit, household


def _coverage() -> GateResult:
    return GateResult(
        name="uk_release_input_coverage",
        passed=True,
        details={"fixture": True},
    )


def _parity(**overrides) -> SimpleNamespace:
    fields = {
        "candidate_columns": {"person.age"},
        "reference_columns": {"person.age"},
        "candidate_targets": {"ons/population"},
        "reference_targets": {"ons/population"},
        "target_relative_errors": {"ons/population": 0.01},
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _reference() -> UKInputMassReference:
    return UKInputMassReference(
        totals={"employment_income": 10.0},
        filename="enhanced_frs_2023_24.h5",
        revision="655dd07e4bb9c777b00dac044949611f1feb824f",
        sha256="584ae33d80ca0431254610a3f8254d132da73477d31966d6446282861ecae50d",
        vintage="2023_24",
    )


def _fixture_coverage_registry():
    """The UK registry with the coverage gate fed by the same fixture the
    differential harness used before retiring schema 3. The real coverage
    gate has its own dedicated tests. The fixture mints the legacy name, so
    the re-minting path stays exercised."""

    return {
        **UK_GATE_REGISTRY,
        "release_input_coverage": UKGateBinding(
            name="release_input_coverage",
            evaluator=lambda context, parameters: _coverage(),
            parameter_keys=frozenset({"check"}),
            legacy_name="uk_release_input_coverage",
            needs_frame=False,
        ),
        "take_up_signal": UKGateBinding(
            name="take_up_signal",
            evaluator=lambda context, parameters: GateResult(
                name="take_up_signal", passed=True
            ),
            parameter_keys=UK_GATE_REGISTRY["take_up_signal"].parameter_keys,
        ),
        "enum_domain": UKGateBinding(
            name="enum_domain",
            evaluator=lambda context, parameters: GateResult(
                name="enum_domain", passed=True
            ),
            parameter_keys=UK_GATE_REGISTRY["enum_domain"].parameter_keys,
        ),
    }


def _fake_cgt_projection(base_year: int = 2023, horizon_year: int = 2030):
    """A projection on the manifest's pinned growth path and exempt amounts.

    The seam reads the engine; tests supply the pins themselves, which is
    also what the seam falls back to when no engine is installed.
    """

    from microcosm.build.uk_runtime.cgt_projection import (
        UK_CGT_EXEMPT_AMOUNT_PARAMETER,
        UK_CGT_GAINS_GROWTH_PARAMETER,
        UKCGTProjection,
    )

    parameters = next(
        entry
        for entry in load_country_spec("uk").gates.gates
        if entry.id == "uk_cgt_projection_entrants"
    ).parameters
    pinned_growth = parameters["expected_yoy_growth_by_year"]
    pinned_exempt = parameters["expected_exempt_amount_by_year"]
    growth: dict[str, float] = {}
    cumulative: dict[str, float] = {}
    factor = 1.0
    for year in range(base_year + 1, horizon_year + 1):
        rate = float(pinned_growth[str(year)])
        factor *= 1.0 + rate
        growth[str(year)] = rate
        cumulative[str(year)] = factor
    return UKCGTProjection(
        base_year=base_year,
        horizon_year=horizon_year,
        growth_parameter=UK_CGT_GAINS_GROWTH_PARAMETER,
        exempt_amount_parameter=UK_CGT_EXEMPT_AMOUNT_PARAMETER,
        yoy_growth_by_year=growth,
        cumulative_gains_factor_by_year=cumulative,
        exempt_amount_by_year={
            str(year): float(pinned_exempt[str(year)])
            for year in range(base_year, horizon_year + 1)
        },
        engine="test",
    )


def _run_battery(
    tables,
    *,
    parity=None,
    fit_records=None,
    armed=True,
    clock=CLOCK,
    extra_artifacts=None,
):
    person, benunit, household = tables
    # A dependant in the first household, so is_household_head varies as it
    # does in a real frame: the degenerate-surface gate refuses a constant
    # column. The dependant adds no income, receipt or taxable gain.
    dependant = person.iloc[[0]].assign(
        person_id=int(person["person_id"].max()) + 1,
        employment_income=0.0,
        universal_credit_reported=0.0,
        capital_gains=250.0,
        age=5,
        is_uc_claimant=False,
        is_household_head=False,
    )
    person = pd.concat([person, dependant], ignore_index=True)
    frame = uk_national_frame(
        person=person, benunit=benunit, household=household, time_period="2023"
    )
    artifacts: dict[str, object] = {
        "coverage_engine": object(),
        "exclusions_evaluated_on": clock,
        # This binding fixture schedules a stage with no declared nonnegative
        # outputs; canonical family completeness has dedicated tests.
        "build_stage_names": ("frs_household_draws",),
    }
    if fit_records is not None:
        artifacts["fit_weight_records"] = fit_records
    if parity is not None:
        artifacts["parity_evidence"] = parity
    if armed:
        artifacts["input_mass_reference"] = _reference()
        artifacts["aggregate_admin"] = {
            "nhs_spending_total": 202_000_000_000,
        }
        artifacts["cgt_projection"] = _fake_cgt_projection()
    if extra_artifacts:
        artifacts.update(extra_artifacts)
    # Small synthetic totals exercise battery behavior without disclosing
    # the licensed 131-column reference (same patch as the legacy tests);
    # the binding's declared-pin check compares spec to runtime constant and
    # needs no patching.
    with patch(VALIDATE_REFERENCE, return_value=None):
        return evaluate_phase(
            load_country_spec("uk").gates,
            "terminal",
            EvidenceContext(frame=frame, artifacts=artifacts),
            registry=_fixture_coverage_registry(),
        )


class TestUKSurfaceAdapter:
    def test_surface_materializes_the_frame_not_fallbacks(self) -> None:
        # The one surviving copy of the legacy duck-attr evidence surface
        # (the national build's adapter consolidated into it at the
        # orchestration swap). Every attr must resolve to the frame's real
        # values — a gate reading household_weight_kind or time_period must
        # never see a fallback.
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        surface = _uk_gate_surface(frame)
        tables = engine_tables(frame)

        pd.testing.assert_frame_equal(surface.person, tables["person"])
        pd.testing.assert_frame_equal(surface.benunit, tables["benunit"])
        pd.testing.assert_frame_equal(surface.household, tables["household"])
        assert surface.time_period == "2023"
        assert surface.household_weight_kind is uk_household_weight_kind(frame)
        assert surface.mass_log == frame.mass_log

    def _nonnegative_frame(self, *, sic: list[float] | None):
        person, benunit, household = _tables()
        if sic is not None:
            person["sic_industry_division"] = sic
        return uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2023",
        )

    def test_uc_column_implication_binding_aggregates_and_checks_carrier(self) -> None:
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2023",
        )
        entry = next(
            gate
            for gate in load_country_spec("uk").gates.gates
            if gate.id == "uk_uc_capital_coherence"
        )

        passing = UK_GATE_REGISTRY["column_implication"].evaluate(
            EvidenceContext(frame=frame), entry.parameters
        )
        assert passing.passed

        frame.table("benunit").loc[0, "would_claim_uc"] = False
        frame.table("benunit").loc[1, "uc_reported_capital"] = -1.0
        failing = UK_GATE_REGISTRY["column_implication"].evaluate(
            EvidenceContext(frame=frame), entry.parameters
        )
        assert not failing.passed
        assert any("must imply" in failure for failure in failing.failures)
        assert any("same sentinel" in failure for failure in failing.failures)
        assert any("must equal" in failure for failure in failing.failures)

    def test_uc_column_implication_binding_is_not_vacuous_at_amount_thresholds(
        self,
    ) -> None:
        # Adversarial-review finding 2: the configured threshold filters the
        # person-level amounts; the aggregated 0/1 indicator must always be
        # compared at 0. With the two conflated, any threshold >= 1 could
        # never flag a violation. This pins the de-conflation: a reporter
        # above a nonzero amount threshold with would_claim_uc=False must
        # still fail the gate.
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2023",
        )
        entry = next(
            gate
            for gate in load_country_spec("uk").gates.gates
            if gate.id == "uk_uc_capital_coherence"
        )
        parameters = {**dict(entry.parameters), "threshold": 100.0}
        frame.table("person")["universal_credit_reported"] = 500.0
        frame.table("benunit")["would_claim_uc"] = False

        failing = UK_GATE_REGISTRY["column_implication"].evaluate(
            EvidenceContext(frame=frame), parameters
        )
        assert not failing.passed
        assert any("must imply" in failure for failure in failing.failures)
        assert failing.details["amount_threshold"] == 100.0
        assert failing.details["threshold"] == 0.0

    def test_uc_column_implication_binding_refuses_the_undefined_interval(
        self,
    ) -> None:
        # Adversarial-review round-2 residual (a): the -1 contract defines
        # exactly two regions — the sentinel and nonnegative amounts. A
        # corrupted -0.5 previously cleared every capital check: above the
        # bare floor, not isclose to the sentinel on either side, finite,
        # and equal to a carrier carrying the same corruption. The domain
        # predicate must refuse it, on both columns.
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2023",
        )
        entry = next(
            gate
            for gate in load_country_spec("uk").gates.gates
            if gate.id == "uk_uc_capital_coherence"
        )
        frame.table("benunit")["uc_reported_capital"] = -0.5
        frame.table("benunit")["frs_benunit_capital"] = -0.5

        failing = UK_GATE_REGISTRY["column_implication"].evaluate(
            EvidenceContext(frame=frame), entry.parameters
        )
        assert not failing.passed
        assert any("outside the declared domain" in f for f in failing.failures)
        assert failing.details["capital_domain_violation_count"] > 0
        assert failing.details["carrier_domain_violation_count"] > 0
        # The corruption must NOT be reported as a sentinel or equality
        # mismatch — those checks legitimately pass on it, which is exactly
        # why the domain check exists.
        assert failing.details["same_source_mismatch_count"] == 0

    def test_uc_column_implication_binding_refuses_near_sentinel_values(
        self,
    ) -> None:
        # #833: the isclose band around -1 admitted the top sliver of the
        # undefined interval and read it as a declared absence. Sentinel
        # equality is exact; a corrupted -1.000005 on both columns must be a
        # domain violation, not a tolerated sentinel.
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2023",
        )
        entry = next(
            gate
            for gate in load_country_spec("uk").gates.gates
            if gate.id == "uk_uc_capital_coherence"
        )
        frame.table("benunit")["uc_reported_capital"] = -1.000005
        frame.table("benunit")["frs_benunit_capital"] = -1.000005

        failing = UK_GATE_REGISTRY["column_implication"].evaluate(
            EvidenceContext(frame=frame), entry.parameters
        )
        assert not failing.passed
        assert any("outside the declared domain" in f for f in failing.failures)
        assert failing.details["capital_domain_violation_count"] > 0
        assert failing.details["carrier_domain_violation_count"] > 0
        assert failing.details["sentinel_mismatch_count"] == 0
        assert failing.details["same_source_mismatch_count"] == 0

    @staticmethod
    def _capital_gate():
        return next(
            gate
            for gate in load_country_spec("uk").gates.gates
            if gate.id == "uk_uc_capital_coherence"
        )

    def test_uc_column_implication_binding_checks_both_capitals_with_property(
        self,
    ) -> None:
        # microcosm#1095: each recorded capital is the carrier plus the
        # household's property, at 90% of its value, for the household
        # reference person's unit: other property for UC, and that with land
        # for Pension Credit.
        person, benunit, household = _tables()
        # Unit 0 reports no Universal Credit here, so it records the property.
        person.loc[0, "universal_credit_reported"] = 0.0
        household.loc[0, "other_residential_property_value"] = 40_000.0
        household.loc[0, "owned_land"] = 10_000.0
        benunit.loc[0, "uc_reported_capital"] = 36_000.0
        benunit.loc[0, "pension_credit_reported_capital"] = 45_000.0
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        entry = self._capital_gate()
        evaluate = UK_GATE_REGISTRY["column_implication"].evaluate
        context = EvidenceContext(frame=frame)

        passing = evaluate(context, entry.parameters)
        assert passing.passed, passing.failures
        for rule in passing.details["property_shares"].values():
            assert rule["owners"] == "household_reference_person_unit"
            assert rule["valuation"] == 0.9

        # The carrier alone, without the property, fails on both columns.
        frame.table("benunit").loc[0, "uc_reported_capital"] = 0.0
        frame.table("benunit").loc[0, "pension_credit_reported_capital"] = 0.0
        failing = evaluate(context, entry.parameters)
        assert not failing.passed
        share_failures = [
            failure
            for failure in failing.failures
            if "plus the unit's declared property share" in failure
        ]
        assert len(share_failures) == 2
        columns = failing.details["capital_columns"]
        for column in ("uc_reported_capital", "pension_credit_reported_capital"):
            assert columns[f"benunit.{column}"]["same_source_mismatch_count"] == 1

    @pytest.mark.parametrize(
        ("reference_reports_uc", "expected"),
        [
            (False, {"uc_reported_capital": 1, "pension_credit_reported_capital": 1}),
            (True, {"uc_reported_capital": 0, "pension_credit_reported_capital": 1}),
        ],
        ids=["both_schemes", "pension_credit_only"],
    )
    def test_uc_column_implication_binding_refuses_a_sentinel_unit_beside_property(
        self, reference_reports_uc: bool, expected: dict[str, int]
    ) -> None:
        # From policyengine-uk 2.136.1 a unit on the sentinel takes its own
        # share of its household's property, which the recording unit beside
        # it already holds (Vahid's review of microcosm#1333). Unit 204 joins
        # household 2 on the sentinel, beside the reference person's unit 202,
        # which records the property: both schemes' shares, or Pension
        # Credit's alone where unit 202 reports Universal Credit.
        person, benunit, household = _tables()
        person.loc[3, "person_household_id"] = 2
        person.loc[3, "is_household_head"] = False
        for column in (
            "frs_benunit_capital",
            "uc_reported_capital",
            "pension_credit_reported_capital",
        ):
            benunit.loc[3, column] = -1.0
        if reference_reports_uc:
            person.loc[1, "universal_credit_reported"] = 10.0
            benunit.loc[1, "would_claim_uc"] = True
            benunit.loc[1, "uc_reported_capital"] = benunit.loc[
                1, "frs_benunit_capital"
            ]
        frame = uk_national_frame(
            person=person,
            benunit=benunit,
            household=household.iloc[:3].copy(),
            time_period="2023",
        )
        entry = self._capital_gate()
        result = UK_GATE_REGISTRY["column_implication"].evaluate(
            EvidenceContext(frame=frame), entry.parameters
        )

        assert not result.passed
        assert result.details["sentinel_units_beside_recorded_property"] == expected
        beside = [failure for failure in result.failures if "would count" in failure]
        assert len(beside) == sum(expected.values())

    def test_uc_column_implication_binding_keeps_a_reporters_receipt(
        self,
    ) -> None:
        # microcosm#1095: a unit that reports Universal Credit records the
        # carrier alone, whatever property its household holds.
        person, benunit, household = _tables()
        assert person.loc[0, "universal_credit_reported"] > 0
        household.loc[0, "other_residential_property_value"] = 40_000.0
        benunit.loc[0, "pension_credit_reported_capital"] = 36_000.0
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        entry = self._capital_gate()
        evaluate = UK_GATE_REGISTRY["column_implication"].evaluate
        context = EvidenceContext(frame=frame)

        passing = evaluate(context, entry.parameters)
        assert passing.passed, passing.failures

        # The property it would hold is refused on a reporting unit.
        frame.table("benunit").loc[0, "uc_reported_capital"] = 36_000.0
        failing = evaluate(context, entry.parameters)
        assert not failing.passed
        columns = failing.details["capital_columns"]
        assert columns["benunit.uc_reported_capital"]["same_source_mismatch_count"] == 1
        assert (
            columns["benunit.pension_credit_reported_capital"][
                "same_source_mismatch_count"
            ]
            == 0
        )

    def test_uc_column_implication_binding_tolerates_a_float32_round_trip(
        self,
    ) -> None:
        person, benunit, household = _tables()
        household.loc[1, "other_residential_property_value"] = 654_321.09
        household.loc[1, "non_residential_property_value"] = 0.0
        benunit.loc[1, "frs_benunit_capital"] = 123_456.78
        benunit.loc[1, "pension_credit_reported_capital"] = 123_456.78 + 0.9 * (
            household.loc[1, "owned_land"] + 654_321.09
        )
        expected = 123_456.78 + 0.9 * 654_321.09
        benunit.loc[1, "uc_reported_capital"] = float(np.float32(expected))
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        entry = self._capital_gate()
        evaluate = UK_GATE_REGISTRY["column_implication"].evaluate
        context = EvidenceContext(frame=frame)

        assert evaluate(context, entry.parameters).passed
        frame.table("benunit").loc[1, "uc_reported_capital"] = expected + 1.0
        assert not evaluate(context, entry.parameters).passed

    def test_uc_column_implication_binding_refuses_a_drifted_property_rule(
        self,
    ) -> None:
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        entry = self._capital_gate()
        parameters = {
            **dict(entry.parameters),
            "property_shares": {
                "uc_reported_capital": {
                    "sources": ["owned_land"],
                    "owners": "is_uc_claimant",
                }
            },
        }
        with pytest.raises(ValueError, match="not the recorded-capital rules"):
            UK_GATE_REGISTRY["column_implication"].evaluate(
                EvidenceContext(frame=frame), parameters
            )

    def test_nonnegative_binding_requires_scheduled_stage_columns(self) -> None:
        # frs_employment declares sic_industry_division nonnegative; a build
        # that scheduled the stage but lost the column must fail — the
        # missing-column path is the reason the required set is never
        # pre-filtered to present columns.
        binding = UK_GATE_REGISTRY["nonnegative_columns"]
        context = EvidenceContext(
            frame=self._nonnegative_frame(sic=None),
            artifacts={"build_stage_names": ("frs_employment",)},
        )

        result = binding.evaluate(context, {})

        assert result.passed is False
        assert "sic_industry_division" in result.failures[0]

    def test_nonnegative_binding_fails_on_negative_values(self) -> None:
        binding = UK_GATE_REGISTRY["nonnegative_columns"]
        context = EvidenceContext(
            frame=self._nonnegative_frame(sic=[1.0, -2.0, 3.0, 4.0]),
            artifacts={"build_stage_names": ("frs_employment",)},
        )

        result = binding.evaluate(context, {})

        assert result.name == "nonnegative_columns"
        assert result.passed is False
        assert (
            "sic_industry_division: 1 finite value(s) below zero" in result.failures[0]
        )

    def test_nonnegative_binding_passes_clean_scheduled_columns(self) -> None:
        binding = UK_GATE_REGISTRY["nonnegative_columns"]
        context = EvidenceContext(
            frame=self._nonnegative_frame(sic=[1.0, 0.0, 3.0, 4.0]),
            artifacts={"build_stage_names": ("frs_employment",)},
        )

        result = binding.evaluate(context, {})

        assert result.passed is True

    def _salary_sacrifice_frames(self, *, carrier):
        """The release frame without the pre-conversion pay carrier, the spine with it.

        The carrier leaves at the release boundary
        (UK_RELEASE_EXPORT_DROPPED_COLUMNS), so a certifier checking the
        salary_sacrifice stage's declared non-negative outputs on the release
        candidate finds it only on the spine frame (microcosm#1063).
        """

        from microcosm.build.uk_runtime.salary_sacrifice import (
            BONUS_SACRIFICE_OUTPUT,
            SALSAC_OUTPUT,
            SALSAC_PRE_CONVERSION_PAY_COLUMN,
        )

        person, benunit, household = _tables()
        person[SALSAC_OUTPUT] = [0.0, 10.0, 0.0, 5.0]
        # The stage's bonus route (#1340) is a declared non-negative output too.
        person[BONUS_SACRIFICE_OUTPUT] = [0.0, 0.0, 3.0, 0.0]
        person["employee_pension_contributions"] = [1.0, 0.0, 2.0, 0.0]
        release = uk_national_frame(
            person=person.copy(),
            benunit=benunit.copy(),
            household=household.copy(),
            time_period="2023",
        )
        spine_person = person.copy()
        spine_person[SALSAC_PRE_CONVERSION_PAY_COLUMN] = carrier
        spine = uk_national_frame(
            person=spine_person,
            benunit=benunit.copy(),
            household=household.copy(),
            time_period="2023",
        )
        return release, spine

    def test_nonnegative_binding_reads_export_dropped_columns_from_the_spine(
        self,
    ) -> None:
        binding = UK_GATE_REGISTRY["nonnegative_columns"]
        release, spine = self._salary_sacrifice_frames(carrier=[0.0, 30.0, 0.0, 20.0])
        # Without the spine frame the dropped carrier is a missing column.
        missing = binding.evaluate(
            EvidenceContext(
                frame=release, artifacts={"build_stage_names": ("salary_sacrifice",)}
            ),
            {},
        )
        assert missing.passed is False
        assert "salary_sacrifice_pre_conversion_pay" in missing.failures[0]
        # With it the carrier is checked where it lives.
        checked = binding.evaluate(
            EvidenceContext(
                frame=release,
                artifacts={
                    "build_stage_names": ("salary_sacrifice",),
                    "spine_frame": spine,
                },
            ),
            {},
        )
        assert checked.passed is True
        # The spine's values are really checked, not just their presence.
        _, negative_spine = self._salary_sacrifice_frames(
            carrier=[0.0, 30.0, -1.0, 20.0]
        )
        failing = binding.evaluate(
            EvidenceContext(
                frame=release,
                artifacts={
                    "build_stage_names": ("salary_sacrifice",),
                    "spine_frame": negative_spine,
                },
            ),
            {},
        )
        assert failing.passed is False
        assert (
            "salary_sacrifice_pre_conversion_pay: 1 finite value(s) below zero"
            in (failing.failures[0])
        )

    def test_only_export_dropped_columns_are_read_from_the_spine(self) -> None:
        from microcosm.build.uk_runtime.battery_bindings import (
            _export_dropped_columns_from_spine,
        )
        from microcosm.build.uk_runtime.frs_disability import (
            UK_INTERNAL_DISABILITY_REPORTED_COLUMNS,
        )

        release, spine = self._salary_sacrifice_frames(carrier=[0.0, 30.0, 0.0, 20.0])
        spine_person = spine.table("person").copy()
        spine_person["sic_industry_division"] = [1.0, 2.0, 3.0, 4.0]
        spine_person[UK_INTERNAL_DISABILITY_REPORTED_COLUMNS[0]] = 0.0
        # microcosm#1106: the WAS property_wealth total is the first household
        # column the export drops; the was_wealth stage declares it
        # non-negative, so the certifier reads it from the spine.
        spine_household = spine.table("household").copy()
        spine_household["property_wealth"] = [5.0, 0.0, 7.0, 0.0]
        spine = uk_national_frame(
            person=spine_person,
            benunit=spine.table("benunit"),
            household=spine_household,
            time_period="2023",
            household_weights=spine.weights_for("household").values,
        )
        found = _export_dropped_columns_from_spine(
            EvidenceContext(frame=release, artifacts={"spine_frame": spine}),
            [
                "salary_sacrifice_pre_conversion_pay",
                UK_INTERNAL_DISABILITY_REPORTED_COLUMNS[0],
                "property_wealth",
                "sic_industry_division",
                "never_declared",
            ],
        )
        # A column the export keeps is never answered from the spine, even
        # when the spine carries it; one absent from the spine is not found.
        assert set(found) == {
            "salary_sacrifice_pre_conversion_pay",
            UK_INTERNAL_DISABILITY_REPORTED_COLUMNS[0],
            "property_wealth",
        }
        assert found["property_wealth"].tolist() == [5.0, 0.0, 7.0, 0.0]
        assert found["salary_sacrifice_pre_conversion_pay"].tolist() == [
            0.0,
            30.0,
            0.0,
            20.0,
        ]
        assert (
            _export_dropped_columns_from_spine(
                EvidenceContext(frame=release, artifacts={}),
                ["salary_sacrifice_pre_conversion_pay"],
            )
            == {}
        )

    def test_nonnegative_binding_does_not_demand_unscheduled_stages(self) -> None:
        # This isolated stage declares no nonnegative outputs. Outputs of
        # unscheduled employment and income stages must not be demanded.
        binding = UK_GATE_REGISTRY["nonnegative_columns"]
        context = EvidenceContext(
            frame=self._nonnegative_frame(sic=None),
            artifacts={"build_stage_names": ("frs_household_draws",)},
        )

        result = binding.evaluate(context, {})

        assert result.passed is True

    def test_support_binding_passes_in_range_was_outputs(self) -> None:
        person, benunit, household = _tables(n=2)
        household["owned_land"] = [0.0, 100.0]
        household["cash_isa"] = [0.0, 1000.0]
        person["student_loan_balance"] = [0.0, 100.0]
        frame = uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2023",
        )
        binding = UK_GATE_REGISTRY["support"]

        result = binding.evaluate(
            EvidenceContext(frame=frame, artifacts={}),
            {
                "support_bounds_resources": [
                    "was_wealth_support_bounds.json",
                    "lcfs_consumption_support_bounds.json",
                    "etb_vat_support_bounds.json",
                    "etb_services_support_bounds.json",
                ]
            },
        )

        assert result.passed is True
        # The fixture's other residential and non-residential property are
        # WAS outputs too.
        assert result.details["columns_checked"] == 5

    def test_support_binding_fails_out_of_range_was_outputs(self) -> None:
        person, benunit, household = _tables(n=1)
        household["cash_isa"] = [99_999_999.0]
        frame = uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2023",
        )
        binding = UK_GATE_REGISTRY["support"]

        result = binding.evaluate(
            EvidenceContext(frame=frame, artifacts={}),
            {"support_bounds_resource": "was_wealth_support_bounds.json"},
        )

        assert result.passed is False
        assert "cash_isa" in result.failures[0]

    def test_support_binding_checks_e6_support_resources(self) -> None:
        person, benunit, household = _tables(n=1)
        household["full_rate_vat_expenditure_rate"] = [999_999.0]
        frame = uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2023",
        )
        binding = UK_GATE_REGISTRY["support"]

        result = binding.evaluate(
            EvidenceContext(frame=frame, artifacts={}),
            {"support_bounds_resources": ["etb_vat_support_bounds.json"]},
        )

        assert result.passed is False
        assert "full_rate_vat_expenditure_rate" in result.failures[0]

    def test_aggregate_admin_binding_checks_declared_anchors(self) -> None:
        binding = UK_GATE_REGISTRY["aggregate_admin"]

        result = binding.evaluate(
            EvidenceContext(
                frame=None,
                artifacts={
                    "aggregate_admin": {
                        "need_electricity_mean_spending": 882.91463,
                        "nhs_spending_total": 202_000_000_000,
                    }
                },
            ),
            {
                "default_rtol": 0.15,
                "anchors": [
                    {
                        "name": "need_electricity_mean_spending",
                        "entity": "household",
                        "measure": "electricity_consumption",
                        "value": 882.91463,
                        "period": "2023",
                        "source": "test",
                        "family": "need_energy",
                    },
                    {
                        "name": "nhs_spending_total",
                        "entity": "person",
                        "measure": "nhs_spending",
                        "value": 202_000_000_000,
                        "period": "2025_26",
                        "source": "test",
                        "family": "nhs",
                    },
                ],
            },
        )

        assert result.passed is True
        assert result.details["anchors_checked"] == 2


class TestUKCompatibility:
    """The BE plumbing test, run over the UK spec: an empty evidence context
    resolves every declared entry to a named gap and blocks the release."""

    def test_the_uk_spec_runs_as_declared_with_named_gaps(
        self, tmp_path, uk_gates
    ) -> None:
        run = GateBatteryRun(
            uk_gates,
            release_id="uk-test-build",
            report_path=tmp_path / "terminal_gates.json",
            release_candidate=True,
            registry=UK_GATE_REGISTRY,
        )
        run.run_phase("preflight", EvidenceContext())
        run.run_phase("assembled", EvidenceContext())
        run.run_phase("transferred", EvidenceContext())
        run.run_phase("terminal", EvidenceContext())
        report = run.report_payload()

        assert set(report["gates"]) == {entry.id for entry in uk_gates.gates}
        assert {gate["status"] for gate in report["gates"].values()} == {
            "evidence_absent"
        }
        assert report["shippable"] is False
        with pytest.raises(GateBatteryBlockedError):
            run.enforce("terminal", mode=BlockingMode.BLOCKS_ARTIFACT)

    def test_missing_evidence_names_its_keys(self, uk_gates) -> None:
        phase = evaluate_phase(
            uk_gates, "terminal", EvidenceContext(), registry=UK_GATE_REGISTRY
        )
        reasons = {o.entry.id: o.reason for o in phase.outcomes}
        assert reasons["uk_weights_audit"] == ("missing evidence: fit_weight_records")
        assert reasons["uk_input_mass_parity"] == (
            "missing evidence: frame, exclusions_evaluated_on, input_mass_reference"
        )
        assert reasons["uk_degenerate_release_surface"] == (
            "missing evidence: frame, exclusions_evaluated_on"
        )


class TestBatteryRegressions:
    def test_fully_armed_battery_evaluates_gate_for_gate(self) -> None:
        battery = _run_battery(
            _tables(),
            parity=_parity(),
            fit_records=(FitWeightRecord("spi_qrf", "importance"),),
        )

        by_id = {o.entry.id: o for o in battery.outcomes}
        passed = [
            entry_id for entry_id, o in by_id.items() if o.status is GateStatus.PASSED
        ]
        # 11 as on main (uk_nonnegative_columns passes with zero required
        # columns — the scheduled stages declare none), the take-up signal
        # gate, the E5 support gate, the E6 aggregate-admin gate, and the E8
        # student-loan enum gate; their evaluators have direct tests. The BRMA
        # enum gate is no longer among them: it moved to the spine battery's
        # assembled boundary, where its column is first written.
        # 15 before this lane, plus uk_uc_capital_coherence from #829 and the
        # UC-deduction enum gate from #685, plus
        # the two frame-only local weight diagnostics that also pass in this
        # unscoped compatibility probe. The local ladder gate fails because
        # this national fixture deliberately carries no ladder columns; the
        # three evidence-backed local arms are named gaps below.
        # Plus the #970 projection fence, armed here with the pinned growth
        # path over distinct sub-exempt gains that never cross the exempt
        # amount, and the release-frame council-tax band check (#1278).
        # Plus uk_local_brma_enum_domain, the terminal recheck of the area-BRMA
        # rewrite (microcosm#1262; its enum_domain binding is stubbed here).
        assert len(passed) == 22
        assert "uk_local_brma_enum_domain" in passed

    def test_input_mass_parity_compares_the_release_surface(self) -> None:
        # The frame carries the spine's WAS property_wealth and the reference
        # stores its own total within tolerance. Compared on the raw frame the
        # signed exclusion would be stale and the gate would fail; the gate
        # compares the export frame, where the column is absent, so the
        # exclusion is in force and nothing is stale (microcosm#1106).
        person, benunit, household = _tables()
        household = household.assign(property_wealth=100.0)
        reference = UKInputMassReference(
            totals={"employment_income": 10.0, "property_wealth": 400.0},
            filename="enhanced_frs_2023_24.h5",
            revision="655dd07e4bb9c777b00dac044949611f1feb824f",
            sha256="584ae33d80ca0431254610a3f8254d132da73477d31966d6446282861ecae50d",
            vintage="2023_24",
        )
        battery = _run_battery(
            (person, benunit, household),
            parity=_parity(),
            fit_records=(FitWeightRecord("spi_qrf", "importance"),),
            extra_artifacts={"input_mass_reference": reference},
        )
        outcome = {o.entry.id: o for o in battery.outcomes}["uk_input_mass_parity"]
        assert outcome.status is GateStatus.PASSED, outcome.result.failures
        details = outcome.result.details
        assert details["stale_exclusions"] == []
        assert "property_wealth" in details["reviewed_exclusions"]

    def test_empty_fit_records_fail_closed(self) -> None:
        # Present-but-empty is not absent: a fit stage that ran and emitted
        # nothing is a failed audit, never a vacuous pass.
        battery = _run_battery(_tables(), fit_records=())

        audit = {o.entry.id: o for o in battery.outcomes}["uk_weights_audit"]
        assert audit.status is GateStatus.FAILED
        assert "an absent audit is not a passing audit" in (audit.result.failures[0])

    def test_seeded_defects_fail_the_expected_gates(self) -> None:
        blown = _tables(weights=[1.0, 1.0, 1.0, 1.0e9])
        seeded_parity = _parity(
            candidate_columns={"person.age", "person.unreviewed_extra"},
            target_relative_errors={"ons/population": -0.40},
        )
        battery = _run_battery(
            blown,
            parity=seeded_parity,
            fit_records=(FitWeightRecord("spi_qrf", "none"),),
        )

        failed = {o.entry.id for o in battery.outcomes if o.status is GateStatus.FAILED}
        assert {
            "uk_weight_ratio",
            "uk_weights_audit",
            "uk_export_surface",
            "uk_target_fit",
        } <= failed

    def test_the_fit_gate_judges_a_thin_national_row(self) -> None:
        """María's ruling of 2026-10-26: no row's figures are withheld, so the
        fit gate judges a row however few source households carry it."""

        thin = "dwp.uc.payment_distribution_single_thin@2025"
        parity = _parity(
            candidate_targets={"ons/population", thin},
            reference_targets={"ons/population", thin},
            target_relative_errors={"ons/population": 0.01, thin: -0.60},
        )
        battery = _run_battery(_tables(), parity=parity)
        fit = {o.entry.id: o for o in battery.outcomes}["uk_target_fit"]
        assert fit.status is GateStatus.FAILED
        assert fit.result.details["failing_targets"] == {thin: -0.60}
        assert "withheld_targets" not in fit.result.details


class TestUnevidencedArms:
    """Missing evidence is explicit; it blocks release candidates, plus any
    entry whose manifest declares absence non-excusable in every posture
    (``uk_weights_audit`` — "an absent audit is not a passing audit", the
    legacy strictness ported during the #654 retirement)."""

    def test_battery_records_evidence_absent(self, uk_gates) -> None:
        battery = _run_battery(_tables(), armed=False)

        absent = {
            o.entry.id: o.reason
            for o in battery.outcomes
            if o.status is GateStatus.EVIDENCE_ABSENT
        }
        assert set(absent) == {
            "uk_weights_audit",
            "uk_export_surface",
            "uk_calibration_reference_coverage",
            "uk_target_surface",
            "uk_target_fit",
            "uk_cgt_projection_entrants",
            "uk_input_mass_parity",
            "uk_aggregate_admin",
            "uk_local_area_support",
            "uk_local_target_fit",
            "uk_local_per_family_fit",
            # microcosm#1240: the two report-only high-value homes gates read
            # the evidence the terminal kernel computes on the calibrated pool.
            "uk_high_value_homes_benchmarks",
            "uk_high_value_homes_support",
        }
        for reason in absent.values():
            assert reason.startswith("missing evidence: ")

        # The audit's absence blocks even the default posture — its status
        # stays honestly evidence_absent; only the enforcement is strict.
        default_blocked = {
            o.entry.id for o in battery.blocking_outcomes(release_candidate=False)
        }
        assert default_blocked == {
            "uk_local_geography_ladder_post_calibration",
            "uk_weights_audit",
            # The #970 projection fence cannot be skipped in any posture.
            "uk_cgt_projection_entrants",
        }
        blocked = {
            o.entry.id for o in battery.blocking_outcomes(release_candidate=True)
        }
        # The two diagnostic gates report their missing evidence and block
        # nothing in any posture.
        diagnostic = {"uk_high_value_homes_benchmarks", "uk_high_value_homes_support"}
        assert blocked == {
            *(set(absent) - diagnostic),
            "uk_local_geography_ladder_post_calibration",
        }

    def test_absent_fit_evidence_is_named(self) -> None:
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        battery = evaluate_phase(
            load_country_spec("uk").gates,
            "terminal",
            EvidenceContext(frame=frame, artifacts={"coverage_engine": object()}),
            registry=_fixture_coverage_registry(),
        )
        audit = {o.entry.id: o for o in battery.outcomes}["uk_weights_audit"]
        assert audit.status is GateStatus.EVIDENCE_ABSENT
        assert audit.reason == "missing evidence: fit_weight_records"


class TestExclusionDiscipline:
    """One expiry clock, a committed register of record, a loud override."""

    EXCLUSION_GATES = (
        "uk_degenerate_release_surface",
        "uk_input_mass_parity",
    )

    def test_every_exclusion_gate_shares_the_injected_clock(self) -> None:
        battery = _run_battery(
            _tables(),
            parity=_parity(),
            fit_records=(FitWeightRecord("spi_qrf", "importance"),),
        )
        by_id = {o.entry.id: o for o in battery.outcomes}
        stamps = {
            entry_id: by_id[entry_id].result.details["exclusions_evaluated_on"]
            for entry_id in self.EXCLUSION_GATES
        }
        assert set(stamps.values()) == {CLOCK.isoformat()}, stamps

    def test_an_expired_register_fails_closed(self) -> None:
        # The committed input-mass register carries no entry since the
        # 2026-09-21 retirements, so the expiry discipline is exercised on an
        # injected receipt that lapses before the clock below; the other two
        # registers still expire from their committed entries.
        lapsed = {
            "efrs-post-calibration": {
                "owned_land": {
                    "reason": "injected: a reviewed exclusion whose window has closed",
                    "approved_by": "juaristi22",
                    "adjudication": "microcosm#714",
                    "approved_on": "2026-08-26",
                    "expires_on": "2026-09-26",
                }
            }
        }
        battery = _run_battery(
            _tables(),
            parity=_parity(),
            fit_records=(FitWeightRecord("spi_qrf", "importance"),),
            clock=date(2027, 3, 1),
            extra_artifacts={"reviewed_input_mass_exclusions": lapsed},
        )
        failed = {o.entry.id for o in battery.outcomes if o.status is GateStatus.FAILED}
        assert {
            "uk_degenerate_release_surface",
            "uk_input_mass_parity",
        } <= failed

    def test_review_override_is_loud_in_the_evidence_payload(self) -> None:
        binding = UK_GATE_REGISTRY["degenerate_release_surface"]
        committed = binding.evidence_payload(
            EvidenceContext(artifacts={"exclusions_evaluated_on": CLOCK}), {}
        )
        assert committed["exclusions_policy"] == "committed"
        assert "household.source_year" in committed["reviewed_exclusions"]

        overridden = binding.evidence_payload(
            EvidenceContext(
                artifacts={
                    "exclusions_evaluated_on": CLOCK,
                    "reviewed_degenerate_exclusions": {},
                }
            ),
            {},
        )
        assert overridden["exclusions_policy"] == "override"
        assert overridden["reviewed_exclusions"] == {}
        assert overridden != committed, "an override must move the evidence digest"

    def test_resupplying_the_committed_register_is_not_an_override(self) -> None:
        # The label follows content, not the artifact's presence: a caller
        # routing the committed register through the artifact (as a driver
        # preflight might) runs the committed policy and must say so — and
        # a review file byte-identical to the register is no deviation.
        from microcosm.build.uk_runtime.terminal_gates import (
            uk_default_degenerate_reviewed_exclusions,
        )

        binding = UK_GATE_REGISTRY["degenerate_release_surface"]
        committed = binding.evidence_payload(
            EvidenceContext(artifacts={"exclusions_evaluated_on": CLOCK}), {}
        )
        resupplied = binding.evidence_payload(
            EvidenceContext(
                artifacts={
                    "exclusions_evaluated_on": CLOCK,
                    "reviewed_degenerate_exclusions": dict(
                        uk_default_degenerate_reviewed_exclusions()
                    ),
                }
            ),
            {},
        )
        assert resupplied == committed
        assert resupplied["exclusions_policy"] == "committed"

    @pytest.mark.parametrize(
        ("line", "stands_aside"),
        # Where no line is declared a scoped entry stands aside (fails closed).
        [("local", True), ("national", False), (None, True)],
    )
    def test_a_line_scoped_entry_stands_aside_off_its_line(
        self, line, stands_aside
    ) -> None:
        # employment_income carries signal in the fixture, so an exclusion for
        # it is stale wherever it applies.
        scoped = {
            "person.employment_income": {
                "reason": "injected: earned on the national line's evidence only",
                "approved_by": "juaristi22",
                "adjudication": "microcosm#1124",
                "approved_on": "2026-08-26",
                "expires_on": "2027-02-26",
                "release_lines": ["national"],
            }
        }
        extra = {"reviewed_degenerate_exclusions": scoped}
        if line is not None:
            extra["release_line"] = line
        battery = _run_battery(
            _tables(),
            parity=_parity(),
            fit_records=(FitWeightRecord("spi_qrf", "importance"),),
            extra_artifacts=extra,
        )
        outcome = {o.entry.id: o for o in battery.outcomes}[
            "uk_degenerate_release_surface"
        ]
        details = outcome.result.details
        stale = "person.employment_income" in details["stale_exclusions"]
        assert stale is not stands_aside
        if stands_aside:
            assert details["release_line"] == line
            assert details["out_of_line_exclusions"] == ["person.employment_income"]
            assert outcome.status is GateStatus.PASSED
        else:
            assert "out_of_line_exclusions" not in details
            assert outcome.status is GateStatus.FAILED

    def test_an_unknown_release_line_is_refused(self) -> None:
        binding = UK_GATE_REGISTRY["degenerate_release_surface"]
        context = EvidenceContext(
            artifacts={"exclusions_evaluated_on": CLOCK, "release_line": "regional"}
        )
        with pytest.raises(ValueError, match="release_line must be one of"):
            binding.evaluate(
                context,
                {"reviewed_exclusions_resource": "degenerate_reviewed_exclusions.json"},
            )

    def test_a_datetime_clock_is_refused(self, uk_gates) -> None:
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        entry = {e.id: e for e in uk_gates.gates}["uk_degenerate_release_surface"]
        binding = UK_GATE_REGISTRY["degenerate_release_surface"]
        result = _evaluate_gate(
            "degenerate_release_surface",
            lambda: binding.evaluate(
                EvidenceContext(
                    frame=frame,
                    artifacts={"exclusions_evaluated_on": datetime(2026, 9, 1, 12, 0)},
                ),
                entry.parameters,
            ),
        )
        assert result.passed is False
        assert "shared clock" in result.details["evaluation_error"]["message"]


class _TerminalCoverageEngine:
    """Minimal engine surface the coverage gate consults."""

    def default_values(self, names):
        return {name: 0.0 for name in names}

    def variables(self):
        return ["employment_income"]

    def variable_entities(self, names):
        return {name: "person" for name in names}


class TestTerminalCoverageBinding:
    def test_terminal_mode_threads_frame_engine_and_manifest(self) -> None:
        # The differential test feeds fixture coverage to both sides (the
        # real gate needs the committed licensed manifest), so this is the
        # one place the binding's terminal branch runs the real gate: same
        # frame surface, engine, and manifest as a direct call, re-minted
        # onto the declared neutral name.
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        engine = _TerminalCoverageEngine()
        manifest = UKReleaseInputCoverageManifest(
            reference={"source": "test"},
            candidate_evidence={"source": "test"},
            columns=(UKReleaseInputColumn("employment_income", "required"),),
            family_coverage={},
        )
        binding = UK_GATE_REGISTRY["release_input_coverage"]
        # The binding needs the spine build state (microcosm#1115 review);
        # here the spine frame is the frame itself.
        result = binding.evaluate(
            EvidenceContext(
                frame=frame,
                artifacts={
                    "coverage_engine": engine,
                    "coverage_manifest": manifest,
                    "spine_frame": frame,
                },
            ),
            {},
        )
        direct = uk_release_input_coverage_gate(
            _uk_gate_surface(frame),
            engine,
            manifest=manifest,
            build_state_frame=_uk_gate_surface(frame),
        )

        assert direct.name == "uk_release_input_coverage"
        assert result.name == "release_input_coverage"
        assert direct.passed is True
        assert result.passed == direct.passed
        assert result.failures == direct.failures
        assert dict(result.details) == dict(direct.details)

    def test_terminal_mode_reads_the_frame_the_release_writes(self) -> None:
        # microcosm#1106: the full build's terminal gates hold the raw
        # calibrated frame, which still carries the WAS property_wealth total
        # the release export drops. A reviewed exclusion that the raw frame
        # still populates reads as stale; the binding reads the export frame.
        person, benunit, household = _tables()
        household["main_residence_value"] = [300_000.0, 0.0, 150_000.0, 0.0]
        household["property_wealth"] = [350_000.0, 0.0, 160_000.0, 0.0]
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )

        class Engine(_TerminalCoverageEngine):
            def variables(self):
                return [
                    "employment_income",
                    "main_residence_value",
                    "property_wealth",
                ]

            def variable_entities(self, names):
                return {
                    name: "person" if name == "employment_income" else "household"
                    for name in names
                }

        engine = Engine()
        manifest = UKReleaseInputCoverageManifest(
            reference={"source": "test"},
            candidate_evidence={"source": "test"},
            columns=(
                UKReleaseInputColumn("employment_income", "required"),
                UKReleaseInputColumn("main_residence_value", "required"),
                UKReleaseInputColumn(
                    "property_wealth",
                    "reviewed_exclusion",
                    reason="formula-owned in policyengine-uk",
                    tracking_note="microcosm#1106",
                ),
            ),
            family_coverage={},
        )
        raw = uk_release_input_coverage_gate(
            _uk_gate_surface(frame), engine, manifest=manifest
        )
        result = UK_GATE_REGISTRY["release_input_coverage"].evaluate(
            EvidenceContext(
                frame=frame,
                artifacts={
                    "coverage_engine": engine,
                    "coverage_manifest": manifest,
                    # The spine build state the family half reads (#1115).
                    "spine_frame": frame,
                },
            ),
            {},
        )

        assert raw.passed is False
        assert any("property_wealth" in failure for failure in raw.failures)
        assert result.passed is True, result.failures


class TestPreflightBindings:
    def test_preflight_passes_with_the_committed_manifest(self, uk_gates) -> None:
        manifest = load_uk_release_input_coverage_manifest()
        phase = evaluate_phase(
            uk_gates,
            "preflight",
            EvidenceContext(
                artifacts={
                    "coverage_engine": None,
                    "build_stage_names": tuple(sorted(manifest.required_build_stages)),
                }
            ),
            registry=UK_GATE_REGISTRY,
        )
        statuses = {o.entry.id: o.status for o in phase.outcomes}
        assert statuses == {
            "uk_release_input_coverage_manifest_current": GateStatus.PASSED,
            "uk_release_family_build_stages": GateStatus.PASSED,
            "uk_ledger_compile_parity_incumbent_2025": GateStatus.EVIDENCE_ABSENT,
            "uk_ledger_compile_parity_local_incumbent_2025": (
                GateStatus.EVIDENCE_ABSENT
            ),
            "uk_target_surface_local_default_2025": GateStatus.EVIDENCE_ABSENT,
        }

    def test_ledger_compile_parity_selects_the_declared_period_registry(self) -> None:
        registry_2023 = TargetRegistry(
            (
                TargetSpec(
                    name="target",
                    entity="household",
                    measure="measure",
                    value=1.0,
                    period=2023,
                    source="synthetic",
                ),
            ),
            country="uk",
        )
        registry_2025 = TargetRegistry(
            (
                TargetSpec(
                    name="target",
                    entity="household",
                    measure="measure",
                    value=2.0,
                    period=2025,
                    source="synthetic",
                ),
            ),
            country="uk",
        )
        context = EvidenceContext(
            artifacts={
                "uk_ledger_compiled_registries": {
                    2023: registry_2023,
                    2025: registry_2025,
                }
            }
        )

        assert _ledger_compile_parity_registry(context, 2023) is registry_2023
        assert _ledger_compile_parity_registry(context, 2025) is registry_2025

    def test_ledger_compile_parity_can_select_the_local_registry_artifact(
        self,
    ) -> None:
        national = TargetRegistry(
            (
                TargetSpec(
                    name="target",
                    entity="household",
                    measure="measure",
                    value=1.0,
                    period=2025,
                    source="synthetic",
                ),
            ),
            country="uk",
        )
        local = TargetRegistry(
            (
                TargetSpec(
                    name="local_target",
                    entity="household",
                    measure="measure",
                    value=2.0,
                    period=2025,
                    source="synthetic",
                ),
            ),
            country="uk",
        )
        context = EvidenceContext(
            artifacts={
                "uk_ledger_compiled_registries": {2025: national},
                "uk_ledger_compiled_local_registries": {2025: local},
            }
        )

        assert _ledger_compile_parity_registry(context, 2025) is national
        assert (
            _ledger_compile_parity_registry(
                context,
                2025,
                registry_artifact="uk_ledger_compiled_local_registries",
            )
            is local
        )

    def test_ledger_compile_parity_requires_the_compiles_unsupported_report(
        self,
    ) -> None:
        # The registries alone are not the evidence. Without the compile's
        # report of unsupported references the gate is evidence_absent, never
        # a verdict that reads an unresolved reference as a fixture-only row.
        from microcosm.build.uk_runtime.battery_bindings import (
            _ledger_compile_parity_required_artifacts,
        )

        assert _ledger_compile_parity_required_artifacts({}) == {
            "uk_ledger_compiled_registries",
            "uk_ledger_compile_unsupported",
        }
        assert _ledger_compile_parity_required_artifacts(
            {"registry_artifact": "uk_ledger_compiled_local_registries"}
        ) == {
            "uk_ledger_compiled_local_registries",
            "uk_ledger_compile_local_unsupported",
        }

    def test_ledger_compile_parity_forwards_the_compiles_unsupported_report(
        self, monkeypatch
    ) -> None:
        from microcosm.build.uk_runtime import battery_bindings

        fixture = TargetRegistry(
            (
                TargetSpec(
                    name="ons.series",
                    entity="household",
                    measure="measure",
                    value=5.0,
                    period=2023,
                    source="synthetic",
                ),
            ),
            country="uk",
        )
        monkeypatch.setattr(
            battery_bindings,
            "_load_ledger_compile_parity_fixture",
            lambda resource: fixture,
        )
        parameters = {
            "fixture_resource": "synthetic.json",
            "target_period": 2023,
            "signed_differences": (
                {
                    "fixture_value": 5.0,
                    "kind": "ledger_absent",
                    "name": "ons.series",
                    "period": 2023,
                    "reason": "reviewed: the pinned feed no longer resolves it",
                },
            ),
        }
        unresolved = {"name": "ons.series", "period": 2023, "reason": "no fact"}

        def evaluate(unsupported):
            return battery_bindings._evaluate_ledger_compile_parity(
                EvidenceContext(
                    artifacts={
                        "uk_ledger_compiled_registries": {
                            2023: TargetRegistry((), country="uk")
                        },
                        "uk_ledger_compile_unsupported": unsupported,
                    }
                ),
                parameters,
            )

        # The compile reported the reference: the signed ledger_absent row
        # reproduces. A serialised surface keys its periods as strings.
        assert evaluate({2023: (unresolved,)}).passed
        assert evaluate({"2023": [unresolved]}).passed
        # A clean compile at that period contradicts the signed row.
        clean = evaluate({2023: ()})
        assert not clean.passed
        assert clean.failures == (
            "signed ledger parity difference ons.series[2023] changed kind: "
            "signed='ledger_absent', live='fixture_only'.",
        )
        # A period the producer compiled and did not report is refused.
        with pytest.raises(KeyError, match="no entry for target period 2023"):
            evaluate({2025: ()})
        with pytest.raises(TypeError, match="must be mappings that name"):
            evaluate({2023: ("ons.series",)})

    def test_ledger_compile_parity_evidence_attests_the_unsupported_report(
        self,
    ) -> None:
        # The report decides the verdict, so it is part of the attested
        # evidence: the same registry with and without it must not share a
        # digest. The names are sorted, so their order in the report is not.
        from microcosm.build.uk_runtime.battery_bindings import (
            _ledger_compile_parity_evidence,
        )

        parameters = {
            "fixture_resource": "registry_parity_fixture_2025.json",
            "target_period": 2025,
        }

        def evidence(unsupported):
            return _ledger_compile_parity_evidence(
                EvidenceContext(
                    artifacts={
                        "uk_ledger_compiled_registries": {
                            2025: TargetRegistry((), country="uk")
                        },
                        "uk_ledger_compile_unsupported": {2025: unsupported},
                    }
                ),
                parameters,
            )

        rows = (
            {"name": "ons.b", "period": 2025, "reason": "no fact"},
            {"name": "ons.a", "period": 2025, "reason": "no fact"},
        )
        assert evidence(rows)["unsupported_references"] == ["ons.a", "ons.b"]
        assert evidence(rows) == evidence(tuple(reversed(rows)))
        assert evidence(())["unsupported_references"] == []
        assert evidence(rows) != evidence(())

    def test_local_default_target_surface_uses_metric_names_and_signed_deferrals(
        self, uk_gates
    ) -> None:
        spec = load_country_spec("uk")
        registry = TargetRegistry(
            (
                TargetSpec(
                    name=reference.name,
                    entity=reference.entity,
                    measure=reference.measure,
                    value=1.0,
                    period=2025,
                    source="synthetic",
                )
                for reference in spec.local_target_references
            ),
            country="uk",
        )
        entry = {entry.id: entry for entry in uk_gates.gates}[
            "uk_target_surface_local_default_2025"
        ]
        result = UK_GATE_REGISTRY["target_surface"].evaluate(
            EvidenceContext(
                artifacts={"uk_ledger_compiled_local_registries": {2025: registry}}
            ),
            entry.parameters,
        )

        assert result.passed is True
        # microcosm#1240 adds 2,679 high-value homes cells (575 constituencies
        # and 318 authorities in England and Wales, three thresholds), six of
        # them deferred by the two micro-authorities' support floor.
        # microcosm#1207 adds 1,050 DESNZ energy cells (350 authorities in
        # Great Britain, three rows), 1,040 of them compiled: the support
        # floor defers five, Orkney and Shetland's gas-connected cells are
        # deferred, and the Isles of Scilly, Orkney and Shetland have no gas
        # kWh row. microcosm#1301 defers 2,323 of the high-value homes cells:
        # all 1,725 constituency cells, which the region rows now control,
        # and the 598 authority cells under the 400-home floor.
        # microcosm#1262 adds Scotland's 18 PIPR BRMA cells, all compiled.
        # microcosm#1095 adds England's single adult discount cells: 296
        # billing authorities, less the two the support floor defers.
        assert result.details["candidate_targets"] == 23_298
        assert result.details["reference_targets"] == 27_429
        # Only the 4,131 signed area deferrals and absences remain (2,100
        # before microcosm#929 bound the Welsh and Scottish council-tax cells;
        # Shetland band H is the one Scottish support deferral; microcosm#1123
        # adds the three higher rate taxpayer gaps; microcosm#1240 the six
        # high-value homes cells above; microcosm#1278 signs the two
        # small-cell rules, which defer 16 council-tax band cells and 192
        # higher-rate cells; microcosm#1207 the ten energy cells above;
        # microcosm#1301 the 2,323 high-value homes cells above; microcosm#1095
        # the two single adult discount cells above); the 1,011 household rows
        # are ordinary Chronicle-compiled references.
        exclusions = result.details["reviewed_exclusions"]
        assert len(exclusions) == 4_131
        households = [
            name for name in exclusions if str(name).startswith("households@")
        ]
        assert households == []
        assert result.details["missing_reference_targets"] == []

    def test_missing_required_stage_fails_with_the_assertion_text(
        self, uk_gates
    ) -> None:
        phase = evaluate_phase(
            uk_gates,
            "preflight",
            EvidenceContext(
                artifacts={"coverage_engine": None, "build_stage_names": ()}
            ),
            registry=UK_GATE_REGISTRY,
        )
        stages = {o.entry.id: o for o in phase.outcomes}[
            "uk_release_family_build_stages"
        ]
        assert stages.status is GateStatus.FAILED
        assert "omits required release family stage(s)" in stages.result.failures[0]


def test_area_support_binding_resolves_register_and_rejects_expired_entry(
    monkeypatch,
    uk_gates,
) -> None:
    import microcosm.build.uk_runtime.battery_bindings as battery_bindings

    entry = {entry.id: entry for entry in uk_gates.gates}["uk_local_area_support"]
    binding = UK_GATE_REGISTRY["area_support"]
    # The committed register carries the two micro local authorities Maria
    # excluded on the measured K=4 shortfalls (microcosm#762 A4); the
    # synthetic roster carries them below the floor, at made-up figures, so
    # the committed entries apply instead of reading as unknown.
    support = pd.DataFrame(
        {
            "geography_level": [
                "constituency",
                "local_authority",
                "local_authority",
                "local_authority",
            ],
            "area_code": ["E14000001", "E06000001", "E06000053", "E09000001"],
            "assigned_households": [50, 50, 30, 40],
            "nonzero_households": [50, 50, 30, 40],
            "effective_sample_size": [50.0, 50.0, 30.0, 40.0],
            "nonzero_source_households": [50, 50, 30, 40],
        }
    )
    monkeypatch.setattr(
        battery_bindings,
        "_local_area_roster",
        lambda _resource, _levels: {
            "constituency": ("E14000001",),
            "local_authority": ("E06000001", "E06000053", "E09000001"),
        },
    )
    context = EvidenceContext(
        artifacts={
            "uk_area_support_summary": support,
            "exclusions_evaluated_on": CLOCK,
        }
    )

    committed = binding.evaluate(context, entry.parameters)
    assert committed.passed is True
    assert set(committed.details["reviewed_exclusions"]) == {
        "local_authority/E06000053",
        "local_authority/E09000001",
    }
    assert committed.details["excluded_area_count"] == 2
    support_register = battery_bindings.load_uk_local_area_support_exclusion_register(
        None,
        resource="local_area_support_exclusions.json",
    )
    exclusions = support_register["exclusions"]
    a14_suffix = (
        " Per microcosm#762 A14 the authority's own local-authority cells are "
        "signed-deferred (`local_authority_support_floor_excluded`)."
    )
    # The exclusions are gate-only (microcosm#1301): only the areas whose cells
    # the reference generator defers on its own constants may claim A14.
    deferred = {f"local_authority/{area}" for area in SUPPORT_FLOOR_DEFERRED_AREA_IDS}
    for key, record in exclusions.items():
        assert record.reason.endswith(a14_suffix) is (key in deferred), key

    monkeypatch.setattr(
        battery_bindings,
        "load_uk_local_area_support_exclusion_register",
        lambda *_args, **_kwargs: {
            "exclusions": {
                "constituency/E14000001": UKReviewedExclusion(
                    reason="synthetic expired support review",
                    approved_by="reviewer",
                    adjudication="microcosm#762",
                    approved_on="2026-01-01",
                    expires_on="2026-02-01",
                )
            },
            "bound_despite_support_floor": {},
        },
    )
    expired = binding.evaluate(context, entry.parameters)
    assert expired.passed is False
    assert expired.details["invalid_reviewed_exclusions"] == [
        "exclusions/constituency/E14000001"
    ]
    assert "outside its approval window" in expired.failures[0]


class TestParameterVocabulary:
    def test_every_declared_uk_parameter_is_inside_its_binding_vocabulary(
        self, uk_gates
    ) -> None:
        # The whole shipped spec arms against the shipped registry: a
        # parameter either routes into its gate or the battery refuses to
        # start. Guards the vocabulary against drifting behind gates.json.
        validate_gate_parameters(uk_gates, UK_GATE_REGISTRY)

    def test_a_stray_preflight_parameter_is_refused_at_arm_time(self) -> None:
        # The preflight coverage evaluator reads `check` selectively rather
        # than splatting, so before vocabulary validation an extra key here
        # was the one place a declared parameter could ship inside
        # policy_sha256 while governing nothing.
        manifest = GatesManifest.from_mapping(
            {
                "version": 1,
                "country": "uk",
                "policy": "test battery",
                "phases": ["preflight"],
                "gates": [
                    {
                        "id": "uk_manifest_current",
                        "gate": "release_input_coverage",
                        "phase": "preflight",
                        "criticality": "release_blocking",
                        "parameters": {
                            "check": "manifest_current",
                            "cheks": "manifest_current",  # typo'd on purpose
                        },
                    }
                ],
            }
        )
        with pytest.raises(ValueError, match=r"'uk_manifest_current'.*cheks"):
            evaluate_phase(
                manifest,
                "preflight",
                EvidenceContext(artifacts={"coverage_engine": None}),
                registry=UK_GATE_REGISTRY,
            )


class TestBindingUnits:
    def test_only_the_declared_legacy_name_is_reminted(self) -> None:
        binding = UKGateBinding(
            name="release_input_coverage",
            evaluator=lambda context, parameters: GateResult(
                name="uk_release_input_coverage", passed=True, details={}
            ),
            legacy_name="uk_release_input_coverage",
        )
        result = binding.evaluate(EvidenceContext(), {})
        assert result.name == "release_input_coverage"

        impostor = UKGateBinding(
            name="release_input_coverage",
            evaluator=lambda context, parameters: GateResult(
                name="weight_ess", passed=True, details={}
            ),
            legacy_name="uk_release_input_coverage",
        )
        checked = _evaluate_gate(
            "release_input_coverage",
            lambda: impostor.evaluate(EvidenceContext(), {}),
        )
        assert checked.passed is False
        assert checked.details["returned_gate"] == "weight_ess"

    def test_frozen_spec_parameters_construct_reviewed_strata(self, uk_gates) -> None:
        # The declared parameters arrive frozen (mappings as proxies, lists
        # as tuples); the binding must build the reviewed declarations from
        # exactly that shape.
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        entry = {e.id: e for e in uk_gates.gates}["uk_zero_weight_strata"]
        binding = UK_GATE_REGISTRY["zero_weight_strata"]
        result = binding.evaluate(EvidenceContext(frame=frame), entry.parameters)
        assert result.passed is True

    def test_unknown_declaration_key_fails_closed(self, uk_gates) -> None:
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        entry = {e.id: e for e in uk_gates.gates}["uk_zero_weight_strata"]
        seeded = {
            "declarations": [
                {**dict(declaration), "surprise": 1}
                for declaration in entry.parameters["declarations"]
            ]
        }
        binding = UK_GATE_REGISTRY["zero_weight_strata"]
        result = _evaluate_gate(
            "zero_weight_strata",
            lambda: binding.evaluate(EvidenceContext(frame=frame), seeded),
        )
        assert result.passed is False
        assert "unknown keys ['surprise']" in result.failures[0]

    def test_unknown_declared_parameter_fails_closed(self, uk_gates) -> None:
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        binding = UK_GATE_REGISTRY["weight_ess"]
        result = _evaluate_gate(
            "weight_ess",
            lambda: binding.evaluate(
                EvidenceContext(frame=frame),
                {"minimum_ess_fraction": 0.01, "unreviewed_knob": 1},
            ),
        )
        assert result.passed is False
        assert result.details["evaluation_error"]["type"] == "TypeError"

    def test_drifted_declared_pin_fails_closed(self, uk_gates) -> None:
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        entry = {e.id: e for e in uk_gates.gates}["uk_input_mass_parity"]
        drifted = dict(entry.parameters)
        drifted["reference_registry"] = {
            name: (
                {
                    **payload,
                    "totals_sha256": "0" * 64,
                }
                if name == "efrs-post-calibration"
                else payload
            )
            for name, payload in drifted["reference_registry"].items()
        }
        binding = UK_GATE_REGISTRY["input_mass_parity"]
        result = _evaluate_gate(
            "input_mass_parity",
            lambda: binding.evaluate(
                EvidenceContext(
                    frame=frame,
                    artifacts={
                        "input_mass_reference": _reference(),
                        "exclusions_evaluated_on": CLOCK,
                    },
                ),
                drifted,
            ),
        )
        assert result.passed is False
        assert "reference_registry" in result.failures[0]


@pytest.mark.parametrize(
    "gate_id,weights,relative_error",
    [
        ("uk_local_target_fit", [1.0] * 200, 0.30),
        ("uk_local_per_family_fit", [1.0] * 200, 0.30),
        ("uk_local_weight_ratio", [1000.0] + [1.0] * 199, 0.0),
        ("uk_local_weight_ess", [1e9] + [1.0] * 199, 0.0),
    ],
)
def test_measured_local_quality_failure_blocks_release(
    gate_id, weights, relative_error, tmp_path
):
    """Use real program errors and weights, then exercise battery enforcement."""
    from microcosm.build.uk_runtime.calibration_run import uk_scoped_gate_manifest

    manifest = uk_scoped_gate_manifest(
        frozenset({gate_id}), phases=("terminal",), policy_suffix="local_candidate"
    )
    person, benunit, household = _tables(n=len(weights), weights=weights)
    frame = uk_national_frame(
        person=person, benunit=benunit, household=household, time_period="2025"
    )
    diagnostics = pd.DataFrame(
        [
            {
                "family": "obr",
                "area_code": "UK",
                "metric": f"program_{i}",
                "relative_error": relative_error,
            }
            for i in range(5)
        ]
    )
    report_path = tmp_path / "gates.json"
    battery = GateBatteryRun(
        manifest,
        release_id="synthetic-quality-refusal",
        report_path=report_path,
        release_candidate=True,
        registry=UK_GATE_REGISTRY,
    )
    battery.run_phase(
        "terminal",
        EvidenceContext(
            frame=frame,
            artifacts={
                "local_target_diagnostics": diagnostics,
                "exclusions_evaluated_on": CLOCK,
            },
        ),
    )
    with pytest.raises(GateBatteryBlockedError):
        battery.enforce("terminal", mode=BlockingMode.BLOCKS_ARTIFACT)
    report = battery.report_payload()
    assert report["shippable"] is False
    assert report["gates"][gate_id]["status"] == "failed"
    assert report["gates"][gate_id]["criticality"] == "release_blocking"


class TestEnumDomainResolution:
    """The enum gates resolve their domain from whichever engine is armed."""

    def _frame_with_plans(self):
        import enum

        person, benunit, household = _tables()
        plans = ["NONE", "PLAN_2"] * len(person)
        person["student_loan_plan"] = plans[: len(person)]
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )

        class StudentLoanPlan(enum.Enum):
            NONE = "NONE"
            PLAN_2 = "PLAN_2"

        return frame, StudentLoanPlan

    def test_public_accessor_is_the_contract(self) -> None:
        from microcosm.build.uk_runtime.battery_bindings import _evaluate_enum_domain

        frame, domain = self._frame_with_plans()

        class Accessor:
            def enum_domain(self, column):
                assert column == "student_loan_plan"
                return domain

        result = _evaluate_enum_domain(
            EvidenceContext(frame=frame, artifacts={"rules_engine": Accessor()}),
            {"columns": ["student_loan_plan"]},
        )
        assert result.passed

    def test_private_variable_lookup_still_resolves(self) -> None:
        from types import SimpleNamespace

        from microcosm.build.uk_runtime.battery_bindings import _evaluate_enum_domain

        frame, domain = self._frame_with_plans()

        class Legacy:
            def _variable(self, column):
                return SimpleNamespace(possible_values=domain)

        result = _evaluate_enum_domain(
            EvidenceContext(frame=frame, artifacts={"rules_engine": Legacy()}),
            {"columns": ["student_loan_plan"]},
        )
        assert result.passed

    def test_an_off_domain_brma_fails_the_local_recheck_as_a_structural_stop(
        self,
    ) -> None:
        """microcosm#1305 review: the unstubbed binding, then the classification.

        The area-BRMA rewrite is rechecked by ``uk_local_brma_enum_domain``;
        a value outside the engine's enum fails it, and the failure stops even
        a development build (no candidate is exported).
        """

        import enum

        from microcosm.build.gate_battery import GateOutcome, GatePhaseReport
        from microcosm.build.uk_runtime import full_gates
        from microcosm.build.uk_runtime.battery_bindings import _evaluate_enum_domain

        person, benunit, household = _tables()
        household["brma"] = ["LOTHIAN", "NOT_A_BRMA", "CARDIFF", "LOTHIAN"][
            : len(household)
        ]
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )

        class BRMAName(enum.Enum):
            CARDIFF = "CARDIFF"
            LOTHIAN = "LOTHIAN"

        class Accessor:
            def enum_domain(self, column):
                assert column == "brma"
                return BRMAName

        entry = next(
            gate
            for gate in load_country_spec("uk").gates.gates
            if gate.id == "uk_local_brma_enum_domain"
        )
        result = _evaluate_enum_domain(
            EvidenceContext(frame=frame, artifacts={"rules_engine": Accessor()}),
            entry.parameters,
        )
        assert not result.passed
        report = GatePhaseReport(
            entry.phase,
            (
                GateOutcome(
                    entry=entry, status=GateStatus.FAILED, result=result, reason=None
                ),
            ),
        )
        for fraction in (0.1, 1.0):
            outcome = full_gates.classify_full_gate_outcomes(
                report, sample_fraction=fraction, release_candidate=False
            )
            assert outcome["artifact_permitted"] is False
            assert outcome["structural_failures"] == ["uk_local_brma_enum_domain"]

    def test_an_engine_with_neither_names_itself(self) -> None:
        from microcosm.build.uk_runtime.battery_bindings import _evaluate_enum_domain

        frame, _domain = self._frame_with_plans()

        class Bare:
            pass

        # The first release-cut run failed both enum gates with an
        # AttributeError from inside the evaluator; the refusal now names the
        # adapter and the missing accessor.
        with pytest.raises(ValueError, match="Bare.*neither enum_domain nor _variable"):
            _evaluate_enum_domain(
                EvidenceContext(frame=frame, artifacts={"rules_engine": Bare()}),
                {"columns": ["student_loan_plan"]},
            )


def test_the_area_support_flag_reads_the_declared_floors() -> None:
    from microcosm.build.uk_runtime.battery_bindings import (
        uk_area_support_below_floor,
        uk_area_support_floors,
    )

    entry = next(
        entry
        for entry in load_country_spec("uk").gates.gates
        if entry.id == "uk_local_area_support"
    )
    floors = uk_area_support_floors()
    assert floors == {key: entry.parameters[key] for key in floors}
    support = pd.DataFrame(
        {
            "geography_level": ["local_authority", "local_authority"],
            "area_code": ["A", "B"],
            "nonzero_households": [floors["minimum_rows"], floors["minimum_rows"] - 1],
            "effective_sample_size": [floors["minimum_effective_sample_size"]] * 2,
            "nonzero_source_households": [floors["minimum_distinct_sources"]] * 2,
        }
    )
    assert uk_area_support_below_floor(support).tolist() == [False, True]


class TestCouncilTaxBandDomain:
    """Each household's band must be one its nation uses (microcosm#1278)."""

    _REGIONS = ["NORTH_EAST", "WALES", "SCOTLAND", "NORTHERN_IRELAND"]

    @staticmethod
    def _parameters() -> dict:
        entry = next(
            entry
            for entry in load_country_spec("uk").gates.gates
            if entry.id == "uk_council_tax_band_domain"
        )
        return dict(entry.parameters)

    def _context(self, bands, regions=None) -> EvidenceContext:
        person, benunit, household = _tables(n=4)
        household["region"] = regions or self._REGIONS
        household["council_tax_band"] = bands
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )

        class Accessor:
            def enum_domain(self, column):
                assert column == "council_tax_band"
                return tuple("ABCDEFGHI")

        return EvidenceContext(frame=frame, artifacts={"rules_engine": Accessor()})

    def test_the_declaration_is_the_shared_constant(self) -> None:
        from microcosm.build.uk_runtime.council_tax_constants import (
            UK_COUNCIL_TAX_BANDS_BY_NATION,
        )

        parameters = self._parameters()
        assert parameters["partition_column"] == "region"
        assert {
            nation: tuple(bands)
            for nation, bands in parameters["domains_by_nation"].items()
        } == dict(UK_COUNCIL_TAX_BANDS_BY_NATION)

    def test_the_release_frame_instance_declares_the_same_check(self) -> None:
        # The spine entry stops at the transferred boundary; the terminal one
        # checks the frame the release ships, after any later step that writes
        # the band (microcosm#1278 review). The release cut owns it.
        from microcosm.build.uk_runtime.calibration_run import (
            UK_NATIONAL_GATE_SCOPE,
            UK_SPINE_GATE_SCOPE,
        )

        entries = {entry.id: entry for entry in load_country_spec("uk").gates.gates}
        spine = entries["uk_council_tax_band_domain"]
        terminal = entries["uk_council_tax_band_domain_terminal"]
        assert (spine.gate, spine.phase) == ("enum_domain", "transferred")
        assert (terminal.gate, terminal.phase) == ("enum_domain", "terminal")
        assert terminal.criticality == "release_blocking"
        assert dict(terminal.parameters) == dict(spine.parameters)
        assert spine.id in UK_SPINE_GATE_SCOPE
        assert terminal.id in UK_NATIONAL_GATE_SCOPE

    def test_each_nation_holds_its_own_bands(self) -> None:
        from microcosm.build.uk_runtime.battery_bindings import _evaluate_enum_domain

        result = _evaluate_enum_domain(
            self._context(["H", "I", "H", "D"]), self._parameters()
        )
        assert result.passed, result.failures

    @pytest.mark.parametrize(
        ("bands", "invalid"),
        [
            (["I", "I", "H", "D"], ["I@ENGLAND"]),
            (["H", "I", "I", "D"], ["I@SCOTLAND"]),
            (["H", "I", "H", "A"], ["A@NORTHERN_IRELAND"]),
        ],
    )
    def test_a_band_its_nation_does_not_use_fails(self, bands, invalid) -> None:
        from microcosm.build.uk_runtime.battery_bindings import _evaluate_enum_domain

        result = _evaluate_enum_domain(self._context(bands), self._parameters())
        assert not result.passed
        assert result.details["invalid_examples"] == {"council_tax_band": invalid}

    def test_a_declaration_other_than_the_constant_is_refused(self) -> None:
        from microcosm.build.uk_runtime.battery_bindings import _evaluate_enum_domain

        parameters = self._parameters()
        parameters["domains_by_nation"] = {
            **parameters["domains_by_nation"],
            "ENGLAND": list("ABCDEFGHI"),
        }
        with pytest.raises(ValueError, match="must equal the shared constant"):
            _evaluate_enum_domain(self._context(["H", "I", "H", "D"]), parameters)

    def test_a_region_without_a_nation_is_refused(self) -> None:
        from microcosm.build.uk_runtime.battery_bindings import _evaluate_enum_domain

        with pytest.raises(ValueError, match="regions with no nation"):
            _evaluate_enum_domain(
                self._context(
                    ["H", "I", "H", "D"],
                    regions=["UNKNOWN", "WALES", "SCOTLAND", "NORTHERN_IRELAND"],
                ),
                self._parameters(),
            )


def test_weight_gate_bindings_fold_support_families_before_evaluating() -> None:
    """The terminal weight gates read the support-family fold (microcosm#1045).

    Four rows: root 1 split into three copies at 1.0 each (ids 1, 2, 3 share
    the lineage key) and one household at 30.0. Row-level the ratio is 30;
    folded it is 10, so a bound of 20 passes only on the fold.
    """
    person, benunit, household = _tables(n=4, weights=[1.0, 1.0, 1.0, 30.0])
    household["household_is_capital_gains_clone"] = False
    household["source_household_id"] = np.asarray([1, 1, 1, 4], dtype=np.int64)
    household["household_support_channel"] = pd.array(["frs"] * 4, dtype="string")
    household["household_support_clone_index"] = np.zeros(4, dtype=np.int64)
    household["household_is_cgt_support_copy"] = [False, True, True, False]
    household["cgt_support_copy_index"] = np.asarray([0, 1, 2, 0], dtype=np.int64)
    household["household_is_cgt_residential_clone"] = False
    household["cgt_residential_clone_index"] = np.zeros(4, dtype=np.int64)
    frame = uk_national_frame(
        person=person, benunit=benunit, household=household, time_period="2023"
    )

    ratio = UK_GATE_REGISTRY["weight_ratio"].evaluate(
        EvidenceContext(frame=frame), {"maximum_max_to_median_ratio": 20.0}
    )
    assert ratio.passed
    assert ratio.details["max_to_median_positive_weight"] == 30.0
    assert ratio.details["evaluated_on"] == "family_folded_weights"
    assert ratio.details["family_folded"]["basis"] == "support_family_fold"
    assert ratio.details["family_folded"]["families"] == 2
    assert ratio.details["family_folded"]["max_to_median_positive_weight"] == (
        30.0 / ((3.0 + 30.0) / 2)
    )

    ess = UK_GATE_REGISTRY["weight_ess"].evaluate(
        EvidenceContext(frame=frame), {"minimum_ess_fraction": 0.01}
    )
    assert ess.details["evaluated_on"] == "family_folded_weights"
    assert ess.details["family_folded"]["n_records"] == 2
    assert ess.details["n_records"] == 4


def _receipt(
    *,
    approved_on: str = "2026-10-01",
    expires_on: str = "2027-01-01",
    release_lines: tuple[str, ...] | None = ("local",),
) -> UKReviewedExclusion:
    return UKReviewedExclusion(
        reason="synthetic reviewed exclusion",
        approved_by="reviewer",
        adjudication="microcosm#1301",
        approved_on=approved_on,
        expires_on=expires_on,
        release_lines=release_lines,
    )


class TestLocalTargetFitRegister:
    """The local cells' own target-fit register (microcosm#1301).

    Synthetic cells and errors only: an entry lifts the fence for its cell,
    on the shared review date and on its release line, and the register
    cannot rot.
    """

    CELL = "council_tax/LA_A/band_c"

    @staticmethod
    def _diagnostics(error: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "family": "council_tax",
                    "area_code": area,
                    "metric": "band_c",
                    "relative_error": value,
                }
                for area, value in (("LA_A", error), ("LA_B", 0.01))
            ]
        )

    def _evaluate(
        self,
        monkeypatch,
        uk_gates,
        register: dict[str, UKReviewedExclusion],
        error: float,
        *,
        line: str | None = "local",
        parameters=None,
    ) -> GateResult:
        import microcosm.build.uk_runtime.battery_bindings as battery_bindings

        monkeypatch.setattr(
            battery_bindings,
            "uk_default_local_target_fit_reviewed_exclusions",
            lambda: MappingProxyType(register),
        )
        entry = {entry.id: entry for entry in uk_gates.gates}["uk_local_target_fit"]
        artifacts = {
            "local_target_diagnostics": self._diagnostics(error),
            "exclusions_evaluated_on": CLOCK,
        }
        if line is not None:
            artifacts["release_line"] = line
        return UK_GATE_REGISTRY["target_fit"].evaluate(
            EvidenceContext(artifacts=artifacts),
            entry.parameters if parameters is None else parameters,
        )

    def test_the_spec_names_the_local_register(self, uk_gates) -> None:
        from microcosm.build.uk_runtime.weighted_integrity import (
            UK_LOCAL_TARGET_FIT_EXCLUSION_REGISTER_RESOURCE,
        )

        entry = {entry.id: entry for entry in uk_gates.gates}["uk_local_target_fit"]
        assert entry.parameters["reviewed_exclusions_resource"] == (
            UK_LOCAL_TARGET_FIT_EXCLUSION_REGISTER_RESOURCE
        )
        assert UK_GATE_REGISTRY["target_fit"].required_artifacts(entry.parameters) == {
            "local_target_diagnostics",
            "exclusions_evaluated_on",
        }

    def test_an_unsigned_breach_fails(self, monkeypatch, uk_gates) -> None:
        result = self._evaluate(monkeypatch, uk_gates, {}, -0.30)
        assert result.passed is False
        assert result.details["failing_targets"] == {self.CELL: -0.30}
        assert result.details["exclusions_evaluated_on"] == CLOCK.isoformat()

    def test_an_in_force_entry_lifts_the_fence_for_its_cell(
        self, monkeypatch, uk_gates
    ) -> None:
        result = self._evaluate(monkeypatch, uk_gates, {self.CELL: _receipt()}, -0.30)
        assert result.passed is True
        assert result.details["failing_targets"] == {}
        assert set(result.details["reviewed_exclusions"]) == {self.CELL}
        assert result.details["exclusions_evaluated_on"] == CLOCK.isoformat()
        assert "out_of_line_exclusions" not in result.details

    def test_a_stale_entry_fails(self, monkeypatch, uk_gates) -> None:
        result = self._evaluate(monkeypatch, uk_gates, {self.CELL: _receipt()}, -0.10)
        assert result.passed is False
        assert result.details["stale_exclusions"] == [self.CELL]

    @pytest.mark.parametrize(
        "window,detail",
        [
            (("2026-01-01", "2026-02-01"), "expired_exclusions"),
            (("2026-12-01", "2027-03-01"), "premature_exclusions"),
        ],
    )
    def test_an_out_of_force_entry_fails_on_the_review_date(
        self, monkeypatch, uk_gates, window, detail
    ) -> None:
        approved_on, expires_on = window
        record = _receipt(approved_on=approved_on, expires_on=expires_on)
        result = self._evaluate(monkeypatch, uk_gates, {self.CELL: record}, -0.30)
        assert result.passed is False
        assert result.details[detail] == [self.CELL]
        assert result.details["failing_targets"] == {self.CELL: -0.30}

    def test_an_entry_scoped_to_the_national_line_stands_aside(
        self, monkeypatch, uk_gates
    ) -> None:
        record = _receipt(release_lines=("national",))
        result = self._evaluate(monkeypatch, uk_gates, {self.CELL: record}, -0.30)
        assert result.passed is False
        assert result.details["failing_targets"] == {self.CELL: -0.30}
        assert result.details["release_line"] == "local"
        assert result.details["out_of_line_exclusions"] == [self.CELL]

    def test_a_missing_review_date_is_evidence_absent(self, uk_gates) -> None:
        phase = evaluate_phase(
            uk_gates,
            "terminal",
            EvidenceContext(
                artifacts={"local_target_diagnostics": self._diagnostics(-0.30)}
            ),
            registry=UK_GATE_REGISTRY,
        )
        outcome = {o.entry.id: o for o in phase.outcomes}["uk_local_target_fit"]
        assert outcome.status is GateStatus.EVIDENCE_ABSENT
        assert outcome.reason == "missing evidence: exclusions_evaluated_on"

    def test_the_national_register_is_refused_on_the_local_surface(
        self, monkeypatch, uk_gates
    ) -> None:
        entry = {entry.id: entry for entry in uk_gates.gates}["uk_local_target_fit"]
        parameters = {
            **entry.parameters,
            "reviewed_exclusions_resource": "target_fit_reviewed_exclusions.json",
        }
        with pytest.raises(ValueError, match="local exclusion register"):
            self._evaluate(monkeypatch, uk_gates, {}, -0.30, parameters=parameters)


class TestThinRowsAreJudged:
    """The production local fit bindings on a thin BRMA row.

    María's ruling of 2026-10-26 dropped the withholding of microcosm#1262: a
    row resting on fewer than ten source households keeps its figures, and
    both fit gates judge it with every other row.
    """

    THIN = "private_rent/S33999999/rent/private_rent_brma"

    @staticmethod
    def _rows() -> list[dict]:
        rows = [
            {
                "family": "private_rent",
                "area_code": f"E0600000{index}",
                "metric": "rent/private_rent",
                "relative_error": error,
            }
            for index, error in enumerate((0.01, 0.02, 0.05, 0.30, 0.04, 0.03), 1)
        ]
        rows += [
            {
                "family": "private_rent",
                "area_code": "S33999999",
                "metric": "rent/private_rent_brma",
                "relative_error": -0.4,
            },
            {
                "family": "private_rent",
                "area_code": "S33999998",
                "metric": "rent/private_rent_brma",
                "relative_error": 0.02,
            },
        ]
        return rows

    def _evaluate(self, monkeypatch, uk_gates, gate_id, rows, register=None):
        import microcosm.build.uk_runtime.battery_bindings as battery_bindings

        monkeypatch.setattr(
            battery_bindings,
            "uk_default_local_target_fit_reviewed_exclusions",
            lambda: MappingProxyType(register or {}),
        )
        entry = {entry.id: entry for entry in uk_gates.gates}[gate_id]
        return UK_GATE_REGISTRY[entry.gate].evaluate(
            EvidenceContext(
                artifacts={
                    "local_target_diagnostics": rows,
                    "exclusions_evaluated_on": CLOCK,
                    "release_line": "local",
                }
            ),
            entry.parameters,
        )

    def test_the_target_fit_judges_the_thin_row(self, monkeypatch, uk_gates) -> None:
        """The thin row misses its target by 40% and fails beside the
        published row beyond the bound; nothing is counted as withheld."""

        result = self._evaluate(
            monkeypatch, uk_gates, "uk_local_target_fit", self._rows()
        )
        assert result.passed is False
        published = "private_rent/E06000004/rent/private_rent"
        assert result.details["failing_targets"] == {published: 0.30, self.THIN: -0.4}
        assert result.details["targets_checked"] == 8
        assert "withheld_targets" not in result.details

    def test_the_thin_rows_register_entry_excuses_it(
        self, monkeypatch, uk_gates
    ) -> None:
        result = self._evaluate(
            monkeypatch,
            uk_gates,
            "uk_local_target_fit",
            self._rows(),
            register={self.THIN: _receipt()},
        )
        assert self.THIN in result.details["reviewed_exclusions"]
        assert self.THIN not in result.details["failing_targets"]

    def test_family_fit_counts_every_row(self, monkeypatch, uk_gates) -> None:
        result = self._evaluate(
            monkeypatch, uk_gates, "uk_local_per_family_fit", self._rows()
        )
        assert result.details["family_sizes"] == {"private_rent": 8}
        assert result.details["family_within_shares"]["private_rent"] == pytest.approx(
            6 / 8
        )
        assert "withheld_targets" not in result.details


class TestGateOnlyAreaSupport:
    """Area-support exclusions excuse an area at the gate only (microcosm#1301).

    Synthetic areas and support figures: a local-authority or constituency
    key is honoured on its line, an entry scoped to another line stands aside
    unchecked, and the bound-despite-floor approvals stay window-checked.
    """

    ROSTER = {
        "constituency": ("PCON_A", "PCON_B"),
        "local_authority": ("LA_A", "LA_B", "LA_C"),
    }

    @staticmethod
    def _support() -> pd.DataFrame:
        # PCON_B, LA_B and LA_C sit below the floors; the rest clear them.
        return pd.DataFrame(
            {
                "geography_level": ["constituency"] * 2 + ["local_authority"] * 3,
                "area_code": ["PCON_A", "PCON_B", "LA_A", "LA_B", "LA_C"],
                "assigned_households": [80, 20, 80, 20, 60],
                "nonzero_households": [80, 20, 80, 20, 60],
                "effective_sample_size": [80.0, 20.0, 80.0, 20.0, 20.0],
                "nonzero_source_households": [80, 20, 80, 20, 60],
            }
        )

    def _evaluate(
        self,
        monkeypatch,
        uk_gates,
        exclusions: dict[str, UKReviewedExclusion],
        *,
        bound: dict[str, UKReviewedExclusion] | None = None,
    ) -> GateResult:
        import microcosm.build.uk_runtime.battery_bindings as battery_bindings

        monkeypatch.setattr(
            battery_bindings,
            "_local_area_roster",
            lambda _resource, _levels: self.ROSTER,
        )
        monkeypatch.setattr(
            battery_bindings,
            "load_uk_local_area_support_exclusion_register",
            lambda *_args, **_kwargs: {
                "exclusions": exclusions,
                "bound_despite_support_floor": bound or {},
            },
        )
        entry = {entry.id: entry for entry in uk_gates.gates}["uk_local_area_support"]
        context = EvidenceContext(
            artifacts={
                "uk_area_support_summary": self._support(),
                "exclusions_evaluated_on": CLOCK,
                "release_line": "local",
            }
        )
        return UK_GATE_REGISTRY["area_support"].evaluate(context, entry.parameters)

    def test_local_authority_and_constituency_keys_are_honoured(
        self, monkeypatch, uk_gates
    ) -> None:
        keys = ("constituency/PCON_B", "local_authority/LA_B", "local_authority/LA_C")
        result = self._evaluate(
            monkeypatch, uk_gates, {key: _receipt() for key in keys}
        )
        assert result.passed is True, result.failures
        assert set(result.details["reviewed_exclusions"]) == set(keys)
        assert result.details["excluded_area_count"] == 3
        assert "out_of_line_exclusions" not in result.details

    def test_an_entry_scoped_to_another_line_stands_aside(
        self, monkeypatch, uk_gates
    ) -> None:
        # Out of its window as well: an entry that stands aside is not checked.
        aside = _receipt(
            approved_on="2026-01-01",
            expires_on="2026-02-01",
            release_lines=("national",),
        )
        result = self._evaluate(
            monkeypatch,
            uk_gates,
            {
                "constituency/PCON_B": _receipt(),
                "local_authority/LA_B": aside,
                "local_authority/LA_C": _receipt(),
            },
        )
        assert result.passed is False
        assert "invalid_reviewed_exclusions" not in result.details
        assert {failure.split(":", 1)[0] for failure in result.failures} == {
            "local_authority/LA_B"
        }
        assert result.details["release_line"] == "local"
        assert result.details["out_of_line_exclusions"] == ["local_authority/LA_B"]

    def test_bound_despite_floor_approvals_are_window_checked_on_every_line(
        self, monkeypatch, uk_gates
    ) -> None:
        expired = _receipt(
            approved_on="2026-01-01", expires_on="2026-02-01", release_lines=None
        )
        aside = _receipt(release_lines=("national",))
        result = self._evaluate(
            monkeypatch,
            uk_gates,
            {"local_authority/LA_B": aside},
            bound={"census_households": expired},
        )
        assert result.passed is False
        assert result.details["invalid_reviewed_exclusions"] == [
            "bound_despite_support_floor/census_households"
        ]
        assert result.details["out_of_line_exclusions"] == ["local_authority/LA_B"]
