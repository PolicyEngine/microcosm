"""The UK consumer half of the gate battery (microcosm#611 increment 1).

The behaviour-preservation contract: ``evaluate_phase`` over ``uk/gates.json``
with ``UK_GATE_REGISTRY`` must reproduce the legacy ``uk_terminal_gate_report``
verdicts gate for gate over identical synthetic evidence — same ``passed``,
same failure lines, same details — with exactly two result names re-minted
onto the shared vocabulary. Where the two paths deliberately differ (the
legacy report *omits* unevidenced gates; the battery records them as
``evidence_absent`` and blocks release candidates only), the difference is
asserted here as a positive statement, not papered over.

Fixtures are synthetic throughout: no UKDS unit records, same discipline as
the legacy battery tests.
"""

from __future__ import annotations

import base64
from datetime import date, datetime
from types import SimpleNamespace
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
    _uk_gate_surface,
)
from microcosm.build.uk_runtime.national_frame import (
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
    UKInputMassParityPolicy,
    UKInputMassReference,
    UKQRFTailConcentrationPolicy,
    UKReleaseParityEvidence,
    uk_terminal_gate_report,
)
from microcosm.frame import engine_tables

KEY = base64.b64encode(b"\x07" * 32).decode("ascii")
RELEASE_ID = "populace-uk-2023-frs-k535080"
DIAGNOSTICS_SHA256 = "c" * 64
#: The shared exclusion-expiry clock, fixed inside the committed register's
#: validity window (approved 2026-08-10, expires 2027-02-10) so the suite
#: never drifts across an expiry boundary.
CLOCK = date(2026, 9, 1)

#: Neutral declared name -> the legacy result name the bindings re-mint.
LEGACY_NAMES = {
    "release_input_coverage": "uk_release_input_coverage",
    "tail_concentration": "qrf_tail_concentration",
}

VALIDATE_REFERENCE = (
    "microcosm.build.uk_runtime.weighted_integrity._validate_input_mass_reference"
)


@pytest.fixture(autouse=True)
def signing_env(monkeypatch) -> None:
    monkeypatch.setenv(gate_signing_key_env("uk"), KEY)


@pytest.fixture(scope="module")
def uk_gates():
    return load_country_spec("uk").gates


def _tables(*, n: int = 4, weights=None):
    if weights is None:
        weights = np.ones(n, dtype=float)
    household_ids = np.arange(1, n + 1, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": np.arange(101, 101 + n, dtype=np.int64),
            "person_household_id": household_ids,
            "person_benunit_id": np.arange(201, 201 + n, dtype=np.int64),
            "employment_income": np.arange(1, n + 1, dtype=float),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(201, 201 + n, dtype=np.int64)})
    household = pd.DataFrame(
        {
            "household_id": household_ids,
            "household_weight": np.asarray(weights, dtype=float),
            "household_is_spi_synthetic": np.arange(n) % 2 == 1,
            "household_is_capital_gains_clone": np.arange(n) % 4 >= 2,
        }
    )
    return person, benunit, household


def _coverage() -> GateResult:
    return GateResult(
        name="uk_release_input_coverage",
        passed=True,
        details={"fixture": True},
    )


def _parity(**overrides) -> UKReleaseParityEvidence:
    fields = {
        "candidate_columns": {"person.age"},
        "reference_columns": {"person.age"},
        "candidate_targets": {"ons/population"},
        "reference_targets": {"ons/population"},
        "target_relative_errors": {"ons/population": 0.01},
    }
    fields.update(overrides)
    return UKReleaseParityEvidence(**fields)


def _reference() -> UKInputMassReference:
    return UKInputMassReference(
        totals={"person.employment_income": 10.0},
        filename="enhanced_frs_2023_24.h5",
        revision="655dd07e4bb9c777b00dac044949611f1feb824f",
        sha256="584ae33d80ca0431254610a3f8254d132da73477d31966d6446282861ecae50d",
        vintage="2023_24",
    )


def _input_mass_policy() -> UKInputMassParityPolicy:
    return UKInputMassParityPolicy(relative_tolerance=0.5, minimum_reference_total=0.0)


def _qrf_policy() -> UKQRFTailConcentrationPolicy:
    return UKQRFTailConcentrationPolicy(
        top_k=1, max_top_share=0.5, min_nonzero_records=2
    )


def _fixture_coverage_registry():
    """The UK registry with the coverage gate fed by the same fixture the
    legacy tests inject, so both differential sides see identical coverage
    evidence (the real coverage gate has its own dedicated tests). The
    fixture mints the legacy name, so the re-minting path stays exercised."""

    return {
        **UK_GATE_REGISTRY,
        "release_input_coverage": UKGateBinding(
            name="release_input_coverage",
            evaluator=lambda context, parameters: _coverage(),
            parameter_keys=frozenset({"check"}),
            legacy_name="uk_release_input_coverage",
            needs_frame=False,
        ),
    }


def _run_both(tables, *, parity=None, fit_records=None, armed=True, clock=CLOCK):
    """Run the legacy battery and the declared battery over one evidence set.

    Both sides are built from the same tables and the same evidence objects
    in one place — evidence asymmetry between the sides would read as a
    false differential failure. That includes the exclusion-expiry clock:
    the legacy aggregator threads ``now`` and the battery threads the
    ``exclusions_evaluated_on`` artifact, both set to the same date here.
    """

    person, benunit, household = tables
    dataset = SimpleNamespace(person=person, benunit=benunit, household=household)
    frame = uk_national_frame(
        person=person, benunit=benunit, household=household, time_period="2023"
    )
    artifacts: dict[str, object] = {
        "coverage_engine": object(),
        "exclusions_evaluated_on": clock,
    }
    legacy_kwargs: dict[str, object] = {"now": clock}
    if fit_records is not None:
        artifacts["fit_weight_records"] = fit_records
        legacy_kwargs["fit_weight_records"] = fit_records
    if parity is not None:
        artifacts["parity_evidence"] = parity
        legacy_kwargs["parity_evidence"] = parity
    if armed:
        reference = _reference()
        input_mass_policy = _input_mass_policy()
        qrf_policy = _qrf_policy()
        artifacts["input_mass_reference"] = reference
        artifacts["input_mass_policy"] = input_mass_policy
        artifacts["qrf_tail_policy"] = qrf_policy
        legacy_kwargs["input_mass_reference"] = reference
        legacy_kwargs["input_mass_policy"] = input_mass_policy
        legacy_kwargs["qrf_tail_policy"] = qrf_policy
    # Small synthetic totals exercise battery behavior without disclosing
    # the licensed 131-column reference (same patch as the legacy tests);
    # the binding's declared-pin check compares spec to runtime constant and
    # needs no patching.
    with patch(VALIDATE_REFERENCE, return_value=None):
        legacy = uk_terminal_gate_report(
            dataset,
            object(),
            release_id=RELEASE_ID,
            calibration_diagnostics_sha256=DIAGNOSTICS_SHA256,
            input_coverage_evaluator=_coverage,
            **legacy_kwargs,
        )
        battery = evaluate_phase(
            load_country_spec("uk").gates,
            "terminal",
            EvidenceContext(frame=frame, artifacts=artifacts),
            registry=_fixture_coverage_registry(),
        )
    return legacy, battery


def _assert_identical_verdicts(legacy, battery) -> None:
    legacy_by_name = {result.name: result for result in legacy.results}
    evaluated = [
        outcome
        for outcome in battery.outcomes
        if outcome.status in (GateStatus.PASSED, GateStatus.FAILED)
    ]
    assert [LEGACY_NAMES.get(o.entry.gate, o.entry.gate) for o in evaluated] == [
        result.name for result in legacy.results
    ]
    for outcome in evaluated:
        legacy_result = legacy_by_name[
            LEGACY_NAMES.get(outcome.entry.gate, outcome.entry.gate)
        ]
        result = outcome.result
        assert result.name == outcome.entry.gate
        assert result.passed == legacy_result.passed, outcome.entry.id
        assert result.failures == legacy_result.failures, outcome.entry.id
        assert dict(result.details) == dict(legacy_result.details), outcome.entry.id


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
            "missing evidence: frame, exclusions_evaluated_on, "
            "input_mass_policy, input_mass_reference"
        )
        assert reasons["uk_degenerate_release_surface"] == (
            "missing evidence: frame, exclusions_evaluated_on"
        )


class TestDifferentialAgainstLegacyBattery:
    def test_fully_armed_battery_matches_gate_for_gate(self) -> None:
        legacy, battery = _run_both(
            _tables(),
            parity=_parity(),
            fit_records=(FitWeightRecord("spi_qrf", "importance"),),
        )

        _assert_identical_verdicts(legacy, battery)
        by_id = {o.entry.id: o for o in battery.outcomes}
        passed = [
            entry_id for entry_id, o in by_id.items() if o.status is GateStatus.PASSED
        ]
        assert len(passed) == 10
        # The armed QRF gate fails identically on both sides: the tiny
        # synthetic frame carries none of the declared QRF output columns.
        # Failure-text parity over a real failure, for free.
        qrf = by_id["uk_qrf_tail_concentration"]
        assert qrf.status is GateStatus.FAILED
        assert qrf.result.failures == (
            legacy.results[-1].failures  # qrf is the last legacy gate
        )

    def test_empty_fit_records_fail_identically(self) -> None:
        # Present-but-empty is not absent: a fit stage that ran and emitted
        # nothing is a failed audit on both sides, never a vacuous pass
        # (the shared binding alone would pass it; the UK override keeps
        # the legacy guard).
        legacy, battery = _run_both(_tables(), fit_records=())

        _assert_identical_verdicts(legacy, battery)
        audit = {o.entry.id: o for o in battery.outcomes}["uk_weights_audit"]
        assert audit.status is GateStatus.FAILED
        assert "an absent audit is not a passing audit" in (audit.result.failures[0])

    def test_seeded_defects_fail_identically(self) -> None:
        blown = _tables(weights=[1.0, 1.0, 1.0, 1.0e9])
        seeded_parity = _parity(
            candidate_columns={"person.age", "person.unreviewed_extra"},
            target_relative_errors={"ons/population": -0.40},
        )
        legacy, battery = _run_both(
            blown,
            parity=seeded_parity,
            fit_records=(FitWeightRecord("spi_qrf", "none"),),
        )

        _assert_identical_verdicts(legacy, battery)
        failed = {o.entry.id for o in battery.outcomes if o.status is GateStatus.FAILED}
        assert {
            "uk_weight_ratio",
            "uk_weights_audit",
            "uk_export_surface",
            "uk_target_fit",
        } <= failed


class TestUnevidencedArms:
    """The chartered semantic difference, stated as a positive assertion.

    The legacy report *omits* gates whose evidence is absent (sealed by its
    membership contract); the battery lists every declared entry and records
    the gap as ``evidence_absent`` with the missing keys named — blocking
    release candidates only. The A2 orchestration swap inherits exactly this
    delta."""

    def test_legacy_omits_where_the_battery_records_evidence_absent(
        self, uk_gates
    ) -> None:
        legacy, battery = _run_both(_tables(), armed=False)

        legacy_names = {result.name for result in legacy.results}
        assert legacy_names == {
            "uk_release_input_coverage",
            "degenerate_release_surface",
            "zero_weight_strata",
            "weight_ess",
            "weight_ratio",
        }
        _assert_identical_verdicts(legacy, battery)
        absent = {
            o.entry.id: o.reason
            for o in battery.outcomes
            if o.status is GateStatus.EVIDENCE_ABSENT
        }
        assert set(absent) == {
            "uk_weights_audit",
            "uk_export_surface",
            "uk_target_surface",
            "uk_target_fit",
            "uk_input_mass_parity",
            "uk_qrf_tail_concentration",
        }
        for reason in absent.values():
            assert reason.startswith("missing evidence: ")

        assert battery.blocking_outcomes(release_candidate=False) == ()
        blocked = {
            o.entry.id for o in battery.blocking_outcomes(release_candidate=True)
        }
        assert blocked == set(absent)

    def test_absent_but_required_fit_evidence_is_the_named_delta(self) -> None:
        # Legacy: a production fit stage without records is an explicit
        # failure. Battery: the absent artifact is a named evidence gap that
        # blocks release candidates. Same shipping decision, different
        # taxonomy — asserted so the A2 review can lean on it.
        person, benunit, household = _tables()
        dataset = SimpleNamespace(person=person, benunit=benunit, household=household)
        legacy = uk_terminal_gate_report(
            dataset,
            object(),
            release_id=RELEASE_ID,
            calibration_diagnostics_sha256=DIAGNOSTICS_SHA256,
            input_coverage_evaluator=_coverage,
            require_fit_weight_records=True,
        )
        legacy_audit = {r.name: r for r in legacy.results}["weights_audit"]
        assert legacy_audit.passed is False

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
        "uk_qrf_tail_concentration",
    )

    def test_every_exclusion_gate_shares_the_injected_clock(self) -> None:
        _legacy, battery = _run_both(
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

    def test_an_expired_register_behaves_identically_on_both_sides(self) -> None:
        # Past the committed register's expiry the exclusion is out of
        # force on both paths; whatever the verdict, it must be the same
        # verdict — the differential contract holds at every clock value.
        legacy, battery = _run_both(
            _tables(),
            parity=_parity(),
            fit_records=(FitWeightRecord("spi_qrf", "importance"),),
            clock=date(2027, 3, 1),
        )
        _assert_identical_verdicts(legacy, battery)

    def test_review_override_is_loud_in_the_evidence_payload(self) -> None:
        binding = UK_GATE_REGISTRY["degenerate_release_surface"]
        committed = binding.evidence_payload(
            EvidenceContext(artifacts={"exclusions_evaluated_on": CLOCK}), {}
        )
        assert committed["exclusions_register"] == "committed"
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
        assert overridden["exclusions_register"] == "override"
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
        assert resupplied["exclusions_register"] == "committed"

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
        result = binding.evaluate(
            EvidenceContext(
                frame=frame,
                artifacts={
                    "coverage_engine": engine,
                    "coverage_manifest": manifest,
                },
            ),
            {},
        )
        direct = uk_release_input_coverage_gate(
            _uk_gate_surface(frame), engine, manifest=manifest
        )

        assert direct.name == "uk_release_input_coverage"
        assert result.name == "release_input_coverage"
        assert direct.passed is True
        assert result.passed == direct.passed
        assert result.failures == direct.failures
        assert dict(result.details) == dict(direct.details)


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
        }

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
            name="tail_concentration",
            evaluator=lambda context, parameters: GateResult(
                name="qrf_tail_concentration", passed=True, details={}
            ),
            legacy_name="qrf_tail_concentration",
        )
        result = binding.evaluate(EvidenceContext(), {})
        assert result.name == "tail_concentration"

        impostor = UKGateBinding(
            name="tail_concentration",
            evaluator=lambda context, parameters: GateResult(
                name="weight_ess", passed=True, details={}
            ),
            legacy_name="qrf_tail_concentration",
        )
        checked = _evaluate_gate(
            "tail_concentration",
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
        drifted["reference_sha256"] = "0" * 64
        binding = UK_GATE_REGISTRY["input_mass_parity"]
        result = _evaluate_gate(
            "input_mass_parity",
            lambda: binding.evaluate(
                EvidenceContext(
                    frame=frame,
                    artifacts={
                        "input_mass_reference": _reference(),
                        "input_mass_policy": _input_mass_policy(),
                    },
                ),
                drifted,
            ),
        )
        assert result.passed is False
        assert "declared pin" in result.failures[0]
