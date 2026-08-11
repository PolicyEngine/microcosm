"""UK terminal-gate batching and seeded-defect coverage."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from microcosm.build.gates import FitWeightRecord, GateReport, GateResult
from microcosm.build.uk_runtime.terminal_gates import (
    UK_DEFAULT_ZERO_WEIGHT_STRATA,
    UK_MAX_TO_MEDIAN_WEIGHT_RATIO,
    UK_TERMINAL_GATE_PRODUCER,
    UK_TERMINAL_GATE_SIGNATURE_ALGORITHM,
    UK_TERMINAL_GATE_SIGNING_KEY_ENV,
    UKInputMassParityPolicy,
    UKInputMassReference,
    UKQRFTailConcentrationPolicy,
    UKReleaseParityEvidence,
    UKZeroWeightStratumDeclaration,
    uk_default_degenerate_reviewed_exclusions,
    uk_degenerate_release_surface_gate,
    uk_export_surface_gate,
    uk_target_fit_gate,
    uk_target_surface_gate,
    uk_terminal_gate_policy_sha256,
    uk_terminal_gate_report,
    uk_weight_ratio_gate,
    write_uk_terminal_gate_report,
)

TEST_UK_TERMINAL_GATE_SIGNING_KEY = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="


def _entry(reason: str, *, expires_on: str = "2027-02-10") -> dict[str, str]:
    """A valid schema-2 approval receipt around the fixture's reason."""

    return {
        "reason": reason,
        "approved_by": "test-reviewer",
        "adjudication": "microcosm#610",
        "approved_on": "2026-08-10",
        "expires_on": expires_on,
    }


TEST_UK_RELEASE_ID = "populace-uk-2023-frs-k535080"
TEST_UK_CALIBRATION_DIAGNOSTICS_SHA256 = "c" * 64


@pytest.fixture(autouse=True)
def _trusted_terminal_gate_signing_key(monkeypatch) -> None:
    monkeypatch.setenv(
        UK_TERMINAL_GATE_SIGNING_KEY_ENV,
        TEST_UK_TERMINAL_GATE_SIGNING_KEY,
    )


def _dataset(
    *,
    n: int = 4,
    weights: np.ndarray | list[float] | None = None,
    signal: object | None = None,
):
    if weights is None:
        weights = np.ones(n, dtype=float)
    values = np.arange(1, n + 1, dtype=float) if signal is None else signal
    if not isinstance(values, (list, tuple, np.ndarray, pd.Series)):
        values = [values] * n
    household_ids = np.arange(1, n + 1, dtype=np.int64)
    return SimpleNamespace(
        person=pd.DataFrame(
            {
                "person_id": np.arange(101, 101 + n, dtype=np.int64),
                "person_household_id": household_ids,
                "person_benunit_id": np.arange(201, 201 + n, dtype=np.int64),
                "employment_income": values,
            }
        ),
        benunit=pd.DataFrame({"benunit_id": np.arange(201, 201 + n, dtype=np.int64)}),
        household=pd.DataFrame(
            {
                "household_id": household_ids,
                "household_weight": np.asarray(weights, dtype=float),
                "household_is_spi_synthetic": np.arange(n) % 2 == 1,
                "household_is_capital_gains_clone": np.arange(n) % 4 >= 2,
            }
        ),
    )


def _coverage(*, passed: bool = True) -> GateResult:
    return GateResult(
        name="uk_release_input_coverage",
        passed=passed,
        failures=() if passed else ("seeded coverage defect",),
        details={"fixture": True},
    )


def _report(dataset=None, **kwargs):
    # Small synthetic totals exercise battery behavior without disclosing the
    # licensed 131-column reference; trust-anchor behavior has dedicated tests.
    with patch(
        "microcosm.build.uk_runtime.weighted_integrity._validate_input_mass_reference",
        return_value=None,
    ):
        return uk_terminal_gate_report(
            _dataset() if dataset is None else dataset,
            object(),
            release_id=TEST_UK_RELEASE_ID,
            calibration_diagnostics_sha256=(TEST_UK_CALIBRATION_DIAGNOSTICS_SHA256),
            input_coverage_evaluator=lambda: _coverage(),
            **kwargs,
        )


def _gates(report) -> dict[str, dict[str, object]]:
    return report.to_manifest()["gates"]


def test_healthy_synthetic_release_passes_the_mandatory_batch() -> None:
    report = _report()

    assert report.passed
    assert list(_gates(report)) == [
        "uk_release_input_coverage",
        "degenerate_release_surface",
        "zero_weight_strata",
        "weight_ess",
        "weight_ratio",
    ]


@pytest.mark.parametrize(
    ("signal", "detail_key"),
    [
        ([0.0] * 4, "all_zero_columns"),
        ([None] * 4, "all_null_columns"),
        ([7.0] * 4, "constant_columns"),
    ],
)
def test_each_degenerate_column_class_produces_its_named_finding(
    signal,
    detail_key,
) -> None:
    gate = _gates(_report(_dataset(signal=signal)))["degenerate_release_surface"]

    assert gate["passed"] is False
    assert gate["details"][detail_key] == ["person.employment_income"]


def test_reviewed_degenerate_exclusion_is_recorded_and_stale_entries_fail() -> None:
    reason = "Fixture intentionally broadcasts this reviewed input."
    live = uk_degenerate_release_surface_gate(
        _dataset(signal=7.0),
        reviewed_exclusions={"person.employment_income": _entry(reason)},
    )
    stale = uk_degenerate_release_surface_gate(
        _dataset(),
        reviewed_exclusions={"person.employment_income": _entry(reason)},
    )

    assert live.passed
    recorded = live.details["reviewed_exclusions"]["person.employment_income"]
    assert recorded["reason"] == reason
    assert recorded["approved_by"] == "test-reviewer"
    assert recorded["adjudication"] == "microcosm#610"
    assert recorded["expires_on"] == "2027-02-10"
    assert not stale.passed
    assert stale.details["stale_exclusions"] == ["person.employment_income"]


def test_undeclared_zero_weight_stratum_produces_named_finding() -> None:
    dataset = _dataset(weights=[0.0, 1.0, 1.0, 1.0])
    gate = _gates(_report(dataset))["zero_weight_strata"]

    assert gate["passed"] is False
    assert gate["details"]["unmatched_zero_weight_rows"] == 1
    assert "match no declared stratum" in gate["failures"][0]


def test_zero_weight_stratum_beyond_declaration_produces_named_finding() -> None:
    dataset = _dataset(weights=[0.0, 1.0, 1.0, 1.0])
    declaration = UKZeroWeightStratumDeclaration(
        name="fixture_base",
        selector={
            "household_is_spi_synthetic": False,
            "household_is_capital_gains_clone": False,
        },
        maximum_zero_weight_rows=0,
        reason="No zero rows are expected in the healthy fixture.",
    )

    gate = _gates(_report(dataset, zero_weight_declarations=(declaration,)))[
        "zero_weight_strata"
    ]

    assert gate["passed"] is False
    assert gate["details"]["declared_strata"][0]["zero_weight_rows"] == 1
    assert "exceed the declared maximum" in gate["failures"][0]


def test_missing_zero_weight_selector_columns_fail_even_with_positive_weights() -> None:
    dataset = _dataset()
    dataset.household.drop(
        columns=[
            "household_is_spi_synthetic",
            "household_is_capital_gains_clone",
        ],
        inplace=True,
    )

    gate = _gates(_report(dataset))["zero_weight_strata"]

    assert gate["passed"] is False
    assert all(
        row["missing_selector_columns"] for row in gate["details"]["declared_strata"]
    )
    assert "selector column(s) are missing" in gate["failures"][0]


def test_default_declarations_name_both_june_100k_zero_strata() -> None:
    assert [row.maximum_zero_weight_rows for row in UK_DEFAULT_ZERO_WEIGHT_STRATA] == [
        100_000,
        100_000,
    ]
    assert [row.selector for row in UK_DEFAULT_ZERO_WEIGHT_STRATA] == [
        {
            "household_is_capital_gains_clone": False,
            "household_is_spi_synthetic": True,
        },
        {
            "household_is_capital_gains_clone": True,
            "household_is_spi_synthetic": True,
        },
    ]


def test_ess_collapse_produces_named_finding() -> None:
    weights = np.ones(200, dtype=float)
    weights[0] = 10_000.0

    gate = _gates(_report(_dataset(n=200, weights=weights)))["weight_ess"]

    assert gate["passed"] is False
    assert gate["details"]["ess_fraction"] < 0.01
    assert "ESS fraction" in gate["failures"][0]


def test_ratio_blowout_produces_named_finding() -> None:
    report = _report(
        _dataset(weights=[1.0, 1.0, 1.0, 20.0]),
        maximum_max_to_median_ratio=10.0,
    )
    gate = _gates(report)["weight_ratio"]

    assert gate["passed"] is False
    assert gate["details"]["max_to_median_positive_weight"] == 20.0
    assert "Max/positive-median" in gate["failures"][0]


def test_ratio_boundary_is_pinned_to_the_certified_june_measurement() -> None:
    assert UK_MAX_TO_MEDIAN_WEIGHT_RATIO == 1_151.2542195939373


def test_certified_june_weight_ratio_passes_at_the_inclusive_boundary() -> None:
    # Ratio-preserving replay of the certified June H5 (sha256 f17306ccb2aad7ff
    # 0130be3589b560afb2e2a12a943570911cd0c77f07934833).  The gate observes
    # only the positive median and maximum, so the full 535,080-row vector is
    # trimmed to those real values plus a representative shipped zero.
    positive_median = 16.202157974243164
    maximum = 18_652.802734375
    gate = uk_weight_ratio_gate([0.0, positive_median, positive_median, maximum])

    assert gate.details["max_to_median_positive_weight"] == (
        UK_MAX_TO_MEDIAN_WEIGHT_RATIO
    )
    assert gate.details["maximum_max_to_median_ratio"] == (
        UK_MAX_TO_MEDIAN_WEIGHT_RATIO
    )
    assert gate.passed


def test_immediate_nextafter_weight_ratio_fails_with_distinct_full_precision() -> None:
    just_above = math.nextafter(UK_MAX_TO_MEDIAN_WEIGHT_RATIO, math.inf)

    gate = uk_weight_ratio_gate([0.0, 1.0, 1.0, just_above])

    assert just_above == 1_151.2542195939375
    assert gate.details["max_to_median_positive_weight"] == just_above
    assert not gate.passed
    assert repr(just_above) in gate.failures[0]
    assert repr(UK_MAX_TO_MEDIAN_WEIGHT_RATIO) in gate.failures[0]


def test_gate_evaluation_error_does_not_mask_later_findings() -> None:
    report = uk_terminal_gate_report(
        _dataset(signal=0.0),
        object(),
        release_id=TEST_UK_RELEASE_ID,
        calibration_diagnostics_sha256=(TEST_UK_CALIBRATION_DIAGNOSTICS_SHA256),
        input_coverage_evaluator=lambda: (_ for _ in ()).throw(
            RuntimeError("seeded coverage crash")
        ),
    )
    gates = _gates(report)

    assert list(gates) == [
        "uk_release_input_coverage",
        "degenerate_release_surface",
        "zero_weight_strata",
        "weight_ess",
        "weight_ratio",
    ]
    assert gates["uk_release_input_coverage"]["passed"] is False
    assert gates["degenerate_release_surface"]["passed"] is False
    assert gates["weight_ratio"]["passed"] is True


def test_malformed_release_surface_still_returns_the_complete_named_batch() -> None:
    dataset = _dataset()
    del dataset.household

    report = uk_terminal_gate_report(
        dataset,
        object(),
        release_id=TEST_UK_RELEASE_ID,
        calibration_diagnostics_sha256=(TEST_UK_CALIBRATION_DIAGNOSTICS_SHA256),
        input_coverage_evaluator=lambda: _coverage(passed=False),
    )
    gates = _gates(report)

    assert list(gates) == [
        "uk_release_input_coverage",
        "degenerate_release_surface",
        "zero_weight_strata",
        "weight_ess",
        "weight_ratio",
    ]
    assert all(not gate["passed"] for gate in gates.values())
    assert "DataFrames" in gates["weight_ratio"]["failures"][0]


def test_bad_optional_evidence_is_contained_by_each_named_gate() -> None:
    def broken_records():
        raise RuntimeError("seeded record materialization failure")
        yield  # pragma: no cover

    report = _report(
        fit_weight_records=broken_records(),
        parity_evidence=object(),
    )
    gates = _gates(report)

    assert gates["weights_audit"]["passed"] is False
    assert (
        "seeded record materialization failure" in gates["weights_audit"]["failures"][0]
    )
    for name in ("export_surface", "target_surface", "target_fit"):
        assert gates[name]["passed"] is False
        assert "must be UKReleaseParityEvidence" in gates[name]["failures"][0]


def test_fit_audit_is_absent_without_evidence_and_required_missing_fails() -> None:
    absent = _gates(_report())
    required = _gates(_report(require_fit_weight_records=True))

    assert "weights_audit" not in absent
    assert required["weights_audit"]["passed"] is False
    assert required["weights_audit"]["details"]["evidence_missing"] is True


def test_fit_audit_uses_real_records_and_rejects_unweighted_fit() -> None:
    passing = _gates(
        _report(fit_weight_records=(FitWeightRecord("spi_qrf", "importance"),))
    )
    failing = _gates(_report(fit_weight_records=(FitWeightRecord("spi_qrf", "none"),)))

    assert passing["weights_audit"]["passed"] is True
    assert failing["weights_audit"]["passed"] is False
    assert failing["weights_audit"]["details"]["unweighted_fits"] == ["spi_qrf"]


def test_parity_trio_is_absent_without_evidence_and_present_with_evidence() -> None:
    absent = _gates(_report())
    evidence = UKReleaseParityEvidence(
        candidate_columns={"person.age"},
        reference_columns={"person.age"},
        candidate_targets={"ons/population"},
        reference_targets={"ons/population"},
        target_relative_errors={"ons/population": 0.01},
    )
    present = _gates(_report(parity_evidence=evidence))

    assert {"export_surface", "target_surface", "target_fit"}.isdisjoint(absent)
    assert all(
        present[name]["passed"]
        for name in ("export_surface", "target_surface", "target_fit")
    )


def test_parity_evidence_must_be_complete_and_nonvacuous() -> None:
    with pytest.raises(ValueError, match="exactly cover candidate_targets"):
        UKReleaseParityEvidence(
            candidate_columns={"person.age"},
            reference_columns={"person.age"},
            candidate_targets={"ons/population"},
            reference_targets={"ons/population"},
            target_relative_errors={"different": 0.01},
        )


def test_ported_june_parity_gates_retain_their_named_failures() -> None:
    export = uk_export_surface_gate(
        {"person.age"},
        {"person.age", "person.attends_private_school"},
    )
    surface = uk_target_surface_gate(
        {"ons/population"},
        {"ons/population", "hmrc/income_tax"},
    )
    fit = uk_target_fit_gate({"ons/population": -0.40})

    assert not export.passed
    assert export.name == "export_surface"
    assert not surface.passed
    assert surface.name == "target_surface"
    assert not fit.passed
    assert fit.name == "target_fit"


def test_ported_june_parity_gates_reject_empty_evidence() -> None:
    export = uk_export_surface_gate((), ())
    surface = uk_target_surface_gate((), ())
    fit = uk_target_fit_gate({})

    assert not export.passed
    assert not surface.passed
    assert not fit.passed
    assert "evidence is empty" in " ".join(export.failures)
    assert "evidence is empty" in " ".join(surface.failures)
    assert "evidence is empty" in " ".join(fit.failures)


def test_unevidenced_gates_are_omitted_not_stubbed_as_passes() -> None:
    """Item-4 gates joined the battery evidence-gated; item 5 remains future."""

    names = set(_gates(_report()))

    assert {
        "input_mass_parity",
        "qrf_tail_concentration",
        "delivered_take_up",
    }.isdisjoint(names)


def _input_mass_reference(totals=None) -> UKInputMassReference:
    return UKInputMassReference(
        totals=({"person.employment_income": 10.0} if totals is None else totals),
        filename="enhanced_frs_2023_24.h5",
        revision="655dd07e4bb9c777b00dac044949611f1feb824f",
        sha256=("584ae33d80ca0431254610a3f8254d132da73477d31966d6446282861ecae50d"),
        vintage="2023_24",
    )


def _input_mass_policy(**overrides) -> UKInputMassParityPolicy:
    fields = {"relative_tolerance": 0.5, "minimum_reference_total": 0.0}
    fields.update(overrides)
    return UKInputMassParityPolicy(**fields)


def _qrf_tail_policy(**overrides) -> UKQRFTailConcentrationPolicy:
    fields = {"top_k": 1, "max_top_share": 0.5, "min_nonzero_records": 2}
    fields.update(overrides)
    return UKQRFTailConcentrationPolicy(**fields)


def test_armed_weighted_integrity_gates_join_membership_and_attestation() -> None:
    """Removing either armed gate from the battery must fail this test."""

    report = _report(
        input_mass_reference=_input_mass_reference(),
        input_mass_policy=_input_mass_policy(),
        qrf_tail_policy=_qrf_tail_policy(),
    )
    gates = _gates(report)

    assert list(gates) == [
        "uk_release_input_coverage",
        "degenerate_release_surface",
        "zero_weight_strata",
        "weight_ess",
        "weight_ratio",
        "input_mass_parity",
        "qrf_tail_concentration",
    ]
    assert gates["input_mass_parity"]["passed"] is True
    assert set(report.evidence_sha256) == {
        "release_dataset",
        "input_mass_parity",
        "qrf_tail_concentration",
    }
    assert report.attestation["evaluated_gates"][-2:] == [
        "input_mass_parity",
        "qrf_tail_concentration",
    ]


def test_zeroed_input_column_fails_the_armed_battery_by_name() -> None:
    report = _report(
        _dataset(signal=0.0),
        input_mass_reference=_input_mass_reference(),
        input_mass_policy=_input_mass_policy(),
    )
    gate = _gates(report)["input_mass_parity"]

    assert not report.passed
    assert gate["passed"] is False
    assert "person.employment_income" in gate["failures"][0]
    assert "mass is zero" in gate["failures"][0]
    assert gate["details"]["reference_identity"]["filename"] == (
        "enhanced_frs_2023_24.h5"
    )


def test_999_permille_mass_loss_fails_the_armed_battery_by_name() -> None:
    report = _report(
        _dataset(signal=[0.001, 0.002, 0.003, 0.004]),
        input_mass_reference=_input_mass_reference(),
        input_mass_policy=_input_mass_policy(),
    )
    gate = _gates(report)["input_mass_parity"]

    assert gate["passed"] is False
    assert "person.employment_income" in gate["failures"][0]
    assert "-99.9%" in gate["failures"][0]


def test_concentrated_qrf_output_fails_the_armed_battery_by_name() -> None:
    n = 10
    values = np.ones(n)
    values[0] = 1_000.0
    dataset = _dataset(n=n)
    dataset.person["self_employment_income"] = values

    report = _report(dataset, qrf_tail_policy=_qrf_tail_policy())
    gate = _gates(report)["qrf_tail_concentration"]

    assert not report.passed
    assert gate["passed"] is False
    assert "self_employment_income" in gate["failures"][0]
    assert gate["details"]["surface"]["declared_qrf_outputs"] >= 47


def test_armed_gate_without_reference_or_thresholds_fails_closed() -> None:
    missing_reference = _gates(_report(input_mass_policy=_input_mass_policy()))[
        "input_mass_parity"
    ]
    missing_policy = _gates(_report(input_mass_reference=_input_mass_reference()))[
        "input_mass_parity"
    ]

    assert missing_reference["passed"] is False
    assert "no UKInputMassReference" in missing_reference["failures"][0]
    assert missing_policy["passed"] is False
    assert "#609 measurement pass" in missing_policy["failures"][0]


def test_weighted_integrity_evaluator_crash_does_not_mask_pending_gates() -> None:
    report = _report(
        input_mass_reference=object(),
        input_mass_policy=_input_mass_policy(),
        qrf_tail_policy=object(),
    )
    gates = _gates(report)

    assert gates["input_mass_parity"]["passed"] is False
    assert "must be UKInputMassReference" in gates["input_mass_parity"]["failures"][0]
    assert gates["qrf_tail_concentration"]["passed"] is False
    assert (
        "must be UKQRFTailConcentrationPolicy"
        in gates["qrf_tail_concentration"]["failures"][0]
    )
    # Pending gates still evaluated and reported (#547).
    assert gates["weight_ratio"]["passed"] is True


def test_weighted_integrity_thresholds_are_bound_into_the_policy_digest() -> None:
    unarmed = _report()
    armed = _report(
        input_mass_reference=_input_mass_reference(),
        input_mass_policy=_input_mass_policy(),
    )
    retuned = _report(
        input_mass_reference=_input_mass_reference(),
        input_mass_policy=_input_mass_policy(relative_tolerance=0.25),
    )

    digests = {
        report.attestation["policy_sha256"] for report in (unarmed, armed, retuned)
    }
    assert len(digests) == 3


def test_stale_weighted_integrity_exclusions_fail_the_armed_battery() -> None:
    report = _report(
        input_mass_reference=_input_mass_reference(),
        input_mass_policy=_input_mass_policy(
            reviewed_exclusions={
                "person.employment_income": _entry("Seeded stale entry."),
            }
        ),
    )
    gate = _gates(report)["input_mass_parity"]

    assert gate["passed"] is False
    assert "Stale reviewed input-mass exclusions" in gate["failures"][0]


def test_terminal_report_writer_round_trips_strict_atomic_json(tmp_path) -> None:
    report = _report()
    output = tmp_path / "terminal_gates.json"

    written = write_uk_terminal_gate_report(report, output)
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 3
    assert payload["enforced"] is True
    assert payload["passed"] is True
    assert payload["gates"] == _gates(report)
    attestation = payload["attestation"]
    assert attestation["schema_version"] == 5
    assert attestation["producer"] == UK_TERMINAL_GATE_PRODUCER
    assert attestation["release_id"] == TEST_UK_RELEASE_ID
    assert (
        attestation["calibration_diagnostics_sha256"]
        == TEST_UK_CALIBRATION_DIAGNOSTICS_SHA256
    )
    assert attestation["policy_sha256"] != uk_terminal_gate_policy_sha256()
    assert attestation["evaluated_gates"] == [
        "uk_release_input_coverage",
        "degenerate_release_surface",
        "zero_weight_strata",
        "weight_ess",
        "weight_ratio",
    ]
    assert set(attestation["evidence_sha256"]) == {"release_dataset"}
    assert len(attestation["gate_results_sha256"]) == 64
    assert attestation["signature_algorithm"] == UK_TERMINAL_GATE_SIGNATURE_ALGORITHM
    assert len(attestation["signing_key_sha256"]) == 64
    assert len(attestation["signature"]) == 64
    unsigned_report = {
        **payload,
        "attestation": {
            key: value for key, value in attestation.items() if key != "signature"
        },
    }
    expected_signature = hmac.new(
        base64.b64decode(TEST_UK_TERMINAL_GATE_SIGNING_KEY),
        json.dumps(
            unsigned_report,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode(),
        hashlib.sha256,
    ).hexdigest()
    assert hmac.compare_digest(attestation["signature"], expected_signature)
    assert list(tmp_path.glob(".terminal_gates.json.*.tmp")) == []


def test_terminal_report_writer_persists_before_missing_signing_key_raise(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv(UK_TERMINAL_GATE_SIGNING_KEY_ENV)
    report = _report()
    output = tmp_path / "terminal_gates.json"

    assert report.passed is False
    assert report.passed == report.report_payload()["passed"]
    with pytest.raises(RuntimeError, match="Unsigned failed report was written"):
        write_uk_terminal_gate_report(report, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["attestation"]["signature"] is None
    assert payload["attestation"]["signing_key_sha256"] is None


@pytest.mark.parametrize(
    ("encoded_key", "match"),
    [
        ("not-base64!", "must be valid base64"),
        (base64.b64encode(b"x" * 31).decode(), "exactly 32 bytes"),
    ],
)
def test_terminal_report_signer_rejects_malformed_or_wrong_length_key(
    monkeypatch,
    tmp_path,
    encoded_key: str,
    match: str,
) -> None:
    monkeypatch.setenv(UK_TERMINAL_GATE_SIGNING_KEY_ENV, encoded_key)
    report = _report()
    output = tmp_path / "terminal_gates.json"

    with pytest.raises(RuntimeError, match=match):
        write_uk_terminal_gate_report(report, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["attestation"]["signature"] is None
    assert payload["attestation"]["signing_key_sha256"] is None


def test_terminal_signature_canonicalizes_dict_order_and_float_formatting() -> None:
    from microcosm.build.uk_runtime import terminal_gates

    left = json.loads('{"z":1e0,"nested":{"beta":2.50,"alpha":3.0}}')
    right = json.loads('{"nested":{"alpha":3e0,"beta":25e-1},"z":1.0}')
    key = base64.b64decode(TEST_UK_TERMINAL_GATE_SIGNING_KEY)

    assert terminal_gates._canonical_json_bytes(left) == (
        terminal_gates._canonical_json_bytes(right)
    )
    assert bytes.fromhex(terminal_gates._terminal_gate_signature(key, left)) == (
        bytes.fromhex(terminal_gates._terminal_gate_signature(key, right))
    )


def test_terminal_report_writer_rejects_sol_composed_raw_parity_trio(
    tmp_path,
) -> None:
    report = GateReport(
        (
            uk_export_surface_gate({"person.age"}, {"person.age"}),
            uk_target_surface_gate({"ons/population"}, {"ons/population"}),
            uk_target_fit_gate({"ons/population": 0.0}),
        )
    )
    assert report.passed
    output = tmp_path / "terminal_gates.json"

    with pytest.raises(TypeError, match="returned by uk_terminal_gate_report"):
        write_uk_terminal_gate_report(report, output)

    assert not output.exists()


def test_private_constructor_cannot_mint_sol_raw_parity_trio() -> None:
    """Even importing underscored internals cannot omit mandatory gates."""

    from microcosm.build.uk_runtime import terminal_gates

    results = (
        uk_export_surface_gate({"person.age"}, {"person.age"}),
        uk_target_surface_gate({"ons/population"}, {"ons/population"}),
        uk_target_fit_gate({"ons/population": 0.0}),
    )

    with pytest.raises(ValueError, match="membership must follow"):
        terminal_gates._AttestedUKTerminalGateReport(
            results,
            release_id=TEST_UK_RELEASE_ID,
            calibration_diagnostics_sha256=(TEST_UK_CALIBRATION_DIAGNOSTICS_SHA256),
            policy_sha256=uk_terminal_gate_policy_sha256(),
            evidence_sha256={
                "release_dataset": "a" * 64,
                "release_parity": "b" * 64,
            },
            attestation={},
            _signing_error=None,
        )


def test_private_constructor_cannot_drop_evidenced_weighted_integrity_gates() -> None:
    """An attested report naming increment-4 evidence must carry the gates."""

    from microcosm.build.uk_runtime import terminal_gates

    healthy = _report(
        input_mass_reference=_input_mass_reference(),
        input_mass_policy=_input_mass_policy(),
        qrf_tail_policy=_qrf_tail_policy(),
    )
    trimmed_results = tuple(
        result
        for result in healthy.results
        if result.name not in ("input_mass_parity", "qrf_tail_concentration")
    )

    with pytest.raises(ValueError, match="membership must follow"):
        terminal_gates._AttestedUKTerminalGateReport(
            trimmed_results,
            release_id=TEST_UK_RELEASE_ID,
            calibration_diagnostics_sha256=(TEST_UK_CALIBRATION_DIAGNOSTICS_SHA256),
            policy_sha256=uk_terminal_gate_policy_sha256(),
            evidence_sha256=dict(healthy.evidence_sha256),
            attestation={},
            _signing_error=None,
        )


def test_terminal_report_writer_cannot_resign_aggregator_output(
    monkeypatch,
    tmp_path,
) -> None:
    """The public persistence seam is not a signing oracle for caller input."""

    report = _report()
    original_key_id = report.attestation["signing_key_sha256"]
    original_signature = report.attestation["signature"]
    monkeypatch.setenv(
        UK_TERMINAL_GATE_SIGNING_KEY_ENV,
        base64.b64encode(b"x" * 32).decode(),
    )

    output = write_uk_terminal_gate_report(
        report,
        tmp_path / "terminal_gates.json",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["attestation"]["signing_key_sha256"] == original_key_id
    assert payload["attestation"]["signature"] == original_signature


def test_production_terminal_report_pins_policy_and_evidence_membership(
    monkeypatch,
    tmp_path,
) -> None:
    from microcosm.build.uk_runtime import terminal_gates

    monkeypatch.setattr(
        terminal_gates,
        "uk_release_input_coverage_gate",
        lambda _dataset, _engine: _coverage(),
    )
    evidence = UKReleaseParityEvidence(
        candidate_columns={"person.age"},
        reference_columns={"person.age"},
        candidate_targets={"ons/population"},
        reference_targets={"ons/population"},
        target_relative_errors={"ons/population": 0.01},
    )
    report = uk_terminal_gate_report(
        _dataset(),
        object(),
        release_id=TEST_UK_RELEASE_ID,
        calibration_diagnostics_sha256=(TEST_UK_CALIBRATION_DIAGNOSTICS_SHA256),
        fit_weight_records=(FitWeightRecord("spi_qrf", "importance"),),
        parity_evidence=evidence,
    )
    output = write_uk_terminal_gate_report(
        report,
        tmp_path / "terminal_gates.json",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    # The #630 source_year reviewed exclusion entered the committed register
    # and #610's schema 2 sealed its full approval receipt (approver,
    # adjudication, dates), so the frozen digest moved — the intended
    # tripwire; the pre-#630 digest stays pinned in microcosm-data for the
    # grandfathered June release.
    assert uk_terminal_gate_policy_sha256() == (
        "ae93bd10a02362a523eb077bcbd32b362cef31f0447acbc40537df696e30c757"
    )
    assert payload["attestation"]["policy_sha256"] == (uk_terminal_gate_policy_sha256())
    assert payload["attestation"]["evaluated_gates"] == [
        "uk_release_input_coverage",
        "degenerate_release_surface",
        "zero_weight_strata",
        "weight_ess",
        "weight_ratio",
        "weights_audit",
        "export_surface",
        "target_surface",
        "target_fit",
    ]
    assert set(payload["attestation"]["evidence_sha256"]) == {
        "release_dataset",
        "hmrc_spi_income",
        "release_parity",
    }
    weight_details = payload["gates"]["weight_ratio"]["details"]
    weight_fields = (
        "n_records",
        "positive_weight_records",
        "zero_weight_records",
        "total_weight",
        "effective_sample_size",
        "ess_fraction",
        "median_positive_weight",
        "max_weight",
        "max_to_median_positive_weight",
        "top_1pct_weight_share",
    )
    release_evidence = {
        "weights": {field: weight_details[field] for field in weight_fields}
    }
    encoded = json.dumps(
        release_evidence,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert payload["attestation"]["evidence_sha256"]["release_dataset"] == (
        hashlib.sha256(encoded).hexdigest()
    )


def test_committed_degenerate_register_is_the_policy_of_record() -> None:
    """None resolves to the committed #630 register; the digest seals it.

    The register asserts the structural approval receipt only — never the
    reason prose, which must stay freely editable (any edit still moves the
    frozen digest, so rewording is visible without a prose pin here).
    """

    register = uk_default_degenerate_reviewed_exclusions()
    assert set(register) == {"household.source_year"}
    record = register["household.source_year"]
    assert record.adjudication == "microcosm#630"
    assert record.approved_by == "juaristi22"
    assert record.approved_on == "2026-08-10"
    assert record.expires_on == "2027-02-10"
    assert record.reason.strip()


def test_policy_of_record_is_immutable_and_loaded_once() -> None:
    """The default register cannot drift from the already-computed digest.

    A mutable module-level mapping would let any caller (or a sloppy test)
    change the policy of record for the rest of the process while the frozen
    digest kept attesting the committed one — the exact failure the digest
    exists to prevent. The accessor returns one cached read-only mapping, and
    loading is lazy so a broken committed register surfaces as this call's
    ValueError rather than an ImportError at module import.
    """

    register = uk_default_degenerate_reviewed_exclusions()
    assert register is uk_default_degenerate_reviewed_exclusions()
    with pytest.raises(TypeError):
        register["household.source_year"] = None  # type: ignore[index]
    with pytest.raises(AttributeError):
        register.pop  # noqa: B018 - MappingProxyType exposes no mutators
    assert uk_terminal_gate_policy_sha256() == uk_terminal_gate_policy_sha256()


def test_expired_degenerate_exclusion_fails_with_renewal_context() -> None:
    """Honored through expires_on; strictly after, the combined message names
    the approver, the adjudication, and the lapse date."""

    from datetime import date

    entry = _entry("Fixture broadcast, admitted.")
    honored = uk_degenerate_release_surface_gate(
        _dataset(signal=7.0),
        reviewed_exclusions={"person.employment_income": entry},
        now=date(2027, 2, 10),
    )
    expired = uk_degenerate_release_surface_gate(
        _dataset(signal=7.0),
        reviewed_exclusions={"person.employment_income": entry},
        now=date(2027, 2, 11),
    )

    assert honored.passed
    assert honored.details["expired_exclusions"] == []
    assert not expired.passed
    assert expired.details["expired_exclusions"] == ["person.employment_income"]
    assert expired.details["exclusions_evaluated_on"] == "2027-02-11"
    assert (
        "its reviewed exclusion expired 2027-02-10 (approved_by test-reviewer, "
        "microcosm#610) — renew the adjudication or remove the entry."
        in expired.failures[0]
    )
    # Expired-but-still-degenerate is not stale: the column carries no signal.
    assert expired.details["stale_exclusions"] == []
