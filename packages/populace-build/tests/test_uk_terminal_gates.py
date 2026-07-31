"""UK terminal-gate batching and seeded-defect coverage."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from populace.build.gates import FitWeightRecord, GateResult
from populace.build.uk_runtime.terminal_gates import (
    UK_DEFAULT_ZERO_WEIGHT_STRATA,
    UKReleaseParityEvidence,
    UKZeroWeightStratumDeclaration,
    uk_degenerate_release_surface_gate,
    uk_export_surface_gate,
    uk_target_fit_gate,
    uk_target_surface_gate,
    uk_terminal_gate_report,
    write_uk_terminal_gate_report,
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
    return uk_terminal_gate_report(
        _dataset() if dataset is None else dataset,
        object(),
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
        reviewed_exclusions={"person.employment_income": reason},
    )
    stale = uk_degenerate_release_surface_gate(
        _dataset(),
        reviewed_exclusions={"person.employment_income": reason},
    )

    assert live.passed
    assert (
        live.details["reviewed_exclusions"]["person.employment_income"]["reason"]
        == reason
    )
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


def test_gate_evaluation_error_does_not_mask_later_findings() -> None:
    report = uk_terminal_gate_report(
        _dataset(signal=0.0),
        object(),
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


def test_item_4_and_5_future_gates_are_not_stubbed_as_passes() -> None:
    names = set(_gates(_report()))

    assert {
        "input_mass_parity",
        "qrf_tail_concentration",
        "delivered_take_up",
    }.isdisjoint(names)


def test_terminal_report_writer_round_trips_strict_atomic_json(tmp_path) -> None:
    report = _report()
    output = tmp_path / "terminal_gates.json"

    written = write_uk_terminal_gate_report(report, output)
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["enforced"] is True
    assert payload["passed"] is True
    assert payload["gates"] == _gates(report)
    assert list(tmp_path.glob(".terminal_gates.json.*.tmp")) == []
