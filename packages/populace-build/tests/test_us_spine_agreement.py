from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from populace.build.gates import GateReport
from populace.build.us_runtime.acs_transfer import (
    acs_derived_transfer_expectations,
    declared_acs_transfer_target_families,
)
from populace.build.us_runtime.spine_agreement import (
    DEFAULT_INCIDENCE_RATIO_BOUNDS,
    DEFAULT_QUANTILE_ENVELOPE_TOLERANCE,
    DEFAULT_SPINE_AGREEMENT_QUANTILES,
    US_SPINE_AGREEMENT_REGISTRY,
    SpineAgreementSpec,
    default_spine_agreement_registry,
    normalize_transfer_family_name,
    spine_agreement_gate,
    validate_spine_agreement_registry,
)


@dataclass
class _Weights:
    values: np.ndarray


class _Frame:
    def __init__(
        self,
        tables: dict[str, pd.DataFrame],
        weights: dict[str, list[float]],
    ) -> None:
        self._tables = tables
        self._weights = weights
        self.entities = tuple(tables)

    def table(self, entity: str) -> pd.DataFrame:
        return self._tables[entity]

    def resolve_weights(self, entity: str) -> _Weights:
        return _Weights(np.asarray(self._weights[entity], dtype=np.float64))


def test_default_registry_exactly_covers_declared_and_derived_transfer_surface() -> (
    None
):
    families = declared_acs_transfer_target_families()
    registry = default_spine_agreement_registry()
    assert registry == US_SPINE_AGREEMENT_REGISTRY
    actual = {(spec.entity, spec.family): set(spec.columns) for spec in registry}
    expected: dict[tuple[str, str], set[str]] = {}
    for entity, entity_families in families.items():
        for family, columns in entity_families.items():
            expected.setdefault(
                (entity, normalize_transfer_family_name(family)), set()
            ).update(columns)
    for column, entity in acs_derived_transfer_expectations(families).items():
        expected.setdefault((entity, "derived_transfer"), set()).add(column)

    assert actual == expected
    assert DEFAULT_INCIDENCE_RATIO_BOUNDS == (0.8, 1.25)
    assert DEFAULT_SPINE_AGREEMENT_QUANTILES == (0.10, 0.25, 0.50, 0.75, 0.90)
    assert DEFAULT_QUANTILE_ENVELOPE_TOLERANCE == 0.25


def test_registry_normalizes_and_merges_qrf_batches() -> None:
    registry = default_spine_agreement_registry(
        {
            "person": {
                "numeric__batch_1": ("first", "second"),
                "numeric__batch_2": ("third",),
                "boolean": ("flag",),
            }
        }
    )

    assert [(spec.entity, spec.family, spec.columns) for spec in registry] == [
        ("person", "boolean", ("flag",)),
        ("person", "numeric", ("first", "second", "third")),
    ]
    assert normalize_transfer_family_name("numeric__batch_02") == "numeric"


def test_gate_passes_equal_weighted_distributions_and_writes_manifest_details() -> None:
    frame = _Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_support_channel": [
                        "asec",
                        "asec",
                        "asec",
                        "acs",
                        "acs",
                        "acs",
                        "acs",
                    ],
                    "amount": [0.0, 10.0, 20.0, 0.0, 0.0, 10.0, 20.0],
                }
            )
        },
        {"person": [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
    )
    spec = SpineAgreementSpec(
        entity="person",
        family="numeric",
        columns=("amount",),
    )

    result = spine_agreement_gate(frame, registry=(spec,))

    assert result.passed
    assert result.name == "us_spine_agreement"
    assert result.details["checked_columns"] == 1
    assert result.details["checked_spine_pairs"] == 1
    comparison = result.details["comparisons"]["person/numeric/amount/acs_vs_asec"]
    assert comparison == {
        "left_incidence": 0.5,
        "right_incidence": 0.5,
        "incidence_ratio_right_over_left": 1.0,
        "quantile_envelope_distance": 0.0,
    }
    manifest = GateReport((result,)).to_manifest()
    assert manifest["passed"]
    assert manifest["gates"]["us_spine_agreement"]["passed"]


def test_gate_batches_incidence_and_quantile_failures_across_columns() -> None:
    frame = _Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_support_channel": [
                        "asec",
                        "asec",
                        "asec",
                        "acs",
                        "acs",
                        "acs",
                    ],
                    "amount": [0.0, 10.0, 20.0, 0.0, 100.0, 200.0],
                    "receipt": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                }
            )
        },
        {"person": [1.0] * 6},
    )
    spec = SpineAgreementSpec(
        entity="person",
        family="example",
        columns=("amount", "receipt"),
    )

    result = spine_agreement_gate(frame, registry=(spec,))

    assert not result.passed
    assert len(result.failures) == 2
    assert any(
        "amount/acs_vs_asec" in failure and "conditional-quantile envelope" in failure
        for failure in result.failures
    )
    assert any(
        "receipt/acs_vs_asec" in failure and "nonzero-incidence ratio" in failure
        for failure in result.failures
    )
    report = GateReport((result,))
    assert not report.passed
    assert len(report.failures) == 2
    assert all(
        failure.startswith("[us_spine_agreement]") for failure in report.failures
    )


def test_gate_reports_missing_evidence_as_batched_failure() -> None:
    frame = _Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_support_channel": ["asec", "acs"],
                    "present": [1.0, 1.0],
                }
            )
        },
        {"person": [1.0, 1.0]},
    )
    spec = SpineAgreementSpec(
        entity="person",
        family="example",
        columns=("missing", "present"),
    )

    result = spine_agreement_gate(frame, registry=(spec,))

    assert not result.passed
    assert result.details["checked_columns"] == 1
    assert result.failures == (
        "person/example/missing: registered column is absent from the frame.",
    )


def test_gate_batches_nullable_positive_weight_values_as_evidence_failure() -> None:
    frame = _Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_support_channel": ["asec", "acs"],
                    "amount": pd.Series([1.0, pd.NA], dtype="Float64"),
                }
            )
        },
        {"person": [1.0, 1.0]},
    )
    spec = SpineAgreementSpec(
        entity="person",
        family="numeric",
        columns=("amount",),
    )

    result = spine_agreement_gate(frame, registry=(spec,))

    assert not result.passed
    assert result.failures == (
        "person/numeric/amount: 1 positive-weight value(s) are missing or non-finite.",
    )


def test_gate_rejects_legacy_clone_role_as_source_spine_provenance() -> None:
    frame = _Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_support_channel": ["asec", "puf_tax_detail"],
                    "amount": [1.0, 1.0],
                }
            )
        },
        {"person": [1.0, 1.0]},
    )
    spec = SpineAgreementSpec(
        entity="person",
        family="numeric",
        columns=("amount",),
    )

    result = spine_agreement_gate(frame, registry=(spec,))

    assert not result.passed
    assert result.failures == (
        "person: 'puf_tax_detail' is a clone role, not valid immutable "
        "source-spine provenance.",
    )


def test_malformed_registry_fails_closed() -> None:
    spec = SpineAgreementSpec(
        entity="person",
        family="numeric",
        columns=("amount",),
    )
    with pytest.raises(ValueError, match="repeats entity/family"):
        validate_spine_agreement_registry((spec, spec))
    with pytest.raises(ValueError, match="batch suffix"):
        normalize_transfer_family_name("numeric__batch_nope")
    with pytest.raises(ValueError, match="does not exactly cover"):
        validate_spine_agreement_registry(
            (spec,),
            target_families={"person": {"numeric": ("amount", "other")}},
        )
    with pytest.raises(ValueError, match="immutable tuple"):
        SpineAgreementSpec(  # type: ignore[arg-type]
            entity="person",
            family="numeric",
            columns=["amount"],
        )
