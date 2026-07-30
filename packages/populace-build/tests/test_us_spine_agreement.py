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
from populace.build.us_runtime.support_provenance import (
    spine_assembly_manifest,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from populace.build.us_runtime.take_up_contract import load_take_up_contract
from populace.frame import EntitySchema, Frame, WeightKind, Weights


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


def test_default_registry_exactly_covers_chartered_agreement_surface() -> None:
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
    registered_columns = {
        (entity, column)
        for (entity, _family), columns in expected.items()
        for column in columns
    }
    for program in load_take_up_contract().programs:
        key = (program.entity, program.variable)
        if key not in registered_columns:
            expected.setdefault((program.entity, "take_up"), set()).add(
                program.variable
            )
            registered_columns.add(key)
    if ("person", "ssi") not in registered_columns:
        expected.setdefault(("person", "simulated_output"), set()).add("ssi")

    assert actual == expected
    registered = {(spec.entity, column) for spec in registry for column in spec.columns}
    assert {
        (program.entity, program.variable)
        for program in load_take_up_contract().programs
    } <= registered
    assert ("person", "takes_up_ssi_if_eligible") in registered
    assert ("person", "ssi") in registered
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

    actual = {(spec.entity, spec.family): spec.columns for spec in registry}
    assert actual[("person", "boolean")] == ("flag",)
    assert actual[("person", "numeric")] == ("first", "second", "third")
    assert actual[("person", "simulated_output")] == ("ssi",)
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
        "status": "tested",
        "left_incidence": 0.5,
        "right_incidence": 0.5,
        "incidence_ratio_right_over_left": 1.0,
        "quantile_envelope_distance": 0.0,
    }
    assert result.details["tested_spine_pairs"] == 1
    assert result.details["untestable_comparisons"] == []
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


def test_default_registry_rejects_403_shaped_ssi_spine_disagreement() -> None:
    """The reviewer's default-registry repro must reach the SSI comparison."""

    tables: dict[str, pd.DataFrame] = {}
    weights: dict[str, list[float]] = {}
    for entity in sorted({spec.entity for spec in US_SPINE_AGREEMENT_REGISTRY}):
        columns = {
            column
            for spec in US_SPINE_AGREEMENT_REGISTRY
            if spec.entity == entity
            for column in spec.columns
        }
        data: dict[str, object] = {
            f"{entity}_support_channel": ["acs", "acs", "asec", "asec"],
            **{column: np.ones(4, dtype=np.float64) for column in columns},
        }
        if entity == "person":
            data["ssi"] = [1.0, 0.0, 1.0, 0.0]
        tables[entity] = pd.DataFrame(data)
        weights[entity] = [2.59, 97.41, 1.85, 98.15]
    frame = _Frame(tables, weights)

    result = spine_agreement_gate(frame)

    assert not result.passed
    assert result.failures == (
        "person/simulated_output/ssi/acs_vs_asec: weighted "
        "nonzero-incidence ratio 0.714286 is outside [0.8, 1.25] "
        "(left=0.0259, right=0.0185).",
    )
    comparison = result.details["comparisons"][
        "person/simulated_output/ssi/acs_vs_asec"
    ]
    assert comparison["incidence_ratio_right_over_left"] == pytest.approx(
        0.0185 / 0.0259
    )
    assert result.details["untestable_comparisons"] == []


def test_gate_rejects_channel_forged_against_assembly_manifest() -> None:
    schema = EntitySchema(group_entities=("household",))
    channels = np.asarray(["asec", "acs"], dtype=object)
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_household_id": [1, 2],
                support_channel_column("person"): channels.copy(),
                support_clone_index_column("person"): [0, 0],
                support_source_id_column("person"): [1, 2],
                spine_source_id_column("person"): [1, 1],
                "amount": [1.0, 1.0],
            }
        ),
        "household": pd.DataFrame(
            {
                "household_id": [1, 2],
                support_channel_column("household"): channels.copy(),
                support_clone_index_column("household"): [0, 0],
                support_source_id_column("household"): [1, 2],
                spine_source_id_column("household"): [1, 1],
            }
        ),
    }
    frame = Frame(
        tables,
        schema,
        {
            "household": Weights(
                np.asarray([1.0, 1.0]),
                WeightKind.DESIGN,
            )
        },
        metadata=spine_assembly_manifest(
            tables,
            channels=("asec", "acs"),
        ),
    )
    frame.table("person").loc[0, support_channel_column("person")] = "forged_source"

    with pytest.raises(ValueError, match="assembly manifest.*unknown channel"):
        spine_agreement_gate(
            frame,
            registry=(SpineAgreementSpec("person", "numeric", ("amount",)),),
        )


def test_gate_fails_when_only_one_spine_has_nonzero_incidence() -> None:
    frame = _Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_support_channel": ["acs", "asec"],
                    "amount": [0.0, 1.0],
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
    assert any("nonzero-incidence ratio inf" in failure for failure in result.failures)
    comparison = result.details["comparisons"]["person/numeric/amount/acs_vs_asec"]
    assert comparison["status"] == "tested"
    assert comparison["incidence_ratio_right_over_left"] == "infinity"
    assert comparison["quantile_envelope_distance"] == "infinity"


def test_gate_records_both_zero_surface_as_untestable_and_fails() -> None:
    frame = _Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_support_channel": ["acs", "asec"],
                    "amount": [0.0, 0.0],
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

    comparison_key = "person/numeric/amount/acs_vs_asec"
    assert not result.passed
    assert result.failures == (
        f"{comparison_key}: both source spines have zero weighted nonzero "
        "incidence; the registered comparison is untestable.",
    )
    assert result.details["comparisons"][comparison_key] == {
        "status": "untestable_both_zero",
        "left_incidence": 0.0,
        "right_incidence": 0.0,
        "incidence_ratio_right_over_left": None,
        "quantile_envelope_distance": None,
    }
    assert result.details["tested_spine_pairs"] == 0
    assert result.details["untestable_comparisons"] == [comparison_key]


def test_gate_fails_and_represents_missing_pairs_for_inconsistent_source_sets() -> None:
    frame = _Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_support_channel": ["acs", "asec", "sipp"],
                    "amount": [1.0, 1.0, 1.0],
                }
            ),
            "tax_unit": pd.DataFrame(
                {
                    "tax_unit_support_channel": ["acs", "asec"],
                    "amount": [1.0, 1.0],
                }
            ),
        },
        {
            "person": [1.0, 1.0, 1.0],
            "tax_unit": [1.0, 1.0],
        },
    )
    specs = (
        SpineAgreementSpec("person", "numeric", ("amount",)),
        SpineAgreementSpec("tax_unit", "numeric", ("amount",)),
    )

    result = spine_agreement_gate(frame, registry=specs)

    assert not result.passed
    assert result.failures == (
        "tax_unit: source-spine set is inconsistent across registered entity "
        "grains; observed ['acs', 'asec'], expected ['acs', 'asec', 'sipp'], "
        "missing ['sipp'].",
    )
    assert result.details["expected_source_channels"] == ["acs", "asec", "sipp"]
    assert result.details["missing_source_channels_by_entity"] == {"tax_unit": ["sipp"]}
    comparisons = result.details["comparisons"]
    assert comparisons["tax_unit/numeric/amount/acs_vs_sipp"] == {
        "status": "untestable_missing_source_spine",
        "missing_source_spines": ["sipp"],
    }
    assert comparisons["tax_unit/numeric/amount/asec_vs_sipp"] == {
        "status": "untestable_missing_source_spine",
        "missing_source_spines": ["sipp"],
    }
    assert result.details["checked_spine_pairs"] == 6
    assert result.details["tested_spine_pairs"] == 4


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
