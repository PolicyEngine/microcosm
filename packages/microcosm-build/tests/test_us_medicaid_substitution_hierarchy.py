"""Engine-free regressions for the reviewed CMS source-month substitution."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from microcosm.build.ledger_targets import (
    LedgerTargetReference,
    compile_ledger_target_references,
)
from microcosm.build.us_runtime.medicaid_take_up import (
    US_MEDICAID_ENROLLMENT_SUBSTITUTIONS,
    US_MEDICAID_ENROLLMENT_TARGET_ROLE,
    apply_us_medicaid_enrollment_substitutions,
    us_medicaid_take_up_gate,
)
from microcosm.calibrate import (
    CalibrationHierarchySeed,
    HierarchyCategory,
    HierarchyDimension,
    HierarchyGeography,
    HierarchyNode,
    TargetRegistry,
    TargetSpec,
)


def _state_spec(
    fips: str = "51", postal: str = "va", label: str = "Virginia"
) -> TargetSpec:
    """Compile the same hierarchy shape as the public CMS consumer facts."""
    record_set = "cms_medicaid.month2024_12.state_enrollment"
    name = f"{record_set}.{postal}.total_medicaid_enrollment"
    fact = {
        "lineage": {"source_record_id": name},
        "label": f"{label} 2024-12 total Medicaid enrollment",
        "value": 1_609_622,
        "period": {"type": "month", "value": "2024-12"},
        "geography": {"level": "state", "id": f"0400000US{fips}", "name": label},
        "aggregation": {"method": "sum"},
        "observed_measure": {
            "source_name": "cms_medicaid",
            "source_measure_id": "total_medicaid_enrollment",
        },
        "layout": {
            "record_set_id": record_set,
            "groupby_dimension": "cms_medicaid.state_abbreviation",
            "groupby_dimension_label": "State",
            "groupby_value_id": postal,
            "groupby_value_label": label,
            "measure_id": "total_medicaid_enrollment",
        },
        # An unrelated dimension must keep its identity and position.
        "dimensions": {"coverage": "medicaid"},
        "dimension_labels": {"coverage": "Coverage"},
        "dimension_value_labels": {"coverage": {"medicaid": "Medicaid"}},
        "aggregate_fact_key": f"aggregate-{postal}-2024-12",
        "semantic_fact_key": f"semantic-{postal}-2024-12",
        "legacy_fact_key": f"legacy-{postal}-2024-12",
    }
    reference = LedgerTargetReference(
        name=name,
        ledger_source_record_id=name,
        entity="household",
        measure=name,
        period=2024,
        source="CMS Medicaid & CHIP monthly enrollment (April 2026 release)",
        family="cms_medicaid",
        metadata={
            "materializer": "policyengine_variable",
            "base_variable": "medicaid_enrolled",
            "measure_mode": "indicator_sum",
            "target_role": US_MEDICAID_ENROLLMENT_TARGET_ROLE,
            "source_period": "2024-12",
            "state_fips": fips,
        },
        hierarchy=CalibrationHierarchySeed(
            provider=HierarchyNode("cms_medicaid", "CMS"),
            category=HierarchyCategory(
                "cms_medicaid.total_medicaid_enrollment",
                "Total Medicaid enrollment",
                "cms_medicaid",
            ),
            target_label="Total Medicaid enrollment",
        ),
    )
    return compile_ledger_target_references([fact], [reference], country="us").specs[0]


@pytest.mark.parametrize(
    ("fips", "postal", "label"),
    [("51", "va", "Virginia"), ("06", "ca", "California")],
)
def test_substitution_rebinds_complete_hierarchy_without_mutating_template(
    fips: str, postal: str, label: str, tmp_path: Path
) -> None:
    template = _state_spec(fips, postal, label)
    registry = TargetRegistry([template], country="us")
    before = asdict(template)
    version = registry.version

    augmented, records = apply_us_medicaid_enrollment_substitutions(registry)

    assert len(augmented) == 2
    assert augmented.specs[0] is template
    assert asdict(template) == before
    assert registry.version == version
    spec = augmented.specs[1]
    substitution = US_MEDICAID_ENROLLMENT_SUBSTITUTIONS[0]
    assert spec.name == (
        f"{substitution.substitute_source_record_id}.medicaid_enrollment_substitution"
    )
    assert spec.measure == spec.name
    assert spec.value == 273_400.0  # Reviewed RI November 2024 count (#386).
    assert (spec.entity, spec.period, spec.source, spec.family) == (
        template.entity,
        template.period,
        template.source,
        template.family,
    )
    hierarchy = spec.hierarchy
    assert hierarchy is not None
    assert hierarchy.provider == template.hierarchy.provider
    assert hierarchy.category == template.hierarchy.category
    assert hierarchy.geography == HierarchyGeography(
        "0400000US44", "Rhode Island", "state"
    )
    assert hierarchy.dimensions == (
        HierarchyDimension(
            "cms_medicaid.state_abbreviation", "State", "ri", "Rhode Island"
        ),
        template.hierarchy.dimensions[1],
    )
    assert hierarchy.target.id == spec.name
    assert hierarchy.target.label == (
        "Rhode Island Medicaid enrollment (2024-11 source substituted for 2024-12)"
    )
    assert (
        spec.metadata.items()
        >= {
            "state_fips": "44",
            "ledger_geography_id": "0400000US44",
            "ledger_geography_name": "Rhode Island",
            "ledger_layout_groupby_value_id": "ri",
            "ledger_layout_groupby_value_label": "Rhode Island",
            "ledger_layout_record_set_id": "cms_medicaid.month2024_11.state_enrollment",
            "ledger_source_record_id": substitution.substitute_source_record_id,
            "ledger_fact_period": "2024-11",
            "source_period": "2024-11",
            "substituted_for_source_record_id": substitution.substituted_for_source_record_id,
            "substituted_for_source_period": "2024-12",
            "medicaid_enrollment_substitution": "true",
            "substitution_issue": "microcosm#386",
            "substitution_reason": substitution.reason,
            "materializer": "policyengine_variable",
            "base_variable": "medicaid_enrolled",
            "measure_mode": "indicator_sum",
            "target_role": US_MEDICAID_ENROLLMENT_TARGET_ROLE,
        }.items()
    )
    for key in (
        "ledger_fact_key",
        "ledger_aggregate_fact_key",
        "ledger_semantic_fact_key",
        "ledger_legacy_fact_key",
        "ledger_fact_label",
    ):
        assert key not in spec.metadata
    assert records[0]["applied"] is True
    assert records[0]["stale"] is False
    target = next(
        target for target in augmented.to_target_set() if target.name == spec.name
    )
    assert target.hierarchy == hierarchy
    assert target.value == spec.value
    assert target.metadata == spec.metadata
    restored = TargetRegistry.from_json(augmented.to_json(tmp_path / "registry.json"))
    assert restored.version == augmented.version
    assert restored.specs == augmented.specs


def test_backfilled_hierarchical_target_still_marks_substitution_stale() -> None:
    registry = TargetRegistry(
        [_state_spec(), _state_spec("44", "ri", "Rhode Island")], country="us"
    )
    augmented, records = apply_us_medicaid_enrollment_substitutions(registry)
    assert augmented.specs == registry.specs
    assert records[0]["applied"] is False
    assert records[0]["stale"] is True
    gate = us_medicaid_take_up_gate(
        {"states": [], "medicaid_enrollment_substitutions": records}
    )
    assert not gate.passed
    assert any(
        "stale" in failure and "microcosm#386" in failure for failure in gate.failures
    )


def test_legacy_target_without_hierarchy_remains_supported() -> None:
    template = replace(_state_spec(), hierarchy=None)
    # Existing callers need not supply the new label for hierarchy-free specs.
    substitution = replace(US_MEDICAID_ENROLLMENT_SUBSTITUTIONS[0], state_label=None)
    augmented, records = apply_us_medicaid_enrollment_substitutions(
        TargetRegistry([template], country="us"), substitutions=[substitution]
    )
    assert augmented.specs[1].hierarchy is None
    assert augmented.specs[1].value == 273_400.0
    assert records[0]["applied"] is True


@pytest.mark.parametrize("state_label", [None, "", "  "])
def test_hierarchical_substitution_requires_explicit_state_label(
    state_label: str | None,
) -> None:
    substitution = replace(
        US_MEDICAID_ENROLLMENT_SUBSTITUTIONS[0], state_label=state_label
    )
    with pytest.raises(ValueError, match="requires a reviewed state_label"):
        apply_us_medicaid_enrollment_substitutions(
            TargetRegistry([_state_spec()], country="us"),
            substitutions=[substitution],
        )


def test_substituted_target_is_not_a_natural_backfill_or_template() -> None:
    augmented, _ = apply_us_medicaid_enrollment_substitutions(
        TargetRegistry([_state_spec()], country="us")
    )
    registry = TargetRegistry([augmented.specs[1]], country="us")
    result, records = apply_us_medicaid_enrollment_substitutions(registry)
    assert result.specs == registry.specs
    assert records[0]["applied"] is False
    assert records[0]["stale"] is False


def test_registry_without_natural_family_remains_a_no_op() -> None:
    registry = TargetRegistry([], country="us")
    augmented, records = apply_us_medicaid_enrollment_substitutions(registry)
    assert augmented.specs == ()
    assert records[0]["applied"] is False
    assert records[0]["stale"] is False
