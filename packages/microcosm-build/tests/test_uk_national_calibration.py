"""Shared UK measure materialization and target-loss contracts."""

from __future__ import annotations

import json
from importlib import resources as importlib_resources

import numpy as np
import pandas as pd
import pytest

from microcosm.build.ledger_targets import LedgerTargetReference
from microcosm.build.uk_runtime import (
    UK_NATIONAL_L0_LAMBDA,
    UK_NATIONAL_LEARNING_RATE,
    UK_NATIONAL_MASS_RULE,
    UK_NATIONAL_MAX_WEIGHT_RATIO,
    UK_NATIONAL_SEED,
    UK_NATIONAL_SOLVE_DOCTRINE,
    UK_NATIONAL_SOLVE_EPOCHS,
    UK_NATIONAL_TARGET_LOSS_CAP,
    UK_NATIONAL_TARGET_WEIGHT_RULE,
    UKNationalSolveDoctrine,
    uk_doctrine_with_overrides,
    uk_national_target_loss_weights,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

ACTIVE_REFERENCE_COUNT = 415


def _fact(
    *,
    concept: str = "dwp.uc_benefit_units",
    source_name: str = "dwp",
    value: float = 30.0,
) -> dict:
    return {
        "label": "Test Universal Credit fact",
        "aggregate_fact_key": "ledger.aggregate_fact.v2:uc-fixture",
        "aggregation": {"method": "sum"},
        "assertion": "observation",
        "entity": {"name": "person"},
        "geography": {
            "level": "country",
            "id": "K02000001",
            "name": "United Kingdom",
        },
        "observed_measure": {
            "source_name": source_name,
            "source_concept": concept,
            "source_measure_id": "total_units",
            "unit": "count",
        },
        "period": {"type": "month", "value": "2025-12"},
        "value": value,
    }


def _reference_by_name(name: str) -> LedgerTargetReference:
    from microcosm.build.country_spec import load_country_spec

    return next(
        reference
        for reference in load_country_spec("uk").target_references
        if reference.name == name
    )


def _fact_for_reference(
    reference: LedgerTargetReference,
    value: float,
) -> dict:
    selector = dict(reference.ledger_selector)
    dimensions = dict(selector.get("dimension_values", {}))
    layout = {
        key: selector[key]
        for key in (
            "groupby_dimension",
            "groupby_value_id",
            "record_set_spec_id",
        )
        if key in selector
    }
    groupby_dimension = str(layout.get("groupby_dimension") or "")
    groupby_value_id = str(layout.get("groupby_value_id") or "")
    hierarchy_dimensions = dict(dimensions)
    if groupby_dimension:
        hierarchy_dimensions.setdefault(groupby_dimension, groupby_value_id)
    dimension_labels = {
        dimension_id: f"Test dimension {dimension_id}"
        for dimension_id in hierarchy_dimensions
    }
    dimension_value_labels = {
        dimension_id: {
            str(dimension_value): f"Test value {dimension_id}={dimension_value}"
        }
        for dimension_id, dimension_value in hierarchy_dimensions.items()
    }
    if groupby_dimension:
        layout["groupby_dimension_label"] = dimension_labels[groupby_dimension]
        if groupby_value_id:
            layout["groupby_value_label"] = dimension_value_labels[groupby_dimension][
                groupby_value_id
            ]
    return {
        "label": f"Test label for {reference.name}",
        "aggregate_fact_key": selector.get(
            "aggregate_fact_key", f"ledger.aggregate_fact.v2:{reference.name}"
        ),
        "aggregation": {"method": "sum"},
        "assertion": "observation",
        "entity": {"name": selector.get("entity_name", reference.entity)},
        "dimensions": dimensions,
        "dimension_labels": dimension_labels,
        "dimension_value_labels": dimension_value_labels,
        "geography": {
            "level": selector.get("geography_level", "country"),
            "id": selector.get("geography_id", "K02000001"),
            "name": "United Kingdom",
        },
        "layout": layout,
        "observed_measure": {
            "source_name": selector["source_name"],
            "source_concept": selector["source_concept"],
            "source_measure_id": selector.get("source_measure_id", "value"),
            "unit": "gbp",
        },
        "period": {"type": "tax_year", "value": 2024}
        if reference.name.startswith("hmrc.cgt.")
        else {"type": "month", "value": f"{reference.period}-12"},
        "value": value,
    }


def _facts_for_reference(reference: LedgerTargetReference, value: float) -> list[dict]:
    """Expand the paid headline into its declared synthetic month/cell grid."""
    if reference.name != "dwp.uc.households":
        return [_fact_for_reference(reference, value)]

    facts = []
    for month in reference.ledger_selector["period_value"]:
        for cell, operand in enumerate(reference.value_operands):
            fact = _fact_for_reference(reference, value / len(reference.value_operands))
            fact["aggregate_fact_key"] += f":{month}:{cell}"
            fact["dimensions"] = dict(operand["dimension_values"])
            fact["dimension_labels"] = {
                dimension_id: f"Test dimension {dimension_id}"
                for dimension_id in fact["dimensions"]
            }
            fact["dimension_value_labels"] = {
                dimension_id: {
                    str(dimension_value): (
                        f"Test value {dimension_id}={dimension_value}"
                    )
                }
                for dimension_id, dimension_value in fact["dimensions"].items()
            }
            fact["period"] = {"type": "month", "value": month}
            fact["source_release_key"] = "ledger.source_release.v2:uc-paid-fixture"
            fact["source"] = {"source_sha256": "a" * 64}
            fact["observed_measure"]["unit"] = "count"
            facts.append(fact)
    return facts


def _materialization_binding_frame(
    *,
    include_counterfactual_delta: bool = True,
) -> Frame:
    household = pd.DataFrame(
        {
            "household_id": np.arange(3, dtype="int64"),
            "region": "LONDON",
            "esa_income": [10.0, 20.0, 0.0],
            "esa_contrib": [1.0, 2.0, 0.0],
        }
    )
    salary_sacrifice_metric = "hmrc/salary_sacrifice_it_relief_basic_rate"
    person_columns = {
        "person_id": np.arange(6, dtype="int64"),
        "person_benunit_id": [0, 0, 1, 2, 2, 2],
        "person_household_id": [0, 0, 1, 2, 2, 2],
        "capital_gains": [0.0, 7_000.0, 12_000.0, 500.0, 0.0, 0.0],
        # Two flagged children in household 0, none in 1, three in 2.
        "uc_is_child_limit_affected": [True, True, False, True, True, True],
    }
    if include_counterfactual_delta:
        person_columns[salary_sacrifice_metric] = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    return Frame(
        {
            "person": pd.DataFrame(person_columns),
            "benunit": pd.DataFrame(
                {
                    "benunit_id": np.arange(3, dtype="int64"),
                    "universal_credit": [1.0, 0.0, 1.0],
                }
            ),
            "household": household,
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.full(3, 10.0), WeightKind.DESIGN)},
        metadata={"time_period": "2025"},
    )


def test_chronicle_184_uc_and_obr_references_compile_fail_closed() -> None:
    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.ledger_targets import compile_ledger_target_references

    spec = load_country_spec("uk")
    assert len(spec.target_references) == ACTIVE_REFERENCE_COUNT
    reference_names = {reference.name for reference in spec.target_references}
    assert "obr.universal_credit_in_cap" in reference_names
    assert "dwp.uc.households" in reference_names

    membership = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("target_reference_membership.json")
        .read_text()
    )
    assert membership["targets"]["dwp.uc.households"]["status"] == "active"
    uc_reference = next(
        reference
        for reference in spec.target_references
        if reference.name == "dwp.uc.households"
    )
    assert uc_reference.value_operation == "monthly_window_sum_average"
    assert uc_reference.period_match_policy == "source_window"
    assert uc_reference.ledger_selector["period_value"] == [
        f"2025-{month:02d}" for month in range(1, 13)
    ]
    assert len(uc_reference.value_operands) == 10
    assert {
        tuple(
            operand["dimension_values"][key]
            for key in ("family_type", "payment_indicator", "child_entitlement")
        )
        for operand in uc_reference.value_operands
    } == {
        (family, "Yes", entitled)
        for family in (
            "Single, no children",
            "Single, with children",
            "Couple, no children",
            "Couple, with children",
            "Unknown or missing family type",
        )
        for entitled in ("No", "Yes")
    }

    references = tuple(
        reference
        for reference in spec.target_references
        if reference.name == "obr.universal_credit_in_cap"
    )
    registry = compile_ledger_target_references(
        [
            _fact(
                concept="obr.universal_credit_in_cap",
                source_name="obr",
                value=40_000_000_000,
            ),
        ],
        references,
        country="uk",
    )

    assert {spec.name for spec in registry.specs} == {"obr.universal_credit_in_cap"}


def test_national_doctrine_constants_are_the_declared_contract() -> None:
    assert UK_NATIONAL_SOLVE_EPOCHS == 256
    assert UK_NATIONAL_LEARNING_RATE == 0.02
    assert UK_NATIONAL_MAX_WEIGHT_RATIO == 10.0
    assert UK_NATIONAL_SEED == 0
    assert UK_NATIONAL_TARGET_LOSS_CAP == 10.0
    assert UK_NATIONAL_L0_LAMBDA == 0.0
    assert UK_NATIONAL_MASS_RULE == "free"
    # María's ruling (2026-08-24): family_equal is vocabulary, never the
    # default — she passes it as an explicit per-run override.
    assert UK_NATIONAL_TARGET_WEIGHT_RULE == "uniform"
    assert UK_NATIONAL_SOLVE_DOCTRINE == UKNationalSolveDoctrine()
    assert UK_NATIONAL_SOLVE_DOCTRINE.scale_rule == "default_target_loss_scales"
    assert UK_NATIONAL_SOLVE_DOCTRINE.target_weight_rule == "uniform"
    assert UKNationalSolveDoctrine(target_weight_rule="family_equal")


def test_uk_doctrine_with_overrides_receipts_effective_diffs_only() -> None:
    doctrine, receipt = uk_doctrine_with_overrides()
    assert doctrine == UK_NATIONAL_SOLVE_DOCTRINE
    assert receipt == {}

    doctrine, receipt = uk_doctrine_with_overrides(
        epochs=UK_NATIONAL_SOLVE_EPOCHS,
        target_weight_rule=UK_NATIONAL_TARGET_WEIGHT_RULE,
    )
    assert doctrine == UK_NATIONAL_SOLVE_DOCTRINE
    assert receipt == {}

    doctrine, receipt = uk_doctrine_with_overrides(
        epochs=128,
        learning_rate=0.01,
        target_weight_rule="family_equal",
        target_loss_cap=5.0,
    )
    assert doctrine.epochs == 128
    assert doctrine.learning_rate == 0.01
    assert doctrine.target_weight_rule == "family_equal"
    assert doctrine.target_loss_cap == 5.0
    assert receipt == {
        "epochs": {"default": 256, "effective": 128},
        "learning_rate": {"default": 0.02, "effective": 0.01},
        "target_loss_cap": {"default": 10.0, "effective": 5.0},
        "target_weight_rule": {"default": "uniform", "effective": "family_equal"},
    }


def test_uk_doctrine_with_overrides_refuses_invalid_or_frozen_fields() -> None:
    with pytest.raises(ValueError, match="target_weight_rule"):
        uk_doctrine_with_overrides(target_weight_rule="per_target")
    with pytest.raises(ValueError, match="epochs"):
        uk_doctrine_with_overrides(epochs=0)
    with pytest.raises(ValueError, match="reviewed constants"):
        uk_doctrine_with_overrides(seed=1)
    with pytest.raises(ValueError, match="unknown"):
        uk_doctrine_with_overrides(not_a_field=1)


def test_family_equal_gives_each_family_one_equal_share() -> None:
    weights = uk_national_target_loss_weights(
        ["hmrc"] * 3 + ["obr"], rule="family_equal"
    )
    assert weights is not None
    assert weights.sum() == pytest.approx(1.0)
    # Three hmrc rows share half the objective; the single obr row holds the
    # other half, so an over-supplied family cannot outvote by count.
    assert weights[:3].sum() == pytest.approx(0.5)
    assert weights[3] == pytest.approx(0.5)


def test_uniform_rule_defers_to_the_kernel_default() -> None:
    assert uk_national_target_loss_weights(["hmrc", "obr"], rule="uniform") is None


def test_family_equal_refuses_an_undeclared_family() -> None:
    with pytest.raises(ValueError, match="must declare a family"):
        uk_national_target_loss_weights(["hmrc", ""], rule="family_equal")


def test_national_doctrine_rejects_tampered_bounds() -> None:
    with pytest.raises(ValueError, match="epochs"):
        UKNationalSolveDoctrine(epochs=0)
    with pytest.raises(ValueError, match="learning_rate"):
        UKNationalSolveDoctrine(learning_rate=0.0)
    with pytest.raises(ValueError, match="max_weight_ratio"):
        UKNationalSolveDoctrine(max_weight_ratio=1.0)
    with pytest.raises(ValueError, match="seed"):
        UKNationalSolveDoctrine(seed=-1)
    with pytest.raises(ValueError, match="target_loss_cap"):
        UKNationalSolveDoctrine(target_loss_cap=float("nan"))
    with pytest.raises(ValueError, match="scale_rule"):
        UKNationalSolveDoctrine(scale_rule="bespoke")
    with pytest.raises(ValueError, match="target_weight_rule"):
        UKNationalSolveDoctrine(target_weight_rule="per_target")
    with pytest.raises(ValueError, match="mass_rule"):
        UKNationalSolveDoctrine(mass_rule="conserve")
    with pytest.raises(ValueError, match="l0_lambda"):
        UKNationalSolveDoctrine(l0_lambda=-0.1)


def test_measure_resolution_never_touches_the_source_frame() -> None:
    """Injection lands on adapter table copies, not on the caller's frame.

    A test that inspects the frame after ``restore`` would pass either way,
    so this one holds the property where it actually lives: the source frame
    gains no column while the resolution loop probes round after round.
    """

    from microcosm.build.target_materialization import resolve_target_measures
    from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter

    frame = _materialization_binding_frame()
    before = {entity: list(frame.table(entity).columns) for entity in frame.entities}

    class ProbeResolver:
        contract_targets = {
            "probe.target": {
                "bindings": {
                    "policyengine": {
                        "from_entity": "household",
                        "value_variable": "probe_measure",
                    }
                }
            }
        }

        def knows(self, entity, variable):
            return (entity, variable) == ("household", "probe_measure")

        def compute(self, entity, variable):
            return np.array([1.0, 2.0, 3.0]), "probe"

        def receipt(self):
            return {"provider": "probe"}

    registry = TargetRegistry(
        [
            TargetSpec(
                name="probe.target",
                entity="household",
                measure="probe/measure",
                value=6.0,
                source="test",
                metadata={"contract_target_id": "probe.target"},
            )
        ],
        country="uk",
    )

    resolution = resolve_target_measures(
        lambda: UKFrameTargetAdapter(frame),
        registry,
        ProbeResolver(),
        period=2025,
    )

    assert ("household", "probe_measure") in resolution.measure_inputs
    for entity in frame.entities:
        assert list(frame.table(entity).columns) == before[entity]


def test_published_child_reduction_translates_to_the_age_predicate():
    """A published fact keeps its publisher's semantics; binding it is our job.

    DWP names a dependent-child concept in `any_child_under`. The model carries
    no `is_child` column because it has no need of one — dependency is derived
    from age where it is wanted — so "any child under N" is exactly "any person
    aged under N" here. The translation is declared, not aliased at the call
    site, and refusing the fact instead would drop a real target.
    """

    from microcosm.build.uk_runtime.ledger_targets import (
        UK_TRANSLATED_HOUSEHOLD_REDUCTIONS,
        UKFrameTargetAdapter,
    )

    assert UK_TRANSLATED_HOUSEHOLD_REDUCTIONS["any_child_under"] == "any"

    frame = _materialization_binding_frame()
    adapter = UKFrameTargetAdapter(frame)
    # Households 0 and 2 each carry an infant; household 1 does not.
    adapter.tables["person"]["age"] = [0.0, 40.0, 35.0, 0.0, 38.0, 41.0]
    condition = {
        "variable": "age",
        "entity": "person",
        "reduce": "any_child_under",
        "operator": "<",
        "value": 1,
    }

    translated = adapter.household_condition(condition)

    assert list(translated) == [True, False, True]
    assert list(translated) == list(
        adapter.household_condition({**condition, "reduce": "any"})
    )


def test_unmapped_household_reduction_names_the_published_reducer():
    from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter

    adapter = UKFrameTargetAdapter(_materialization_binding_frame())

    with pytest.raises(ValueError, match="any_grandchild_under"):
        adapter.household_condition(
            {
                "variable": "capital_gains",
                "entity": "person",
                "reduce": "any_grandchild_under",
                "operator": "<",
                "value": 1,
            }
        )


@pytest.mark.parametrize("defect", ["missing", "duplicate"])
def test_packaged_uc_headline_refuses_incomplete_month_cell_fixture(defect) -> None:
    from microcosm.build.ledger_targets import compile_ledger_target_references

    reference = _reference_by_name("dwp.uc.households")
    facts = _facts_for_reference(reference, 20.0)
    if defect == "missing":
        # Keep every month represented but omit one paid/entitlement cell.
        facts.pop(0)
    else:
        # Keep the expected total size and unique IDs; repeat a cell instead.
        facts[0]["dimensions"] = dict(facts[1]["dimensions"])

    with pytest.raises(ValueError, match="monthly window"):
        compile_ledger_target_references(facts, [reference], country="uk")
