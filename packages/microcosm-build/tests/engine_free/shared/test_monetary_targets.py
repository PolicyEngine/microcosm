"""Synthetic contracts for exact monetary bindings."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.ledger_targets import LedgerTargetReference
from microcosm.build.monetary_targets import (
    MonetaryBasis,
    bind_monetary_target,
    prepare_monetary_measure,
)
from microcosm.calibrate import TargetRegistry, build_constraint_matrix
from microcosm.calibrate.monetary_binding import MonetaryBindingIntegrityError
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


def _basis(**changes):
    fields = {
        "currency": "XXX",
        "unit": "base_currency",
        "period": "2024-12-31",
        "temporal_basis": "closing_stock",
        "sector": "S14",
        "perimeter": "resident household deposits",
        "valuation": "nominal",
    }
    return MonetaryBasis(**(fields | changes))


def _prepared(values=(1.0, 2.0), **changes):
    fields = {
        "record_ids": np.arange(1, len(values) + 1),
        "basis": _basis(),
        "factor": 2.0,
        "source_identity_sha256": "a" * 64,
        "bridge_description": "Synthetic explicit level bridge.",
        "bridge_source_sha256": "b" * 64,
    }
    return prepare_monetary_measure(values, **(fields | changes))


def _reference(**changes):
    fields = {
        "name": "synthetic/deposits",
        "ledger_source_record_id": "synthetic.fact",
        "entity": "household",
        "measure": "deposits",
        "period": 2024,
        "family": "financial_assets",
        "source": "Synthetic observation",
    }
    return LedgerTargetReference(**(fields | changes))


def _bind(**changes):
    fields = {
        "value": 6,
        "source_basis": _basis(),
        "source_assertion": "observation",
        "source_identity_sha256": "c" * 64,
        "prepared": _prepared(),
    }
    return bind_monetary_target(_reference(), **(fields | changes))


def test_direct_binding_receipts_exact_basis_values_and_entity_ids():
    target = _bind()
    receipt = json.loads(target.metadata["monetary_binding"])
    assert target.value == 6
    assert target.measure == "deposits"
    assert receipt["prepared"]["basis"]["period"] == "2024-12-31"
    assert receipt["prepared"]["values_sha256"] == _prepared().values_sha256
    assert receipt["source_identity_sha256"] == "c" * 64
    assert receipt["prepared"]["source_identity_sha256"] == "a" * 64
    assert len(target.metadata["monetary_binding_sha256"]) == 64
    assert target.metadata["activation_status"] == "active"
    assert target.metadata["monetary_source_activation_status"] == "ready"


def test_binding_preserves_existing_target_role_without_repurposing_it():
    target = bind_monetary_target(
        _reference(metadata={"target_role": "federal_income_tax_total"}),
        value=6,
        source_basis=_basis(),
        source_assertion="observation",
        source_identity_sha256="c" * 64,
        prepared=_prepared(),
    )
    assert target.metadata["target_role"] == "federal_income_tax_total"
    assert target.metadata["monetary_target_role"] == "calibration"


def test_binding_rejects_predeclared_generated_receipt_metadata():
    with pytest.raises(ValueError, match="may not predeclare"):
        bind_monetary_target(
            _reference(metadata={"monetary_binding_sha256": "d" * 64}),
            value=6,
            source_basis=_basis(),
            source_assertion="observation",
            source_identity_sha256="c" * 64,
            prepared=_prepared(),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"currency": "USD"},
        {"period": "2023-12-31"},
        {"sector": "S15"},
        {"perimeter": "all deposits"},
        {"valuation": "market"},
    ],
)
def test_any_accounting_basis_difference_is_rejected(change):
    with pytest.raises(ValueError, match="basis mismatch"):
        _bind(source_basis=_basis(**change))


def test_annual_flow_binding_accepts_only_the_exact_annual_basis():
    annual = _basis(period="2024", temporal_basis="annual_flow")
    target = _bind(source_basis=annual, prepared=_prepared(basis=annual))
    assert target.period == 2024


def test_prepared_values_are_copied_immutable_and_fail_closed_on_metadata_change():
    values = np.array([1.0, 2.0])
    prepared = _prepared(values)
    values[:] = 0
    np.testing.assert_array_equal(prepared.values, [2, 4])
    with pytest.raises(ValueError):
        prepared.values.setflags(write=True)
    prepared.values.shape = (2, 1)
    with pytest.raises(ValueError, match="metadata was mutated"):
        prepared.receipt()


def test_checked_weighted_total_rejects_misaligned_or_negative_weights():
    assert _prepared().total([3, 4]) == 22
    with pytest.raises(ValueError, match="weights"):
        _prepared().total([-1, 1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_assertion": "forecast"},
        {"prepared": _prepared(readiness="unbound")},
        {"prepared": _prepared(measure_kind="policy_derived")},
        {"reference": _reference(filter="included")},
    ],
)
def test_unready_indirect_or_filtered_measures_never_bind(kwargs):
    if "reference" in kwargs:
        reference = kwargs.pop("reference")
        with pytest.raises(ValueError):
            bind_monetary_target(
                reference,
                value=6,
                source_basis=_basis(),
                source_assertion="observation",
                source_identity_sha256="c" * 64,
                prepared=_prepared(),
            )
    else:
        with pytest.raises(ValueError):
            _bind(**kwargs)


def test_bound_target_uses_existing_sparse_compiler_api():
    prepared = _prepared()
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": [1, 2], "person_household_id": [1, 2]}
            ),
            "household": pd.DataFrame(
                {"household_id": [1, 2], "deposits": prepared.values}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array([3.0, 4.0]), WeightKind.DESIGN)},
    )
    problem = build_constraint_matrix(
        frame, TargetRegistry([_bind()], country="xx").to_target_set()
    )
    np.testing.assert_array_equal(problem.estimates(np.array([3.0, 4.0])), [22])
    with pytest.raises(MonetaryBindingIntegrityError, match="prepared values differ"):
        stale = Frame(
            {
                "person": pd.DataFrame(
                    {"person_id": [1, 2], "person_household_id": [1, 2]}
                ),
                "household": pd.DataFrame(
                    {"household_id": [1, 2], "deposits": [20, 40]}
                ),
            },
            EntitySchema(group_entities=("household",)),
            {"household": Weights(np.array([3.0, 4.0]), WeightKind.DESIGN)},
        )
        build_constraint_matrix(
            stale,
            TargetRegistry([_bind()], country="xx").to_target_set(),
        )


def test_person_receipt_refuses_unreceipted_household_weight_collapse():
    prepared = _prepared(record_ids=[11, 12])
    target = bind_monetary_target(
        _reference(entity="person"),
        value=6,
        source_basis=_basis(),
        source_assertion="observation",
        source_identity_sha256="c" * 64,
        prepared=prepared,
    )
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [11, 12],
                    "person_household_id": [1, 1],
                    "deposits": prepared.values,
                }
            ),
            "household": pd.DataFrame({"household_id": [1]}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array([3.0]), WeightKind.DESIGN)},
    )
    with pytest.raises(MonetaryBindingIntegrityError, match="cross-entity"):
        build_constraint_matrix(
            frame, TargetRegistry([target], country="xx").to_target_set()
        )
