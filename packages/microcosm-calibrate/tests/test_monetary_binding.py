"""Compiler integrity tests that do not depend on microcosm-build."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.calibrate import Target, TargetSet, build_constraint_matrix
from microcosm.calibrate.monetary_binding import (
    MonetaryBindingIntegrityError,
    monetary_digest,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


def _frame(values=(2.0, 4.0), ids=(1, 2)):
    return Frame(
        {
            "person": pd.DataFrame({"person_id": [11, 12], "person_household_id": ids}),
            "household": pd.DataFrame({"household_id": ids, "deposits": values}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(len(ids)), WeightKind.DESIGN)},
    )


def _target(values=(2.0, 4.0), ids=(1, 2)):
    source_metadata = {
        "activation_status": "requires_prepared_measure",
        "monetary_target_role": "calibration",
    }
    target = Target(
        "synthetic/deposits",
        "household",
        "deposits",
        6,
        2024,
        tolerance=0.01,
        source="Synthetic",
        metadata={
            "activation_status": "active",
            "monetary_source_activation_status": "requires_prepared_measure",
            "monetary_target_role": "calibration",
        },
    )
    prepared = {
        "schema_version": 2,
        "basis": {
            "currency": "XXX",
            "unit": "base_currency",
            "period": "2024-12-31",
            "temporal_basis": "closing_stock",
            "sector": "S14",
            "perimeter": "deposits",
            "valuation": "nominal",
        },
        "n_records": len(values),
        "source_identity_sha256": "b" * 64,
        "source_values_sha256": "c" * 64,
        "values_sha256": hashlib.sha256(
            np.asarray(values, dtype="<f8").tobytes()
        ).hexdigest(),
        "record_ids_sha256": hashlib.sha256(
            np.asarray(ids, dtype="<i8").tobytes()
        ).hexdigest(),
        "bridge": {
            "factor": 1.0,
            "description": "Synthetic bridge",
            "source_sha256": "d" * 64,
        },
        "readiness": "ready",
        "measure_kind": "direct",
    }
    prepared["bridge_sha256"] = monetary_digest(prepared["bridge"])
    prepared["receipt_sha256"] = monetary_digest(prepared)
    source_reference = {
        "name": target.name,
        "ledger_fact_key": "",
        "ledger_source_record_id": "synthetic.fact",
        "ledger_selector": {},
        "value_operation": "identity",
        "entity": target.entity,
        "measure": target.measure,
        "filter": target.filter,
        "period": target.period,
        "source": target.source,
        "family": "synthetic",
        "signed": False,
        "se": None,
        "tolerance": target.tolerance,
        "notes": "",
        "metadata": source_metadata,
        "assertion_policy": "observed_only",
        "period_match_policy": "latest_not_after",
        "uprating_index": None,
        "uprating_from_period": None,
        "uprating_to_period": None,
    }
    binding = {
        "reference": {
            **source_reference,
            "metadata": target.metadata,
        },
        "source_reference": source_reference,
        "value": target.value,
        "source": target.source,
        "source_assertion": "observed",
        "source_identity_sha256": "a" * 64,
        "prepared": prepared,
    }
    return replace(
        target,
        metadata={
            **target.metadata,
            "monetary_binding": json.dumps(binding),
            "monetary_binding_sha256": monetary_digest(binding),
        },
    )


def test_matching_receipt_compiles_and_stale_values_fail_whole_compilation():
    assert build_constraint_matrix(_frame(), TargetSet((_target(),))).n_targets == 1
    with pytest.raises(MonetaryBindingIntegrityError, match="prepared values differ"):
        build_constraint_matrix(_frame((20, 40)), TargetSet((_target(),)))


@pytest.mark.parametrize("key", ["monetary_binding", "monetary_binding_sha256"])
def test_partial_receipt_cannot_be_used_as_an_opt_out(key):
    target = _target()
    metadata = dict(target.metadata)
    del metadata[key]
    with pytest.raises(MonetaryBindingIntegrityError):
        build_constraint_matrix(
            _frame(), TargetSet((replace(target, metadata=metadata),))
        )


def test_complete_receipt_removal_cannot_bypass_remaining_monetary_markers():
    target = _target()
    metadata = {
        key: value
        for key, value in target.metadata.items()
        if key not in {"monetary_binding", "monetary_binding_sha256"}
    }
    with pytest.raises(MonetaryBindingIntegrityError, match="markers remain"):
        build_constraint_matrix(
            _frame((20, 40)),
            TargetSet((replace(target, metadata=metadata),)),
        )


def test_receipt_carried_tolerance_cannot_drift():
    with pytest.raises(MonetaryBindingIntegrityError, match="tolerance differs"):
        build_constraint_matrix(
            _frame(),
            TargetSet((replace(_target(), tolerance=1e30),)),
        )


def test_source_reference_must_match_the_activated_reference():
    target = _target()
    binding = json.loads(target.metadata["monetary_binding"])
    binding["source_reference"]["family"] = "different"
    metadata = {
        **target.metadata,
        "monetary_binding": json.dumps(binding),
        "monetary_binding_sha256": monetary_digest(binding),
    }
    with pytest.raises(MonetaryBindingIntegrityError, match="source reference"):
        build_constraint_matrix(
            _frame(),
            TargetSet((replace(target, metadata=metadata),)),
        )


def test_schema_two_receipt_rejects_extra_keys_even_when_rehashed():
    target = _target()
    binding = json.loads(target.metadata["monetary_binding"])
    binding["ignored_future_semantics"] = "must bump the schema instead"
    metadata = {
        **target.metadata,
        "monetary_binding": json.dumps(binding),
        "monetary_binding_sha256": monetary_digest(binding),
    }
    with pytest.raises(MonetaryBindingIntegrityError, match="binding keys differ"):
        build_constraint_matrix(
            _frame(),
            TargetSet((replace(target, metadata=metadata),)),
        )


def test_reordered_ids_fail_even_when_values_are_equal():
    with pytest.raises(MonetaryBindingIntegrityError, match="IDs/order"):
        build_constraint_matrix(_frame((2, 2)), TargetSet((_target((2, 2), (2, 1)),)))


def test_ordinary_targets_keep_their_existing_skip_behavior():
    valid = Target("ordinary", "household", "deposits", 6)
    missing = Target("missing", "household", "absent", 1)
    problem = build_constraint_matrix(_frame(), TargetSet((valid, missing)))
    assert problem.n_targets == 1
    assert [skipped.target.name for skipped in problem.skipped] == ["missing"]
