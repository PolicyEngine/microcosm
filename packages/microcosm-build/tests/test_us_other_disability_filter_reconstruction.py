"""Literal structural reconstruction checks; no source owners or model fitting."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_us_other_disability_host as host
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import KernelResult, Node, NodeRejectedError, StructuralDelta


def _incoming():
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.array([10, 20, 30], dtype=np.int64),
                    "person_household_id": np.array([100, 100, 200], dtype=np.int64),
                    "amount": np.array([np.nan, 0.0, 20.0], dtype="float64"),
                },
                index=[3, 8, 13],
            ),
            "household": pd.DataFrame(
                {"household_id": np.array([100, 200], dtype=np.int64)}, index=[4, 9]
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array([1.5, 4.0]), WeightKind.DESIGN)},
        pd.Series(["a", "b", "b"], index=[3, 8, 13], name="stratum"),
    )
    return host.population_ops.Population.from_frame(frame, "invented.source")


def _node(incoming, *, keep_all):
    return Node(
        host.fragment.VERSION_NODE if keep_all else host.fragment.DONOR_NODE,
        "invented.filter@1",
        base=incoming.version,
        structural=StructuralDelta.FILTER,
        mass="conserve" if keep_all else "free",
    )


@pytest.mark.parametrize("keep_all", [False, True])
@pytest.mark.parametrize("dtype", ["bool", "boolean"])
def test_both_host_filters_materialize_exact_executor_selection(keep_all, dtype):
    incoming = _incoming()
    before = host.physical._population_stamp(incoming)
    node = _node(incoming, keep_all=keep_all)
    # Deliberately scrambled labels and non-contiguous table indices. The donor
    # subset retains both households, with person 10 as the excluded observation.
    raw = KernelResult(
        keep=pd.Series([True, keep_all, True], index=[30, 10, 20], dtype=dtype),
        artifacts={"proof": b"invented artifact"},
        receipt={"scope": "invented structural result"},
    )
    mask_before = raw.keep.copy(deep=True)
    materialized = host._materialize_filter_result(incoming, node, raw)
    assert materialized.frame is not None and materialized.keep is None
    assert materialized.artifacts is raw.artifacts
    assert materialized.receipt is raw.receipt
    assert materialized.columns is raw.columns
    assert materialized.weights is raw.weights
    assert raw.frame is None
    pd.testing.assert_series_equal(raw.keep, mask_before)
    actual = host.population_ops.patch(incoming, node, materialized)
    expected = host.executor._apply_result(node, raw, incoming)
    host.physical.replay.same_replayed_population(actual, expected)
    person_ids = [10, 20, 30] if keep_all else [20, 30]
    assert actual.frame.person.person_id.tolist() == person_ids
    pd.testing.assert_frame_equal(
        actual.frame.person,
        incoming.frame.person.loc[[3, 8, 13] if keep_all else [8, 13]],
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        actual.frame.table("household"),
        incoming.frame.table("household").reset_index(drop=True),
        check_exact=True,
    )
    np.testing.assert_array_equal(
        actual.frame.weights_for("household").values,
        incoming.frame.weights_for("household").values,
    )
    assert actual.frame.weights_for("household").kind is WeightKind.DESIGN
    pd.testing.assert_series_equal(
        actual.frame.strata,
        incoming.frame.strata.loc[[3, 8, 13] if keep_all else [8, 13]],
    )
    assert actual.mass_ledger[:-1] == incoming.mass_ledger
    record = actual.mass_ledger[-1]
    assert record.node_id == node.id and record.policy == node.mass
    assert record.before_total == 7.0
    assert record.after_total == (7.0 if keep_all else 5.5)
    if keep_all:
        assert record.before_by_stratum == record.after_by_stratum
    assert actual.version == node.id
    assert host.physical._population_stamp(incoming) == before


@pytest.mark.parametrize(
    "mask,error_type",
    [
        (None, NodeRejectedError),
        (pd.Series(False, index=[10, 20, 30], dtype=bool), ValueError),
        (pd.Series([True, True], index=[10, 20]), NodeRejectedError),
        (pd.Series([True, True, True], index=[10, 10, 30]), NodeRejectedError),
        (pd.Series([True, True, True], index=[10, 20, 40]), NodeRejectedError),
        (
            pd.Series([True, pd.NA, True], index=[10, 20, 30], dtype="boolean"),
            NodeRejectedError,
        ),
        (pd.Series([1, 0, 1], index=[10, 20, 30]), NodeRejectedError),
    ],
)
def test_malformed_filter_masks_refuse_without_mutation(mask, error_type):
    incoming = _incoming()
    before = host.physical._population_stamp(incoming)
    with pytest.raises(error_type):
        host._materialize_filter_result(
            incoming, _node(incoming, keep_all=False), KernelResult(keep=mask)
        )
    assert host.physical._population_stamp(incoming) == before


@pytest.mark.parametrize("defect", ["node", "structure", "base", "frame"])
def test_filter_materialization_is_limited_to_declared_host_filters(defect):
    incoming = _incoming()
    node = _node(incoming, keep_all=True)
    raw = KernelResult(keep=pd.Series(True, index=[10, 20, 30], dtype=bool))
    if defect == "node":
        node = replace(node, id="invented.other")
    elif defect == "structure":
        node = replace(
            node,
            base=None,
            population=incoming.version,
            structural=StructuralDelta.NONE,
        )
    elif defect == "base":
        node = replace(node, base="invented.other")
    else:
        raw = replace(raw, frame=incoming.frame)
    with pytest.raises(ValueError, match="HOST_FILTER_DECLARATION"):
        host._materialize_filter_result(incoming, node, raw)
