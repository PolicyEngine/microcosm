"""Transport source-qualified budget values through their actual graph edges.

The country runner issues and checks the live budget. This source-composition
kernel carries immutable bytes only; it cannot issue authority. Its numeric
output removes source identity fields before the ordinary calibration kernel.
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from microcosm.frame import WeightKind
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Node,
    Numeric,
    Slice,
    source_hash,
)
from microcosm.graph.keys import opaque_artifact_key

from . import graph_combined_clone as clone
from . import graph_survey_calibration as numeric
from . import graph_survey_population as graph
from . import survey_origin_budget as budgets
from .support_provenance import support_clone_index_column

BUDGET_NODE = "survey.sampling_budget"


def _require(condition, reason):
    if not condition:
        raise ValueError("SURVEY_BUDGET_TRANSPORT_" + reason)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _document(payload):
    _require(
        type(payload) is bytes and 0 < len(payload) <= budgets.MAX_PAYLOAD_BYTES,
        "PAYLOAD",
    )
    try:
        document = json.loads(payload)
    except (ValueError, UnicodeError):
        raise ValueError("SURVEY_BUDGET_TRANSPORT_JSON") from None
    _require(type(document) is dict, "DOCUMENT")
    _require(
        document.get("protocol") == budgets.BUDGET_PROTOCOL
        and document.get("release_eligible") is False,
        "PROTOCOL",
    )
    for field in ("preparation_sha256", "allocation_sha256"):
        _require(numeric._digest(document.get(field)), "SOURCE_EDGE_DIGEST")
    return document


def numeric_survey_budget_payload(payload: bytes) -> bytes:
    """Project numbers from bounded bytes; this pure operation authenticates nothing."""
    document = _document(payload)
    ids = document.get("household_ids")
    groups = document.get("group_indices")
    records = document.get("origins")
    _require(
        type(ids) is list
        and 0 < len(ids) <= numeric.MAX_ROWS
        and type(groups) is list
        and len(groups) == len(ids)
        and type(records) is list
        and 0 < len(records) <= budgets.MAX_GROUPS
        and type(document.get("group_count")) is int
        and document["group_count"] == len(records)
        and len(ids) == 2 * len(records),
        "COMPLETE_CLONE_SHAPE",
    )
    _require(
        all(type(i) is int and -(2**63) <= i < 2**63 for i in ids)
        and len(set(ids)) == len(ids),
        "HOUSEHOLD_IDS",
    )
    positions = {i: position for position, i in enumerate(ids)}
    incoming, row_upper = [None] * len(ids), [None] * len(ids)
    upper, seen = [], set()
    for group, record in enumerate(records):
        _require(type(record) is dict, "ORIGIN_RECORD")
        # Validate before duplicating per-origin tokens into per-row output.
        numeric._vector([record.get("design_bound_float64_hex")], 1)
        numeric._vector([record.get("upper_float64_hex")], 1)
        members, weights = (
            record.get("members"),
            record.get("incoming_clone_float64_bytes"),
        )
        _require(
            type(members) is list
            and len(members) == 2
            and type(weights) is list
            and len(weights) == 2,
            "COMPLETE_ROLES",
        )
        roles = set()
        for member, weight_bytes in zip(members, weights, strict=True):
            _require(
                type(member) is list
                and len(member) == 2
                and type(member[0]) is int
                and member[0] in positions
                and member[0] not in seen
                and type(member[1]) is int
                and member[1] in (0, 1),
                "MEMBERSHIP",
            )
            seen.add(member[0])
            roles.add(member[1])
            position = positions[member[0]]
            _require(
                type(groups[position]) is int and groups[position] == group,
                "GROUP_ORDER",
            )
            _require(
                type(weight_bytes) is str
                and len(weight_bytes) == 16
                and all(c in "0123456789abcdef" for c in weight_bytes),
                "WEIGHT_BYTES",
            )
            value = np.frombuffer(bytes.fromhex(weight_bytes), dtype=np.float64)[0]
            incoming[position] = float(value).hex()
            row_upper[position] = record.get("design_bound_float64_hex")
        _require(roles == {0, 1}, "COMPLETE_ROLES")
        upper.append(record.get("upper_float64_hex"))
    _require(seen == set(ids), "COMPLETE_MEMBERSHIP")
    output = graph._bounded_json(
        {
            "protocol": numeric.BOUNDS_PROTOCOL,
            "budget_sha256": _sha(payload),
            "population": clone.COMBINED_CLONE_NODE,
            "household_ids": ids,
            "group_indices": groups,
            "group_upper_hex": upper,
            "row_upper_hex": row_upper,
            "incoming_hex": incoming,
        },
        numeric.MAX_BYTES,
    )
    _require(len(output) <= numeric.MAX_BYTES, "NUMERIC_LIMIT")
    numeric.decode_numeric_survey_bounds(output)
    return output


def survey_sampling_budget_node(*, budget_sha256):
    _require(numeric._digest(budget_sha256), "BUDGET_DIGEST")
    return Node(
        BUDGET_NODE,
        SurveySamplingBudgetKernel.ref,
        population=clone.COMBINED_CLONE_NODE,
        # This provenance input forces the actual ownership claim to precede
        # the budget transport. It is not an imputation feature.
        inputs=(Slice("household", (support_clone_index_column("household"),)),),
        params={"budget_sha256": budget_sha256, "authority": "country_runner_required"},
        artifact_inputs=(
            ArtifactInput(
                "preparation", graph.CREATE_NODE, "preparation", graph.PREPARATION_TYPE
            ),
            ArtifactInput(
                "allocation", graph.ALLOCATION_NODE, "allocation", graph.ALLOCATION_TYPE
            ),
        ),
        artifact_outputs=(
            ArtifactOutput("budget", budgets.BUDGET_TYPE),
            ArtifactOutput("numeric_bounds", numeric.BOUNDS_TYPE),
        ),
    )


class SurveySamplingBudgetKernel(KernelBase):
    """Carry immutable caller values with checked source and clone projections."""

    ref = "us.survey_sampling_budget@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def __init__(self, payload):
        _document(payload)
        self._payload = payload

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            budgets,
            numeric,
            numeric.group_bounds,
            support_clone_index_column,
            graph,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        payload = self._payload
        document = _document(payload)
        expected = survey_sampling_budget_node(budget_sha256=_sha(payload))
        _require(context.node.normative() == expected.normative(), "DECLARATION")
        _require(dict(context.params) == dict(expected.params), "PARAMETERS")
        _require(
            set(context.tables) == set(context.weights) == {"household"}
            and not context.sources
            and set(context.artifacts) == {"preparation", "allocation"},
            "CONTEXT",
        )
        for edge in expected.artifact_inputs:
            value = context.artifacts[edge.name]
            _require(
                value.type == edge.type
                and value.key == opaque_artifact_key(value.producer_key, edge.artifact)
                and _sha(value.payload) == document[edge.name + "_sha256"],
                "SOURCE_EDGE",
            )
        numbers_raw = numeric_survey_budget_payload(payload)
        numbers = numeric.decode_numeric_survey_bounds(numbers_raw)
        household = context.tables["household"]
        role_column = support_clone_index_column("household")
        _require(set(household.columns) == {"household_id", role_column}, "PROJECTION")
        _require(
            tuple(household.household_id) == numbers.grouped.household_ids,
            "ORDERED_IDS",
        )
        roles = {
            member[0]: member[1]
            for record in document["origins"]
            for member in record["members"]
        }
        _require(
            tuple(household[role_column])
            == tuple(roles[i] for i in household.household_id),
            "CLONE_ROLES",
        )
        weights = context.weights["household"]
        _require(
            weights.kind is WeightKind.IMPORTANCE
            and weights.values.dtype == numbers.incoming.dtype
            and weights.values.tobytes() == numbers.incoming.tobytes(),
            "ACTUAL_INCOMING_WEIGHTS",
        )
        result = KernelResult(
            artifacts={"budget": payload, "numeric_bounds": numbers_raw},
            receipt={
                "budget_sha256": _sha(payload),
                "numeric_bounds_sha256": _sha(numbers_raw),
                "constraint_digest": numbers.grouped.digest,
                "group_count": numbers.grouped.group_count,
                "release_eligible": False,
                "source_admission": "required_from_country_runner",
            },
        )
        _require(
            type(self._payload) is bytes and self._payload == payload, "FINAL_PAYLOAD"
        )
        return result
