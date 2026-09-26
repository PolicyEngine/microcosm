"""Bind the canonical state input to existing post-clone atomic geography.

This nonstructural fragment converts a derived code; it never assigns geography.
The maintained host owns live source admission and replay verification. Typed
gate bytes and detached kernel results do not issue source or release authority.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from microcosm.build.graph_atomic_geography import ATOMIC_GEOGRAPHY_VALIDATION_TYPE
from microcosm.calibrate import geography_constants
from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_POSTAL
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    ArtifactValue,
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Node,
    Numeric,
    Owned,
    Slice,
    source_hash,
)
from microcosm.graph import population as population_ops
from microcosm.graph.canonical import canonical_json

NODE = "survey_geography.bind_state_fips"
REF = "us.survey_geography.bind_state_fips@1"
PROTOCOL = "microcosm.us.current-survey-state-binding.v1"
INPUTS = ("assigned_state_fips", "survey_observed_state")
OUTPUT = "state_fips"
BINDING_TYPE = ArtifactType("microcosm.us.current_survey_state_binding", 1)


def _require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_SURVEY_STATE_" + reason)


def bind_state_fips(households):
    """Exact representation change on supplied values, without source authority."""
    _require(
        type(households) is pd.DataFrame
        and households.columns.is_unique
        and {"household_id", *INPUTS} <= set(households),
        "COLUMNS",
    )
    ids = households.household_id
    _require(
        ids.dtype == np.dtype("int64")
        and ids.is_unique
        and len(ids) > 0
        and ids.ge(0).all(),
        "HOUSEHOLD_AXIS",
    )
    assigned, observed = (households[name] for name in INPUTS)
    for values in (assigned, observed):
        _require(
            values.dtype == population_ops.dtype_for_token("string")
            and all(
                type(value) is str and value in US_STATE_FIPS_TO_POSTAL
                for value in values
            ),
            "STATE_DOMAIN",
        )
    _require(assigned.equals(observed), "OBSERVED_STATE_MISMATCH")
    # Exact two-character literals enter int directly. No numeric coercion,
    # default state, block parsing, lookup reconstruction, or float intermediary.
    dtype = _output_dtype(households)
    return pd.Series(
        [int(value) for value in assigned],
        index=pd.Index(ids.to_numpy(copy=True), name="household_id"),
        name=OUTPUT,
        dtype=dtype,
    )


def _output_dtype(households):
    if OUTPUT not in households:
        return "int64"
    dtype = population_ops.token_for_dtype(households[OUTPUT].dtype)
    _require(dtype in ("int64", "Int64"), "INCUMBENT_DTYPE")
    return dtype


def _node(households, *, population, after):
    _require(
        type(population) is str and population and population != NODE, "POPULATION"
    )
    _require(after is None or type(after) is ArtifactInput, "AFTER")
    bind_state_fips(households)
    edges = (
        ArtifactInput(
            "geography_validation",
            "geography.gate",
            "validation",
            ATOMIC_GEOGRAPHY_VALIDATION_TYPE,
        ),
    )
    if after is not None:
        _require(after.name != edges[0].name and after.producer != NODE, "AFTER")
        edges += (after,)
    return Node(
        NODE,
        REF,
        population=population,
        inputs=(Slice("household", INPUTS),),
        outputs=(
            Owned(
                "household",
                OUTPUT,
                _output_dtype(households),
                rewrite=OUTPUT in households,
            ),
        ),
        params={"protocol": PROTOCOL},
        artifact_inputs=edges,
        artifact_outputs=(ArtifactOutput("binding", BINDING_TYPE),),
        description="Bind integer state_fips from the existing block-derived state, requiring equality with qualified observed survey state; replace any carried canonical value and preserve all other cells.",
    )


def state_binding_node(receiving, *, population, after=None):
    """Declare the alias after the existing geography gate and optional host edge.

    The caller must provide the genuine receiving population through its own
    retained owner. This declaration checks storage and cannot qualify ancestry.
    """
    return _node(receiving.table("household"), population=population, after=after)


def state_binding_result(node, households, artifacts):
    """Recompute the exact canonical leaf; detached results remain descriptive."""
    after = node.artifact_inputs[1] if len(node.artifact_inputs) == 2 else None
    expected = _node(households, population=node.population, after=after)
    _require(node == expected, "DECLARATION")
    _require(
        set(artifacts) == {edge.name for edge in node.artifact_inputs},
        "ARTIFACT_ROSTER",
    )
    for edge in node.artifact_inputs:
        value = artifacts[edge.name]
        _require(
            type(value) is ArtifactValue
            and value.type == edge.type
            and type(value.payload) is bytes,
            "ARTIFACT_TYPE",
        )
    gate = artifacts["geography_validation"].payload
    document = json.loads(gate)
    _require(
        type(document) is dict
        and canonical_json(document) == gate
        and document.get("outcome") == "pass"
        and document.get("scope") == "atomic_geography_mapping_integrity",
        "GEOGRAPHY_GATE",
    )
    # The gate may precede later support-preserving graph fragments. Its bytes
    # order the graph; they do not authenticate the receiving rows or ancestry.
    values = bind_state_fips(households)
    receipt = {
        "protocol": PROTOCOL,
        "input": INPUTS[0],
        "output": OUTPUT,
        "observed_state_equality": True,
        "incumbent_policy": "replace_from_qualified_derived_state",
        "households": len(values),
        "new_geography_assignment": False,
        "source_admission_issued": False,
        "release_eligible": False,
    }
    return KernelResult(
        columns={("household", OUTPUT): values},
        artifacts={"binding": canonical_json(receipt)},
        receipt=receipt,
    )


class CurrentSurveyStateKernel(KernelBase):
    """Pure representation binding; the host retains source and population proof."""

    ref = REF
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            geography_constants,
            population_ops,
            canonical_json,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        _require(
            set(context.tables) == {"household"}
            and len(context.node.outputs) == 1
            and tuple(context.tables["household"].columns)
            == (
                "household_id",
                *INPUTS,
                *((OUTPUT,) if context.node.outputs[0].rewrite else ()),
            )
            and not context.sources
            and dict(context.params) == dict(context.node.params),
            "CONTEXT",
        )
        return state_binding_result(
            context.node, context.tables["household"], context.artifacts
        )
