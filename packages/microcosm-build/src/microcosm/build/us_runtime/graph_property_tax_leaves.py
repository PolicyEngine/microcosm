"""Strict deterministic property splits and a numerical completeness gate.

The host owns source qualification, clone pairing and full-Frame lifetime checks.
These operations grant no source or release authority and never complete unknown
components. The receiving FILTER opens a real population version and ledger entry.
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.frame import Frame
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelResult,
    KernelRole,
    Node,
    Numeric,
    Owned,
    Slice,
    StructuralDelta,
    source_hash,
)
from microcosm.graph.keys import opaque_artifact_key

from . import cps_carried
from .property_income_constants import PROPERTY_COMPONENTS

PROTOCOL = "microcosm.us.property-tax-leaves.v1"
RECEIVING_NODE = "survey_property.tax_receiving"
TAX_LEAVES_NODE = "survey_property.tax_leaves"
GATE_NODE = "survey_property.tax_leaf_gate"
TAX_LEAF_COLUMNS = (
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
)
DIAGNOSTICS_TYPE = ArtifactType("microcosm.us.property_tax_leaf_diagnostics", 1)
VERIFICATION_TYPE = ArtifactType("microcosm.us.property_tax_leaf_verification", 1)
_INPUTS = (PROPERTY_COMPONENTS[0], PROPERTY_COMPONENTS[2], PROPERTY_COMPONENTS[1])


def _require(condition, reason):
    if not condition:
        raise ValueError("PROPERTY_TAX_LEAVES_" + reason)


def _parameters(atol, rtol):
    _require(
        all(type(v) is float and np.isfinite(v) and v >= 0 for v in (atol, rtol)),
        "TOLERANCES",
    )
    fractions = (
        cps_carried.TAXABLE_INTEREST_FRACTION,
        cps_carried.QUALIFIED_DIVIDEND_FRACTION,
    )
    _require(
        all(type(v) is float and np.isfinite(v) and 0 < v < 1 for v in fractions),
        "FRACTIONS",
    )
    return {
        "protocol": PROTOCOL,
        "taxable_interest_fraction": fractions[0],
        "qualified_dividend_fraction": fractions[1],
        "complements": "total_minus_computed_primary",
        "unknown": "nan",
        "retirement_interest": "auxiliary_unchanged",
        "atol": atol,
        "rtol": rtol,
    }


def _person(person):
    _require(
        type(person) is pd.DataFrame
        and person.columns.is_unique
        and {"person_id", *_INPUTS} <= set(person),
        "PERSON_COLUMNS",
    )
    _require(
        person.person_id.dtype == np.dtype("int64") and person.person_id.is_unique,
        "PERSON_IDENTITY",
    )
    for name in _INPUTS:
        _require(person[name].dtype == np.dtype("float64"), "INPUT_DTYPE:" + name)
        values = person[name].to_numpy(copy=False)
        _require(not np.isinf(values).any(), "INFINITE_COMPONENT:" + name)
        if name != PROPERTY_COMPONENTS[1]:
            _require(not (values < 0).any(), "NEGATIVE_COMPONENT:" + name)
    return pd.Index(person.person_id.to_numpy(copy=True), name="person_id")


def split_property_tax_leaves(person: pd.DataFrame) -> pd.DataFrame:
    """Split individually known O/D amounts; retain NaN independently per family.

    Returned float64 columns are keyed by exact integer person IDs. Source age,
    net anchors, donor eligibility, retirement earnings, and old tax predictions
    never establish knownness or select the amount of either split.
    """
    index = _person(person)
    params = _parameters(0.0, 0.0)
    columns = {}
    for component, primary, complement, fraction in (
        (
            PROPERTY_COMPONENTS[0],
            *TAX_LEAF_COLUMNS[:2],
            params["taxable_interest_fraction"],
        ),
        (
            PROPERTY_COMPONENTS[2],
            *TAX_LEAF_COLUMNS[2:],
            params["qualified_dividend_fraction"],
        ),
    ):
        amount = person[component].to_numpy(copy=False)
        known = np.isfinite(amount)
        first = np.full(len(person), np.nan, dtype=np.float64)
        second = first.copy()
        first[known] = amount[known] * fraction
        second[known] = amount[known] - first[known]
        _require(
            np.isfinite(first[known]).all() and np.isfinite(second[known]).all(),
            "UNSTABLE_SPLIT",
        )
        columns[primary], columns[complement] = first, second
    return pd.DataFrame(columns, index=index)


def _slices(frame):
    _require(type(frame) is Frame and frame.schema.person_entity == "person", "FRAME")
    _require(not frame.links, "UNSUPPORTED_LINKS")
    _person(frame.person)
    _require(len(frame.person) > 0, "EMPTY_RECEIVING_FRAME")
    result = []
    for entity in frame.entities:
        table = frame.table(entity)
        columns = tuple(c for c in table if not c.endswith("_id"))
        _require(bool(columns), "NO_READABLE_ENTITY_COLUMN:" + entity)
        if entity != "person":
            _require(
                type(table.index) is pd.RangeIndex
                and table.index.equals(pd.RangeIndex(len(table)))
                and table.index.name is None,
                "UNSUPPORTED_GROUP_INDEX:" + entity,
            )
            ids = table[frame.schema.entity_id_column(entity)]
            members = frame.person[frame.schema.membership_column(entity)]
            _require(set(ids) == set(members), "ORPHAN_GROUP:" + entity)
        result.append(Slice(entity, columns))
    for name in TAX_LEAF_COLUMNS:
        _require(
            name in frame.person and frame.person[name].dtype == np.dtype("float64"),
            "INCUMBENT_LEAF:" + name,
        )
    return tuple(result)


def _nodes(base, slices, projection, reconciliation, atol, rtol):
    _require(
        type(projection) is type(reconciliation) is ArtifactInput
        and projection.name == "projection"
        and reconciliation.name == "reconciliation",
        "ARTIFACT_EDGES",
    )
    params = _parameters(atol, rtol)
    edges = (projection, reconciliation)
    receiving = Node(
        RECEIVING_NODE,
        PropertyTaxReceivingKernel.ref,
        base=base,
        structural=StructuralDelta.FILTER,
        inputs=slices,
        params={"protocol": PROTOCOL, "mode": "all_rows_supported_shape"},
        artifact_inputs=edges,
        description="Open a tax-rebase version with all original rows; refuse orphan groups, links and group-index normalization.",
    )
    rebase = Node(
        TAX_LEAVES_NODE,
        PropertyTaxLeavesKernel.ref,
        population=RECEIVING_NODE,
        inputs=(Slice("person", (*_INPUTS, *TAX_LEAF_COLUMNS)),),
        outputs=tuple(
            Owned("person", c, "float64", rewrite=True) for c in TAX_LEAF_COLUMNS
        ),
        params=params,
        artifact_inputs=edges,
        artifact_outputs=(ArtifactOutput("diagnostics", DIAGNOSTICS_TYPE),),
        description="Rebase four tax leaves from individually known ordinary interest and dividends; preserve unknowns and retirement-account earnings.",
    )
    gate = Node(
        GATE_NODE,
        PropertyTaxLeafGateKernel.ref,
        population=RECEIVING_NODE,
        inputs=(Slice("person", (*_INPUTS, *TAX_LEAF_COLUMNS)),),
        params=params,
        artifact_inputs=(
            *edges,
            ArtifactInput("rebase", TAX_LEAVES_NODE, "diagnostics", DIAGNOSTICS_TYPE),
        ),
        artifact_outputs=(ArtifactOutput("verification", VERIFICATION_TYPE),),
        description="Recompute all four leaves, partition sums and individual knownness; missing member inputs keep completeness false.",
    )
    return receiving, rebase, gate


def property_tax_leaf_nodes(
    frame: Frame,
    *,
    population: str,
    projection: ArtifactInput,
    reconciliation: ArtifactInput,
    atol: float,
    rtol: float,
) -> tuple[Node, Node, Node]:
    """Declare an explicit receiving version, four rewrites and numeric gate.

    The host must bind/recheck this complete input Frame. FILTER-all cannot
    preserve orphan groups, link tables or nondefault group indices; refuse
    those shapes before execution. Group tables need a declared non-ID column
    so the receiving kernel can check their actual IDs through KernelContext.
    """
    return _nodes(population, _slices(frame), projection, reconciliation, atol, rtol)


def _bindings(context):
    _require(
        set(context.artifacts) == {e.name for e in context.node.artifact_inputs},
        "ARTIFACT_ROSTER",
    )
    result = {}
    for edge in context.node.artifact_inputs:
        value = context.artifacts[edge.name]
        _require(
            value.type == edge.type
            and value.key == opaque_artifact_key(value.producer_key, edge.artifact),
            "ARTIFACT_BINDING",
        )
        if edge.name != "rebase":
            result[edge.name] = {
                "producer": edge.producer,
                "producer_key": value.producer_key,
                "artifact_key": value.key,
                "payload_sha256": hashlib.sha256(value.payload).hexdigest(),
                "type": (edge.type.name, edge.type.schema_version),
            }
    return result


def _diagnostics(person, split, params, bindings):
    interest = np.isfinite(person[PROPERTY_COMPONENTS[0]].to_numpy())
    dividend = np.isfinite(person[PROPERTY_COMPONENTS[2]].to_numpy())
    return {
        "protocol": PROTOCOL,
        "parameters": dict(params),
        "input_artifacts": bindings,
        "person_ids": [str(value) for value in person.person_id],
        "interest_known": interest.tolist(),
        "dividend_known": dividend.tolist(),
        "known_leaf_origin": "derived_maintained_fraction_and_subtraction_complement",
        "unknown_leaf_origin": "unknown_component_no_completion",
        "leaf_knownness_groups": (TAX_LEAF_COLUMNS[:2], TAX_LEAF_COLUMNS[2:]),
        "input_sha256": {
            c: hashlib.sha256(
                person[c].to_numpy().astype("<f8", copy=False).tobytes()
            ).hexdigest()
            for c in _INPUTS
        },
        "leaf_sha256": {
            c: hashlib.sha256(
                split[c].to_numpy().astype("<f8", copy=False).tobytes()
            ).hexdigest()
            for c in TAX_LEAF_COLUMNS
        },
        "rows": len(person),
        "unknown_interest_persons": int((~interest).sum()),
        "unknown_dividend_persons": int((~dividend).sum()),
        "complete": bool(len(person) and (interest & dividend).all()),
        "required_scope": "every_physical_person_including_zero_weight_and_dependents",
        "clone_pairing": "host_required",
        "source_authority": False,
        "whole_frame_verification": "host_required",
    }


class _Kernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            cps_carried,
            dependencies=self.capabilities.dependencies,
        )

    def _check(self, context, gate):
        _require(set(context.tables) == {"person"} and not context.sources, "CONTEXT")
        edges = {e.name: e for e in context.node.artifact_inputs}
        _require({"projection", "reconciliation"} <= set(edges), "ARTIFACT_EDGES")
        expected = _nodes(
            "unused_base",
            (),
            edges["projection"],
            edges["reconciliation"],
            context.params.get("atol"),
            context.params.get("rtol"),
        )[2 if gate else 1]
        _require(
            context.node.normative() == expected.normative()
            and dict(context.params) == dict(expected.params),
            "DECLARATION",
        )
        return _bindings(context)


class PropertyTaxReceivingKernel(_Kernel):
    """FILTER-all, with explicit refusal of shapes Frame.select would alter."""

    ref = "us.property_tax_receiving@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        structural=StructuralDelta.FILTER,
        dependencies=("numpy", "pandas"),
    )

    def run(self, context: KernelContext) -> KernelResult:
        node = context.node
        edges = {e.name: e for e in node.artifact_inputs}
        _require(set(edges) == {"projection", "reconciliation"}, "ARTIFACT_EDGES")
        expected = _nodes(
            node.base,
            node.inputs,
            edges["projection"],
            edges["reconciliation"],
            0.0,
            0.0,
        )[0]
        _require(
            node.normative() == expected.normative()
            and not context.sources
            and dict(context.params) == dict(node.params),
            "RECEIVING_DECLARATION",
        )
        _bindings(context)
        _require(
            {s.entity for s in node.inputs} == set(context.tables)
            and "person" in context.tables,
            "RECEIVING_TABLES",
        )
        person = context.tables["person"]
        ids = _person(person)
        _require(len(ids) > 0, "EMPTY_RECEIVING_FRAME")
        for entity, table in context.tables.items():
            if entity == "person":
                continue
            _require(
                table.index.equals(pd.RangeIndex(len(table)))
                and table.index.name is None,
                "UNSUPPORTED_GROUP_INDEX:" + entity,
            )
            _require(
                set(table[entity + "_id"]) == set(person["person_" + entity + "_id"]),
                "ORPHAN_GROUP:" + entity,
            )
        return KernelResult(
            keep=pd.Series(True, index=ids),
            receipt={
                "protocol": PROTOCOL,
                "persons": len(ids),
                "all_rows_retained": True,
                "population_ledger_transition": "filter_conserve",
                "source_authority": False,
            },
        )


class PropertyTaxLeavesKernel(_Kernel):
    ref = "us.property_tax_leaves@1"

    def run(self, context: KernelContext) -> KernelResult:
        bindings = self._check(context, False)
        person = context.tables["person"]
        split = split_property_tax_leaves(person)
        document = _diagnostics(person, split, context.params, bindings)
        return KernelResult(
            columns={("person", c): split[c] for c in TAX_LEAF_COLUMNS},
            artifacts={"diagnostics": codec.encode_json(document)},
            receipt=document,
        )


class PropertyTaxLeafGateKernel(_Kernel):
    ref = "us.property_tax_leaf_gate@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        role=KernelRole.GATE,
        dependencies=("numpy", "pandas"),
    )

    def run(self, context: KernelContext) -> KernelResult:
        bindings = self._check(context, True)
        person = context.tables["person"]
        split = split_property_tax_leaves(person)
        for name in TAX_LEAF_COLUMNS:
            _require(
                name in person and person[name].dtype == np.dtype("float64"),
                "LEAF_DTYPE",
            )
            _require(
                np.array_equal(
                    person[name].to_numpy().view("uint64"),
                    split[name].to_numpy().view("uint64"),
                ),
                "LEAF_BITS:" + name,
            )
        for component, names in (
            (PROPERTY_COMPONENTS[0], TAX_LEAF_COLUMNS[:2]),
            (PROPERTY_COMPONENTS[2], TAX_LEAF_COLUMNS[2:]),
        ):
            amount = person[component].to_numpy()
            known = np.isfinite(amount)
            with np.errstate(over="ignore", invalid="ignore"):
                summed = split.loc[:, list(names)].to_numpy()[known].sum(axis=1)
            _require(np.isfinite(summed).all(), "CONSERVATION_OVERFLOW")
            error = np.abs(summed - amount[known])
            scale = amount[known]
            relative = np.divide(
                error, scale, out=np.zeros_like(error), where=scale != 0
            )
            _require(
                (
                    (error <= context.params["atol"])
                    | ((scale != 0) & (relative <= context.params["rtol"]))
                ).all(),
                "PARTITION_CONSERVATION",
            )
        document = _diagnostics(person, split, context.params, bindings)
        _require(
            context.artifacts["rebase"].payload == codec.encode_json(document),
            "DIAGNOSTIC_PAYLOAD",
        )
        document = {**document, "numeric_verified": True}
        return KernelResult(
            artifacts={"verification": codec.encode_json(document)},
            receipt={
                "outcome": "pass" if document["complete"] else "evidence_absent",
                "evidence": document,
            },
        )
