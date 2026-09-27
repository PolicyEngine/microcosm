"""Four explicit development mappings, authenticated before graph publication.

These are qualified values chosen for preservation, not observed taxable
amounts. Source custody remains in the financial host and its preparation.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd

from microcosm.frame import Frame
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
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
from microcosm.graph.keys import _capabilities_projection

from . import current_survey_health_coverage as attachment
from . import current_survey_health_source as original_source
from . import graph_current_survey_predictors as financial
from . import puf55_survey_observed as fixed
from . import support_provenance as provenance

source = financial.values.source
codec = financial.codec
routing = fixed.routing
PROTOCOL = "microcosm.us.source-qualified-development-inputs.v1"
SOURCE_NODE = "survey_development_inputs.source_projection"
ATTACH_NODE = "survey_development_inputs.attach"
NODE_IDS = (SOURCE_NODE, ATTACH_NODE)
OUTPUTS = fixed.DEVELOPMENT_TARGETS
PROJECTION_TYPE = ArtifactType("microcosm.us.asec_development_projection", 1)
BINDING_TYPE = ArtifactType("microcosm.us.asec_development_attachment", 1)


def require(condition, reason):
    if not condition:
        raise ValueError("SURVEY_DEVELOPMENT_INPUTS_" + reason)


def _table(table):
    return {
        "index": [source._cell(v) for v in table.index],
        "index_name": table.index.name,
        "columns": [(c, str(table[c].dtype)) for c in table],
        "rows": [
            [source._cell(v) for v in row]
            for row in table.itertuples(index=False, name=None)
        ],
    }


def qualification_payload(qualified):
    require(type(qualified) is routing.CurrentAsecIncomeRoutingValues, "QUALIFIED_TYPE")
    values, known = fixed._development_person_values(qualified.person)
    return codec.encode_json(
        {
            "protocol": PROTOCOL,
            "source_evidence": qualified.evidence,
            "source_basis": _table(qualified.person),
            "source_basis_sha256": fixed.recipients._table_digest(qualified.person),
            "source_literals": _table(qualified.asec_literals),
            "source_literals_sha256": fixed.recipients._table_digest(
                qualified.asec_literals
            ),
            "rule_metadata": fixed.development_rule_metadata(fixed.DEVELOPMENT_RULES),
            "values": _table(values),
            "known": _table(known),
            "value_bits": {
                c: values[c].to_numpy(dtype="<f8").tobytes().hex() for c in OUTPUTS
            },
            "source_admission_issued": False,
            "release_eligible": False,
        }
    )


def _identity_columns():
    return tuple(
        f("person")
        for f in (
            provenance.support_source_id_column,
            provenance.support_channel_column,
            provenance.spine_source_id_column,
            provenance.support_clone_index_column,
        )
    )


def columns(qualified, originals, people):
    """Use the maintained exact two-clone mapper, with source-qualified values."""
    require(not set(OUTPUTS) & set(people), "INCUMBENT_COLUMNS")
    basis = qualified.person
    require(
        originals.index.is_unique
        and originals.index.dtype == np.dtype("int64")
        and originals.source.isin(("asec", "acs")).all()
        and basis.index.is_unique
        and basis.native_person_id.is_unique
        and set(basis.index) == set(originals.index[originals.source.eq("asec")])
        and np.array_equal(
            basis.native_person_id.to_numpy(),
            originals.loc[basis.index, "native_person_id"].to_numpy(),
        ),
        "SOURCE_JOIN",
    )
    values, _ = fixed._development_person_values(basis)
    complete = pd.DataFrame(
        np.nan, index=originals.index, columns=OUTPUTS, dtype=np.float64
    )
    complete.loc[values.index, list(OUTPUTS)] = values
    return attachment.attach_columns(
        originals, SimpleNamespace(person=people), complete
    )


def projection_edge():
    return ArtifactInput(
        "development_projection", SOURCE_NODE, "projection", PROJECTION_TYPE
    )


def nodes(
    qualified,
    receiving,
    *,
    population,
    host_edges,
    host_pins,
    after_columns=(),
    after_edges=(),
):
    require(not set(OUTPUTS) & set(receiving.person), "INCUMBENT_COLUMNS")
    params = {
        "protocol": PROTOCOL,
        "projection_sha256": codec.sha(qualification_payload(qualified)),
        "rule_metadata": codec.encode_json(
            fixed.development_rule_metadata(fixed.DEVELOPMENT_RULES)
        ).decode(),
        "host_edges": codec.encode_json(host_pins).decode(),
    }
    inputs = (
        Slice("person", tuple(dict.fromkeys((*_identity_columns(), *after_columns)))),
    )
    return (
        Node(
            SOURCE_NODE,
            "us.survey_development_inputs.source@1",
            population=population,
            inputs=inputs,
            params=params,
            artifact_inputs=(*host_edges, *after_edges),
            artifact_outputs=(ArtifactOutput("projection", PROJECTION_TYPE),),
            description="Publish authentic ASEC routing, raw totals and explicit development mapping assumptions; retain unknown ACS and off-route cells.",
        ),
        Node(
            ATTACH_NODE,
            "us.survey_development_inputs.attach@1",
            population=population,
            inputs=inputs,
            params=params,
            artifact_inputs=(projection_edge(),),
            artifact_outputs=(ArtifactOutput("binding", BINDING_TYPE),),
            outputs=tuple(Owned("person", c, "float64") for c in OUTPUTS),
            description="Attach four qualified development values by complete original clone pairs; no NIU zero filling or taxability observation claim.",
        ),
    )


def projection_payload(qualified, node, people):
    # The declared parent columns are genuine inputs: preserve their exact
    # semantic values in the projection and require the same view at attach.
    names = tuple(
        dict.fromkeys(
            ("person_id", *(c for selection in node.inputs for c in selection.columns))
        )
    )
    return codec.encode_json(
        {
            "qualification": codec.decode_json(qualification_payload(qualified)),
            "parent_inputs": _table(people.loc[:, list(names)]),
        }
    )


def result(qualified, originals, node, people):
    payload = projection_payload(qualified, node, people)
    receipt = {
        "protocol": PROTOCOL,
        "projection_sha256": codec.sha(payload),
        "rule_metadata": fixed.development_rule_metadata(fixed.DEVELOPMENT_RULES),
        "source_admission_issued": False,
        "release_eligible": False,
    }
    if node.id == SOURCE_NODE:
        return KernelResult(artifacts={"projection": payload}, receipt=receipt)
    require(node.id == ATTACH_NODE, "NODE")
    attached = columns(qualified, originals, people)
    binding = codec.encode_json(
        {
            **receipt,
            "columns": {
                c: {
                    "index": [str(int(v)) for v in attached["person", c].index],
                    "float64_bits": attached["person", c]
                    .to_numpy(dtype="<f8")
                    .tobytes()
                    .hex(),
                }
                for c in OUTPUTS
            },
        }
    )
    return KernelResult(
        columns=attached, artifacts={"binding": binding}, receipt=receipt
    )


def complement(population):
    """Remove only these additions for the independently retained old verifier."""
    require(set(OUTPUTS) <= set(population.frame.person), "MISSING_COLUMNS")
    frame = population.frame
    tables = {e: frame.table(e).copy() for e in frame.entities}
    tables.update({name: frame.link(name).copy() for name in frame.links})
    tables["person"] = tables["person"].drop(columns=list(OUTPUTS))
    return replace(
        population,
        frame=Frame(
            tables,
            frame.schema,
            dict(frame._weights),
            frame.strata,
            metadata=frame.metadata,
            mass_log=frame.mass_log,
        ),
        owners={
            key: value
            for key, value in population.owners.items()
            if key not in {("person", c) for c in OUTPUTS}
        },
    )


def _live():
    modules = (
        sys.modules[__name__],
        fixed,
        routing,
        fixed.leaves,
        provenance,
        fixed.recipients,
        financial,
        financial.values,
        attachment,
        original_source,
        codec,
    )
    return (
        tuple(
            (
                m,
                tuple(
                    (n, source._function_seal(v))
                    for n, v in vars(m).items()
                    if isinstance(v, FunctionType)
                ),
            )
            for m in modules
        ),
        source,
        codec,
        financial,
        attachment,
        attachment.provenance,
        original_source,
        original_source.health,
        PROTOCOL,
        SOURCE_NODE,
        ATTACH_NODE,
        NODE_IDS,
        PROJECTION_TYPE,
        BINDING_TYPE,
        ArtifactType,
        ArtifactInput,
        ArtifactOutput,
        Owned,
        Slice,
        Frame,
        Node,
        KernelResult,
        DevelopmentKernel,
        tuple(
            (n, source._function_seal(v))
            for n, v in vars(DevelopmentKernel).items()
            if isinstance(v, FunctionType)
        ),
        codec.encode_json(_capabilities_projection(DevelopmentKernel.capabilities)),
        OUTPUTS,
        type(fixed.DEVELOPMENT_TARGETS),
        fixed.DEVELOPMENT_TARGETS,
        fixed.DEVELOPMENT_RULES,
        codec.encode_json(fixed.development_rule_metadata(fixed.DEVELOPMENT_RULES)),
        source._runtime_marker(
            (
                routing.READ_COLUMNS,
                routing.KNOWN_AMOUNT_STATUSES,
                routing.RECEIPT_ENTRIES,
                routing.ACCOUNT_ENTRIES,
                routing.ALLOCATION_ENTRIES,
            )
        ),
    )


class DevelopmentKernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def __init__(self, boundary, node):
        self.boundary, self.node, self.ref = boundary, node, node.kernel

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            fixed,
            routing,
            fixed.leaves,
            fixed.recipients,
            financial,
            financial.values,
            attachment,
            original_source,
            codec,
            provenance,
            source,
            population_ops,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        b = self.boundary
        before = b.seal()
        b.pure()
        require(context.node == self.node, "DECLARATION")
        b.context(context)
        output = result(b.qualified, b.originals, self.node, context.tables["person"])
        stamp = _result_seal(output)
        b.pure()
        b.context(context)
        require(before == b.seal() and stamp == _result_seal(output), "RESULT_CHANGED")
        return output


def _result_seal(value):
    return codec.encode_json(
        {
            "columns": [
                (e, c, _table(v.to_frame()), v.to_numpy(copy=False).tobytes().hex())
                for (e, c), v in value.columns.items()
            ],
            "artifacts": {n: codec.sha(p) for n, p in value.artifacts.items()},
            "receipt": value.receipt,
        }
    )


_LIVE = _live()
