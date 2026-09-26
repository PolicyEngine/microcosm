"""Retain realized original immigration pairs and fan them to existing clones.

The enrichment host owns all validation and issuance. This fragment performs no
allocation, fitting, random draw, reconciliation or population/weight change.
"""

from __future__ import annotations

import json
import sys
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd

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

from . import current_survey_health_coverage as attachment
from . import current_survey_immigration_transfer as owner
from .graph_survey_population import SOURCE_NAME

PROTOCOL = "microcosm.us.realized-original-immigration-graph.v1"
SOURCE_NODE = "survey_immigration.original_pairs"
ATTACH_NODE = "survey_immigration.attach"
PAIRS_TYPE = ArtifactType("microcosm.us.realized_original_immigration_pairs", 1)
ATTACHMENT_TYPE = ArtifactType("microcosm.us.immigration_clone_attachment", 1)
_REFS = {
    SOURCE_NODE: "us.survey_immigration.original_pairs@1",
    ATTACH_NODE: "us.survey_immigration.attach@1",
}
STRING = pd.StringDtype(storage="python")
# A guard cannot authenticate itself after a caller has rebound it. Use the
# original owner's baseline, even if this fragment is first imported later.
_OWNER_CLASS = owner._LIVE[owner.__name__, "CurrentSurveyImmigrationTransfer"]
_OWNER_CALLABLES = tuple(
    (owner, name, owner._LIVE[owner.__name__, name])
    for name in ("_pure", "_live", "_modules", "_require")
) + (
    (
        _OWNER_CLASS,
        "validate",
        owner._LIVE[owner.__name__, "CurrentSurveyImmigrationTransfer", "validate"],
    ),
)


def _require(condition, reason):
    if not condition:
        raise ValueError("IMMIGRATION_GRAPH_" + reason)


def configuration():
    return (
        owner,
        _OWNER_CLASS,
        _OWNER_CALLABLES,
        attachment,
        attachment.provenance,
        population_ops,
        PROTOCOL,
        SOURCE_NODE,
        ATTACH_NODE,
        SOURCE_NAME,
        tuple(
            (type(t), t.name, t.schema_version) for t in (PAIRS_TYPE, ATTACHMENT_TYPE)
        ),
        tuple(sorted(_REFS.items())),
        STRING,
    )


def retained_entry(transfer):
    """Borrow the genuine owner; an equal frame or receipt is not authority."""
    _require(
        owner.CurrentSurveyImmigrationTransfer is _OWNER_CLASS
        and all(
            type(current := getattr(container, name, None)) is FunctionType
            and owner.source._function_seal(current) == seal
            for container, name, seal in _OWNER_CALLABLES
        ),
        "OWNER_IMPLEMENTATION_CHANGED",
    )
    entry = owner._ISSUED.get(id(transfer))
    _require(
        type(transfer) is owner.CurrentSurveyImmigrationTransfer
        and entry is not None
        and entry[0]() is transfer,
        "ISSUED_TRANSFER_REQUIRED",
    )
    owner._pure(transfer, entry)
    return entry


def canonical_pairs(table):
    """Lossless defensive string projection, not an issuer or an ID coercion."""
    _require(
        type(table) is pd.DataFrame
        and tuple(table.columns) == owner.OUTPUTS
        and table.index.dtype == np.dtype("int64")
        and table.index.name == "person_id"
        and table.index.is_unique,
        "PAIR_AXIS",
    )
    domains = (
        owner.rules.SSN_CARD_TYPE_VALUES,
        owner.rules.IMMIGRATION_STATUS_VALUES,
    )
    for name, domain in zip(owner.OUTPUTS, domains, strict=True):
        _require(
            all(type(v) in (str, np.str_) and v in domain for v in table[name]),
            "PAIR_DOMAIN",
        )
    result = table.copy(deep=True)
    for name in owner.OUTPUTS:
        result[name] = pd.array(table[name], dtype=STRING)
        _require(result[name].tolist() == table[name].tolist(), "PAIR_VALUE_CHANGED")
    _require(result.index.equals(table.index), "PAIR_AXIS_CHANGED")
    return result


def pair_bytes(table):
    """Private typed JSON: exact int64 IDs never pass through a float array."""
    table = canonical_pairs(table)
    return owner.source._encode(
        {
            "protocol": PROTOCOL,
            "person_id": [int(value) for value in table.index],
            "columns": list(owner.OUTPUTS),
            "values": [list(row) for row in table.itertuples(index=False, name=None)],
        }
    )


def read_pairs(payload):
    _require(type(payload) is bytes, "PAIR_BYTES")
    document = json.loads(payload)
    _require(
        type(document) is dict
        and set(document) == {"protocol", "person_id", "columns", "values"}
        and document["protocol"] == PROTOCOL
        and document["columns"] == list(owner.OUTPUTS),
        "PAIR_SCHEMA",
    )
    ids, rows = document["person_id"], document["values"]
    _require(
        type(ids) is list
        and all(type(i) is int and -(2**63) <= i < 2**63 for i in ids)
        and type(rows) is list
        and len(rows) == len(ids)
        and all(type(row) is list and len(row) == len(owner.OUTPUTS) for row in rows),
        "PAIR_ENCODING",
    )
    result = canonical_pairs(
        pd.DataFrame(
            rows,
            columns=list(owner.OUTPUTS),
            index=pd.Index(ids, dtype="int64", name="person_id"),
        )
    )
    _require(pair_bytes(result) == payload, "PAIR_CANONICAL_ENCODING")
    return result


def _pair_edge():
    return ArtifactInput("immigration_pairs", SOURCE_NODE, "pairs", PAIRS_TYPE)


def immigration_nodes(transfer, receiving, *, receiving_version, after):
    entry = retained_entry(transfer)
    state = entry[2]
    _require(
        type(after) is ArtifactInput
        and type(receiving_version) is str
        and receiving_version,
        "FRAGMENT_INPUT",
    )
    columns = canonical_pairs(transfer.pairs)
    origins = state.acs_entry[2].qualified.origins
    # Check the entire retained two-role roster before any unrelated new fit.
    attachment.attach_columns(origins, receiving, columns)
    params = {
        "protocol": PROTOCOL,
        "transfer_receipt_sha256": owner.source._sha(transfer.receipt),
        "pairs_sha256": owner.source._sha(pair_bytes(columns)),
        "original_frame_sha256": json.loads(transfer.receipt)["original_frame_sha256"],
        "original_population_version": state.original.version,
        "original_rows": len(columns),
        "receiving_rows": len(receiving.person),
        "scope": "realized_original_pairs",
        "graph_fit_artifact_qualified": False,
    }
    source_node = Node(
        SOURCE_NODE,
        _REFS[SOURCE_NODE],
        population=receiving_version,
        sources=(SOURCE_NAME,),
        params=params,
        artifact_inputs=(after,),
        artifact_outputs=(ArtifactOutput("pairs", PAIRS_TYPE),),
        description="Retain the final original-person immigration pair privately; no fit, draw or observed-status claim.",
    )
    attach = Node(
        ATTACH_NODE,
        _REFS[ATTACH_NODE],
        population=receiving_version,
        inputs=(
            Slice(
                "person",
                tuple(
                    function("person")
                    for function in (
                        attachment.provenance.support_source_id_column,
                        attachment.provenance.support_clone_index_column,
                        attachment.provenance.spine_source_id_column,
                        attachment.provenance.support_channel_column,
                    )
                ),
            ),
        ),
        outputs=tuple(Owned("person", name, "string") for name in owner.OUTPUTS),
        params=params,
        artifact_inputs=(after, _pair_edge()),
        artifact_outputs=(ArtifactOutput("attachment", ATTACHMENT_TYPE),),
        description="Copy each original's realized immigration pair to its two retained clones without redrawing or changing rows, geography or weights.",
    )
    return source_node, attach


def immigration_result(transfer, node, artifacts, people=None):
    """Pure independent reconstruction from the live retained original result."""
    state = retained_entry(transfer)[2]
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
    payload = pair_bytes(transfer.pairs)
    if node.id == SOURCE_NODE:
        return KernelResult(artifacts={"pairs": payload})
    _require(node.id == ATTACH_NODE and people is not None, "ATTACH_NODE")
    _require(artifacts[_pair_edge().name].payload == payload, "PAIR_ARTIFACT_CHANGED")
    columns = read_pairs(payload)
    attached = attachment.attach_columns(
        state.acs_entry[2].qualified.origins,
        SimpleNamespace(person=people),
        columns,
    )
    mapping = people.loc[
        :,
        [
            "person_id",
            attachment.provenance.support_source_id_column("person"),
            attachment.provenance.support_clone_index_column("person"),
            attachment.provenance.spine_source_id_column("person"),
            attachment.provenance.support_channel_column("person"),
        ],
    ]
    evidence = {
        "protocol": PROTOCOL,
        "transfer_receipt_sha256": owner.source._sha(transfer.receipt),
        "pairs_sha256": owner.source._sha(payload),
        "original_rows": len(columns),
        "receiving_rows": len(people),
        "receiving_version": node.population,
        "mapping_sha256": owner.assignment_owner.donor_owner.literals._table_seal(
            mapping
        ),
        "realized_original_pairs_preserved": True,
        "new_draw_performed": False,
        "weights_changed": False,
        "source_admission_issued": False,
        "graph_fit_artifact_qualified": False,
        "national_stock_alignment_qualified": False,
        "release_eligible": False,
    }
    return KernelResult(
        columns=attached,
        artifacts={"attachment": owner.source._encode(evidence)},
        receipt=evidence,
    )


def _result_seal(result):
    return (
        owner.assignment_owner.donor_owner.literals._table_seal(
            pd.concat(
                [value.rename(name) for (_, name), value in result.columns.items()],
                axis=1,
            )
        )
        if result.columns
        else None,
        tuple(sorted(result.artifacts.items())),
        owner.source._encode(dict(result.receipt)),
    )


class _ImmigrationKernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def __init__(self, boundary, ref):
        self.boundary, self.ref = boundary, ref

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            *owner._modules(),
            attachment,
            attachment.provenance,
            population_ops,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        self.boundary.context(context)
        _require(
            context.node in self.boundary.immigration_nodes
            and context.node.kernel == self.ref,
            "KERNEL_NODE",
        )
        result = immigration_result(
            self.boundary.immigration_transfer,
            context.node,
            context.artifacts,
            context.tables.get("person"),
        )
        stamp = _result_seal(result)
        # The owning host performs full owner validation at admission and its
        # existing final I/O fences. Ordinary kernels use the same pure seals.
        self.boundary.pure()
        _require(_result_seal(result) == stamp, "FINAL_RESULT_CHANGED")
        return result


def immigration_kernels(boundary):
    return tuple(
        _ImmigrationKernel(boundary, node.kernel) for node in boundary.immigration_nodes
    )
