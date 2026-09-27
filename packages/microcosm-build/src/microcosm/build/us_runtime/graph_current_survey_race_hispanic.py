"""Visible source-race_hispanic binding once per original person, then exact clone copy.

The host owns admission and final source requalification. Ordinary kernels use
the retained projection and pure seals; they never infer source authority from
an inherited canonical column or a serialized receipt.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from microcosm.frame import Frame
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
    StructuralDelta,
    source_hash,
)
from microcosm.graph import population as population_ops

from . import common_frame_export_contract as comparison
from . import current_survey_race_hispanic_source as source
from .graph_survey_population import SOURCE_NAME

require = source.require
SOURCE_NODE = "survey_race_hispanic.source"
RAW_NODE = "survey_race_hispanic.raw_columns"
BIND_NODE = "survey_race_hispanic.bind_demographics"
ATTACH_NODE = "survey_race_hispanic.attach"
PROJECTION_TYPE = ArtifactType(
    "microcosm.us.current_survey_race_hispanic_projection", 1
)
COLUMNS_TYPE = ArtifactType("microcosm.us.current_survey_race_hispanic_columns", 1)
ATTACHMENT_TYPE = ArtifactType(
    "microcosm.us.current_survey_race_hispanic_attachment", 1
)


def _bytes(table):
    return table.to_json(orient="table").encode()


def _projection(qualified):
    return source.source._encode(
        {
            "evidence": qualified.evidence.decode(),
            "raw": _bytes(qualified.raw).decode(),
        }
    )


def _outputs(table):
    return tuple(
        Owned("person", name, population_ops.token_for_dtype(table[name].dtype))
        for name in table
    )


def _source_edge():
    return ArtifactInput(
        "race_hispanic_projection", SOURCE_NODE, "projection", PROJECTION_TYPE
    )


def _columns_edge(node_id):
    return ArtifactInput(
        "race_hispanic_raw" if node_id == RAW_NODE else "race_hispanic_canonical",
        node_id,
        "columns",
        COLUMNS_TYPE,
    )


def _completed(qualified):
    return qualified.raw.join(source.recode(qualified.raw))


def race_hispanic_nodes(qualified, receiving, *, receiving_version, after):
    source.retained(qualified)
    require(
        type(after) is ArtifactInput
        and type(receiving_version) is str
        and receiving_version,
        "FRAGMENT_INPUT",
    )
    require(
        not set(_completed(qualified)) & set(receiving.person),
        "ATTACH_OWNERSHIP_COLLISION",
    )
    params = {
        "protocol": source.PROTOCOL,
        "source_evidence": qualified.evidence.decode(),
        "projection_sha256": source.source._sha(_projection(qualified)),
        "mapping": "exact CPS categories and Hispanic origin; unresolved stays null",
    }
    create = Node(
        SOURCE_NODE,
        "us.survey_race_hispanic.source@1",
        structural=StructuralDelta.CREATE,
        sources=(SOURCE_NAME,),
        outputs=(Owned("person", "race_hispanic_native_person_id", "int64"),),
        params=params,
        artifact_inputs=(after,),
        artifact_outputs=(ArtifactOutput("projection", PROJECTION_TYPE),),
        description="Borrow authenticated original ACS/ASEC support and private race_hispanic/allocation observations.",
    )
    raw = Node(
        RAW_NODE,
        "us.survey_race_hispanic.raw@1",
        population=SOURCE_NODE,
        outputs=_outputs(qualified.raw),
        params=params,
        artifact_inputs=(_source_edge(),),
        artifact_outputs=(ArtifactOutput("columns", COLUMNS_TYPE),),
        description="Expose source race_hispanic codes, ASEC allocation flags, observation year and unresolved provenance.",
    )
    bind = Node(
        BIND_NODE,
        "us.survey_race_hispanic.bind@1",
        population=SOURCE_NODE,
        inputs=(Slice("person", source.RAW_COLUMNS),),
        outputs=_outputs(source.recode(qualified.raw)),
        params=params,
        artifact_inputs=(_source_edge(), _columns_edge(RAW_NODE)),
        artifact_outputs=(ArtifactOutput("columns", COLUMNS_TYPE),),
        description="Bind nullable CPS race and Hispanic origin once per original; unsupported categories remain unknown.",
    )
    provenance = source.attachment.provenance
    attach = Node(
        ATTACH_NODE,
        "us.survey_race_hispanic.attach@1",
        population=receiving_version,
        inputs=(
            Slice(
                "person",
                tuple(
                    f("person")
                    for f in (
                        provenance.support_source_id_column,
                        provenance.support_clone_index_column,
                        provenance.spine_source_id_column,
                        provenance.support_channel_column,
                    )
                ),
            ),
        ),
        outputs=_outputs(_completed(qualified)),
        params=params,
        artifact_inputs=(
            after,
            _source_edge(),
            _columns_edge(RAW_NODE),
            _columns_edge(BIND_NODE),
        ),
        artifact_outputs=(ArtifactOutput("attachment", ATTACHMENT_TYPE),),
        description="Copy each qualified original's race_hispanic and provenance to its existing two clones, preserving unknowns and weights.",
    )
    return create, raw, bind, attach


def _check_artifacts(qualified, node, artifacts):
    require(
        set(artifacts) == {edge.name for edge in node.artifact_inputs},
        "ARTIFACT_ROSTER",
    )
    expected = {
        (SOURCE_NODE, "projection"): _projection(qualified),
        (RAW_NODE, "columns"): _bytes(qualified.raw),
        (BIND_NODE, "columns"): _bytes(source.recode(qualified.raw)),
    }
    for edge in node.artifact_inputs:
        value = artifacts[edge.name]
        require(
            type(value) is ArtifactValue
            and value.type == edge.type
            and type(value.payload) is bytes,
            "ARTIFACT_TYPE",
        )
        if (edge.producer, edge.artifact) in expected:
            require(
                value.payload == expected[edge.producer, edge.artifact],
                "ARTIFACT_PAYLOAD",
            )
        # The retained host authenticates the preceding fragment's edge and key.


def race_hispanic_result(qualified, node, artifacts, people=None):
    _check_artifacts(qualified, node, artifacts)
    if node.id == SOURCE_NODE:
        original = source.source._copy_source(qualified.source_frame)
        tables = {}
        for entity in original.entities:
            columns = [original.schema.entity_id_column(entity)]
            if entity == original.schema.person_entity:
                columns += [
                    original.schema.membership_column(e)
                    for e in original.schema.group_entities
                ]
            tables[entity] = original.table(entity).loc[:, columns].copy()
        require(
            np.array_equal(
                original.person.person_id.to_numpy(), qualified.origins.index.to_numpy()
            ),
            "CREATE_ORIGIN_AXIS",
        )
        tables["person"]["race_hispanic_native_person_id"] = (
            qualified.origins.native_person_id.to_numpy(copy=True)
        )
        frame = Frame(
            tables,
            original.schema,
            dict(original._weights),
            original.strata,
            metadata=original.metadata,
            mass_log=original.mass_log,
        )
        return KernelResult(
            frame=frame, artifacts={"projection": _projection(qualified)}
        )
    if node.id == ATTACH_NODE:
        require(people is not None, "RECEIVING_REQUIRED")
        attached = source.attachment.attach_columns(
            qualified.origins, SimpleNamespace(person=people), _completed(qualified)
        )
        evidence = source.source._encode(
            {
                "protocol": source.PROTOCOL,
                "projection_sha256": source.source._sha(_projection(qualified)),
                "receiving_version": node.population,
                "rows": len(people),
                "unknown_policy": "preserved",
                "weights_changed": False,
                "source_admission_issued": False,
                "release_eligible": False,
            }
        )
        return KernelResult(columns=attached, artifacts={"attachment": evidence})
    require(node.id in (RAW_NODE, BIND_NODE) and people is not None, "BIND_NODE")
    require(
        np.array_equal(people.person_id.to_numpy(), qualified.raw.index.to_numpy()),
        "ORIGINAL_PERSON_AXIS",
    )
    table = qualified.raw if node.id == RAW_NODE else source.recode(qualified.raw)
    if node.id == BIND_NODE:
        actual = people.set_index("person_id")
        require(
            all(
                comparison._same_series(qualified.raw[c], actual[c], readback=True)
                for c in source.RAW_COLUMNS
            ),
            "BIND_SOURCE_SLICE",
        )
    return KernelResult(
        columns={("person", c): table[c].copy() for c in table},
        artifacts={"columns": _bytes(table)},
    )


def _result_seal(result):
    return (
        source.source._frame_identity(result.frame)
        if result.frame is not None
        else None,
        tuple(
            (key, source.physical._table_stamp(series.to_frame()))
            for key, series in result.columns.items()
        ),
        tuple(sorted(result.artifacts.items())),
    )


class _RaceHispanicKernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def __init__(self, qualified, nodes, require_current, require_context, ref):
        self.qualified, self.nodes, self.ref = qualified, nodes, ref
        self.require_current, self.require_context = require_current, require_context
        self.stamp = source.seal(qualified)
        if ref == "us.survey_race_hispanic.source@1":
            self.capabilities = replace(
                self.capabilities, structural=StructuralDelta.CREATE
            )

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            source,
            source.original.asec,
            source.original.records,
            source.original.housing,
            source.literals,
            source.source_csv_builtin,
            source.original,
            source.attachment,
            source.source,
            source.physical,
            comparison,
            source.source.acs_native,
            population_ops,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        self.require_current()
        self.require_context(context)
        source.retained(self.qualified)
        require(source.seal(self.qualified) == self.stamp, "QUALIFIED_CHANGED")
        require(
            context.node in self.nodes and context.node.kernel == self.ref,
            "KERNEL_NODE",
        )
        result = race_hispanic_result(
            self.qualified,
            context.node,
            context.artifacts,
            context.tables.get("person"),
        )
        stamp = _result_seal(result)
        self.require_current()
        require(
            source.seal(self.qualified) == self.stamp and _result_seal(result) == stamp,
            "RESULT_CHANGED",
        )
        return result


def race_hispanic_kernels(
    qualified, receiving, *, receiving_version, after, require_current, require_context
):
    require(callable(require_current) and callable(require_context), "HOST_CALLBACK")
    nodes = race_hispanic_nodes(
        qualified, receiving, receiving_version=receiving_version, after=after
    )
    return tuple(
        _RaceHispanicKernel(qualified, nodes, require_current, require_context, ref)
        for ref in dict.fromkeys(n.kernel for n in nodes)
    )
