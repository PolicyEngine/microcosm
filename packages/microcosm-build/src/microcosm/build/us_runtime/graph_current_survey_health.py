"""Private health fragment for the one country enrichment host.

The host retains the actual checked PUF run, preparation, kernel registry and
source files. This fragment never issues a run or creates a second receiving
branch. Its final ordinary attachment preserves the incoming population version.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

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

from . import current_survey_health_coverage as health
from . import current_survey_health_source as source
from . import graph_full_puf_enrichment as physical
from .graph_survey_population import SOURCE_NAME

qualify_health_coverage = source.qualify_current_survey_health
require = health.require
SOURCE_NODE = "survey_health.source"
RAW_NODE = "survey_health.raw_columns"
RECODE_PREFIX = "survey_health.recode."
ATTACH_NODE = "survey_health.attach"
PROJECTION_TYPE = ArtifactType("microcosm.us.current_survey_health_projection", 1)
COLUMNS_TYPE = ArtifactType("microcosm.us.current_survey_health_columns", 1)
ATTACHMENT_TYPE = ArtifactType("microcosm.us.current_survey_health_attachment", 1)


def _json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _table_bytes(table):
    return table.to_json(orient="table").encode()


def health_coverage_seal(qualified):
    require(type(qualified) is source.QualifiedSurveyHealthCoverage, "QUALIFIED_TYPE")
    require(
        qualified.projection
        == qualified.raw.reset_index().to_json(orient="table", index=False).encode()
        and qualified.evidence["projection_sha256"]
        == hashlib.sha256(qualified.projection).hexdigest(),
        "PROJECTION_BINDING",
    )
    return (
        source.source._frame_identity(qualified.source_frame),
        physical._table_stamp(qualified.origins),
        physical._table_stamp(qualified.raw),
        qualified.projection,
        _json(qualified.evidence),
    )


def _source_edge():
    return ArtifactInput(
        "health_projection", SOURCE_NODE, "projection", PROJECTION_TYPE
    )


def _column_artifact(node_id, alias):
    return ArtifactInput(alias, node_id, "columns", COLUMNS_TYPE)


def _dtype(series):
    return population_ops.token_for_dtype(series.dtype)


def _outputs(table):
    return tuple(Owned("person", name, _dtype(table[name])) for name in table)


def _completed(qualified):
    return pd.concat(
        [
            health.source_columns(qualified.raw),
            *(health.recode_field(qualified.raw, f.output) for f in health.FIELDS),
        ],
        axis=1,
    )


def _params(qualified):
    return {
        "protocol": health.PROTOCOL,
        "projection_sha256": hashlib.sha256(qualified.projection).hexdigest(),
        "source_evidence": _json(qualified.evidence).decode(),
    }


def health_coverage_nodes(qualified, *, receiving_version, after):
    health_coverage_seal(qualified)
    require(
        type(receiving_version) is str
        and receiving_version
        and type(after) is ArtifactInput,
        "FRAGMENT_INPUT",
    )
    params = _params(qualified)
    raw = health.source_columns(qualified.raw)
    create = Node(
        SOURCE_NODE,
        "us.survey_health.source@1",
        structural=StructuralDelta.CREATE,
        sources=(SOURCE_NAME,),
        outputs=(Owned("person", "health_native_person_id", "int64"),),
        params=params,
        artifact_inputs=(after,),
        artifact_outputs=(ArtifactOutput("projection", PROJECTION_TYPE),),
        description="Borrow original ACS/ASEC person support and retain authenticated current-coverage source projection.",
    )
    literal = Node(
        RAW_NODE,
        "us.survey_health.raw_columns@1",
        population=SOURCE_NODE,
        outputs=_outputs(raw),
        params=params,
        artifact_inputs=(_source_edge(),),
        artifact_outputs=(ArtifactOutput("columns", COLUMNS_TYPE),),
        description="Expose literal coverage, allocation and edit codes with survey observation years.",
    )
    recodes = []
    for field in health.FIELDS:
        inputs = [
            health.SOURCE_PREFIX + "source",
            health.SOURCE_PREFIX + field.asec,
            health.SOURCE_PREFIX + "I_" + field.asec,
        ]
        if field.acs:
            inputs += [
                health.SOURCE_PREFIX + field.acs,
                health.SOURCE_PREFIX + "F" + field.acs + "P",
            ]
        recodes.append(
            Node(
                RECODE_PREFIX + field.output,
                "us.survey_health.recode@1",
                population=SOURCE_NODE,
                inputs=(Slice("person", tuple(inputs)),),
                outputs=_outputs(health.recode_field(qualified.raw, field.output)),
                params={**params, "field": field.output},
                artifact_inputs=(
                    _source_edge(),
                    _column_artifact(RAW_NODE, "health_raw_columns"),
                ),
                artifact_outputs=(ArtifactOutput("columns", COLUMNS_TYPE),),
                description=f"Current coverage: {field.asec}; ACS {field.acs or field.acs_gap}. Missing codes remain unknown.",
            )
        )
    identity = tuple(
        f("person")
        for f in (
            health.provenance.support_source_id_column,
            health.provenance.support_clone_index_column,
            health.provenance.spine_source_id_column,
            health.provenance.support_channel_column,
        )
    )
    attach = Node(
        ATTACH_NODE,
        "us.survey_health.attach@1",
        population=receiving_version,
        inputs=(Slice("person", identity),),
        outputs=_outputs(_completed(qualified)),
        params=params,
        artifact_inputs=(
            after,
            _source_edge(),
            _column_artifact(RAW_NODE, "health_raw_columns"),
            *(
                _column_artifact(n.id, "health_field_" + str(i))
                for i, n in enumerate(recodes)
            ),
        ),
        artifact_outputs=(ArtifactOutput("attachment", ATTACHMENT_TYPE),),
        description="Attach only exact source-person observations and explicit unknowns to both PUF clones; preserve all incoming fields.",
    )
    return (create, literal, *recodes, attach)


def _expected_artifacts(qualified):
    return {
        (SOURCE_NODE, "projection"): qualified.projection,
        (RAW_NODE, "columns"): _table_bytes(health.source_columns(qualified.raw)),
        **{
            (RECODE_PREFIX + f.output, "columns"): _table_bytes(
                health.recode_field(qualified.raw, f.output)
            )
            for f in health.FIELDS
        },
    }


def _check_artifacts(node, artifacts, qualified):
    require(set(artifacts) == {a.name for a in node.artifact_inputs}, "ARTIFACT_ROSTER")
    expected = _expected_artifacts(qualified)
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
        # The incoming amount edge is authenticated by the owning host, which
        # retains its node/store keys. The fragment never treats its JSON as an issuer.


def _result(qualified, node, people):
    if node.id == SOURCE_NODE:
        original = source.source._copy_source(qualified.source_frame)
        # A projection branch keeps original support/design and structural IDs;
        # its coverage cells are introduced visibly by the following nodes.
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
                qualified.origins.index.to_numpy(), original.person.person_id.to_numpy()
            ),
            "CREATE_ORIGIN_AXIS",
        )
        tables["person"]["health_native_person_id"] = (
            qualified.origins.native_person_id.to_numpy(copy=True)
        )
        projected = Frame(
            tables,
            original.schema,
            dict(original._weights),
            original.strata,
            metadata=original.metadata,
            mass_log=original.mass_log,
        )
        return KernelResult(
            frame=projected, artifacts={"projection": qualified.projection}
        )
    if node.id == RAW_NODE:
        table = health.source_columns(qualified.raw)
    elif node.id.startswith(RECODE_PREFIX):
        table = health.recode_field(qualified.raw, node.params["field"])
    else:
        require(node.id == ATTACH_NODE, "NODE_ID")
        columns = health.attach_columns(
            qualified.origins, SimpleNamespace(person=people), _completed(qualified)
        )
        return KernelResult(
            columns=columns,
            artifacts={
                "attachment": _json(
                    {
                        "protocol": health.PROTOCOL,
                        "projection_sha256": hashlib.sha256(
                            qualified.projection
                        ).hexdigest(),
                        "receiving_version": node.population,
                        "rows": len(people),
                        "source_admission_issued": False,
                        "release_eligible": False,
                    }
                )
            },
        )
    require(
        people is not None
        and np.array_equal(people.person_id.to_numpy(), table.index.to_numpy()),
        "SOURCE_PERSON_AXIS",
    )
    return KernelResult(
        columns={("person", c): table[c].copy() for c in table},
        artifacts={"columns": _table_bytes(table)},
    )


def expected_health_population(node_id, incoming, *, qualified, node, artifacts):
    """Reconstruct the complete output, independently of executor cache results."""
    require(node_id == node.id, "EXPECTED_NODE")
    _check_artifacts(node, artifacts, qualified)
    result = _result(
        qualified, node, None if incoming is None else incoming.frame.person
    )
    if node.structural is StructuralDelta.CREATE:
        require(incoming is None, "CREATE_INCOMING")
        return population_ops.Population.from_frame(result.frame, node.id)
    require(incoming is not None, "ORDINARY_INCOMING")
    return population_ops.patch(incoming, node, result)


class _HealthKernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def __init__(self, qualified, nodes, require_current, *, create=False, ref):
        self.ref, self.qualified, self.nodes = ref, qualified, nodes
        self.require_current, self.seal = (
            require_current,
            health_coverage_seal(qualified),
        )
        if create:
            self.capabilities = replace(
                self.capabilities, structural=StructuralDelta.CREATE
            )

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            health,
            source,
            physical,
            source.source,
            source.housing,
            source.records,
            source.asec,
            source.source_csv_builtin,
            sys.modules[health.__package__ + ".cps_carried"],
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        self.require_current()
        require(health_coverage_seal(self.qualified) == self.seal, "QUALIFIED_CHANGED")
        require(
            context.node in self.nodes and context.node.kernel == self.ref,
            "KERNEL_NODE",
        )
        _check_artifacts(context.node, context.artifacts, self.qualified)
        if context.node.id.startswith(RECODE_PREFIX):
            expected = health.source_columns(self.qualified.raw)
            actual = context.tables["person"].set_index("person_id")
            for selection in context.node.inputs:
                for column in selection.columns:
                    require(
                        population_ops.storage_equal(actual[column], expected[column]),
                        "RECODE_SOURCE_SLICE",
                    )
        result = _result(self.qualified, context.node, context.tables.get("person"))
        result_seal = (
            physical._table_stamp(
                pd.concat(
                    [v.rename(c) for (_e, c), v in result.columns.items()], axis=1
                )
            )
            if result.columns
            else None
        )
        frame_seal = (
            source.source._frame_identity(result.frame)
            if result.frame is not None
            else None
        )
        artifacts = tuple(sorted(result.artifacts.items()))
        self.require_current()
        require(health_coverage_seal(self.qualified) == self.seal, "QUALIFIED_CHANGED")
        require(
            (
                physical._table_stamp(
                    pd.concat(
                        [v.rename(c) for (_e, c), v in result.columns.items()], axis=1
                    )
                )
                if result.columns
                else None
            )
            == result_seal
            and (
                source.source._frame_identity(result.frame)
                if result.frame is not None
                else None
            )
            == frame_seal
            and tuple(sorted(result.artifacts.items())) == artifacts,
            "RESULT_CHANGED",
        )
        return result


def health_coverage_kernels(qualified, *, receiving_version, after, require_current):
    nodes = health_coverage_nodes(
        qualified, receiving_version=receiving_version, after=after
    )
    require(callable(require_current), "HOST_CALLBACK")
    return tuple(
        _HealthKernel(
            qualified,
            nodes,
            require_current,
            create=ref == "us.survey_health.source@1",
            ref=ref,
        )
        for ref in dict.fromkeys(n.kernel for n in nodes)
    )
