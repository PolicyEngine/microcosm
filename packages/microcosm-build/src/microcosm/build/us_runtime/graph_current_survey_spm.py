"""Three ordinary SPM input nodes for the retained survey enrichment host.

Pure artifacts describe source observations and exact clone transport. Only the
host retains source authority; this fragment creates no population, model, source
issuer or release permission.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from microcosm.build.spm_input_contract import ROLE_INPUT, UNIVERSE_INPUT
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

from . import current_survey_spm_source as source
from .graph_survey_population import SOURCE_NAME

projection = source.projection
PROTOCOL = "microcosm.us.current-survey-spm-graph.v1"
SOURCE_NODE = "survey_spm.source"
PROJECT_NODE = "survey_spm.project"
ATTACH_NODE = "survey_spm.attach"
SOURCE_TYPE = ArtifactType("microcosm.us.current_survey_spm_source", 1)
PROJECTION_TYPE = ArtifactType("microcosm.us.current_survey_spm_projection", 1)
ATTACHMENT_TYPE = ArtifactType("microcosm.us.current_survey_spm_attachment", 1)
_REFS = {
    SOURCE_NODE: "us.survey_spm.source@1",
    PROJECT_NODE: "us.survey_spm.project@1",
    ATTACH_NODE: "us.survey_spm.attach@1",
}
_ENTITIES = ("person", "spm_unit", "household")


def _require(condition, reason):
    if not condition:
        raise ValueError("SPM_GRAPH_" + reason)


def _objects(qualified):
    return tuple(
        getattr(qualified, name)
        for name in (
            "source_frame",
            "origins",
            "roles",
            "unit_status",
            "unit_evidence",
            "asec_raw",
        )
    )


def _artifact_type_seal(value):
    _require(
        type(value) is ArtifactType
        and type(value.name) is str
        and bool(value.name)
        and type(value.schema_version) is int
        and value.schema_version > 0,
        "ARTIFACT_TYPE_CONFIGURATION",
    )
    return type(value), value.name, value.schema_version


def spm_seal(qualified):
    """Pure value identity; descriptive values alone cannot issue kernels."""
    _require(
        type(qualified) is source.QualifiedNativeSpmInputs
        and qualified._receiving_run is None,
        "PREPARATION_QUALIFICATION_REQUIRED",
    )
    return (
        (
            source,
            projection,
            population_ops,
            json,
            source_hash,
            PROTOCOL,
            SOURCE_NODE,
            PROJECT_NODE,
            ATTACH_NODE,
            SOURCE_NAME,
            _artifact_type_seal(SOURCE_TYPE),
            _artifact_type_seal(PROJECTION_TYPE),
            _artifact_type_seal(ATTACHMENT_TYPE),
            ROLE_INPUT,
            UNIVERSE_INPUT,
            _ENTITIES,
            tuple(sorted(_REFS.items())),
        ),
        source.source._frame_identity(qualified.source_frame),
        *(source.seals._table_seal(t) for t in source._result_tables(qualified)),
        (type(qualified.roles.name).__name__, qualified.roles.name),
        (type(qualified.unit_status.name).__name__, qualified.unit_status.name),
        qualified.receipt,
    )


def _table(table):
    # Keep separate typed entity tables, including their exact integer indexes.
    return json.loads(table.to_json(orient="table", double_precision=15))


def _source_bytes(qualified):
    return source.source._encode(
        {
            "protocol": PROTOCOL,
            "receipt": json.loads(qualified.receipt),
            "origins": _table(qualified.origins),
            "roles": _table(qualified.roles.rename("role").to_frame()),
            "role_name": qualified.roles.name,
            "unit_status": _table(qualified.unit_status.rename("status").to_frame()),
            "unit_evidence": _table(qualified.unit_evidence),
            "asec_raw": _table(qualified.asec_raw),
        }
    )


def _projection_bytes(projected):
    projection.spm_projection_seal(projected)
    return source.source._encode(
        {
            "protocol": PROTOCOL,
            "year": projected.year,
            "columns": {
                entity: _table(projected.columns[entity, name].to_frame())
                for entity, name in (
                    ("person", ROLE_INPUT),
                    ("spm_unit", UNIVERSE_INPUT),
                )
            },
            "nullable_source_roles": _table(
                projected.nullable_source_roles.rename("role").to_frame()
            ),
            "source_role_name": projected.nullable_source_roles.name,
            "placeholder_person_ids": projected.placeholder_person_ids,
            "unit_mapping": projected.unit_mapping,
        }
    )


def _params(qualified, outside_role_placeholder):
    _require(type(outside_role_placeholder) is bool, "NODE_PARAMETERS")
    evidence = json.loads(qualified.receipt)
    _require(
        tuple(
            evidence.get(k)
            for k in ("income_year", "acs_survey_year", "asec_survey_year")
        )
        == (2024, 2024, 2025),
        "SOURCE_PERIOD",
    )
    return {
        "protocol": PROTOCOL,
        "source_receipt_sha256": source.source._sha(qualified.receipt),
        "source_artifact_sha256": source.source._sha(_source_bytes(qualified)),
        "acs_profile_sha256": source.source._sha(
            source.source._encode(evidence["acs_profile"])
        ),
        "asec_scope_policy_sha256": source.source._sha(
            source.source._encode(evidence["asec_scope_policy"])
        ),
        "income_year": 2024,
        "acs_survey_year": 2024,
        "asec_survey_year": 2025,
        "outside_role_placeholder": outside_role_placeholder,
    }


def _project(qualified, receiving, outside_role_placeholder):
    result = projection.project_spm_inputs(
        qualified.source_frame,
        qualified.origins,
        qualified.roles,
        qualified.unit_status,
        receiving,
        source_year=2024,
        year=2024,
        outside_role_placeholder=outside_role_placeholder,
    )
    projection.spm_projection_seal(result)
    return result


def _source_edge():
    return ArtifactInput("spm_source", SOURCE_NODE, "source", SOURCE_TYPE)


def _projection_edge():
    return ArtifactInput("spm_projection", PROJECT_NODE, "projection", PROJECTION_TYPE)


def _inputs():
    provenance = projection.provenance
    return tuple(
        Slice(
            entity,
            tuple(
                function(entity)
                for function in (
                    provenance.support_source_id_column,
                    provenance.support_clone_index_column,
                    provenance.spine_source_id_column,
                    provenance.support_channel_column,
                )
            ),
        )
        for entity in _ENTITIES
    )


def spm_nodes(
    qualified, receiving, *, receiving_version, after, outside_role_placeholder
):
    """Declare source, complete-unit mapping and two-grain ordinary attachment."""
    spm_seal(qualified)
    _require(
        type(after) is ArtifactInput
        and type(receiving_version) is str
        and bool(receiving_version),
        "FRAGMENT_INPUT",
    )
    params = _params(qualified, outside_role_placeholder)
    _project(qualified, receiving, outside_role_placeholder)
    return (
        Node(
            SOURCE_NODE,
            _REFS[SOURCE_NODE],
            population=receiving_version,
            sources=(SOURCE_NAME,),
            params=params,
            artifact_inputs=(after,),
            artifact_outputs=(ArtifactOutput("source", SOURCE_TYPE),),
            description="Retain private original SPM roles and annual scope from the preparation owner.",
        ),
        Node(
            PROJECT_NODE,
            _REFS[PROJECT_NODE],
            population=receiving_version,
            inputs=_inputs(),
            params=params,
            artifact_inputs=(_source_edge(),),
            artifact_outputs=(ArtifactOutput("projection", PROJECTION_TYPE),),
            description="Map complete original persons, SPM units and households to both exact clones.",
        ),
        Node(
            ATTACH_NODE,
            _REFS[ATTACH_NODE],
            population=receiving_version,
            inputs=_inputs(),
            outputs=(
                Owned("person", ROLE_INPUT, "bool"),
                Owned("spm_unit", UNIVERSE_INPUT, "string"),
            ),
            params=params,
            artifact_inputs=(after, _source_edge(), _projection_edge()),
            artifact_outputs=(ArtifactOutput("attachment", ATTACHMENT_TYPE),),
            description="Attach only source roles and 2024 SPM scope; preserve all incumbent cells and weights.",
        ),
    )


def spm_result(qualified, node, artifacts, receiving=None):
    """Pure reconstruction; the host separately authenticates producer keys."""
    spm_seal(qualified)
    _require(node.id in _REFS and node.kernel == _REFS[node.id], "KERNEL_NODE")
    params = _params(qualified, node.params.get("outside_role_placeholder"))
    _require(dict(node.params) == params, "NODE_PARAMETERS")
    _require(
        set(artifacts) == {e.name for e in node.artifact_inputs}, "ARTIFACT_ROSTER"
    )
    source_bytes = _source_bytes(qualified)
    projected = None
    if node.id != SOURCE_NODE:
        _require(receiving is not None, "RECEIVING_REQUIRED")
        projected = _project(qualified, receiving, params["outside_role_placeholder"])
    mapped_bytes = None if projected is None else _projection_bytes(projected)
    expected = {
        (SOURCE_NODE, "source"): source_bytes,
        (PROJECT_NODE, "projection"): mapped_bytes,
    }
    for edge in node.artifact_inputs:
        value = artifacts[edge.name]
        _require(
            type(value) is ArtifactValue
            and value.type == edge.type
            and type(value.payload) is bytes,
            "ARTIFACT_TYPE",
        )
        if (edge.producer, edge.artifact) in expected:
            _require(
                value.payload == expected[edge.producer, edge.artifact],
                "ARTIFACT_PAYLOAD",
            )
    if node.id == SOURCE_NODE:
        return KernelResult(artifacts={"source": source_bytes})
    if node.id == PROJECT_NODE:
        return KernelResult(artifacts={"projection": mapped_bytes})
    evidence = {
        **params,
        "projection_sha256": source.source._sha(mapped_bytes),
        "receiving_version": node.population,
        "person_rows": len(projected.columns["person", ROLE_INPUT]),
        "spm_unit_rows": len(projected.columns["spm_unit", UNIVERSE_INPUT]),
        "outside_placeholders": len(projected.placeholder_person_ids),
        "complete_unit_clone_mapping": True,
        "weights_changed": False,
        "source_admission_issued": False,
        "release_eligible": False,
    }
    return KernelResult(
        columns=dict(projected.columns),
        artifacts={"attachment": source.source._encode(evidence)},
        receipt=evidence,
    )


def _result_seal(result):
    _require(
        all(
            type(name) is str and type(payload) is bytes
            for name, payload in result.artifacts.items()
        ),
        "RESULT_ARTIFACT_TYPE",
    )
    return (
        tuple(
            (entity, name, source.seals._table_seal(values.to_frame()))
            for (entity, name), values in sorted(result.columns.items())
        ),
        tuple(sorted(result.artifacts.items())),
        source.source._encode(dict(result.receipt)),
    )


def _validate_result(result, revalidate):
    """Pure output preservation across the owning host's final validation."""
    expected = _result_seal(result)
    revalidate()
    # No callbacks or source I/O may follow this output comparison.
    try:
        unchanged = _result_seal(result) == expected
    except (ValueError, TypeError, KeyError, AttributeError):
        unchanged = False
    _require(unchanged, "FINAL_RESULT_CHANGED")
    return result


def _receiving_context(context, nodes, ref, require_context):
    """Delegate actual source/producer authentication to the retaining host."""
    require_context(context)
    _require(context.node in nodes and context.node.kernel == ref, "KERNEL_NODE")
    if context.node.id == SOURCE_NODE:
        return None
    _require(set(context.tables) == set(_ENTITIES), "RECEIVING_TABLES")
    return SimpleNamespace(**context.tables)


class _SpmKernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def __init__(self, qualified, nodes, require_current, require_context, ref):
        qualified.validate()
        self.qualified = qualified
        self.retained_entry = source._ISSUED.get(qualified)
        self.objects = _objects(qualified)
        self.seal = spm_seal(qualified)
        self.nodes = nodes
        self.require_current = require_current
        self.require_context = require_context
        self.ref = ref
        source._check_retained_output(qualified, self.retained_entry)

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            source,
            projection,
            source.acs,
            source.asec_roles,
            source.seals,
            source.original,
            source.source,
            source.contract,
            projection.attachment,
            projection.provenance,
            population_ops,
            dependencies=self.capabilities.dependencies,
        )

    def _pure(self):
        source._check_retained_output(self.qualified, self.retained_entry)
        _require(
            all(
                a is b
                for a, b in zip(self.objects, _objects(self.qualified), strict=True)
            )
            and spm_seal(self.qualified) == self.seal,
            "QUALIFIED_CHANGED",
        )

    def run(self, context):
        self.qualified.validate()
        receiving = _receiving_context(
            context, self.nodes, self.ref, self.require_context
        )
        self.require_current()
        self._pure()
        result = spm_result(self.qualified, context.node, context.artifacts, receiving)

        def final_validation():
            self.qualified.validate()
            self.require_current()
            self._pure()

        return _validate_result(result, final_validation)


def spm_kernels(
    qualified,
    receiving,
    *,
    receiving_version,
    after,
    outside_role_placeholder,
    require_current,
    require_context,
):
    _require(callable(require_current) and callable(require_context), "HOST_CALLBACK")
    nodes = spm_nodes(
        qualified,
        receiving,
        receiving_version=receiving_version,
        after=after,
        outside_role_placeholder=outside_role_placeholder,
    )
    return tuple(
        _SpmKernel(qualified, nodes, require_current, require_context, node.kernel)
        for node in nodes
    )
