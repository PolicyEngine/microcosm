"""Source hours operations for the retained US enrichment host.

The source owner authenticates original literals and the complete age-15 donor
cohort. Recode kernels execute the pure transformations; attachment consumes
their checked columns and transports each original draw to its exact clones.
This fragment creates no source issuer, receiving branch or release authority.
"""

from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace

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

from . import current_survey_hours_source as source
from .graph_survey_population import SOURCE_NAME

hours, asec_hours = source.hours, source.asec_hours
require = source.require
PROTOCOL = "microcosm.us.native-usual-hours-graph.v1"
SOURCE_NODE = "survey_hours.source"
ASEC_NODE = "survey_hours.asec_recode"
ACS_NODE = "survey_hours.acs_recode_impute"
ATTACH_NODE = "survey_hours.attach"
PROJECTION_TYPE = ArtifactType("microcosm.us.native_usual_hours_projection", 1)
COLUMNS_TYPE = ArtifactType("microcosm.us.native_usual_hours_columns", 1)
ATTACHMENT_TYPE = ArtifactType("microcosm.us.native_usual_hours_attachment", 1)


def hours_seal(qualified):
    """Pure retained-value seal; the owning host separately validates sources."""
    require(
        type(qualified) is source.QualifiedSurveyHoursProposals,
        "RETAINED_OWNER_REQUIRED",
    )
    return (
        source.source._frame_identity(qualified.source_frame),
        *(
            source._table_seal(getattr(qualified, name))
            for name in (
                "origins",
                "acs_raw",
                "asec_selected_raw",
                "donor_raw",
                "person_hours",
            )
        ),
        source._proposal_seal(qualified.proposals),
        source._proposal_seal(qualified.asec_proposals),
        qualified.receipt,
    )


def _projection(qualified):
    # Private artifact: literal source identifiers never enter public params.
    return source.source._encode(
        {
            "source_receipt": json.loads(qualified.receipt),
            "acs": qualified.acs_raw.to_json(orient="table"),
            "asec": qualified.asec_selected_raw.to_json(orient="table"),
            "age15_donors": qualified.donor_raw.to_json(orient="table"),
        }
    )


def _bytes(table):
    return table.to_json(orient="table", double_precision=15).encode()


def _expected_columns(qualified, node_id):
    require(node_id in (ASEC_NODE, ACS_NODE), "RECODE_NODE")
    arm = "asec" if node_id == ASEC_NODE else "acs"
    return qualified.person_hours.loc[qualified.origins.source.eq(arm)].copy()


def _recode(qualified, node_id):
    """Execute source recoders, preserving literal history in the source owner."""
    if node_id == ASEC_NODE:
        proposals = tuple(
            asec_hours.propose_asec_usual_hours(
                row, under15_policy=qualified.asec_proposals.under15_policy
            )
            for row in qualified.asec_selected_raw.to_dict("records")
        )
        index = qualified.asec_selected_raw.index
        require(proposals == qualified.asec_proposals.proposals, "ASEC_PROPOSAL_PARITY")
    else:
        require(node_id == ACS_NODE, "RECODE_NODE")
        batch = hours.propose_acs_usual_hours(
            qualified.acs_raw.to_dict("records"),
            donors=qualified.donor_raw.to_dict("records"),
            age15_policy=qualified.proposals.age15_policy,
            under15_policy=qualified.proposals.under15_policy,
        )
        require(batch == qualified.proposals, "ACS_PROPOSAL_PARITY")
        proposals, index = batch.proposals, qualified.acs_raw.index
    columns = pd.DataFrame(
        {
            hours.TARGET: pd.array([p.hours for p in proposals], dtype="float64"),
            "hours_provenance": pd.array(
                [p.provenance for p in proposals], dtype=source.STRING
            ),
            "hours_policy": pd.array(
                [p.policy for p in proposals], dtype=source.STRING
            ),
        },
        index=index.copy(),
    )
    require(
        source._table_seal(columns)
        == source._table_seal(_expected_columns(qualified, node_id)),
        "RECODE_PARITY",
    )
    return columns


def _source_edge():
    return ArtifactInput("hours_projection", SOURCE_NODE, "projection", PROJECTION_TYPE)


def _column_edge(node_id):
    return ArtifactInput(
        "hours_asec" if node_id == ASEC_NODE else "hours_acs",
        node_id,
        "columns",
        COLUMNS_TYPE,
    )


def hours_nodes(qualified, receiving, *, receiving_version, after):
    """Declare private projection, two source transformations and clone attachment."""
    hours_seal(qualified)
    require(
        type(after) is ArtifactInput
        and type(receiving_version) is str
        and receiving_version,
        "FRAGMENT_INPUT",
    )
    require(
        not set(qualified.person_hours) & set(receiving.person),
        "ATTACH_OWNERSHIP_COLLISION",
    )
    evidence = json.loads(qualified.receipt)
    params = {
        "protocol": PROTOCOL,
        "source_receipt_sha256": source.source._sha(qualified.receipt),
        "projection_sha256": source.source._sha(_projection(qualified)),
        "income_year": evidence["income_year"],
        "acs_survey_year": evidence["acs_survey_year"],
        "asec_survey_year": evidence["asec_survey_year"],
        "age15_policy": evidence["age15_policy"],
        "under15_policy": evidence["under15_policy"],
        "seed": evidence["seed"],
    }
    projection = Node(
        SOURCE_NODE,
        "us.survey_hours.source@1",
        population=receiving_version,
        sources=(SOURCE_NAME,),
        params=params,
        artifact_inputs=(after,),
        artifact_outputs=(ArtifactOutput("projection", PROJECTION_TYPE),),
        description="Retain authenticated original ACS/ASEC hours literals and the complete ASEC age-15 donor cohort privately.",
    )
    recodes = tuple(
        Node(
            node_id,
            "us.survey_hours.recode@1",
            population=receiving_version,
            params={**params, "survey": arm},
            artifact_inputs=(_source_edge(),),
            artifact_outputs=(ArtifactOutput("columns", COLUMNS_TYPE),),
            description=description,
        )
        for node_id, arm, description in (
            (
                ASEC_NODE,
                "asec",
                "Recode ASEC source hours/nonwork and explicitly complete under-15 hours.",
            ),
            (
                ACS_NODE,
                "acs",
                "Recode ACS observed hours/nonwork, draw age-15 hours by original native key, and explicitly complete under-15 hours.",
            ),
        )
    )
    attach = Node(
        ATTACH_NODE,
        "us.survey_hours.attach@1",
        population=receiving_version,
        inputs=(
            Slice(
                "person",
                tuple(
                    f("person")
                    for f in (
                        source.attachment.provenance.support_source_id_column,
                        source.attachment.provenance.support_clone_index_column,
                        source.attachment.provenance.spine_source_id_column,
                        source.attachment.provenance.support_channel_column,
                    )
                ),
            ),
        ),
        outputs=tuple(
            Owned(
                "person",
                c,
                population_ops.token_for_dtype(qualified.person_hours[c].dtype),
            )
            for c in qualified.person_hours
        ),
        params=params,
        artifact_inputs=(after, _source_edge(), *(_column_edge(n.id) for n in recodes)),
        artifact_outputs=(ArtifactOutput("attachment", ATTACHMENT_TYPE),),
        description="Transport checked original hours and provenance to both exact clones without redrawing, changing weights, or replacing existing columns.",
    )
    return projection, *recodes, attach


def _check_artifacts(qualified, node, artifacts):
    require(set(artifacts) == {e.name for e in node.artifact_inputs}, "ARTIFACT_ROSTER")
    expected = {
        (SOURCE_NODE, "projection"): _projection(qualified),
        **{
            (n, "columns"): _bytes(_expected_columns(qualified, n))
            for n in (ASEC_NODE, ACS_NODE)
        },
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
        # The preceding housing edge is authenticated by the receiving host.


def hours_result(qualified, node, artifacts, people=None, *, execute_recoders=False):
    """Evaluate a kernel or reconstruct its independently qualified expectation."""
    _check_artifacts(qualified, node, artifacts)
    if node.id == SOURCE_NODE:
        return KernelResult(artifacts={"projection": _projection(qualified)})
    if node.id in (ASEC_NODE, ACS_NODE):
        table = (
            _recode(qualified, node.id)
            if execute_recoders
            else _expected_columns(qualified, node.id)
        )
        return KernelResult(artifacts={"columns": _bytes(table)})
    require(node.id == ATTACH_NODE and people is not None, "ATTACH_NODE")
    columns = pd.concat(
        [
            pd.read_json(
                io.StringIO(artifacts[_column_edge(n).name].payload.decode()),
                orient="table",
            )
            for n in (ASEC_NODE, ACS_NODE)
        ]
    ).reindex(qualified.origins.index)
    columns[hours.TARGET] = columns[hours.TARGET].astype("float64")
    for name in ("hours_provenance", "hours_policy"):
        columns[name] = columns[name].astype(source.STRING)
    require(
        source._table_seal(columns) == source._table_seal(qualified.person_hours),
        "ATTACH_COLUMNS",
    )
    attached = source.attachment.attach_columns(
        qualified.origins, SimpleNamespace(person=people), columns
    )
    evidence = {
        "protocol": PROTOCOL,
        "source_receipt_sha256": source.source._sha(qualified.receipt),
        "projection_sha256": source.source._sha(artifacts["hours_projection"].payload),
        "column_sha256": {
            n: source.source._sha(artifacts[_column_edge(n).name].payload)
            for n in (ASEC_NODE, ACS_NODE)
        },
        "receiving_version": node.population,
        "rows": len(people),
        "native_keyed_draws_preserved": True,
        "weights_changed": False,
        "prior_wages_consumed": False,
        "source_admission_issued": False,
        "release_eligible": False,
    }
    return KernelResult(
        columns=attached,
        artifacts={"attachment": source.source._encode(evidence)},
        receipt=evidence,
    )


def _result_seal(result):
    return (
        source._table_seal(
            pd.concat([v.rename(c) for (_, c), v in result.columns.items()], axis=1)
        )
        if result.columns
        else None,
        tuple(sorted(result.artifacts.items())),
        source.source._encode(dict(result.receipt)),
    )


class _HoursKernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def __init__(self, qualified, nodes, require_current, ref):
        qualified.validate()
        self.qualified, self.nodes, self.require_current, self.ref = (
            qualified,
            nodes,
            require_current,
            ref,
        )
        self.seal = hours_seal(qualified)

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            source,
            hours,
            asec_hours,
            source.attachment,
            source.original,
            source.source,
            population_ops,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        self.qualified.validate()
        self.require_current()
        require(hours_seal(self.qualified) == self.seal, "QUALIFIED_CHANGED")
        require(
            context.node in self.nodes and context.node.kernel == self.ref,
            "KERNEL_NODE",
        )
        result = hours_result(
            self.qualified,
            context.node,
            context.artifacts,
            context.tables.get("person"),
            execute_recoders=True,
        )
        stamp = _result_seal(result)
        self.qualified.validate()
        self.require_current()
        require(
            hours_seal(self.qualified) == self.seal and _result_seal(result) == stamp,
            "FINAL_HOURS_CHANGED",
        )
        return result


def hours_kernels(qualified, receiving, *, receiving_version, after, require_current):
    require(callable(require_current), "HOST_CALLBACK")
    nodes = hours_nodes(
        qualified, receiving, receiving_version=receiving_version, after=after
    )
    return tuple(
        _HoursKernel(qualified, nodes, require_current, ref)
        for ref in dict.fromkeys(n.kernel for n in nodes)
    )
