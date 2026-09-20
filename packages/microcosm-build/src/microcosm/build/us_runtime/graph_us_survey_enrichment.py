"""Fixed US post-PUF host for source-qualified survey enrichment fragments.

The owning boundary retains a real checked PUF run and authenticated source
projection. Models use separate original-design donor branches; attachment
adds declared amounts, coverage and participation to the existing receiving frame.
"""

from __future__ import annotations

import json
import sys
import weakref
from dataclasses import asdict, dataclass, replace
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import model_input, qrf, qrf_target
from microcosm.fit.graph_legacy_apply_matrix import (
    MATRIX_APPLY_STATE_TYPE,
    decode_matrix_apply_state,
)
from microcosm.fit.graph_legacy_qrf import (
    legacy_qrf_apply_matrix_nodes,
    legacy_qrf_train_nodes,
)
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    ContentStore,
    Determinism,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    Slice,
    StructuralDelta,
    codecs,
    compile_graph,
    run_graph,
    source_hash,
)
from microcosm.graph import population as population_ops
from microcosm.graph.executor import (
    _all_node_keys,
    _source_paths_and_keys,
    _structural_columns,
)
from microcosm.graph.serialize import graph_to_json

from . import current_survey_amounts as values
from . import graph_current_survey_health as health_graph
from . import graph_current_survey_health_completion as health_completion_graph
from . import graph_current_survey_hours as hours_graph
from . import graph_current_survey_housing as housing_graph
from . import graph_current_survey_immigration as immigration_graph
from . import graph_current_survey_predictors as predictor_graph
from . import graph_current_survey_race_hispanic as race_graph
from . import graph_current_survey_sex as sex_graph
from . import graph_current_survey_spm as spm_graph
from . import graph_current_survey_state as state_graph

parent = values.parent_host
physical = values.physical
require = values.require
PROJECTION_NODE = "survey_amounts.source_projection"
FULL_DONOR_SOURCE_NODE = "survey_amounts.full_original_asec_source"
ATTACH_NODE = "survey_amounts.attach"
CHILD_VERSION_NODE = "survey_amounts.child_support.version"
CHILD_ATTACH_NODE = "survey_amounts.child_support.attach"
CANONICAL_VERSION_NODE = "survey_amounts.canonical.version"
CANONICAL_ATTACH_NODE = "survey_amounts.canonical.attach"
PROJECTION_TYPE = ArtifactType("microcosm.us.current_survey_amount_projection", 1)
ATTACHMENT_TYPE = ArtifactType("microcosm.us.current_survey_amount_attachment", 1)
STATE_VERSION_NODE = "survey_geography.canonical_state_version"
STATE_VERSION_TYPE = ArtifactType("microcosm.us.current_survey_state_version", 1)
_ISSUED = {}


def _live():
    result = []
    for module in (
        sys.modules[__name__],
        values,
        values.unemployment,
        values.workers_compensation,
        values.workers_compensation.mapper,
        values.veterans,
        values.veterans.routing,
        values.child_source,
        values.child_source.routing,
        values.child_mapper,
        values.full_donor,
        values.full_donor.demographics,
        values.full_donor.demographics.demographic,
        values.full_donor.demographics.household,
        health_graph,
        health_graph.health,
        health_graph.source,
        health_completion_graph,
        health_completion_graph.values,
        health_completion_graph.values.demographics,
        hours_graph,
        hours_graph.source,
        hours_graph.hours,
        hours_graph.asec_hours,
        housing_graph,
        housing_graph.housing,
        housing_graph.participation,
        spm_graph,
        immigration_graph,
        immigration_graph.owner,
        sex_graph,
        sex_graph.source,
        sex_graph.comparison,
        race_graph,
        race_graph.source,
        state_graph,
    ):
        for name, item in vars(module).items():
            if type(item) is FunctionType:
                result.append(
                    (
                        module.__name__,
                        name,
                        values.predictors.source._function_seal(item),
                    )
                )
            elif isinstance(item, type) and item.__module__ == module.__name__:
                result.append((module.__name__, name, item))
                for member, function in vars(item).items():
                    if isinstance(function, (classmethod, staticmethod)):
                        function = function.__func__
                    if isinstance(function, property):
                        function = function.fget
                    if type(function) is FunctionType:
                        result.append(
                            (
                                module.__name__,
                                name,
                                member,
                                values.predictors.source._function_seal(function),
                            )
                        )
    result.append(
        (
            "amount_configuration",
            values.SEED,
            tuple((g.key, g.fields, g.targets) for g in values.GROUPS),
            values.UC_REPORT_COLUMNS,
            values.unemployment.PROTOCOL,
            values.unemployment.READ_COLUMNS,
            codec.encode_json(values.unemployment.DICTIONARY),
            values.workers_compensation.PROTOCOL,
            values.workers_compensation.READ_COLUMNS,
            codec.encode_json(values.workers_compensation.DICTIONARY),
            values.veterans.PROTOCOL,
            values.veterans.READ_COLUMNS,
            values.veterans.OUTPUT,
            type(values.veterans.ALLOCATION_CODES),
            tuple(
                (name, type(codes), codes)
                for name, codes in values.veterans.ALLOCATION_CODES.items()
            ),
            type(values.veterans.DICTIONARY),
            codec.encode_json(values.veterans.DICTIONARY),
            values.child_source.PROTOCOL,
            values.child_source.READ_COLUMNS,
            values.child_source.AMOUNT_FIELDS,
            values.child_mapper.US_CHILD_SUPPORT_OUTPUT_COLUMNS,
            tuple(
                (
                    name,
                    type(getattr(values.child_source, name)),
                    codec.encode_json(getattr(values.child_source, name)),
                )
                for name in (
                    "RESPONSE_ENTRIES",
                    "RESPONSE_MEANINGS",
                    "ALLOCATION_ENTRIES",
                    "TOPCODE_ENTRIES",
                )
            ),
            values.child_source.routing.DOMAINS_RESOURCE,
            values.child_source.routing.DOMAINS_SHA256,
            values.child_source.routing.DICTIONARY_SHA256,
            values.child_source.routing.KNOWN_AMOUNT_STATUSES,
            values.child_source.routing.CURRENT_INCOME_YEAR,
            values.child_source.routing.COORDINATE_COLUMNS,
            values.child_source.routing.RECEIPT_CODE_DOMAIN,
            values.child_source.routing.ALLOCATION_ANNVAL_CODES,
            values.child_source._cached_amount_entries_json,
            values.predictors.source._function_seal(
                values.child_source._cached_amount_entries_json.__wrapped__
            ),
            values.full_donor.PROTOCOL,
            type(values.full_donor.US_STATE_NUMERIC_FIPS_TO_POSTAL),
            tuple(sorted(values.full_donor.US_STATE_NUMERIC_FIPS_TO_POSTAL.items())),
            tuple(
                (
                    name,
                    type(getattr(values.full_donor.demographics.demographic, name)),
                    tuple(
                        vars(
                            getattr(values.full_donor.demographics.demographic, name)
                        ).items()
                    ),
                )
                for name in ("A_SEX", "AXSEX")
            ),
        )
    )
    result.append(
        (
            "health_configuration",
            health_graph.health.PROTOCOL,
            tuple(
                (f.output, f.asec, f.acs, f.acs_gap) for f in health_graph.health.FIELDS
            ),
            health_graph.health.RAW_COLUMNS,
            health_graph.health.SOURCE_PREFIX,
            health_graph.source.ASEC_COLUMNS,
            health_graph.source.ACS_COLUMNS,
            health_completion_graph.values.PROTOCOL,
            tuple(
                (f.output, f.asec, f.acs, f.acs_gap)
                for f in health_completion_graph.values.FIELDS
            ),
            health_completion_graph.values.FEATURES,
            health_completion_graph.values.TARGETS,
            health_completion_graph.values.ELIGIBLE,
            health_completion_graph.values.SEED,
            tuple(
                sorted(
                    health_completion_graph.values.US_STATE_NUMERIC_FIPS_TO_POSTAL.items()
                )
            ),
            health_completion_graph.PHASE,
        )
    )
    result.append(
        (
            "hours_configuration",
            hours_graph.PROTOCOL,
            hours_graph.source._live(),
        )
    )
    result.append(("spm_configuration", spm_graph.PROTOCOL, spm_graph.source._live()))
    result.append(("immigration_configuration", immigration_graph.configuration()))
    result.append(("sex_configuration", sex_graph.source._live()))
    result.append(("race_hispanic_configuration", race_graph.source._live()))
    result.append(
        (
            "state_configuration",
            state_graph.PROTOCOL,
            state_graph.NODE,
            state_graph.REF,
            state_graph.INPUTS,
            state_graph.OUTPUT,
            state_graph.BINDING_TYPE,
            state_graph.ATOMIC_GEOGRAPHY_VALIDATION_TYPE,
            type(state_graph.US_STATE_FIPS_TO_POSTAL),
            tuple(sorted(state_graph.US_STATE_FIPS_TO_POSTAL.items())),
            STATE_VERSION_NODE,
            STATE_VERSION_TYPE,
        )
    )
    result.append(
        (
            "housing_configuration",
            housing_graph.housing.PROTOCOL,
            housing_graph.housing.TARGET,
            housing_graph.SEED,
            housing_graph.PHASE,
            housing_graph.participation.HOUSING_PARTICIPATION_ASSUMPTIONS,
            housing_graph.housing.live(),
        )
    )
    return tuple(result)


def _amount_edge():
    return ArtifactInput(
        "amount_attachment", ATTACH_NODE, "attachment", ATTACHMENT_TYPE
    )


def _housing_after_edge():
    return ArtifactInput(
        "health_attachment",
        health_graph.ATTACH_NODE,
        "attachment",
        health_graph.ATTACHMENT_TYPE,
    )


def _hours_after_edge():
    return ArtifactInput(
        "housing_attachment",
        housing_graph.ATTACH_NODE,
        "attachment",
        housing_graph.ATTACHMENT_TYPE,
    )


def _spm_after_edge():
    return ArtifactInput(
        "hours_attachment",
        hours_graph.ATTACH_NODE,
        "attachment",
        hours_graph.ATTACHMENT_TYPE,
    )


def _immigration_after_edge(spm_enabled):
    if spm_enabled:
        return ArtifactInput(
            "spm_attachment",
            spm_graph.ATTACH_NODE,
            "attachment",
            spm_graph.ATTACHMENT_TYPE,
        )
    return _spm_after_edge()


def _sex_after_edge(spm_enabled, immigration_enabled):
    if immigration_enabled:
        return ArtifactInput(
            "previous_attachment",
            immigration_graph.ATTACH_NODE,
            "attachment",
            immigration_graph.ATTACHMENT_TYPE,
        )
    return _immigration_after_edge(spm_enabled)


def _race_after_edge(spm_enabled, immigration_enabled, sex_enabled):
    if sex_enabled:
        return ArtifactInput(
            "previous_attachment",
            sex_graph.ATTACH_NODE,
            "attachment",
            sex_graph.ATTACHMENT_TYPE,
        )
    return _sex_after_edge(spm_enabled, immigration_enabled)


def _state_after_edge(spm_enabled, immigration_enabled, sex_enabled, race_enabled):
    if race_enabled:
        return ArtifactInput(
            "previous_attachment",
            race_graph.ATTACH_NODE,
            "attachment",
            race_graph.ATTACHMENT_TYPE,
        )
    return _race_after_edge(spm_enabled, immigration_enabled, sex_enabled)


def _state_version_node(*, receiving_version, after):
    return Node(
        STATE_VERSION_NODE,
        CurrentSurveyStateVersionKernel.ref,
        structural=StructuralDelta.FILTER,
        base=receiving_version,
        inputs=(Slice("person", ("person_support_clone_index",)),),
        params={"protocol": state_graph.PROTOCOL},
        artifact_inputs=(after,),
        artifact_outputs=(ArtifactOutput("version", STATE_VERSION_TYPE),),
        description="Keep every existing support clone and weight in an explicit population version before replacing the canonical state input; no geography assignment.",
    )


def _state_nodes(receiving, *, receiving_version, after):
    version = _state_version_node(receiving_version=receiving_version, after=after)
    binding = state_graph.state_binding_node(
        receiving,
        population=version.id,
        after=ArtifactInput("state_version", version.id, "version", STATE_VERSION_TYPE),
    )
    return version, binding


def _state_version_result(node, person, artifacts):
    require(
        len(node.artifact_inputs) == 1
        and node
        == _state_version_node(
            receiving_version=node.base, after=node.artifact_inputs[0]
        )
        and tuple(person.columns) == ("person_id", "person_support_clone_index")
        and person.person_id.is_unique
        and person.person_id.dtype == population_ops.dtype_for_token("int64")
        and set(artifacts) == {node.artifact_inputs[0].name},
        "STATE_VERSION_CONTEXT",
    )
    edge = node.artifact_inputs[0]
    value = artifacts[edge.name]
    require(
        type(value) is state_graph.ArtifactValue
        and value.type == edge.type
        and type(value.payload) is bytes,
        "STATE_VERSION_ARTIFACT",
    )
    receipt = {
        "protocol": state_graph.PROTOCOL,
        "scope": "canonical_state_population_boundary",
        "persons": len(person),
        "selection": "keep_all",
        "new_geography_assignment": False,
        "source_admission_issued": False,
        "release_eligible": False,
    }
    return KernelResult(
        keep=pd.Series(
            True, index=pd.Index(person.person_id, name="person_id"), dtype="bool"
        ),
        artifacts={"version": codec.encode_json(receipt)},
        receipt=receipt,
    )


def _state_expected_population(incoming, node, artifacts, persisted):
    """Independently reconstruct the complete keep-all or canonical transition."""
    if node.id == STATE_VERSION_NODE:
        result = _state_version_result(
            node,
            incoming.frame.person[["person_id", "person_support_clone_index"]],
            artifacts,
        )
    else:
        require(node.id == state_graph.NODE, "STATE_RECONSTRUCTION_NODE")
        result = state_graph.state_binding_result(
            node, incoming.frame.table("household"), artifacts
        )
    require(persisted == result.artifacts, "STATE_RESULT_ARTIFACT")
    expected = health_completion_graph.expected_population(incoming, node, result)
    if node.structural is StructuralDelta.FILTER:
        # Selecting every person can still drop an orphan group. A canonical
        # representation boundary must preserve every entity axis, not prune it.
        require(
            all(
                expected.frame.table(entity)[
                    expected.frame.schema.entity_id_column(entity)
                ].equals(
                    incoming.frame.table(entity)[
                        incoming.frame.schema.entity_id_column(entity)
                    ]
                )
                for entity in incoming.frame.entities
            ),
            "STATE_VERSION_AXIS_CHANGED",
        )
    return expected


def _spm_configuration(acs_profile, asec_scope_policy, outside_role_placeholder):
    """An explicit opt-in; absent ASEC policy retains UNRESOLVED source scope."""
    if all(
        x is None for x in (acs_profile, asec_scope_policy, outside_role_placeholder)
    ):
        return None
    require(
        type(acs_profile) is spm_graph.source.acs.ACSAnalysisProfile
        and type(outside_role_placeholder) is bool,
        "SPM_CONFIGURATION",
    )
    acs_profile.__post_init__()
    spm_graph.source._policy(asec_scope_policy)
    return codec.encode_json(
        {
            "acs_profile": asdict(acs_profile),
            "asec_scope_policy": None
            if asec_scope_policy is None
            else asdict(asec_scope_policy),
            "outside_role_placeholder": outside_role_placeholder,
        }
    )


def _upstream_edge():
    return ArtifactInput(
        "puf_finalization",
        parent.attach.ATTACH_NODE,
        "finalization",
        parent.attach.FINALIZATION_TYPE,
    )


def _projection_edge():
    return ArtifactInput("projection", PROJECTION_NODE, "projection", PROJECTION_TYPE)


def _ids(group):
    base = "survey_amounts." + group.spec.key
    return base + ".donor", base + ".columns", base + ".fit", base + ".apply"


def _full_donor_inputs(frame):
    """Declare actual source data; structural-only groups need no Slice."""
    result = []
    for entity in frame.schema.entities:
        id_column = frame.schema.entity_id_column(entity)
        structural = {id_column}
        if entity == frame.schema.person_entity:
            structural.update(
                frame.schema.membership_column(e) for e in frame.schema.group_entities
            )
        columns = tuple(c for c in frame.table(entity) if c not in structural)
        if columns:
            result.append(Slice(entity, columns))
    return tuple(result)


def _full_donor_context(context, expected):
    """Check the declared all-row donor Slice, weights and strata exactly.

    Structural-only group tables remain sealed on the host's full Frame and
    checked by complete materialized reconstruction; they are not fake inputs.
    """
    inputs = _full_donor_inputs(expected)
    require(context.node.inputs == inputs, "FULL_DONOR_INPUTS")
    require(set(context.tables) == {s.entity for s in inputs}, "FULL_DONOR_TABLES")
    weights = {}
    for selection in inputs:
        entity = selection.entity
        columns = [expected.schema.entity_id_column(entity)]
        if entity == expected.schema.person_entity:
            columns.extend(
                expected.schema.membership_column(e)
                for e in expected.schema.group_entities
            )
        columns.extend(selection.columns)
        table = expected.table(entity).loc[[True] * expected.n(entity), columns]
        require(
            physical._table_stamp(context.tables[entity])
            == physical._table_stamp(table),
            "FULL_DONOR_TABLE",
        )
        try:
            weights[entity] = expected.resolve_weights(entity)
        except ValueError:
            if entity in expected.weighted_entities:
                raise
    require(set(context.weights) == set(weights), "FULL_DONOR_WEIGHTS")
    for entity, weight in weights.items():
        actual = context.weights[entity]
        require(
            actual.kind is weight.kind
            and actual.values.dtype == weight.values.dtype
            and actual.values.shape == weight.values.shape
            and actual.values.tobytes() == weight.values.tobytes(),
            "FULL_DONOR_WEIGHT_VALUES",
        )
    expected_strata = expected.strata.loc[[True] * len(expected.strata)]
    require(
        physical._table_stamp(context.strata.to_frame())
        == physical._table_stamp(expected_strata.to_frame()),
        "FULL_DONOR_STRATA",
    )


def _child_enabled(qualified):
    return any(g.spec.key == "child_support" for g in qualified.groups)


def _canonical_enabled(qualified):
    return bool(values.canonical_outputs(qualified))


def _canonical_ids(qualified):
    if any(g.spec.key == "veterans_benefits" for g in qualified.groups):
        return CANONICAL_VERSION_NODE, CANONICAL_ATTACH_NODE
    return CHILD_VERSION_NODE, CHILD_ATTACH_NODE


def amount_nodes(qualified, receiving, *, parent_digest, n_estimators):
    require(
        type(n_estimators) is int and n_estimators > 0 and codec._hash(parent_digest),
        "NODE_PARAMETERS",
    )
    params = {
        "projection_sha256": codec.sha(qualified.projection),
        "parent_sha256": parent_digest,
        "n_estimators": n_estimators,
        "groups": tuple(g.spec.key for g in qualified.groups),
        "protocol": values.PROTOCOL,
    }
    nodes = [
        Node(
            PROJECTION_NODE,
            CurrentSurveyAmountProjectionKernel.ref,
            population=parent.attach.FILTER_NODE,
            inputs=predictor_graph._inputs(receiving),
            params=params,
            artifact_inputs=(_upstream_edge(),),
            artifact_outputs=(
                ArtifactOutput("projection", PROJECTION_TYPE),
                *(
                    ArtifactOutput(
                        g.spec.key + "_matrix", model_input.RECIPIENT_MATRIX_TYPE
                    )
                    for g in qualified.groups
                ),
            ),
            description="Qualify original current survey amount and reporting-universe evidence; preserve source unknowns.",
        )
    ]
    if qualified.full_donor_source_frame is not None:
        nodes.append(
            Node(
                FULL_DONOR_SOURCE_NODE,
                CurrentSurveyFullAmountSourceKernel.ref,
                structural=StructuralDelta.CREATE,
                sources=(health_graph.SOURCE_NAME,),
                params=params,
                outputs=tuple(
                    Owned(
                        selection.entity,
                        c,
                        population_ops.token_for_dtype(
                            qualified.donor_source_frame.table(selection.entity)[
                                c
                            ].dtype
                        ),
                    )
                    for selection in _full_donor_inputs(qualified.donor_source_frame)
                    for c in selection.columns
                    if c
                    != qualified.donor_source_frame.schema.entity_id_column(
                        selection.entity
                    )
                ),
                artifact_inputs=(_projection_edge(),),
                description="Borrow complete original current ASEC DESIGN support independently of selected receiving households.",
            )
        )
    attach_edges = [_projection_edge(), _upstream_edge()]
    for group in qualified.groups:
        donor, columns, fit_prefix, apply_prefix = _ids(group)
        group_params = {**params, "group": group.spec.key}
        matrix_edge = ArtifactInput(
            group.spec.key + "_matrix",
            PROJECTION_NODE,
            group.spec.key + "_matrix",
            model_input.RECIPIENT_MATRIX_TYPE,
        )
        nodes.extend(
            (
                Node(
                    donor,
                    CurrentSurveyAmountDonorKernel.ref,
                    base=(
                        values.predictors.host.survey_graph.CREATE_NODE
                        if qualified.full_donor_source_frame is None
                        else FULL_DONOR_SOURCE_NODE
                    ),
                    structural=StructuralDelta.FILTER,
                    mass="free",
                    inputs=(
                        predictor_graph._inputs(qualified.donor_source_frame)
                        if qualified.full_donor_source_frame is None
                        else _full_donor_inputs(qualified.donor_source_frame)
                    ),
                    params=group_params,
                    artifact_inputs=(_projection_edge(),),
                    description="Select source ASEC persons with jointly known targets; retain original design weights before allocation and clones.",
                ),
                Node(
                    columns,
                    CurrentSurveyAmountColumnsKernel.ref,
                    population=donor,
                    inputs=(
                        predictor_graph._inputs(group.donor_frame)
                        if qualified.full_donor_source_frame is None
                        else _full_donor_inputs(group.donor_frame)
                    ),
                    params=group_params,
                    outputs=tuple(
                        Owned("person", c, "float64")
                        for c in (*qualified.features, *group.spec.targets)
                    ),
                    artifact_inputs=(_projection_edge(),),
                    description="Map qualified current amounts and source predictors without numeric or reporting-unknown fills.",
                ),
            )
        )
        fits = legacy_qrf_train_nodes(
            fit_prefix,
            population=donor,
            entity="person",
            predictors=qualified.features,
            targets=group.spec.targets,
            seed=values.SEED,
            n_estimators=n_estimators,
            zero_atol=0,
            phase="current_survey_amounts." + group.spec.key,
        )
        applies = legacy_qrf_apply_matrix_nodes(
            apply_prefix,
            population=parent.attach.FILTER_NODE,
            fit_nodes=fits,
            matrix_producer=PROJECTION_NODE,
            matrix_artifact=group.spec.key + "_matrix",
            seed=values.SEED,
            phase="current_survey_amounts." + group.spec.key,
        )
        nodes.extend((*fits, *applies))
        attach_edges.append(matrix_edge)
        for i, node in enumerate(applies):
            attach_edges.extend(
                (
                    ArtifactInput(
                        f"{group.spec.key}_raw_{i}",
                        node.id,
                        "raw_draw",
                        codec.RAW_TARGET_TYPE,
                    ),
                    ArtifactInput(
                        f"{group.spec.key}_state_{i}",
                        node.id,
                        "apply_state",
                        MATRIX_APPLY_STATE_TYPE,
                    ),
                )
            )
    columns = pd.concat([qualified.native, qualified.reports], axis=1)
    child_columns = (
        columns.loc[:, list(values.canonical_outputs(qualified))]
        if _canonical_enabled(qualified)
        else None
    )
    if child_columns is not None:
        columns = columns.drop(columns=list(child_columns))
    nodes.append(
        Node(
            ATTACH_NODE,
            CurrentSurveyAmountAttachKernel.ref,
            population=parent.attach.FILTER_NODE,
            inputs=predictor_graph._inputs(receiving),
            params=params,
            outputs=tuple(
                Owned(
                    "person",
                    c,
                    values.attachment_dtype(
                        qualified, receiving, c, str(columns[c].dtype)
                    ),
                    rewrite=(c in receiving.person),
                )
                for c in columns
            ),
            artifact_inputs=tuple(attach_edges),
            artifact_outputs=(ArtifactOutput("attachment", ATTACHMENT_TYPE),),
            description="Join source observations and ACS conditional draws to both support clones; retain all PUF and source columns, geography, weights and ledger.",
        )
    )
    if child_columns is not None:
        nodes.extend(
            (
                Node(
                    _canonical_ids(qualified)[0],
                    CurrentSurveyChildVersionKernel.ref,
                    base=parent.attach.FILTER_NODE,
                    structural=StructuralDelta.FILTER,
                    inputs=predictor_graph._inputs(receiving),
                    params=params,
                    artifact_inputs=(_projection_edge(),),
                    description="Keep every receiving row in a derived version after prior enrichment; preserve all memberships, weights and source cells before canonical amount writes.",
                ),
                Node(
                    _canonical_ids(qualified)[1],
                    CurrentSurveyChildAttachKernel.ref,
                    population=_canonical_ids(qualified)[0],
                    inputs=predictor_graph._inputs(receiving),
                    params=params,
                    outputs=tuple(
                        Owned(
                            "person",
                            c,
                            values.attachment_dtype(
                                qualified, receiving, c, str(child_columns[c].dtype)
                            ),
                            rewrite=(c in receiving.person),
                        )
                        for c in child_columns
                    ),
                    artifact_inputs=tuple(attach_edges),
                    artifact_outputs=(ArtifactOutput("attachment", ATTACHMENT_TYPE),),
                    description="Attach qualified original canonical amounts and ACS draws to existing clones in a new version; explicitly replace only selected child/veterans outputs.",
                ),
            )
        )

    return tuple(nodes)


class Boundary:
    """Internal retained-value seam; a detached projection cannot construct it."""

    def __init__(
        self,
        run,
        *,
        groups,
        n_estimators,
        spm_acs_profile=None,
        spm_asec_scope_policy=None,
        spm_outside_role_placeholder=None,
        immigration_transfer=None,
        health_completion=False,
        demographic_inputs=False,
        race_hispanic_inputs=False,
        full_original_amount_donors=False,
        canonical_state_input=False,
    ):
        require(type(canonical_state_input) is bool, "CANONICAL_STATE_OPTION")
        self.canonical_state_input = canonical_state_input
        require(
            type(full_original_amount_donors) is bool, "FULL_ORIGINAL_DONORS_OPTION"
        )
        self.full_original_amount_donors = full_original_amount_donors
        require(type(demographic_inputs) is bool, "DEMOGRAPHIC_OPTION")
        require(type(race_hispanic_inputs) is bool, "RACE_HISPANIC_OPTION")
        live = _live()
        require(type(health_completion) is bool, "HEALTH_COMPLETION_OPTION")
        self.health_completion_enabled = health_completion
        self.demographic_inputs = demographic_inputs
        self.race_hispanic_inputs = race_hispanic_inputs
        self.spm_options = (
            spm_acs_profile,
            spm_asec_scope_policy,
            spm_outside_role_placeholder,
        )
        self.spm_configuration = _spm_configuration(*self.spm_options)
        self.run = run
        immigration_entry = (
            None
            if immigration_transfer is None
            else immigration_graph.retained_entry(immigration_transfer)
        )
        self.parent_view = parent.check_survey_puf55_run(run)
        self.parent_entry = parent._run_entry(run)
        if canonical_state_input:
            # Refuse unsupported incumbent storage before any new model fitting.
            state_graph.bind_state_fips(run.population.frame.table("household"))
        self.immigration_transfer = immigration_transfer
        self.immigration_entry = self.immigration_origins = (
            self.immigration_pairs_bytes
        ) = None
        self.immigration_nodes = ()
        if immigration_transfer is not None:
            self.immigration_entry = immigration_entry
            self.immigration_origins = (
                self.immigration_entry[2].acs_entry[2].qualified.origins
            )
            self.immigration_pairs_bytes = immigration_graph.pair_bytes(
                immigration_transfer.pairs
            )
            self._immigration_pure()
            immigration_transfer.validate()
            self._immigration_pure()
        self.qualified = values.qualify_current_survey_amounts(
            run, groups=groups, full_original_donors=full_original_amount_donors
        )
        self.qualified_stamp = values.seal(self.qualified)
        self.parent_stamp = physical._population_stamp(run.population)
        self.preparation = run.financial_run.prefix.preparation
        self.sex = (
            sex_graph.source.qualify_current_survey_sex(self.preparation)
            if demographic_inputs
            else None
        )
        self.sex_stamp = (
            sex_graph.source.seal(self.sex) if self.sex is not None else None
        )
        self.sex_nodes = ()
        self.race = (
            race_graph.source.qualify_current_survey_race_hispanic(self.preparation)
            if race_hispanic_inputs
            else None
        )
        self.race_stamp = (
            race_graph.source.seal(self.race) if self.race is not None else None
        )
        self.race_nodes = ()
        self.health = health_graph.qualify_health_coverage(self.preparation)
        self.health_stamp = health_graph.health_coverage_seal(self.health)
        self.health_completion = (
            health_completion_graph.values.qualify_current_survey_health_completion(
                self.preparation, self.health
            )
            if health_completion
            else None
        )
        self.health_completion_stamp = (
            None
            if self.health_completion is None
            else health_completion_graph.values.seal(self.health_completion)
        )
        self.housing = housing_graph.housing.qualify_current_survey_housing(run)
        self.housing_stamp = housing_graph.housing.seal(self.housing)
        self.hours = hours_graph.source.qualify_current_survey_hours(
            self.preparation,
            age15_policy=hours_graph.hours.AGE15_POLICY,
            under15_policy=hours_graph.hours.UNDER15_POLICY,
        )
        self.hours_stamp = hours_graph.hours_seal(self.hours)
        self.spm = None
        self.spm_entry = self.spm_objects = self.spm_stamp = None
        self.spm_nodes = ()
        if self.spm_configuration is not None:
            self.spm = spm_graph.source.qualify_current_survey_spm(
                self.preparation,
                acs_profile=spm_acs_profile,
                asec_scope_policy=spm_asec_scope_policy,
            )
            self.spm_entry = spm_graph.source._ISSUED.get(self.spm)
            self.spm_objects = self._spm_objects()
            self.spm_stamp = spm_graph.spm_seal(self.spm)
            # Refuse incompatible receiving membership or missing roles before
            # compiling or running any new enrichment fits.
            preflight = spm_graph.projection.project_spm_inputs(
                self.spm.source_frame,
                self.spm.origins,
                self.spm.roles,
                self.spm.unit_status,
                run.population.frame,
                source_year=2024,
                year=2024,
                outside_role_placeholder=spm_outside_role_placeholder,
            )
            spm_graph.projection.validate_spm_projection(preflight, self.spm.validate)
            self.spm_nodes = spm_graph.spm_nodes(
                self.spm,
                run.population.frame,
                receiving_version=parent.attach.FILTER_NODE,
                after=_spm_after_edge(),
                outside_role_placeholder=spm_outside_role_placeholder,
            )
        self.n_estimators = n_estimators
        self.amount_nodes = amount_nodes(
            self.qualified,
            run.population.frame,
            parent_digest=self.parent_view.digest,
            n_estimators=n_estimators,
        )
        self.health_completion_nodes = self._health_completion_nodes()
        self.health_nodes = self._health_nodes()
        self.housing_nodes = housing_graph.housing_nodes(
            self.housing,
            run.population.frame,
            after=_housing_after_edge(),
            n_estimators=n_estimators,
        )
        self.hours_nodes = hours_graph.hours_nodes(
            self.hours,
            run.population.frame,
            receiving_version=parent.attach.FILTER_NODE,
            after=_hours_after_edge(),
        )
        if self.immigration_transfer is not None:
            self.immigration_nodes = immigration_graph.immigration_nodes(
                self.immigration_transfer,
                run.population.frame,
                receiving_version=parent.attach.FILTER_NODE,
                after=_immigration_after_edge(self.spm is not None),
            )
        self.nodes = (
            *self.amount_nodes,
            *self.health_nodes,
            *self.housing_nodes,
            *self.hours_nodes,
            *self.spm_nodes,
            *self.immigration_nodes,
        )
        if self.sex is not None:
            self.sex_nodes = sex_graph.sex_nodes(
                self.sex,
                run.population.frame,
                receiving_version=parent.attach.FILTER_NODE,
                after=_sex_after_edge(
                    self.spm is not None, self.immigration_transfer is not None
                ),
            )
            self.nodes = (*self.nodes, *self.sex_nodes)
        if self.race is not None:
            self.race_nodes = race_graph.race_hispanic_nodes(
                self.race,
                run.population.frame,
                receiving_version=parent.attach.FILTER_NODE,
                after=_race_after_edge(
                    self.spm is not None,
                    self.immigration_transfer is not None,
                    self.sex is not None,
                ),
            )
            self.nodes = (*self.nodes, *self.race_nodes)
        self.state_nodes = self._state_nodes()
        self.nodes = (*self.nodes, *self.state_nodes)
        self.declaration = tuple(self.nodes)
        self.live = _live()
        require(self.live == live, "QUALIFIER_CALLBACK_CHANGED_IMPLEMENTATION")
        self.compiled = self.kernels = self.store = None
        self.paths = self.source_keys = self.keys = self.implementations = None
        self.parent_objects = (
            run.population,
            run.compiled,
            run.manifest,
            run.store,
            run.kernels,
            run.sources,
        )
        self._spm_pure()
        self._immigration_pure()

    def _state_nodes(self):
        require(type(self.canonical_state_input) is bool, "CANONICAL_STATE_OPTION")
        if not self.canonical_state_input:
            return ()
        child = _canonical_enabled(self.qualified)
        return _state_nodes(
            self.run.population.frame,
            receiving_version=_canonical_ids(self.qualified)[0]
            if child
            else parent.attach.FILTER_NODE,
            after=(
                ArtifactInput(
                    "child_support_attachment",
                    _canonical_ids(self.qualified)[1],
                    "attachment",
                    ATTACHMENT_TYPE,
                )
                if child
                else _state_after_edge(
                    self.spm is not None,
                    self.immigration_transfer is not None,
                    self.sex is not None,
                    self.race is not None,
                )
            ),
        )

    def _health_completion_nodes(self):
        if self.health_completion is None:
            return ()
        return health_completion_graph.nodes(
            self.health_completion,
            self.health,
            receiving_version=parent.attach.FILTER_NODE,
            after=_amount_edge(),
            n_estimators=self.n_estimators,
        )

    def _health_nodes(self):
        observed = health_graph.health_coverage_nodes(
            self.health,
            receiving_version=parent.attach.FILTER_NODE,
            after=_amount_edge(),
        )
        return (
            observed
            if self.health_completion is None
            else (*observed[:-1], *self._health_completion_nodes())
        )

    def _sex_pure(self):
        require(type(self.demographic_inputs) is bool, "DEMOGRAPHIC_OPTION")
        require(
            (
                self.demographic_inputs
                and self.sex is not None
                and sex_graph.source.seal(self.sex) == self.sex_stamp
            )
            or (
                not self.demographic_inputs
                and self.sex is None
                and self.sex_stamp is None
                and self.sex_nodes == ()
            ),
            "DEMOGRAPHIC_STATE_CHANGED",
        )

    def _race_pure(self):
        require(type(self.race_hispanic_inputs) is bool, "RACE_HISPANIC_OPTION")
        require(
            (
                self.race_hispanic_inputs
                and self.race is not None
                and race_graph.source.seal(self.race) == self.race_stamp
            )
            or (
                not self.race_hispanic_inputs
                and self.race is None
                and self.race_stamp is None
                and self.race_nodes == ()
            ),
            "RACE_HISPANIC_STATE_CHANGED",
        )

    def _immigration_pure(self):
        if self.immigration_transfer is None:
            require(
                self.immigration_entry is None
                and self.immigration_origins is None
                and self.immigration_pairs_bytes is None
                and self.immigration_nodes == (),
                "IMMIGRATION_DISABLED_STATE",
            )
            return
        require(
            immigration_graph.retained_entry(self.immigration_transfer)
            is self.immigration_entry,
            "IMMIGRATION_ENTRY_CHANGED",
        )
        state = self.immigration_entry[2]
        require(
            state.run is self.run.financial_run
            and state.run_entry is parent.financial._run_entry(self.run.financial_run)
            and state.original is self.run.financial_run.prefix.allocated_population
            and state.acs_entry[2].qualified.origins is self.immigration_origins
            and self.immigration_origins.index.equals(
                self.immigration_transfer.pairs.index
            )
            and immigration_graph.pair_bytes(self.immigration_transfer.pairs)
            == self.immigration_pairs_bytes,
            "IMMIGRATION_COMMON_PARENT",
        )

    def _spm_objects(self):
        return (
            self.spm.source_frame,
            self.spm.origins,
            self.spm.roles,
            self.spm.unit_status,
            self.spm.unit_evidence,
            self.spm.asec_raw,
        )

    def _spm_pure(self):
        require(
            _spm_configuration(*self.spm_options) == self.spm_configuration,
            "SPM_CONFIGURATION_CHANGED",
        )
        if self.spm_configuration is None:
            require(
                self.spm is None
                and self.spm_entry is None
                and self.spm_objects is None
                and self.spm_stamp is None
                and self.spm_nodes == (),
                "SPM_DISABLED_STATE",
            )
            return
        spm_graph.source._check_retained_output(self.spm, self.spm_entry)
        require(
            self.spm._receiving_run is None
            and all(
                a is b
                for a, b in zip(self.spm_objects, self._spm_objects(), strict=True)
            )
            and spm_graph.spm_seal(self.spm) == self.spm_stamp,
            "SPM_BOUNDARY_CHANGED",
        )

    def pure(self):
        self._sex_pure()
        self._race_pure()
        require(self.state_nodes == self._state_nodes(), "STATE_DECLARATIONS")
        parent._pure_run(self.run, self.parent_entry)
        require(
            parent._run_entry(self.run) is self.parent_entry
            and all(
                a is b
                for a, b in zip(
                    self.parent_objects,
                    (
                        self.run.population,
                        self.run.compiled,
                        self.run.manifest,
                        self.run.store,
                        self.run.kernels,
                        self.run.sources,
                    ),
                    strict=True,
                )
            )
            and physical._population_stamp(self.run.population) == self.parent_stamp
            and values.seal(self.qualified) == self.qualified_stamp
            and type(self.full_original_amount_donors) is bool
            and self.full_original_amount_donors
            == (self.qualified.full_donor_source_frame is not None)
            and self.run.financial_run.prefix.preparation is self.preparation
            and health_graph.health_coverage_seal(self.health) == self.health_stamp
            and type(self.health_completion_enabled) is bool
            and self.health_completion_enabled == (self.health_completion is not None)
            and self.health_completion_stamp
            == (
                None
                if self.health_completion is None
                else health_completion_graph.values.seal(self.health_completion)
            )
            and housing_graph.housing.seal(self.housing) == self.housing_stamp
            and hours_graph.hours_seal(self.hours) == self.hours_stamp
            and self.nodes == self.declaration
            and _live() == self.live,
            "BOUNDARY_CHANGED",
        )
        require(
            self.amount_nodes
            == amount_nodes(
                self.qualified,
                self.run.population.frame,
                parent_digest=self.parent_view.digest,
                n_estimators=self.n_estimators,
            )
            and self.health_nodes == self._health_nodes()
            and self.health_completion_nodes == self._health_completion_nodes()
            and self.housing_nodes
            == housing_graph.housing_nodes(
                self.housing,
                self.run.population.frame,
                after=_housing_after_edge(),
                n_estimators=self.n_estimators,
            )
            and self.hours_nodes
            == hours_graph.hours_nodes(
                self.hours,
                self.run.population.frame,
                receiving_version=parent.attach.FILTER_NODE,
                after=_hours_after_edge(),
            )
            and self.spm_nodes
            == (
                ()
                if self.spm_configuration is None
                else spm_graph.spm_nodes(
                    self.spm,
                    self.run.population.frame,
                    receiving_version=parent.attach.FILTER_NODE,
                    after=_spm_after_edge(),
                    outside_role_placeholder=self.spm_options[2],
                )
            )
            and self.immigration_nodes
            == (
                ()
                if self.immigration_transfer is None
                else immigration_graph.immigration_nodes(
                    self.immigration_transfer,
                    self.run.population.frame,
                    receiving_version=parent.attach.FILTER_NODE,
                    after=_immigration_after_edge(self.spm is not None),
                )
            )
            and self.nodes
            == (
                *self.amount_nodes,
                *self.health_nodes,
                *self.housing_nodes,
                *self.hours_nodes,
                *self.spm_nodes,
                *self.immigration_nodes,
                *self.sex_nodes,
                *self.race_nodes,
                *self.state_nodes,
            ),
            "BOUNDARY_DECLARATIONS",
        )
        if self.compiled is not None:
            require(
                all(
                    a is b
                    for a, b in zip(
                        self.bound_objects,
                        (self.compiled, self.store, self.kernels, self.store.codecs),
                        strict=True,
                    )
                )
                and tuple(self.kernels.as_mapping().items()) == self.kernel_items
                and tuple(self.store.codecs.as_mapping().items()) == self.codec_items
                and tuple(self.store.codecs.as_bytes_mapping().items())
                == self.bytes_codec_items,
                "BOUND_REGISTRY_CHANGED",
            )
            require(
                self.compiled == compile_graph(self.compiled.graph)
                and graph_to_json(self.compiled.graph) == self.graph_json,
                "COMPILED_CHANGED",
            )
        self._spm_pure()
        self._immigration_pure()
        require(
            self.sex_nodes
            == (
                ()
                if self.sex is None
                else sex_graph.sex_nodes(
                    self.sex,
                    self.run.population.frame,
                    receiving_version=parent.attach.FILTER_NODE,
                    after=_sex_after_edge(
                        self.spm is not None, self.immigration_transfer is not None
                    ),
                )
            ),
            "DEMOGRAPHIC_DECLARATIONS",
        )
        require(
            self.race_nodes
            == (
                ()
                if self.race is None
                else race_graph.race_hispanic_nodes(
                    self.race,
                    self.run.population.frame,
                    receiving_version=parent.attach.FILTER_NODE,
                    after=_race_after_edge(
                        self.spm is not None,
                        self.immigration_transfer is not None,
                        self.sex is not None,
                    ),
                )
            ),
            "RACE_HISPANIC_DECLARATIONS",
        )

    def borrow(self):
        require(
            parent.check_survey_puf55_run(self.run).payload == self.parent_view.payload,
            "PARENT_IDENTITY",
        )
        if self.compiled is not None:
            paths, source_keys = _source_paths_and_keys(
                self.compiled, dict(self.paths), self.store
            )
            keys, implementations = _all_node_keys(
                self.compiled, self.kernels, source_keys
            )
            require(
                tuple(sorted(paths.items())) == self.paths
                and tuple(sorted(source_keys.items())) == self.source_keys
                and tuple(sorted(keys.items())) == self.keys
                and tuple(sorted(implementations.items())) == self.implementations,
                "SOURCE_OR_IMPLEMENTATION_CHANGED",
            )
        self.hours.validate()
        if self.spm is not None:
            self.spm.validate()
        if self.immigration_transfer is not None:
            self.immigration_transfer.validate()
        if self.sex is not None:
            self.sex.validate()
        if self.race is not None:
            self.race.validate()
        self.pure()

    def context(self, context):
        # Entry/final host fences check the complete PUF owner. These ordinary
        # fragment calls consume retained, source-qualified values and check
        # their pure seals; they do not repeatedly reread the full PUF pipeline.
        self.pure()
        require(
            set(context.sources) == set(context.node.sources)
            and all(
                context.sources[name] == dict(self.paths)[name]
                for name in context.sources
            )
            and context.node in self.nodes
            and set(context.artifacts)
            == {e.name for e in context.node.artifact_inputs},
            "CONTEXT_DECLARATION",
        )
        for edge in context.node.artifact_inputs:
            value = values.predictors.host.shared.artifact(
                context, edge.name, edge.type
            )
            require(
                value.producer_key == dict(self.keys)[edge.producer],
                "ARTIFACT_PRODUCER_KEY",
            )
            if edge.name == "projection":
                require(value.payload == self.qualified.projection, "PROJECTION_BYTES")
            if edge.name == "housing_projection":
                require(
                    value.payload == self.housing.projection, "HOUSING_PROJECTION_BYTES"
                )
            if edge == _upstream_edge():
                record = self.run.manifest.node(edge.producer)
                payload = self.run.store.load_bytes(
                    record.opaque_artifacts[edge.artifact]
                )
                require(
                    value.producer_key == record.key
                    and value.key == record.opaque_artifacts[edge.artifact]
                    and value.payload == payload,
                    "PARENT_ARTIFACT",
                )
        self.pure()
        return self.qualified

    def requalify(self):
        """Reconstruct owned source transformations at the host's final I/O fence."""
        fresh = values.qualify_current_survey_amounts(
            self.run,
            groups=tuple(g.spec.key for g in self.qualified.groups),
            full_original_donors=self.full_original_amount_donors,
        )
        require(
            values.seal(fresh) == self.qualified_stamp, "SOURCE_REQUALIFICATION_CHANGED"
        )
        fresh_health = health_graph.qualify_health_coverage(self.preparation)
        require(
            health_graph.health_coverage_seal(fresh_health) == self.health_stamp,
            "HEALTH_SOURCE_REQUALIFICATION_CHANGED",
        )
        if self.health_completion is not None:
            fresh_completion = (
                health_completion_graph.values.qualify_current_survey_health_completion(
                    self.preparation, fresh_health
                )
            )
            require(
                health_completion_graph.values.seal(fresh_completion)
                == self.health_completion_stamp,
                "HEALTH_COMPLETION_REQUALIFICATION_CHANGED",
            )
        fresh_housing = housing_graph.housing.qualify_current_survey_housing(self.run)
        require(
            housing_graph.housing.seal(fresh_housing) == self.housing_stamp,
            "HOUSING_SOURCE_REQUALIFICATION_CHANGED",
        )
        self.hours.validate()
        if self.spm is not None:
            self.spm.validate()
        if self.immigration_transfer is not None:
            self.immigration_transfer.validate()
        if self.sex is not None:
            self.sex.validate()
        if self.race is not None:
            self.race.validate()
        self.pure()


class _Kernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=predictor_graph._Kernel.capabilities.dependencies,
    )

    def __init__(self, boundary):
        self.boundary = boundary

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            values,
            values.unemployment,
            values.workers_compensation,
            values.workers_compensation.mapper,
            values.veterans,
            values.veterans.routing,
            values.child_source,
            values.child_source.routing,
            values.child_mapper,
            values.full_donor,
            values.full_donor.demographics,
            values.full_donor.demographics.demographic,
            values.full_donor.demographics.household,
            health_graph,
            state_graph,
            state_graph.geography_constants,
            health_completion_graph,
            health_completion_graph.values,
            hours_graph,
            hours_graph.source,
            hours_graph.hours,
            hours_graph.asec_hours,
            spm_graph,
            *spm_graph.source._modules(),
            parent,
            physical,
            predictor_graph,
            values.predictors,
            values.unemployment.coverage,
            values.unemployment.source_csv_builtin,
            population_ops,
            _structural_columns,
            model_input,
            qrf,
            qrf_target,
            dependencies=self.capabilities.dependencies,
        )


class CurrentSurveyStateVersionKernel(_Kernel):
    ref = "us.survey_geography.canonical_state_version@1"
    capabilities = replace(_Kernel.capabilities, structural=StructuralDelta.FILTER)

    def run(self, context):
        self.boundary.context(context)
        require(
            set(context.tables) == {"person"} and not context.sources,
            "STATE_VERSION_TABLES",
        )
        person = context.tables["person"]
        frame = self.boundary.run.population.frame
        columns = [*_structural_columns(frame, "person"), "person_support_clone_index"]
        require(person.equals(frame.person[columns]), "STATE_VERSION_PERSONS")
        result = _state_version_result(
            context.node,
            person[["person_id", "person_support_clone_index"]],
            context.artifacts,
        )
        self.boundary.pure()
        return result


class CurrentSurveyCanonicalStateKernel(_Kernel):
    ref = state_graph.REF

    def run(self, context):
        self.boundary.context(context)
        columns = ["household_id", *state_graph.INPUTS]
        if context.node.outputs[0].rewrite:
            columns.append(state_graph.OUTPUT)
        require(
            set(context.tables) == {"household"}
            and context.tables["household"].equals(
                self.boundary.run.population.frame.table("household")[columns]
            ),
            "STATE_RECEIVING_HOUSEHOLDS",
        )
        result = state_graph.CurrentSurveyStateKernel().run(context)
        self.boundary.pure()
        return result


class CurrentSurveyAmountProjectionKernel(_Kernel):
    ref = "us.survey_amounts.source_projection@1"

    def run(self, context):
        qualified = self.boundary.context(context)
        values.predictors.host._current_context_frame(
            context, self.boundary.run.population.frame
        )
        return KernelResult(
            artifacts={
                "projection": qualified.projection,
                **{g.spec.key + "_matrix": g.matrix for g in qualified.groups},
            },
            receipt=qualified.evidence,
        )


class CurrentSurveyFullAmountSourceKernel(_Kernel):
    ref = "us.survey_amounts.full_original_source@1"
    capabilities = replace(_Kernel.capabilities, structural=StructuralDelta.CREATE)

    def run(self, context):
        qualified = self.boundary.context(context)
        require(
            qualified.full_donor_source_frame is not None, "FULL_DONOR_SOURCE_DISABLED"
        )
        return KernelResult(
            frame=values.predictors.source._copy_source(qualified.donor_source_frame)
        )


class CurrentSurveyAmountDonorKernel(_Kernel):
    ref = "us.survey_amounts.source_donor@1"
    capabilities = replace(_Kernel.capabilities, structural=StructuralDelta.FILTER)

    def run(self, context):
        qualified = self.boundary.context(context)
        if qualified.full_donor_source_frame is None:
            values.predictors.host._current_context_frame(
                context, qualified.donor_source_frame
            )
        else:
            _full_donor_context(context, qualified.donor_source_frame)
        group = next(
            g for g in qualified.groups if g.spec.key == context.params["group"]
        )
        return KernelResult(
            keep=pd.Series(
                group.keep.copy(),
                index=pd.Index(
                    qualified.donor_source_frame.person.person_id.to_numpy(),
                    name="person_id",
                ),
            ),
            receipt={
                "group": group.spec.key,
                "donor_persons": int(group.keep.sum()),
                "selection": "joint_known_current_ASEC_person_targets",
                "weight_kind": "design",
            },
        )


class CurrentSurveyAmountColumnsKernel(_Kernel):
    ref = "us.survey_amounts.source_columns@1"

    def run(self, context):
        qualified = self.boundary.context(context)
        group = next(
            g for g in qualified.groups if g.spec.key == context.params["group"]
        )
        if qualified.full_donor_source_frame is None:
            values.predictors.host._current_context_frame(context, group.donor_frame)
        else:
            _full_donor_context(context, group.donor_frame)
        return KernelResult(
            columns={
                ("person", c): group.donor_columns[c] for c in group.donor_columns
            },
            receipt={
                "group": group.spec.key,
                "projection_sha256": codec.sha(qualified.projection),
            },
        )


def read_draws(qualified, artifacts):
    """Check target history, exact matrix producer and immutable draw bytes."""
    draws = {}
    for group in qualified.groups:
        name = group.spec.key
        matrix_value = artifacts[name + "_matrix"]
        require(matrix_value.payload == group.matrix, "MATRIX_BYTES")
        matrix = model_input.decode_recipient_matrix(group.matrix)
        require(tuple(matrix.features) == qualified.features, "MATRIX_FEATURES")
        result = pd.DataFrame(index=matrix.features.index)
        models, raw_history = [], []
        for i, target in enumerate(group.spec.targets):
            raw = artifacts[f"{name}_raw_{i}"].payload
            state_value = artifacts[f"{name}_state_{i}"]
            require(
                state_value.producer_key == artifacts[f"{name}_raw_{i}"].producer_key,
                "RAW_STATE_SIBLINGS",
            )
            packet = decode_matrix_apply_state(state_value.payload)
            application, chain = codec.read_application(
                codec.encode_json(packet["application"])
            )
            raw_history.append({"target": target, "sha256": codec.sha(raw)})
            require(
                packet["matrix_sha256"] == codec.sha(group.matrix)
                and packet["matrix_producer_key"] == matrix_value.producer_key
                and chain.entity == "person"
                and tuple(chain.predictors) == qualified.features
                and tuple(chain.targets) == group.spec.targets
                and tuple(chain.completed_targets) == group.spec.targets[: i + 1]
                and chain.recipient_index == qrf._index_identity(matrix.features.index)
                and application["seed"] == values.SEED
                and application["raw_targets"] == raw_history
                and application["models"][:i] == models
                and len(application["models"]) == i + 1,
                "DRAW_CHAIN",
            )
            models = application["models"]
            result[target] = codec.read_raw_target(
                raw, target=target, index=matrix.features.index
            )
        draws[name] = result
    return draws


def _attachment_result(boundary, artifacts, *, child_only=False):
    qualified = boundary.qualified
    require(
        artifacts["projection"].payload == qualified.projection, "ATTACHMENT_PROJECTION"
    )
    draws = read_draws(qualified, artifacts)
    columns = values.attach_columns(qualified, boundary.run.population.frame, draws)
    require(
        type(child_only) is bool and (not child_only or _canonical_enabled(qualified)),
        "CHILD_ATTACHMENT_OPTION",
    )
    if _canonical_enabled(qualified):
        columns = {
            key: value
            for key, value in columns.items()
            if (key[1] in values.canonical_outputs(qualified)) == child_only
        }
    receipt = {
        "protocol": values.PROTOCOL,
        "parent_sha256": boundary.parent_view.digest,
        "projection_sha256": codec.sha(qualified.projection),
        "draw_sha256": {
            n: codec.sha(v.payload) for n, v in artifacts.items() if "_raw_" in n
        },
        **(
            {
                "canonical_replacements": {
                    name: str(columns["person", name].dtype)
                    for name in values.canonical_outputs(qualified)
                    if name in boundary.run.population.frame.person
                }
            }
            if child_only
            else {}
        ),
        **(
            {
                "child_support_attachment": "canonical_final"
                if child_only
                else "deferred_to_derived_version"
            }
            if _child_enabled(qualified)
            else {}
        ),
        **(
            {
                "canonical_amount_attachment": "canonical_final"
                if child_only
                else "deferred_to_derived_version"
            }
            if any(g.spec.key == "veterans_benefits" for g in qualified.groups)
            else {}
        ),
        "source_unknowns_preserved": True,
        "weights_changed": False,
        "prior_wages_consumed": False,
        "release_eligible": False,
    }
    return KernelResult(
        columns=columns,
        artifacts={"attachment": codec.encode_json(receipt)},
        receipt=receipt,
    )


class CurrentSurveyAmountAttachKernel(_Kernel):
    ref = "us.survey_amounts.attach@1"

    def run(self, context):
        self.boundary.context(context)
        values.predictors.host._current_context_frame(
            context, self.boundary.run.population.frame
        )
        result = _attachment_result(self.boundary, context.artifacts)
        self.boundary.pure()
        return result


class CurrentSurveyChildVersionKernel(_Kernel):
    ref = "us.survey_amounts.child_version@1"
    capabilities = replace(_Kernel.capabilities, structural=StructuralDelta.FILTER)

    def run(self, context):
        qualified = self.boundary.context(context)
        require(_canonical_enabled(qualified), "CHILD_VERSION_DISABLED")
        values.predictors.host._current_context_frame(
            context, self.boundary.run.population.frame
        )
        return KernelResult(
            keep=pd.Series(
                True,
                index=pd.Index(
                    self.boundary.run.population.frame.person.person_id.to_numpy(),
                    name="person_id",
                ),
                dtype=bool,
            )
        )


class CurrentSurveyChildAttachKernel(_Kernel):
    ref = "us.survey_amounts.child_attach@1"

    def run(self, context):
        self.boundary.context(context)
        values.predictors.host._current_context_frame(
            context, self.boundary.run.population.frame
        )
        result = _attachment_result(self.boundary, context.artifacts, child_only=True)
        self.boundary.pure()
        return result


def _child_version_population(population, node):
    """Independently reconstruct keep-all with the actual completed base Frame."""
    require(
        node.id in (CHILD_VERSION_NODE, CANONICAL_VERSION_NODE)
        and node.structural is StructuralDelta.FILTER
        and node.mass == "conserve",
        "CHILD_VERSION_DECLARATION",
    )
    keep = np.ones(population.frame.n("person"), dtype=bool)
    selected = population.frame.select(keep)
    require(
        all(
            np.array_equal(
                selected.table(entity)[
                    selected.schema.entity_id_column(entity)
                ].to_numpy(),
                population.frame.table(entity)[
                    population.frame.schema.entity_id_column(entity)
                ].to_numpy(),
            )
            for entity in population.frame.entities
        ),
        "CHILD_VERSION_AXIS_CHANGED",
    )
    return population_ops.patch(population, node, KernelResult(frame=selected))


def _construct(
    run,
    *,
    groups,
    n_estimators,
    spm_acs_profile=None,
    spm_asec_scope_policy=None,
    spm_outside_role_placeholder=None,
    immigration_transfer=None,
    health_completion=False,
    demographic_inputs=False,
    race_hispanic_inputs=False,
    full_original_amount_donors=False,
    canonical_state_input=False,
):
    boundary = Boundary(
        run,
        groups=groups,
        n_estimators=n_estimators,
        spm_acs_profile=spm_acs_profile,
        spm_asec_scope_policy=spm_asec_scope_policy,
        spm_outside_role_placeholder=spm_outside_role_placeholder,
        immigration_transfer=immigration_transfer,
        health_completion=health_completion,
        demographic_inputs=demographic_inputs,
        race_hispanic_inputs=race_hispanic_inputs,
        full_original_amount_donors=full_original_amount_donors,
        canonical_state_input=canonical_state_input,
    )
    compiled = compile_graph(
        replace(run.compiled.graph, nodes=(*run.compiled.graph.nodes, *boundary.nodes))
    )
    kernels = KernelRegistry()
    for kernel in run.kernels.as_mapping().values():
        kernels.register(kernel)
    for cls in (
        CurrentSurveyAmountProjectionKernel,
        CurrentSurveyFullAmountSourceKernel,
        CurrentSurveyAmountDonorKernel,
        CurrentSurveyAmountColumnsKernel,
        CurrentSurveyAmountAttachKernel,
        CurrentSurveyChildVersionKernel,
        CurrentSurveyChildAttachKernel,
    ):
        require(cls.ref not in kernels.refs(), "KERNEL_COLLISION")
        kernels.register(cls(boundary))
    for kernel in health_graph.health_coverage_kernels(
        boundary.health,
        receiving_version=parent.attach.FILTER_NODE,
        after=_amount_edge(),
        require_current=boundary.pure,
    ):
        require(kernel.ref not in kernels.refs(), "HEALTH_KERNEL_COLLISION")
        kernels.register(kernel)
    for kernel in housing_graph.kernels(boundary):
        require(kernel.ref not in kernels.refs(), "HOUSING_KERNEL_COLLISION")
        kernels.register(kernel)
    for kernel in health_completion_graph.kernels(boundary):
        require(kernel.ref not in kernels.refs(), "HEALTH_COMPLETION_KERNEL_COLLISION")
        kernels.register(kernel)
    for kernel in hours_graph.hours_kernels(
        boundary.hours,
        run.population.frame,
        receiving_version=parent.attach.FILTER_NODE,
        after=_hours_after_edge(),
        require_current=boundary.pure,
    ):
        require(kernel.ref not in kernels.refs(), "HOURS_KERNEL_COLLISION")
        kernels.register(kernel)
    if boundary.spm is not None:
        for kernel in spm_graph.spm_kernels(
            boundary.spm,
            run.population.frame,
            receiving_version=parent.attach.FILTER_NODE,
            after=_spm_after_edge(),
            outside_role_placeholder=boundary.spm_options[2],
            require_current=boundary.pure,
            require_context=boundary.context,
        ):
            require(kernel.ref not in kernels.refs(), "SPM_KERNEL_COLLISION")
            kernels.register(kernel)
    for kernel in immigration_graph.immigration_kernels(boundary):
        require(kernel.ref not in kernels.refs(), "IMMIGRATION_KERNEL_COLLISION")
        kernels.register(kernel)
    if boundary.sex is not None:
        for kernel in sex_graph.sex_kernels(
            boundary.sex,
            run.population.frame,
            receiving_version=parent.attach.FILTER_NODE,
            after=_sex_after_edge(
                boundary.spm is not None, boundary.immigration_transfer is not None
            ),
            require_current=boundary.pure,
            require_context=boundary.context,
        ):
            require(kernel.ref not in kernels.refs(), "SEX_KERNEL_COLLISION")
            kernels.register(kernel)
    if boundary.race is not None:
        for kernel in race_graph.race_hispanic_kernels(
            boundary.race,
            run.population.frame,
            receiving_version=parent.attach.FILTER_NODE,
            after=_race_after_edge(
                boundary.spm is not None,
                boundary.immigration_transfer is not None,
                boundary.sex is not None,
            ),
            require_current=boundary.pure,
            require_context=boundary.context,
        ):
            require(kernel.ref not in kernels.refs(), "RACE_HISPANIC_KERNEL_COLLISION")
            kernels.register(kernel)
    if boundary.canonical_state_input:
        for cls in (CurrentSurveyStateVersionKernel, CurrentSurveyCanonicalStateKernel):
            require(cls.ref not in kernels.refs(), "STATE_KERNEL_COLLISION")
            kernels.register(cls(boundary))
    registry = parent._registry(run.store.codecs, codecs.SourceCodecRegistry())
    store = ContentStore(run.store.root, codecs=registry)
    paths, source_keys = _source_paths_and_keys(compiled, dict(run.sources), store)
    keys, implementations = _all_node_keys(compiled, kernels, source_keys)
    require(
        all(
            keys[n] == run.manifest.node(n).key
            and implementations[n] == run.manifest.node(n).kernel_impl_hash
            and compiled.graph.node(n) == run.compiled.graph.node(n)
            for n in run.compiled.order
        ),
        "PREFIX_DECLARATION_OR_KEY",
    )
    boundary.compiled, boundary.kernels, boundary.store = compiled, kernels, store
    boundary.paths = tuple(sorted(paths.items()))
    boundary.source_keys = tuple(sorted(source_keys.items()))
    boundary.keys = tuple(sorted(keys.items()))
    boundary.implementations = tuple(sorted(implementations.items()))
    boundary.graph_json = graph_to_json(compiled.graph)
    boundary.bound_objects = (compiled, store, kernels, store.codecs)
    boundary.kernel_items = tuple(kernels.as_mapping().items())
    boundary.codec_items = tuple(store.codecs.as_mapping().items())
    boundary.bytes_codec_items = tuple(store.codecs.as_bytes_mapping().items())
    boundary.borrow()
    return boundary


def _verify_models(boundary, loaded, donors):
    """Verify persisted models against actual donor values after artifact admission."""
    for group in boundary.qualified.groups:
        donor_id, _, fit_prefix, apply_prefix = _ids(group)
        donor = donors[donor_id]
        first = boundary.compiled.graph.node(fit_prefix + ".000")
        model_frame = codec.model_frame(
            SimpleNamespace(weights={"person": donor.frame.resolve_weights("person")}),
            first.inputs[0],
            donor.frame.person,
        )
        model = qrf.RegimeGatedQRF(
            seed=values.SEED,
            n_estimators=boundary.n_estimators,
            zero_atol=0,
            max_samples_leaf=None,
        )
        before = qrf_target.LegacyQRFTrainingState.from_chain(
            model.start_chain(
                model_frame,
                list(boundary.qualified.features),
                list(group.spec.targets),
                weights="design",
            )
        )
        history = []
        for i, target in enumerate(group.spec.targets):
            node = f"{fit_prefix}.{i:03d}"
            payload = loaded[node, "model"]
            packet, after = codec.read_training(loaded[node, "training_state"])
            fitted = qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes(
                payload, expected_sha256=codec.sha(payload)
            )
            require(
                fitted.training_state == before
                and fitted.next_training_state == after
                and fitted.donor_sha256
                == qrf_target._consumed_values_sha256(
                    model_frame.person,
                    (*boundary.qualified.features, *group.spec.targets[:i], target),
                ),
                "TRAINING_DONOR",
            )
            history.append(
                {
                    "target": target,
                    "sha256": codec.sha(payload),
                    "training_id": fitted.training_id,
                }
            )
            require(
                packet["models"] == history
                and decode_matrix_apply_state(
                    loaded[f"{apply_prefix}.{i:03d}", "apply_state"]
                )["application"]["models"]
                == history,
                "TRAINING_APPLY_HISTORY",
            )
            before = after


@dataclass(frozen=True)
class SurveyEnrichmentRun:
    parent_run: parent.SurveyPuf55Run
    population: population_ops.Population
    manifest: object
    compiled: object
    store: ContentStore
    kernels: KernelRegistry
    sources: tuple
    receipt: bytes

    def checked_view(self):
        return check_survey_enrichment_run(self)


@dataclass(frozen=True)
class CheckedSurveyEnrichmentRun:
    """Descriptive values; authority remains in the original retained host."""

    payload: bytes
    digest: str
    population: population_ops.Population


def _run_seal(run):
    return (
        run.receipt,
        run.manifest.to_json(),
        graph_to_json(run.compiled.graph),
        physical._population_stamp(run.population),
        parent._heterogeneous_manifest_seals(run.manifest, run.compiled),
        tuple(sorted(run.manifest.populations)),
        tuple(sorted(run.manifest.mass_ledgers)),
    )


def check_survey_enrichment_run(run):
    entry = _ISSUED.get(id(run))
    require(
        type(run) is SurveyEnrichmentRun and entry is not None and entry[0]() is run,
        "UNISSUED_RUN",
    )
    _, boundary, stamp, objects, artifacts = entry
    try:
        boundary.borrow()
        require(
            all(
                a is b
                for a, b in zip(
                    objects,
                    (
                        run.parent_run,
                        run.population,
                        run.manifest,
                        run.compiled,
                        run.store,
                        run.kernels,
                        run.sources,
                    ),
                    strict=True,
                )
            ),
            "RUN_OBJECTS",
        )
        loaded = parent.financial._artifacts(
            run.manifest,
            run.compiled,
            run.store,
            run.kernels,
            dict(boundary.keys),
            dict(boundary.implementations),
        )
        require(
            tuple(sorted((n, a, codec.sha(p)) for (n, a), p in loaded.items()))
            == artifacts,
            "RUN_ARTIFACTS",
        )
        boundary.borrow()
        boundary.pure()
        require(_run_seal(run) == stamp, "RUN_CHANGED")
    except BaseException:
        if _ISSUED.get(id(run)) is entry:
            del _ISSUED[id(run)]
        raise
    return CheckedSurveyEnrichmentRun(
        run.receipt, codec.sha(run.receipt), run.population
    )


def run_us_survey_enrichment(
    run,
    *,
    groups=("unemployment", "health_costs"),
    n_estimators=100,
    resume="auto",
    spm_acs_profile=None,
    spm_asec_scope_policy=None,
    spm_outside_role_placeholder=None,
    immigration_transfer=None,
    health_completion=False,
    demographic_inputs=False,
    race_hispanic_inputs=False,
    full_original_amount_donors=False,
    canonical_state_input=False,
):
    """Execute and verify enrichment with optional SPM and realized immigration.

    SPM requires an explicit ACS profile and OUTSIDE role representation. An
    absent ASEC scope policy preserves UNRESOLVED for the country refusal gate.
    Immigration consumes the same financial parent's genuine original transfer;
    this host only fans its final pair to existing clones, without another draw.
    Health completion is an explicit development model from original ASEC 2025
    coverage onto original ACS 2024 people; scientific qualification is pending.
    Demographic inputs bind source-qualified nullable sex on originals and copy
    it to both clones; unknown values remain unknown. Race/Hispanic inputs have
    a separate opt-in and preserve unsupported ACS categories as unknown.
    Full-original amount donors are an explicit opt-in; they change only fitting
    support, while receiving source observations and clone identity stay fixed.
    Canonical state is a separate opt-in after all existing fragments. Its visible
    keep-all version prevents earlier readers from consuming a later rewrite.
    """
    require(resume in ("auto", "require"), "RESUME")
    boundary = _construct(
        run,
        groups=groups,
        n_estimators=n_estimators,
        spm_acs_profile=spm_acs_profile,
        spm_asec_scope_policy=spm_asec_scope_policy,
        spm_outside_role_placeholder=spm_outside_role_placeholder,
        immigration_transfer=immigration_transfer,
        health_completion=health_completion,
        demographic_inputs=demographic_inputs,
        race_hispanic_inputs=race_hispanic_inputs,
        full_original_amount_donors=full_original_amount_donors,
        canonical_state_input=canonical_state_input,
    )
    observed, stamps = {}, {}

    def observe(node_id, population):
        require(node_id not in observed, "OBSERVER_DUPLICATE")
        observed[node_id] = population
        stamps[node_id] = physical._population_stamp(population)

    manifest = run_graph(
        boundary.compiled,
        sources=dict(boundary.paths),
        store=boundary.store,
        kernels=boundary.kernels,
        resume=resume,
        _population_observer=observe,
    )
    require(tuple(observed) == boundary.compiled.order, "OBSERVER_ROSTER")
    if resume == "require":
        require(all(record.hit for record in manifest.nodes.values()), "REQUIRED_HITS")
    boundary.borrow()
    loaded = parent.financial._artifacts(
        manifest,
        boundary.compiled,
        boundary.store,
        boundary.kernels,
        dict(boundary.keys),
        dict(boundary.implementations),
    )
    for node in run.compiled.graph.nodes:
        old = run.manifest.node(node.id)
        for name, key in old.opaque_artifacts.items():
            require(
                loaded[node.id, name] == run.store.load_bytes(key), "PREFIX_ARTIFACT"
            )
    physical.replay.same_replayed_population(
        run.population, observed[parent.attach.ATTACH_NODE]
    )
    require(
        loaded[PROJECTION_NODE, "projection"] == boundary.qualified.projection,
        "SOURCE_PROJECTION",
    )
    attach = boundary.compiled.graph.node(ATTACH_NODE)
    artifacts = parent._loaded_values(boundary, manifest, loaded, attach)
    result = _attachment_result(boundary, artifacts)
    require(
        loaded[ATTACH_NODE, "attachment"] == result.artifacts["attachment"],
        "ATTACHMENT_ARTIFACT",
    )
    # Compare every unchanged prefix terminal Frame and ledger against the
    # retained parent manifest. Full execution owners/design anchors on the
    # receiving population are checked separately above against the original.
    prefix_terminal = {}
    for node_id in run.compiled.order:
        prefix_terminal[run.compiled.versions[node_id]] = observed[node_id]
    for version, population in prefix_terminal.items():
        physical.replay.same_replayed_population(
            population_ops.Population.from_frame(
                population.frame, version, mass_ledger=population.mass_ledger
            ),
            population_ops.Population.from_frame(
                run.manifest.population(version),
                version,
                mass_ledger=run.manifest.mass_ledger(version),
            ),
        )
    housing_attach = boundary.compiled.graph.node(housing_graph.ATTACH_NODE)
    housing_artifacts = parent._loaded_values(
        boundary, manifest, loaded, housing_attach
    )
    housing_result = housing_graph.attachment_result(boundary, housing_artifacts)
    require(
        loaded[housing_graph.PROJECTION_NODE, "projection"]
        == boundary.housing.projection
        and loaded[housing_graph.ATTACH_NODE, "attachment"]
        == housing_result.artifacts["attachment"],
        "HOUSING_RESULT_ARTIFACT",
    )
    current, donors = {}, {}
    housing_donor = None
    health_completion_donor = None
    health_completion_ids = {n.id for n in boundary.health_completion_nodes}
    health_ids = {n.id for n in boundary.health_nodes}
    hours_ids = {n.id for n in boundary.hours_nodes}
    spm_ids = {n.id for n in boundary.spm_nodes}
    immigration_ids = {n.id for n in boundary.immigration_nodes}
    sex_ids = {n.id for n in boundary.sex_nodes}
    race_ids = {n.id for n in boundary.race_nodes}
    state_ids = {n.id for n in boundary.state_nodes}
    group_nodes = {_ids(g)[0]: g for g in boundary.qualified.groups}
    column_nodes = {_ids(g)[1]: g for g in boundary.qualified.groups}
    original = population_ops.Population.from_frame(
        boundary.qualified.source_frame, values.predictors.host.survey_graph.CREATE_NODE
    )
    for node_id in boundary.compiled.order:
        node = boundary.compiled.graph.node(node_id)
        version = boundary.compiled.versions[node_id]
        if node_id in run.compiled.order:
            current[version] = observed[node_id]
            continue
        if node_id in state_ids:
            incoming = current[
                node.base if node.structural is StructuralDelta.FILTER else version
            ]
            expected = _state_expected_population(
                incoming,
                node,
                parent._loaded_values(boundary, manifest, loaded, node),
                {a.name: loaded[node_id, a.name] for a in node.artifact_outputs},
            )
        elif node_id in race_ids:
            race_result = race_graph.race_hispanic_result(
                boundary.race,
                node,
                parent._loaded_values(boundary, manifest, loaded, node),
                None if current.get(version) is None else current[version].frame.person,
            )
            require(
                all(
                    loaded[node_id, name] == payload
                    for name, payload in race_result.artifacts.items()
                ),
                "RACE_HISPANIC_RESULT_ARTIFACT",
            )
            expected = (
                population_ops.Population.from_frame(race_result.frame, node.id)
                if node.structural is StructuralDelta.CREATE
                else population_ops.patch(current[version], node, race_result)
            )
        elif node_id in sex_ids:
            sex_result = sex_graph.sex_result(
                boundary.sex,
                node,
                parent._loaded_values(boundary, manifest, loaded, node),
                None if current.get(version) is None else current[version].frame.person,
            )
            require(
                all(
                    loaded[node_id, name] == payload
                    for name, payload in sex_result.artifacts.items()
                ),
                "SEX_RESULT_ARTIFACT",
            )
            expected = (
                population_ops.Population.from_frame(sex_result.frame, node.id)
                if node.structural is StructuralDelta.CREATE
                else population_ops.patch(current[version], node, sex_result)
            )
        elif node_id in immigration_ids:
            immigration_artifacts = parent._loaded_values(
                boundary, manifest, loaded, node
            )
            immigration_result = immigration_graph.immigration_result(
                boundary.immigration_transfer,
                node,
                immigration_artifacts,
                current[version].frame.person,
            )
            require(
                all(
                    loaded[node_id, name] == payload
                    for name, payload in immigration_result.artifacts.items()
                ),
                "IMMIGRATION_RESULT_ARTIFACT",
            )
            expected = population_ops.patch(current[version], node, immigration_result)
        elif node_id in spm_ids:
            spm_artifacts = parent._loaded_values(boundary, manifest, loaded, node)
            spm_result = spm_graph.spm_result(
                boundary.spm, node, spm_artifacts, current[version].frame
            )
            require(
                all(
                    loaded[node_id, name] == payload
                    for name, payload in spm_result.artifacts.items()
                ),
                "SPM_RESULT_ARTIFACT",
            )
            expected = population_ops.patch(current[version], node, spm_result)
        elif node_id in hours_ids:
            hours_artifacts = parent._loaded_values(boundary, manifest, loaded, node)
            hours_result = hours_graph.hours_result(
                boundary.hours, node, hours_artifacts, current[version].frame.person
            )
            require(
                all(
                    loaded[node_id, name] == payload
                    for name, payload in hours_result.artifacts.items()
                ),
                "HOURS_RESULT_ARTIFACT",
            )
            expected = population_ops.patch(current[version], node, hours_result)
        elif node_id in health_completion_ids:
            if node_id.startswith(
                (
                    health_completion_graph.FIT_PREFIX + ".",
                    health_completion_graph.APPLY_PREFIX + ".",
                )
            ):
                expected = current[version]
            else:
                health_artifacts = parent._loaded_values(
                    boundary, manifest, loaded, node
                )
                incoming = current.get(
                    node.base if node.structural is StructuralDelta.FILTER else version
                )
                health_result = health_completion_graph.result(
                    boundary.health_completion,
                    boundary.health,
                    node,
                    health_artifacts,
                    None if incoming is None else incoming.frame.person,
                )
                require(
                    all(
                        loaded[node_id, name] == payload
                        for name, payload in health_result.artifacts.items()
                    ),
                    "HEALTH_COMPLETION_ARTIFACT",
                )
                expected = health_completion_graph.expected_population(
                    incoming, node, health_result
                )
                if node_id == health_completion_graph.DONOR_NODE:
                    health_completion_donor = expected
        elif node_id in health_ids:
            health_artifacts = parent._loaded_values(boundary, manifest, loaded, node)
            expected = health_graph.expected_health_population(
                node_id,
                current.get(version),
                qualified=boundary.health,
                node=node,
                artifacts=health_artifacts,
            )
            # Bind every persisted source/recode/attachment artifact to its
            # independent domain result, including attachment metadata.
            expected_result = health_graph._result(
                boundary.health,
                node,
                None if current.get(version) is None else current[version].frame.person,
            )
            require(
                all(
                    loaded[node_id, name] == payload
                    for name, payload in expected_result.artifacts.items()
                ),
                "HEALTH_RESULT_ARTIFACT",
            )
        elif node_id == _canonical_ids(boundary.qualified)[0]:
            expected = _child_version_population(current[node.base], node)
        elif node_id == _canonical_ids(boundary.qualified)[1]:
            child_result = _attachment_result(
                boundary,
                parent._loaded_values(boundary, manifest, loaded, node),
                child_only=True,
            )
            require(
                loaded[node_id, "attachment"] == child_result.artifacts["attachment"],
                "CHILD_ATTACHMENT_ARTIFACT",
            )
            expected = population_ops.patch(current[version], node, child_result)
        elif node_id == FULL_DONOR_SOURCE_NODE:
            expected = population_ops.Population.from_frame(
                boundary.qualified.donor_source_frame, node_id
            )
        elif node_id in group_nodes:
            donor_base = (
                original
                if boundary.qualified.full_donor_source_frame is None
                else current[FULL_DONOR_SOURCE_NODE]
            )
            expected = population_ops.patch(
                donor_base, node, KernelResult(frame=group_nodes[node_id].donor_frame)
            )
        elif node_id in column_nodes:
            group = column_nodes[node_id]
            expected = population_ops.patch(
                current[version],
                node,
                KernelResult(
                    columns={
                        ("person", c): group.donor_columns[c]
                        for c in group.donor_columns
                    }
                ),
            )
            donors[_ids(group)[0]] = expected
        elif node_id == housing_graph.DONOR_NODE:
            source_population = population_ops.Population.from_frame(
                boundary.housing.source_frame,
                values.predictors.host.survey_graph.CREATE_NODE,
            )
            expected = population_ops.patch(
                source_population,
                node,
                KernelResult(frame=boundary.housing.donor_frame),
            )
        elif node_id == housing_graph.COLUMNS_NODE:
            expected = population_ops.patch(
                current[version],
                node,
                KernelResult(
                    columns={
                        ("household", name): boundary.housing.donor_columns[name]
                        for name in boundary.housing.donor_columns
                    }
                ),
            )
            housing_donor = expected
        elif node_id == housing_graph.ATTACH_NODE:
            expected = population_ops.patch(current[version], node, housing_result)
        elif node_id == ATTACH_NODE:
            expected = population_ops.patch(current[version], node, result)
        else:
            expected = current[version]
        physical.replay.same_replayed_population(expected, observed[node_id])
        current[version] = expected
    _verify_models(boundary, loaded, donors)
    housing_graph.verify_model(boundary, loaded, housing_donor)
    health_completion_graph.verify_models(boundary, loaded, health_completion_donor)
    for version, population in current.items():
        physical.replay.same_replayed_population(
            population_ops.Population.from_frame(
                population.frame, version, mass_ledger=population.mass_ledger
            ),
            population_ops.Population.from_frame(
                manifest.population(version),
                version,
                mass_ledger=manifest.mass_ledger(version),
            ),
        )
    spm_receipt = {"enabled": False}
    if boundary.spm is not None:
        spm_receipt = {
            "enabled": True,
            **json.loads(boundary.spm_configuration),
            "source_receipt_sha256": codec.sha(boundary.spm.receipt),
            "source_sha256": codec.sha(loaded[spm_graph.SOURCE_NODE, "source"]),
            "projection_sha256": codec.sha(
                loaded[spm_graph.PROJECT_NODE, "projection"]
            ),
            "attachment_sha256": codec.sha(loaded[spm_graph.ATTACH_NODE, "attachment"]),
        }
    immigration_receipt = {}
    if boundary.immigration_transfer is not None:
        immigration_receipt["immigration"] = {
            "enabled": True,
            "transfer_receipt_sha256": codec.sha(boundary.immigration_transfer.receipt),
            "pairs_sha256": codec.sha(loaded[immigration_graph.SOURCE_NODE, "pairs"]),
            "attachment_sha256": codec.sha(
                loaded[immigration_graph.ATTACH_NODE, "attachment"]
            ),
            "graph_fit_artifact_qualified": False,
            "national_stock_alignment_qualified": False,
        }
    receipt = codec.encode_json(
        {
            "protocol": values.PROTOCOL,
            "parent_sha256": boundary.parent_view.digest,
            "manifest_key": manifest.key,
            "node_count": len(boundary.compiled.order),
            "groups": list(groups),
            "complete_population_compared": True,
            "projection_sha256": codec.sha(boundary.qualified.projection),
            "attachment_sha256": codec.sha(result.artifacts["attachment"]),
            "health_projection_sha256": codec.sha(boundary.health.projection),
            "health_attachment_sha256": codec.sha(
                loaded[health_graph.ATTACH_NODE, "attachment"]
            ),
            "health_fields": [f.output for f in health_graph.health.FIELDS],
            **(
                {}
                if boundary.health_completion is None
                else {
                    "health_completion": {
                        "enabled": True,
                        "projection_sha256": codec.sha(
                            boundary.health_completion.projection
                        ),
                        "features": list(health_completion_graph.values.FEATURES),
                        "source_knownness_preserved": True,
                        "one_draw_per_original": True,
                        "temporal_equivalence_claim": False,
                        "scientific_qualification": "pending",
                    }
                }
            ),
            "housing_projection_sha256": codec.sha(boundary.housing.projection),
            "housing_attachment_sha256": codec.sha(
                loaded[housing_graph.ATTACH_NODE, "attachment"]
            ),
            "housing_participation_assumptions": boundary.housing.evidence[
                "assumptions"
            ],
            "hours_source_receipt_sha256": codec.sha(boundary.hours.receipt),
            "hours_attachment_sha256": codec.sha(
                loaded[hours_graph.ATTACH_NODE, "attachment"]
            ),
            "hours_age15_policy": boundary.hours.proposals.age15_policy,
            "hours_under15_policy": boundary.hours.proposals.under15_policy,
            "spm": spm_receipt,
            **immigration_receipt,
            **(
                {
                    "demographics": {
                        "enabled": True,
                        "fields": [sex_graph.source.OUTPUT],
                        "source_evidence_sha256": codec.sha(boundary.sex.evidence),
                        "attachment_sha256": codec.sha(
                            loaded[sex_graph.ATTACH_NODE, "attachment"]
                        ),
                    }
                }
                if boundary.sex is not None
                else {}
            ),
            **(
                {
                    "race_hispanic": {
                        "enabled": True,
                        "fields": list(race_graph.source.OUTPUTS),
                        "source_evidence_sha256": codec.sha(boundary.race.evidence),
                        "attachment_sha256": codec.sha(
                            loaded[race_graph.ATTACH_NODE, "attachment"]
                        ),
                        "unsupported_categories": "explicitly nullable; no imputation",
                    }
                }
                if boundary.race is not None
                else {}
            ),
            **(
                {
                    "child_support_attachment_sha256": codec.sha(
                        loaded[_canonical_ids(boundary.qualified)[1], "attachment"]
                    )
                }
                if _child_enabled(boundary.qualified)
                else {}
            ),
            **(
                {
                    "canonical_amount_attachment_sha256": codec.sha(
                        loaded[CANONICAL_ATTACH_NODE, "attachment"]
                    )
                }
                if any(
                    g.spec.key == "veterans_benefits" for g in boundary.qualified.groups
                )
                else {}
            ),
            **(
                {
                    "canonical_state": {
                        "enabled": True,
                        "binding_sha256": codec.sha(
                            loaded[state_graph.NODE, "binding"]
                        ),
                        "version_sha256": codec.sha(
                            loaded[STATE_VERSION_NODE, "version"]
                        ),
                        "new_geography_assignment": False,
                    }
                }
                if boundary.canonical_state_input
                else {}
            ),
            "release_eligible": False,
        }
    )
    output = SurveyEnrichmentRun(
        run,
        observed[
            state_graph.NODE
            if boundary.canonical_state_input
            else _canonical_ids(boundary.qualified)[1]
            if _canonical_enabled(boundary.qualified)
            else race_graph.ATTACH_NODE
            if boundary.race is not None
            else sex_graph.ATTACH_NODE
            if boundary.sex is not None
            else immigration_graph.ATTACH_NODE
            if boundary.immigration_transfer is not None
            else spm_graph.ATTACH_NODE
            if boundary.spm is not None
            else hours_graph.ATTACH_NODE
        ],
        manifest,
        boundary.compiled,
        boundary.store,
        boundary.kernels,
        boundary.paths,
        receipt,
    )
    objects = (
        output.parent_run,
        output.population,
        output.manifest,
        output.compiled,
        output.store,
        output.kernels,
        output.sources,
    )
    stamp = _run_seal(output)
    artifact_hashes = tuple(
        sorted((n, a, codec.sha(p)) for (n, a), p in loaded.items())
    )
    boundary.requalify()
    boundary.borrow()
    fresh = parent.financial._artifacts(
        manifest,
        boundary.compiled,
        boundary.store,
        boundary.kernels,
        dict(boundary.keys),
        dict(boundary.implementations),
    )
    boundary.borrow()
    boundary.pure()
    # All source/artifact I/O and host callbacks precede these detached output
    # comparisons. Do not add revalidation callbacks after this fence.
    require(
        tuple(sorted((n, a, codec.sha(p)) for (n, a), p in fresh.items()))
        == artifact_hashes
        and _run_seal(output) == stamp
        and all(physical._population_stamp(observed[n]) == stamps[n] for n in observed),
        "FINAL_OUTPUT",
    )
    require(id(output) not in _ISSUED, "RUN_ALREADY_ISSUED")
    ident = id(output)

    def forget(ref):
        old = _ISSUED.get(ident)
        if old is not None and old[0] is ref:
            del _ISSUED[ident]

    entry = (weakref.ref(output, forget), boundary, stamp, objects, artifact_hashes)
    _ISSUED[ident] = entry
    return output
