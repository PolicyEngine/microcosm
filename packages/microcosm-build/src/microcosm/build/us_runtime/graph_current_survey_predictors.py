"""Graph training, draw and attachment of current survey financial predictors.

The ASEC fitting branch leaves the source CREATE before importance allocation;
its household-derived person weights remain DESIGN. No PUF clone is a donor.
All fitted totals, prior-target conditioning and final source-origin joins are
visible typed graph dependencies. Cold and replay results require the same
materialized verifier, independently of cache or node receipt labels.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import model_input, qrf
from microcosm.fit.graph_legacy_apply_matrix import (
    MATRIX_APPLY_STATE_TYPE,
    decode_matrix_apply_state,
)
from microcosm.fit.graph_legacy_qrf import (
    legacy_qrf_apply_matrix_nodes,
    legacy_qrf_train_nodes,
)
from microcosm.frame import US_SCHEMA
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelResult,
    Node,
    Numeric,
    Owned,
    Slice,
    StructuralDelta,
    source_hash,
)
from microcosm.graph import population as population_ops

from . import current_survey_predictors as values
from . import survey_population_replay as replay

host = values.host
require = values.require
PROJECTION_NODE = "survey_predictors.source_projection"
DONOR_NODE = "survey_predictors.asec_design_donor"
DONOR_COLUMNS_NODE = "survey_predictors.asec_current_columns"
FIT_PREFIX = "survey_predictors.fit"
APPLY_PREFIX = "survey_predictors.apply"
ATTACH_NODE = "survey_predictors.attach"
PROJECTION_TYPE = ArtifactType("microcosm.us.current_survey_predictor_projection", 1)


def _inputs(frame):
    result = []
    for entity in US_SCHEMA.entities:
        structural = {US_SCHEMA.entity_id_column(entity)}
        if entity == "person":
            structural.update(
                US_SCHEMA.membership_column(e) for e in US_SCHEMA.group_entities
            )
        result.append(
            Slice(entity, tuple(c for c in frame.table(entity) if c not in structural))
        )
    return tuple(result)


def _projection_edges():
    return (
        ArtifactInput("projection", PROJECTION_NODE, "projection", PROJECTION_TYPE),
        ArtifactInput(
            "matrix", PROJECTION_NODE, "matrix", model_input.RECIPIENT_MATRIX_TYPE
        ),
    )


def _attach_edges():
    result = list(_projection_edges())
    for i in range(len(values.TARGETS)):
        result.extend(
            (
                ArtifactInput(
                    f"raw_{i:03d}",
                    f"{APPLY_PREFIX}.{i:03d}",
                    "raw_draw",
                    codec.RAW_TARGET_TYPE,
                ),
                ArtifactInput(
                    f"state_{i:03d}",
                    f"{APPLY_PREFIX}.{i:03d}",
                    "apply_state",
                    MATRIX_APPLY_STATE_TYPE,
                ),
            )
        )
    return tuple(result)


def _geography_edge():
    return ArtifactInput(
        "geography_validation",
        "geography.gate",
        "validation",
        host.survey_budget.geography.atomic_graph.ATOMIC_GEOGRAPHY_VALIDATION_TYPE,
    )


def _params(qualified, host_pins, n_estimators):
    require(type(n_estimators) is int and n_estimators > 0, "TREE_COUNT")
    values.feature_columns(qualified.demographic_conditioning)
    geography_enabled = qualified.geography_config_payload is not None
    require(
        (type(qualified.geography_validation) is bytes)
        if geography_enabled
        else qualified.geography_validation is None,
        "GEOGRAPHY_VALIDATION_VALUES",
    )
    names = {e.name for e in host.current_survey_host_edges()}
    if geography_enabled:
        names.add(_geography_edge().name)
    require(
        type(host_pins) is dict and set(host_pins) == names,
        "HOST_PINS",
    )
    for pin in host_pins.values():
        require(
            type(pin) is dict
            and set(pin) == {"producer_key", "artifact_key", "payload_sha256"}
            and all(codec._hash(v) for v in pin.values()),
            "HOST_PIN_DIGESTS",
        )
    return {
        "protocol": values.PROTOCOL,
        "source_projection_sha256": codec.sha(qualified.projection),
        "host_edges": codec.encode_json(host_pins).decode(),
        "seed": values.SEED,
        "n_estimators": n_estimators,
        "demographic_conditioning": qualified.demographic_conditioning,
        "geography_config_sha256": None
        if qualified.geography_config_payload is None
        else codec.sha(qualified.geography_config_payload),
    }


def current_survey_predictor_nodes(
    qualified, clone_frame, *, host_pins, n_estimators=100
):
    require(
        type(qualified) is values.QualifiedSurveyPredictors, "QUALIFIED_VALUES_TYPE"
    )
    params = _params(qualified, host_pins, n_estimators)
    predictors = values.feature_columns(qualified.demographic_conditioning)
    projection = Node(
        PROJECTION_NODE,
        CurrentSurveyPredictorProjectionKernel.ref,
        # This ordinary node must live outside CREATE: its typed allocation
        # evidence is downstream of that version's structural allocation node.
        population=DONOR_NODE,
        inputs=_inputs(qualified.donor_frame),
        params=params,
        artifact_inputs=host.current_survey_host_edges(),
        artifact_outputs=(
            ArtifactOutput("projection", PROJECTION_TYPE),
            ArtifactOutput("matrix", model_input.RECIPIENT_MATRIX_TYPE),
        ),
    )
    donor = Node(
        DONOR_NODE,
        CurrentSurveyPredictorDonorFilterKernel.ref,
        base=host.survey_graph.CREATE_NODE,
        structural=StructuralDelta.FILTER,
        inputs=_inputs(qualified.source_frame),
        mass="free",
        params=params,
        # Branch from the authenticated CREATE before its weight transition.
        # The later projection depends on this FILTER, never conversely.
        artifact_inputs=(
            host.current_survey_host_edges()[0],
            *(
                (_geography_edge(),)
                if qualified.geography_config_payload is not None
                else ()
            ),
        ),
        description="Select native ASEC whole households before allocation; retain original design weights for survey financial fitting.",
    )
    columns = Node(
        DONOR_COLUMNS_NODE,
        CurrentSurveyPredictorDonorColumnsKernel.ref,
        population=DONOR_NODE,
        inputs=_inputs(qualified.donor_frame),
        params=params,
        outputs=tuple(
            Owned("person", c, "float64") for c in (*predictors, *values.TARGETS)
        ),
        artifact_inputs=_projection_edges(),
    )
    fit = legacy_qrf_train_nodes(
        FIT_PREFIX,
        population=DONOR_NODE,
        entity="person",
        predictors=predictors,
        targets=values.TARGETS,
        seed=values.SEED,
        phase=values.PHASE,
        n_estimators=n_estimators,
        zero_atol=0,
    )
    apply = legacy_qrf_apply_matrix_nodes(
        APPLY_PREFIX,
        population=host.survey_clone.COMBINED_CLONE_NODE,
        fit_nodes=fit,
        matrix_producer=PROJECTION_NODE,
        seed=values.SEED,
        phase=values.PHASE,
    )
    attach = Node(
        ATTACH_NODE,
        CurrentSurveyPredictorAttachKernel.ref,
        population=host.survey_clone.COMBINED_CLONE_NODE,
        inputs=_inputs(clone_frame),
        params=params,
        outputs=tuple(
            Owned("person", c, "float64", rewrite=(c in clone_frame.person))
            for c in values.OUTPUTS
        ),
        artifact_inputs=_attach_edges(),
    )
    return (projection, donor, columns, *fit, *apply, attach)


class _Kernel(host._CurrentSurveyKernel):
    def __init__(
        self,
        preparation,
        allocated_population,
        clone_population,
        *,
        host_pins,
        n_estimators=100,
        demographic_conditioning=False,
        geography_config=None,
    ):
        self.preparation = preparation
        self.allocated_population = allocated_population
        self.clone_population = clone_population
        self.host_pins = codec.decode_json(codec.encode_json(host_pins))
        self.n_estimators = n_estimators
        values.feature_columns(demographic_conditioning)
        self.demographic_conditioning = demographic_conditioning
        self.geography_config = geography_config
        self.geography_config_payload = host.survey_budget._config_payload(
            geography_config
        )

    def implementation_hash(self):
        demographics = values.observed_geography.demographics
        return codec.sha(
            codec.encode_json(
                {
                    "host": super().implementation_hash(),
                    "predictors": source_hash(
                        sys.modules[__name__],
                        values,
                        values.leaves,
                        values.universe,
                        values.observed_geography,
                        demographics,
                        demographics.qualify_current_asec_demographics,
                        demographics.demographic,
                        demographics.demographic.load_authenticated_asec_demographic_source,
                        demographics.demographic._snapshot,
                        demographics.household,
                        demographics.source_csv_builtin,
                        population_ops,
                        replay,
                        qrf,
                        model_input,
                        sys.modules[decode_matrix_apply_state.__module__],
                        sys.modules[legacy_qrf_train_nodes.__module__],
                        # The optional postclone geography admission invokes
                        # the budget owner's complete reconstruction closure.
                        *host.survey_budget._modules(),
                        dependencies=self.capabilities.dependencies,
                    ),
                }
            )
        )

    def _qualified(self, context):
        require(not context.sources, "UNDECLARED_SOURCE")
        require(
            host.survey_budget._config_payload(self.geography_config)
            == self.geography_config_payload,
            "GEOGRAPHY_CONFIG_CHANGED",
        )
        result = values.qualify_current_survey_predictors(
            self.preparation,
            self.allocated_population,
            self.clone_population,
            demographic_conditioning=self.demographic_conditioning,
            geography_config=self.geography_config,
        )
        require(
            result.geography_config_payload == self.geography_config_payload,
            "GEOGRAPHY_CONFIG_CHANGED",
        )
        nodes = current_survey_predictor_nodes(
            result,
            self.clone_population.frame,
            host_pins=self.host_pins,
            n_estimators=self.n_estimators,
        )
        expected = {n.id: n for n in nodes}
        require(context.node == expected.get(context.node.id), "NODE_DECLARATION")
        require(
            set(context.artifacts) == {e.name for e in context.node.artifact_inputs},
            "ARTIFACT_ROSTER",
        )
        if context.node.id == DONOR_NODE:
            edge = host.current_survey_host_edges()[0]
            require(edge.name == "preparation", "PREPARATION_EDGE")
            value = host.shared.artifact(context, edge.name, edge.type)
            require(
                self.host_pins[edge.name]
                == {
                    "producer_key": value.producer_key,
                    "artifact_key": value.key,
                    "payload_sha256": codec.sha(value.payload),
                }
                and value.payload == self.preparation.payload,
                "DONOR_PREPARATION_BINDING",
            )
            if result.geography_config_payload is not None:
                edge = _geography_edge()
                value = host.shared.artifact(context, edge.name, edge.type)
                require(
                    self.host_pins[edge.name]
                    == {
                        "producer_key": value.producer_key,
                        "artifact_key": value.key,
                        "payload_sha256": codec.sha(value.payload),
                    }
                    and type(value.payload) is bytes
                    and value.payload == result.geography_validation,
                    "GEOGRAPHY_VALIDATION_BINDING",
                )
        elif context.node.id != PROJECTION_NODE:
            projection = host.shared.artifact(context, "projection", PROJECTION_TYPE)
            matrix = host.shared.artifact(
                context, "matrix", model_input.RECIPIENT_MATRIX_TYPE
            )
            host.shared.siblings(context, ("projection", "matrix"))
            require(
                projection.payload == result.projection
                and matrix.payload == result.matrix,
                "SOURCE_ARTIFACT_BYTES",
            )
        return result


class CurrentSurveyPredictorProjectionKernel(_Kernel):
    ref = "us.survey_predictors.source_projection@1"

    def run(self, context):
        qualified = self._qualified(context)
        host._current_context_frame(context, qualified.donor_frame)
        loaded = {}
        for edge in host.current_survey_host_edges():
            value = host.shared.artifact(context, edge.name, edge.type)
            require(
                self.host_pins[edge.name]
                == {
                    "producer_key": value.producer_key,
                    "artifact_key": value.key,
                    "payload_sha256": codec.sha(value.payload),
                },
                "HOST_EDGE_PIN",
            )
            loaded[edge.name] = value.payload
        host.shared.siblings(context, ("allocation", "frame_context"))
        require(
            loaded["preparation"] == self.preparation.payload
            and codec.sha(loaded["allocation"])
            == qualified.evidence["allocation_sha256"]
            and codec.sha(loaded["frame_context"])
            == codec.decode_json(loaded["allocation"])["output_context_sha256"],
            "HOST_EDGE_BYTES",
        )
        return KernelResult(
            artifacts={"projection": qualified.projection, "matrix": qualified.matrix},
            receipt=qualified.evidence,
        )


class CurrentSurveyPredictorDonorFilterKernel(_Kernel):
    ref = "us.survey_predictors.asec_design_donor@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.FILTER,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=host._CurrentSurveyKernel.capabilities.dependencies,
    )

    def run(self, context):
        self._qualified(context)
        # CREATE carries raw ACS NIU blanks. The qualified feature frame has
        # separately receipted materialized zeros and is not this context.
        original = self.preparation._checked()[2].frame
        host._current_context_frame(context, original)
        person = original.person
        keep = (
            person[values.provenance.support_channel_column("person")]
            .eq("asec")
            .to_numpy()
        )
        return KernelResult(
            keep=pd.Series(
                keep,
                index=pd.Index(person.person_id.to_numpy(), name="person_id"),
                dtype=bool,
            ),
            receipt={
                "selection": "ASEC_native_whole_households",
                "fit_weight_kind": "design",
                "release_eligible": False,
            },
        )


class CurrentSurveyPredictorDonorColumnsKernel(_Kernel):
    ref = "us.survey_predictors.asec_current_columns@1"

    def run(self, context):
        qualified = self._qualified(context)
        host._current_context_frame(context, qualified.donor_frame)
        return KernelResult(
            columns={
                ("person", c): qualified.donor_columns[c]
                for c in (
                    *values.feature_columns(qualified.demographic_conditioning),
                    *values.TARGETS,
                )
            },
            receipt=qualified.evidence,
        )


def read_current_survey_draws(
    matrix,
    matrix_producer_key,
    raw_draws,
    apply_states,
    *,
    demographic_conditioning=False,
):
    predictors = values.feature_columns(demographic_conditioning)
    require(
        codec._hash(matrix_producer_key)
        and type(raw_draws) is tuple
        and type(apply_states) is tuple
        and len(raw_draws) == len(apply_states) == len(values.TARGETS),
        "DRAW_CHAIN_ROSTER",
    )
    prepared = model_input.decode_recipient_matrix(matrix)
    require(
        prepared.entity == "person" and tuple(prepared.features.columns) == predictors,
        "MATRIX_FEATURE_ROSTER",
    )
    result = pd.DataFrame(index=prepared.features.index)
    models = []
    for i, target in enumerate(values.TARGETS):
        packet = decode_matrix_apply_state(apply_states[i])
        require(
            packet["matrix_sha256"] == codec.sha(matrix)
            and packet["matrix_producer_key"] == matrix_producer_key,
            "DRAW_MATRIX_BINDING",
        )
        application, chain = codec.read_application(
            codec.encode_json(packet["application"])
        )
        require(
            chain.entity == "person"
            and tuple(chain.predictors) == predictors
            and tuple(chain.targets) == values.TARGETS
            and tuple(chain.completed_targets) == values.TARGETS[: i + 1]
            and chain.recipient_index == qrf._index_identity(prepared.features.index)
            and application["seed"] == values.SEED
            and application["raw_targets"]
            == [
                {"target": name, "sha256": codec.sha(raw)}
                for name, raw in zip(
                    values.TARGETS[: i + 1], raw_draws[: i + 1], strict=True
                )
            ],
            "DRAW_CHAIN_BINDING",
        )
        current_models = application["models"]
        require(
            current_models[:i] == models and len(current_models) == i + 1,
            "DRAW_MODEL_HISTORY",
        )
        models = current_models
        values_array = codec.read_raw_target(
            raw_draws[i], target=target, index=prepared.features.index
        )
        require(np.isfinite(values_array).all(), "DRAW_UNKNOWN")
        result[target] = values_array
    return result


class CurrentSurveyPredictorAttachKernel(_Kernel):
    ref = "us.survey_predictors.attach@1"

    def run(self, context):
        qualified = self._qualified(context)
        host._current_context_frame(context, self.clone_population.frame)
        matrix = host.shared.artifact(
            context, "matrix", model_input.RECIPIENT_MATRIX_TYPE
        )
        raw, states = [], []
        for i in range(len(values.TARGETS)):
            r, s = f"raw_{i:03d}", f"state_{i:03d}"
            host.shared.siblings(context, (r, s))
            raw.append(host.shared.artifact(context, r, codec.RAW_TARGET_TYPE).payload)
            states.append(
                host.shared.artifact(context, s, MATRIX_APPLY_STATE_TYPE).payload
            )
        drawn = read_current_survey_draws(
            matrix.payload,
            matrix.producer_key,
            tuple(raw),
            tuple(states),
            demographic_conditioning=self.demographic_conditioning,
        )
        columns = values.complete_predictor_columns(
            qualified, self.clone_population.frame, drawn
        )
        return KernelResult(
            columns=columns,
            receipt={
                **qualified.evidence,
                "raw_sha256": [codec.sha(r) for r in raw],
                "all_output_cells_available": True,
                "ACS_financial_origin": "modeled",
                "paired_draws": "source_origin_join",
                "host_weights_changed": False,
            },
        )


def verify_materialized_current_survey_predictors(
    preparation,
    allocated_population,
    clone_population,
    *,
    population,
    projection,
    matrix,
    matrix_producer_key,
    raw_draws,
    apply_states,
    host_pins,
    n_estimators=100,
    demographic_conditioning=False,
    geography_config=None,
):
    """Use actual executor-observed Population and authenticated typed artifacts.

    Call only after graph store/type/producer-key checks, both cold and replay.
    Requalifies the current source owners, then compares every column, identity,
    owner, metadata value, weight and mass ledger of the expected full result.
    """
    qualified = values.qualify_current_survey_predictors(
        preparation,
        allocated_population,
        clone_population,
        demographic_conditioning=demographic_conditioning,
        geography_config=geography_config,
    )
    require(
        projection == qualified.projection and matrix == qualified.matrix,
        "MATERIALIZED_SOURCE",
    )
    drawn = read_current_survey_draws(
        matrix,
        matrix_producer_key,
        raw_draws,
        apply_states,
        demographic_conditioning=demographic_conditioning,
    )
    attach = current_survey_predictor_nodes(
        qualified,
        clone_population.frame,
        host_pins=host_pins,
        n_estimators=n_estimators,
    )[-1]
    expected = population_ops.patch(
        clone_population,
        attach,
        KernelResult(
            columns=values.complete_predictor_columns(
                qualified, clone_population.frame, drawn
            )
        ),
    )
    replay.same_replayed_population(expected, population)
    return {
        **qualified.evidence,
        "all_output_cells_available": True,
        "paired_draws": "source_origin_join",
    }
