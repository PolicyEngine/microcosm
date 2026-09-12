"""Opt-in property components on the complete initial survey clone frame.

The existing country host retains source authority and issues its financial
run. These nodes borrow that owner's sources; neither options nor descriptive
projections are an admission token. Legacy tax leaves and their CAP chain remain
unchanged. Broad property receipts and retirement-account earnings are not tax
rental income or retirement withdrawals.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import graph_signed_reconciliation as signed_graph
from microcosm.fit import model_input, qrf
from microcosm.fit.graph_legacy_qrf import LegacyQRFApplyKernel
from microcosm.frame import Frame
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
from microcosm.graph.kernel import KernelContext
from microcosm.graph.keys import opaque_artifact_key

from . import current_property_income_sources as sources
from . import current_survey_predictors as shared
from . import graph_current_survey_predictors as financial
from . import graph_property_income as model
from . import survey_population_replay as replay
from .property_income_constants import (
    PROPERTY_COMPONENTS,
    PROPERTY_DRAW_COLUMNS,
    PROPERTY_REPORTED_TOTAL,
)

host = shared.host
PROTOCOL = "microcosm.us.current-survey-property.v1"
PREFIX = "survey_property"
PROJECTION_NODE = PREFIX + ".source_projection"
DONOR_NODE = PREFIX + ".asec_eligible_donor"
DONOR_COLUMNS_NODE = PREFIX + ".asec_donor_columns"
RECIPIENT_NODE = PREFIX + ".acs_eligible_recipient"
RECIPIENT_COLUMNS_NODE = PREFIX + ".acs_recipient_columns"
ATTACH_NODE = PREFIX + ".attach"
PROJECTION_TYPE = ArtifactType("microcosm.us.current_survey_property_projection", 1)
CAP_LIMITATION = "legacy_CAP_conditions_on_legacy_INT_DIV; reconciled_component_consistency_not_claimed"
LEGACY_DIFFERENCE = "property_legacy_interest_draw_minus_reconciled_interest"
BASIS_DIAGNOSTICS = (
    "property_interest_component_discrepancy",
    "property_reported_minus_component_total",
)


def require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_SURVEY_PROPERTY_" + reason)


@dataclass(frozen=True, kw_only=True)
class PropertyIncomeOptions:
    scales: tuple[float, float, float, float]
    atol: float
    rtol: float
    n_estimators: int

    def __post_init__(self):
        # Reuse the real numeric declaration's domain checks, including finite
        # positive scales and finite nonnegative tolerances; no fitted model.
        self.nodes((PROPERTY_REPORTED_TOTAL,))

    def nodes(self, features):
        return model.property_income_nodes(
            PREFIX,
            donor_population=DONOR_NODE,
            recipient_population=RECIPIENT_NODE,
            features=features,
            seed=shared.SEED,
            n_estimators=self.n_estimators,
            scales=self.scales,
            atol=self.atol,
            rtol=self.rtol,
        )

    def to_bytes(self):
        return codec.encode_json(self.document())

    def document(self):
        return {
            "scales": self.scales,
            "atol": self.atol,
            "rtol": self.rtol,
            "n_estimators": self.n_estimators,
        }


def _features(qualified):
    return (
        *shared.feature_columns(qualified.shared_predictors.demographic_conditioning),
        PROPERTY_REPORTED_TOTAL,
    )


def _diagnostics():
    return signed_graph._diagnostics(PROPERTY_COMPONENTS, PREFIX + "_reconciliation")


def owned_columns():
    adjustments, active, residual, objective = _diagnostics()
    floats = (
        *PROPERTY_COMPONENTS,
        *PROPERTY_DRAW_COLUMNS,
        PROPERTY_REPORTED_TOTAL,
        *adjustments,
        residual,
        objective,
        LEGACY_DIFFERENCE,
        *BASIS_DIAGNOSTICS,
    )
    return (
        *tuple(Owned("person", c, "float64") for c in floats),
        *tuple(Owned("person", c, "boolean") for c in active),
        Owned("person", "property_anchor_known", "bool"),
        Owned("person", "property_components_known", "bool"),
    )


def _seal_json(value):
    if type(value) is bytes:
        return {"bytes_sha256": codec.sha(value)}
    if type(value) is tuple:
        return [_seal_json(x) for x in value]
    if type(value) is dict:
        return {k: _seal_json(v) for k, v in value.items()}
    return value


def _projection_values(qualified):
    require(type(qualified) is sources.QualifiedPropertyIncomeSources, "QUALIFIED_TYPE")
    require(
        qualified.donor_frame is not None
        and qualified.recipient_frame is not None
        and qualified.recipient_matrix is not None,
        "EMPTY_MODEL_BRANCH",
    )
    donor_matrix = model_input.encode_recipient_matrix(
        qualified.donor_columns,
        entity="person",
        entity_ids=qualified.donor_columns.index.to_numpy(dtype="<i8", copy=True),
    )
    # Physical seals bind unknown masks/backing storage and complete detached
    # source descriptions, not just the eligible model rows or aggregate counts.
    evidence = {
        "protocol": PROTOCOL,
        "source": codec.decode_json(qualified.projection),
        "source_values_sha256": codec.sha(
            codec.encode_json(
                _seal_json(sources.property_income_sources_seal(qualified))
            )
        ),
        "donor_matrix_sha256": codec.sha(donor_matrix),
        "recipient_matrix_sha256": codec.sha(qualified.recipient_matrix),
        "donor_exclusion_summary": json.loads(
            qualified.donor_basis.summary.to_json(orient="split")
        ),
        "recipient_exclusions": {
            name: int(qualified.recipient_diagnostics[name].sum())
            for name in (
                "excluded_under15",
                "excluded_unknown_anchor",
                "eligible_recipient",
            )
        },
        "donor_summary_sha256": sources.physical._table_stamp(
            qualified.donor_basis.summary
        ),
        "recipient_diagnostics_sha256": sources.physical._table_stamp(
            qualified.recipient_diagnostics
        ),
        "tax_split_rebased": False,
        "capital_gains": CAP_LIMITATION,
        "paired_draws": "one_original_ACS_draw_fanned_to_exact_clone_pair",
        "asec_components": "qualified_individual_values_even_when_ineligible_as_donor",
        "acs_excluded": "unknown_components_and_anchor; source_anchor_status_retained_in_projection",
        "release_eligible": False,
    }
    return {
        "projection": codec.encode_json(evidence),
        "donor_matrix": donor_matrix,
        "recipient_matrix": qualified.recipient_matrix,
    }


def _source_edges():
    return (
        ArtifactInput(
            "property_projection", PROJECTION_NODE, "projection", PROJECTION_TYPE
        ),
        ArtifactInput(
            "donor_matrix",
            PROJECTION_NODE,
            "donor_matrix",
            model_input.RECIPIENT_MATRIX_TYPE,
        ),
        ArtifactInput(
            "recipient_matrix",
            PROJECTION_NODE,
            "recipient_matrix",
            model_input.RECIPIENT_MATRIX_TYPE,
        ),
    )


def _host_edges(qualified):
    return (
        *host.current_survey_host_edges(),
        *(
            (financial._geography_edge(),)
            if qualified.shared_predictors.geography_config_payload is not None
            else ()
        ),
        ArtifactInput(
            "predictor_projection",
            financial.PROJECTION_NODE,
            "projection",
            financial.PROJECTION_TYPE,
        ),
        ArtifactInput(
            "predictor_matrix",
            financial.PROJECTION_NODE,
            "matrix",
            model_input.RECIPIENT_MATRIX_TYPE,
        ),
    )


def _attach_edges():
    return (
        *_source_edges(),
        *tuple(
            edge
            for i in range(4)
            for edge in (
                ArtifactInput(
                    f"raw_{i}",
                    f"{PREFIX}.apply.{i:03d}",
                    "raw_draw",
                    codec.RAW_TARGET_TYPE,
                ),
                ArtifactInput(
                    f"state_{i}",
                    f"{PREFIX}.apply.{i:03d}",
                    "apply_state",
                    codec.APPLY_STATE_TYPE,
                ),
            )
        ),
        ArtifactInput(
            "training", PREFIX + ".fit.003", "training_state", codec.TRAINING_STATE_TYPE
        ),
        ArtifactInput(
            "draw_summary", PREFIX + ".draws", "summary", model.DRAW_SUMMARY_TYPE
        ),
        ArtifactInput(
            "reconciliation_summary",
            PREFIX + ".reconcile",
            "summary",
            signed_graph.SUMMARY_TYPE,
        ),
        *tuple(replace(e, name="legacy_" + e.name) for e in financial._attach_edges()),
    )


def _params(qualified, host_pins, options):
    require(type(options) is PropertyIncomeOptions, "OPTIONS_TYPE")
    # Reuse exact host-pin validation; tree count here only validates an option.
    financial._params(qualified.shared_predictors, host_pins, options.n_estimators)
    projected = _projection_values(qualified)
    return {
        "protocol": PROTOCOL,
        "source_projection_sha256": codec.sha(projected["projection"]),
        "host_edges": codec.encode_json(host_pins).decode(),
        "tax_split_rebased": False,
        "capital_gains": CAP_LIMITATION,
    }


def _attachment_inputs(frame):
    return tuple(
        Slice(
            s.entity,
            tuple(dict.fromkeys((*s.columns, *shared.OUTPUTS)))
            if s.entity == "person"
            else s.columns,
        )
        for s in financial._inputs(frame)
    )


def current_survey_property_nodes(qualified, clone_frame, *, host_pins, options):
    """Declare 16 nodes; source admission remains with the existing host."""
    params = _params(qualified, host_pins, options)
    require(
        not set(c.column for c in owned_columns()).intersection(clone_frame.person),
        "OWNED_COLUMN_COLLISION",
    )
    source = Node(
        PROJECTION_NODE,
        CurrentSurveyPropertyProjectionKernel.ref,
        population=financial.DONOR_NODE,
        inputs=financial._inputs(qualified.shared_predictors.donor_frame),
        params=params,
        artifact_inputs=_host_edges(qualified),
        artifact_outputs=(
            ArtifactOutput("projection", PROJECTION_TYPE),
            ArtifactOutput("donor_matrix", model_input.RECIPIENT_MATRIX_TYPE),
            ArtifactOutput("recipient_matrix", model_input.RECIPIENT_MATRIX_TYPE),
        ),
    )
    branches = []
    for donor, node_id, column_id, frame, columns in (
        (
            True,
            DONOR_NODE,
            DONOR_COLUMNS_NODE,
            qualified.donor_frame,
            qualified.donor_columns,
        ),
        (
            False,
            RECIPIENT_NODE,
            RECIPIENT_COLUMNS_NODE,
            qualified.recipient_frame,
            qualified.recipient_columns,
        ),
    ):
        branches.extend(
            (
                Node(
                    node_id,
                    CurrentSurveyPropertyFilterKernel.ref,
                    base=host.survey_graph.CREATE_NODE,
                    structural=StructuralDelta.FILTER,
                    mass="free",
                    inputs=financial._inputs(qualified.source_frame),
                    params={**params, "donor": donor},
                    artifact_inputs=_source_edges(),
                    description="Select eligible original persons before allocation; retain DESIGN weights and exact source identity.",
                ),
                Node(
                    column_id,
                    CurrentSurveyPropertyColumnsKernel.ref,
                    population=node_id,
                    inputs=financial._inputs(frame),
                    params={**params, "donor": donor},
                    artifact_inputs=_source_edges(),
                    outputs=tuple(
                        Owned("person", c, "float64", rewrite=c in frame.person)
                        for c in columns
                    ),
                ),
            )
        )
    attach = Node(
        ATTACH_NODE,
        CurrentSurveyPropertyAttachKernel.ref,
        population=host.survey_clone.COMBINED_CLONE_NODE,
        inputs=_attachment_inputs(clone_frame),
        outputs=owned_columns(),
        params={**params, **options.document()},
        artifact_inputs=_attach_edges(),
        description="Attach source components or one reconciled original ACS draw to both initial clones; preserve legacy tax leaves and CAP limitation.",
    )
    return (source, *branches, *options.nodes(_features(qualified)), attach)


def _payloads(artifacts, edges):
    return {e.name: artifacts[e.producer, e.artifact] for e in edges}


def _read_draws(qualified, artifacts):
    raw = tuple(artifacts[f"{PREFIX}.apply.{i:03d}", "raw_draw"] for i in range(4))
    states = tuple(
        artifacts[f"{PREFIX}.apply.{i:03d}", "apply_state"] for i in range(4)
    )
    training, trained = codec.read_training(
        artifacts[PREFIX + ".fit.003", "training_state"]
    )
    frame = qualified.recipient_frame
    features = _features(qualified)
    training_state = trained.to_dict()
    require(
        training_state["entity"] == "person"
        and tuple(training_state["predictors"]) == features
        and tuple(training_state["targets"]) == PROPERTY_COMPONENTS
        and tuple(training_state["completed_targets"]) == PROPERTY_COMPONENTS,
        "TRAINING_ROSTER",
    )
    index = frame.person.index
    draws = pd.DataFrame(
        index=pd.Index(frame.person.person_id.to_numpy(copy=True), name="person_id")
    )
    require(draws.index.equals(qualified.recipient_columns.index), "DRAW_PERSON_AXIS")
    for i, (target, column) in enumerate(
        zip(PROPERTY_COMPONENTS, PROPERTY_DRAW_COLUMNS, strict=True)
    ):
        application, chain = codec.read_application(states[i])
        require(
            chain.entity == "person"
            and tuple(chain.predictors) == features
            and tuple(chain.targets) == PROPERTY_COMPONENTS
            and tuple(chain.completed_targets) == PROPERTY_COMPONENTS[: i + 1]
            and chain.recipient_index == qrf._index_identity(index)
            and application["seed"] == shared.SEED
            and application["models"] == training["models"][: i + 1]
            and application["raw_targets"]
            == [
                {"target": name, "sha256": codec.sha(value)}
                for name, value in zip(
                    PROPERTY_COMPONENTS[: i + 1], raw[: i + 1], strict=True
                )
            ],
            "ORDERED_DRAW_HISTORY",
        )
        values = codec.read_raw_target(raw[i], target=target, index=index)
        require(np.isfinite(values).all(), "DRAW_UNKNOWN")
        draws[column] = values
    expected_summary = codec.encode_json(
        {
            "protocol": model.PROTOCOL,
            "rows": len(draws),
            "components": PROPERTY_COMPONENTS,
            "models": training["models"],
            "raw_sha256": [codec.sha(r) for r in raw],
            "allocation": "one_joint_draw_per_input_person",
        }
    )
    require(artifacts[PREFIX + ".draws", "summary"] == expected_summary, "DRAW_SUMMARY")
    return draws


def _reconciliation_result(qualified, options, draws):
    # Invoke the existing deterministic kernel, never a second QRF computation.
    node = options.nodes(_features(qualified))[-1]
    table = qualified.recipient_frame.person[["person_id"]].copy(deep=True)
    table[PROPERTY_REPORTED_TOTAL] = qualified.recipient_columns[
        PROPERTY_REPORTED_TOTAL
    ].to_numpy(copy=True)
    for name in PROPERTY_DRAW_COLUMNS:
        table[name] = draws[name].to_numpy(copy=True)
    context = KernelContext(
        node=node,
        tables={"person": table},
        weights={},
        strata=None,
        params=node.params,
        rng=None,
        sources={},
        artifacts={},
        tolerances={},
        numerics={},
    )
    return signed_graph.SignedReconciliationKernel().run(context)


def _legacy_draws(qualified, artifacts, matrix_producer_key):
    require(
        artifacts[financial.PROJECTION_NODE, "projection"]
        == qualified.shared_predictors.projection
        and artifacts[financial.PROJECTION_NODE, "matrix"]
        == qualified.shared_predictors.matrix,
        "LEGACY_PROJECTION",
    )
    return financial.read_current_survey_draws(
        qualified.shared_predictors.matrix,
        matrix_producer_key,
        tuple(
            artifacts[f"{financial.APPLY_PREFIX}.{i:03d}", "raw_draw"] for i in range(3)
        ),
        tuple(
            artifacts[f"{financial.APPLY_PREFIX}.{i:03d}", "apply_state"]
            for i in range(3)
        ),
        demographic_conditioning=qualified.shared_predictors.demographic_conditioning,
    )


def _clone_lookup(qualified, clone_frame):
    person = clone_frame.person
    provenance = shared.provenance
    ids = person[provenance.support_source_id_column("person")].to_numpy(dtype=np.int64)
    lookup = qualified.origins.reindex(ids)
    require(not lookup.isna().any().any(), "CLONE_ORIGIN_COVERAGE")
    require(
        np.array_equal(
            lookup.native_person_id.to_numpy(),
            person[provenance.spine_source_id_column("person")].to_numpy(),
        )
        and np.array_equal(
            lookup.source.to_numpy(),
            person[provenance.support_channel_column("person")].astype(str).to_numpy(),
        ),
        "CLONE_ORIGIN_IDENTITY",
    )
    clones = person[provenance.support_clone_index_column("person")].to_numpy()
    require(np.isin(clones, (0, 1)).all(), "CLONE_ROLE_DOMAIN")
    pairs = pd.DataFrame({"source": ids, "clone": clones})
    counts = pairs.groupby("source", sort=False).size()
    require(
        counts.eq(2).all()
        and len(counts) == len(qualified.origins)
        and not pairs.duplicated().any(),
        "WHOLE_CLONE_PAIRS",
    )
    require(
        person.person_id.dtype == np.dtype("int64") and person.person_id.is_unique,
        "CLONE_PERSON_IDS",
    )
    return ids, pd.Index(person.person_id.to_numpy(copy=True), name="person_id")


def complete_property_columns(
    qualified, clone_frame, draws, reconciliation, legacy_draws
):
    """Pure origin join; source owners and authenticated graph artifacts are external."""
    ids, index = _clone_lookup(qualified, clone_frame)
    original = pd.DataFrame(index=qualified.origins.index)
    for owned in owned_columns():
        if owned.dtype == "boolean":
            original[owned.column] = pd.array([pd.NA] * len(original), dtype="boolean")
        elif owned.dtype == "bool":
            original[owned.column] = False
        else:
            original[owned.column] = np.nan
    basis = qualified.donor_basis.person
    for name in (*PROPERTY_COMPONENTS, PROPERTY_REPORTED_TOTAL):
        original.loc[basis.index, name] = basis[name].to_numpy(copy=True)
    for output, name in zip(
        BASIS_DIAGNOSTICS,
        ("interest_component_discrepancy", "reported_minus_component_total"),
        strict=True,
    ):
        original.loc[basis.index, output] = basis[name].to_numpy(copy=True)
    for name in PROPERTY_DRAW_COLUMNS:
        original.loc[draws.index, name] = draws[name].to_numpy(copy=True)
    original.loc[draws.index, PROPERTY_REPORTED_TOTAL] = qualified.recipient_columns[
        PROPERTY_REPORTED_TOTAL
    ].to_numpy(copy=True)
    for (_, name), column in reconciliation.columns.items():
        require(column.index.equals(draws.index), "RECONCILIATION_IDENTITY")
        original.loc[draws.index, name] = column.to_numpy(copy=True)
    original.loc[draws.index, LEGACY_DIFFERENCE] = (
        legacy_draws.loc[draws.index, shared.TARGETS[0]].to_numpy()
        - original.loc[draws.index, list(PROPERTY_COMPONENTS[:2])]
        .sum(axis=1)
        .to_numpy()
    )
    original["property_anchor_known"] = np.isfinite(original[PROPERTY_REPORTED_TOTAL])
    original["property_components_known"] = np.isfinite(
        original.loc[:, list(PROPERTY_COMPONENTS)]
    ).all(axis=1)
    aligned = original.reindex(ids)
    return {
        ("person", owned.column): pd.Series(
            aligned[owned.column].array.copy(),
            index=index,
            dtype=owned.dtype,
            name=owned.column,
        )
        for owned in owned_columns()
    }


def reconstruct_property_results(
    qualified, clone_frame, *, host_pins, options, artifacts, legacy_matrix_producer_key
):
    """Reconstruct non-fit results from qualified sources and verified store bytes.

    The host validates artifact type/key/producer and model training against the
    actual donor, then patches these results at each declared graph version.
    No graph/model execution or source authority is inferred from this mapping.
    """
    current_survey_property_nodes(
        qualified, clone_frame, host_pins=host_pins, options=options
    )
    projected = _projection_values(qualified)
    require(
        all(
            artifacts[PROJECTION_NODE, name] == payload
            for name, payload in projected.items()
        ),
        "PROJECTION_ARTIFACTS",
    )
    result = {
        PROJECTION_NODE: KernelResult(
            artifacts=projected, receipt=codec.decode_json(projected["projection"])
        )
    }
    for donor, filter_id, columns_id, frame, table in (
        (
            True,
            DONOR_NODE,
            DONOR_COLUMNS_NODE,
            qualified.donor_frame,
            qualified.donor_columns,
        ),
        (
            False,
            RECIPIENT_NODE,
            RECIPIENT_COLUMNS_NODE,
            qualified.recipient_frame,
            qualified.recipient_columns,
        ),
    ):
        people = qualified.source_frame.person
        keep = people.person_id.isin(frame.person.person_id).to_numpy()
        result[filter_id] = KernelResult(
            keep=pd.Series(
                keep, index=pd.Index(people.person_id.to_numpy(), name="person_id")
            ),
            receipt={
                "protocol": PROTOCOL,
                "branch": "ASEC_eligible_original_design"
                if donor
                else "ACS_adult_known_anchor_original",
                "persons": len(table),
            },
        )
        result[columns_id] = KernelResult(
            columns={("person", c): table[c] for c in table},
            receipt={
                "protocol": PROTOCOL,
                "matrix_sha256": codec.sha(
                    projected["donor_matrix" if donor else "recipient_matrix"]
                ),
            },
        )
    draws = _read_draws(qualified, artifacts)
    reconciliation = _reconciliation_result(qualified, options, draws)
    require(
        artifacts[PREFIX + ".reconcile", "summary"]
        == reconciliation.artifacts["summary"],
        "RECONCILIATION_SUMMARY",
    )
    result[PREFIX + ".draws"] = KernelResult(
        columns={("person", c): draws[c] for c in draws},
        artifacts={"summary": artifacts[PREFIX + ".draws", "summary"]},
        receipt={"protocol": model.PROTOCOL, "rows": len(draws)},
    )
    result[PREFIX + ".reconcile"] = reconciliation
    legacy = _legacy_draws(qualified, artifacts, legacy_matrix_producer_key)
    result[ATTACH_NODE] = KernelResult(
        columns=complete_property_columns(
            qualified, clone_frame, draws, reconciliation, legacy
        ),
        receipt={
            **codec.decode_json(projected["projection"]),
            "options": options.document(),
            "raw_sha256": [
                codec.sha(artifacts[f"{PREFIX}.apply.{i:03d}", "raw_draw"])
                for i in range(4)
            ],
            "attached_columns": [o.column for o in owned_columns()],
            "host_weights_changed": False,
        },
    )
    return result


class _Kernel(host._CurrentSurveyKernel):
    def __init__(
        self,
        preparation,
        allocated_population,
        clone_population,
        *,
        host_pins,
        options,
        demographic_conditioning=False,
        geography_config=None,
    ):
        self.preparation, self.allocated_population, self.clone_population = (
            preparation,
            allocated_population,
            clone_population,
        )
        self.host_pins = codec.decode_json(codec.encode_json(host_pins))
        self.options = options
        self.demographic_conditioning = demographic_conditioning
        self.geography_config = geography_config

    def implementation_hash(self):
        return codec.sha(
            codec.encode_json(
                {
                    "host": host._CurrentSurveyKernel().implementation_hash(),
                    "fragment": source_hash(
                        sys.modules[__name__],
                        sources,
                        sources.acs,
                        sources.interest,
                        sources.routing,
                        sources.dividend,
                        sys.modules[sources.build_asec_property_basis.__module__],
                        financial,
                        shared,
                        model,
                        sources.physical,
                        model_input,
                        population_ops,
                        replay,
                        dependencies=self.capabilities.dependencies,
                    ),
                    "reconciliation": signed_graph.SignedReconciliationKernel().implementation_hash(),
                    "draws": model.PropertyIncomeDrawColumnsKernel().implementation_hash(),
                }
            )
        )

    def _qualified(self, context):
        require(not context.sources, "UNDECLARED_SOURCE")
        qualified = sources.qualify_current_property_income_sources(
            self.preparation,
            self.allocated_population,
            self.clone_population,
            demographic_conditioning=self.demographic_conditioning,
            geography_config=self.geography_config,
        )
        nodes = current_survey_property_nodes(
            qualified,
            self.clone_population.frame,
            host_pins=self.host_pins,
            options=self.options,
        )
        require(
            context.node == {n.id: n for n in nodes}.get(context.node.id),
            "NODE_DECLARATION",
        )
        require(
            set(context.artifacts) == {e.name for e in context.node.artifact_inputs},
            "ARTIFACT_ROSTER",
        )
        for edge in context.node.artifact_inputs:
            value = host.shared.artifact(context, edge.name, edge.type)
            require(
                value.key == opaque_artifact_key(value.producer_key, edge.artifact),
                "ARTIFACT_KEY",
            )
        if context.node.id != PROJECTION_NODE:
            projected = _projection_values(qualified)
            for edge in _source_edges():
                require(
                    context.artifacts[edge.name].payload == projected[edge.artifact],
                    "SOURCE_ARTIFACT_BYTES",
                )
            host.shared.siblings(context, tuple(e.name for e in _source_edges()))
        return qualified


class CurrentSurveyPropertyProjectionKernel(_Kernel):
    ref = "us.survey_property.source_projection@1"

    def run(self, context):
        qualified = self._qualified(context)
        host._current_context_frame(context, qualified.shared_predictors.donor_frame)
        for edge in host.current_survey_host_edges():
            value = context.artifacts[edge.name]
            require(
                self.host_pins[edge.name]
                == {
                    "producer_key": value.producer_key,
                    "artifact_key": value.key,
                    "payload_sha256": codec.sha(value.payload),
                },
                "HOST_PIN",
            )
        require(
            context.artifacts["preparation"].payload == self.preparation.payload,
            "PREPARATION_BYTES",
        )
        require(
            codec.sha(context.artifacts["allocation"].payload)
            == qualified.shared_predictors.evidence["allocation_sha256"]
            and codec.sha(context.artifacts["frame_context"].payload)
            == codec.decode_json(context.artifacts["allocation"].payload)[
                "output_context_sha256"
            ],
            "ALLOCATION_BYTES",
        )
        host.shared.siblings(context, ("allocation", "frame_context"))
        if qualified.shared_predictors.geography_config_payload is not None:
            value = context.artifacts["geography_validation"]
            require(
                self.host_pins["geography_validation"]
                == {
                    "producer_key": value.producer_key,
                    "artifact_key": value.key,
                    "payload_sha256": codec.sha(value.payload),
                }
                and value.payload == qualified.shared_predictors.geography_validation,
                "GEOGRAPHY_BYTES",
            )
        require(
            context.artifacts["predictor_projection"].payload
            == qualified.shared_predictors.projection
            and context.artifacts["predictor_matrix"].payload
            == qualified.shared_predictors.matrix,
            "SHARED_PREDICTOR_BYTES",
        )
        host.shared.siblings(context, ("predictor_projection", "predictor_matrix"))
        projected = _projection_values(qualified)
        return KernelResult(
            artifacts=projected, receipt=codec.decode_json(projected["projection"])
        )


class CurrentSurveyPropertyFilterKernel(_Kernel):
    ref = "us.survey_property.filter@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.FILTER,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=_Kernel.capabilities.dependencies,
    )

    def run(self, context):
        qualified = self._qualified(context)
        qualified_seal = sources.property_income_sources_seal(qualified)
        original = self.preparation._checked()[2].frame
        require(
            sources.property_income_sources_seal(qualified) == qualified_seal,
            "FILTER_FINAL_SOURCE_BORROW",
        )
        host._current_context_frame(context, original)
        frame = (
            qualified.donor_frame
            if context.params["donor"]
            else qualified.recipient_frame
        )
        keep = qualified.source_frame.person.person_id.isin(
            frame.person.person_id
        ).to_numpy()
        return KernelResult(
            keep=pd.Series(
                keep,
                index=pd.Index(
                    qualified.source_frame.person.person_id.to_numpy(), name="person_id"
                ),
            ),
            receipt={
                "protocol": PROTOCOL,
                "branch": "ASEC_eligible_original_design"
                if context.params["donor"]
                else "ACS_adult_known_anchor_original",
                "persons": len(frame.person),
            },
        )


class CurrentSurveyPropertyColumnsKernel(_Kernel):
    ref = "us.survey_property.columns@1"

    def run(self, context):
        qualified = self._qualified(context)
        donor = context.params["donor"]
        frame = qualified.donor_frame if donor else qualified.recipient_frame
        table = qualified.donor_columns if donor else qualified.recipient_columns
        host._current_context_frame(context, frame)
        return KernelResult(
            columns={("person", c): table[c] for c in table},
            receipt={
                "protocol": PROTOCOL,
                "matrix_sha256": codec.sha(
                    context.artifacts[
                        "donor_matrix" if donor else "recipient_matrix"
                    ].payload
                ),
            },
        )


def _clone_context(context, frame):
    # The existing financial attachment owns exactly these eight rewrites.
    # They are separately reconstructed from source observations and legacy
    # draws below; compare every other raw clone input to its retained value.
    retained = Frame(
        {
            e: frame.table(e).loc[
                :, [c for c in frame.table(e) if c not in shared.OUTPUTS]
            ]
            for e in frame.entities
        },
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
    )
    # Keep the executor's structural-first order, independently of Frame order.
    tables = {
        e: context.tables[e].loc[
            :, [c for c in context.tables[e] if c in retained.table(e)]
        ]
        for e in retained.entities
    }
    return host._current_context_frame(
        replace(
            context,
            node=replace(context.node, inputs=financial._inputs(retained)),
            tables=tables,
        ),
        retained,
    )


class CurrentSurveyPropertyAttachKernel(_Kernel):
    ref = "us.survey_property.attach@1"

    def run(self, context):
        qualified = self._qualified(context)
        # Compare every original clone input independently of newly supplied
        # legacy leaves; the final host compares complete Population ownership.
        frame = self.clone_population.frame
        _clone_context(context, frame)
        loaded = {
            (e.producer, e.artifact): context.artifacts[e.name].payload
            for e in context.node.artifact_inputs
        }
        for i in range(4):
            host.shared.siblings(context, (f"raw_{i}", f"state_{i}"))
        legacy_key = context.artifacts["legacy_matrix"].producer_key
        legacy = _legacy_draws(qualified, loaded, legacy_key)
        expected = shared.complete_predictor_columns(
            qualified.shared_predictors, frame, legacy
        )
        for (_, name), column in expected.items():
            require(
                np.array_equal(
                    context.tables["person"][name].to_numpy(),
                    column.to_numpy(),
                    equal_nan=True,
                ),
                "LEGACY_LEAF_CHANGED:" + name,
            )
        results = reconstruct_property_results(
            qualified,
            frame,
            host_pins=self.host_pins,
            options=self.options,
            artifacts=loaded,
            legacy_matrix_producer_key=legacy_key,
        )
        return results[ATTACH_NODE]


def register_property_kernels(
    kernels,
    preparation,
    allocated_population,
    clone_population,
    *,
    host_pins,
    options,
    demographic_conditioning=False,
    geography_config=None,
):
    for cls in (
        CurrentSurveyPropertyProjectionKernel,
        CurrentSurveyPropertyFilterKernel,
        CurrentSurveyPropertyColumnsKernel,
        CurrentSurveyPropertyAttachKernel,
    ):
        kernels.register(
            cls(
                preparation,
                allocated_population,
                clone_population,
                host_pins=host_pins,
                options=options,
                demographic_conditioning=demographic_conditioning,
                geography_config=geography_config,
            )
        )
    for kernel in (
        model.PropertyIncomeTrainKernel(),
        LegacyQRFApplyKernel(),
        model.PropertyIncomeDrawColumnsKernel(),
        signed_graph.SignedReconciliationKernel(),
    ):
        kernels.register(kernel)


def verify_materialized_property_income(
    preparation,
    allocated_population,
    clone_population,
    *,
    legacy_population,
    population,
    host_pins,
    options,
    artifacts,
    legacy_matrix_producer_key,
    demographic_conditioning=False,
    geography_config=None,
):
    """Requalify sources then compare the final complete retained population."""
    qualified = sources.qualify_current_property_income_sources(
        preparation,
        allocated_population,
        clone_population,
        demographic_conditioning=demographic_conditioning,
        geography_config=geography_config,
    )
    results = reconstruct_property_results(
        qualified,
        clone_population.frame,
        host_pins=host_pins,
        options=options,
        artifacts=artifacts,
        legacy_matrix_producer_key=legacy_matrix_producer_key,
    )
    node = current_survey_property_nodes(
        qualified, clone_population.frame, host_pins=host_pins, options=options
    )[-1]
    expected = population_ops.patch(legacy_population, node, results[ATTACH_NODE])
    replay.same_replayed_population(expected, population)
    return dict(results[ATTACH_NODE].receipt)
