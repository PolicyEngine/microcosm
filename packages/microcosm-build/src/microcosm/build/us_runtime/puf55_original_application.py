"""Explicit original-arm apply declarations and strict detached v2 merging.

No model is deserialized here and no population is attached or issued. The host
must authenticate the genuine fitted producers against its compiled graph and
requalify the financial/fixed inputs before and after its I/O. ArtifactValue,
a receipt, or a caller digest alone cannot establish source/model authority.
Mixed-person knownness remains unresolved for attachment; it is not permission
to allocate an aggregate draw over known people.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.fit import qrf, qrf_target
from microcosm.fit.qrf_target import LEGACY_QRF_TARGET_TYPE
from microcosm.graph import ArtifactValue, Numeric, platform_fingerprint
from microcosm.graph.keys import opaque_artifact_key

from . import graph_puf55_route_attachment as attachment
from . import graph_puf55_survey_observed as fixed_graph

values = fixed_graph.values
codec, observed = values.codec, values.observed
PROTOCOL = "microcosm.us.puf55-original-conditioning-merge.v1"


def require(condition, reason):
    if not condition:
        raise ValueError("PUF55_ORIGINAL_APPLICATION_" + reason)


def _seeds(clone_one_seed, original_application_seed):
    require(
        type(clone_one_seed) is type(original_application_seed) is int
        and 0 <= clone_one_seed < 2**64
        and 0 <= original_application_seed < 2**64
        and clone_one_seed != original_application_seed,
        "SEEDS",
    )


def original_application_nodes(
    qualified, routes, *, population, clone_one_seed, original_application_seed
):
    """Reuse each actual route's fits; declare all55 original-arm applications."""
    _seeds(clone_one_seed, original_application_seed)
    values.check_fixed_input_binding(qualified)
    qualified_seal = values.fixed_input_stamp(qualified)
    require(qualified.recipients.arm == 0, "ARM")
    require(type(routes) is tuple and routes, "ROUTES")
    require(
        all(type(route) is attachment.Route for route in routes)
        and tuple(route.profile.value for route in routes)
        == tuple(name for name, _ in qualified.recipients.matrices),
        "ROUTE_ROSTER",
    )
    result = []
    for route in routes:
        # Recreate only the declarations, never fit or change donor/RNG state.
        # This binds every existing fit/apply node, not just its target label.
        require(len(route.fits) == len(route.profile.targets) == 55, "FIT_ROSTER")
        first = route.fits[0]
        expected = attachment.route_nodes(
            route.profile,
            seed=clone_one_seed,
            n_estimators=first.params["n_estimators"],
            zero_atol=first.params["zero_atol"],
        )
        require(route == expected, "REUSED_ROUTE_DECLARATION")
        prefix = "survey_puf55.original." + (
            "nine" if route.profile is attachment.PROFILES[0] else "eight"
        )
        result.extend(
            observed.legacy_qrf_apply_observed_matrix_nodes(
                prefix + ".apply",
                population=population,
                fit_nodes=route.fits,
                matrix_producer=fixed_graph.parent.recipient_node_ids(0)[1],
                matrix_artifact=fixed_graph.parent._NAMES[route.profile.value],
                observed_producer=fixed_graph.fixed_input_node_id(0),
                observed_artifacts=fixed_graph.observed_artifact_names(
                    qualified, route.profile.value
                ),
                seed=original_application_seed,
                phase=route.profile.phase,
            )
        )
    require(
        values.fixed_input_stamp(qualified) == qualified_seal,
        "DECLARATION_INPUT_CHANGED",
    )
    return tuple(result)


@dataclass(frozen=True)
class OriginalTargetArtifacts:
    """Actual typed transport values; source authority is an upstream duty."""

    model: ArtifactValue
    training_state: ArtifactValue
    raw_draw: ArtifactValue
    conditioning: ArtifactValue
    apply_state: ArtifactValue


@dataclass(frozen=True)
class OriginalRouteArtifacts:
    profile: values.recipients.full.PufOutputProfile
    matrix: ArtifactValue
    steps: tuple[OriginalTargetArtifacts, ...]
    fixed_inputs: tuple[tuple[str, ArtifactValue], ...]


def _route_stamp(route):
    require(type(route) is OriginalRouteArtifacts, "ROUTE_TYPE")
    value = route.matrix
    return (
        route.profile,
        (value.payload, value.type, value.key, value.producer_key, value.numerics),
        _step_stamp(route.steps),
        tuple(
            (k, v.payload, v.type, v.key, v.producer_key, v.numerics)
            for k, v in route.fixed_inputs
        ),
    )


def _edge(value, type_, name):
    require(
        type(value) is ArtifactValue
        and type(value.payload) is bytes
        and value.type == type_
        and codec._hash(value.producer_key)
        and value.key == opaque_artifact_key(value.producer_key, name)
        and value.numerics.numeric is Numeric.PLATFORM_BITWISE
        and value.numerics.platform == platform_fingerprint(),
        "TYPED_EDGE",
    )
    return value.payload


def _step_stamp(steps):
    require(
        type(steps) is tuple and all(type(s) is OriginalTargetArtifacts for s in steps),
        "STEP_TYPES",
    )
    return tuple(
        tuple(
            (v.payload, v.type, v.key, v.producer_key, v.numerics)
            for v in (
                s.model,
                s.training_state,
                s.raw_draw,
                s.conditioning,
                s.apply_state,
            )
        )
        for s in steps
    )


def _decode_chain(
    *,
    matrix,
    matrix_name,
    steps,
    fixed_inputs,
    expected_fixed,
    targets,
    predictors,
    entity,
    clone_one_seed,
    original_application_seed,
):
    """Verify actual per-step envelopes/bytes without executing or unpickling.

    Private generic checker permits a genuine tiny component proof. The public
    US merger below separately requires the exact closed55-target profiles.
    """
    _seeds(clone_one_seed, original_application_seed)
    matrix_bytes = _edge(
        matrix, fixed_graph.parent.model_input.RECIPIENT_MATRIX_TYPE, matrix_name
    )
    prepared = fixed_graph.parent.model_input.decode_recipient_matrix(matrix_bytes)
    require(
        prepared.entity == entity and tuple(prepared.features) == predictors,
        "MATRIX_PROFILE",
    )
    require(
        type(targets) is tuple
        and len(steps) == len(targets)
        and len(set(targets)) == len(targets),
        "TARGET_ROSTER",
    )
    stamp = _step_stamp(steps)
    require(
        len({s.model.producer_key for s in steps}) == len(steps)
        and len({s.apply_state.producer_key for s in steps}) == len(steps),
        "UNIQUE_STEP_PRODUCERS",
    )
    require(
        type(fixed_inputs) is tuple
        and all(type(p) is tuple and len(p) == 2 for p in fixed_inputs),
        "FIXED_ROSTER",
    )
    fixed = dict(fixed_inputs)
    require(
        len(fixed) == len(fixed_inputs)
        and tuple(fixed) == tuple(expected_fixed)
        and set(fixed) <= set(targets),
        "FIXED_ROSTER",
    )
    fixed_stamp = tuple(
        (k, v.payload, v.type, v.key, v.producer_key, v.numerics)
        for k, v in fixed_inputs
    )
    expected_stamp = tuple(expected_fixed.items())
    if fixed:
        require(len({v.producer_key for v in fixed.values()}) == 1, "FIXED_PRODUCER")
    for target, artifact in fixed.items():
        name, payload = expected_fixed[target]
        require(
            _edge(artifact, observed.OBSERVED_TARGET_TYPE, name) == payload,
            "FIXED_SOURCE_VALUES",
        )
    models, history, prior_keys, columns, evidence = [], [], [], {}, []
    binding = dict(
        matrix_sha256=codec.sha(matrix_bytes), matrix_producer_key=matrix.producer_key
    )
    for i, (target, step) in enumerate(zip(targets, steps, strict=True)):
        model = _edge(step.model, LEGACY_QRF_TARGET_TYPE, "model")
        training_bytes = _edge(
            step.training_state, codec.TRAINING_STATE_TYPE, "training_state"
        )
        raw_bytes = _edge(step.raw_draw, codec.RAW_TARGET_TYPE, "raw_draw")
        conditioning_bytes = _edge(
            step.conditioning, observed.CONDITIONING_TARGET_TYPE, "conditioning"
        )
        state_bytes = _edge(
            step.apply_state, observed.OBSERVED_MATRIX_APPLY_STATE_TYPE, "apply_state"
        )
        require(
            step.model.producer_key == step.training_state.producer_key
            and len(
                {
                    step.raw_draw.producer_key,
                    step.conditioning.producer_key,
                    step.apply_state.producer_key,
                }
            )
            == 1,
            "STEP_PRODUCERS",
        )
        train_packet, train_state = codec.read_training(training_bytes)
        training = train_state.to_dict()
        require(
            tuple(training["targets"]) == targets
            and tuple(training["completed_targets"]) == targets[: i + 1]
            and tuple(training["predictors"]) == predictors
            and training["entity"] == entity
            and training["model_config"]["seed"] == clone_one_seed
            and train_packet["models"][:-1] == models
            and train_packet["models"][-1]["sha256"] == codec.sha(model),
            "TRAINING_HISTORY",
        )
        packet = observed.decode_observed_matrix_apply_state(state_bytes)
        application = packet["application"]
        chain = qrf.QRFChainState.from_dict(application["state"])
        require(
            {k: packet[k] for k in binding} == binding
            and application["seed"] == original_application_seed
            and application["models"] == train_packet["models"]
            and qrf_target.LegacyQRFTrainingState.from_chain(chain) == train_state
            and chain.recipient_index == qrf._index_identity(prepared.features.index)
            and application["raw_targets"][:-1] == history
            and application["prior_producer_keys"] == prior_keys,
            "APPLICATION_HISTORY",
        )
        raw = codec.read_raw_target(
            raw_bytes, target=target, index=prepared.features.index
        )
        merged = raw.copy()
        observation = fixed.get(target)
        if observation is not None:
            metadata, supplied, known = observed._read_observed(
                observation.payload,
                target=target,
                index=prepared.features.index,
                **binding,
            )
            merged[known] = supplied[known]
        else:
            metadata, known = {}, np.zeros(len(raw), dtype=bool)
        expected_record = dict(
            target=target,
            draw_sha256=codec.sha(raw_bytes),
            conditioning_sha256=codec.sha(conditioning_bytes),
            model_producer_key=step.model.producer_key,
            observed_sha256=None
            if observation is None
            else codec.sha(observation.payload),
            observed_producer_key=None
            if observation is None
            else observation.producer_key,
            source_sha256=None if observation is None else metadata["source_sha256"],
            observed_rows=int(known.sum()),
        )
        require(
            np.isfinite(raw).all()
            and np.isfinite(merged).all()
            and codec.encode_raw_target(
                merged, target=target, index=prepared.features.index
            )
            == conditioning_bytes
            and application["raw_targets"][-1] == expected_record,
            "RAW_CONDITIONING_BINDING",
        )
        columns[target] = merged
        evidence.append(
            {
                **expected_record,
                "fit_seed": clone_one_seed,
                "application_seed": original_application_seed,
                "apply_producer_key": step.apply_state.producer_key,
                "training_sha256": codec.sha(training_bytes),
                "state_sha256": codec.sha(state_bytes),
            }
        )
        models, history = train_packet["models"], application["raw_targets"]
        prior_keys.append(step.apply_state.producer_key)
    require(
        _step_stamp(steps) == stamp
        and tuple(
            (k, v.payload, v.type, v.key, v.producer_key, v.numerics)
            for k, v in fixed_inputs
        )
        == fixed_stamp
        and tuple(expected_fixed.items()) == expected_stamp
        and _edge(
            matrix, fixed_graph.parent.model_input.RECIPIENT_MATRIX_TYPE, matrix_name
        )
        == matrix_bytes,
        "FINAL_INPUT_CHANGED",
    )
    return pd.DataFrame(columns, index=prepared.features.index), evidence


def merge_puf55_original_conditioning(
    qualified, routes, *, clone_one_seed, original_application_seed
):
    """Merge complete arm0 v2 routes only; never attach or infer owner authority."""
    _seeds(clone_one_seed, original_application_seed)
    values.check_fixed_input_binding(qualified)
    seal = values.fixed_input_stamp(qualified)
    require(qualified.recipients.arm == 0, "ARM")
    require(
        type(routes) is tuple
        and all(type(r) is OriginalRouteArtifacts for r in routes),
        "ROUTES",
    )
    require(
        tuple(r.profile.value for r in routes)
        == tuple(p for p, _ in qualified.recipients.matrices),
        "ROUTE_ROSTER",
    )
    tables, evidence, ids = [], [], []
    snapshots = tuple(_route_stamp(r) for r in routes)
    for route, (_, expected_matrix) in zip(
        routes, qualified.recipients.matrices, strict=True
    ):
        require(
            route.profile in attachment.PROFILES
            and len(route.profile.targets) == 55
            and route.matrix.payload == expected_matrix,
            "PROFILE_MATRIX",
        )
        names = fixed_graph.observed_artifact_names(qualified, route.profile.value)
        fixed = values.observed_target_artifacts(
            qualified,
            profile=route.profile.value,
            matrix_payload=expected_matrix,
            matrix_producer_key=route.matrix.producer_key,
        )
        table, history = _decode_chain(
            matrix=route.matrix,
            matrix_name=fixed_graph.parent._NAMES[route.profile.value],
            steps=route.steps,
            fixed_inputs=route.fixed_inputs,
            expected_fixed={
                target: (names[target], payload) for target, payload in fixed.items()
            },
            targets=route.profile.targets,
            predictors=route.profile.predictors,
            entity="tax_unit",
            clone_one_seed=clone_one_seed,
            original_application_seed=original_application_seed,
        )
        tables.append(table)
        ids.extend(table.index.tolist())
        evidence.append(
            {
                "profile": route.profile.value,
                "matrix_sha256": codec.sha(expected_matrix),
                "matrix_producer_key": route.matrix.producer_key,
                "history": history,
            }
        )
    # The qualifier's matrix roster is the exact complete arm0 axis. Preserve
    # its receiving tax-unit order instead of concatenated route order.
    require(len(ids) == len(set(ids)), "OVERLAPPING_ROUTES")
    ordered = qualified.recipients.tax_unit.index
    ordered = ordered[ordered.isin(ids)]
    require(set(ordered) == set(ids), "INCOMPLETE_AXIS")
    combined = pd.concat(tables).loc[ordered].copy(deep=True)
    table_seal = values.recipients._table_digest(combined)
    receipt = codec.encode_json(
        {
            "protocol": PROTOCOL,
            "recipient_arm": 0,
            "fit_seed": clone_one_seed,
            "clone_one_application_seed": clone_one_seed,
            "original_application_seed": original_application_seed,
            "qualification_sha256": codec.sha(qualified.receipt),
            "routes": evidence,
            "conditioning_table_sha256": table_seal,
            "raw_draws_retained": True,
            "finalization_performed": False,
            "mixed_person_knownness_resolved": False,
            "source_admission_issued": False,
            "release_eligible": False,
        }
    )
    require(
        values.fixed_input_stamp(qualified) == seal
        and snapshots == tuple(_route_stamp(r) for r in routes)
        and values.recipients._table_digest(combined) == table_seal,
        "FINAL_MERGE_CHANGED",
    )
    # Rebind the detached result to each immutable per-target conditioning
    # payload after all route decoding and caller-visible qualifier checks.
    for route in routes:
        index = fixed_graph.parent.model_input.decode_recipient_matrix(
            route.matrix.payload
        ).features.index
        for target, step in zip(route.profile.targets, route.steps, strict=True):
            require(
                codec.encode_raw_target(
                    combined.loc[index, target].to_numpy(), target=target, index=index
                )
                == step.conditioning.payload,
                "FINAL_CONDITIONING_CHANGED",
            )
    return combined, receipt
