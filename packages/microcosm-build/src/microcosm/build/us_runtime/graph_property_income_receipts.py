"""Verify property model receipts against authenticated population/artifact inputs.

This is a source-blind numerical check, not a producer or source issuer. Callers
must authenticate store payloads and the typed producer/population dependency
closure before supplying model bytes: decoding a pickle is not authentication.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import qrf, qrf_target
from microcosm.fit.graph_legacy_qrf import (
    LegacyQRFApplyKernel,
    legacy_qrf_apply_nodes,
    legacy_qrf_train_nodes,
)
from microcosm.frame import WeightKind
from microcosm.graph import Node
from microcosm.graph.canonical import canonical_json, normative
from microcosm.graph.population import Population

from . import graph_property_income as property_graph
from .property_income_constants import PROPERTY_COMPONENTS, PROPERTY_REPORTED_TOTAL


def _require(condition, reason):
    if not condition:
        raise ValueError("PROPERTY_MODEL_RECEIPTS_" + reason)


def _declarations(nodes, donor, recipient):
    _require(
        isinstance(nodes, tuple)
        and all(type(node) is Node for node in nodes)
        and len({node.id for node in nodes}) == len(nodes),
        "NODE_ROSTER",
    )
    fits = tuple(
        n for n in nodes if n.kernel == property_graph.PropertyIncomeTrainKernel.ref
    )
    applies = tuple(n for n in nodes if n.kernel == LegacyQRFApplyKernel.ref)
    _require(len(fits) == len(applies) == len(PROPERTY_COMPONENTS), "MODEL_ROSTER")
    # Declaration order must describe the ordered chain; other fragment nodes
    # (source projection, attachment, reconciliation) are the host's concern.
    first = fits[0]
    _require(first.id.endswith(".fit.000"), "FIT_PREFIX")
    prefix = first.id.removesuffix(".fit.000")
    params = first.params
    features = property_graph._features(params.get("predictors"))
    seed, trees = params.get("seed"), params.get("n_estimators")
    _require(
        bool(prefix)
        and type(seed) is int
        and seed >= 0
        and type(trees) is int
        and trees > 0
        and donor.version != recipient.version,
        "PARAMETERS",
    )
    phase = property_graph.PROTOCOL + ":" + prefix
    expected_fits = legacy_qrf_train_nodes(
        prefix + ".fit",
        population=donor.version,
        entity="person",
        predictors=features,
        targets=PROPERTY_COMPONENTS,
        seed=seed,
        n_estimators=trees,
        zero_atol=0,
        phase=phase,
    )
    expected_applies = legacy_qrf_apply_nodes(
        prefix + ".apply",
        population=recipient.version,
        fit_nodes=expected_fits,
        seed=seed,
        phase=phase,
    )
    expected_fits = tuple(
        replace(node, kernel=property_graph.PropertyIncomeTrainKernel.ref)
        for node in expected_fits
    )
    _require(
        all(
            canonical_json(normative(actual)) == canonical_json(normative(expected))
            for actual, expected in zip(
                (*fits, *applies), (*expected_fits, *expected_applies), strict=True
            )
        ),
        "DECLARATION",
    )
    return fits, applies, features, seed, trees, phase


def _table(population, columns):
    table = population.frame.person
    _require(
        table.person_id.dtype == np.dtype("int64")
        and table.person_id.is_unique
        and all(c in table and table[c].dtype == np.dtype("float64") for c in columns)
        and np.isfinite(table.loc[:, list(columns)].to_numpy()).all(),
        "PERSON_IDENTITY_OR_VALUES",
    )
    return table


def verify_property_model_receipts(
    nodes, donor_population, recipient_population, artifacts
):
    """Return the published eight fit/apply receipts after numerical validation.

    ``nodes`` is the ordered property fragment (additional non-model nodes may
    be present); populations are the exact original donor/recipient branches.
    ``artifacts[(node_id, output_name)]`` holds caller-authenticated bytes.
    All four target fits are checked against the actual design donor and
    chain-start state without fitting again. Applications are replayed using
    those models and the current recipient features, because the legacy apply
    checkpoint carries an index but no independent feature digest.

    The caller owns population identity, source qualification and lifetime
    checks. This function preserves the supplied frames and does not confer
    authority on detached populations, arbitrary model bytes or its result.
    """
    _require(
        type(donor_population) is Population
        and type(recipient_population) is Population
        and isinstance(artifacts, Mapping),
        "INPUT_TYPES",
    )
    fits, applies, features, seed, trees, phase = _declarations(
        nodes, donor_population, recipient_population
    )
    required = {
        (node.id, output.name)
        for node in (*fits, *applies)
        for output in node.artifact_outputs
    }
    _require(
        all(key in artifacts and type(artifacts[key]) is bytes for key in required),
        "ARTIFACT_ROSTER",
    )
    donor = _table(donor_population, (*features, *PROPERTY_COMPONENTS))
    recipient = _table(recipient_population, features).loc[:, list(features)].copy()
    weights = donor_population.frame.resolve_weights("person")
    _require(weights.kind is WeightKind.DESIGN, "ORIGINAL_DESIGN_WEIGHTS")
    _require(
        (donor.loc[:, list(PROPERTY_COMPONENTS[:3])].to_numpy() >= 0).all()
        and np.array_equal(
            donor.loc[:, list(PROPERTY_COMPONENTS)].sum(axis=1).to_numpy(),
            donor[PROPERTY_REPORTED_TOTAL].to_numpy(),
        ),
        "RESOLVED_DONOR_COMPONENTS",
    )
    model_frame = codec.model_frame(
        SimpleNamespace(weights={"person": weights}), fits[0].inputs[0], donor
    )
    model = qrf.RegimeGatedQRF(
        seed=seed, n_estimators=trees, zero_atol=0, max_samples_leaf=None
    )
    application_state = model.start_chain(
        model_frame, list(features), list(PROPERTY_COMPONENTS), weights="design"
    )
    before = qrf_target.LegacyQRFTrainingState.from_chain(application_state)
    receipts, history, raw_history = {}, [], []
    raw = pd.DataFrame(index=recipient.index)
    for i, (fit, apply, target) in enumerate(
        zip(fits, applies, PROPERTY_COMPONENTS, strict=True)
    ):
        payload = artifacts[fit.id, "model"]
        packet, after = codec.read_training(artifacts[fit.id, "training_state"])
        _require(
            len(packet["models"]) == i + 1
            and packet["models"][:i] == history
            and packet["models"][-1]["sha256"] == codec.sha(payload),
            "TRAINING_HISTORY",
        )
        fitted = qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes(
            payload, expected_sha256=packet["models"][-1]["sha256"]
        )
        _require(
            fitted.training_state == before
            and fitted.next_training_state == after
            and fitted.donor_sha256
            == qrf_target._consumed_values_sha256(
                model_frame.person, (*features, *PROPERTY_COMPONENTS[:i], target)
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
        _require(packet["models"] == history, "TRAINING_HISTORY")
        before = after
        application, chain = codec.read_application(artifacts[apply.id, "apply_state"])
        raw_payload = artifacts[apply.id, "raw_draw"]
        actual_raw = codec.read_raw_target(
            raw_payload, target=target, index=recipient.index
        )
        raw_history.append({"target": target, "sha256": codec.sha(raw_payload)})
        _require(
            application["seed"] == seed
            and application["models"] == history
            and application["raw_targets"] == raw_history
            and qrf_target.LegacyQRFTrainingState.from_chain(chain) == after
            and chain.recipient_index == qrf._index_identity(recipient.index),
            "APPLICATION_HISTORY",
        )
        result = qrf_target.apply_target(
            fitted, recipient, raw, state=application_state
        )
        _require(
            result.state == chain
            and codec.encode_raw_target(
                result.raw_draw, target=target, index=recipient.index
            )
            == raw_payload,
            "APPLICATION_VALUES",
        )
        application_state = result.state
        raw[target] = actual_raw
        receipts[fit.id] = {
            "phase": phase,
            "target": target,
            "training_id": fitted.training_id,
            "model_sha256": codec.sha(payload),
            "donor_rows": len(donor),
            "entity": "person",
            "weight_kind": "design",
            "regime": fitted.regime,
        }
        receipts[apply.id] = {
            "phase": phase,
            "target": target,
            "recipient_rows": len(recipient),
            "entity": "person",
            "model_sha256": codec.sha(payload),
            "raw_sha256": codec.sha(raw_payload),
            "regime": fitted.regime,
        }
    return receipts
