"""Recompute supported age diagnostics without rerunning an optimizer.

This is a value checker, not source authority. The caller supplies independently
verified ordered counts, sampling bounds and receiving weights. Optimizer history
has a bounded schema but cannot be reconstructed from those final values.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from microcosm.calibrate import diagnostics_payload
from microcosm.calibrate.score import score_targets
from microcosm.frame import WeightKind, Weights
from microcosm.graph.canonical import canonical_json

from . import graph_survey_calibration as numeric

MAX_DIAGNOSTIC_BYTES = 8 * 1024**2
HISTORY_FIELDS = (
    "initial_loss",
    "loss_trajectory",
    "options.matrix_format",
    "options.grouped_upper_bounds.last_corrected_group_count",
)


def _require(condition, reason):
    if not condition:
        raise ValueError("SURVEY_DIAGNOSTICS_" + reason)


def _history(document, epochs, group_count):
    options = document.get("options")
    _require(type(options) is dict, "OPTIONS")
    grouped = options.get("grouped_upper_bounds")
    _require(type(grouped) is dict, "GROUPED_OPTIONS")
    corrected = grouped.get("last_corrected_group_count")
    _require(type(corrected) is int and 0 <= corrected <= group_count, "HISTORY_COUNT")
    layout = options.get("matrix_format")
    _require(
        type(layout) is str and layout in {"dense", "sparse_csr"}, "HISTORY_LAYOUT"
    )
    trajectory = document.get("loss_trajectory")
    _require(type(trajectory) is list and len(trajectory) == epochs, "HISTORY_LENGTH")
    _require(
        all(type(v) is float and np.isfinite(v) and 0 <= v <= 10 for v in trajectory),
        "HISTORY_LOSS",
    )
    _require(
        type(document.get("initial_loss")) is float
        and document["initial_loss"] == trajectory[0],
        "HISTORY_INITIAL",
    )
    return corrected, layout, trajectory


def validate_survey_calibration_diagnostics(
    payload,
    *,
    counts_payload,
    bounds_payload,
    weights,
    registry,
    epochs,
    learning_rate,
    anchors,
    activation_binding=None,
):
    """Return defensive schema-8 values plus an explicit verification annotation.

    Every field except the four named history fields is reconstructed using
    score_targets and the package's diagnostics encoder. Only warning-free
    schema 8 (every registry-backed target carrying its calibration hierarchy)
    with target-loss attribution is supported in this first profile.
    No convergence, optimality, target eligibility or release claim is granted.
    """
    _require(
        type(payload) is bytes and 0 < len(payload) <= MAX_DIAGNOSTIC_BYTES,
        "PAYLOAD_LIMIT",
    )
    try:
        document = json.loads(payload)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("SURVEY_DIAGNOSTICS_JSON") from None
    _require(type(document) is dict, "DOCUMENT")
    _require(canonical_json(document) == payload, "CANONICAL")
    # Reuse the actual declaration's option/registry restrictions. This is a
    # declaration and numeric work Frame, never a population/source substitute.
    factory = numeric.survey_age_calibration_node
    profile_options = {}
    if activation_binding is not None:
        factory = numeric.survey_age_development_node
        profile_options["activation_binding"] = activation_binding
    factory(
        registry,
        **profile_options,
        base="diagnostic-values",
        budget_node="bounds",
        count_node="counts",
        epochs=epochs,
        learning_rate=learning_rate,
    )
    bounds = numeric.decode_numeric_survey_bounds(bounds_payload)
    counts = numeric.ages.decode_survey_age_counts(counts_payload)
    _require(
        counts.population == bounds.population
        and tuple(counts.household_ids) == bounds.grouped.household_ids,
        "ORDERED_POPULATION",
    )
    numeric.check_numeric_survey_weights(bounds, weights)
    accepted_weights = weights.tobytes()
    corrected, layout, trajectory = _history(
        document, epochs, bounds.grouped.group_count
    )
    table = counts.counts.copy(deep=True)
    table.insert(0, "household_id", counts.household_ids.copy())
    work = numeric.demographic.kernels_module._frame_from_context(
        SimpleNamespace(
            tables={"household": table},
            weights={
                "household": Weights(bounds.incoming.copy(), WeightKind.IMPORTANCE)
            },
        ),
        "household",
    )
    targets = registry.to_target_set()
    initial = score_targets(
        work, targets, weights=bounds.incoming, target_loss_cap=10.0
    )
    final = score_targets(work, targets, weights=weights, target_loss_cap=10.0)
    _require(not initial.skipped and not final.skipped, "SKIPPED_TARGETS")
    options = {
        # The closed grouped profile supplies no gates or record-budget search.
        # Reconstruct these solver fields; never inherit them from the payload.
        "gate_initialization_supplied": False,
        "budget_basis": "nonzero_count",
        "feasible_draw_pi_hi": None,
        "budget_search": None,
        "grouped_preserve_zeros": {
            "enabled": True,
            "fixed_zero_count": int(np.count_nonzero(bounds.incoming == 0)),
            "ordered_zero_mask_sha256": hashlib.sha256(
                np.asarray(bounds.incoming == 0, dtype=np.uint8).tobytes()
            ).hexdigest(),
        },
        "grouped_upper_bounds": bounds.grouped.diagnostics(weights, corrected),
        "method": "adam",
        "epochs": epochs,
        "learning_rate": learning_rate,
        # This profile always uses grouped bounds, whose accepted weights are
        # the closing state. Ungrouped Adam's best-iterate history cannot apply.
        "iterate_selection": "closing_state",
        "iterate_selection_receipt": {},
        "mass": "free",
        "mass_reason": None,
        "max_weight_ratio": None,
        "target_records": None,
        "l1_lambda": 0.0,
        "l1_penalty": "mean_initial_weight_ratio_abs",
        "l2_lambda": 0.0,
        "l2_anchor": "initial",
        "l2_anchor_weights_supplied": False,
        "l2_penalty": "mean_initial_pre_gate_weight_ratio_squared",
        "seed": 0,
        "target_loss_weights": final.options["target_loss_weights"],
        "target_loss_scales": final.options["target_loss_scales"],
        "warm_start_weights": {"enabled": False, "kind": None},
        "matrix_format": layout,
    }
    combined = replace(
        final,
        diagnostics=tuple(
            replace(after, initial_estimate=before.final_estimate)
            for before, after in zip(
                initial.diagnostics, final.diagnostics, strict=True
            )
        ),
        options=options,
        loss_trajectory=np.asarray(trajectory, dtype=np.float64),
    )
    expected = diagnostics_payload(combined, target_registry=registry, build=anchors)
    _require(
        expected.get("schema_version") == 8
        and expected.get("diagnostic_warnings") == []
        and "target_loss_basis" in expected,
        "UNSUPPORTED_DIAGNOSTIC_SCHEMA",
    )
    _require(canonical_json(expected) == payload, "RECOMPUTED_VALUES")
    # Make the distinction visible to downstream callers without altering the
    # stored schema-8 evidence or its digest.
    document["verification"] = {
        "protocol": "microcosm.us.survey-age-diagnostics-verification.v1",
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "recomputed": "all artifact fields except the listed optimizer history",
        "optimizer_history_schema_checked_only": list(HISTORY_FIELDS),
        "optimizer_rerun": False,
        "release_eligible": False,
    }
    numeric.check_numeric_survey_weights(bounds, weights)
    _require(weights.tobytes() == accepted_weights, "FINAL_WEIGHT_BYTES")
    return document
