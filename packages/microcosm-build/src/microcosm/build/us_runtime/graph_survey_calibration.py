"""Source-blind age calibration over typed numeric budget and count artifacts.

These kernels do not authenticate a survey. The country runner must reconstruct
the source-qualified budget and independently admit the complete population on
cold execution, cached restoration and final return. Frozen incoming weights
are numeric artifact values, never replacement original DESIGN anchors.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, replace

import numpy as np

from microcosm.calibrate import calibrate, diagnostics_payload, group_bounds
from microcosm.frame import WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Node,
    Numeric,
    SeedSource,
    StructuralDelta,
    WeightTransition,
    source_hash,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.keys import opaque_artifact_key

from . import demographic_calibration_graph as demographic
from . import graph_survey_age_artifact as ages
from . import survey_age_activation as age_activation

BOUNDS_TYPE = ArtifactType("microcosm.us.survey_calibration_numeric_bounds", 1)
BOUNDS_PROTOCOL = "microcosm.us.survey-calibration-numeric-bounds.v1"
MAX_BYTES = 64 * 1024**2
MAX_ROWS = MAX_BYTES // 128
CALIBRATION_NODE = "survey.age_calibration"
_FIELDS = frozenset(
    {
        "protocol",
        "budget_sha256",
        "population",
        "household_ids",
        "group_indices",
        "group_upper_hex",
        "row_upper_hex",
        "incoming_hex",
    }
)


def _require(condition, reason):
    if not condition:
        raise ValueError("SURVEY_CALIBRATION_" + reason)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _digest(value):
    return (
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _vector(raw, length):
    _require(type(raw) is list and len(raw) == length, "VECTOR_LENGTH")
    _require(all(type(v) is str and 0 < len(v) <= 32 for v in raw), "FLOAT_TOKEN")
    try:
        values = np.array([float.fromhex(v) for v in raw], dtype=np.float64)
    except (ValueError, OverflowError):
        raise ValueError("SURVEY_CALIBRATION_FLOAT_TOKEN") from None
    _require(
        np.all(np.isfinite(values) & (values >= 0))
        and all(float(v).hex() == text for v, text in zip(values, raw, strict=True)),
        "FLOAT_VALUES",
    )
    return values


@dataclass(frozen=True)
class NumericSurveyBounds:
    """Decoded numbers only; neither a source certificate nor a live budget."""

    budget_sha256: str
    population: str
    grouped: group_bounds.GroupedUpperBounds
    incoming: np.ndarray
    row_upper: np.ndarray


def decode_numeric_survey_bounds(payload: bytes) -> NumericSurveyBounds:
    _require(type(payload) is bytes and 0 < len(payload) <= MAX_BYTES, "PAYLOAD")
    try:
        document = json.loads(payload)
    except (ValueError, UnicodeError):
        raise ValueError("SURVEY_CALIBRATION_JSON") from None
    _require(type(document) is dict and set(document) == _FIELDS, "FIELDS")
    _require(canonical_json(document) == payload, "CANONICAL")
    _require(document["protocol"] == BOUNDS_PROTOCOL, "PROTOCOL")
    _require(_digest(document["budget_sha256"]), "BUDGET_DIGEST")
    _require(
        type(document["population"]) is str and 0 < len(document["population"]) <= 256,
        "POPULATION",
    )
    ids, indices = document["household_ids"], document["group_indices"]
    _require(type(ids) is list and 0 < len(ids) <= MAX_ROWS, "ROW_COUNT")
    _require(
        all(type(v) is int and -(2**63) <= v < 2**63 for v in ids)
        and all(a < b for a, b in zip(ids, ids[1:], strict=False)),
        "ORDERED_IDS",
    )
    _require(type(indices) is list and len(indices) == len(ids), "GROUP_INDICES")
    raw_upper = document["group_upper_hex"]
    _require(type(raw_upper) is list and 0 < len(raw_upper) <= len(ids), "GROUP_COUNT")
    _require(
        all(type(v) is int and 0 <= v < len(raw_upper) for v in indices),
        "GROUP_INDICES",
    )
    upper = _vector(raw_upper, len(raw_upper))
    incoming = _vector(document["incoming_hex"], len(ids))
    row_upper = _vector(document["row_upper_hex"], len(ids))
    grouped = group_bounds.GroupedUpperBounds(ids, indices, upper)
    grouped.check(incoming, positive=False)
    _require(np.all(incoming <= row_upper), "INITIAL_ROW_CAP")
    _require(np.any(incoming > 0), "EMPTY_POSITIVE_SUPPORT")
    return NumericSurveyBounds(
        document["budget_sha256"], document["population"], grouped, incoming, row_upper
    )


def check_numeric_survey_weights(bounds, values):
    """Independent numerical checks, including strict retained zero support."""
    _require(type(bounds) is NumericSurveyBounds, "BOUNDS_TYPE")
    _require(
        type(values) is np.ndarray
        and values.dtype == np.dtype("float64")
        and values.shape == bounds.incoming.shape,
        "WEIGHT_ARRAY",
    )
    bounds.grouped.check(values, positive=False)
    _require(np.all(values <= bounds.row_upper), "ROW_REFERENCE_CAP")
    _require(np.array_equal(values == 0, bounds.incoming == 0), "FIXED_ZERO_SUPPORT")


def survey_age_calibration_node(
    registry, *, base, budget_node, count_node, epochs, learning_rate
):
    frozen = demographic._registry_from_json(demographic._registry_json(registry))
    _require(
        tuple(s.measure for s in frozen) == ages._COLUMNS
        and all(s.metadata["evidence_scope"] == "invented" for s in frozen),
        "INVENTED_AGE_REGISTRY_REQUIRED",
    )
    return _survey_age_calibration_node(
        frozen,
        base=base,
        budget_node=budget_node,
        count_node=count_node,
        epochs=epochs,
        learning_rate=learning_rate,
    )


def survey_age_development_node(
    registry,
    *,
    activation_binding,
    base,
    budget_node,
    count_node,
    epochs,
    learning_rate,
):
    """Declare source-documented numbers; the country runner activates sources."""
    frozen = age_activation.validate_survey_age_registry(registry, activation_binding)
    return _survey_age_calibration_node(
        frozen,
        base=base,
        budget_node=budget_node,
        count_node=count_node,
        epochs=epochs,
        learning_rate=learning_rate,
        activation_binding=activation_binding,
    )


def _survey_age_calibration_node(
    frozen,
    *,
    base,
    budget_node,
    count_node,
    epochs,
    learning_rate,
    activation_binding=None,
):
    _require(type(epochs) is int and 1 <= epochs <= 1000, "EPOCHS")
    _require(
        type(learning_rate) in (int, float)
        and np.isfinite(learning_rate)
        and 0 < learning_rate <= 1,
        "LEARNING_RATE",
    )
    activation_params = {}
    if activation_binding is not None:
        declaration = age_activation.declaration_from_binding(activation_binding)
        activation_params["activation"] = canonical_json(
            age_activation.activation_binding(declaration)
        ).decode()
    return Node(
        CALIBRATION_NODE,
        SurveyAgeCalibrationKernel.ref,
        base=base,
        structural=StructuralDelta.REWEIGHT,
        weights=WeightTransition("household", "calibrated", mass="free"),
        mass="free",
        params={
            **activation_params,
            "registry": demographic._registry_json(frozen),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_anchor": "source_qualified_sampling_reference",
            "cap_enforcement": "mandatory_country_runner_cold_cache_and_final",
            "grouped_preserve_zeros": True,
            "method": "adam",
            "seed": 0,
            "l2_lambda": 0.0,
            "l2_anchor": "initial",
            "initial_anchor_meaning": "frozen_incoming_clone_weights",
        },
        artifact_inputs=(
            ArtifactInput("bounds", budget_node, "numeric_bounds", BOUNDS_TYPE),
            ArtifactInput("counts", count_node, "counts", ages.COUNTS_TYPE),
        ),
        artifact_outputs=(ArtifactOutput("diagnostics", demographic.DIAGNOSTICS_TYPE),),
    )


def calibration_receipt_scope(params):
    """Identify numerical evidence scope without claiming source admission."""
    return (
        "source_documented_survey_age_calibration_numbers_only"
        if "activation" in params
        else "invented_survey_age_calibration_numbers_only"
    )


def calibration_activation_binding(params):
    """Decode the bounded canonical-text profile without acquiring its sources."""
    if "activation" not in params:
        return None
    value = params["activation"]
    _require(type(value) is str and 0 < len(value.encode()) <= 65536, "ACTIVATION_TEXT")
    try:
        binding = json.loads(value)
        encoded = canonical_json(binding).decode()
    except (ValueError, TypeError, RecursionError, OverflowError):
        raise ValueError("SURVEY_CALIBRATION_ACTIVATION_JSON") from None
    _require(encoded == value, "ACTIVATION_CANONICAL")
    age_activation.declaration_from_binding(binding)
    return binding


def _artifact(context, name, type_, output):
    value = context.artifacts[name]
    _require(
        value.type == type_
        and value.key == opaque_artifact_key(value.producer_key, output)
        and type(value.payload) is bytes,
        "ARTIFACT_EDGE",
    )
    return value.payload


class SurveyAgeCalibrationKernel(KernelBase):
    """Real fixed-zero grouped Adam on a minimal numeric work Frame."""

    ref = "us.survey_age_calibration@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.NONE,
        structural=StructuralDelta.REWEIGHT,
        consumes_se=False,
        dependencies=("numpy", "pandas", "scipy", "torch"),
    )

    def implementation_hash(self):
        return _sha(
            canonical_json(
                {
                    "adapter": source_hash(
                        sys.modules[__name__],
                        group_bounds,
                        dependencies=self.capabilities.dependencies,
                    ),
                    "age_artifact": ages.SurveyAgeCountArtifactKernel().implementation_hash(),
                    "activation_declaration": source_hash(
                        age_activation,
                        age_activation.age,
                        age_activation.survey_age_requests,
                    ),
                    "solver": demographic.DemographicCalibrationKernel().implementation_hash(),
                }
            )
        )

    def run(self, context):
        aliases = context.node.artifact_inputs
        _require(tuple(a.name for a in aliases) == ("bounds", "counts"), "ALIASES")
        registry = demographic._registry_from_json(context.params["registry"])
        factory = survey_age_calibration_node
        profile_options = {}
        if "activation" in context.params:
            factory = survey_age_development_node
            profile_options["activation_binding"] = calibration_activation_binding(
                context.params
            )
        expected = factory(
            registry,
            **profile_options,
            base=context.node.base,
            budget_node=aliases[0].producer,
            count_node=aliases[1].producer,
            epochs=context.params["epochs"],
            learning_rate=context.params["learning_rate"],
        )
        _require(context.node.normative() == expected.normative(), "DECLARATION")
        _require(dict(context.params) == dict(expected.params), "PARAMETERS")
        _require(
            not context.tables and not context.weights and not context.sources,
            "CONTEXT",
        )
        bounds_raw = _artifact(context, "bounds", BOUNDS_TYPE, "numeric_bounds")
        counts_raw = _artifact(context, "counts", ages.COUNTS_TYPE, "counts")
        bounds = decode_numeric_survey_bounds(bounds_raw)
        counts = ages.decode_survey_age_counts(counts_raw)
        _require(
            bounds.population == counts.population == context.node.base
            and tuple(counts.household_ids) == bounds.grouped.household_ids,
            "COMPLETE_ORDERED_POPULATION",
        )
        table = counts.counts.copy(deep=True)
        table.insert(0, "household_id", counts.household_ids.copy())
        incoming = Weights(bounds.incoming.copy(), WeightKind.IMPORTANCE)
        work = demographic.kernels_module._frame_from_context(
            replace(
                context, tables={"household": table}, weights={"household": incoming}
            ),
            "household",
        )
        result = calibrate(
            work,
            registry.to_target_set(),
            weight_entity="household",
            method="adam",
            seed=0,
            epochs=context.params["epochs"],
            learning_rate=context.params["learning_rate"],
            mass="free",
            max_weight_ratio=None,
            grouped_upper_bounds=bounds.grouped,
            grouped_preserve_zeros=True,
            l2_lambda=context.params["l2_lambda"],
            l2_anchor=context.params["l2_anchor"],
        )
        _require(not result.skipped, "SKIPPED_TARGETS")
        weights = result.frame.weights_for("household")
        accepted = weights.values.tobytes()
        check_numeric_survey_weights(bounds, weights.values)
        anchors = {
            "budget_sha256": bounds.budget_sha256,
            "numeric_bounds_sha256": _sha(bounds_raw),
            "counts_sha256": _sha(counts_raw),
            "accepted_weight_sha256": _sha(accepted),
            "constraint_digest": bounds.grouped.digest,
            "weight_anchor": expected.params["weight_anchor"],
            "cap_enforcement": expected.params["cap_enforcement"],
            "fixed_zero_rows": int(np.count_nonzero(bounds.incoming == 0)),
        }
        payload = canonical_json(
            diagnostics_payload(result, target_registry=registry, build=anchors)
        )
        output = KernelResult(
            weights=weights,
            artifacts={"diagnostics": payload},
            receipt={
                **anchors,
                "diagnostics_sha256": _sha(payload),
                "scope": calibration_receipt_scope(expected.params),
                "release_eligible": False,
                "source_admission": "required_from_country_runner",
            },
        )
        # Validation follows diagnostics and return-object materialization.
        fresh = decode_numeric_survey_bounds(bounds_raw)
        check_numeric_survey_weights(fresh, weights.values)
        _require(weights.values.tobytes() == accepted, "FINAL_WEIGHT_BYTES")
        _require(
            tuple(result.frame.table("household").household_id)
            == fresh.grouped.household_ids,
            "FINAL_HOUSEHOLD_IDS",
        )
        return output
