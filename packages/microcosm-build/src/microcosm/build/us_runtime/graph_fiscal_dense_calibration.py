"""Numerical dense fiscal calibration; complete-parent admission stays external.

The caller supplies original DESIGN anchors and origin-derived absolute caps.
This module transports and checks those numbers but cannot authenticate their
source or the complete Population. The host must compare original anchors and
admit the complete parent and successor on cold execution, replay and return.

One existing grouped Adam/free-mass solve preserves every initial zero exactly.
There is no pruning, restart, repeated calibration or row-cap relaxation. CSR
rows enter the existing public Target callable API one at a time: O(N) temporary
row memory, plus sparse storage, rather than a dense target-by-household table.
The existing solver chooses sparse/dense Torch storage for the resulting matrix;
its capped scaled-MAPE objective uses float32 and accepted weights use float64.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from microcosm.calibrate import (
    TargetSet,
    TargetSnapshotObserver,
    calibrate,
    diagnostics_payload,
    group_bounds,
)
from microcosm.calibrate import kernels as calibration_kernels
from microcosm.calibrate import target as target_module
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
from microcosm.graph.canonical import canonical_json, normative
from microcosm.graph.keys import opaque_artifact_key

from . import graph_fiscal_measurement as measurement
from .demographic_calibration_graph import DIAGNOSTICS_TYPE

BOUNDS_TYPE = ArtifactType("microcosm.us.fiscal_calibration_numeric_bounds", 1)
BOUNDS_PROTOCOL = "microcosm.us.fiscal-calibration-numeric-bounds.v1"
ORIGIN_TYPE = ArtifactType("microcosm.us.fiscal_calibration_origin_diagnostics", 1)
MAX_BYTES = 64 * 1024**2
_FIELDS = {
    "protocol",
    "population",
    "budget_sha256",
    "household_ids",
    "original_design",
    "incoming",
    "group_indices",
    "group_upper",
    "row_upper",
}


def _require(condition, reason):
    if not condition:
        raise ValueError("FISCAL_CALIBRATION_" + reason)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _digest(value):
    return (
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _same_array(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@dataclass(frozen=True)
class FiscalCalibrationBounds:
    """Decoded numerical references only; never an issued sampling budget."""

    document: dict
    household_ids: np.ndarray
    original_design: Weights
    incoming: Weights
    grouped: group_bounds.GroupedUpperBounds
    row_upper: np.ndarray


def decode_fiscal_calibration_bounds(payload: bytes) -> FiscalCalibrationBounds:
    """Read bounded, canonical typed-ID and exact float64 reference arrays."""
    _require(type(payload) is bytes and 0 < len(payload) <= MAX_BYTES, "BOUNDS_PAYLOAD")
    try:
        raw = json.loads(payload)
    except (ValueError, UnicodeError):
        raise ValueError("FISCAL_CALIBRATION_BOUNDS_JSON") from None
    _require(type(raw) is dict and set(raw) == _FIELDS, "BOUNDS_FIELDS")
    _require(canonical_json(raw) == payload, "BOUNDS_CANONICAL")
    _require(raw["protocol"] == BOUNDS_PROTOCOL, "BOUNDS_PROTOCOL")
    _require(_digest(raw["budget_sha256"]), "BUDGET_DIGEST")
    _require(
        type(raw["population"]) is str and 0 < len(raw["population"]) <= 256,
        "POPULATION",
    )
    ids = measurement._read_array(raw["household_ids"], {"<i8", "<u8"})
    _require(len(ids) > 0 and len(set(ids.tolist())) == len(ids), "HOUSEHOLD_IDS")
    vectors = {
        name: measurement._read_array(raw[name], {"<f8"})
        for name in ("original_design", "incoming", "group_upper", "row_upper")
    }
    _require(
        all(np.isfinite(v).all() and np.all(v >= 0) for v in vectors.values()),
        "REFERENCE_VALUES",
    )
    _require(
        all(
            len(vectors[n]) == len(ids)
            for n in ("original_design", "incoming", "row_upper")
        ),
        "REFERENCE_ALIGNMENT",
    )
    _require(np.any(vectors["incoming"] > 0), "EMPTY_POSITIVE_SUPPORT")
    groups = group_bounds.GroupedUpperBounds(
        ids,
        measurement._read_array(raw["group_indices"], {"<i8"}),
        vectors["group_upper"],
    )
    result = FiscalCalibrationBounds(
        raw,
        ids,
        Weights(vectors["original_design"], WeightKind.DESIGN),
        Weights(vectors["incoming"], WeightKind.IMPORTANCE),
        groups,
        vectors["row_upper"],
    )
    check_fiscal_calibration_weights(result, result.incoming.values)
    return result


def encode_fiscal_calibration_bounds(
    *,
    population: str,
    household_ids: np.ndarray,
    original_design_weights: Weights,
    incoming_weights: Weights,
    grouped_upper_bounds: group_bounds.GroupedUpperBounds,
    row_upper: np.ndarray,
    budget_sha256: str,
) -> bytes:
    """Freeze explicitly supplied original references without issuing authority.

    ``row_upper`` is already the original source-reference cap, not an inferred
    multiple of incoming or cloned DESIGN values. Its provenance and arithmetic
    belong to the original budget owner. Full uint64 IDs and row order survive.
    """
    _require(
        type(original_design_weights) is Weights
        and original_design_weights.kind is WeightKind.DESIGN,
        "DESIGN_KIND",
    )
    _require(
        type(incoming_weights) is Weights
        and incoming_weights.kind is WeightKind.IMPORTANCE,
        "IMPORTANCE_KIND",
    )
    _require(
        type(grouped_upper_bounds) is group_bounds.GroupedUpperBounds, "GROUP_TYPE"
    )
    _require(
        type(household_ids) is np.ndarray
        and household_ids.dtype.str in {"<i8", "<u8"}
        and household_ids.ndim == 1,
        "HOUSEHOLD_ID_DTYPE",
    )
    _require(
        tuple(household_ids.tolist()) == grouped_upper_bounds.household_ids,
        "GROUP_HOUSEHOLD_ALIGNMENT",
    )
    _require(
        type(row_upper) is np.ndarray
        and row_upper.dtype == np.dtype("float64")
        and row_upper.ndim == 1,
        "ROW_UPPER_DTYPE",
    )
    _require(
        household_ids.size * 96 + grouped_upper_bounds.group_count * 16 < MAX_BYTES,
        "BOUNDS_SIZE",
    )
    payload = canonical_json(
        {
            "protocol": BOUNDS_PROTOCOL,
            "population": population,
            "budget_sha256": budget_sha256,
            "household_ids": measurement._array(household_ids),
            "original_design": measurement._array(original_design_weights.values),
            "incoming": measurement._array(incoming_weights.values),
            "group_indices": measurement._array(grouped_upper_bounds.group_indices),
            "group_upper": measurement._array(grouped_upper_bounds.absolute_bounds),
            "row_upper": measurement._array(row_upper),
        }
    )
    decode_fiscal_calibration_bounds(payload)
    return payload


def check_fiscal_calibration_weights(
    bounds: FiscalCalibrationBounds, values: np.ndarray
):
    """Refuse every group/row overage and any changed initial zero coordinate."""
    _require(type(bounds) is FiscalCalibrationBounds, "BOUNDS_TYPE")
    _require(
        type(values) is np.ndarray
        and values.dtype == np.dtype("float64")
        and values.shape == bounds.incoming.values.shape,
        "WEIGHT_ARRAY",
    )
    bounds.grouped.check(values, positive=False)
    _require(np.all(values <= bounds.row_upper), "ROW_REFERENCE_CAP")
    zeros = bounds.incoming.values == 0
    _require(
        values[zeros].tobytes() == bounds.incoming.values[zeros].tobytes()
        and np.all(values[~zeros] > 0),
        "FIXED_ZERO_SUPPORT",
    )


def verify_fiscal_calibration_bounds(
    payload,
    *,
    household_ids,
    incoming_weights,
    original_design_weights,
) -> FiscalCalibrationBounds:
    """Compare numeric bytes to a host's live references; not source admission.

    The host must obtain original_design_weights from retained Population
    anchors and separately authenticate the original group and row caps.
    """
    bounds = decode_fiscal_calibration_bounds(payload)
    _require(_same_array(bounds.household_ids, household_ids), "HOUSEHOLD_ALIGNMENT")
    _require(
        type(incoming_weights) is Weights
        and incoming_weights.kind is WeightKind.IMPORTANCE,
        "IMPORTANCE_KIND",
    )
    _require(
        _same_array(bounds.incoming.values, incoming_weights.values), "INCOMING_ANCHOR"
    )
    _require(
        type(original_design_weights) is Weights
        and original_design_weights.kind is WeightKind.DESIGN
        and _same_array(bounds.original_design.values, original_design_weights.values),
        "DESIGN_ANCHOR",
    )
    return bounds


def _measurement_node(raw):
    params = raw["params"]
    result = measurement.fiscal_measurement_node(
        measurement._registry(json.loads(params["registry"])),
        population=raw["population"],
        node_id=raw["id"],
        input_columns={s["entity"]: tuple(s["columns"]) for s in raw["inputs"]},
        contract_targets=json.loads(params["contract_targets"]),
        advertised_scopes=measurement._scopes(json.loads(params["advertised_scopes"])),
        geography_vintage=params["geography_vintage"],
        period=params["period"],
        model_outputs=tuple(params["model_outputs"]),
    )
    _require(
        canonical_json(normative(result)) == canonical_json(raw), "MEASUREMENT_NODE"
    )
    return result


def fiscal_dense_calibration_node(
    *,
    measurement_node: Node,
    bounds_node: str,
    epochs: int = 256,
    learning_rate: float = 0.02,
    node_id: str = "us.fiscal_dense_calibration",
) -> Node:
    """Declare one weight-only, fixed-zero grouped solve on the measured parent."""
    _require(type(epochs) is int and 1 <= epochs <= 10000, "EPOCHS")
    _require(
        type(learning_rate) in (int, float)
        and np.isfinite(learning_rate)
        and 0 < learning_rate <= 1,
        "LEARNING_RATE",
    )
    raw = canonical_json(normative(measurement_node))
    _require(len(raw) <= 2 * measurement.MAX_DECLARATION_BYTES, "DECLARATION_SIZE")
    verified = _measurement_node(json.loads(raw))
    return Node(
        node_id,
        FiscalDenseCalibrationKernel.ref,
        base=verified.population,
        inputs=verified.inputs,
        structural=StructuralDelta.REWEIGHT,
        weights=WeightTransition("household", "calibrated", mass="free"),
        mass="free",
        params={
            "measurement_node": raw.decode(),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "method": "adam",
            "seed": 0,
            "grouped_preserve_zeros": True,
            "l0_lambda": 0.0,
            "l1_lambda": 0.0,
            "l2_lambda": 0.0,
            "l2_anchor": "initial",
            "weight_anchor": "explicit_original_sampling_references",
            "cap_enforcement": "numeric_group_projection_and_final_row_refusal",
        },
        artifact_inputs=(
            ArtifactInput(
                "measurement", verified.id, "measurement", measurement.MEASUREMENT_TYPE
            ),
            ArtifactInput("bounds", bounds_node, "numeric_bounds", BOUNDS_TYPE),
        ),
        artifact_outputs=(
            ArtifactOutput("diagnostics", DIAGNOSTICS_TYPE),
            ArtifactOutput("origin_diagnostics", ORIGIN_TYPE),
        ),
    )


def _artifact(context, name, type_, output):
    value = context.artifacts[name]
    _require(
        value.type == type_
        and value.key == opaque_artifact_key(value.producer_key, output),
        "ARTIFACT_EDGE",
    )
    return value.payload


class FiscalDenseCalibrationKernel(KernelBase):
    """Use the existing solver and independently check the compiled measurement.

    ``target_snapshots`` is optional host-owned instrumentation, held only on
    this kernel instance. Observer configuration is absent from the graph and
    cache identity. Sinks finish before the existing final measurement, ID and
    weight checks; their exceptions propagate. A required cache hit does not
    execute this kernel or emit optimizer snapshots.
    """

    ref = "us.fiscal_dense_calibration@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.NONE,
        structural=StructuralDelta.REWEIGHT,
        consumes_se=False,
        dependencies=("numpy", "pandas", "scipy", "torch"),
    )

    def __init__(self, *, target_snapshots: TargetSnapshotObserver | None = None):
        if target_snapshots is not None and not isinstance(
            target_snapshots, TargetSnapshotObserver
        ):
            raise TypeError("target_snapshots must be a TargetSnapshotObserver or None")
        self._target_snapshots = target_snapshots

    def implementation_hash(self):
        return _sha(
            canonical_json(
                {
                    "adapter": source_hash(
                        sys.modules[__name__],
                        measurement,
                        group_bounds,
                        target_module,
                        measurement.TargetSpec,
                        measurement.CalibrationHierarchy,
                        calibration_kernels.diagnostics_module.assemble_target_loss_attribution,
                        dependencies=self.capabilities.dependencies,
                    ),
                    "solver": calibration_kernels.CALIBRATE_ADAM.implementation_hash(),
                }
            )
        )

    def run(self, context):
        params = context.params
        _require(
            tuple(a.name for a in context.node.artifact_inputs)
            == ("measurement", "bounds"),
            "ALIASES",
        )
        measured_node = _measurement_node(json.loads(params["measurement_node"]))
        expected = fiscal_dense_calibration_node(
            measurement_node=measured_node,
            bounds_node=context.node.artifact_inputs[1].producer,
            epochs=params["epochs"],
            learning_rate=params["learning_rate"],
            node_id=context.node.id,
        )
        _require(
            normative(context.node) == normative(expected)
            and dict(params) == dict(expected.params),
            "DECLARATION",
        )
        _require(
            context.weights["household"].kind is WeightKind.IMPORTANCE,
            "IMPORTANCE_REQUIRED",
        )
        _require(not context.sources, "UNDECLARED_SOURCE")
        raw_measurement = _artifact(
            context, "measurement", measurement.MEASUREMENT_TYPE, "measurement"
        )
        measured_context = replace(
            context, node=measured_node, params=measured_node.params, artifacts={}
        )
        measured = measurement.verify_fiscal_measurement(
            raw_measurement,
            context=measured_context,
            expected_sha256=_sha(raw_measurement),
        )
        raw_bounds = _artifact(context, "bounds", BOUNDS_TYPE, "numeric_bounds")
        bounds = decode_fiscal_calibration_bounds(raw_bounds)
        _require(
            bounds.document["population"] == context.node.base, "BOUNDS_POPULATION"
        )
        _require(
            _same_array(bounds.household_ids, measured.household_ids),
            "HOUSEHOLD_ALIGNMENT",
        )
        _require(
            _same_array(bounds.incoming.values, context.weights["household"].values),
            "INCOMING_ANCHOR",
        )
        table = pd.DataFrame({"household_id": measured.household_ids.copy()})
        work = calibration_kernels._frame_from_context(
            replace(
                context,
                tables={"household": table},
                weights={"household": bounds.incoming},
            ),
            "household",
        )
        targets = []
        for i, spec in enumerate(measured.registry):
            # The public callable API materializes only this one sparse row.
            def row(_frame, position=i):
                return measured.matrix[position : position + 1].toarray().ravel()

            targets.append(
                replace(spec.to_target(), entity="household", measure=row, filter=None)
            )
        result = calibrate(
            work,
            TargetSet(targets),
            weight_entity="household",
            method="adam",
            seed=0,
            epochs=params["epochs"],
            learning_rate=params["learning_rate"],
            mass="free",
            max_weight_ratio=None,
            target_records=None,
            l0_lambda=0.0,
            l1_lambda=0.0,
            l2_lambda=0.0,
            l2_anchor="initial",
            grouped_upper_bounds=bounds.grouped,
            grouped_preserve_zeros=True,
            target_snapshots=self._target_snapshots,
        )
        _require(
            not result.skipped
            and result.problem.names
            == tuple(s.to_target().row_name for s in measured.registry),
            "TARGET_ALIGNMENT",
        )
        _require(
            _same_array(result.problem.target_vector, measured.target_values)
            and result.problem.matrix.shape == measured.matrix.shape
            and all(
                _same_array(
                    getattr(result.problem.matrix, k), getattr(measured.matrix, k)
                )
                for k in ("indptr", "indices", "data")
            ),
            "MATRIX_IDENTITY",
        )
        _require(
            _same_array(result.initial_weights, bounds.incoming.values),
            "SOLVER_INITIAL",
        )
        weights = result.frame.weights_for("household")
        _require(weights.kind is WeightKind.CALIBRATED, "CALIBRATED_KIND")
        check_fiscal_calibration_weights(bounds, weights.values)
        accepted = weights.values.tobytes()
        _require(
            np.isfinite(result.loss_trajectory).all()
            and np.isfinite(result.final_loss),
            "NONFINITE_LOSS",
        )
        initial_estimates = measured.matrix @ bounds.incoming.values
        final_estimates = measured.matrix @ weights.values
        _require(
            np.isfinite(initial_estimates).all()
            and np.isfinite(final_estimates).all()
            and np.isfinite(final_estimates - measured.target_values).all(),
            "NONFINITE_RESIDUAL",
        )
        _require(
            _same_array(
                np.asarray([d.initial_estimate for d in result.diagnostics]),
                initial_estimates,
            )
            and _same_array(
                np.asarray([d.final_estimate for d in result.diagnostics]),
                final_estimates,
            ),
            "DIAGNOSTIC_ESTIMATES",
        )
        build = {
            "measurement_sha256": _sha(raw_measurement),
            "numeric_bounds_sha256": _sha(raw_bounds),
            "budget_sha256": bounds.document["budget_sha256"],
            "original_design_sha256": _sha(bounds.original_design.values.tobytes()),
            "incoming_sha256": _sha(bounds.incoming.values.tobytes()),
            "accepted_weight_sha256": _sha(accepted),
            "constraint_digest": bounds.grouped.digest,
            "matrix_matches_measurement": True,
            "numeric_measurement_entity": "household",
            "original_measurement_entities": {
                s.name: s.entity for s in measured.registry
            },
            "weight_anchor": params["weight_anchor"],
            "cap_enforcement": params["cap_enforcement"],
        }
        diagnostics = canonical_json(
            diagnostics_payload(
                result,
                target_registry=measured.registry,
                build=build,
            )
        )
        origin = canonical_json(
            {
                "protocol": "microcosm.us.fiscal-calibration-origin-diagnostics.v1",
                **build,
                "household_ids": measurement._array(bounds.household_ids),
                "accepted_weights": measurement._array(weights.values),
                "group_totals": measurement._array(
                    bounds.grouped.totals(weights.values)
                ),
                "group_upper": measurement._array(bounds.grouped.absolute_bounds),
                "group_diagnostics": bounds.grouped.diagnostics(
                    weights.values,
                    result.options["grouped_upper_bounds"][
                        "last_corrected_group_count"
                    ],
                ),
                "fixed_zero_rows": int(np.count_nonzero(bounds.incoming.values == 0)),
                "active_rows": int(np.count_nonzero(weights.values > 0)),
                "binding_row_caps": int(
                    np.count_nonzero(weights.values == bounds.row_upper)
                ),
                "all_row_caps_satisfied": True,
                "release_eligible": False,
                "source_admission": "required_from_complete_parent_owner",
            }
        )
        _require(
            len(diagnostics) <= MAX_BYTES and len(origin) <= MAX_BYTES,
            "DIAGNOSTIC_SIZE",
        )
        output = KernelResult(
            weights=weights,
            artifacts={"diagnostics": diagnostics, "origin_diagnostics": origin},
            receipt={
                **build,
                "diagnostics_sha256": _sha(diagnostics),
                "origin_diagnostics_sha256": _sha(origin),
                "release_eligible": False,
                "source_admission": "required_from_complete_parent_owner",
                "scope": "one_numeric_dense_fiscal_solve_no_ancestry_admission",
            },
        )
        check_fiscal_calibration_weights(
            decode_fiscal_calibration_bounds(raw_bounds), weights.values
        )
        _require(weights.values.tobytes() == accepted, "FINAL_WEIGHT_BYTES")
        _require(
            _same_array(
                work.table("household").household_id.to_numpy(), bounds.household_ids
            )
            and _same_array(
                result.frame.table("household").household_id.to_numpy(),
                bounds.household_ids,
            ),
            "FINAL_HOUSEHOLD_ALIGNMENT",
        )
        measurement.verify_fiscal_measurement(
            raw_measurement,
            context=measured_context,
            expected_sha256=_sha(raw_measurement),
        )
        return output
