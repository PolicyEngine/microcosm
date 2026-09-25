"""Portable ordered calibration problems and solutions, without pickles.

The authoritative numerical input is CSR plus aligned target and entity axes.
Country adapters supply declarative row metadata and identity bindings. Python
measure closures are not serialized: ``to_target_set`` can reconstruct exact
compiled contributions, checking the consuming Frame's ordered entity IDs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from io import BytesIO
from numbers import Integral
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from scipy import sparse

from microcosm.frame import Frame, WeightKind, Weights
from microcosm.graph import ArtifactType
from microcosm.graph.canonical import canonical_json

from .matrix import CalibrationProblem, SkippedTarget
from .target import Target, TargetSet

PROBLEM_TYPE = ArtifactType("microcosm.calibrate.ordered-problem", 1)
SOLUTION_TYPE = ArtifactType("microcosm.calibrate.ordered-solution", 1)
RESULT_TYPE = ArtifactType("microcosm.calibrate.calibration-result", 1)


def _ids(values: Sequence[int | str]) -> tuple[int | str, ...]:
    if isinstance(values, str | bytes):
        raise ValueError("Entity axis must be a sequence of IDs.")
    result = []
    for value in values:
        if isinstance(value, Integral) and not isinstance(value, bool):
            result.append(int(value))
        elif isinstance(value, str) and value:
            result.append(value)
        else:
            raise ValueError("Entity IDs must be integers or nonempty strings.")
    if not result or len(set(result)) != len(result):
        raise ValueError("Entity IDs must be nonempty and unique.")
    return tuple(result)


def _sha(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("A problem binding must be a lowercase SHA-256 digest.")
    return value


def _freeze(array: np.ndarray) -> np.ndarray:
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _pack(metadata: Mapping, arrays: Mapping[str, np.ndarray]) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        members = {
            **arrays,
            "metadata": np.frombuffer(canonical_json(metadata), dtype=np.uint8),
        }
        for name, array in sorted(members.items()):
            stream = BytesIO()
            np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
            info = ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, stream.getvalue())
    return output.getvalue()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate calibration metadata key.")
        result[key] = value
    return result


def _unpack(payload: bytes, *, schema: str, members: set[str]) -> tuple[dict, dict]:
    if type(payload) is not bytes:
        raise TypeError("Calibration artifacts require immutable bytes.")
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)) or set(
                archive.files
            ) != members | {"metadata"}:
                raise ValueError("Calibration artifact members differ from its schema.")
            raw = archive["metadata"]
            if raw.dtype != np.uint8 or raw.ndim != 1:
                raise ValueError("Calibration artifact metadata must be UTF-8 bytes.")
            metadata = json.loads(raw.tobytes(), object_pairs_hook=_unique)
            if (
                canonical_json(metadata) != raw.tobytes()
                or metadata.get("schema") != schema
            ):
                raise ValueError(
                    "Calibration artifact schema/canonical metadata differs."
                )
            arrays = {key: _freeze(archive[key]) for key in members}
        return metadata, arrays
    except (OSError, KeyError, TypeError, UnicodeError) as error:
        raise ValueError("Malformed calibration artifact.") from error


def _target(target: Target) -> dict[str, object]:
    return {
        "name": target.name,
        "entity": target.entity,
        "value": target.value,
        "period": target.period,
        "source": target.source,
        "tolerance": target.tolerance,
        "metadata": dict(target.metadata),
        "measure": target.measure if isinstance(target.measure, str) else None,
        "filter": target.filter if isinstance(target.filter, str) else None,
        "compiled_measure": callable(target.measure),
        "compiled_filter": callable(target.filter),
    }


def _read_target(raw: Mapping) -> Target:
    expected = {
        "name",
        "entity",
        "value",
        "period",
        "source",
        "tolerance",
        "metadata",
        "measure",
        "filter",
        "compiled_measure",
        "compiled_filter",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise ValueError("Malformed calibration target descriptor.")
    if (
        type(raw["compiled_measure"]) is not bool
        or type(raw["compiled_filter"]) is not bool
    ):
        raise ValueError("Malformed compiled target flags.")
    if raw["compiled_measure"] != (raw["measure"] is None):
        raise ValueError("Target measure descriptor is inconsistent.")
    if raw["compiled_filter"] and raw["filter"] is not None:
        raise ValueError("Target filter descriptor is inconsistent.")
    return Target(
        **{
            key: raw[key]
            for key in (
                "name",
                "entity",
                "value",
                "period",
                "source",
                "tolerance",
                "metadata",
            )
        },
        measure=raw["measure"] or "__compiled_contribution__",
        filter=raw["filter"],
    )


@dataclass(frozen=True)
class _CompiledRow:
    entity: str
    entity_ids: tuple[int | str, ...]
    values: sparse.csr_array

    def __call__(self, frame: Frame) -> np.ndarray:
        column = frame.schema.entity_id_column(self.entity)
        if tuple(frame.table(self.entity)[column]) != self.entity_ids:
            raise ValueError(
                "Compiled contribution requires its exact ordered entity axis."
            )
        # Keep target definitions sparse; compilation materializes one row
        # at a time rather than retaining a targets × households dense array.
        return self.values.toarray().ravel()


@dataclass(frozen=True)
class OrderedProblem:
    """An identified numerical problem with explicit ordered entity IDs."""

    problem: CalibrationProblem
    entity_ids: tuple[int | str, ...]
    target_metadata: tuple[Mapping, ...]
    bindings: Mapping
    sha256: str

    def to_target_set(self) -> TargetSet:
        """Use the compiled rows in the public solver without original closures.

        Contributions already include original filters and entity aggregation.
        No raw measure is recomputed; consuming a different row axis refuses.
        """
        return TargetSet(
            [
                replace(
                    target,
                    entity=self.problem.weight_entity,
                    filter=None,
                    measure=_CompiledRow(
                        self.problem.weight_entity,
                        self.entity_ids,
                        self.problem.matrix[index : index + 1],
                    ),
                )
                for index, target in enumerate(self.problem.targets)
            ]
        )


def encode_problem(
    problem: CalibrationProblem,
    *,
    entity_ids: Sequence[int | str],
    target_metadata: Sequence[Mapping] | None = None,
    bindings: Mapping | None = None,
) -> bytes:
    """Encode numerical rows, axes, skipped facts and declarative bindings."""
    ids = _ids(entity_ids)
    if not isinstance(problem, CalibrationProblem):
        raise TypeError("encode_problem requires CalibrationProblem.")
    if len(ids) != problem.n_weights:
        raise ValueError("Problem entity axis differs from the weight columns.")
    matrix = sparse.csr_array(problem.matrix, dtype=np.float64, copy=True)
    rows = (
        tuple({} for _ in problem.targets)
        if target_metadata is None
        else tuple(target_metadata)
    )
    if len(rows) != problem.n_targets or any(
        not isinstance(row, Mapping) for row in rows
    ):
        raise ValueError("Target metadata must match the ordered target axis.")
    metadata = {
        "schema": PROBLEM_TYPE.name + ".v1",
        "shape": list(matrix.shape),
        "entity_ids": ids,
        "weight_entity": problem.weight_entity,
        "weight_kind": problem.initial_weights.kind.value,
        "names": problem.names,
        "targets": [_target(target) for target in problem.targets],
        "skipped": [
            {"target": _target(item.target), "reason": item.reason}
            for item in problem.skipped
        ],
        "target_metadata": rows,
        "bindings": {} if bindings is None else bindings,
    }
    payload = _pack(
        metadata,
        {
            "data": matrix.data.astype("<f8"),
            "indices": matrix.indices.astype("<i8"),
            "indptr": matrix.indptr.astype("<i8"),
            "targets": np.asarray(problem.target_vector, dtype="<f8"),
            "weights": np.asarray(problem.initial_weights.values, dtype="<f8"),
        },
    )
    decode_problem(payload)
    return payload


def decode_problem(payload: bytes) -> OrderedProblem:
    """Decode and validate a portable problem, preserving CSR row order."""
    metadata, arrays = _unpack(
        payload,
        schema=PROBLEM_TYPE.name + ".v1",
        members={"data", "indices", "indptr", "targets", "weights"},
    )
    if set(metadata) != {
        "schema",
        "shape",
        "entity_ids",
        "weight_entity",
        "weight_kind",
        "names",
        "targets",
        "skipped",
        "target_metadata",
        "bindings",
    }:
        raise ValueError("Problem metadata fields differ.")
    ids = _ids(metadata["entity_ids"])
    shape = metadata["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(n) is not int or n < 0 for n in shape)
    ):
        raise ValueError("Malformed problem shape.")
    if shape[1] != len(ids):
        raise ValueError("Problem entity axis differs from the weight columns.")
    for key, values in arrays.items():
        dtype = np.dtype("<i8" if key in {"indices", "indptr"} else "<f8")
        if values.ndim != 1 or values.dtype != dtype or not np.isfinite(values).all():
            raise ValueError(
                "Problem arrays must be finite, typed one-dimensional arrays."
            )
    data, indices, indptr = (arrays[key] for key in ("data", "indices", "indptr"))
    if (
        len(indptr) != shape[0] + 1
        or indptr[0] != 0
        or indptr[-1] != len(data)
        or len(indices) != len(data)
        or (np.diff(indptr) < 0).any()
        or (indices < 0).any()
        or (indices >= shape[1]).any()
    ):
        raise ValueError("Malformed CSR index arrays.")
    matrix = sparse.csr_array((data, indices, indptr), shape=tuple(shape))
    targets = tuple(_read_target(raw) for raw in metadata["targets"])
    if len({target.key for target in targets}) != len(targets) or tuple(
        metadata["names"]
    ) != tuple(target.row_name for target in targets):
        raise ValueError("Problem target IDs/names must form a unique ordered axis.")
    if arrays["targets"].tolist() != [target.value for target in targets]:
        raise ValueError("Problem target values disagree with their descriptors.")
    skipped = []
    for item in metadata["skipped"]:
        if (
            set(item) != {"target", "reason"}
            or not isinstance(item["reason"], str)
            or not item["reason"]
        ):
            raise ValueError("Malformed skipped-target evidence.")
        skipped.append(SkippedTarget(_read_target(item["target"]), item["reason"]))
    rows = metadata["target_metadata"]
    if (
        len(rows) != len(targets)
        or any(not isinstance(row, dict) for row in rows)
        or not isinstance(metadata["bindings"], dict)
    ):
        raise ValueError("Problem row metadata/bindings differ from their contract.")
    problem = CalibrationProblem(
        matrix,
        arrays["targets"],
        tuple(metadata["names"]),
        Weights(arrays["weights"], WeightKind(metadata["weight_kind"])),
        metadata["weight_entity"],
        targets,
        tuple(skipped),
    )
    return OrderedProblem(
        problem,
        ids,
        tuple(rows),
        metadata["bindings"],
        hashlib.sha256(payload).hexdigest(),
    )


@dataclass(frozen=True)
class OrderedSolution:
    """Aligned calibrated weights bound to an exact problem artifact."""

    weights: np.ndarray
    entity_ids: tuple[int | str, ...]
    problem_sha256: str
    diagnostics: Mapping
    sha256: str


def encode_solution(
    weights: Sequence[float] | np.ndarray,
    *,
    entity_ids: Sequence[int | str],
    problem_sha256: str,
    diagnostics: Mapping | None = None,
) -> bytes:
    """Encode a solution once; later population installation need not solve."""
    ids = _ids(entity_ids)
    payload = _pack(
        {
            "schema": SOLUTION_TYPE.name + ".v1",
            "entity_ids": ids,
            "problem_sha256": _sha(problem_sha256),
            "diagnostics": {} if diagnostics is None else diagnostics,
        },
        {"weights": np.asarray(weights, dtype="<f8")},
    )
    decode_solution(payload)
    return payload


def decode_solution(
    payload: bytes,
    *,
    problem_sha256: str | None = None,
    entity_ids: Sequence[int | str] | None = None,
) -> OrderedSolution:
    """Validate solution axes, finite nonnegative weights and optional binding."""
    metadata, arrays = _unpack(
        payload, schema=SOLUTION_TYPE.name + ".v1", members={"weights"}
    )
    if set(metadata) != {
        "schema",
        "entity_ids",
        "problem_sha256",
        "diagnostics",
    } or not isinstance(metadata["diagnostics"], dict):
        raise ValueError("Solution metadata fields differ.")
    ids = _ids(metadata["entity_ids"])
    digest = _sha(metadata["problem_sha256"])
    weights = arrays["weights"]
    if (
        weights.dtype != np.dtype("<f8")
        or weights.shape != (len(ids),)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
    ):
        raise ValueError(
            "Solution weights must be finite, nonnegative and match the axis."
        )
    if problem_sha256 is not None and digest != _sha(problem_sha256):
        raise ValueError("Solution belongs to a different problem.")
    if entity_ids is not None and ids != _ids(entity_ids):
        raise ValueError("Solution belongs to a different ordered entity axis.")
    return OrderedSolution(
        weights,
        ids,
        digest,
        metadata["diagnostics"],
        hashlib.sha256(payload).hexdigest(),
    )


def encode_calibration_result(
    result, *, entity_ids: Sequence[int | str], problem_sha256: str
) -> bytes:
    """Persist the complete numerical state needed by dense/search continuation.

    A result is bound to a separately encoded ordered problem. Optimizer state
    is not claimed: continuation rebuilds completed results, never mid-epoch
    training. Grouped solves require a separate constraints-aware protocol.
    """
    ids = _ids(entity_ids)
    if len(ids) != len(result.weights):
        raise ValueError("Calibration result differs from the ordered entity axis.")
    if (
        "grouped_upper_bounds" in result.options
        or "grouped_preserve_zeros" in result.options
    ):
        raise ValueError(
            "Grouped calibration results require their original constraints."
        )
    probabilities = result.gate_open_probabilities
    return _pack(
        {
            "schema": RESULT_TYPE.name + ".v1",
            "entity_ids": ids,
            "problem_sha256": _sha(problem_sha256),
            "l0_lambda": float(result.l0_lambda),
            "n_nonzero": int(result.n_nonzero),
            "target_loss_cap": float(result.target_loss_cap),
            "closing_loss": float(result.final_loss),
            "options": dict(result.options),
            "has_probabilities": probabilities is not None,
        },
        {
            "weights": np.asarray(result.weights, dtype="<f8"),
            "initial_weights": np.asarray(result.initial_weights, dtype="<f8"),
            "loss_trajectory": np.asarray(result.loss_trajectory, dtype="<f8"),
            "target_loss_weights": np.asarray(result.target_loss_weights, dtype="<f8"),
            "target_loss_scales": np.asarray(result.target_loss_scales, dtype="<f8"),
            "probabilities": np.asarray(
                [] if probabilities is None else probabilities, dtype="<f8"
            ),
        },
    )


def decode_calibration_result(payload: bytes, *, frame: Frame, problem: OrderedProblem):
    """Rebuild a completed result with the public, non-optimizing replay seam."""
    from .solve import rebuild_calibration_result

    metadata, arrays = _unpack(
        payload,
        schema=RESULT_TYPE.name + ".v1",
        members={
            "weights",
            "initial_weights",
            "loss_trajectory",
            "target_loss_weights",
            "target_loss_scales",
            "probabilities",
        },
    )
    if set(metadata) != {
        "schema",
        "entity_ids",
        "problem_sha256",
        "l0_lambda",
        "n_nonzero",
        "target_loss_cap",
        "closing_loss",
        "options",
        "has_probabilities",
    }:
        raise ValueError("Calibration result metadata fields differ.")
    if _sha(metadata["problem_sha256"]) != problem.sha256:
        raise ValueError("Calibration result belongs to a different problem.")
    if _ids(metadata["entity_ids"]) != problem.entity_ids:
        raise ValueError("Calibration result differs from its ordered problem axis.")
    entity = problem.problem.weight_entity
    if (
        tuple(frame.table(entity)[frame.schema.entity_id_column(entity)])
        != problem.entity_ids
    ):
        raise ValueError("Calibration result differs from the Frame entity axis.")
    initial = frame.weights_for(entity)
    if (
        initial.kind != problem.problem.initial_weights.kind
        or not np.array_equal(initial.values, problem.problem.initial_weights.values)
        or not np.array_equal(arrays["initial_weights"], initial.values)
    ):
        raise ValueError(
            "Calibration result initial weights differ from the bound pool."
        )
    for values in arrays.values():
        if (
            values.dtype != np.dtype("<f8")
            or values.ndim != 1
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "Calibration result arrays must be finite float64 vectors."
            )
    flag = metadata["has_probabilities"]
    if (
        type(flag) is not bool
        or (not flag and len(arrays["probabilities"]))
        or (flag and len(arrays["probabilities"]) != len(problem.entity_ids))
    ):
        raise ValueError(
            "Calibration result probabilities do not match their declaration."
        )
    if problem.problem.skipped:
        raise ValueError("Cannot rebuild a calibration result with skipped targets.")
    return rebuild_calibration_result(
        frame,
        problem.to_target_set(),
        weight_entity=entity,
        weights=arrays["weights"],
        loss_trajectory=arrays["loss_trajectory"],
        l0_lambda=metadata["l0_lambda"],
        n_nonzero=metadata["n_nonzero"],
        target_loss_weights=arrays["target_loss_weights"],
        target_loss_scales=arrays["target_loss_scales"],
        target_loss_cap=metadata["target_loss_cap"],
        options=metadata["options"],
        gate_open_probabilities=arrays["probabilities"] if flag else None,
        closing_loss=metadata["closing_loss"],
    )
