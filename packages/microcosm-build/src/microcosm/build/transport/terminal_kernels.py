"""Terminal kernels: calibration diagnostics, export and the package receipt.

These follow the UK full build's terminal pattern
(``uk_runtime/graph_terminal.py``) without sharing its code, so no UK node
re-keys:

- ``diagnostics.calibration@1`` rebuilds a completed calibration from the
  ordered ``problem``/``solution``/``result`` artifacts (no solve), checks
  that the population it reads carries exactly that solution's weights, and
  emits the ``microcosm-diagnostics`` schema-8 model as canonical JSON. The
  target hierarchy comes from the target surface's registry.
- ``export.prepare@1`` describes the exact entity tables an export must
  write (columns, dtypes, row count and a content hash per table, plus the
  period) and refuses when any declared gate report does not permit the
  artifact.
- Writing the H5 is an outer service (:func:`materialize_export`), so a cache
  hit can never silently skip a file write. It writes through
  :class:`~microcosm.frame.adapters.axiom.AxiomEntityTableDataset`, whose
  HDF5 writer records object times: two writes of one descriptor can differ
  in bytes while their tables are identical, so the readback and the package
  re-key per written file.
- ``export.readback@1`` is a GATE that reads the written file as a declared
  source and compares every table, the table set and the period with the
  descriptor; an unreadable file fails the node (charter F4) with a message
  that names the source but not its path, and a readable file that differs
  fails its outcome.
- ``transport.package@1`` binds every artifact the node declares, the gate
  outcomes and the readback into one receipt. It never authorizes a release.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

import microcosm.build.artifact_files as artifact_files_module
import microcosm.calibrate._target_loss_attribution as loss_attribution_module
import microcosm.calibrate.artifacts as calibrate_artifacts_module
import microcosm.calibrate.diagnostics as calibrate_diagnostics_module
import microcosm.calibrate.matrix as matrix_module
import microcosm.calibrate.registry as registry_module
import microcosm.calibrate.solve as solve_module
import microcosm.calibrate.target as target_module
import microcosm.diagnostics.schema as diagnostics_schema_module
import microcosm.frame.adapters.axiom as axiom_adapter_module
import microcosm.frame.bundle as frame_bundle_module
import microcosm.frame.materialize as materialize_module
import microcosm.frame.schema as frame_schema_module
import microcosm.frame.weights as frame_weights_module
from microcosm.build.artifact_files import file_artifact
from microcosm.calibrate import diagnostics_payload
from microcosm.calibrate.artifacts import (
    PROBLEM_TYPE,
    RESULT_TYPE,
    SOLUTION_TYPE,
    OrderedProblem,
    decode_calibration_result,
    decode_problem,
    decode_solution,
)
from microcosm.diagnostics import CalibrationDiagnosticsV8
from microcosm.frame import EntitySchema, Frame, engine_tables
from microcosm.frame.adapters.axiom import AxiomEntityTableDataset
from microcosm.graph import (
    SOURCE_CODECS,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    KernelRole,
    Numeric,
    SeedSource,
    SourceCodecRegistry,
    source_hash,
)
from microcosm.graph.canonical import canonical_json

from . import artifact_types, gate_kernels, graph_inputs, target_kernels
from .artifact_types import (
    EXPORT_DESCRIPTOR_TYPE,
    EXPORT_READBACK_TYPE,
    TARGET_SURFACE_TYPE,
)
from .gate_kernels import require_terminal_gate_reports
from .graph_inputs import (
    canonical_document_param,
    context_frame,
    entities_param,
    require_outputs,
    require_params,
    sha256_text,
    single_source,
    string_param,
)
from .target_kernels import decode_target_surface

__all__ = [
    "DIAGNOSTICS_CALIBRATION",
    "EXPORT_DTYPES",
    "EXPORT_PREPARE",
    "EXPORT_READBACK",
    "EXPORT_SOURCE_CODEC",
    "TRANSPORT_PACKAGE",
    "ExportUnreadableError",
    "DiagnosticsCalibrationKernel",
    "ExportPrepareKernel",
    "ExportReadbackKernel",
    "TransportPackageKernel",
    "describe_export",
    "materialize_export",
    "register_terminal_kernels",
    "validate_export",
]

#: The codec a written export is declared under as a graph source.
EXPORT_SOURCE_CODEC = "axiom-entity-table-h5-v1"
#: Column dtypes the entity-table writer and reader round-trip exactly
#: (``AxiomEntityTableDataset.save`` and its reader). Nullable ``Int64`` does
#: not write, and nullable ``boolean`` comes back as ``bool`` without missing
#: values and as ``object`` with them, so an export carrying either is refused
#: before anything is written.
EXPORT_DTYPES = frozenset({"bool", "int32", "int64", "float32", "float64", "string"})

_EXPORT_KIND = "transport_export"
_READBACK_KIND = "transport_export_readback"
_PACKAGE_KIND = "transport_package"
_EXPORT_FORMAT = "axiom-entity-table-h5"
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def _typed_artifact(
    context: KernelContext, ref: str, alias: str, expected: ArtifactType
) -> bytes:
    value = context.artifacts.get(alias)
    if value is None or value.type != expected:
        raise ValueError(
            f"{ref} reads a {expected.name} v{expected.schema_version} artifact "
            f"under the alias {alias!r}."
        )
    return value.payload


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------


def _axis_frame(problem: OrderedProblem) -> Frame:
    """A one-entity frame carrying the problem's ordered axis and start weights.

    The same pattern ``calibrate.adam@1`` uses for its context: the
    calibrated entity stands in as the person entity and a one-row dummy
    group supplies the linkage ``EntitySchema`` requires.
    """

    entity = problem.problem.weight_entity
    table = pd.DataFrame({f"{entity}_id": list(problem.entity_ids)})
    group = "__diagnostics_axis_group"
    table[f"{entity}_{group}_id"] = np.zeros(len(table), dtype=np.int64)
    return Frame(
        {entity: table, group: pd.DataFrame({f"{group}_id": np.zeros(1, np.int64)})},
        EntitySchema(person_entity=entity, group_entities=(group,)),
        {entity: problem.problem.initial_weights},
    )


def _installed_weights(context: KernelContext, entity: str, ref: str) -> pd.Series:
    if entity not in context.tables or entity not in context.weights:
        raise ValueError(
            f"{ref} needs the {entity!r} table and weights: declare one "
            f"data-column slice on {entity!r}."
        )
    ids = context.tables[entity][f"{entity}_id"].tolist()
    return pd.Series(context.weights[entity].values, index=ids)


class DiagnosticsCalibrationKernel(KernelBase):
    """``diagnostics.calibration@1``: one calibration's schema-8 diagnostics."""

    ref = "diagnostics.calibration@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy", "pandas", "scipy", "pydantic"),
    )
    _required = frozenset({"weight_entity"})
    _optional = frozenset({"build"})

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            graph_inputs,
            artifact_types,
            target_kernels,
            calibrate_artifacts_module,
            calibrate_diagnostics_module,
            diagnostics_schema_module,
            loss_attribution_module,
            solve_module,
            matrix_module,
            target_module,
            registry_module,
            frame_bundle_module,
            frame_schema_module,
            frame_weights_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        ref = self.ref
        require_params(context, ref, required=self._required, optional=self._optional)
        require_outputs(context, ref)
        entity = string_param(context, ref, "weight_entity")
        build = (
            dict(canonical_document_param(context, ref, "build"))
            if "build" in context.params
            else {}
        )
        problem_payload = _typed_artifact(context, ref, "problem", PROBLEM_TYPE)
        ordered = decode_problem(problem_payload)
        if ordered.problem.weight_entity != entity:
            raise ValueError(
                f"{ref} problem calibrates {ordered.problem.weight_entity!r}, "
                f"not {entity!r}."
            )
        solution_payload = _typed_artifact(context, ref, "solution", SOLUTION_TYPE)
        solution = decode_solution(
            solution_payload,
            problem_sha256=ordered.sha256,
            entity_ids=ordered.entity_ids,
        )
        result_payload = _typed_artifact(context, ref, "result", RESULT_TYPE)
        result = decode_calibration_result(
            result_payload, frame=_axis_frame(ordered), problem=ordered
        )
        if result.weights.tobytes() != solution.weights.tobytes():
            raise ValueError(f"{ref} result and solution weights differ.")
        installed = _installed_weights(context, entity, ref)
        expected = pd.Series(solution.weights, index=list(ordered.entity_ids))
        if (
            set(installed.index) != set(expected.index)
            or installed.reindex(expected.index).to_numpy().tobytes()
            != expected.to_numpy().tobytes()
        ):
            raise ValueError(
                f"{ref} reads a population whose {entity!r} weights are not this "
                "solution's; the node must live downstream of its calibration."
            )
        surface_payload = _typed_artifact(context, ref, "surface", TARGET_SURFACE_TYPE)
        surface = decode_target_surface(surface_payload)
        bound = ordered.bindings.get("surface_sha256")
        if bound is not None and bound != surface.sha256:
            raise ValueError(f"{ref} problem was compiled from a different surface.")
        # Report every row against the target as compiled: the decoded problem
        # keeps each target's own entity, measure and filter, where the
        # rebuilt result carries only the compiled contribution rows. The
        # matrix, target vector and starting weights are the same object's.
        result = replace(result, problem=ordered.problem)
        payload = diagnostics_payload(
            result,
            target_registry=surface.registry,
            build={
                **build,
                "problem_sha256": ordered.sha256,
                "solution_sha256": solution.sha256,
                "result_sha256": sha256_text(result_payload),
                "surface_sha256": surface.sha256,
            },
        )
        model = CalibrationDiagnosticsV8.model_validate(payload)
        diagnostics = canonical_json(
            model.model_dump(mode="python", exclude_defaults=True)
        )
        return KernelResult(
            artifacts={"diagnostics": diagnostics},
            receipt={
                "schema_version": model.schema_version,
                "diagnostics_sha256": sha256_text(diagnostics),
                "n_targets": len(model.targets),
                "fraction_within_10pct": model.fraction_within_10pct,
                # The diagnostics' ratio is against the problem's starting
                # weights; the executor's ``realized_max_weight_ratio`` on the
                # calibrate node is against the CREATE design weights.
                "ratio_to_starting_weights_max": model.realized_max_weight_ratio,
            },
        )


# ---------------------------------------------------------------------------
# Export description, materialization and readback
# ---------------------------------------------------------------------------


def _table_description(table: pd.DataFrame) -> dict[str, object]:
    descriptor = {
        "rows": len(table),
        "columns": [str(column) for column in table.columns],
        "dtypes": [str(dtype) for dtype in table.dtypes],
    }
    digest = hashlib.sha256(canonical_json(descriptor))
    digest.update(
        np.ascontiguousarray(
            pd.util.hash_pandas_object(table, index=False).to_numpy()
        ).tobytes()
    )
    return {**descriptor, "content_sha256": digest.hexdigest()}


def _content(
    tables: Mapping[str, pd.DataFrame],
    *,
    entities: tuple[str, ...],
    time_period: int,
) -> dict[str, object]:
    content = {
        "format": _EXPORT_FORMAT,
        "entities": list(entities),
        "tables": {
            entity: _table_description(tables[entity].reset_index(drop=True))
            for entity in entities
        },
        "time_period": time_period,
        "hash_environment": {
            "pandas": metadata.version("pandas"),
            "numpy": metadata.version("numpy"),
        },
    }
    return {**content, "content_sha256": sha256_text(canonical_json(content))}


def _export_tables(
    frame: Frame, *, entities: tuple[str, ...], weight_entity: str
) -> dict[str, pd.DataFrame]:
    tables = engine_tables(frame, weighted_entities=(weight_entity,))
    exported = {}
    for entity in entities:
        table = tables[entity].reset_index(drop=True)
        if table.empty:
            raise ValueError(f"Export refuses an empty {entity!r} table.")
        unsupported = {
            str(column): str(dtype)
            for column, dtype in table.dtypes.items()
            if str(dtype) not in EXPORT_DTYPES
        }
        if unsupported:
            raise ValueError(
                f"Export refuses {entity!r} column dtype(s) the entity-table "
                f"writer cannot round-trip exactly: {unsupported}; allowed "
                f"{sorted(EXPORT_DTYPES)}."
            )
        exported[entity] = table
    return exported


def describe_export(
    frame: Frame,
    *,
    entities: tuple[str, ...],
    weight_entity: str,
    time_period: int,
    bindings: Mapping[str, object],
) -> dict[str, object]:
    """Describe the exact tables an export writes, without writing them."""

    tables = _export_tables(frame, entities=entities, weight_entity=weight_entity)
    return {
        "schema_version": 1,
        "kind": _EXPORT_KIND,
        **_content(tables, entities=entities, time_period=time_period),
        "weight_entity": weight_entity,
        "weight_kind": frame.weights_for(weight_entity).kind.value,
        "bindings": dict(bindings),
        "graph_only": ["strata", "metadata", "mass_log", "weight_kind"],
    }


def _check_descriptor(descriptor: Mapping[str, object]) -> tuple[str, ...]:
    if (
        descriptor.get("schema_version") != 1
        or descriptor.get("kind") != _EXPORT_KIND
        or descriptor.get("format") != _EXPORT_FORMAT
    ):
        raise ValueError("Unsupported export descriptor.")
    content = {
        key: descriptor[key]
        for key in ("format", "entities", "tables", "time_period", "hash_environment")
    }
    if sha256_text(canonical_json(content)) != descriptor.get("content_sha256"):
        raise ValueError("Export descriptor content identity is inconsistent.")
    return tuple(descriptor["entities"])


def materialize_export(
    frame: Frame, descriptor: Mapping[str, object], path: str | Path
) -> dict[str, object]:
    """Write exactly the described tables; refuse a population that differs.

    The outer step after ``export.prepare@1``: it runs on every build, cache
    hit or not, and writes only what the graph described. ``frame`` may carry
    more columns than the export (the full population version); the
    descriptor's columns are selected in its own order.
    """

    entities = _check_descriptor(descriptor)
    weight_entity = str(descriptor["weight_entity"])
    full = engine_tables(frame, weighted_entities=(weight_entity,))
    tables = {
        entity: full[entity].loc[:, list(descriptor["tables"][entity]["columns"])]
        for entity in entities
    }
    time_period = int(descriptor["time_period"])
    current = _content(tables, entities=entities, time_period=time_period)
    if current["content_sha256"] != descriptor["content_sha256"]:
        raise ValueError("Export population differs from its graph descriptor.")
    destination = Path(path)
    if destination.suffix != ".h5":
        raise ValueError("An entity-table export must have an .h5 filename.")
    AxiomEntityTableDataset(
        tables={entity: tables[entity].reset_index(drop=True) for entity in entities},
        time_period=time_period,
    ).save(destination)
    return file_artifact(destination)


class ExportUnreadableError(ValueError):
    """The written file cannot be bound or read as an entity-table H5.

    Its message never names the path: a gate's exception message is recorded
    in the node's cached receipt (charter F4), and a host path there would
    outlive a source that moved.
    """


def _bind_file(path: str | Path) -> dict[str, object]:
    try:
        return file_artifact(path)
    except Exception as error:  # noqa: BLE001 - re-raised without the path
        raise ExportUnreadableError(
            f"Exported file cannot be bound as a regular file ({type(error).__name__})."
        ) from None


def _read_export(path: str | Path) -> tuple[dict[str, object], AxiomEntityTableDataset]:
    dataset = _bind_file(path)
    try:
        written = AxiomEntityTableDataset(file_path=path)
    except Exception as error:  # noqa: BLE001 - re-raised without the path
        raise ExportUnreadableError(
            f"Exported file cannot be read as an entity-table H5 "
            f"({type(error).__name__})."
        ) from None
    return dataset, written


def validate_export(
    path: str | Path, descriptor: Mapping[str, object]
) -> dict[str, object]:
    """Compare the written file's tables and period with the descriptor."""

    entities = _check_descriptor(descriptor)
    dataset, written = _read_export(path)
    failures = []
    extra = sorted(set(written.tables) - set(entities))
    missing = sorted(set(entities) - set(written.tables))
    if extra:
        failures.append(f"Exported file carries undescribed table(s) {extra}.")
    if missing:
        failures.append(f"Exported file lacks described table(s) {missing}.")
    present = tuple(entity for entity in entities if entity in written.tables)
    actual = _content(
        written.tables, entities=present, time_period=int(written.time_period)
    )
    for entity in present:
        if actual["tables"][entity] != descriptor["tables"][entity]:
            failures.append(f"Exported {entity!r} table differs from its descriptor.")
    if actual["time_period"] != descriptor["time_period"]:
        failures.append("Exported time period differs from its descriptor.")
    if actual["hash_environment"] != descriptor["hash_environment"]:
        failures.append("Readback hash environment differs from the descriptor's.")
    if _bind_file(path) != dataset:
        raise ValueError("Exported file changed during readback.")
    return {
        "schema_version": 1,
        "kind": _READBACK_KIND,
        "passed": not failures,
        "failures": failures,
        "dataset": {key: dataset[key] for key in ("sha256", "size_bytes")},
        "content_sha256": actual["content_sha256"],
        "expected_content_sha256": descriptor["content_sha256"],
        "bindings": dict(descriptor["bindings"]),
    }


def _load_export_bytes(path: Path, *, store: object | None = None) -> bytes:
    """The export codec: an HDF5 file's verified bytes."""

    del store
    payload = Path(path).read_bytes()
    if not payload.startswith(_HDF5_SIGNATURE):
        raise ValueError("The declared export source is not an HDF5 file.")
    return payload


class ExportPrepareKernel(KernelBase):
    """``export.prepare@1``: describe the export; refuse when gates block."""

    ref = "export.prepare@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy", "pandas"),
    )
    _required = frozenset({"entities", "weight_entity", "time_period"})
    _optional = frozenset({"bindings"})

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            graph_inputs,
            artifact_types,
            gate_kernels,
            materialize_module,
            axiom_adapter_module,
            frame_bundle_module,
            frame_schema_module,
            frame_weights_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        ref = self.ref
        require_params(context, ref, required=self._required, optional=self._optional)
        require_outputs(context, ref)
        entities = entities_param(context, ref, "entities")
        weight_entity = string_param(context, ref, "weight_entity")
        time_period = context.params.get("time_period")
        if isinstance(time_period, bool) or not isinstance(time_period, int):
            raise TypeError(f"{ref} parameter 'time_period' must be an integer.")
        gate_reports = require_terminal_gate_reports(context, ref)
        blocked = sorted(
            f"{alias} ({report.phase}: {report.outcome})"
            for alias, report in gate_reports.items()
            if not report.artifact_permitted
        )
        if blocked:
            raise ValueError(f"{ref} refused by blocking gate report(s): {blocked}.")
        bindings = (
            dict(canonical_document_param(context, ref, "bindings"))
            if "bindings" in context.params
            else {}
        )
        if "gates" in bindings:
            raise ValueError(
                f"{ref} records the gate reports itself; parameter 'bindings' "
                "must not carry a 'gates' key."
            )
        bindings["gates"] = {
            alias: {
                "phase": report.phase,
                "outcome": report.outcome,
                "gates_sha256": report.gates_sha256,
                "release_candidate": report.release_candidate,
                "synthetic_smoke": report.synthetic_smoke,
                "artifact_key": context.artifacts[alias].key,
            }
            for alias, report in gate_reports.items()
        }
        frame = context_frame(
            context, ref, entities=entities, weight_entity=weight_entity
        )
        descriptor = describe_export(
            frame,
            entities=entities,
            weight_entity=weight_entity,
            time_period=time_period,
            bindings=bindings,
        )
        payload = canonical_json(descriptor)
        return KernelResult(
            artifacts={"export_descriptor": payload},
            receipt={
                "content_sha256": descriptor["content_sha256"],
                "rows": {
                    entity: descriptor["tables"][entity]["rows"] for entity in entities
                },
            },
        )


class ExportReadbackKernel(KernelBase):
    """``export.readback@1``: the written file against its descriptor (GATE)."""

    ref = "export.readback@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        role=KernelRole.GATE,
        dependencies=("numpy", "pandas", "tables"),
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            graph_inputs,
            artifact_types,
            artifact_files_module,
            materialize_module,
            axiom_adapter_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        ref = self.ref
        require_params(context, ref, required=frozenset())
        require_outputs(context, ref)
        descriptor = json.loads(
            _typed_artifact(context, ref, "export_descriptor", EXPORT_DESCRIPTOR_TYPE)
        )
        source = single_source(context, ref)
        try:
            report = validate_export(context.sources[source], descriptor)
        except ExportUnreadableError as error:
            raise ExportUnreadableError(
                f"{ref} cannot read declared source {source!r}: {error}"
            ) from None
        payload = canonical_json(report)
        return KernelResult(
            artifacts={"export_readback": payload},
            receipt={
                "outcome": "pass" if report["passed"] else "fail",
                "evidence": {
                    "failures": list(report["failures"]),
                    "dataset_sha256": report["dataset"]["sha256"],
                    "content_sha256": report["content_sha256"],
                },
            },
        )


class TransportPackageKernel(KernelBase):
    """``transport.package@1``: one receipt binding a build's terminal evidence."""

    ref = "transport.package@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            graph_inputs,
            artifact_types,
            gate_kernels,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        ref = self.ref
        require_params(
            context, ref, required=frozenset(), optional=frozenset({"bindings"})
        )
        require_outputs(context, ref)
        readback = json.loads(
            _typed_artifact(context, ref, "export_readback", EXPORT_READBACK_TYPE)
        )
        if (
            readback.get("kind") != _READBACK_KIND
            or readback.get("schema_version") != 1
        ):
            raise ValueError(f"{ref} requires a typed export readback report.")
        if readback["passed"] is not True:
            raise ValueError(f"{ref} refused: the exported file failed readback.")
        gates = {}
        reports = require_terminal_gate_reports(context, ref)
        for alias, report in reports.items():
            if not report.artifact_permitted:
                raise ValueError(
                    f"{ref} refused: gate report {alias!r} ({report.phase}) "
                    f"does not permit the artifact."
                )
            gates[alias] = {
                "phase": report.phase,
                "outcome": report.outcome,
                "gates_sha256": report.gates_sha256,
                "release_candidate": report.release_candidate,
                "synthetic_smoke": report.synthetic_smoke,
            }
        # The reports must be the ones the export was prepared under, which the
        # descriptor recorded and the readback carried through.
        prepared = {
            entry["artifact_key"]
            for entry in readback["bindings"].get("gates", {}).values()
        }
        if {context.artifacts[alias].key for alias in reports} != prepared:
            raise ValueError(
                f"{ref} refused: its gate reports are not the ones the export "
                "was prepared under."
            )
        bindings = (
            dict(canonical_document_param(context, ref, "bindings"))
            if "bindings" in context.params
            else {}
        )
        payload = canonical_json(
            {
                "schema_version": 1,
                "kind": _PACKAGE_KIND,
                "readback": {
                    key: readback[key]
                    for key in ("passed", "dataset", "content_sha256")
                },
                "artifacts": {
                    alias: {
                        "type": value.type.name,
                        "schema_version": value.type.schema_version,
                        "key": value.key,
                        "producer_key": value.producer_key,
                        "sha256": sha256_text(value.payload),
                        "size_bytes": len(value.payload),
                    }
                    for alias, value in sorted(context.artifacts.items())
                },
                "gates": gates,
                "bindings": bindings,
                "release_authorized": False,
            }
        )
        return KernelResult(
            artifacts={"receipt": payload},
            receipt={"receipt_sha256": sha256_text(payload), "gates": gates},
        )


DIAGNOSTICS_CALIBRATION = DiagnosticsCalibrationKernel()
EXPORT_PREPARE = ExportPrepareKernel()
EXPORT_READBACK = ExportReadbackKernel()
TRANSPORT_PACKAGE = TransportPackageKernel()


def register_terminal_kernels(
    registry: KernelRegistry, *, codecs: SourceCodecRegistry = SOURCE_CODECS
) -> None:
    """Register the four kernels and the export codec; idempotent."""

    codecs.register_bytes(EXPORT_SOURCE_CODEC, _load_export_bytes)
    for kernel in (
        DIAGNOSTICS_CALIBRATION,
        EXPORT_PREPARE,
        EXPORT_READBACK,
        TRANSPORT_PACKAGE,
    ):
        registry.register(kernel)
