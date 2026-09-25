"""Full-build terminal artifacts, with streamed H5 materialization and readback.

Graph kernels describe and validate the exact numerical export. Atomic disk
materialization is an outer service, so a cache hit cannot silently skip a
required file write. The resulting H5 is a content-bound source to the graph
continuation; large H5 payloads are never duplicated in a byte artifact.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, replace
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.frame import Frame, engine_tables
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    KernelRole,
    Node,
    Numeric,
    SeedSource,
    SourceRef,
    source_hash,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import SOURCE_CODECS

from ..artifact_files import file_artifact
from . import geography_ladder, national_frame
from .atomic_area_support import UK_NATIVE_ALIAS_COLUMNS
from .geography_ladder import uk_geography_ladder_gate
from .graph_population import context_frame, population_columns, population_slices
from .national_frame import (
    _read_uk_national_tables,
    _write_uk_single_year_tables,
    uk_household_weight_kind,
    uk_time_period,
    validate_uk_national_frame,
)
from .rowwise_dataset import (
    ARTIFACT_CLONE_INDEX_COLUMN,
    ladder_clone_index_column,
    load_uk_rowwise_dataset,
)

EXPORT_DESCRIPTOR_TYPE = ArtifactType("microcosm.uk.full-export-descriptor", 1)
EXPORT_READBACK_TYPE = ArtifactType("microcosm.uk.full-export-readback", 1)
PACKAGE_INVENTORY_TYPE = ArtifactType("microcosm.full-package-inventory", 1)
EXPORT_SOURCE_CODEC = "uk-single-year-h5@1"


def _tables(frame: Frame) -> dict[str, pd.DataFrame]:
    tables = engine_tables(frame, weighted_entities=("household",))
    renamed = {}
    for entity in ("person", "benunit", "household"):
        column = ladder_clone_index_column(entity)
        if column not in tables[entity]:
            raise ValueError(
                f"Full-build export lacks {entity} geographic replicate lineage {column!r}."
            )
        renamed[entity] = tables[entity].rename(
            columns={column: ARTIFACT_CLONE_INDEX_COLUMN}
        )
    # Nation-native aliases of derived layers are NA outside their own nation;
    # the single-year artifact carries the ten ladder columns plus the
    # identity-keyed assignment columns, never the aliases.
    aliases = [c for c in UK_NATIVE_ALIAS_COLUMNS if c in renamed["household"]]
    if aliases:
        renamed["household"] = renamed["household"].drop(columns=aliases)
    return renamed


def _table_description(table: pd.DataFrame) -> dict[str, object]:
    descriptor = {
        "rows": len(table),
        "columns": list(table.columns),
        "dtypes": [str(dtype) for dtype in table.dtypes],
        "index_dtype": str(table.index.dtype),
    }
    digest = hashlib.sha256(canonical_json(descriptor))
    digest.update(
        np.ascontiguousarray(
            pd.util.hash_pandas_object(table, index=True).to_numpy()
        ).tobytes()
    )
    return {**descriptor, "content_sha256": digest.hexdigest()}


def _content_descriptor(
    tables, *, time_period, weight_kind, mass_log
) -> dict[str, object]:
    descriptor = {
        "tables": {
            entity: _table_description(tables[entity])
            for entity in ("person", "benunit", "household")
        },
        "time_period": str(time_period),
        "weight_kind": weight_kind.value,
        "mass_log": [asdict(record) for record in mass_log],
        "hash_environment": {
            "pandas": metadata.version("pandas"),
            "numpy": metadata.version("numpy"),
        },
    }
    return {
        **descriptor,
        "content_sha256": hashlib.sha256(canonical_json(descriptor)).hexdigest(),
    }


def describe_uk_export(
    frame: Frame, *, bindings: Mapping[str, object]
) -> dict[str, object]:
    """Validate and describe the maintained H5 layout without serializing it."""
    validate_uk_national_frame(frame)
    tables = _tables(frame)
    gate = uk_geography_ladder_gate(
        tables["household"], frame.weights_for("household").values
    )
    if not gate.passed:
        raise ValueError(
            "UK export geography integrity failed: " + "; ".join(gate.failures)
        )
    return {
        "schema_version": 1,
        "kind": "uk_full_build_export",
        **_content_descriptor(
            tables,
            time_period=uk_time_period(frame),
            weight_kind=uk_household_weight_kind(frame),
            mass_log=frame.mass_log,
        ),
        "bindings": dict(bindings),
        "geography_integrity": {"passed": gate.passed, "failures": list(gate.failures)},
        "graph_only_metadata": [
            "strata",
            "metadata_other_than_time_period",
            "structural_ancestry",
        ],
    }


def _check_descriptor(descriptor: Mapping[str, object]) -> None:
    if (
        descriptor.get("schema_version") != 1
        or descriptor.get("kind") != "uk_full_build_export"
    ):
        raise ValueError("Unsupported UK export descriptor.")
    fields = ("tables", "time_period", "weight_kind", "mass_log", "hash_environment")
    payload = {key: descriptor[key] for key in fields}
    if hashlib.sha256(canonical_json(payload)).hexdigest() != descriptor.get(
        "content_sha256"
    ):
        raise ValueError("UK export descriptor content identity is inconsistent.")


def materialize_uk_export(
    frame: Frame, descriptor: Mapping[str, object], path: str | Path
) -> dict[str, object]:
    """Perform only the declared serialization, recreating files on cache hits."""
    _check_descriptor(descriptor)
    tables = _tables(frame)
    current = _content_descriptor(
        tables,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        mass_log=frame.mass_log,
    )
    if current["content_sha256"] != descriptor["content_sha256"]:
        raise ValueError("UK export population differs from its graph descriptor.")
    destination = Path(path)
    if destination.suffix != ".h5":
        raise ValueError("UK full-build dataset must have an .h5 filename.")
    _write_uk_single_year_tables(
        **tables,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        mass_log=frame.mass_log,
        path=destination,
    )
    return file_artifact(destination)


def validate_uk_export(
    path: str | Path, descriptor: Mapping[str, object]
) -> dict[str, object]:
    """Compare actual written table values/dtypes/weights/periods to the request."""
    _check_descriptor(descriptor)
    dataset = file_artifact(path)
    payload, _, _ = _read_uk_national_tables(path)
    actual = _content_descriptor(
        payload,
        time_period=payload["time_period"],
        weight_kind=payload["household_weight_kind"],
        mass_log=payload["mass_log"],
    )
    failures = [
        f"Exported {entity} table differs from its graph descriptor."
        for entity in ("person", "benunit", "household")
        if actual["tables"][entity] != descriptor["tables"][entity]
    ]
    for key in ("time_period", "weight_kind", "mass_log", "hash_environment"):
        if actual[key] != descriptor[key]:
            failures.append(f"Exported {key} differs from its graph descriptor.")
    if file_artifact(path) != dataset:
        raise ValueError("UK exported file changed during graph readback validation.")
    return {
        "schema_version": 1,
        "kind": "uk_full_build_export_readback",
        "passed": not failures,
        "failures": failures,
        "dataset": dataset,
        "content_sha256": actual["content_sha256"],
        "expected_content_sha256": descriptor["content_sha256"],
        "bindings": dict(descriptor["bindings"]),
    }


class UKExportPrepareKernel(KernelBase):
    ref = "uk.full-export.prepare@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, numeric=Numeric.BITWISE, seed_source=SeedSource.NONE
    )

    def implementation_hash(self) -> str:
        return source_hash(sys.modules[__name__], national_frame, geography_ladder)

    def run(self, context: KernelContext) -> KernelResult:
        for value in context.artifacts.values():
            if value.type == FULL_GATE_REPORT_TYPE:
                _, enforcement = decode_full_gate_report(value.payload)
                if not enforcement["artifact_permitted"]:
                    raise ValueError(
                        "UK export preparation refused by a structural full-build gate."
                    )
        bindings = json.loads(str(context.params["bindings"]))
        bindings["artifacts"] = {
            name: value.key for name, value in context.artifacts.items()
        }
        descriptor = describe_uk_export(context_frame(context), bindings=bindings)
        return KernelResult(artifacts={"export_descriptor": canonical_json(descriptor)})


class UKExportReadbackKernel(KernelBase):
    ref = "uk.full-export.readback@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        role=KernelRole.GATE,
    )

    def implementation_hash(self) -> str:
        return source_hash(sys.modules[__name__], national_frame)

    def run(self, context: KernelContext) -> KernelResult:
        descriptor = json.loads(context.artifacts["export_descriptor"].payload)
        report = validate_uk_export(context.sources["exported_dataset"], descriptor)
        return KernelResult(
            artifacts={"export_readback": canonical_json(report)},
            receipt={"outcome": "pass" if report["passed"] else "fail"},
        )


class UKPackageInventoryKernel(KernelBase):
    ref = "uk.full-export.package@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, numeric=Numeric.BITWISE, seed_source=SeedSource.NONE
    )

    def run(self, context: KernelContext) -> KernelResult:
        report = json.loads(context.artifacts["export_readback"].payload)
        if (
            report.get("kind") != "uk_full_build_export_readback"
            or report.get("schema_version") != 1
        ):
            raise ValueError(
                "Package inventory requires a typed export readback report."
            )
        if not report["passed"]:
            raise ValueError("UK package refused because exported H5 readback failed.")
        manifest = json.loads(str(context.params["manifest_binding"]))
        files = {}
        for alias, filename in json.loads(
            str(context.params.get("evidence_files", "{}"))
        ).items():
            record = file_artifact(context.sources["exported_evidence_" + alias])
            payload = context.artifacts[alias].payload
            if (
                record["filename"] != filename
                or record["sha256"] != hashlib.sha256(payload).hexdigest()
                or record["size_bytes"] != len(payload)
            ):
                raise ValueError(
                    f"Materialized UK evidence {alias!r} differs from its graph artifact."
                )
            files[alias] = {
                **record,
                "graph_artifact_key": context.artifacts[alias].key,
            }
        return KernelResult(
            artifacts={
                "package_inventory": canonical_json(
                    {
                        "schema_version": 1,
                        "kind": "uk_full_build_package",
                        "readback_passed": report["passed"],
                        "dataset": report["dataset"],
                        "content_sha256": report["content_sha256"],
                        "build_bindings": report["bindings"],
                        "numerical_graph": manifest,
                        "artifacts": {
                            name: value.key for name, value in context.artifacts.items()
                        },
                        "evidence_files": files,
                        "release_authorized": False,
                    }
                )
            }
        )


def add_uk_export_preparation(
    graph: Graph,
    *,
    population: str,
    bindings: Mapping[str, object],
    artifact_inputs: tuple[ArtifactInput, ...] = (),
) -> Graph:
    node = Node(
        id="uk.full.export.prepare",
        kernel=UKExportPrepareKernel.ref,
        population=population,
        inputs=population_slices(population_columns(graph, population)),
        params={"bindings": canonical_json(dict(bindings)).decode()},
        artifact_inputs=artifact_inputs,
        artifact_outputs=(ArtifactOutput("export_descriptor", EXPORT_DESCRIPTOR_TYPE),),
        description="Validate and describe exact H5 values, schema, weights and geographic lineage.",
    )
    return replace(graph, nodes=(*graph.nodes, node))


def add_uk_export_continuation(
    graph: Graph,
    *,
    population: str,
    manifest_binding: Mapping[str, object],
    artifact_inputs: tuple[ArtifactInput, ...] = (),
    evidence_files: Mapping[str, str] | None = None,
) -> Graph:
    """Continue the same graph after its declared H5 materialization boundary."""
    evidence_files = {} if evidence_files is None else dict(evidence_files)
    aliases = {item.name for item in artifact_inputs}
    for alias, filename in evidence_files.items():
        if alias not in aliases or not alias.replace("_", "").isalnum():
            raise ValueError(
                "Materialized evidence must name a declared simple artifact alias."
            )
        if not filename or Path(filename).name != filename:
            raise ValueError(
                "Materialized evidence filenames must be simple path components."
            )
    evidence_sources = tuple("exported_evidence_" + alias for alias in evidence_files)
    readback = Node(
        id="uk.full.export.readback",
        kernel=UKExportReadbackKernel.ref,
        population=population,
        sources=("exported_dataset",),
        artifact_inputs=(
            ArtifactInput(
                "export_descriptor",
                "uk.full.export.prepare",
                "export_descriptor",
                EXPORT_DESCRIPTOR_TYPE,
            ),
        ),
        artifact_outputs=(ArtifactOutput("export_readback", EXPORT_READBACK_TYPE),),
        description="Read the written H5 and compare exact exported tables to the graph descriptor.",
    )
    package = Node(
        id="uk.full.package",
        sources=evidence_sources,
        kernel=UKPackageInventoryKernel.ref,
        population=population,
        artifact_inputs=(
            ArtifactInput(
                "export_readback", readback.id, "export_readback", EXPORT_READBACK_TYPE
            ),
            *artifact_inputs,
        ),
        artifact_outputs=(ArtifactOutput("package_inventory", PACKAGE_INVENTORY_TYPE),),
        params={
            "manifest_binding": canonical_json(dict(manifest_binding)).decode(),
            "evidence_files": canonical_json(evidence_files).decode(),
        },
        description="Bind output bytes, numerical graph identity, scope/sizing and terminal evidence.",
    )
    return replace(
        graph,
        sources=(
            *graph.sources,
            *(SourceRef(name, "raw-bytes-v1") for name in evidence_sources),
            SourceRef(
                "exported_dataset",
                EXPORT_SOURCE_CODEC,
                "Materialized UK H5; streamed identity and graph-owned readback.",
            ),
        ),
        nodes=(*graph.nodes, readback, package),
    )


def _load_export_frame(path: Path) -> Frame:
    return load_uk_rowwise_dataset(path)[0]


def register_uk_terminal_kernels(registry: KernelRegistry) -> None:
    SOURCE_CODECS.register(EXPORT_SOURCE_CODEC, _load_export_frame)
    for kernel in (
        UKExportPrepareKernel(),
        UKExportReadbackKernel(),
        UKPackageInventoryKernel(),
    ):
        registry.register(kernel)


FULL_GATE_REPORT_TYPE = ArtifactType("microcosm.uk.full-gate-report", 1)
FULL_DIAGNOSTICS_TYPE = ArtifactType("microcosm.uk.full-calibration-diagnostics", 1)
FULL_DIAGNOSTICS_CSV_TYPE = ArtifactType("microcosm.uk.full-target-diagnostics-csv", 1)
FULL_SUPPORT_CSV_TYPE = ArtifactType("microcosm.uk.full-area-support-csv", 1)
FULL_HOLDOUT_TYPE = ArtifactType("microcosm.uk.full-rotated-holdout", 1)


def _full_gate_enforcement(document: Mapping, report):
    """Carry earlier phase policy forward without reevaluating its population."""
    from ..country_spec import load_country_spec
    from ..gate_battery import gate_phase_report_from_payload
    from .full_gates import classify_full_gate_outcomes, uk_full_gate_manifest
    from .graph_evidence import uk_spine_gate_manifest

    upstream = document.get("upstream_phase_reports", {})
    allowed = {"spine_assembled": "assembled", "spine_transferred": "transferred"}
    if report.phase == "terminal":
        allowed["full_preflight"] = "preflight"
    if not isinstance(upstream, Mapping) or set(upstream) - set(allowed):
        raise ValueError("Unexpected upstream phase in the full gate artifact.")
    reports = []
    for name, payload in upstream.items():
        gates = (
            uk_full_gate_manifest(document["selection_receipt"])
            if name == "full_preflight"
            else uk_spine_gate_manifest(load_country_spec("uk"))
        )
        previous = gate_phase_report_from_payload(payload, gates=gates)
        if previous.phase != allowed[name]:
            raise ValueError("Upstream full gate report has a different phase.")
        reports.append(previous)
    reports.append(report)
    classifications = [
        classify_full_gate_outcomes(
            item,
            sample_fraction=document["sample_fraction"],
            release_candidate=document["release_candidate"],
        )
        for item in reports
    ]
    enforcement = dict(classifications[-1])
    for key in (
        "structural_failures",
        "enforced_blocking",
        "exportable_blocking",
        "unenforced_release_failures",
        "diagnostic_failures",
    ):
        enforcement[key] = list(
            dict.fromkeys(value for item in classifications for value in item[key])
        )
    enforcement["artifact_permitted"] = all(
        item["artifact_permitted"] for item in classifications
    )
    enforcement["release_blocking_gates_passed"] = all(
        item["release_blocking_gates_passed"] for item in classifications
    )
    return enforcement


def decode_full_gate_report(payload: bytes | Mapping):
    """Restore a phase report only after checking its declared gate scope."""
    from ..gate_battery import gate_phase_report_from_payload
    from .full_gates import uk_full_gate_manifest

    document = json.loads(payload) if isinstance(payload, bytes) else dict(payload)
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "uk_full_gate_report"
    ):
        raise ValueError("Unsupported UK full gate artifact.")
    gates = uk_full_gate_manifest(document["selection_receipt"])
    report = gate_phase_report_from_payload(document["report"], gates=gates)
    enforcement = _full_gate_enforcement(document, report)
    if enforcement != document["enforcement"]:
        raise ValueError("UK gate enforcement differs from its bound phase report.")
    return report, enforcement


def _spine_gate_evidence(context: KernelContext):
    from ..stage_evidence import decode_stage_evidence

    names = tuple(context.params["spine_stage_names"])
    if "spine_provenance" in context.artifacts:
        provenance = json.loads(context.artifacts["spine_provenance"].payload)
        if tuple(provenance["stages"]) != names:
            raise ValueError(
                "Bound spine stage roster differs from its declared gate input."
            )
        evidence = {
            name: provenance.get("stage_evidence", {}).get(name) for name in names
        }
        records = provenance.get("fit_weight_records")
        return evidence, records
    documents = {
        name: decode_stage_evidence(context.artifacts[name].payload, stage=name)
        for name in names
    }
    return (
        {name: document["evidence"] for name, document in documents.items()},
        {
            name: document["fit_weight_records"]
            for name, document in documents.items()
            if "fit_weight_records" in document
        },
    )


def _source_gate_evidence(context: KernelContext, engine):
    from datetime import date

    from .graph_targets import registry_from_payload

    surface = json.loads(context.artifacts["surface"].payload)
    return {
        "coverage_engine": engine,
        "build_stage_names": tuple(context.params["spine_stage_names"]),
        "reference_registry": registry_from_payload(surface["national_registry"]),
        "uk_ledger_compiled_registries": {
            int(period): registry_from_payload(registry)
            for period, registry in surface["uk_ledger_compiled_registries"].items()
        },
        "uk_ledger_compiled_local_registries": {
            int(period): registry_from_payload(registry)
            for period, registry in surface[
                "uk_ledger_compiled_local_registries"
            ].items()
        },
        "exclusions_evaluated_on": date.fromisoformat(
            str(context.params["review_date"])
        ),
    }


class UKFullGateKernel(KernelBase):
    ref = "uk.full-gates@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        role=KernelRole.GATE,
        dependencies=("policyengine-uk",),
    )

    def __init__(self, *, coverage_engine, engine_identity: str):
        self.engine = coverage_engine
        self.engine_identity = engine_identity

    def implementation_hash(self) -> str:
        from .. import gate_battery
        from ..country_spec import load_country_spec
        from . import (
            battery_bindings,
            diagnostics,
            full_gates,
            local_rowwise,
            weighted_integrity,
        )

        return hashlib.sha256(
            canonical_json(
                {
                    "code": source_hash(
                        sys.modules[__name__],
                        gate_battery,
                        battery_bindings,
                        full_gates,
                        local_rowwise,
                        weighted_integrity,
                        diagnostics,
                    ),
                    "country_resources": load_country_spec("uk").fingerprint,
                }
            )
        ).hexdigest()

    def run(self, context: KernelContext) -> KernelResult:
        from microcosm.calibrate.artifacts import decode_problem, decode_solution

        from ..gate_battery import (
            EvidenceContext,
            evaluate_phase,
            gate_phase_report_payload,
        )
        from ..stage_evidence import encode_stage_evidence
        from .battery_bindings import UK_GATE_REGISTRY
        from .full_gates import (
            build_full_gate_context,
            uk_full_gate_manifest,
            uk_full_gate_scope_receipt,
        )
        from .geography_ladder import load_uk_oa_ladder
        from .local_rowwise import uk_ladder_area_support_summary
        from .release_certification import rehydrate_uk_fit_weight_records
        from .weighted_integrity import load_uk_input_mass_reference

        if context.params["engine_identity"] != self.engine_identity:
            raise ValueError(
                "UK full gate engine identity differs from its declared binding."
            )
        selection = json.loads(context.artifacts["selection"].payload)["receipt"]
        gates = uk_full_gate_manifest(selection)
        from ..gate_battery import _gates_manifest_payload, _json_safe

        if (
            canonical_json(_gates_manifest_payload(uk_full_gate_manifest())).decode()
            != context.params["gate_manifest"]
        ):
            raise ValueError("UK full gate manifest differs from its declared binding.")
        stage_evidence, fit_weight_records = _spine_gate_evidence(context)
        supporting = _source_gate_evidence(context, self.engine)
        phase = str(context.params["phase"])
        diagnostics = []
        support = []
        output_artifacts = {}
        upstream_reports = {
            name: json.loads(context.artifacts[name].payload)
            for name in ("spine_assembled", "spine_transferred")
            if name in context.artifacts
        }
        if phase == "preflight":
            # These source/reference and roster checks have no final-weight or
            # contribution-matrix dependency. They run before dense calibration.
            evidence = EvidenceContext(artifacts=supporting)
        elif phase == "terminal":
            preflight_document = json.loads(context.artifacts["preflight"].payload)
            _, preflight = decode_full_gate_report(preflight_document)
            if not preflight["artifact_permitted"]:
                raise ValueError(
                    "Full calibration reached terminal gates despite a blocking source preflight."
                )
            upstream_reports = {
                **preflight_document.get("upstream_phase_reports", {}),
                "full_preflight": preflight_document["report"],
            }
            frame = context_frame(context)
            problem = decode_problem(context.artifacts["problem"].payload)
            solution = decode_solution(context.artifacts["solution"].payload)
            from microcosm.calibrate.artifacts import decode_calibration_result

            # The ordered result binds its initial weights. Rehydrate against
            # the same selected table axis, then install the actual graph Frame
            # so scoring uses the graph-owned mass ledger and final population.
            initial_frame = Frame(
                {entity: frame.table(entity) for entity in frame.entities},
                frame.schema,
                {"household": problem.problem.initial_weights},
                frame.strata,
                metadata=frame.metadata,
            )
            result = decode_calibration_result(
                context.artifacts["result"].payload,
                frame=initial_frame,
                problem=problem,
            )
            if not np.array_equal(result.weights, solution.weights):
                raise ValueError(
                    "Final diagnostics result differs from the installed solution."
                )
            result = replace(result, frame=frame)
            supporting["calibration_result"] = result
            fit_records = rehydrate_uk_fit_weight_records(
                {"fit_weight_records": fit_weight_records}
            )
            if fit_records is not None:
                supporting["fit_weight_records"] = fit_records
            if "uk_input_mass_reference" in context.sources:
                supporting["input_mass_reference"] = load_uk_input_mass_reference(
                    context.sources["uk_input_mass_reference"]
                )
            household = frame.table("household").copy()
            household["household_weight"] = frame.weights_for("household").values
            summaries = uk_ladder_area_support_summary(
                household, load_uk_oa_ladder(context.sources["uk_ladder"])
            )
            support_frame = pd.concat(
                (
                    summaries["constituency"].assign(geography_level="constituency"),
                    summaries["la"].assign(geography_level="local_authority"),
                ),
                ignore_index=True,
            )
            supporting["uk_area_support_summary"] = support_frame
            evidence = build_full_gate_context(
                frame,
                ordered_problem=problem,
                solution=solution,
                selection_receipt=selection,
                stage_evidence=stage_evidence,
                supporting_evidence=supporting,
            )
            diagnostics = evidence.artifacts["target_diagnostics"]
            support = support_frame.to_dict(orient="records")
            from .diagnostics import uk_calibration_diagnostics_payload
            from .graph_targets import registry_from_payload

            selected_registry = registry_from_payload(
                json.loads(context.artifacts["selection"].payload)["registry"]
            )
            holdout = json.loads(context.artifacts["holdout"].payload)
            complete_diagnostics = uk_calibration_diagnostics_payload(
                result,
                frame,
                target_geography_levels={
                    target.row_name: str(row["geography_level"])
                    for target, row in zip(
                        problem.problem.targets, problem.target_metadata, strict=True
                    )
                },
                target_registry=selected_registry,
                local_area_support=support_frame,
                rotated_holdout=holdout,
                build={
                    "build_kind": "uk_full_build",
                    "target_scope": selection["selector"],
                },
            )
            output_artifacts.update(
                {
                    "calibration_diagnostics": encode_stage_evidence(
                        _json_safe(complete_diagnostics)
                    ),
                    "target_diagnostics_csv": pd.DataFrame(diagnostics)
                    .to_csv(index=False)
                    .encode(),
                    "area_support_csv": support_frame.to_csv(index=False).encode(),
                }
            )
        else:
            raise ValueError(f"Unknown full gate phase {phase!r}.")
        report = evaluate_phase(
            gates, phase=phase, context=evidence, registry=UK_GATE_REGISTRY
        )
        payload = {
            "schema_version": 1,
            "kind": "uk_full_gate_report",
            "selection_receipt": selection,
            "sample_fraction": float(context.params["sample_fraction"]),
            "release_candidate": bool(context.params["release_candidate"]),
            "scope": uk_full_gate_scope_receipt(selection),
            "report": gate_phase_report_payload(report, gates=gates),
            "upstream_phase_reports": upstream_reports,
            "target_diagnostics": _json_safe(diagnostics),
            "area_support": _json_safe(support),
            "artifacts": {name: value.key for name, value in context.artifacts.items()},
        }
        enforcement = _full_gate_enforcement(payload, report)
        payload["enforcement"] = enforcement
        return KernelResult(
            artifacts={
                "gate_report": encode_stage_evidence(payload),
                **output_artifacts,
            },
            receipt={
                "outcome": "pass" if enforcement["artifact_permitted"] else "fail"
            },
        )


def append_uk_full_gate_nodes(
    graph: Graph,
    *,
    calibration,
    spine_stage_names: tuple[str, ...],
    engine_identity: str,
    review_date,
    sample_fraction: float = 1.0,
    release_candidate: bool = False,
    spine_provenance: ArtifactInput | None = None,
    input_population: str = "uk.full.pool",
    skip_holdout: bool = False,
) -> Graph:
    """Own source preflight before solve and final diagnostics after installation."""
    from microcosm.calibrate.artifacts import PROBLEM_TYPE, RESULT_TYPE, SOLUTION_TYPE

    from ..gate_battery import _gates_manifest_payload
    from ..stage_evidence import STAGE_EVIDENCE_TYPE
    from .full_gates import uk_full_gate_manifest
    from .graph_evidence import SPINE_GATE_REPORT_TYPE
    from .graph_targets import TARGET_SELECTION_TYPE, TARGET_SURFACE_TYPE

    if not engine_identity:
        raise ValueError("Full-build gates require a declared engine identity.")
    if any(node.id == "uk.full.gates.preflight" for node in graph.nodes):
        raise ValueError("Full gate nodes are already registered.")
    params = {
        "spine_stage_names": tuple(spine_stage_names),
        "engine_identity": engine_identity,
        "review_date": str(review_date),
        "sample_fraction": sample_fraction,
        "release_candidate": release_candidate,
        "gate_manifest": canonical_json(
            _gates_manifest_payload(uk_full_gate_manifest())
        ).decode(),
    }
    provenance = (
        (replace(spine_provenance, name="spine_provenance"),)
        if spine_provenance
        else tuple(
            ArtifactInput(
                name,
                "create_uk_frs" if name == "frs_spine" else name,
                "stage_evidence",
                STAGE_EVIDENCE_TYPE,
            )
            for name in spine_stage_names
        )
    )
    common = (
        ArtifactInput(
            "surface", "uk.full.target_compilation", "surface", TARGET_SURFACE_TYPE
        ),
        ArtifactInput(
            "selection", "uk.full.target_selection", "selection", TARGET_SELECTION_TYPE
        ),
        *provenance,
        *(
            ArtifactInput(
                f"spine_{phase}",
                f"spine.gates.{phase}",
                "gate_report",
                SPINE_GATE_REPORT_TYPE,
            )
            for phase in ("assembled", "transferred")
            if any(node.id == f"spine.gates.{phase}" for node in graph.nodes)
        ),
    )
    preflight = Node(
        "uk.full.gates.preflight",
        UKFullGateKernel.ref,
        population=input_population,
        params={**params, "phase": "preflight"},
        artifact_inputs=common,
        artifact_outputs=(ArtifactOutput("gate_report", FULL_GATE_REPORT_TYPE),),
        description="Validate complete source/reference registries and spine stage ownership before dense calibration.",
    )
    prerequisite = ArtifactInput(
        "preflight", preflight.id, "gate_report", FULL_GATE_REPORT_TYPE
    )
    nodes = tuple(
        replace(node, artifact_inputs=(*node.artifact_inputs, prerequisite))
        if node.id == calibration.dense_producer
        else node
        for node in graph.nodes
    )
    sources = ("uk_ladder",) + (
        ("uk_input_mass_reference",)
        if any(source.name == "uk_input_mass_reference" for source in graph.sources)
        else ()
    )
    final = Node(
        "uk.full.gates.calibrated",
        UKFullGateKernel.ref,
        population=calibration.population,
        inputs=population_slices(population_columns(graph, calibration.population)),
        sources=sources,
        params={**params, "phase": "terminal"},
        artifact_inputs=(
            *common,
            prerequisite,
            ArtifactInput(
                "problem", calibration.problem_producer, "problem", PROBLEM_TYPE
            ),
            ArtifactInput(
                "solution",
                calibration.solution_producer,
                "refit_solution" if calibration.size_producer else "solution",
                SOLUTION_TYPE,
            ),
        ),
        artifact_outputs=(ArtifactOutput("gate_report", FULL_GATE_REPORT_TYPE),),
        description="Compute final identified-row diagnostics once and evaluate all applicable national/local release gates.",
    )
    holdout = uk_full_holdout_node(
        graph,
        calibration=calibration,
        input_population=input_population,
        skip_holdout=skip_holdout,
    )
    final = replace(
        final,
        artifact_inputs=(
            *final.artifact_inputs,
            ArtifactInput("result", calibration.result_producer, "result", RESULT_TYPE),
            ArtifactInput("holdout", holdout.id, "holdout", FULL_HOLDOUT_TYPE),
        ),
        artifact_outputs=(
            *final.artifact_outputs,
            ArtifactOutput("calibration_diagnostics", FULL_DIAGNOSTICS_TYPE),
            ArtifactOutput("target_diagnostics_csv", FULL_DIAGNOSTICS_CSV_TYPE),
            ArtifactOutput("area_support_csv", FULL_SUPPORT_CSV_TYPE),
        ),
    )
    return replace(graph, nodes=(*nodes, preflight, holdout, final))


def register_uk_full_gate_kernels(
    registry: KernelRegistry, *, coverage_engine, engine_identity: str
) -> None:
    registry.register(
        UKFullGateKernel(
            coverage_engine=coverage_engine, engine_identity=engine_identity
        )
    )

    registry.register(UKFullHoldoutKernel())


class UKFullHoldoutKernel(KernelBase):
    ref = "uk.full.rotated-holdout@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.PARAM,
        dependencies=("policyengine-uk", "torch"),
    )

    def implementation_hash(self) -> str:
        from ..country_spec import load_country_spec
        from . import dataset_size, graph_targets, local_rowwise

        return hashlib.sha256(
            canonical_json(
                {
                    "code": source_hash(
                        sys.modules[__name__],
                        graph_targets,
                        local_rowwise,
                        dataset_size,
                        dependencies=self.capabilities.dependencies,
                    ),
                    "country_resources": load_country_spec("uk").fingerprint,
                }
            )
        ).hexdigest()

    def run(self, context: KernelContext) -> KernelResult:
        from microcosm.calibrate.artifacts import decode_problem

        from ..gate_battery import _json_safe
        from ..stage_evidence import encode_stage_evidence
        from .graph_targets import reconstruct_uk_full_problem_inputs
        from .local_rowwise import rotated_uk_local_holdout

        _, preflight = decode_full_gate_report(context.artifacts["preflight"].payload)
        if not preflight["artifact_permitted"]:
            raise ValueError(
                "Rotated holdout cannot run after a blocking source preflight."
            )
        if context.params["skip_holdout"]:
            report = {
                "report_only": True,
                "skipped": True,
                "reason": "Explicit development request; no holdout claim.",
            }
        else:
            inputs = reconstruct_uk_full_problem_inputs(context)
            original = decode_problem(context.artifacts["problem"].payload)
            if original.entity_ids != tuple(
                inputs.frame.table("household")["household_id"]
            ):
                raise ValueError(
                    "Holdout original pool differs from its bound problem axis."
                )
            report = rotated_uk_local_holdout(
                inputs.frame,
                inputs.local_problem,
                bound_families=inputs.bound_families,
                national_rows=inputs.national_rows,
                target_weight_rule=str(context.params["target_weight_rule"]),
                epochs=int(context.params["epochs"]),
                learning_rate=float(context.params["learning_rate"]),
                conserve_mass=False,
                target_records=None,
                l0_lambda=0.0,
                budget_iters=10,
                dataset_households=context.params.get("dataset_households"),
                solve_seed=int(context.params["seed"]),
                selection_seed=context.params.get("selection_seed"),
                selection_pi_hi=float(context.params["selection_pi_hi"]),
                baseline_pi_floor=float(context.params["baseline_pi_floor"]),
            )
        report = {
            **report,
            "graph_binding": {
                "original_problem_artifact": context.artifacts["problem"].key,
                "artifacts": {
                    name: value.key for name, value in context.artifacts.items()
                },
            },
        }
        return KernelResult(
            artifacts={"holdout": encode_stage_evidence(_json_safe(report))}
        )


def uk_full_holdout_node(
    graph: Graph, *, calibration, input_population: str, skip_holdout: bool
) -> Node:
    from microcosm.calibrate.artifacts import PROBLEM_TYPE

    original = graph.node("uk.full.problem")
    dense = graph.node(calibration.dense_producer)
    size = (
        None
        if calibration.size_producer is None
        else graph.node(calibration.size_producer)
    )
    return Node(
        "uk.full.holdout",
        UKFullHoldoutKernel.ref,
        population=input_population,
        inputs=population_slices(population_columns(graph, input_population)),
        sources=original.sources,
        params={
            **original.params,
            **dense.params,
            "skip_holdout": skip_holdout,
            "dataset_households": None if size is None else size.params["households"],
            "selection_seed": None if size is None else size.params["seed"],
            "selection_pi_hi": 1.0 if size is None else size.params["pi_hi"],
            "baseline_pi_floor": (
                0.0 if size is None else size.params["baseline_pi_floor"]
            ),
        },
        artifact_inputs=(
            *original.artifact_inputs,
            ArtifactInput("problem", original.id, "problem", PROBLEM_TYPE),
            ArtifactInput(
                "preflight",
                "uk.full.gates.preflight",
                "gate_report",
                FULL_GATE_REPORT_TYPE,
            ),
        ),
        artifact_outputs=(ArtifactOutput("holdout", FULL_HOLDOUT_TYPE),),
        description="Preserve five rotated local-target holdouts with national constraints fixed in training and unchanged sizing doctrine.",
    )


def materialize_uk_terminal_artifacts(
    manifest, store, *, directory: str | Path, stem: str
) -> dict[str, dict[str, object]]:
    """Write graph-produced diagnostic bytes atomically, including after cache hits."""
    from ..artifact_files import materialize_bytes

    if not stem or Path(stem).name != stem:
        raise ValueError(
            "UK terminal artifact stem must be a simple filename component."
        )
    root = Path(directory)
    artifacts = {
        "calibration_diagnostics": (
            "uk.full.gates.calibrated",
            "calibration_diagnostics",
            ".diagnostics.json",
        ),
        "target_diagnostics": (
            "uk.full.gates.calibrated",
            "target_diagnostics_csv",
            ".targets.csv",
        ),
        "area_support": (
            "uk.full.gates.calibrated",
            "area_support_csv",
            ".area_support.csv",
        ),
        "holdout": ("uk.full.holdout", "holdout", ".holdout.json"),
        "target_registry": (
            "uk.full.target_selection",
            "selection",
            ".target_selection.json",
        ),
    }
    inventory = {}
    for role, (node, output, suffix) in artifacts.items():
        key = manifest.nodes[node].opaque_artifacts[output]
        inventory[role] = {
            **materialize_bytes(store.load_bytes(key), root / (stem + suffix)),
            "graph_artifact_key": key,
        }
    return inventory


# ---------------------------------------------------------------------------
# The rowwise candidate manifest projected from a finished graph
# ---------------------------------------------------------------------------
#
# ``rowwise_candidate_manifest.json`` is the dense line's evidence contract:
# the release pre-flight (``tools/preflight_uk_local_release_candidate.py``),
# the dense release assembler (``tools/assemble_uk_dense_release_dir.py``) and
# the staged-dataset lane read its schema-4 shape. The rowwise tool renders
# it from live objects (the solve, the clone, the problem); the graph keeps
# those as stored artifacts, so this projection reads the same facts back
# from the run manifest and the content store. It is new UK code by
# necessity: main's ``_manifest`` cannot run without the live objects. The
# projection adds one ``graph`` block naming the artifacts it stood on.
#
# Per-epoch ``calibration_progress`` rows: the dense solve node forwards its
# epochs through the observer registered on ``UKDenseSolveKernel``; the size
# search and refit nodes solve through ``dataset_size`` without an observer,
# so a size run stages no epoch rows for those two phases (recorded in the
# manifest's ``graph.epoch_rows`` field).

_LADDER_TARGET_PREFIX = "ons.census.households@"
_NATIONAL_MATERIALIZATION = "uk_national_measure"


def _graph_payload(manifest, store, node: str, artifact: str) -> bytes:
    return store.load_bytes(manifest.nodes[node].opaque_artifacts[artifact])


def _optional_graph_json(manifest, store, node: str, artifact: str):
    if node not in manifest.nodes:
        return None
    return json.loads(_graph_payload(manifest, store, node, artifact))


def _gate_rows(document: Mapping) -> dict[str, dict]:
    """The battery-shaped ``gates`` mapping of one graph gate report."""
    rows = {}
    for outcome in document["report"]["outcomes"]:
        entry = dict(outcome)
        rows[str(entry.pop("id"))] = entry
    return rows


def _fit_rows(target_rows: pd.DataFrame, materialization: Mapping[str, str]):
    """Split the terminal target diagnostics into local and national rows."""
    kinds = target_rows["name"].map(materialization)
    if kinds.isna().any():
        raise ValueError(
            "Terminal target diagnostics name targets the ordered problem lacks."
        )
    national = target_rows[kinds == _NATIONAL_MATERIALIZATION].reset_index(drop=True)
    local = target_rows[kinds != _NATIONAL_MATERIALIZATION].reset_index(drop=True)
    return local, national


def rowwise_candidate_manifest_from_graph(
    final_manifest,
    store,
    *,
    args,
    posture,
    pins: Mapping[str, Mapping[str, object]],
    terminal_files: Mapping[str, Mapping[str, object]],
    frame: Frame,
    outputs: Mapping[str, Mapping[str, object]],
    source_year: int,
    inputs: Mapping[str, Mapping[str, object]],
    ladder_provenance: Mapping[str, object],
    code: Mapping[str, object],
    runtime: Mapping[str, str],
    created_at: str,
) -> dict:
    """Project the schema-4 rowwise candidate manifest from stored artifacts.

    ``frame`` is the exported population, ``outputs`` the published file
    records (``path``/``sha256``/``bytes``) keyed as the rowwise tool keys
    them, ``inputs`` the ``dataset``/``ladder`` artifact records with their
    ``pin_verified`` flags and ``code``/``runtime`` the git and package pins.
    Every numerical fact comes from a graph artifact named in ``graph``.
    """
    from microcosm.calibrate.artifacts import decode_problem

    from .calibration_run import UK_LOCAL_GATE_SCOPE
    from .diagnostics import uk_fit_by_family, uk_support_limited_misses
    from .graph_targets import registry_from_payload
    from .national_sampling import UK_SAMPLE_RUNG_TOKENS
    from .rowwise_cli import (
        gate_failures_by_criticality,
        local_vintage_census,
        release_verdict,
        rowwise_parameters,
    )

    nodes = final_manifest.nodes
    problem = decode_problem(
        _graph_payload(final_manifest, store, "uk.full.problem", "problem")
    )
    bindings = dict(problem.bindings)
    surface = json.loads(
        _graph_payload(final_manifest, store, "uk.full.target_compilation", "surface")
    )
    gate_document = json.loads(
        _graph_payload(final_manifest, store, "uk.full.gates.calibrated", "gate_report")
    )
    diagnostics = json.loads(
        _graph_payload(
            final_manifest, store, "uk.full.gates.calibrated", "calibration_diagnostics"
        )
    )
    target_rows = pd.read_csv(
        io.BytesIO(
            _graph_payload(
                final_manifest,
                store,
                "uk.full.gates.calibrated",
                "target_diagnostics_csv",
            )
        )
    )
    support = pd.read_csv(
        io.BytesIO(
            _graph_payload(
                final_manifest, store, "uk.full.gates.calibrated", "area_support_csv"
            )
        )
    )
    holdout = json.loads(
        _graph_payload(final_manifest, store, "uk.full.holdout", "holdout")
    )
    geography_gate = json.loads(
        _graph_payload(final_manifest, store, "uk.full.geography_gate", "gate")
    )
    sampling = json.loads(
        _graph_payload(final_manifest, store, "uk.full.sample", "sampling")
    )["receipt"]
    size_receipt = _optional_graph_json(
        final_manifest, store, "uk.full.size_refit", "size"
    )
    spine_provenance = (
        _optional_graph_json(
            final_manifest, store, "uk.full.spine_checkpoint", "spine_provenance"
        )
        or {}
    )
    # The request's geography binding (assignment mode, definition digest,
    # support pins, identity column, stream) travels in the export descriptor.
    export_descriptor = (
        _optional_graph_json(
            final_manifest, store, "uk.full.export.prepare", "export_descriptor"
        )
        or {}
    )
    geography_binding = dict(
        (export_descriptor.get("bindings") or {}).get("geography") or {}
    )
    enforcement = dict(gate_document["enforcement"])
    gate_rows = _gate_rows(gate_document)
    local_scope = {
        gate_id: gate_rows[gate_id]
        for gate_id in UK_LOCAL_GATE_SCOPE
        if gate_id in gate_rows
    }
    blocking_lines, diagnostic_lines = gate_failures_by_criticality(
        {"gates": gate_rows}
    )
    enforced = set(enforcement["enforced_blocking"])
    unenforced = set(enforcement["unenforced_release_failures"])
    blocking_failures = [
        line for line in blocking_lines if line[1:].split("]")[0] in enforced
    ]
    unenforced_failures = [
        line for line in blocking_lines if line[1:].split("]")[0] in unenforced
    ]
    releasable, release_posture = release_verdict(
        sample_fraction=args.sample_fraction,
        engine_blocks=args.engine_blocks,
        release_blocking_gates_passed=bool(
            enforcement["release_blocking_gates_passed"]
        ),
    )
    materialization = {
        str(problem.problem.targets[index].row_name): str(row.get("materialization"))
        for index, row in enumerate(problem.target_metadata)
    }
    ladder_rows = sum(
        1
        for name, kind in materialization.items()
        if kind != _NATIONAL_MATERIALIZATION and name.startswith(_LADDER_TARGET_PREFIX)
    )
    national_rows = sum(
        1 for kind in materialization.values() if kind == _NATIONAL_MATERIALIZATION
    )
    local_rows = len(materialization) - ladder_rows - national_rows
    local_fit, national_fit = _fit_rows(target_rows, materialization)
    abs_errors = target_rows["abs_relative_error"].to_numpy(dtype=np.float64)
    support_by_grain = {
        ("la" if grain == "local_authority" else str(grain)): rows.reset_index(
            drop=True
        )
        for grain, rows in support.groupby("geography_level", sort=True)
    }
    household_weights = np.asarray(
        frame.weights_for("household").values, dtype=np.float64
    )
    design = pd.Series(
        np.asarray(problem.problem.initial_weights.values, dtype=np.float64),
        index=pd.Index(list(problem.entity_ids)),
    )
    exported_ids = frame.table("household")["household_id"].tolist()
    design_weights = design.reindex(exported_ids).to_numpy(dtype=np.float64)
    if np.isnan(design_weights).any():
        raise ValueError("Exported households are not a subset of the ordered pool.")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_vs_design = float(np.nanmax(np.divide(household_weights, design_weights)))
    calibration_record = frame.mass_log[-1]
    if "calibration" not in str(calibration_record.reason):
        raise ValueError(
            "exported frame's latest mass record is not the calibration record: "
            f"{calibration_record.reason!r}."
        )
    old_total = float(calibration_record.old_total)
    new_total = float(calibration_record.new_total)
    area_gate = gate_rows.get("uk_local_area_support", {})
    area_details = (
        area_gate.get("details", {}) if isinstance(area_gate, Mapping) else {}
    )
    past_cap = diagnostics.get("past_cap_census") or {}
    weight_kind = uk_household_weight_kind(frame).value
    graph_keys = {
        node_id: dict(receipt.opaque_artifacts)
        for node_id, receipt in nodes.items()
        if node_id
        in {
            "uk.full.problem",
            "uk.full.target_compilation",
            "uk.full.target_selection",
            "uk.full.gates.calibrated",
            "uk.full.holdout",
            "uk.full.package",
            "uk.full.sample",
            "uk.full.geography_gate",
            "uk.full.size_refit",
            "uk.full.spine_checkpoint",
        }
    }
    return {
        "schema_version": 4,
        "build_kind": "uk_rowwise_calibrated_candidate",
        "release_role": posture.role,
        "release_id": posture.release_id,
        "candidate_scope": "adjudicated_partial",
        "created_at": created_at,
        "git_commit": code.get("git_commit"),
        "git_dirty": code.get("git_dirty"),
        "bound_target_families": list(bindings.get("bound_families", ())),
        "binding_adjudications": dict(bindings.get("binding_adjudications", {})),
        "cross_grain": dict(bindings.get("cross_geography", {})),
        "ladder_assignment_provenance": dict(ladder_provenance),
        "household_dispersion": dict(surface.get("household_dispersion", {})),
        "parameters": rowwise_parameters(args, source_year=source_year),
        "inputs": {
            "dataset": dict(inputs["dataset"]),
            "ladder": dict(inputs["ladder"]),
        },
        "identity": {
            "spine": {
                **dict(inputs["dataset"]),
                "spine_provenance": dict(spine_provenance),
            },
            "ladder": {
                **dict(inputs["ladder"]),
                "layer_vintages": dict(ladder_provenance),
                "matches_local_area_crosswalk_pin": True,
            },
            "targets": dict(surface["source_validation"]["targets"]),
            "code": dict(code),
            "runtime": dict(runtime),
            "sampling": dict(sampling),
            "survey_year": int(source_year),
            "calibration_year": int(surface["calibration_year"]),
        },
        "sampling": dict(sampling),
        "rung_surface": {
            **dict(bindings.get("rung_surface", {})),
            "rung": UK_SAMPLE_RUNG_TOKENS[float(args.sample_fraction)],
            "fraction": float(args.sample_fraction),
            "unreachable_check": "completed",
        },
        "outputs": {key: dict(value) for key, value in outputs.items()},
        "geography": {
            "constituencies_assigned": int(
                support.loc[
                    support["geography_level"] == "constituency", "area_code"
                ].nunique()
            ),
            "local_authorities_assigned": int(
                support.loc[
                    support["geography_level"] == "local_authority", "area_code"
                ].nunique()
            ),
            "missing_geography_rows": 0,
            "ladder_gate": {**geography_gate, "phase": "post_calibration"},
            "assignment": geography_binding,
        },
        "gate": {**geography_gate, "phase": "post_calibration"},
        "weights": {
            "household_weight_kind": weight_kind,
            "household_weight_kind_chain": [
                {"stage": "staging", "kind": "importance"},
                *(
                    []
                    if float(args.sample_fraction) == 1.0
                    else [{"stage": "sample", "kind": "importance"}]
                ),
                {"stage": "ladder_clone", "kind": "importance"},
                {"stage": "rowwise_calibration", "kind": weight_kind},
            ],
            "mass_log_records_before_calibration": len(frame.mass_log) - 1,
            "mass_log_records": len(frame.mass_log),
            "calibration_mass_change": {
                "entity": str(calibration_record.entity),
                "old_total": old_total,
                "new_total": new_total,
                "relative_shift": (new_total - old_total) / old_total,
                "declared_factor": calibration_record.declared_factor,
                "reason": str(calibration_record.reason),
            },
            "abs_delta": abs(new_total - old_total),
            "declared_stretch_bound": float(posture.doctrine.max_weight_ratio),
            "stretch_reference": "pool_design"
            if size_receipt is None
            else "normalized_horvitz_thompson_w_over_q",
            "realized_max_weight_ratio_vs_stretch_reference": (
                ratio_vs_design
                if size_receipt is None
                else float(diagnostics["realized_max_weight_ratio"])
            ),
            "realized_max_weight_ratio_vs_design": ratio_vs_design,
        },
        "solve": {
            "n_targets": int(len(materialization)),
            "n_targets_by_kind": {
                "local": int(local_rows),
                "ladder": int(ladder_rows),
                "national": int(national_rows),
            },
            "n_households": int(frame.n("household")),
            "pool_households": int(len(problem.entity_ids)),
            "dataset_size": None if size_receipt is None else dict(size_receipt),
            "initial_loss": diagnostics["initial_loss"],
            "final_loss": diagnostics["final_loss"],
            "max_abs_relative_error": float(abs_errors.max())
            if len(abs_errors)
            else None,
            "median_abs_relative_error": float(np.median(abs_errors))
            if len(abs_errors)
            else None,
            "n_nonzero": int(diagnostics["n_nonzero"]),
            "past_cap": {
                "n_targets": past_cap.get("n_targets"),
                "past_at_init": past_cap.get("initial_past_cap"),
                "past_at_final": past_cap.get("final_past_cap"),
                "escaped": past_cap.get("escaped"),
                "frozen": past_cap.get("frozen"),
                "pushed_out": past_cap.get("pushed_out"),
            },
            "loss_shape": "capped_relative_error",
            "target_weight_rule": args.target_weight_rule,
            "target_weight_rule_override": (
                {}
                if args.target_weight_rule == posture.target_weight_rule
                else {
                    "target_weight_rule": {
                        "default": posture.target_weight_rule,
                        "effective": args.target_weight_rule,
                    }
                }
            ),
            "measure_resolution": dict(bindings.get("measure_resolution", {})),
            "cross_grain": dict(bindings.get("cross_geography", {})),
            "binding_adjudications": dict(bindings.get("binding_adjudications", {})),
            "area_support_exclusions": {
                "resource": "local_area_support_exclusions.json",
                "entries_stood_on": sorted(area_details.get("reviewed_exclusions", {})),
                "stale": list(area_details.get("stale_exclusions", [])),
                "unknown": list(area_details.get("unknown_exclusions", [])),
            },
        },
        "diagnostics": {
            "schema_version": diagnostics["schema_version"],
            "target_registry": diagnostics.get("target_registry"),
            "weakest_families": diagnostics["uk_diagnostics"].get("weakest_families"),
            "weakest_areas_by_fit": diagnostics["uk_diagnostics"].get(
                "weakest_areas_by_fit"
            ),
            "rotated_holdout": diagnostics["uk_diagnostics"].get("rotated_holdout"),
        },
        "support": {
            "min_assigned_households": int(support["assigned_households"].min()),
            "min_nonzero_households": int(support["nonzero_households"].min()),
            "min_effective_sample_size": float(support["effective_sample_size"].min()),
            "by_geography_level": {
                str(level): {
                    "min_assigned_households": int(rows["assigned_households"].min()),
                    "min_nonzero_households": int(rows["nonzero_households"].min()),
                    "min_effective_sample_size": float(
                        rows["effective_sample_size"].min()
                    ),
                    "min_nonzero_source_households": int(
                        rows["nonzero_source_households"].min()
                    ),
                }
                for level, rows in support.groupby("geography_level", sort=True)
            },
        },
        "fit": {
            "local_by_family": uk_fit_by_family(local_fit),
            "national_by_family": uk_fit_by_family(national_fit),
            "weakest_families": sorted(
                [*uk_fit_by_family(local_fit), *uk_fit_by_family(national_fit)],
                key=lambda row: (
                    -float(row["worst_abs_relative_error"]),
                    row["family"],
                ),
            )[:10],
            "weakest_areas_by_fit": dict(
                diagnostics["uk_diagnostics"].get("weakest_areas_by_fit") or {}
            ),
            "support_limited_misses": dict(
                uk_support_limited_misses(
                    local_fit, support_by_grain, max_abs_relative_error=0.25
                )
                if len(local_fit)
                else {}
            ),
            "rotated_holdout": dict(holdout),
        },
        "vintages": local_vintage_census(
            registry_from_payload(surface["local_registry"])
        ),
        "failing_gate_ids": sorted(
            gate_id
            for gate_id, payload in gate_rows.items()
            if not isinstance(payload, Mapping) or payload.get("status") != "passed"
        ),
        "releasable": bool(releasable and args.dataset_households is None),
        "release_posture": {
            **release_posture,
            **(
                {}
                if args.dataset_households is None
                else {"size_certification_present": False}
            ),
        },
        "census_household_uprating": dict(
            surface.get("census_household_uprating")
            or {"applied": False, "reason": "no cross-grain receipt"}
        ),
        "measure_exclusions": {
            str(name): dict(record)
            for name, record in sorted(
                (surface.get("measure_exclusions") or {}).items()
            )
        },
        "blocked_at_f100": bool(blocking_failures),
        "blocking_failures": blocking_failures,
        "diagnostic_failures": diagnostic_lines,
        "release_gate_failures_not_enforced": unenforced_failures,
        "local_gate_scope": sorted(local_scope),
        "graph": {
            "artifacts": graph_keys,
            "terminal_files": {
                key: dict(value) for key, value in terminal_files.items()
            },
            "enforcement": enforcement,
            "epoch_rows": "dense_solve_only",
        },
    }
