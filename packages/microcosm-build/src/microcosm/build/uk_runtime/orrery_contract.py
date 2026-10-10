"""UK inputs to the shared presentation and execution-evidence contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from microcosm.build.country_spec import CountrySpec, load_country_spec
from microcosm.calibrate.artifacts import RESULT_TYPE, decode_problem
from microcosm.calibrate.graph_evidence import (
    CALIBRATION_SUMMARY_PROVIDERS,
    result_summary,
)
from microcosm.graph import (
    compile_graph,
    publish_run_evidence,
    save_graph_schema,
    save_run_evidence,
)
from microcosm.graph.evidence import ArtifactSummaryContext, read_json
from microcosm.graph.evidence import checkpoint_references as _checkpoint_references
from microcosm.graph.presentation import PRESENTATION_PROTOCOL

from .graph import _manifest_stages, uk_spine_operation_inventory
from .graph_calibration import SIZE_DRAW_TYPE, SIZE_RECEIPT_TYPE, SIZE_SEARCH_TYPE
from .graph_evidence import SPINE_GATE_REPORT_TYPE
from .graph_national import NATIONAL_GATE_REPORT_TYPE, NATIONAL_TARGET_TYPE
from .graph_targets import TARGET_SELECTION_TYPE, registry_from_payload
from .graph_terminal import FULL_GATE_REPORT_TYPE


def _uk_result_summary(context: ArtifactSummaryContext) -> dict:
    data = result_summary(context)
    ancestors, pending = set(), [context.node_id]
    while pending:
        current = pending.pop()
        if current in ancestors:
            continue
        ancestors.add(current)
        pending.extend(context.run.compiled.predecessors[current])
    specs = {}
    for node_id in ancestors:
        receipt = context.run.manifest.nodes.get(node_id)
        if receipt is None:
            continue
        for name, descriptor in receipt.typed_artifacts.get("outputs", {}).items():
            if name not in receipt.opaque_artifacts:
                continue
            type_ = descriptor["type"]
            if type_ == {
                "name": TARGET_SELECTION_TYPE.name,
                "schema_version": TARGET_SELECTION_TYPE.schema_version,
            }:
                raw = json.loads(context.store.load_bytes(descriptor["key"]))[
                    "registry"
                ]
            elif type_ == {
                "name": NATIONAL_TARGET_TYPE.name,
                "schema_version": NATIONAL_TARGET_TYPE.schema_version,
            }:
                raw = json.loads(context.store.load_bytes(descriptor["key"]))[
                    "national_registry"
                ]
            else:
                continue
            for spec in registry_from_payload(raw).specs:
                row_name = spec.to_target().row_name
                if row_name in specs and specs[row_name] != spec:
                    raise ValueError("Conflicting recorded target specifications.")
                specs[row_name] = spec
    for row in data["tables"]["targets"]:
        spec = specs.get(row["name"])
        if spec is None:
            continue
        if row["target"] != spec.value:
            raise ValueError("Recorded target registry differs from the result.")
        row["source_entity"] = spec.entity
        row["family"] = spec.family
        row["uncertainty"] = (
            {"status": "recorded", "se": spec.se}
            if spec.se is not None
            else {"status": "not_recorded"}
        )
    return data


def _size_summary(context: ArtifactSummaryContext) -> dict:
    """Project the maintained size artifact, excluding IDs, masks and draws."""
    value = json.loads(context.payload)
    problem = decode_problem(context.input_payload("problem"))
    if value.get("problem_sha256") != problem.sha256:
        raise ValueError("Size summary belongs to a different ordered problem.")
    method = value.get("method")
    if method not in {
        "full_pool",
        "contribution_informed_l0",
        "exact_count",
        "contribution_informed_l0_exact_count_refit",
    }:
        raise ValueError("Unknown recorded UK size method.")
    fields = (
        "method",
        "requested_households",
        "realized_households",
        "pool_households",
        "households",
        "seed",
        "protected_carriers",
        "certainty_share",
        "boundary_draws",
        "zero_target_rows",
        "selection_pi_hi",
        "selection_search_pi_hi",
        "selection_reused",
        "selection_budget_basis",
        "selection_l0_lambda",
        "selection_epochs",
        "refit_epochs",
        "refit_baseline",
        "stretch_reference",
        "baseline_pi_floor",
        "baseline_floored_rows",
        "baseline_mass_share_certainties",
        "dense_loss",
        "compact_loss",
        "max_target_scaled_change",
        "certification",
        "pi_hi",
    )
    summary = {key: value[key] for key in fields if key in value}
    summary["problem_sha256"] = problem.sha256
    for key in ("support", "protected", "household_ids", "pool_row_indices"):
        if key in value:
            summary[f"{key}_count"] = len(value[key])
    if "inclusion_probabilities" in value:
        probabilities = np.asarray(value["inclusion_probabilities"], dtype=float)
        if (
            probabilities.ndim != 1
            or not np.isfinite(probabilities).all()
            or (probabilities <= 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError("Invalid recorded inclusion probabilities.")
        summary["inclusion_probability_range"] = {
            "minimum": float(probabilities.min()),
            "maximum": float(probabilities.max()),
        }
    return {"overview": summary}


_GATE_STATUSES = ("passed", "failed", "not_applicable", "evidence_absent", "unreached")
_GATE_ROW_FIELDS = ("id", "gate", "phase", "criticality", "status", "reason")
_FULL_ENFORCEMENT_FIELDS = (
    "structural_failures",
    "enforced_blocking",
    "exportable_blocking",
    "unenforced_release_failures",
    "diagnostic_failures",
    "artifact_permitted",
    "release_blocking_gates_passed",
)


def _gate_rows(report: object) -> tuple[list[dict], dict]:
    """Project one phase report's per-gate verdicts; details stay in the artifact."""
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("Unsupported gate phase report.")
    outcomes = report.get("outcomes")
    if not isinstance(outcomes, list):
        raise ValueError("Gate phase report has no outcome list.")
    rows = []
    counts = dict.fromkeys(_GATE_STATUSES, 0)
    for outcome in outcomes:
        if not isinstance(outcome, dict) or outcome.get("status") not in counts:
            raise ValueError("Gate outcome has an unknown status.")
        row = {field: outcome.get(field) for field in _GATE_ROW_FIELDS}
        if not isinstance(row["id"], str) or not row["id"]:
            raise ValueError("Gate outcome has no identifier.")
        failures = outcome.get("failures", [])
        if not isinstance(failures, list) or not all(
            isinstance(line, str) for line in failures
        ):
            raise ValueError("Gate failures must be text lines.")
        row["failures"] = failures
        row["evidence_sha256"] = outcome.get("evidence_sha256")
        counts[row["status"]] += 1
        rows.append(row)
    overview = {
        "phase": report.get("phase"),
        "gates_manifest_sha256": report.get("gates_manifest_sha256"),
        "gates": len(rows),
        "status_counts": counts,
    }
    return rows, overview


def _gate_report_summary(context: ArtifactSummaryContext) -> dict:
    """Per-gate outcomes for spine, full and national gate batteries.

    The spine artifact is a bare phase report; the full and national artifacts
    wrap one in a document with enforcement or blocking verdicts. Gate detail
    payloads and target or area tables are not projected.
    """
    document = json.loads(context.payload)
    if not isinstance(document, dict):
        raise ValueError("Gate report artifact must be a JSON object.")
    kind = document.get("kind")
    if kind is None:
        rows, overview = _gate_rows(document)
        overview["kind"] = "gate_phase_report"
    elif kind in {"uk_full_gate_report", "uk_national_gate_report"}:
        if document.get("schema_version") != 1:
            raise ValueError("Unsupported UK gate report schema.")
        rows, overview = _gate_rows(document.get("report"))
        overview["kind"] = kind
        if kind == "uk_full_gate_report":
            enforcement = document.get("enforcement")
            if not isinstance(enforcement, dict):
                raise ValueError("UK full gate report has no enforcement block.")
            overview["enforcement"] = {
                field: enforcement.get(field) for field in _FULL_ENFORCEMENT_FIELDS
            }
            overview["sample_fraction"] = document.get("sample_fraction")
            overview["release_candidate"] = document.get("release_candidate")
            upstream = document.get("upstream_phase_reports", {})
            if not isinstance(upstream, dict):
                raise ValueError("Upstream phase reports must be an object.")
            overview["upstream_phases"] = {
                name: _gate_rows(payload)[1] for name, payload in upstream.items()
            }
        else:
            blocking = document.get("blocking", [])
            if not isinstance(blocking, list):
                raise ValueError("National blocking gates must be a list.")
            overview["blocking"] = blocking
            overview["artifact_permitted"] = document.get("artifact_permitted")
            overview["posture"] = document.get("posture")
            overview["scope"] = document.get("scope")
    else:
        raise ValueError(f"Unknown UK gate report kind {kind!r}.")
    return {"overview": overview, "tables": {"gates": rows}}


UK_SUMMARY_PROVIDERS = {
    **CALIBRATION_SUMMARY_PROVIDERS,
    (RESULT_TYPE.name, RESULT_TYPE.schema_version): _uk_result_summary,
    (SIZE_SEARCH_TYPE.name, SIZE_SEARCH_TYPE.schema_version): _size_summary,
    (SIZE_DRAW_TYPE.name, SIZE_DRAW_TYPE.schema_version): _size_summary,
    (SIZE_RECEIPT_TYPE.name, SIZE_RECEIPT_TYPE.schema_version): _size_summary,
    (
        SPINE_GATE_REPORT_TYPE.name,
        SPINE_GATE_REPORT_TYPE.schema_version,
    ): _gate_report_summary,
    (FULL_GATE_REPORT_TYPE.name, FULL_GATE_REPORT_TYPE.schema_version): (
        _gate_report_summary
    ),
    (
        NATIONAL_GATE_REPORT_TYPE.name,
        NATIONAL_GATE_REPORT_TYPE.schema_version,
    ): _gate_report_summary,
}


def checkpoint_references(sidecar_path: Path | None) -> tuple[list[dict], str | None]:
    """Use only recorded checkpoint lineage, never today's reconstructed spine."""
    if sidecar_path is None or not sidecar_path.is_file():
        return [], "The checkpoint has no recorded producing graph and manifest."
    sidecar, _ = read_json(sidecar_path)
    return _checkpoint_references(sidecar, base=sidecar_path.parent)


# Presentation groups are an explicit roster, not a substring rule: a new node
# lands in "other" until it is classified here, and the maintained declarations
# are tested to contain no unclassified operation.
UK_GROUPS = (
    ("sources", "Sources", "Declared external inputs and their codecs."),
    (
        "enrichment",
        "Enrichment (spine stages)",
        "Source assembly and imputation stages of the canonical FRS spine, with "
        "their ownership boundaries.",
    ),
    (
        "population",
        "Population pool",
        "Checkpoint admission, sampling, normalisation, expansion into K "
        "geographic copies and the pooled problem checkpoint.",
    ),
    (
        "geography",
        "Geography",
        "Location draws, derived geographies and the geography integrity gate.",
    ),
    (
        "targets",
        "Targets",
        "Target compilation and selection, measure evaluation and the ordered "
        "calibration problem.",
    ),
    (
        "calibration",
        "Calibration",
        "Dense solve and the optional size search, draw and refit path.",
    ),
    (
        "checks",
        "Checks and gates",
        "Gate batteries and holdout diagnostics; verdicts are recorded, not inferred.",
    ),
    (
        "export",
        "Export",
        "Dataset export, read-back verification and package binding.",
    ),
    ("other", "Unclassified operations", "Operations not yet classified."),
)
_FULL_BUILD_GROUPS = {
    "population": (
        "uk.full.spine_checkpoint",
        "uk.full.sample",
        "uk.full.normalize",
        "uk.full.expand",
        "uk.full.pool",
    ),
    "geography": (
        "uk.full.identity",
        "uk.full.locations",
        "uk.full.geography_mapping",
        "uk.full.geography_gate",
        "uk.full.geography.assign",
        "uk.full.geography.derive",
        "uk.full.geography.local_authority",
        "uk.full.geography.gate",
    ),
    "targets": (
        "uk.full.target_compilation",
        "uk.full.target_selection",
        "uk.full.measures",
        "uk.full.problem",
        "uk.full.national_targets",
        "uk.full.national_problem",
    ),
    "calibration": (
        "uk.full.dense",
        "uk.full.size_search",
        "uk.full.size_draw",
        "uk.full.size_refit",
        "uk.full.selected",
        "uk.full.calibrated",
    ),
    "checks": (
        "spine.gates.assembled",
        "spine.gates.transferred",
        "uk.full.gates.preflight",
        "uk.full.gates.calibrated",
        "uk.full.holdout",
    ),
    "export": (
        "uk.full.export.prepare",
        "uk.full.export.readback",
        "uk.full.package",
        "uk.full.national.readback",
    ),
}
_OWNERSHIP_SUFFIXES = ("boundary", "owned", "checkpoint")
_FULL_BUILD_GROUP_OF = {
    node_id: group
    for group, members in _FULL_BUILD_GROUPS.items()
    for node_id in members
}


def uk_spine_stage_ids(spec: CountrySpec | None = None) -> frozenset[str]:
    """Stage node ids of the maintained spine, including its CREATE root."""
    resolved = load_country_spec("uk") if spec is None else spec
    return frozenset(
        {"create_uk_frs", *(stage.stage for stage in _manifest_stages(resolved))}
    )


def uk_operation_group(node_id: str, *, spine_stages: frozenset[str]) -> str:
    """Classify one operation; an ownership helper follows its stage."""
    stem, _, suffix = node_id.rpartition(".")
    base = stem if suffix in _OWNERSHIP_SUFFIXES and stem else node_id
    if base in spine_stages:
        return "enrichment"
    return _FULL_BUILD_GROUP_OF.get(base, "other")


def uk_graph_presentation(
    graph, *, scope: str, sidecar_path: Path | None = None, spec=None
) -> dict:
    """Country-owned labels and grouping; shared code owns contract validation."""
    ids = {node.id for node in graph.nodes}
    spine_stages = uk_spine_stage_ids(spec)
    members = {group: [] for group, _label, _description in UK_GROUPS}
    for node in graph.nodes:
        members[uk_operation_group(node.id, spine_stages=spine_stages)].append(node.id)
    operations = {}
    # Composite descriptions come from the maintained executable stage roster.
    if "create_uk_frs" in ids:
        for item in uk_spine_operation_inventory(graph, spec):
            operations[item["node"]] = {"composite": item}
    scope_record = {"id": scope, "label": scope.replace("_", " ")}
    if "uk.full.spine_checkpoint" in ids and any(
        source.name == "uk_spine" for source in graph.sources
    ):
        refs, missing = checkpoint_references(sidecar_path)
        boundary = {
            "operation": "uk.full.spine_checkpoint",
            "kind": "checkpoint",
            "upstream": refs,
        }
        if missing:
            boundary["missing"] = missing
        scope_record["boundaries"] = [boundary]
    groups = []
    for group, label, description in UK_GROUPS:
        if group == "sources":
            if graph.sources:
                groups.append(
                    {
                        "id": group,
                        "label": label,
                        "description": description,
                        "sources": [source.name for source in graph.sources],
                    }
                )
        elif members[group]:
            groups.append(
                {
                    "id": group,
                    "label": label,
                    "description": description,
                    "operations": members[group],
                }
            )
    return {
        "protocol": PRESENTATION_PROTOCOL,
        "scope": scope_record,
        "groups": groups,
        "operations": operations,
    }


def save_uk_graph_schema(
    graph, directory: Path, *, scope: str, sidecar_path: Path | None = None, spec=None
) -> Path:
    contract = uk_graph_presentation(
        graph, scope=scope, sidecar_path=sidecar_path, spec=spec
    )
    return save_graph_schema(graph, Path(directory), presentation=contract)


def persist_uk_run_evidence(graph, manifest, store, args, *, phase: str) -> Path:
    """Preserve native evidence; the operator separately requests Orrery export."""
    attempt_id = Path(args.attempt_evidence).name
    index_path = save_run_evidence(
        compile_graph(graph),
        manifest,
        store=store,
        directory=Path(args.attempt_evidence) / "graph-evidence",
        attempt_id=attempt_id,
        phase=phase,
        artifact_summaries=UK_SUMMARY_PROVIDERS,
    )
    # The build publisher transacts over files in one directory. Keep the
    # published evidence flat and merged across attempts, while retaining the
    # durable attempt bundle.
    return publish_run_evidence(index_path, Path(args.out))
