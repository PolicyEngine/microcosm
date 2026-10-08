"""UK inputs to the shared presentation and execution-evidence contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from microcosm.calibrate.artifacts import RESULT_TYPE, decode_problem
from microcosm.calibrate.graph_evidence import (
    CALIBRATION_SUMMARY_PROVIDERS,
    result_summary,
)
from microcosm.graph import compile_graph, graph_schema, save_run_evidence
from microcosm.graph.canonical import canonical_json
from microcosm.graph.evidence import ArtifactSummaryContext, read_json, sha256
from microcosm.graph.presentation import PRESENTATION_EXTENSION, PRESENTATION_PROTOCOL

from .graph import uk_spine_operation_inventory
from .graph_calibration import SIZE_DRAW_TYPE, SIZE_RECEIPT_TYPE, SIZE_SEARCH_TYPE
from .graph_national import NATIONAL_TARGET_TYPE
from .graph_targets import TARGET_SELECTION_TYPE, registry_from_payload


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


UK_SUMMARY_PROVIDERS = {
    **CALIBRATION_SUMMARY_PROVIDERS,
    (RESULT_TYPE.name, RESULT_TYPE.schema_version): _uk_result_summary,
    (SIZE_SEARCH_TYPE.name, SIZE_SEARCH_TYPE.schema_version): _size_summary,
    (SIZE_DRAW_TYPE.name, SIZE_DRAW_TYPE.schema_version): _size_summary,
    (SIZE_RECEIPT_TYPE.name, SIZE_RECEIPT_TYPE.schema_version): _size_summary,
}


def checkpoint_references(sidecar_path: Path | None) -> tuple[list[dict], str | None]:
    """Use only recorded checkpoint lineage, never today's reconstructed spine."""
    if sidecar_path is None or not sidecar_path.is_file():
        return [], "The checkpoint has no recorded producing graph and manifest."
    sidecar, _ = read_json(sidecar_path)
    refs = []
    missing = []
    recorded_files = [
        ("graph_declaration", "Producing graph"),
        ("graph_manifest", "Producing run manifest"),
    ]
    if "graph_schema" in sidecar:
        recorded_files.append(("graph_schema", "Producing compiler schema"))
    for name, label in recorded_files:
        record = sidecar.get(name)
        if (
            not isinstance(record, dict)
            or not record.get("path")
            or not record.get("sha256")
        ):
            missing.append(label)
            continue
        path = Path(record["path"])
        if not path.is_absolute():
            path = sidecar_path.parent / path
        if not path.is_file():
            missing.append(f"{label} bytes unavailable")
        elif sha256(read_json(path)[1]) != record["sha256"]:
            raise ValueError(f"Recorded checkpoint {label.lower()} digest mismatch.")
        refs.append(
            {"label": label, "url": str(path.resolve()), "sha256": record["sha256"]}
        )
    return refs, "; ".join(missing) if missing else None


def uk_graph_presentation(
    graph, *, scope: str, sidecar_path: Path | None = None, spec=None
) -> dict:
    """Country-owned labels and grouping; shared code owns contract validation."""
    groups = {
        key: []
        for key in (
            "enrichment",
            "geography",
            "targets",
            "calibration",
            "checks",
            "export",
        )
    }
    operations = {}
    ids = {node.id for node in graph.nodes}
    # Composite descriptions come from the maintained executable stage roster.
    if "create_uk_frs" in ids:
        for item in uk_spine_operation_inventory(graph, spec):
            operations[item["node"]] = {"composite": item}
    for node in graph.nodes:
        if "gates" in node.id or "readback" in node.id or "certification" in node.id:
            group = "checks"
        elif node.id.startswith("uk.full.export") or node.id == "uk.full.package":
            group = "export"
        elif node.id.startswith(
            ("uk.full.dense", "uk.full.size", "uk.full.selected", "uk.full.calibrated")
        ):
            group = "calibration"
        elif "target" in node.id or "problem" in node.id:
            group = "targets"
        elif node.id.startswith("uk.full.") and any(
            part in node.id for part in ("geograph", "clone", "support", "atomic")
        ):
            group = "geography"
        else:
            group = "enrichment"
        groups[group].append(node.id)
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
    return {
        "protocol": PRESENTATION_PROTOCOL,
        "scope": scope_record,
        "groups": [
            {
                "id": "sources",
                "label": "Sources",
                "sources": [source.name for source in graph.sources],
            },
            *(
                {"id": key, "label": key.capitalize(), "operations": members}
                for key, members in groups.items()
                if members
            ),
        ],
        "operations": operations,
    }


def save_uk_graph_schema(
    graph, directory: Path, *, scope: str, sidecar_path: Path | None = None, spec=None
) -> Path:
    contract = uk_graph_presentation(
        graph, scope=scope, sidecar_path=sidecar_path, spec=spec
    )
    # Copy only explicitly recorded graph/manifest bytes, so links survive the
    # temporary build directory's publication and the HTML can sit beside them.
    for boundary in contract["scope"].get("boundaries", []):
        for ref in boundary["upstream"]:
            source = Path(ref["url"])
            if source.is_file():
                relative = Path(f"upstream-{ref['sha256']}.json")
                destination = Path(directory) / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                payload = read_json(source)[1]
                if sha256(payload) != ref["sha256"]:
                    raise ValueError("Checkpoint reference changed during capture.")
                destination.write_bytes(payload)
                ref["url"] = relative.as_posix()
    schema = graph_schema(
        compile_graph(graph), extensions={PRESENTATION_EXTENSION: contract}
    )
    path = Path(directory) / "graph.schema.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(schema))
    return path


def persist_uk_run_evidence(graph, manifest, store, args, *, phase: str) -> Path:
    """Preserve native evidence; the operator separately requests Orrery export."""
    attempt_id = Path(args.attempt_evidence).name
    durable = Path(args.attempt_evidence) / "graph-evidence"
    index_path = save_run_evidence(
        compile_graph(graph),
        manifest,
        store=store,
        directory=durable,
        attempt_id=attempt_id,
        phase=phase,
        artifact_summaries=UK_SUMMARY_PROVIDERS,
    )
    # The build publisher transacts over files in one directory. Keep the
    # published evidence flat, while retaining the durable attempt bundle.
    index, _ = read_json(index_path)
    for entry in index["runs"]:
        for field in ("graph", "manifest", "binding", "summaries"):
            relative = Path(entry[field]["path"])
            filename = f"evidence-{attempt_id}-{relative.parent.name}-{field}.json"
            payload = read_json(durable / relative)[1]
            if sha256(payload) != entry[field]["sha256"]:
                raise ValueError("Native phase evidence changed during capture.")
            (args.out / filename).write_bytes(payload)
            entry[field]["path"] = filename
    destination = args.out / "execution.evidence.json"
    destination.write_bytes(canonical_json(index))
    return destination
