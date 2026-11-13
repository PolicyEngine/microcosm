"""Export saved UK execution evidence before staging the build's artifact bundle.

Only the graph, declarations, digest-bound native receipts, and reviewed
aggregate summaries are eligible. Population files and content-store payloads
are never exported.

A build uploads no graph to the runs application. Its staged bundle carries
the Orrery inputs (``graph.schema.json``, ``graph.execution.json`` and the
``upstream-<sha256>.json`` files) under the interface's Orrery roles
(:func:`microcosm.build.staging_dataset.output_role`), and the emitter submits
them to the collector, which builds the Orrery graph files and keeps a
licensed build's private. ``graph.orrery.json`` is still projected, for
viewing the graph offline.
"""

from __future__ import annotations

import json
import re
from argparse import Namespace
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from microcosm_emitter_interface.orrery import (
    PublicationFileRole,
    PublicationReceipt,
)

from microcosm.build.artifact_files import file_artifact, materialize_bytes
from microcosm.build.staging_dataset import StagedDatasetDelivery
from microcosm.build.telemetry_emitter import publication_inventory
from microcosm.graph import ContentStore
from microcosm.graph.canonical import canonical_json
from microcosm.graph.evidence import collect_execution_evidence, load_run_evidence
from microcosm.graph.execution_overlay import execution_overlay_bytes
from microcosm.graph.orrery import orrery_json_from_schema

from .orrery_contract import public_summary_findings

RECEIPT_NAME = "orrery.publication.json"
INVENTORY_NAME = "orrery.upload.json"
EVIDENCE_MANIFEST_NAME = "orrery.evidence-manifest.json"
EXECUTION_OVERLAY_NAME = "graph.execution.json"
FAILURE_MARKER_NAME = "orrery.failure.json"
FAILURE_BUNDLE_DIRECTORY = "orrery-failure"


class GraphEvidenceOutput(TypedDict):
    """A preserved public file registered in the build's output manifest."""

    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class GraphFinalization:
    """The build-directory files written for one graph and its receipt snapshot."""

    files: tuple[str, ...]
    receipt: PublicationReceipt


def _urls(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "url" and isinstance(item, str):
                yield item
            else:
                yield from _urls(item)
    elif isinstance(value, list):
        for item in value:
            yield from _urls(item)


def _output(path: Path, published_root: Path) -> GraphEvidenceOutput:
    info = file_artifact(path)
    return {
        "path": str(published_root / path.name),
        "sha256": info["sha256"],
        "bytes": info["size_bytes"],
    }


def finalize_graph(
    args: Namespace,
    *,
    manifest: MutableMapping[str, Any] | None,
    published_root: Path | None = None,
) -> GraphFinalization:
    """Freeze bytes and a receipt snapshot; retry never rewrites that snapshot.

    This runs inside the temporary build directory, including on exceptions.
    A dataset ``manifest``, when the build produced one, registers the graph
    files and receipt. A graph export failure does not replace a dataset's
    scientific verdict. ``published_root`` is where the registered files will
    be published, by default the build's output directory. Nothing is
    uploaded: the receipt stays ``skipped`` (``local_only``), and the staged
    bundle submits the Orrery inputs to the collector.
    """
    from . import rowwise_staging

    output = Path(args.out)
    if published_root is None:
        published_root = Path(getattr(args, "published_out", output))
    schema_path = output / "graph.schema.json"
    receipt: PublicationReceipt = {
        "version": 1,
        "status": "no_evidence",
        "publication_id": None,
        "error_code": None,
        "url": None,
    }
    names = []
    try:
        if schema_path.is_file():
            schema = json.loads(schema_path.read_bytes())
            index_path = output / "execution.evidence.json"
            execution = None
            roles: dict[str, PublicationFileRole] = {
                schema_path.name: "schema",
                "graph.orrery.json": "graph",
            }
            if index_path.is_file():
                store = ContentStore(args.graph_store, create=False)
                runs = load_run_evidence(index_path, store=store, reference_base=output)
                execution = collect_execution_evidence(schema, runs=runs, store=store)
                # Checked before anything is preserved: without the inventory
                # neither the build nor the emitter's retry can submit them.
                if public_summary_findings(execution["summaries"]):
                    receipt.update(status="failed", error_code="unpublishable_summary")
                    raise RuntimeError("Graph summaries carry unpublishable fields.")
                roles[index_path.name] = "index"
                index = json.loads(index_path.read_bytes())
                for run in index["runs"]:
                    evidence_roles: tuple[tuple[str, PublicationFileRole], ...] = (
                        ("graph", "declaration"),
                        ("manifest", "manifest"),
                        ("binding", "binding"),
                        ("summaries", "summaries"),
                    )
                    for field, role in evidence_roles:
                        roles[run[field]["path"]] = role
            payload = orrery_json_from_schema(schema, execution=execution).encode()
            materialize_bytes(payload, output / "graph.orrery.json")
            # The overlay is the projection's only input besides the schema,
            # so the collector projects the Orrery graph file from these two
            # files and the upstream files. It is a registered build output,
            # which the staged bundle submits under its Orrery role; it is not
            # in the emitter's publication inventory, which nothing uploads.
            unpublished: list[str] = []
            if execution is not None:
                materialize_bytes(
                    execution_overlay_bytes(execution),
                    output / EXECUTION_OVERLAY_NAME,
                )
                unpublished.append(EXECUTION_OVERLAY_NAME)
            for url in _urls(json.loads(payload)):
                name = url.split("#", 1)[0]
                if re.fullmatch(r"upstream-[a-f0-9]{64}\.json", name):
                    if file_artifact(output / name)["sha256"] != name[9:-5]:
                        raise ValueError("Upstream graph evidence digest mismatch.")
                    roles[name] = "upstream"
            # Keep exact sources outside the build's temporary directory and
            # outside HF's immutable receipt snapshot, including when offline.
            names = [*roles, *unpublished]
            preserved = Path(args.attempt_evidence) / "orrery-publication"
            for name in roles:
                materialize_bytes((output / name).read_bytes(), preserved / name)
            emitter = rowwise_staging.active_telemetry_emitter()
            inventory = (
                publication_inventory(preserved, roles)
                if emitter is None
                else emitter.publication_inventory(preserved, roles)
            )
            if inventory is None:
                receipt.update(
                    status="failed",
                    error_code="local_service_unavailable",
                    preserved_directory=str(preserved),
                    recorded_execution=execution is not None,
                )
                raise RuntimeError("Local graph inventory service is unavailable.")
            materialize_bytes(canonical_json(inventory), preserved / INVENTORY_NAME)
            materialize_bytes(canonical_json(inventory), output / INVENTORY_NAME)
            names = [*roles, *unpublished, INVENTORY_NAME]
            # No upload to the runs application: the collector builds the
            # Orrery graph files from the inputs the staged bundle submits.
            receipt = {
                "version": 1,
                "publication_id": inventory["publication_id"],
                "status": "skipped",
                "error_code": "local_only",
                "url": None,
                "preserved_directory": str(preserved),
                "recorded_execution": execution is not None,
            }
    except Exception:
        # No credential, exception text, paths to raw input data, or signed URL
        # is copied from an exception into public evidence or telemetry.
        if receipt["status"] != "failed":
            receipt["error_code"] = "graph_export_or_enqueue_failed"
        receipt["status"] = "failed"
    materialize_bytes(canonical_json(receipt), output / RECEIPT_NAME)
    names.append(RECEIPT_NAME)
    if manifest is not None:
        from .rowwise_cli import MANIFEST_FILENAME, json_text

        outputs = manifest.setdefault("outputs", {})
        for name in names:
            outputs[f"orrery/{name}"] = _output(output / name, published_root)
        manifest["orrery_publication"] = receipt
        materialize_bytes(json_text(manifest).encode(), output / MANIFEST_FILENAME)
        marker = output / "build.json"
        if marker.is_file():
            completion = json.loads(marker.read_bytes())
            completion["rowwise_candidate_manifest"] = file_artifact(
                output / MANIFEST_FILENAME
            )
            materialize_bytes(canonical_json(completion), marker)
    elif names:
        # Failed/refused builds have no dataset manifest. Give their saved
        # evidence its own inventory rather than pretending an H5 was built.
        evidence = {
            "schema_version": 1,
            "build_kind": "uk_graph_evidence",
            "releasable": False,
            "orrery_publication": receipt,
            "outputs": {name: _output(output / name, published_root) for name in names},
        }
        materialize_bytes(canonical_json(evidence), output / EVIDENCE_MANIFEST_NAME)
    return GraphFinalization(files=tuple(names), receipt=receipt)


def stage_graph_evidence(
    args: Namespace, output: Path, *, run_id: str, build_id: str | None = None
) -> StagedDatasetDelivery | None:
    """Stage failure evidence as a bundle, through the build's emitter.

    The hand-over names the build for the collector (``build_id``, the build's
    attempt ID, by default ``run_id``; no candidate release), so the emitter
    submits the evidence's Orrery inputs once the bundle's commit lands.
    """
    from microcosm.build.staging_dataset import (
        StagedDatasetBundle,
        disabled_staged_dataset,
        local_only_staged_dataset,
        stage_bundle,
        write_sidecars,
    )

    from . import rowwise_staging
    from .staging import UK_STAGED_DATASET_PREFIX

    if not (output / EVIDENCE_MANIFEST_NAME).is_file():
        return None
    mode = rowwise_staging.staged_dataset_mode(args)
    if mode == "disabled":
        return disabled_staged_dataset("--no-staging or --no-staged-dataset")
    repository = None if mode == "local_only" else args.staged_dataset_repo_id
    bundle = StagedDatasetBundle.from_manifest(
        output, run_id=run_id, manifest_name=EVIDENCE_MANIFEST_NAME
    )
    write_sidecars(
        bundle, repository=repository, prefix=UK_STAGED_DATASET_PREFIX, telemetry=None
    )
    if mode == "local_only":
        return local_only_staged_dataset(bundle, prefix=UK_STAGED_DATASET_PREFIX)
    assert repository is not None, "Remote staging requires a repository."
    # Nothing in the evidence directory is rewritten after the hand-over.
    return stage_bundle(
        bundle,
        emitter=rowwise_staging.bundle_emitter(),
        repository=repository,
        prefix=UK_STAGED_DATASET_PREFIX,
        collector=rowwise_staging.bundle_provenance({}, build_id=build_id or run_id),
    ).delivery
