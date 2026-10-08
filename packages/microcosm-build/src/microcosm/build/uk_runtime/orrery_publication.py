"""Export saved UK execution evidence before staging the build's artifact bundle.

Only the graph, declarations, digest-bound native receipts, and reviewed
aggregate summaries are eligible. Population files and content-store payloads
are never uploaded to the public visualization service.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from microcosm.build.artifact_files import file_artifact, materialize_bytes
from microcosm.build.telemetry_emitter_service.graph_publication import (
    publication_inventory,
)
from microcosm.graph import ContentStore
from microcosm.graph.canonical import canonical_json
from microcosm.graph.evidence import collect_execution_evidence, load_run_evidence
from microcosm.graph.orrery import orrery_json_from_schema

RECEIPT_NAME = "orrery.publication.json"
INVENTORY_NAME = "orrery.upload.json"
EVIDENCE_MANIFEST_NAME = "orrery.evidence-manifest.json"


def publication_enabled(args) -> bool:
    explicit = getattr(args, "publish_orrery", None)
    if explicit is not None:
        return explicit
    return not (
        getattr(args, "no_staging", False)
        or getattr(args, "staging_local_only", False)
        or getattr(args, "no_staged_dataset", False)
    )


def _urls(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "url" and isinstance(item, str):
                yield item
            else:
                yield from _urls(item)
    elif isinstance(value, list):
        for item in value:
            yield from _urls(item)


def _output(path, published_root):
    info = file_artifact(path)
    return {
        "path": str(published_root / path.name),
        "sha256": info["sha256"],
        "bytes": info["size_bytes"],
    }


def finalize_graph(args, record: dict) -> dict:
    """Freeze bytes and a receipt snapshot; retry never rewrites that snapshot.

    This runs inside the temporary build directory, including on exceptions.
    Graph publication failure does not replace a dataset's scientific verdict.
    """
    from . import rowwise_staging

    output = Path(args.out)
    published_root = Path(getattr(args, "published_out", output))
    schema_path = output / "graph.schema.json"
    receipt = {
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
            roles = {schema_path.name: "schema", "graph.orrery.json": "graph"}
            if index_path.is_file():
                store = ContentStore(args.graph_store, create=False)
                runs = load_run_evidence(index_path, store=store, reference_base=output)
                execution = collect_execution_evidence(schema, runs=runs, store=store)
                roles[index_path.name] = "index"
                index = json.loads(index_path.read_bytes())
                for run in index["runs"]:
                    for field, role in (
                        ("graph", "declaration"),
                        ("manifest", "manifest"),
                        ("binding", "binding"),
                        ("summaries", "summaries"),
                    ):
                        roles[run[field]["path"]] = role
            payload = orrery_json_from_schema(schema, execution=execution).encode()
            materialize_bytes(payload, output / "graph.orrery.json")
            for url in _urls(json.loads(payload)):
                name = url.split("#", 1)[0]
                if re.fullmatch(r"upstream-[a-f0-9]{64}\.json", name):
                    if file_artifact(output / name)["sha256"] != name[9:-5]:
                        raise ValueError("Upstream graph evidence digest mismatch.")
                    roles[name] = "upstream"
            # Keep exact sources outside the build's temporary directory and
            # outside HF's immutable receipt snapshot, including when offline.
            names = list(roles)
            preserved = Path(args.attempt_evidence) / "orrery-publication"
            for name in roles:
                materialize_bytes((output / name).read_bytes(), preserved / name)
            inventory = publication_inventory(preserved, roles)
            materialize_bytes(canonical_json(inventory), preserved / INVENTORY_NAME)
            materialize_bytes(canonical_json(inventory), output / INVENTORY_NAME)
            names = [*roles, INVENTORY_NAME]
            receipt = {
                "version": 1,
                "publication_id": inventory["publication_id"],
                "status": "skipped",
                "error_code": "local_only",
                "url": None,
                "preserved_directory": str(preserved),
                "recorded_execution": execution is not None,
            }
            if publication_enabled(args):
                emitter = rowwise_staging._ACTIVE_EMITTER
                if emitter is None:
                    # Durable enqueue is still possible without a running
                    # emitter; the retry command requires no telemetry run.
                    from microcosm.build.telemetry_emitter import _cache_dir
                    from microcosm.build.telemetry_emitter_constants import (
                        TELEMETRY_SPOOL_FILENAME,
                    )
                    from microcosm.build.telemetry_emitter_service.graph_publication import (
                        GraphPublicationQueue,
                    )

                    queue = GraphPublicationQueue(
                        _cache_dir() / TELEMETRY_SPOOL_FILENAME
                    )
                    queue.enqueue(preserved, inventory)
                    receipt.update(queue.receipt(inventory["publication_id"]))
                else:
                    receipt.update(emitter.publish_graph(preserved, inventory))
    except Exception:
        # No credential, exception text, paths to raw input data, or signed URL
        # is copied from an exception into public evidence or telemetry.
        receipt.update(status="failed", error_code="graph_export_or_enqueue_failed")
    materialize_bytes(canonical_json(receipt), output / RECEIPT_NAME)
    names.append(RECEIPT_NAME)
    record["orrery_files"] = names
    record["orrery_publication"] = receipt
    manifest = record.get("manifest")
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
    return receipt


def stage_graph_evidence(args, output: Path, *, run_id: str) -> dict | None:
    """Stage failure evidence through the existing generic HF bundle adapter."""
    from microcosm.build.staging_dataset import (
        StagedDatasetBundle,
        disabled_staged_dataset,
        local_only_staged_dataset,
        stage_bundle,
        write_sidecars,
    )
    from microcosm.build.staging_storage import HuggingFaceDatasetStorage

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
    return stage_bundle(
        bundle,
        storage=HuggingFaceDatasetStorage(repository, api=rowwise_staging._hub_api()),
        prefix=UK_STAGED_DATASET_PREFIX,
    )
