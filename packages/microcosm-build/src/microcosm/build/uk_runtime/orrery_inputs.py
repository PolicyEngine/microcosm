"""Write the collector's Orrery inputs for a UK build.

The collector builds the Orrery graph files, with its own copy of the
projection, from three kinds of file: the graph schema (``graph.schema.json``),
the execution overlay (``graph.execution.json``) and each
``upstream-<sha256>.json`` file the schema or the overlay links. Microcosm
writes only those files, so it makes no Orrery document, publication inventory
or receipt.

:func:`finalize_graph` is the build's graph step: it writes the overlay beside
the schema and registers the inputs, with the native evidence files they were
made from, in the build's manifest, so the staged bundle hands them to the
emitter under the interface's Orrery roles
(:func:`microcosm.build.staging_dataset.output_role`).
:func:`stage_graph_evidence` stages a failed build's graph evidence the same
way. ``python -m microcosm.build.uk_runtime.orrery_inputs`` writes the same
inputs into a new directory from a saved schema and, optionally, the build's
recorded evidence index and content store, for a build whose files exist only
on one machine.

Only the graph, declarations, digest-bound native receipts and reviewed
aggregate summaries are eligible. Population files and content-store payloads
are never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import uuid
from argparse import Namespace
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from microcosm.build.artifact_files import file_artifact, materialize_bytes
from microcosm.build.staging_dataset import StagedDatasetDelivery
from microcosm.graph.canonical import canonical_json
from microcosm.graph.errors import GraphRuntimeError
from microcosm.graph.execution_overlay import execution_overlay_bytes
from microcosm.graph.presentation import validate_presentation
from microcosm.graph.schema import validate_graph_schema

from .orrery_contract import public_summary_findings

__all__ = [
    "EVIDENCE_INDEX_NAME",
    "EVIDENCE_MANIFEST_NAME",
    "EXECUTION_OVERLAY_NAME",
    "FAILURE_BUNDLE_DIRECTORY",
    "FAILURE_MARKER_NAME",
    "SCHEMA_NAME",
    "GraphFinalization",
    "UnpublishableSummariesError",
    "finalize_graph",
    "linked_upstream_names",
    "main",
    "recorded_overlay",
    "stage_graph_evidence",
    "write_orrery_inputs",
]

SCHEMA_NAME = "graph.schema.json"
EXECUTION_OVERLAY_NAME = "graph.execution.json"
EVIDENCE_INDEX_NAME = "execution.evidence.json"
EVIDENCE_MANIFEST_NAME = "orrery.evidence-manifest.json"
FAILURE_MARKER_NAME = "orrery.failure.json"
FAILURE_BUNDLE_DIRECTORY = "orrery-failure"
#: The bound the collector's projection applies to a schema or overlay input.
_MAX_INPUT_BYTES = 32 * 1024 * 1024
#: The collector's bound on one upstream file of an ``orrery-inputs`` submission.
_MAX_UPSTREAM_BYTES = 64 * 1024 * 1024
_UPSTREAM_FILE = re.compile(r"upstream-([0-9a-f]{64})\.json")
#: Each recorded run's native evidence files, by their field in the index.
_RUN_EVIDENCE_FIELDS = ("graph", "manifest", "binding", "summaries")
#: Printed when the overlay would carry a target row's figures or a check's
#: failure lines (Vahid's review of #1396, M1).
UNPUBLISHABLE_WARNING = (
    "warning: the graph's recorded summaries carry a target row's figures or "
    "a check's failure lines, so the build hands the collector no Orrery inputs."
)


class GraphEvidenceOutput(TypedDict):
    """A graph input or evidence file registered in the build's output manifest."""

    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class GraphFinalization:
    """The build-directory files the graph step registered, by name."""

    files: tuple[str, ...]


class UnpublishableSummariesError(ValueError):
    """The recorded summaries carry fields the collector must never receive."""


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


def linked_upstream_names(
    schema: Mapping[str, object], execution: object | None = None
) -> list[str]:
    """The ``upstream-<sha256>.json`` files the schema or the overlay links.

    Links come from the schema's producer extensions (the presentation
    contract's scope boundaries and references) and from the overlay's phase
    references, which are what the collector's projection copies into the
    document. A link's fragment is ignored.
    """

    names = {
        url.split("#", 1)[0]
        for url in (*_urls(schema.get("extensions", {})), *_urls(execution))
    }
    return sorted(name for name in names if _UPSTREAM_FILE.fullmatch(name))


def recorded_overlay(
    schema: Mapping[str, object],
    index_path: Path,
    store_path: Path,
    *,
    reference_base: Path,
) -> dict:
    """Collect the execution overlay from a recorded evidence index and its store.

    Raises :class:`UnpublishableSummariesError` when a summary carries a target
    row's estimates or a check's failure lines, which the collector must
    never receive.
    """

    from microcosm.graph import ContentStore
    from microcosm.graph.evidence import collect_execution_evidence, load_run_evidence

    if not Path(store_path).is_dir():
        raise ValueError(f"Evidence store {store_path} does not exist.")
    store = ContentStore(store_path, create=False)
    runs = load_run_evidence(index_path, store=store, reference_base=reference_base)
    execution = collect_execution_evidence(schema, runs=runs, store=store)
    if public_summary_findings(execution["summaries"]):
        raise UnpublishableSummariesError(
            "The recorded summaries carry a target row's figures or a check's "
            "failure lines."
        )
    return execution


def _output(path: Path, published_root: Path) -> GraphEvidenceOutput:
    info = file_artifact(path)
    return {
        "path": str(published_root / path.name),
        "sha256": info["sha256"],
        "bytes": info["size_bytes"],
    }


def _graph_files(args: Namespace, output: Path) -> list[str]:
    """Write the overlay and return every graph file to register, by name."""

    schema_path = output / SCHEMA_NAME
    if not schema_path.is_file():
        return []
    schema = json.loads(schema_path.read_bytes())
    # The collector's projection refuses an invalid contract.
    validate_presentation(schema)
    names = [SCHEMA_NAME]
    execution = None
    index_path = output / EVIDENCE_INDEX_NAME
    if index_path.is_file():
        # Checked before anything is written: summaries the collector must
        # never receive leave the bundle without any Orrery input.
        execution = recorded_overlay(
            schema, index_path, Path(args.graph_store), reference_base=output
        )
        names.append(index_path.name)
        index = json.loads(index_path.read_bytes())
        for run in index["runs"]:
            names.extend(run[field]["path"] for field in _RUN_EVIDENCE_FIELDS)
        materialize_bytes(
            execution_overlay_bytes(execution), output / EXECUTION_OVERLAY_NAME
        )
        names.append(EXECUTION_OVERLAY_NAME)
    for name in linked_upstream_names(schema, execution):
        _read_upstream_file(output, name)
        names.append(name)
    return list(dict.fromkeys(names))


def finalize_graph(
    args: Namespace,
    *,
    manifest: MutableMapping[str, Any] | None,
    published_root: Path | None = None,
) -> GraphFinalization:
    """Write the overlay and register the build's graph inputs and evidence.

    This runs inside the temporary build directory, including on exceptions.
    A dataset ``manifest``, when the build produced one, registers the files
    under ``orrery/``; a failed or refused build, which has none, gets its own
    evidence manifest (``orrery.evidence-manifest.json``), or, when nothing
    is registered, none, and an earlier attempt's is removed from the build
    and published directories. ``published_root``
    is where the registered files will be published, by default the build's
    output directory. A graph failure does not replace the dataset's
    verdict: it registers no graph file and says so.
    """

    output = Path(args.out)
    if published_root is None:
        published_root = Path(getattr(args, "published_out", output))
    try:
        names = _graph_files(args, output)
    except UnpublishableSummariesError:
        print(UNPUBLISHABLE_WARNING, file=sys.stderr, flush=True)
        names = []
    except Exception as error:
        # No exception text: it could carry a path to raw input data.
        print(
            "warning: the build's Orrery inputs were not written "
            f"({type(error).__name__}); the build's result is unchanged.",
            file=sys.stderr,
            flush=True,
        )
        names = []
    if manifest is not None:
        from .rowwise_cli import MANIFEST_FILENAME, json_text

        outputs = manifest.setdefault("outputs", {})
        for name in names:
            outputs[f"orrery/{name}"] = _output(output / name, published_root)
        materialize_bytes(json_text(manifest).encode(), output / MANIFEST_FILENAME)
        marker = output / "build.json"
        if marker.is_file():
            completion = json.loads(marker.read_bytes())
            completion["rowwise_candidate_manifest"] = file_artifact(
                output / MANIFEST_FILENAME
            )
            materialize_bytes(canonical_json(completion), marker)
    elif names:
        # Failed and refused builds have no dataset manifest. Give their graph
        # evidence its own record rather than pretending an H5 was built.
        evidence = {
            "schema_version": 1,
            "build_kind": "uk_graph_evidence",
            "releasable": False,
            "outputs": {name: _output(output / name, published_root) for name in names},
        }
        materialize_bytes(canonical_json(evidence), output / EVIDENCE_MANIFEST_NAME)
    else:
        # Nothing registered: an earlier attempt's record in a reused output
        # directory must not be staged as this attempt's.
        for directory in {output, published_root}:
            (directory / EVIDENCE_MANIFEST_NAME).unlink(missing_ok=True)
    return GraphFinalization(files=tuple(names))


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


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r} in the graph schema.")
        result[key] = value
    return result


def _reject_constant(constant: str) -> None:
    raise ValueError(f"The graph schema holds a non-finite number ({constant}).")


def _read_regular_file(path: Path, label: str, limit: int) -> bytes:
    """Read a file once, refusing links, devices and files over ``limit`` bytes."""

    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise ValueError(
            f"{label} {path} is not a regular file that can be read ({error.strerror})."
        ) from error
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} {path} is not a regular file.")
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"{label} {path} is larger than {limit // 2**20} MiB.")
    return payload


def _read_upstream_file(directory: Path, name: str) -> bytes:
    """An upstream file's bytes, read once and checked against its name's digest."""

    payload = _read_regular_file(
        directory / name, "Upstream graph evidence", _MAX_UPSTREAM_BYTES
    )
    if hashlib.sha256(payload).hexdigest() != _UPSTREAM_FILE.fullmatch(name)[1]:
        raise ValueError(f"Upstream graph evidence digest mismatch: {name}.")
    return payload


def _read_schema(path: Path) -> tuple[bytes, dict[str, object]]:
    """Read a saved schema's bytes, refusing links, devices and oversized files."""

    payload = _read_regular_file(path, "Graph schema", _MAX_INPUT_BYTES)
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    return payload, validate_graph_schema(value)


def _write_new_directory(output: Path, files: Mapping[str, bytes]) -> None:
    """Write every file into a hidden sibling, then move it into place whole."""

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}.partial"
    staging.mkdir()
    try:
        for name, payload in files.items():
            with (staging / name).open("xb") as stream:
                stream.write(payload)
        if os.path.lexists(output):
            raise ValueError(f"Output {output} already exists.")
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def write_orrery_inputs(
    schema_path: Path,
    output: Path,
    *,
    evidence: Path | None = None,
    store: Path | None = None,
) -> tuple[str, ...]:
    """Write the collector's inputs for one saved schema into a new directory.

    ``schema_path`` is a build's ``graph.schema.json``; the upstream files it
    links are read from beside it and must match the digests in their names.
    ``evidence`` (the build's ``execution.evidence.json``) and ``store`` (its
    content store) are given together, and add the execution overlay the
    build would have written, refused like the build's when its summaries
    carry fields the collector must never receive. The schema and upstream
    bytes are copied unchanged. Returns the names written.
    """

    schema_path, requested_output = Path(schema_path), Path(output)
    if (evidence is None) != (store is None):
        raise ValueError("The evidence index and its store are given together.")
    output = requested_output.resolve()
    if os.path.lexists(requested_output) or os.path.lexists(output):
        raise ValueError(f"Output {requested_output} already exists.")
    schema_bytes, schema = _read_schema(schema_path)
    # The collector's projection refuses an invalid contract.
    validate_presentation(schema)
    files = {SCHEMA_NAME: schema_bytes}
    execution = None
    if evidence is not None and store is not None:
        execution = recorded_overlay(
            schema, Path(evidence), Path(store), reference_base=Path(evidence).parent
        )
        files[EXECUTION_OVERLAY_NAME] = execution_overlay_bytes(execution)
    for name in linked_upstream_names(schema, execution):
        files[name] = _read_upstream_file(schema_path.parent, name)
    _write_new_directory(output, files)
    return tuple(sorted(files))


def main(argv: list[str] | None = None) -> int:
    """Write the collector's Orrery inputs for a UK build into a new directory."""

    parser = argparse.ArgumentParser(
        prog="python -m microcosm.build.uk_runtime.orrery_inputs",
        description=main.__doc__,
    )
    parser.add_argument(
        "--schema",
        type=Path,
        required=True,
        help="the build's graph.schema.json; linked upstream files are read beside it",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        help="the build's execution.evidence.json, to add the execution overlay",
    )
    parser.add_argument(
        "--store", type=Path, help="the build's content store, with --evidence"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new directory; it must not exist"
    )
    args = parser.parse_args(argv)
    if (args.evidence is None) != (args.store is None):
        parser.error("--evidence and --store must be given together")
    written_output = args.output.resolve()
    try:
        names = write_orrery_inputs(
            args.schema, args.output, evidence=args.evidence, store=args.store
        )
    except UnpublishableSummariesError as error:
        parser.error(f"{error} The collector must never receive them.")
    except (OSError, TypeError, ValueError, RecursionError, GraphRuntimeError) as error:
        parser.error(str(error))
    print(
        f"Wrote the collector's Orrery inputs to {written_output}: {', '.join(names)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
