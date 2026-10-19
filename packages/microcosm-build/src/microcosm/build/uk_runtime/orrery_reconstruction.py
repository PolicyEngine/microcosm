"""Rebuild a declaration-only Orrery publication for a UK build before #1144.

UK dense and national builds from before microcosm#1144 wrote their graph
declaration (``graph.json``) and their code pin, but no run binding, so their
recorded execution cannot be validated. This module projects the recorded
declaration alone into the normal publication layout, marked as reconstructed
by :mod:`microcosm.graph.reconstruction`. Only the declaration's bytes come
from the build: the schema's compiled sections and the Orrery document are
computed by the code running this module, whose revision the marker records
beside the build's. It reads only the build directory, executes no kernel and
uploads nothing: the publication directory is submitted later, if at all,
through the usual publication path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from microcosm.build.staging_dataset import SHA256SUMS_FILENAME, parse_sha256sums
from microcosm.graph.canonical import canonical_json
from microcosm.graph.orrery import orrery_json_from_schema
from microcosm.graph.reconstruction import (
    DECLARATION_ONLY,
    DECLARATION_ROLE,
    GIT_COMMIT_PATTERN,
    code_revision,
    reconstructed_schema,
    reconstruction_marker,
)

from .orrery_publication import INVENTORY_NAME
from .rowwise_cli import MANIFEST_FILENAME, json_text

__all__ = [
    "GRAPH_NAME",
    "REPORT_NAME",
    "REPORT_PROTOCOL",
    "SCHEMA_NAME",
    "main",
    "reconstruct_uk_publication",
    "summary_text",
]

DECLARATION_NAME = "graph.json"
COMPLETION_NAME = "build.json"
SCHEMA_NAME = "graph.schema.json"
EVIDENCE_INDEX_NAME = "execution.evidence.json"
GRAPH_NAME = "graph.orrery.json"
REPORT_NAME = "orrery.reconstruction.json"
REPORT_PROTOCOL = "microcosm.uk.orrery-reconstruction-report.v1"
PUBLICATION_ID_PREFIX = "reconstructed-"
# The bound the Orrery command applies to a declaration or schema input.
_MAX_INPUT_BYTES = 32 * 1024 * 1024
# Every build file the reconstruction reads, with its role in the marker.
_INPUT_ROLES = {
    DECLARATION_NAME: DECLARATION_ROLE,
    COMPLETION_NAME: "completion_record",
    MANIFEST_FILENAME: "code_pin",
    SHA256SUMS_FILENAME: "digest_list",
}
_ABSENT_REASONS = {
    COMPLETION_NAME: (
        "completion record; without it the candidate manifest's recorded "
        "digest cannot be checked"
    ),
    MANIFEST_FILENAME: "candidate manifest, where the build recorded its code revision",
    SHA256SUMS_FILENAME: "digest list, written only when the build staged its bundle",
}
# Evidence a pre-#1144 build left that a declaration-only reconstruction finds
# but deliberately does not bind to operations.
_UNUSED_REASONS = {
    ".graph-store": (
        "content store; with no run binding there is nothing to verify it against"
    ),
    "evidence-index.json": (
        "index of materialized artifacts; not linked to operations in a "
        "declaration-only reconstruction"
    ),
    "operations.json": (
        "operation inventory written beside the declaration; not needed to project it"
    ),
}
_RUN_MANIFEST_SUFFIX = ".graph.json"
_RUN_MANIFEST_REASON = (
    "run manifest; its receipts are not bound to operations in a "
    "declaration-only reconstruction"
)
_BUILD_FIELDS = ("build_kind", "release_role", "release_id", "build_id", "created_at")
# The note the UK drivers wrote beside the candidate manifest's digest in
# build.json. The close step then appended the staging receipts
# (``staging_delivery``, ``staged_dataset``) to the manifest and refreshed only
# sha256sums.txt, so the digest beside the note is of the manifest as first
# written. ``_refresh_completion`` (cb0d39045, merged with #1057 on 2026-09-30)
# reissues build.json over the final bytes and removes the note; dense builds
# closed by the driver merged with #901 (2026-09-29) until then kept it.
STAGING_RECEIPTS_NOTE = "staging receipts are appended after publication"


@dataclass(frozen=True)
class _RecordedDigest:
    """One digest a build file recorded for another file."""

    recorder: str
    sha256: str
    size_bytes: int | None = None
    # Set when the recorder says the digest was taken before the file was
    # rewritten, so a mismatch is expected rather than evidence of tampering.
    stale_note: str | None = None


def _refuse(message: str) -> NoReturn:
    raise ValueError(f"UK Orrery reconstruction: {message}.")


def _require(condition: bool, message: str) -> None:
    if not condition:
        _refuse(message)


def _read_input(directory: Path, name: str) -> bytes:
    """Read one regular input file, refusing links, devices and oversized files."""

    path = directory / name
    _require(not path.is_symlink(), f"{name} must be a regular file, not a link")
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    )
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode), f"{name} must be a regular file")
        _require(info.st_size <= _MAX_INPUT_BYTES, f"{name} exceeds the input limit")
        payload = stream.read(_MAX_INPUT_BYTES + 1)
    _require(len(payload) <= _MAX_INPUT_BYTES, f"{name} exceeds the input limit")
    return payload


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(constant: str) -> NoReturn:
    _refuse(f"non-finite JSON number {constant}")


def _json_object(payload: bytes, name: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"UK Orrery reconstruction: {name} is not valid JSON."
        ) from error
    _require(type(value) is dict, f"{name} must contain a JSON object")
    return value


def _recorded_digests(
    payloads: Mapping[str, bytes], records: Mapping[str, dict[str, object]]
) -> dict[str, list[_RecordedDigest]]:
    """Digests the build itself recorded, by the name of the file they cover."""

    recorded: dict[str, list[_RecordedDigest]] = {}
    if SHA256SUMS_FILENAME in payloads:
        listing = payloads[SHA256SUMS_FILENAME].decode("utf-8")
        for digest, name in parse_sha256sums(listing):
            recorded.setdefault(name, []).append(
                _RecordedDigest(SHA256SUMS_FILENAME, digest)
            )
    completion = records.get(COMPLETION_NAME, {})
    manifest_record = completion.get("rowwise_candidate_manifest")
    if isinstance(manifest_record, dict) and "sha256" in manifest_record:
        _require(
            manifest_record.get("filename", MANIFEST_FILENAME) == MANIFEST_FILENAME,
            f"{COMPLETION_NAME} records an unexpected candidate manifest name",
        )
        size = manifest_record.get("size_bytes")
        note = manifest_record.get("note")
        recorded.setdefault(MANIFEST_FILENAME, []).append(
            _RecordedDigest(
                COMPLETION_NAME,
                str(manifest_record["sha256"]),
                size if type(size) is int else None,
                note if note == STAGING_RECEIPTS_NOTE else None,
            )
        )
    return recorded


def _verification(
    name: str, payload: bytes, records: Sequence[_RecordedDigest]
) -> tuple[str, list[dict[str, object]]]:
    """Check ``payload`` against its recorded digests: ``(state, stale records)``.

    A mismatch is refused, except against a record that says it was taken
    before the file was rewritten. Such a record is skipped and reported, and
    the file counts as verified only if another recorded digest matches it.
    """

    digest, size = hashlib.sha256(payload).hexdigest(), len(payload)
    matched, skipped = [], []
    for record in records:
        if record.sha256 == digest and record.size_bytes in (None, size):
            matched.append(record.recorder)
        elif record.stale_note is not None:
            skipped.append(record)
        else:
            _refuse(f"{name} does not match the digest recorded in {record.recorder}")
    stale = [
        {
            "name": name,
            "recorded_in": record.recorder,
            "sha256": record.sha256,
            "bytes": record.size_bytes,
            "note": record.stale_note,
            "checked_against": matched,
        }
        for record in skipped
    ]
    if not records:
        return "no_recorded_digest", stale
    if not skipped:
        return "recorded_digest_matched", stale
    return (
        "fallback_digest_matched" if matched else "only_stale_digest_recorded"
    ), stale


def _recorded_code(manifest: Mapping[str, object]) -> tuple[str | None, bool | None]:
    """The code pin a candidate manifest records, top level and under identity."""

    top_level = {
        key: manifest[key] for key in ("git_commit", "git_dirty") if key in manifest
    }
    identity = manifest.get("identity")
    code = identity.get("code") if isinstance(identity, dict) else None
    nested = (
        {key: code[key] for key in ("git_commit", "git_dirty") if key in code}
        if isinstance(code, dict)
        else {}
    )
    _require(
        all(top_level[key] == nested[key] for key in top_level.keys() & nested.keys()),
        f"{MANIFEST_FILENAME} records two different code revisions",
    )
    pin = {**nested, **top_level}
    commit, dirty = pin.get("git_commit"), pin.get("git_dirty")
    _require(
        commit is None
        or (type(commit) is str and GIT_COMMIT_PATTERN.fullmatch(commit) is not None),
        f"{MANIFEST_FILENAME} records a malformed code revision",
    )
    _require(
        type(dirty) in (bool, type(None)),
        f"{MANIFEST_FILENAME} records a malformed working-tree flag",
    )
    return commit, dirty  # type: ignore[return-value]


def _stale_digest_warning(stale: Mapping[str, Any]) -> str:
    """Name a recorded digest skipped as stale, and what checked the file instead."""

    skipped = (
        f"{stale['recorded_in']} records the digest {stale['name']} had before the "
        "staging receipts were appended, as its note says"
    )
    if stale["checked_against"]:
        return (
            f"{skipped}, so {stale['name']} was checked against "
            f"{', '.join(stale['checked_against'])} instead."
        )
    return (
        f"{skipped}, and the build recorded no later digest, so nothing the build "
        f"recorded checks {stale['name']}."
    )


def _source_revision(
    build: Path, recorded: tuple[str | None, bool | None], revision: str | None
) -> tuple[str, bool | None, str]:
    """Resolve ``(git_commit, git_dirty, origin)`` from the build and the caller."""

    if revision is not None:
        _require(
            GIT_COMMIT_PATTERN.fullmatch(revision) is not None,
            "--revision must be a full lowercase git commit",
        )
    recorded_commit, recorded_dirty = recorded
    if recorded_commit is not None:
        _require(
            revision in (None, recorded_commit),
            f"--revision {revision} does not match the recorded revision "
            f"{recorded_commit}",
        )
        return recorded_commit, recorded_dirty, "recorded"
    _require(
        revision is not None,
        f"{build} records no code revision; pass --revision with the commit the "
        "build ran from",
    )
    assert revision is not None
    return revision, None, "caller_asserted"


def _build_identity(*records: Mapping[str, object]) -> dict[str, str]:
    """Identifiers the build recorded, from the first record that names each."""

    identity = {}
    for field in _BUILD_FIELDS:
        value = next((record[field] for record in records if field in record), None)
        if type(value) is str:
            identity[field] = value
    return identity


def _within(directory: Path, candidate: Path) -> bool:
    """Whether ``candidate`` is ``directory`` or lies beneath it, links resolved.

    Each existing ancestor of the resolved path is compared with ``directory``
    as a file, so ``..``, symbolic links and case-insensitive file systems
    cannot disguise the build directory.
    """

    resolved = candidate.resolve()
    return any(
        ancestor.exists() and os.path.samefile(ancestor, directory)
        for ancestor in (resolved, *resolved.parents)
    )


def _unused_evidence(build: Path) -> list[dict[str, str]]:
    found = []
    for name in sorted(os.listdir(build)):
        if name in _UNUSED_REASONS:
            found.append({"name": name, "reason": _UNUSED_REASONS[name]})
        elif name.endswith(_RUN_MANIFEST_SUFFIX):
            found.append({"name": name, "reason": _RUN_MANIFEST_REASON})
    return found


def _published_file(name: str, role: str, payload: bytes) -> dict[str, object]:
    return {
        "name": name,
        "role": role,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_publication(output: Path, files: Mapping[str, bytes]) -> None:
    """Write every file into a hidden sibling, then move it into place whole."""

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}.partial"
    staging.mkdir()
    try:
        for name, payload in files.items():
            with (staging / name).open("xb") as stream:
                stream.write(payload)
        _require(not os.path.lexists(output), f"output {output} already exists")
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def reconstruct_uk_publication(
    build: Path, output: Path, *, revision: str | None = None
) -> dict[str, object]:
    """Write a declaration-only publication for ``build`` into a new ``output``.

    ``build`` is a UK build's output directory from before run bindings; it is
    only read. ``revision`` must equal the code revision the build recorded,
    or supplies one (marked as caller-asserted) when the build recorded none.
    The returned summary report is also written as ``orrery.reconstruction.json``
    beside the publication files, outside the upload inventory.
    """

    build, requested_output = Path(build), Path(output)
    # Check and write the same resolved path. Otherwise a path that enters a
    # missing folder inside the build and leaves it again with ``..`` passes
    # the check, and creating its parents makes that folder in the build.
    output = requested_output.resolve()
    _require(build.is_dir(), f"build directory {build} does not exist")
    _require(
        not _within(build, output),
        f"output {requested_output} is inside the build directory {build}, which "
        "is only read",
    )
    _require(
        not os.path.lexists(requested_output) and not os.path.lexists(output),
        f"output {requested_output} already exists",
    )
    recorded_evidence = [
        name
        for name in (SCHEMA_NAME, EVIDENCE_INDEX_NAME)
        if os.path.lexists(build / name)
    ]
    _require(
        not recorded_evidence,
        f"{build} records Orrery evidence ({', '.join(recorded_evidence)}); export "
        "it with python -m microcosm.graph.orrery instead",
    )
    present = [name for name in sorted(_INPUT_ROLES) if os.path.lexists(build / name)]
    _require(
        DECLARATION_NAME in present,
        f"{build} has no recorded graph declaration ({DECLARATION_NAME}); staged "
        "bundles and builds from before the UK graph driver never wrote one. "
        + (f"Found {', '.join(present)}" if present else "No build record found"),
    )
    payloads = {name: _read_input(build, name) for name in present}
    records = {
        name: _json_object(payloads[name], name)
        for name in (COMPLETION_NAME, MANIFEST_FILENAME)
        if name in payloads
    }
    recorded = _recorded_digests(payloads, records)
    verification: dict[str, str] = {}
    stale_digests: list[dict[str, object]] = []
    for name, payload in payloads.items():
        verification[name], stale_records = _verification(
            name, payload, recorded.get(name, [])
        )
        stale_digests.extend(stale_records)
    manifest = records.get(MANIFEST_FILENAME, {})
    git_commit, git_dirty, origin = _source_revision(
        build, _recorded_code(manifest), revision
    )
    # The code computing the schema's sections and the Orrery projection,
    # identified from this module's file rather than the working directory.
    reconstructing = code_revision(Path(__file__))
    marker = reconstruction_marker(
        git_commit=git_commit,
        git_dirty=git_dirty,
        revision_origin=origin,
        reconstruction_revision=reconstructing,
        inputs=[
            {
                "name": name,
                "role": _INPUT_ROLES[name],
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "verification": verification[name],
            }
            for name, payload in payloads.items()
        ],
    )
    schema = reconstructed_schema(payloads[DECLARATION_NAME], marker)
    country = str(schema["country"])
    schema_bytes = canonical_json(schema)
    graph_bytes = orrery_json_from_schema(
        schema,
        title=f"{country.upper()} · reconstructed declaration ({git_commit[:12]})",
    ).encode("utf-8")
    published = [
        _published_file(GRAPH_NAME, "graph", graph_bytes),
        _published_file(SCHEMA_NAME, "schema", schema_bytes),
    ]
    # Derived from the published bytes, so reconstructing the same build twice
    # yields the same publication rather than a second one.
    publication_id = (
        PUBLICATION_ID_PREFIX
        + hashlib.sha256(canonical_json(published)).hexdigest()[:32]
    )
    inventory = {"version": 1, "publication_id": publication_id, "files": published}
    warnings = []
    if origin == "caller_asserted":
        warnings.append(
            "The build records no code revision; the revision was supplied with "
            "--revision and cannot be checked against the build."
        )
    elif git_dirty is True:
        warnings.append(
            "The build ran from a working tree with uncommitted changes, so its "
            "recorded revision does not fully identify the code that declared "
            "the graph."
        )
    elif git_dirty is None:
        warnings.append("The build did not record whether its working tree was clean.")
    if reconstructing["git_commit"] is None:
        warnings.append(
            "The code running this reconstruction is not a file tracked by a git "
            "checkout, so the marker records no reconstruction revision."
        )
    elif reconstructing["git_dirty"] is None:
        warnings.append(
            "git could not say whether the working tree running this "
            "reconstruction was clean."
        )
    elif reconstructing["git_dirty"]:
        warnings.append(
            "This reconstruction ran from a working tree with uncommitted changes, "
            "so its reconstruction revision does not fully identify the code that "
            "compiled the schema and projected the Orrery document."
        )
    warnings.extend(_stale_digest_warning(stale) for stale in stale_digests)
    if verification[DECLARATION_NAME] == "no_recorded_digest":
        warnings.append(
            f"The build recorded no digest for {DECLARATION_NAME}; it is bound only "
            "by the digest in the reconstruction marker."
        )
    graph: Any = schema["graph"]
    report: dict[str, object] = {
        "protocol": REPORT_PROTOCOL,
        "status": "reconstructed",
        "basis": DECLARATION_ONLY,
        "publication_id": publication_id,
        "published_files": [item["name"] for item in published] + [INVENTORY_NAME],
        "source_revision": marker["source_revision"],
        "reconstruction_revision": marker["reconstruction_revision"],
        "build": _build_identity(manifest, records.get(COMPLETION_NAME, {})),
        "declaration": {
            "name": DECLARATION_NAME,
            "country": country,
            "graph_sha256": schema["graph_sha256"],
            "operations": len(graph["nodes"]),
            "sources": len(graph["sources"]),
        },
        "inputs": marker["inputs"],
        "stale_recorded_digests": stale_digests,
        "missing": [
            {"name": name, "reason": _ABSENT_REASONS[name]}
            for name in sorted(_ABSENT_REASONS)
            if name not in payloads
        ],
        "unused_evidence": _unused_evidence(build),
        "warnings": warnings,
    }
    _write_publication(
        output,
        {
            SCHEMA_NAME: schema_bytes,
            GRAPH_NAME: graph_bytes,
            INVENTORY_NAME: canonical_json(inventory),
            REPORT_NAME: json_text(report).encode("utf-8"),
        },
    )
    return report


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def summary_text(report: Mapping[str, Any], output: Path) -> str:
    """A short human-readable account of one reconstruction."""

    revision, declaration = report["source_revision"], report["declaration"]
    origin = (
        "recorded by the build"
        if revision["origin"] == "recorded"
        else "asserted with --revision"
    )
    reconstructing = report["reconstruction_revision"]
    if reconstructing["git_commit"] is None:
        reconstructed_by = "unknown revision"
    else:
        tree = {False: "clean", True: "uncommitted changes"}.get(
            reconstructing["git_dirty"], "working tree not checked"
        )
        reconstructed_by = f"{reconstructing['git_commit']} ({tree})"
    lines = [
        f"Wrote a reconstructed Orrery publication (declaration only) to {output}",
        f"  revision: {revision['git_commit']} ({origin})",
        f"  reconstructed by: {reconstructed_by}",
        f"  declaration: {declaration['name']}, "
        f"{_count(declaration['operations'], 'operation')}, "
        f"{_count(declaration['sources'], 'source')}",
        "  inputs: "
        + ", ".join(
            f"{item['name']} ({item['verification'].replace('_', ' ')})"
            for item in report["inputs"]
        ),
    ]
    for label, key in (("missing", "missing"), ("found, not bound", "unused_evidence")):
        if report[key]:
            names = ", ".join(item["name"] for item in report[key])
            lines.append(f"  {label}: {names}")
    lines.extend(f"  warning: {warning}" for warning in report["warnings"])
    lines.append(f"  report: {output / REPORT_NAME}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write a declaration-only Orrery publication for a UK build before #1144."""

    parser = argparse.ArgumentParser(
        prog="python -m microcosm.build.uk_runtime.orrery_reconstruction",
        description=main.__doc__,
    )
    parser.add_argument(
        "--build",
        type=Path,
        required=True,
        help="the past build's output directory, which is only read",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "new publication directory, outside the build directory; it must not exist"
        ),
    )
    parser.add_argument(
        "--revision",
        help=(
            "full git commit the build ran from; it must match the revision the "
            "build recorded, or supplies one when the build recorded none"
        ),
    )
    args = parser.parse_args(argv)
    try:
        report = reconstruct_uk_publication(
            args.build, args.output, revision=args.revision
        )
    except (OSError, ValueError, RecursionError) as error:
        parser.error(str(error))
    print(summary_text(report, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
