"""Assemble an annual build of a US national build (microcosm#1570).

An annual projection family is published as a build of the release
``microcosm-us-2024-annual``, into its own folder through the emitter. This
tool writes that build from three inputs:

- ``--base-build-dir``: a local copy of the base national build's folder,
  ``releases/microcosm-us-2024-national/<base build>/``, with every file its
  release manifest declares, its H5 included, under the base build's names;
- ``--candidate-dir``: the directory
  ``microcosm.build.us_annual_static_aging`` wrote for that base build, built
  with ``--base-dataset`` set to the base build's dataset key;
- ``--acceptance``: the separate passing ``us_annual_projection_acceptance``
  report bound to the candidate manifest's SHA-256.

It writes ``<out>/releases/microcosm-us-2024-annual/`` and
``<out>/artifacts/``, neither of which may exist. The release directory holds
the base build's files other than its H5 and its manifests, byte for byte,
plus the candidate manifest (``annual_manifest.json``), the acceptance report
(``annual_acceptance.json``) and every per-year receipt. The artifact root
holds the base-year H5, under the base build's file name, and one
``microcosm_us_<year>.h5`` per projected year; each is a hard link to its
input where the file system allows, and a copy otherwise. Every file is
checked against its SHA-256.

Both manifests are the base build's, with ``release`` and ``build`` changed
and ``migrated_from`` dropped. The release manifest's ``metadata`` gains
``dataset_years``, ``annual_projection_manifest``,
``annual_projection_acceptance`` and ``annual_projection_base``, which names
the base build (``repo_id``, ``release``, ``build``, ``path``, ``sha256``),
and its artifacts gain the added files, each with its role and no revision.
An artifact that is one of the build's manifests or its checksum ledger is
dropped, since the emitter publishes those itself. A checksum ledger in the
base folder keeps its lines for the carried files, gets the rewritten
manifests' digests and gains a line for each added file.

The build ID is ``<base build>-annual-<YYYYMMDDTHHMMSSZ>-<hex8>``: the time
is when the tool ran, in UTC, and ``<hex8>`` is the first eight hex digits of
the candidate manifest's SHA-256, so the ID names the candidate it publishes.

The build is written in a private directory under ``--out`` and validated
there with the release contract; only a build that passes is moved into
place. The summary printed on success gives the build, the release directory
and the two emitter commands: ``publish_command`` publishes the build, and
``promote_command`` makes it the annual release's current build. The annual
release is never the repository's default release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from microcosm.data.build_folder_manifests import publication_commands
from microcosm.data.contract import (
    CHECKSUMS_FILE,
    US_ANNUAL_RELEASE_ID,
    US_NATIONAL_RELEASE_ID,
    release_artifact_role,
    validate_release_dir,
)

_RELEASE_MANIFEST = "release_manifest.json"
_BUILD_MANIFEST = "build_manifest.json"
#: The files the emitter publishes for every build itself.
_PUBLISHER_FILES = frozenset({_RELEASE_MANIFEST, _BUILD_MANIFEST, CHECKSUMS_FILE})
#: The names the candidate manifest and the acceptance report take in the
#: build's folder, and their artifact keys.
_EVIDENCE_FILE = "annual_manifest.json"
_ACCEPTANCE_FILE = "annual_acceptance.json"
_EVIDENCE_KEY = "annual_manifest"
_ACCEPTANCE_KEY = "annual_acceptance"
#: One line of a checksum ledger, as ``sha256sum`` writes it.
_LEDGER_LINE = re.compile(r"(?P<digest>[0-9a-f]{64}) [ *](?P<name>.+)")


def _utc_now() -> datetime:
    """The assembly time (test seam)."""
    return datetime.now(UTC)


def _refuse(message: str) -> SystemExit:
    return SystemExit(f"error: {message}")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise _refuse(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict):
        raise _refuse(f"{path} is not a JSON object")
    return document


def _write_object(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n")


def _bare_name(value: object, what: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or set(value) & {"/", "\\"}
    ):
        raise _refuse(f"{what} must be a bare file name, not {value!r}")
    return value


def _place(source: Path, destination: Path, sha256: str, *, link: bool) -> None:
    """Put ``source`` at ``destination`` and check its bytes.

    An H5 is hard-linked where the file system allows, and copied otherwise;
    every other file is copied.
    """
    if not source.is_file():
        raise _refuse(f"{source} does not exist")
    if destination.exists():
        raise _refuse(f"two inputs are both published as {destination.name}")
    linked = False
    if link:
        try:
            os.link(source.resolve(), destination)
            linked = True
        except OSError:
            linked = False
    if not linked:
        shutil.copyfile(source, destination)
    if _sha256(destination) != sha256:
        raise _refuse(
            f"{source} does not have the sha256 its manifest records ({sha256})"
        )


def _base_build(base_dir: Path) -> tuple[dict, dict, str, str, dict]:
    """The base build's two manifests, its build ID, dataset key and dataset."""
    release_manifest = _read_object(base_dir / _RELEASE_MANIFEST)
    build_manifest = _read_object(base_dir / _BUILD_MANIFEST)
    build = release_manifest.get("build")
    if release_manifest.get("release") != US_NATIONAL_RELEASE_ID or not (
        isinstance(build, str) and build
    ):
        raise _refuse(
            f"{base_dir} is not a build of {US_NATIONAL_RELEASE_ID}: its release "
            f"manifest names release {release_manifest.get('release')!r} and build "
            f"{build!r}. An annual build projects a national build in its folder."
        )
    if (build_manifest.get("release"), build_manifest.get("build")) != (
        US_NATIONAL_RELEASE_ID,
        build,
    ):
        raise _refuse(
            f"{base_dir}/{_BUILD_MANIFEST} does not name release "
            f"{US_NATIONAL_RELEASE_ID} and build {build}"
        )
    metadata = release_manifest.get("metadata")
    if isinstance(metadata, Mapping) and "dataset_years" in metadata:
        raise _refuse(f"base build {build} already declares metadata.dataset_years")
    artifacts = release_manifest.get("artifacts")
    defaults = release_manifest.get("default_datasets")
    base_key = defaults.get("national") if isinstance(defaults, Mapping) else None
    dataset = artifacts.get(base_key) if isinstance(artifacts, Mapping) else None
    if not isinstance(dataset, Mapping) or dataset.get("kind") != "microdata":
        raise _refuse(
            f"base build {build}'s default_datasets.national is not a microdata "
            "artifact"
        )
    return release_manifest, build_manifest, build, base_key, dict(dataset)


def assemble_annual_release_dir(
    *,
    base_build_dir: Path,
    candidate_dir: Path,
    acceptance: Path,
    out: Path,
    assembled_at: datetime,
) -> dict[str, Any]:
    """Write, validate and move into place an annual build; return its summary."""
    base_release, base_build_manifest, base_build, base_key, base_dataset = _base_build(
        base_build_dir
    )
    base_artifacts = base_release["artifacts"]
    repositories = {
        entry.get("repo_id")
        for entry in base_artifacts.values()
        if isinstance(entry, Mapping)
    }
    if len(repositories) != 1 or not isinstance(base_dataset.get("repo_id"), str):
        raise _refuse(
            f"base build {base_build} declares artifacts in more than one repository"
        )
    repo_id = base_dataset["repo_id"]
    base_path = _bare_name(base_dataset.get("path"), f"base build {base_build}'s H5")
    base_sha256 = base_dataset.get("sha256")

    evidence_source = candidate_dir / _EVIDENCE_FILE
    evidence = _read_object(evidence_source)
    evidence_sha256 = _sha256(evidence_source)
    if (
        evidence.get("kind") != "us_annual_static_aging_candidate"
        or evidence.get("status") != "complete"
    ):
        raise _refuse(f"{evidence_source} is not a complete annual candidate manifest")
    base = evidence.get("base")
    base = base if isinstance(base, Mapping) else {}
    if base.get("dataset") != base_key:
        raise _refuse(
            f"the candidate's base dataset is {base.get('dataset')!r}, not base "
            f"build {base_build}'s dataset key {base_key!r}; build the candidate "
            f"again with --base-dataset {base_key}"
        )
    if base.get("parent_release") != base_build:
        raise _refuse(
            f"the candidate's base.parent_release is {base.get('parent_release')!r}, "
            f"not the base build {base_build}"
        )
    if base.get("sha256") != base_sha256:
        raise _refuse(
            f"the candidate's base.sha256 is {base.get('sha256')!r}, but base build "
            f"{base_build}'s H5 has sha256 {base_sha256!r}"
        )
    accepted = _read_object(acceptance)
    if (
        accepted.get("kind") != "us_annual_projection_acceptance"
        or accepted.get("projection_manifest_sha256") != evidence_sha256
    ):
        raise _refuse(
            f"{acceptance} is not an acceptance report for {evidence_source}: it "
            f"must bind its sha256 {evidence_sha256}"
        )
    candidate_metadata = evidence.get("metadata")
    families = (
        candidate_metadata.get("dataset_years")
        if isinstance(candidate_metadata, Mapping)
        else None
    )
    if not isinstance(families, Mapping) or set(families) != {base_key}:
        raise _refuse(
            f"{evidence_source} must declare one dataset_years family, {base_key!r}"
        )
    years = families[base_key]
    candidate_artifacts = evidence.get("artifacts")
    if not isinstance(years, Mapping) or not isinstance(candidate_artifacts, Mapping):
        raise _refuse(f"{evidence_source} has no year map or artifacts")

    build = (
        f"{base_build}-annual-"
        f"{assembled_at.astimezone(UTC):%Y%m%dT%H%M%SZ}-{evidence_sha256[:8]}"
    )
    destination = out / "releases" / US_ANNUAL_RELEASE_ID
    artifacts_destination = out / "artifacts"
    for path in (destination, artifacts_destination):
        if path.exists() or path.is_symlink():
            raise _refuse(f"{path} already exists; assemble into a new --out")

    # The files the annual build adds: (key, kind, source, published name,
    # sha256, whether it is an H5 in the artifact root).
    added: list[tuple[str, str, Path, str, str, bool]] = [
        (
            _EVIDENCE_KEY,
            "diagnostics",
            evidence_source,
            _EVIDENCE_FILE,
            evidence_sha256,
            False,
        ),
        (
            _ACCEPTANCE_KEY,
            "diagnostics",
            acceptance,
            _ACCEPTANCE_FILE,
            _sha256(acceptance),
            False,
        ),
    ]
    for year, key in sorted(years.items()):
        if key == base_key:
            continue
        record = candidate_artifacts.get(key)
        record = record if isinstance(record, Mapping) else {}
        name = _bare_name(record.get("path"), f"the candidate's {year} H5")
        added.append(
            (key, "microdata", candidate_dir / name, name, record.get("sha256"), True)
        )
        receipt = record.get("projection_receipt")
        receipt = receipt if isinstance(receipt, Mapping) else {}
        receipt_name = _bare_name(receipt.get("path"), f"the {year} receipt")
        added.append(
            (
                Path(receipt_name).stem,
                "diagnostics",
                candidate_dir / receipt_name,
                receipt_name,
                receipt.get("sha256"),
                False,
            )
        )
    carried = {
        key: dict(entry)
        for key, entry in base_artifacts.items()
        if isinstance(entry, Mapping) and entry.get("path") not in _PUBLISHER_FILES
    }
    clashes = sorted(
        {key for key, *_ in added} & set(carried)
        | {name for *_, name, _sha, _h5 in added}
        & {entry.get("path") for entry in carried.values()}
    )
    if clashes:
        raise _refuse(f"the candidate's files clash with base build files: {clashes}")

    out.mkdir(parents=True, exist_ok=True)
    releases_created = not destination.parent.exists()
    staging = out / f".assemble-{uuid.uuid4().hex}"
    staged_release = staging / "releases" / US_ANNUAL_RELEASE_ID
    staged_artifacts = staging / "artifacts"
    staged_release.mkdir(parents=True)
    staged_artifacts.mkdir()
    try:
        for entry in carried.values():
            name = _bare_name(entry.get("path"), "a base build artifact")
            h5 = entry.get("kind") == "microdata"
            _place(
                base_build_dir / name,
                (staged_artifacts if h5 else staged_release) / name,
                entry.get("sha256"),
                link=h5,
            )
        new_artifacts = {}
        for key, kind, source, name, sha256, h5 in added:
            _place(
                source,
                (staged_artifacts if h5 else staged_release) / name,
                sha256,
                link=h5,
            )
            new_artifacts[key] = {
                "kind": kind,
                "path": name,
                "repo_id": repo_id,
                "sha256": sha256,
                "role": release_artifact_role(key, kind),
            }
        metadata = dict(base_release.get("metadata") or {})
        metadata.update(
            dataset_years={base_key: dict(years)},
            annual_projection_manifest=_EVIDENCE_KEY,
            annual_projection_acceptance=_ACCEPTANCE_KEY,
            annual_projection_base={
                "repo_id": repo_id,
                "release": US_NATIONAL_RELEASE_ID,
                "build": base_build,
                "path": base_path,
                "sha256": base_sha256,
            },
        )
        release_manifest = {
            key: value
            for key, value in base_release.items()
            if key not in {"migrated_from", "metadata", "artifacts"}
        }
        release_manifest.update(
            release=US_ANNUAL_RELEASE_ID,
            build=build,
            metadata=metadata,
            artifacts={**carried, **new_artifacts},
        )
        build_manifest = {
            key: value
            for key, value in base_build_manifest.items()
            if key != "migrated_from"
        }
        build_manifest.update(release=US_ANNUAL_RELEASE_ID, build=build)
        _write_object(staged_release / _RELEASE_MANIFEST, release_manifest)
        _write_object(staged_release / _BUILD_MANIFEST, build_manifest)
        if (base_build_dir / CHECKSUMS_FILE).is_file():
            _write_ledger(
                base_build_dir / CHECKSUMS_FILE,
                staged_release,
                staged_artifacts,
                added_names=[name for *_, name, _sha, _h5 in added],
            )
        validate_release_dir(staged_release, artifact_root=staged_artifacts)
        # Only a build that passed its contract moves into place. The H5 files
        # land first and the release directory is the commit point: if it
        # cannot move, the H5 files go back.
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged_artifacts.rename(artifacts_destination)
        try:
            staged_release.rename(destination)
        except BaseException:
            artifacts_destination.rename(staged_artifacts)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if (
            releases_created
            and destination.parent.is_dir()
            and not any(destination.parent.iterdir())
        ):
            destination.parent.rmdir()
    return {
        "release": US_ANNUAL_RELEASE_ID,
        "build": build,
        "release_dir": str(destination),
        "artifact_root": str(artifacts_destination),
        "base": {
            "release": US_NATIONAL_RELEASE_ID,
            "build": base_build,
            "dataset": base_path,
            "sha256": base_sha256,
        },
        "candidate_manifest_sha256": evidence_sha256,
        "years": sorted(years),
        **publication_commands(
            destination, repo_id=repo_id, artifact_root=artifacts_destination
        ),
    }


def _write_ledger(
    base_ledger: Path,
    release_dir: Path,
    artifact_root: Path,
    *,
    added_names: list[str],
) -> None:
    """The base build's ledger, refreshed for the annual build.

    A carried file keeps its line, which must still match it; the two
    manifests' lines get their rewritten digests; each added file gets a
    line. A line for a file the annual build does not carry is refused.
    """

    def digest(name: str) -> str:
        local = release_dir / name
        return _sha256(local if local.is_file() else artifact_root / name)

    lines = []
    for line in base_ledger.read_text().splitlines():
        match = _LEDGER_LINE.fullmatch(line)
        if match is None:
            raise _refuse(f"{base_ledger} has a line that is not a checksum: {line!r}")
        name = match["name"]
        if not ((release_dir / name).is_file() or (artifact_root / name).is_file()):
            raise _refuse(
                f"{base_ledger} lists {name}, which the base build's release "
                "manifest does not declare; the annual build carries only the "
                "declared files"
            )
        if name in {_RELEASE_MANIFEST, _BUILD_MANIFEST}:
            lines.append(f"{digest(name)}  {name}")
        elif match["digest"] != digest(name):
            raise _refuse(f"{base_ledger} records another sha256 for {name}")
        else:
            lines.append(line)
    lines.extend(f"{digest(name)}  {name}" for name in added_names)
    (release_dir / CHECKSUMS_FILE).write_text("\n".join(lines) + "\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-build-dir",
        required=True,
        type=Path,
        help=(
            "A local copy of releases/microcosm-us-2024-national/<base build>/, "
            "with its H5."
        ),
    )
    parser.add_argument(
        "--candidate-dir",
        required=True,
        type=Path,
        help="The annual candidate directory built from that base build.",
    )
    parser.add_argument(
        "--acceptance",
        required=True,
        type=Path,
        help="The passing acceptance report bound to the candidate manifest.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Where releases/microcosm-us-2024-annual/ and artifacts/ are written.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = assemble_annual_release_dir(
        base_build_dir=args.base_build_dir,
        candidate_dir=args.candidate_dir,
        acceptance=args.acceptance,
        out=args.out,
        assembled_at=_utc_now(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
