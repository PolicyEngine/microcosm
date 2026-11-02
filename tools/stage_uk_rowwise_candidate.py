"""Stage a finished UK rowwise candidate run on the private dataset repository.

The build driver stages its bundle at the end of every run; this command
covers the other cases: a run whose upload failed or was still queued
(recorded as ``staged_dataset.status == "failed"`` in its manifest), a run
that kept its bundle local, and a run built before the staged-dataset lane
existed. It hands over exactly what the driver would have: every file the
manifest registers as an output, the manifest itself, ``staged_manifest.json``
and ``sha256sums.txt``, which the emitter commits in one commit under
``staged/<run_id>/``. The command starts a short-lived emitter of its own,
with only its Hugging Face module, and waits for its first upload attempt as
a build does. Re-running on an already staged directory is a no-op; a
directory whose bytes differ from the remote bundle is refused.

Staging is not publication: ``releases/`` and ``latest.json`` are untouched.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

from microcosm.build.staging_dataset import (
    SHA256SUMS_FILENAME,
    StagedBundleHandOver,
    StagedDatasetBundle,
    delivery_covers_bundle,
    hub_offline,
    local_only_staged_dataset,
    refresh_sha256sums_entry,
    stage_bundle,
    validate_staged_dataset_delivery,
    write_access_problem,
    write_sidecars,
)
from microcosm.build.telemetry_emitter import start_bundle_emitter
from microcosm.build.uk_runtime.staging import (
    UK_STAGED_DATASET_PREFIX,
    UK_STAGED_DATASET_REPOSITORY,
)

MANIFEST_FILENAME = "rowwise_candidate_manifest.json"


def _start_bundle_emitter():
    """This command's own emitter (test seam)."""

    return start_bundle_emitter()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="A finished build_uk_rowwise_candidate.py output directory.",
    )
    parser.add_argument(
        "--repo-id",
        default=UK_STAGED_DATASET_REPOSITORY.repo_id(os.environ),
        help=(
            "Private Hugging Face dataset repository (default from "
            f"{UK_STAGED_DATASET_REPOSITORY.repo_id_environment_variable})."
        ),
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Staged run id; defaults to the manifest's staged_dataset.run_id, "
            "then the Logbook build id, then the directory name."
        ),
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Verify the bundle and write the sidecars without uploading.",
    )
    return parser.parse_args(argv)


def _default_run_id(run_dir: Path, manifest: dict) -> str:
    staged = manifest.get("staged_dataset")
    if isinstance(staged, dict) and staged.get("run_id"):
        return str(staged["run_id"])
    rows = sorted(glob.glob(str(run_dir / "logbook-spool" / "*.json")))
    for row_path in reversed(rows):
        try:
            row = json.loads(Path(row_path).read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("build_id"):
            return str(row["build_id"])
    return run_dir.name


def _rewrite_manifest(path: Path, manifest: dict) -> None:
    text = json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.staging")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise SystemExit(f"error: {manifest_path} is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = args.run_id or _default_run_id(run_dir, manifest)
    repository = None if args.local_only else str(args.repo_id).strip()
    if repository == "":
        raise SystemExit("error: --repo-id must be non-empty unless --local-only")
    bundle = StagedDatasetBundle.from_manifest(
        run_dir,
        run_id=run_id,
        manifest_name=MANIFEST_FILENAME,
        extra_summary={
            "dataset_households": (manifest.get("parameters") or {}).get(
                "dataset_households"
            ),
            "release_posture": manifest.get("release_posture"),
        },
    )
    existing = manifest.get("staged_dataset")
    if delivery_covers_bundle(existing, bundle, repository=repository):
        # Already on the Hub with these very outputs: keep the driver's record
        # (its revision is the bundle's commit) and touch nothing on disk.
        print(
            f"already staged at {existing['repository']}/{existing['prefix']} "
            f"(revision {existing['revision']}); nothing to do.",
            file=sys.stderr,
            flush=True,
        )
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0
    telemetry = manifest.get("staging_delivery")
    write_sidecars(
        bundle,
        repository=repository,
        prefix=UK_STAGED_DATASET_PREFIX,
        telemetry=(
            None
            if not isinstance(telemetry, dict)
            else {
                "repository": telemetry.get("configured_repository"),
                "prefix": (
                    None
                    if telemetry.get("run_id") is None
                    else f"runs/{telemetry['run_id']}"
                ),
                "mode": telemetry.get("mode"),
            }
        ),
    )
    if repository is None:
        staged = StagedBundleHandOver(
            local_only_staged_dataset(bundle, prefix=UK_STAGED_DATASET_PREFIX)
        )
    else:
        if hub_offline():
            raise SystemExit(
                "error: HF_HUB_OFFLINE is set, so the emitter cannot upload the "
                "bundle; unset it, or pass --local-only."
            )
        print(
            f"staging {run_dir} to {repository} under "
            f"{bundle.remote_prefix(UK_STAGED_DATASET_PREFIX)}...",
            file=sys.stderr,
            flush=True,
        )
        emitter = _start_bundle_emitter()
        try:
            problem = write_access_problem(emitter, repository)
            if problem is not None:
                raise SystemExit(f"error: {problem}.")
            staged = stage_bundle(
                bundle,
                emitter=emitter,
                repository=repository,
                prefix=UK_STAGED_DATASET_PREFIX,
            )
        finally:
            emitter.close()
    delivery = staged.delivery
    manifest["staged_dataset"] = validate_staged_dataset_delivery(delivery)
    # The emitter uploads an uncopied file from disk later; it keeps its bytes.
    if staged.may_rewrite(MANIFEST_FILENAME):
        _rewrite_manifest(manifest_path, manifest)
        if staged.may_rewrite(SHA256SUMS_FILENAME):
            refresh_sha256sums_entry(run_dir, MANIFEST_FILENAME)
    else:
        print(
            f"warning: {MANIFEST_FILENAME} stays as handed over: the emitter "
            "uploads it from disk; the delivery record is printed below.",
            file=sys.stderr,
            flush=True,
        )
    print(json.dumps(delivery, indent=2, sort_keys=True))
    return 0 if delivery["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
