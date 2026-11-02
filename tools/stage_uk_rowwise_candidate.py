"""Stage a finished UK rowwise candidate run on the private dataset repository.

The build driver stages its bundle at the end of every run; this command
covers the other cases: a run whose upload failed or was still queued
(recorded as ``staged_dataset.status == "failed"`` in its manifest), a run
that kept its bundle local, and a run built before the staged-dataset lane
existed. It hands over exactly what the driver would have: every file the
manifest registers as an output, the manifest itself, ``staged_manifest.json``
and ``sha256sums.txt``, which the emitter commits in one commit under
``staged/<run_id>/``. The command starts a short-lived emitter of its own,
with only its Hugging Face module, and waits for its first upload attempt,
for up to three hours by default (``--wait-seconds``). When it stops waiting
it closes its emitter, which abandons an upload still running; the next
emitter started on the machine retries it.

When the manifest names the build's own hand-over
(``staged_dataset_handover``) and this machine's emitter still holds it, the
command records that hand-over's outcome instead of handing over a second
bundle: a national build rewrites its build record after the hand-over, so a
bundle built again from the directory would differ from the one on the Hub.
Re-running on an already staged directory is a no-op; a directory whose bytes
differ from the remote bundle is refused.

Staging is not publication: ``releases/`` and ``latest.json`` are untouched.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from microcosm.build.staging_dataset import (
    HANDOVER_RECORD_KEY,
    RESTAGE_WAIT_SECONDS,
    SHA256SUMS_FILENAME,
    StagedBundleHandOver,
    StagedDatasetBundle,
    delivery_covers_bundle,
    handover_record,
    hub_offline,
    local_only_staged_dataset,
    recorded_handover_id,
    refresh_sha256sums_entry,
    resume_hand_over,
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


def _wait_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number of seconds") from None
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number")
    return seconds


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
    parser.add_argument(
        "--wait-seconds",
        type=_wait_seconds,
        default=RESTAGE_WAIT_SECONDS,
        help=(
            "How long to wait for the emitter's upload attempt before closing it "
            f"(default {RESTAGE_WAIT_SECONDS:g}: a 2.4 GB bundle at 2 Mbit/s). "
            "Closing the emitter abandons an upload still running; the next "
            "emitter started on this machine retries it."
        ),
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
    if repository is None:
        _write_sidecars(bundle, manifest, repository=None)
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
            staged = _resume(bundle, manifest, emitter, repository, args.wait_seconds)
            if staged is None:
                _write_sidecars(bundle, manifest, repository=repository)
                staged = stage_bundle(
                    bundle,
                    emitter=emitter,
                    repository=repository,
                    prefix=UK_STAGED_DATASET_PREFIX,
                    wait_seconds=args.wait_seconds,
                )
        finally:
            emitter.close()
    delivery = staged.delivery
    if _same_landed_record(existing, delivery):
        print(
            f"already staged at {existing['repository']}/{existing['prefix']} "
            f"(revision {existing['revision']}); nothing to do.",
            file=sys.stderr,
            flush=True,
        )
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0
    manifest["staged_dataset"] = validate_staged_dataset_delivery(delivery)
    record = handover_record(staged)
    if record is not None:
        manifest[HANDOVER_RECORD_KEY] = record
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
    _note_unchanged_records(existing, delivery)
    print(json.dumps(delivery, indent=2, sort_keys=True))
    return 0 if delivery["status"] != "failed" else 1


def _write_sidecars(
    bundle: StagedDatasetBundle, manifest: dict, *, repository: str | None
) -> None:
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


def _resume(
    bundle: StagedDatasetBundle,
    manifest: Mapping[str, Any],
    emitter: Any,
    repository: str,
    wait_seconds: float,
) -> StagedBundleHandOver | None:
    """The outcome of the build's own hand-over, if this machine still holds it.

    The run directory's sidecars stay as they were handed over, so they keep
    describing that bundle.
    """
    handover_id = recorded_handover_id(
        manifest, repository=repository, run_id=bundle.run_id
    )
    if handover_id is None:
        return None
    return resume_hand_over(
        bundle,
        emitter=emitter,
        handover_id=handover_id,
        repository=repository,
        prefix=UK_STAGED_DATASET_PREFIX,
        wait_seconds=wait_seconds,
    )


_LANDED = ("uploaded", "already_staged")


def _same_landed_record(existing: Any, delivery: Mapping[str, Any]) -> bool:
    """Whether the manifest already records this landed bundle and commit."""
    if not isinstance(existing, Mapping):
        return False
    try:
        existing = validate_staged_dataset_delivery(existing)
    except ValueError:
        return False
    return (
        existing["status"] in _LANDED
        and delivery["status"] in _LANDED
        and all(
            existing[key] == delivery[key]
            for key in ("repository", "prefix", "run_id", "revision", "files")
        )
    )


def _note_unchanged_records(existing: Any, delivery: Mapping[str, Any]) -> None:
    """Say which of the build's records still say its bundle failed, and why."""
    if delivery["status"] not in _LANDED or not isinstance(existing, Mapping):
        return
    if existing.get("status") != "failed":
        return
    print(
        f"note: the manifest now records the bundle as {delivery['status']}. The "
        "build's staging run (its staged_dataset and fit_summary artifacts, which "
        "the dashboard reads, and its dataset_staging event) and its Logbook row "
        f"still record {existing.get('error_code')}: a completed staging run "
        "accepts no changes, and a Logbook row is identified by the digest of "
        "its content and chained to the next row, so this command cannot change "
        "them.",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
