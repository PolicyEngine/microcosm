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
It reads that hand-over's receipt first: one that landed is recorded wherever
the directory is now. It hands the directory over as a new job instead only
when a hand-over that has not landed can never finish: its files changed, or a
file it reads is gone from the paths its job names (the directory moved). The
emitter keeps retrying the abandoned job; Microcosm cannot cancel it.

A record that says the bundle landed is never replaced by a failure: a
manifest that records the outputs as handed over is left untouched. Where the
emitter does not hold the build's hand-over (another machine, or another user
or cache directory), a national directory's new hand-over comes back
``REMOTE_DIFFERS`` once the build's bundle is on the Hub, and the emitter
cannot say whether the folder holds this build's bundle; the command then
keeps the manifest's records and says so. A directory whose bytes differ from
the remote bundle is otherwise refused.

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
    AT_BUILD_RECORD_KEY,
    HANDOVER_RECORD_KEY,
    RESTAGE_WAIT_SECONDS,
    RETRYING_ERROR_CODES,
    SHA256SUMS_FILENAME,
    StagedBundleHandOver,
    StagedDatasetBundle,
    StagedDatasetDelivery,
    StagedDatasetError,
    at_build_record,
    delivery_covers_bundle,
    handover_record,
    hub_offline,
    local_only_staged_dataset,
    recorded_handover,
    refresh_sha256sums_entry,
    resume_hand_over,
    stage_bundle,
    validate_staged_dataset_delivery,
    write_access_problem,
    write_sidecars,
)
from microcosm.build.telemetry_emitter import BundleEmitter, start_bundle_emitter
from microcosm.build.uk_runtime.rowwise_cli import (
    BUILD_RECORD_FILENAME,
    SCORE_RECEIPT_FILENAME,
)
from microcosm.build.uk_runtime.rowwise_staging import STAGED_DATASET_PHASES
from microcosm.build.uk_runtime.staging import (
    UK_STAGED_DATASET_PREFIX,
    UK_STAGED_DATASET_REPOSITORY,
)

MANIFEST_FILENAME = "rowwise_candidate_manifest.json"


def _start_bundle_emitter() -> BundleEmitter:
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
    existing = _valid_record(manifest.get("staged_dataset"))
    handed_over = (
        None
        if repository is None
        else recorded_handover(manifest, repository=repository, run_id=run_id)
    )
    if delivery_covers_bundle(
        existing, bundle, repository=repository
    ) or _landed_as_handed_over(existing, handed_over, bundle):
        # Already on the Hub with these very outputs, or with the outputs the
        # build handed over: keep the record (its revision is the bundle's
        # commit) and touch nothing on disk.
        return _nothing_to_do(existing)
    # Whether the record is the outcome of the build's own hand-over.
    resumed = False
    if repository is None:
        if _landed(existing):
            return _keep_landed(existing, "--local-only does not replace it", 0)
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
            staged = None
            if handed_over is not None:
                staged = resume_hand_over(
                    bundle,
                    emitter=emitter,
                    record=handed_over,
                    repository=repository,
                    prefix=UK_STAGED_DATASET_PREFIX,
                    wait_seconds=args.wait_seconds,
                )
                resumed = staged is not None
            if staged is None:
                _write_sidecars(bundle, manifest, repository=repository)
                staged = stage_bundle(
                    bundle,
                    emitter=emitter,
                    repository=repository,
                    prefix=UK_STAGED_DATASET_PREFIX,
                    wait_seconds=args.wait_seconds,
                    earlier_files=None if handed_over is None else handed_over["files"],
                )
        finally:
            emitter.close()
    delivery = staged.delivery
    if _landed(existing) and not _landed(delivery):
        # The landed record stays; the attempt still failed.
        return _keep_landed(existing, f"this attempt ended {_describe(delivery)}", 1)
    if _landed(existing) and _same_bundle(existing, delivery):
        return _nothing_to_do(existing)
    if (
        not resumed
        and delivery["error_code"] == "REMOTE_DIFFERS"
        and handed_over is not None
        and handed_over["files"] != bundle.digests()
    ):
        # The folder may hold the build's own bundle; only the emitter that
        # holds the build's hand-over can say, so nothing here is replaced.
        print(
            f"the manifest keeps its staged dataset record "
            f"({_describe(existing)}) and the build's hand-over record.",
            file=sys.stderr,
            flush=True,
        )
        print(json.dumps(delivery, indent=2, sort_keys=True))
        return 1
    # Keep what the build recorded before replacing its record for the first
    # time (a manifest the build wrote already has it).
    if AT_BUILD_RECORD_KEY not in manifest and existing is not None:
        manifest[AT_BUILD_RECORD_KEY] = at_build_record(existing)
    manifest["staged_dataset"] = validate_staged_dataset_delivery(delivery)
    if _landed(delivery) or delivery["error_code"] in RETRYING_ERROR_CODES:
        # A hand-over the emitter holds replaces the record; one it settled
        # as a failure leaves the build's record for a later re-stage.
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
    _note_unchanged_records(manifest.get(AT_BUILD_RECORD_KEY), delivery)
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


_LANDED = ("uploaded", "already_staged")


def _valid_record(value: Any) -> StagedDatasetDelivery | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return validate_staged_dataset_delivery(value)
    except StagedDatasetError:
        return None


def _landed(record: Mapping[str, Any] | None) -> bool:
    return record is not None and record["status"] in _LANDED


def _same_bundle(existing: Mapping[str, Any], delivery: Mapping[str, Any]) -> bool:
    return all(
        existing[key] == delivery[key]
        for key in ("repository", "prefix", "run_id", "revision", "files")
    )


#: The outputs a national build changes after its hand-over: it rewrites the
#: build record and adds the score receipt.
_CHANGED_AFTER_HAND_OVER = frozenset({BUILD_RECORD_FILENAME, SCORE_RECEIPT_FILENAME})


def _landed_as_handed_over(
    existing: StagedDatasetDelivery | None,
    handed_over: Mapping[str, Any] | None,
    bundle: StagedDatasetBundle,
) -> bool:
    """Whether the record says the bundle the build handed over landed, and the
    directory still holds it.

    A national build rewrites its build record and adds its score receipt
    after the hand-over, so those may differ; every other output must still
    have the digest it was handed over with.
    """
    if (
        not _landed(existing)
        or handed_over is None
        or existing is None
        or existing["repository"] != handed_over["repository"]
        or existing["run_id"] != handed_over["run_id"]
        or existing["files"] != handed_over["files"]
    ):
        return False
    current = bundle.digests()
    names = (set(current) | set(handed_over["files"])) - _CHANGED_AFTER_HAND_OVER
    return all(
        (current.get(name) or {}).get("sha256")
        == (handed_over["files"].get(name) or {}).get("sha256")
        for name in names
    )


def _describe(record: Mapping[str, Any] | None) -> str:
    if record is None:
        return "none"
    code = record["error_code"]
    return record["status"] if code is None else f"{record['status']}, {code}"


def _nothing_to_do(existing: Mapping[str, Any]) -> int:
    print(
        f"already staged at {existing['repository']}/{existing['prefix']} "
        f"(revision {existing['revision']}); nothing to do.",
        file=sys.stderr,
        flush=True,
    )
    print(json.dumps(existing, indent=2, sort_keys=True))
    return 0


def _keep_landed(existing: Mapping[str, Any], why: str, status: int) -> int:
    """A landed record is never replaced by a failure or a local record.

    ``status`` is 0 when nothing was attempted (``--local-only``), and 1 when
    an attempt failed: the directory differs from the bundle that landed, or
    the stage went to another repository.
    """
    print(
        f"warning: the manifest says the bundle landed at {existing['repository']}/"
        f"{existing['prefix']} (revision {existing['revision']}); {why}, so the "
        "record stays.",
        file=sys.stderr,
        flush=True,
    )
    print(json.dumps(existing, indent=2, sort_keys=True))
    return status


def _note_unchanged_records(
    at_build: Mapping[str, Any] | None, delivery: Mapping[str, Any]
) -> None:
    """Say which of the build's records still say its bundle did not land.

    They hold what the build recorded (``staged_dataset_at_build``), never a
    re-stage's later code: the staging run's ``staged_dataset`` artifact and
    its ``dataset_staging`` event, unless the build disabled the staged
    dataset, and the Logbook row's phase.
    """
    if delivery["status"] not in _LANDED or not isinstance(at_build, Mapping):
        return
    status = at_build.get("status")
    if status in _LANDED or status not in STAGED_DATASET_PHASES:
        return
    code = at_build.get("error_code")
    recorded = status if code is None else f"{status} ({code})"
    phase = STAGED_DATASET_PHASES[status]
    staging_run = (
        ""
        if at_build.get("mode") == "disabled"
        else (
            "The build's staging run (its staged_dataset and fit_summary "
            "artifacts, which the dashboard reads, and its dataset_staging event) "
            f"still records {recorded}, and its "
        )
    )
    logbook = f"Logbook row still lists the phase {phase}"
    print(
        f"note: the manifest now records the bundle as {delivery['status']}. "
        + (staging_run + logbook if staging_run else f"The build's {logbook}")
        + ": a completed staging run accepts no changes, and a Logbook row is "
        "identified by the digest of its content and chained to the next row, so "
        "this command cannot change them.",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
