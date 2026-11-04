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

No UK program is registered with the Modal runner yet. A UK build run there
would stage nothing from its container: the runner sets ``HF_HUB_OFFLINE``,
so the build would record its bundle as ``local_only``. Once ``modal volume
get`` has copied such a run back, ``--modal-receipt`` names the stage's
receipt. The receipt must be of a stage that finished, and the run directory
must be inside the run's copied-back state directory; both are checked before
any output is read. The receipt must list the build manifest and every output
it registers, at their paths under the state directory. Every output must have
the receipt's SHA-256 and size. So must the build manifest, unless its
``staged_dataset`` record is ``local_and_remote``, as a hand-over from the
workstation leaves it when it rewrites the manifest. The command reads that mode from the manifest itself,
so an edit that sets it also ends the manifest's comparison. Edits to the
manifest's other fields, including which files the stage wrote it registers
as outputs and under which roles, are then not caught; an output whose bytes
differ from the receipt's is still refused. Otherwise the command refuses before it starts an emitter; when everything
compared matches, it stages the directory as above. The staged manifest and
the checksum file are not compared, because staging writes them again. With
``--local-only`` the command checks the directory and writes those two files,
but leaves the build manifest as it is, so later runs with the receipt can
still compare it.

Staging is not publication: ``releases/`` and ``latest.json`` are untouched.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
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

# The Modal runner's receipt format lives beside this command, in tools/.
_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import modal_us_stage_plan as modal_plan  # noqa: E402

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
        help=(
            "Verify the bundle and write the sidecars (staged_manifest.json and "
            "sha256sums.txt) without uploading. Without --modal-receipt, also "
            "record the bundle as local_only in the build manifest; with it, "
            "leave the build manifest as it is. A directory whose manifest "
            "records a landed bundle is left unchanged."
        ),
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
    parser.add_argument(
        "--modal-receipt",
        type=Path,
        help=(
            "The receipt of the Modal stage that wrote --run-dir, as "
            "`modal volume get` copies it back "
            "(modal-runs/<run_id>/receipts/<stage>-<utc>.json). Every output the "
            "manifest registers must be in it with the same SHA-256 and size, "
            "and so must the build manifest unless its staged_dataset record "
            "is local_and_remote."
        ),
    )
    parser.add_argument(
        "--modal-state-root",
        type=Path,
        help=(
            "The copied-back state directory the receipt describes (default: "
            "the state directory beside the receipt's directory)."
        ),
    )
    args = parser.parse_args(argv)
    if args.modal_state_root is not None and args.modal_receipt is None:
        parser.error("--modal-state-root needs --modal-receipt")
    return args


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
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except ValueError:
        raise SystemExit(f"error: {manifest_path} is not valid JSON") from None
    if not isinstance(manifest, dict):
        raise SystemExit(f"error: {manifest_path} must be a JSON object")
    run_id = args.run_id or _default_run_id(run_dir, manifest)
    repository = None if args.local_only else str(args.repo_id).strip()
    if repository == "":
        raise SystemExit("error: --repo-id must be non-empty unless --local-only")
    copied_back = (
        None
        if args.modal_receipt is None
        else _copied_back_stage(
            args.modal_receipt,
            state_root=args.modal_state_root,
            run_dir=run_dir,
            manifest=manifest,
            manifest_bytes=manifest_bytes,
        )
    )
    try:
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
    except StagedDatasetError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if copied_back is not None:
        problems = copied_back.output_problems(bundle)
        if problems:
            raise _copy_back_refusal(run_dir, problems)
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
        local_only = local_only_staged_dataset(bundle, prefix=UK_STAGED_DATASET_PREFIX)
        if copied_back is not None:
            return _leave_the_copied_back_manifest(local_only)
        staged = StagedBundleHandOver(local_only)
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


class _CopiedBackStage:
    """A finished Modal stage's receipt, and where the run directory sits in it.

    ``relative`` is the run directory's path under the copied-back state
    directory, where the receipt lists each file by its path. (A plain class:
    the tests load this command without registering it as a module, which a
    dataclass needs.)
    """

    def __init__(self, receipt: Mapping[str, Any], relative: PurePosixPath) -> None:
        self.receipt = receipt
        self.relative = relative

    def path(self, name: str) -> str:
        return (self.relative / name).as_posix()

    def listed(self) -> dict[str, Mapping[str, Any]]:
        return {
            str(item.get("path")): item
            for item in self.receipt.get("outputs") or ()
            if isinstance(item, Mapping)
        }

    def output_problems(self, bundle: StagedDatasetBundle) -> list[str]:
        """Outputs whose measured SHA-256 or size differ from the receipt's.

        ``bundle`` has already measured each output; the receipt lists every
        one of them (``_copied_back_stage`` checked).
        """
        listed = self.listed()
        problems = []
        for output in bundle.files:
            path = self.path(output.name)
            entry = listed.get(path, {})
            if (
                entry.get("sha256") != output.sha256
                or entry.get("bytes") != output.bytes
            ):
                problems.append(
                    f"{path} is not the file the Modal stage wrote: its SHA-256 or "
                    "size differs from the receipt's"
                )
        return problems


def _copied_back_stage(
    receipt_path: Path,
    *,
    state_root: Path | None,
    run_dir: Path,
    manifest: Mapping[str, Any],
    manifest_bytes: bytes,
) -> _CopiedBackStage:
    """Read the receipt and check what can be checked without reading outputs.

    The receipt must be of a stage that finished, and ``run_dir`` (resolved)
    must be inside the stage's state directory as copied back: by default
    ``state/`` beside the receipt's ``receipts/`` directory, as ``modal volume
    get`` leaves them, resolved as ``--modal-state-root`` is. The receipt must
    list the build manifest and every output it registers; the build manifest
    must also have the receipt's bytes unless its ``staged_dataset`` record is
    ``local_and_remote`` (:func:`_record_is_the_builds`). The command refuses
    otherwise, before any output is hashed.
    """

    receipt_path = receipt_path.expanduser()
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _copy_back_refusal(
            run_dir, [f"cannot read the Modal receipt {receipt_path}: {exc}"]
        ) from None
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") != modal_plan.RECEIPT_SCHEMA
    ):
        raise _copy_back_refusal(
            run_dir, [f"{receipt_path} is not a {modal_plan.RECEIPT_SCHEMA} receipt"]
        )
    problems = modal_plan.receipt_status_problems(receipt)
    root = (
        receipt_path.resolve().parent.parent / "state"
        if state_root is None
        else state_root.expanduser()
    ).resolve()
    try:
        relative = PurePosixPath(*run_dir.relative_to(root).parts)
    except ValueError:
        problems.append(
            f"{run_dir} is not inside the Modal run's state directory {root} "
            "(pass --modal-state-root)"
        )
        raise _copy_back_refusal(run_dir, problems) from None
    stage = _CopiedBackStage(receipt, relative)
    listed = stage.listed()
    manifest_path = stage.path(MANIFEST_FILENAME)
    manifest_entry = listed.get(manifest_path)
    if manifest_entry is None:
        problems.append(f"the receipt lists no {manifest_path}")
    elif _record_is_the_builds(manifest) and (
        manifest_entry.get("sha256") != hashlib.sha256(manifest_bytes).hexdigest()
        or manifest_entry.get("bytes") != len(manifest_bytes)
    ):
        problems.append(
            f"{manifest_path} is not the build manifest the Modal stage wrote: its "
            "SHA-256 or size differs from the receipt's"
        )
    not_mirrored = set(receipt.get("outputs_not_mirrored") or ())
    outputs = manifest.get("outputs")
    for entry in outputs.values() if isinstance(outputs, Mapping) else ():
        recorded = entry.get("path") if isinstance(entry, Mapping) else None
        name = PurePosixPath(str(recorded or "")).name
        if not name:
            continue  # The bundle refuses it.
        path = stage.path(name)
        if path in not_mirrored and not (run_dir / name).is_file():
            problems.append(
                f"{path} was hashed in the Modal container but never copied to the "
                "runs volume (the receipt's outputs_not_mirrored), so no copy-back "
                "holds it"
            )
        elif path not in listed:
            problems.append(f"the receipt lists no {path}")
    if problems:
        raise _copy_back_refusal(run_dir, problems)
    return stage


def _record_is_the_builds(manifest: Mapping[str, Any]) -> bool:
    """Whether the build manifest is to be compared with the receipt.

    A build run with ``HF_HUB_OFFLINE``, as the Modal runner runs every tool,
    records its bundle as ``local_only`` (``disabled`` with staging off), so
    any record but a ``local_and_remote`` one counts as the build's own. A
    hand-over from the workstation replaces it with a ``local_and_remote``
    record when the emitter copied the manifest, rewriting it; a manifest the
    emitter uploads from disk stays as handed over, with the receipt's bytes.
    ``--local-only`` with ``--modal-receipt`` leaves the manifest as it is
    (:func:`_leave_the_copied_back_manifest`).

    This reads only ``staged_dataset.mode``, from the manifest itself, so a
    manifest edited to say ``local_and_remote`` is not compared either. No
    condition read from the manifest can tell such an edit from a record a
    hand-over wrote; every output it registers is still compared.
    """

    record = manifest.get("staged_dataset")
    return not isinstance(record, Mapping) or record.get("mode") != "local_and_remote"


def _leave_the_copied_back_manifest(local_only: Mapping[str, Any]) -> int:
    """``--local-only`` with ``--modal-receipt``: the sidecars are written, and
    the build manifest stays as it is.

    The ``local_only`` record differs from the manifest's whenever the build
    disabled its staged dataset, ``--run-id`` names another run id, a
    national build changed its outputs after recording its staged dataset,
    or a hand-over replaced the record. Writing it would leave a manifest
    that still counts as the build's (:func:`_record_is_the_builds`) without
    the receipt's bytes, which every later run with the receipt refuses.
    """
    print(
        "the build manifest stays as it is, so later runs with --modal-receipt "
        "can still compare it with the receipt; staged_manifest.json and "
        f"{SHA256SUMS_FILENAME} are written. The local_only record of the bundle, "
        "which the manifest does not get:",
        file=sys.stderr,
        flush=True,
    )
    print(json.dumps(local_only, indent=2, sort_keys=True))
    return 0


def _copy_back_refusal(run_dir: Path, problems: list[str]) -> SystemExit:
    return SystemExit(
        f"error: {run_dir} does not hold what the Modal stage wrote, so nothing is "
        "staged:\n" + "\n".join(f"  - {problem}" for problem in problems)
    )


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
