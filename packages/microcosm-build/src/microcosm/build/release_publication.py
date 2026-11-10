"""Publish a release build into its own folder through the emitter.

A release whose builds can publish into their own folders (the UK
national, dense and local-area releases, which
``microcosm.data.contract.publishes_to_build_folder`` names) is handed to the
separately installed emitter with the interface's ``publish_release`` call.
Microcosm decides what the build contains and checks it with the release
contract; the emitter's release writer puts the files in
``releases/<release>/<build>/`` and records the build in the repository's
``releases.json`` in one commit, refusing a change that breaks the shared
rules in ``microcosm_emitter_interface.releases``. Nothing here commits,
tags or writes a pointer file.

A repository's switch to ``releases.json`` deletes its ``latest*.json``
pointer files in the same commit, and the emitter changes no current build
while one exists. So the command lists the repository's files first and
refuses, before any emitter starts, while any pointer file is there: in a
repository without ``releases.json`` it says the switch has not run, since
the loader follows those files and the switch reads them, and in one with
it, that the file appeared after the switch and should be removed. The
tagged publisher refuses these releases, and any repository that has
``releases.json``.

A build is current only when the emitter's receipt says so: ``status`` is
``published`` and ``current`` is the build. A publication with
``make_current`` can land without making its build current, when a change to
the release's current build requested after it landed first.

The emitter finds its Hugging Face credential the way a build's emitter
does: it is this command's child process and inherits its environment
(``HF_TOKEN``, then ``HUGGINGFACE_TOKEN``, then the token saved by
``hf auth login``). It inherits ``HF_HUB_OFFLINE`` and
``TRANSFORMERS_OFFLINE`` too, so with either set the command refuses before
it starts one, and it hands nothing over until the emitter's write-access
check passes, as the staged-bundle hand-over does: otherwise the change
would wait, pending, until the emitter dropped it an hour later. The
command's own listing of the repository's files uses ``huggingface_hub`` in
this process, with the credential the emitter reads. Publication sends no Slack alert;
the dashboard announces changes to ``releases.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from microcosm_emitter_interface import PromoteBuild, PublishRelease, ReleaseFile
from microcosm_emitter_interface.publication import (
    ReleaseChange,
    ReleaseReceipt,
    unavailable_release_receipt,
)
from microcosm_emitter_interface.releases import is_pointer_file

from microcosm.build.staging_dataset import hub_offline, write_access_problem
from microcosm.build.telemetry_emitter import start_bundle_emitter
from microcosm.data.contract import publishes_to_build_folder
from microcosm.data.publish_cli import publication_refusal
from microcosm.data.release import (
    TAGGED_RELEASE_ADVICE,
    BuildFolderRelease,
    build_folder_identity,
    prepare_build_folder_release,
)
from microcosm.data.releases_record import RELEASES_RECORD_PATH

if TYPE_CHECKING:
    from microcosm_emitter_interface import EmitterStartupError, RequestFailure
    from microcosm_emitter_interface.bundles import WriteAccess

__all__ = [
    "PROMOTION_WAIT_SECONDS",
    "PUBLICATION_WAIT_SECONDS",
    "ReleaseEmitter",
    "ReleasePublication",
    "main",
    "promote_release_build",
    "publish_release_build",
]

#: How long the command waits for a publication to land or be refused. The
#: emitter uploads a build's files in its one commit, and the largest UK
#: build recorded (about 2.4 GB) takes about 16 minutes at 20 Mbit/s; the
#: emitter refuses a change still pending an hour after it was handed over
#: (``REQUEST_EXPIRED``), so the wait covers that hour and one long upload.
PUBLICATION_WAIT_SECONDS = 2 * 3600.0
#: How long the command waits for a promotion, which changes only
#: ``releases.json``.
PROMOTION_WAIT_SECONDS = 300.0
#: What the emitter's refusal codes mean for the operator, where its own
#: message does not say what to do next.
_REFUSAL_ADVICE = {
    "POINTER_FILES_PRESENT": (
        "A latest*.json pointer file appeared in the repository after this "
        "command checked it. Nothing writes those files once a repository "
        "records its builds in releases.json, and the emitter changes no "
        "current build while one exists, so find the commit that added it, "
        "remove it, then run the command again; nothing was changed."
    ),
    "BUILD_FOLDER_EXISTS": (
        "Another build with this name was published already, with a different "
        "release manifest; builds are never changed after publishing."
    ),
    "NEWER_CHANGE_LANDED": (
        "A change to the release's current build requested after this one "
        "landed first; run the command again if this build should still be "
        "current."
    ),
    "REQUEST_EXPIRED": (
        "The change was still waiting an hour after it was handed over, so "
        "the emitter dropped it; run the command again."
    ),
}
#: Why the command refuses with ``HF_HUB_OFFLINE`` or ``TRANSFORMERS_OFFLINE``
#: set: the emitter it starts inherits the variable.
_OFFLINE_REFUSAL = (
    "HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE is set, so the emitter this command "
    "starts could not reach the Hugging Face Hub, and the change would stay "
    "pending until the emitter dropped it an hour later; unset it and run the "
    "command again"
)


class ReleaseEmitter(Protocol):
    """The interface ``EmitterClient``'s release calls and state."""

    startup_error: EmitterStartupError | None
    last_failure: RequestFailure | None

    @property
    def available(self) -> bool: ...

    @property
    def modules(self) -> frozenset[str]: ...

    def check_write_access(self, repository: str) -> WriteAccess: ...

    def publish_release(
        self, request: PublishRelease, *, wait_seconds: float = 300.0
    ) -> ReleaseReceipt: ...

    def promote_build(
        self, request: PromoteBuild, *, wait_seconds: float = 60.0
    ) -> ReleaseReceipt: ...

    def close(self) -> None: ...


#: Starts the short-lived emitter a publication uses (only its Hugging Face
#: module); tests replace it.
_start_release_emitter = start_bundle_emitter


@dataclass(frozen=True)
class ReleasePublication:
    """What became of one release change: the emitter's receipt, read.

    ``wanted_current`` says whether the change asked for the build to become
    current: a promotion, or a publication with ``make_current``.
    ``problem`` says why nothing was handed over, when the emitter could not
    take the change or its write-access check refused it.
    """

    repository: str
    release: str
    build: str
    wanted_current: bool
    receipt: ReleaseReceipt
    problem: str | None = None

    @property
    def published(self) -> bool:
        """Whether the build is listed: the change landed, or had already."""
        return self.receipt["status"] == "published"

    @property
    def current(self) -> bool:
        """Whether the build is the release's current build after the change."""
        return self.published and self.receipt.get("current") == self.build

    @property
    def succeeded(self) -> bool:
        """Published, and current when that was asked for."""
        return self.published and (self.current or not self.wanted_current)

    def explanation(self) -> str:
        """A sentence or two on the outcome, for the operator."""
        status, code = self.receipt["status"], self.receipt.get("error_code")
        if self.problem is not None:
            return f"Nothing was handed over: {self.problem}."
        if status == "unavailable":
            return "The emitter did not answer; nothing was handed over."
        if status == "pending":
            return (
                f"The emitter is still trying ({code}); it keeps the change "
                "queued for an hour after the hand-over. Run the same command "
                "again later to see whether it landed."
            )
        if status == "refused":
            advice = _REFUSAL_ADVICE.get(str(code))
            reason = self.receipt.get("message") or code
            return f"The emitter refused it ({code}): {reason}" + (
                f" {advice}" if advice else ""
            )
        if self.wanted_current and not self.current:
            current = self.receipt.get("current")
            note = self.receipt.get("message")
            return (
                f"Build {self.build} is published but is not current: the "
                f"release's current build is {current}." + (f" {note}" if note else "")
            )
        return f"Build {self.build} of {self.release} is published" + (
            " and current." if self.current else "."
        )

    def as_json(self) -> dict[str, Any]:
        """The outcome as the command prints it."""
        return {
            "repository": self.repository,
            "release": self.release,
            "build": self.build,
            "status": self.receipt["status"],
            "current": self.current,
            "current_build": self.receipt.get("current"),
            "error_code": self.receipt.get("error_code"),
            "message": self.receipt.get("message"),
            "commit": self.receipt.get("commit"),
            "handover_id": self.receipt.get("handover_id"),
        }


def _hand_over(
    change: ReleaseChange,
    *,
    client: ReleaseEmitter | None,
    wait_seconds: float,
    wanted_current: bool,
) -> ReleasePublication:
    owned = client is None
    emitter: ReleaseEmitter = _start_release_emitter() if owned else client
    try:
        # The staged-bundle hand-over's check: the emitter is running with its
        # Hugging Face module, and its credential can write the repository.
        problem = write_access_problem(
            emitter,
            change.repository,
            action="the release change",
            emitter_task="the emitter, which commits it",
            proof="the change itself",
        )
        if problem is not None:
            receipt = unavailable_release_receipt(change)
        elif isinstance(change, PublishRelease):
            receipt = emitter.publish_release(change, wait_seconds=wait_seconds)
        else:
            receipt = emitter.promote_build(change, wait_seconds=wait_seconds)
    finally:
        if owned:
            emitter.close()
    return ReleasePublication(
        repository=change.repository,
        release=change.release,
        build=change.build,
        wanted_current=wanted_current,
        receipt=receipt,
        problem=problem,
    )


def _refuse_offline() -> None:
    """Refuse before an emitter starts when it could not reach the Hub."""
    if hub_offline():
        raise ValueError(_OFFLINE_REFUSAL)


def _repository_files(repository: str) -> list[str]:
    """Every file path on the dataset repository's ``main`` (test seam).

    It reads with the credential the emitter uses: ``HF_TOKEN``, then
    ``HUGGINGFACE_TOKEN``, then (``huggingface_hub``'s own order) the saved
    login. ``huggingface_hub`` does not read ``HUGGINGFACE_TOKEN`` itself.
    """
    import huggingface_hub

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None
    api = huggingface_hub.HfApi(token=token)
    return list(api.list_repo_files(repository, repo_type="dataset"))


def _listing_errors() -> tuple[type[BaseException], ...]:
    """The failures a listing raises: the Hub's, offline mode's and httpx's.

    huggingface_hub's HTTP and offline errors are ``OSError``; httpx's
    transport errors, such as a refused connection, are not.
    """
    try:
        import httpx
    except ImportError:  # pragma: no cover - huggingface_hub 1.x requires it
        return (OSError, ValueError)
    return (OSError, ValueError, httpx.HTTPError)


def _refuse_pointer_files(repository: str) -> None:
    """Refuse a repository that has any ``latest*.json`` pointer file.

    The interface's ``is_pointer_file`` rule says which files those are, and
    the emitter applies it too. A repository's switch to ``releases.json``
    deletes them, and nothing writes them after it. Without ``releases.json``
    the repository has not switched: the loader follows its pointer files
    and the switch reads them, so the refusal says to run the switch and
    keeps them. With it, a pointer file appeared after the switch, and the
    refusal says to remove it. A repository whose files cannot be listed is
    refused too.
    """
    try:
        paths = _repository_files(repository)
    except _listing_errors() as error:
        raise ValueError(
            f"could not list the files of {repository} to check for "
            f"latest*.json pointer files ({type(error).__name__}: {error}); "
            "check the Hugging Face credential (HF_TOKEN or `hf auth login`) "
            "and run the command again"
        ) from error
    pointer_files = sorted(path for path in paths if is_pointer_file(path))
    if pointer_files and RELEASES_RECORD_PATH not in paths:
        raise ValueError(
            f"{repository} has not switched to releases.json: it has no "
            f"releases.json and has the pointer file(s) {', '.join(pointer_files)}, "
            "which the loader follows and the switch reads. Its builds publish "
            "here only after the switch, which the `copy` and then `switch` "
            "steps of `python -m microcosm.build.build_folder_switch` make with "
            "--write; keep the files in place and run the command again once "
            "the switch has run"
        )
    if pointer_files:
        raise ValueError(
            f"{repository} has the pointer file(s) {', '.join(pointer_files)}. "
            "Its builds publish here only once its switch to releases.json has "
            "deleted them, and nothing writes them after it, so remove the "
            "file(s) that appeared, then run the command again"
        )


def publication_request(
    prepared: BuildFolderRelease, *, make_current: bool
) -> PublishRelease:
    """The ``publish_release`` request for a checked build."""
    return PublishRelease(
        repository=prepared.repository,
        release=prepared.release,
        build=prepared.build,
        release_role=prepared.release_role,
        make_current=make_current,
        files=[
            ReleaseFile(
                name=item.name,
                role=item.role,
                local_path=item.path,
                sha256=item.sha256,
                bytes=item.bytes,
            )
            for item in prepared.files
        ],
    )


def publish_release_build(
    release_dir: Path | str,
    *,
    repository: str | None = None,
    artifact_root: Path | str | None = None,
    make_current: bool = False,
    client: ReleaseEmitter | None = None,
    wait_seconds: float = PUBLICATION_WAIT_SECONDS,
) -> ReleasePublication:
    """Check a build and hand it to the emitter to publish into its folder.

    The build is checked first (:func:`prepare_build_folder_release`), so a
    build that breaks its release contract is never handed over. Without a
    ``client``, a short-lived emitter is started and closed afterwards; its
    queue is the one builds on this machine share, so a change still pending
    when it closes is retried by the next emitter started here. It refuses
    (``ValueError``) before anything else when ``HF_HUB_OFFLINE`` or
    ``TRANSFORMERS_OFFLINE`` is set, then, before the build is checked or any
    emitter starts, when the repository still has ``latest*.json`` pointer
    files; it hands nothing over unless the emitter's write-access check
    passes.
    """
    if client is None:
        _refuse_offline()
    named_repository, _, _ = build_folder_identity(release_dir)
    _refuse_pointer_files(named_repository)
    prepared = prepare_build_folder_release(
        release_dir, artifact_root=artifact_root, repository=repository
    )
    return _hand_over(
        publication_request(prepared, make_current=make_current),
        client=client,
        wait_seconds=wait_seconds,
        wanted_current=make_current,
    )


def promote_release_build(
    release_dir: Path | str,
    *,
    repository: str | None = None,
    client: ReleaseEmitter | None = None,
    wait_seconds: float = PROMOTION_WAIT_SECONDS,
) -> ReleasePublication:
    """Make a published build, named by its local manifest, the current build.

    It refuses and checks write access as :func:`publish_release_build` does.
    """
    if client is None:
        _refuse_offline()
    named_repository, release, build = build_folder_identity(release_dir)
    if repository is not None and repository != named_repository:
        raise ValueError(
            f"release {release!r} names {named_repository} in its artifacts, not "
            f"{repository}."
        )
    _refuse_pointer_files(named_repository)
    return _hand_over(
        PromoteBuild(repository=named_repository, release=release, build=build),
        client=client,
        wait_seconds=wait_seconds,
        wanted_current=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microcosm-publish-release-build",
        description=(
            "Publish a build of a release whose builds publish into their own "
            "folders (the UK national, dense and local-area releases) through "
            "the emitter, or make a published build current. " + TAGGED_RELEASE_ADVICE
        ),
    )
    parser.add_argument("release_dir", type=Path, help="The local release directory.")
    parser.add_argument(
        "--repo-id",
        help=(
            "The Hugging Face dataset repository; it must be the one the "
            "release manifest's artifacts name, which is the default."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Where artifacts not in the release directory are, such as the H5.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--make-current",
        action="store_true",
        help="Make the build the release's current build when it is published.",
    )
    action.add_argument(
        "--promote",
        action="store_true",
        help="Make this already published build the release's current build.",
    )
    action.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check the build and list what would be published; start no emitter.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        help=(
            "How long to wait for the change to land or be refused (default "
            f"{PUBLICATION_WAIT_SECONDS:.0f} for a publication, "
            f"{PROMOTION_WAIT_SECONDS:.0f} for a promotion)."
        ),
    )
    parser.add_argument(
        "--allow-missing-staging",
        action="store_true",
        help=(
            "Publish even if the build recorded staging that never arrived. A "
            "build that declared --no-staging publishes without the flag."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """``microcosm-publish-release-build``: 0 when the change did what it asked.

    A build that fails its checks (the release contract, a file whose bytes
    differ, another repository) is refused with the reason and status 1,
    before any emitter starts, and so is any change while ``HF_HUB_OFFLINE``
    or ``TRANSFORMERS_OFFLINE`` is set or while the repository still has
    ``latest*.json`` pointer files. A change the emitter's write-access check
    refuses is handed nothing and exits 1 with the reason.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    release_dir: Path = args.release_dir
    if not publishes_to_build_folder(release_dir.name):
        parser.error(
            f"release {release_dir.name!r} does not publish into build folders. "
            + TAGGED_RELEASE_ADVICE
        )
    try:
        return _run(args)
    except (ValueError, FileNotFoundError) as error:
        print(f"refusing to publish: {error}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    release_dir: Path = args.release_dir
    if args.promote:
        outcome = promote_release_build(
            release_dir,
            repository=args.repo_id,
            wait_seconds=(
                PROMOTION_WAIT_SECONDS
                if args.wait_seconds is None
                else args.wait_seconds
            ),
        )
    else:
        refusal = publication_refusal(
            release_dir, allow_missing_staging=args.allow_missing_staging
        )
        if refusal is not None:
            print(refusal, file=sys.stderr)
            return 1
        if args.preflight_only:
            prepared = prepare_build_folder_release(
                release_dir, artifact_root=args.artifact_root, repository=args.repo_id
            )
            verdict: dict[str, Any] = {
                "valid": True,
                "published": False,
                "release": prepared.release,
                "build": prepared.build,
                "repository": prepared.repository,
                "release_role": prepared.release_role,
            }
            if prepared.contract_warnings:
                verdict["contract_warnings"] = list(prepared.contract_warnings)
            print(json.dumps(verdict))
            return 0
        outcome = publish_release_build(
            release_dir,
            repository=args.repo_id,
            artifact_root=args.artifact_root,
            make_current=args.make_current,
            wait_seconds=(
                PUBLICATION_WAIT_SECONDS
                if args.wait_seconds is None
                else args.wait_seconds
            ),
        )
    print(json.dumps(outcome.as_json(), indent=2))
    print(outcome.explanation(), file=sys.stderr)
    return 0 if outcome.succeeded else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
