"""Publish a release build into its own folder through the emitter.

A release whose builds can publish into their own folders (the UK
national, dense and local-area releases and the US releases, which
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

``microcosm-withdraw-release-build`` (:func:`withdraw_release_build`) names a
published build by its repository, release and build and hands its
withdrawal to the same writer, which sets the build's ``withdrawn`` flag in
``releases.json`` and leaves its folder as published. The writer refuses the
current build (make another build current first) and a build the record does
not list, and never makes a withdrawn build current. The loader loads a
withdrawn build only when ``build=`` pins it, with a warning.

A build is current only when the emitter's receipt says so: ``status`` is
``published`` and ``current`` is the build. A publication with
``make_current`` can land without making its build current, when a change to
the release's current build requested after it landed first.

A publication names its build for the collector (:func:`collector_provenance`),
so the emitter, whose collector submissions are on, submits the build's
calibration diagnostics, with its build and release manifests, to the
collector after the release commit, under the build's folder
``releases/<release>/<build>``. Once the publication has landed, the command
waits for the collector to accept them before it closes its emitter, which
starts no new submission once it is closing, and prints the collector's
status; a collector that receives nothing or turns them down is a warning,
and never fails the publication.

A promotion (``--promote``) and a withdrawal name their own release and
build for the collector too, as the interface requires. Once the change has
landed, the emitter submits ``releases.json`` at the commit its receipt
reports (for a change that writes nothing, such as a promotion of the build
already current or a withdrawal of a withdrawn build, the commit that last
changed it) to the collector as a
``releases-record`` submission naming the release's current build after the
change: the promoted build, or, for a withdrawal, the current build it left
in place. The collector then reads ``releases.json`` at the repository's
head and points the country's latest calibration tree at its default
release's current build, an older build as well as a newer one, when it has
that build's tree. The command waits for the collector to accept the
submission, as for a publication, and says nothing of the collector for a
change that has not landed: a refused change submits nothing, and a pending
one is submitted by the emitter that lands it later. A release with
no current build after the change sends nothing (``NO_CURRENT_BUILD``). A
publication with ``make_current`` needs no record of its own: its
calibration diagnostics submission already names the build it made current.

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
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from microcosm_emitter_interface import (
    CollectorProvenance,
    PromoteBuild,
    PublishRelease,
    ReleaseFile,
    WithdrawBuild,
)
from microcosm_emitter_interface.publication import (
    ReleaseChange,
    ReleaseReceipt,
    unavailable_release_receipt,
)
from microcosm_emitter_interface.releases import is_pointer_file

from microcosm.build.collector_submissions import (
    COLLECTOR_RECEIPT_POLL_SECONDS,
    COLLECTOR_SUBMISSION_WAIT_SECONDS,
    UK_COLLECTOR_COUNTRY_CODE,
    US_COLLECTOR_COUNTRY_CODE,
    CollectorReceipts,
    follow_collector_submissions,
)
from microcosm.build.staging_dataset import hub_offline, write_access_problem
from microcosm.build.telemetry_emitter import start_bundle_emitter
from microcosm.data.contract import US_RELEASES, publishes_to_build_folder
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
    from microcosm_emitter_interface.submissions import CollectorSubmissionReceipt

__all__ = [
    "PROMOTION_WAIT_SECONDS",
    "PUBLICATION_WAIT_SECONDS",
    "ReleaseEmitter",
    "ReleasePublication",
    "collector_provenance",
    "main",
    "promote_release_build",
    "publication_request",
    "publish_release_build",
    "withdraw_main",
    "withdraw_release_build",
]

#: How long the command waits for a publication to land or be refused. The
#: emitter uploads a build's files in its one commit: the largest UK build
#: recorded (about 2.4 GB) takes about 16 minutes at 20 Mbit/s, and a US ACS
#: local-area build (about 10 GB) about 67 minutes. The emitter refuses a
#: change still pending an hour after it was handed over
#: (``REQUEST_EXPIRED``), so the wait covers that hour and the longest
#: upload.
PUBLICATION_WAIT_SECONDS = 3 * 3600.0
#: How long the command waits for a promotion, which changes only
#: ``releases.json``.
PROMOTION_WAIT_SECONDS = 300.0
#: How long the withdraw command waits; a withdrawal also changes only
#: ``releases.json``.
WITHDRAWAL_WAIT_SECONDS = PROMOTION_WAIT_SECONDS
#: The three release changes a command hands to the emitter.
type ReleaseChangeKind = Literal["publish", "promote", "withdraw"]
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
    "WITHDRAWING_CURRENT_BUILD": (
        "A release's current build cannot be withdrawn: make another build "
        "current first (microcosm-publish-release-build <that build's release "
        "directory> --promote), then withdraw this one."
    ),
    "CURRENT_WITHDRAWN": (
        "The build has been withdrawn, so it cannot be made current."
    ),
    "REQUEST_EXPIRED": (
        "The change was still waiting an hour after it was handed over, so "
        "the emitter dropped it; run the command again."
    ),
}
#: Advice that depends on the change as well as the code.
_CHANGE_ADVICE: dict[tuple[str, str], str] = {
    ("withdraw", "BUILD_NOT_LISTED"): (
        "The repository's releases.json does not list that build of that "
        "release; check the names, which are its folder releases/<release>/"
        "<build>/."
    ),
    ("promote", "BUILD_NOT_LISTED"): (
        "The repository's releases.json does not list this build: it has not "
        "been published, or its publication has not landed yet. Publish it "
        "first, or run the publish command again to see whether it landed."
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


class ReleaseEmitter(CollectorReceipts, Protocol):
    """The interface ``EmitterClient``'s release calls and state.

    Its collector receipts come from ``CollectorReceipts``.
    """

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

    def withdraw_build(
        self, request: WithdrawBuild, *, wait_seconds: float = 60.0
    ) -> ReleaseReceipt: ...

    def close(self) -> None: ...


#: The shape of an emitter's hand-over ID.
_HANDOVER_ID = re.compile(r"^[0-9a-f]{32}$")
#: Starts the short-lived emitter a publication uses (only its Hugging Face
#: module); tests replace it.
_start_release_emitter = start_bundle_emitter
#: The clock and sleep of the wait for the collector; tests replace them.
_clock = time.monotonic
_sleep = time.sleep


def collector_provenance(release: str, build: str) -> CollectorProvenance | None:
    """The collector's provenance of a build of ``release``, or None.

    It names the publication's, promotion's or withdrawal's own release and
    build, as the interface requires, since the collector refuses a build
    folder whose provenance names another. A UK build (the national, dense and local-area releases)
    used licensed microdata, so the collector keeps its Orrery outputs
    private; a US build is published in the public US repository, so it used
    none whose licence keeps its outputs private. None for a release whose
    builds do not publish into their own folders. The interface raises
    ``ValueError`` for a build ID it does not accept.
    """
    if not publishes_to_build_folder(release):
        return None
    us_build = release in US_RELEASES
    return CollectorProvenance(
        country_code=US_COLLECTOR_COUNTRY_CODE
        if us_build
        else UK_COLLECTOR_COUNTRY_CODE,
        build_id=build,
        licensed_data=not us_build,
        release=release,
    )


@dataclass(frozen=True)
class ReleasePublication:
    """What became of one release change: the emitter's receipt, read.

    ``wanted_current`` says whether the change asked for the build to become
    current: a promotion, or a publication with ``make_current``. ``change``
    is ``publish``, ``promote`` or ``withdraw``. ``problem`` says why nothing
    was handed over, when the emitter could not take the change or its
    write-access check refused it.
    ``collector_receipt`` is the emitter's last collector receipt of the
    change (a publication's calibration diagnostics, or the ``releases.json``
    a promotion or withdrawal submits), or None for a change handed nothing,
    one that names no build for the collector, a promotion or withdrawal that
    has not landed, or one the emitter has no collector row for.
    """

    repository: str
    release: str
    build: str
    wanted_current: bool
    receipt: ReleaseReceipt
    problem: str | None = None
    change: ReleaseChangeKind = "publish"
    collector_receipt: CollectorSubmissionReceipt | None = None

    @property
    def withdrawn(self) -> bool:
        """Whether this change was a withdrawal that landed.

        A publication or promotion says nothing about whether the build was
        withdrawn since; :meth:`as_json` reports that as unknown.
        """
        return self.change == "withdraw" and self.published

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
            advice = _CHANGE_ADVICE.get(
                (self.change, str(code)), _REFUSAL_ADVICE.get(str(code))
            )
            reason = self.receipt.get("message") or code
            return f"The emitter refused it ({code}): {reason}" + (
                f" {advice}" if advice else ""
            )
        if self.withdrawn:
            return (
                f"Build {self.build} of {self.release} is withdrawn: its folder "
                "stays, it can no longer be made current, and it loads only "
                "when build= pins it."
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
            "change": self.change,
            "repository": self.repository,
            "release": self.release,
            "build": self.build,
            "status": self.receipt["status"],
            "withdrawn": self.withdrawn if self.change == "withdraw" else None,
            "current": self.current,
            "current_build": self.receipt.get("current"),
            "error_code": self.receipt.get("error_code"),
            "message": self.receipt.get("message"),
            "commit": self.receipt.get("commit"),
            "handover_id": self.receipt.get("handover_id"),
            "collector_submission": (
                None
                if self.collector_receipt is None
                else {
                    "status": self.collector_receipt.get("status"),
                    "error_code": self.collector_receipt.get("error_code"),
                }
            ),
        }


def _hand_over(
    change: ReleaseChange,
    *,
    client: ReleaseEmitter | None,
    wait_seconds: float,
    wanted_current: bool,
    collector_wait_seconds: float = COLLECTOR_SUBMISSION_WAIT_SECONDS,
) -> ReleasePublication:
    owned = client is None
    emitter: ReleaseEmitter = _start_release_emitter() if owned else client
    collector_receipt: CollectorSubmissionReceipt | None = None
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
        kind: ReleaseChangeKind
        if isinstance(change, PublishRelease):
            kind = "publish"
        elif isinstance(change, WithdrawBuild):
            kind = "withdraw"
        else:
            kind = "promote"
        if problem is not None:
            receipt = unavailable_release_receipt(change)
        else:
            if isinstance(change, PublishRelease):
                receipt = emitter.publish_release(change, wait_seconds=wait_seconds)
            elif isinstance(change, WithdrawBuild):
                receipt = emitter.withdraw_build(change, wait_seconds=wait_seconds)
            else:
                receipt = emitter.promote_build(change, wait_seconds=wait_seconds)
            landed = receipt["status"] == "published"
            # A publication is asked about once even before it lands, so a
            # collector that will receive nothing is said at once. A promotion
            # or withdrawal is asked about only once it has landed: a refused
            # one submits nothing, a pending one is submitted by the emitter
            # that lands it after this command ends, and before it lands a
            # warning would name a releases.json change the writer may never
            # make. The emitter's own startup warning still says when it
            # refused its collector settings.
            asks_collector = (
                receipt["status"] != "unavailable" if kind == "publish" else landed
            )
            if change.collector is not None and asks_collector:
                # Before the emitter closes: a closing emitter starts no new
                # submission. The emitter keeps the collector row under the
                # hand-over ID its receipt names.
                collector_receipt = follow_collector_submissions(
                    emitter,
                    _receipt_handover_id(receipt) or change.handover_id(),
                    subject=_collector_subject(kind, change.release, change.build),
                    wait_seconds=collector_wait_seconds if landed else None,
                    poll_seconds=COLLECTOR_RECEIPT_POLL_SECONDS,
                    clock=_clock,
                    sleep=_sleep,
                    reads_local_files=kind == "publish",
                )
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
        change=kind,
        collector_receipt=collector_receipt,
    )


def _collector_subject(kind: ReleaseChangeKind, release: str, build: str) -> str:
    """What the command calls a change's collector submission in its messages."""
    if kind == "promote":
        return f"releases.json with build {build} current in {release}"
    if kind == "withdraw":
        return f"releases.json with build {build} of {release} withdrawn"
    return f"build {build} of {release}"


def _described[RecordChange: (PromoteBuild, WithdrawBuild)](
    change: RecordChange,
) -> RecordChange:
    """``change`` naming its own release and build for the collector.

    The change is built first without it, so a name the interface refuses is
    reported as the change's own (``build``), not the provenance's
    (``build_id``).
    """
    return replace(change, collector=collector_provenance(change.release, change.build))


def _receipt_handover_id(receipt: ReleaseReceipt) -> str | None:
    """The hand-over ID a release receipt names, when it has the emitter's shape."""
    handover_id = receipt.get("handover_id")
    return (
        handover_id
        if isinstance(handover_id, str) and _HANDOVER_ID.fullmatch(handover_id)
        else None
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


def _hub_api() -> Any:
    """A Hub client with the credential :func:`_repository_files` uses (test
    seam), for the recertification parent check and the annual base check."""
    import huggingface_hub

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None
    return huggingface_hub.HfApi(token=token)


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
    """The ``publish_release`` request for a checked build.

    It names the build for the collector (:func:`collector_provenance`).
    """
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
        collector=collector_provenance(prepared.release, prepared.build),
    )


def _refuse_recertification_as_current(release_dir: Path | str) -> None:
    """Refuse to make a compatibility recertification its release's current build.

    A recertification republishes its parent's data under a wider
    compatibility claim. As with the tagged publisher, which never moved
    ``latest.json`` to one, it is published without becoming current, and
    consumers load it with ``build=``.
    """
    from microcosm.data import compatibility_recertification

    if compatibility_recertification.is_recertification(release_dir):
        raise ValueError(
            f"{Path(release_dir)} holds a compatibility recertification, which "
            "is published without becoming its release's current build; "
            "publish it without --make-current, and load it with build=."
        )


def publish_release_build(
    release_dir: Path | str,
    *,
    repository: str | None = None,
    artifact_root: Path | str | None = None,
    make_current: bool = False,
    client: ReleaseEmitter | None = None,
    wait_seconds: float = PUBLICATION_WAIT_SECONDS,
    parent_h5: Path | str | None = None,
    compatibility_wheels: tuple[Path | str, ...] = (),
    collector_wait_seconds: float = COLLECTOR_SUBMISSION_WAIT_SECONDS,
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
    passes. A recertification is handed over only once the Hub shows its
    parent published with the manifest it carries, as the tagged publisher
    required, and never with ``make_current``. Once the publication has
    landed, it waits at most ``collector_wait_seconds`` for the collector to
    accept the build's submissions.
    """
    if client is None:
        _refuse_offline()
    if make_current:
        _refuse_recertification_as_current(release_dir)
    named_repository, _, _ = build_folder_identity(release_dir)
    _refuse_pointer_files(named_repository)
    prepared = prepare_build_folder_release(
        release_dir,
        artifact_root=artifact_root,
        repository=repository,
        parent_h5=parent_h5,
        compatibility_wheels=tuple(compatibility_wheels),
    )
    from microcosm.data import compatibility_recertification

    if compatibility_recertification.is_recertification(prepared.release_dir):
        compatibility_recertification.require_parent_publication(
            _hub_api(),
            prepared.release_dir,
            repo_id=prepared.repository,
            artifact_root=artifact_root,
        )
    from microcosm.data import annual_projections

    if annual_projections.is_annual_build(prepared.release_dir):
        # An annual build projects a published national build: the Hub must
        # show that build with the same base-year H5 before anything is
        # handed over.
        annual_projections.require_base_publication(
            _hub_api(), prepared.release_dir, repo_id=prepared.repository
        )
    return _hand_over(
        publication_request(prepared, make_current=make_current),
        client=client,
        wait_seconds=wait_seconds,
        wanted_current=make_current,
        collector_wait_seconds=collector_wait_seconds,
    )


def promote_release_build(
    release_dir: Path | str,
    *,
    repository: str | None = None,
    client: ReleaseEmitter | None = None,
    wait_seconds: float = PROMOTION_WAIT_SECONDS,
    collector_wait_seconds: float = COLLECTOR_SUBMISSION_WAIT_SECONDS,
) -> ReleasePublication:
    """Make a published build, named by its local manifest, the current build.

    It refuses and checks write access as :func:`publish_release_build` does,
    and refuses a compatibility recertification, which is never current. The
    promotion names the build for the collector, so the emitter submits
    ``releases.json`` once the promotion has landed; it then waits at most
    ``collector_wait_seconds`` for the collector to accept it.
    """
    if client is None:
        _refuse_offline()
    _refuse_recertification_as_current(release_dir)
    named_repository, release, build = build_folder_identity(release_dir)
    if repository is not None and repository != named_repository:
        raise ValueError(
            f"release {release!r} names {named_repository} in its artifacts, not "
            f"{repository}."
        )
    _refuse_pointer_files(named_repository)
    return _hand_over(
        _described(
            PromoteBuild(repository=named_repository, release=release, build=build)
        ),
        client=client,
        wait_seconds=wait_seconds,
        wanted_current=True,
        collector_wait_seconds=collector_wait_seconds,
    )


def withdraw_release_build(
    repository: str,
    release: str,
    build: str,
    *,
    client: ReleaseEmitter | None = None,
    wait_seconds: float = WITHDRAWAL_WAIT_SECONDS,
    collector_wait_seconds: float = COLLECTOR_SUBMISSION_WAIT_SECONDS,
) -> ReleasePublication:
    """Withdraw a published build: set its ``withdrawn`` flag in ``releases.json``.

    The emitter's release writer makes the change and runs its checks: it
    refuses the release's current build (``WITHDRAWING_CURRENT_BUILD``) and a
    build the record does not list (``BUILD_NOT_LISTED``). The build's folder
    and files stay as published. It refuses (``ValueError``) offline and while
    the repository has ``latest*.json`` pointer files, before any emitter
    starts, and hands nothing over unless the emitter's write-access check
    passes, as :func:`publish_release_build` does. The withdrawal names the
    build for the collector, so the emitter submits ``releases.json``, naming
    the release's current build, once the withdrawal has landed; it then
    waits at most ``collector_wait_seconds`` for the collector to accept it.
    """
    request = _described(
        WithdrawBuild(repository=repository, release=release, build=build)
    )
    if client is None:
        _refuse_offline()
    _refuse_pointer_files(repository)
    return _hand_over(
        request,
        client=client,
        wait_seconds=wait_seconds,
        wanted_current=False,
        collector_wait_seconds=collector_wait_seconds,
    )


def _withdraw_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microcosm-withdraw-release-build",
        description=(
            "Withdraw a published build of a release through the emitter: set "
            "its withdrawn flag in the repository's releases.json. Its folder "
            "stays; it can no longer be made current, and the loader loads it "
            "only when build= pins it. The current build cannot be withdrawn: "
            "make another build current first."
        ),
    )
    parser.add_argument("repository", help="The Hugging Face dataset repository.")
    parser.add_argument("release", help="The release, as releases.json names it.")
    parser.add_argument("build", help="The build to withdraw: its folder's name.")
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=WITHDRAWAL_WAIT_SECONDS,
        help=(
            "How long to wait for the change to land or be refused (default "
            f"{WITHDRAWAL_WAIT_SECONDS:.0f})."
        ),
    )
    return parser


def withdraw_main(argv: list[str] | None = None) -> int:
    """``microcosm-withdraw-release-build``: 0 when the build is withdrawn.

    A withdrawal refused before any emitter starts (offline, pointer files,
    a name the interface refuses) prints the reason and returns 1, as does
    one the write-access check or the release writer refuses.
    """
    args = _withdraw_parser().parse_args(argv)
    try:
        outcome = withdraw_release_build(
            args.repository,
            args.release,
            args.build,
            wait_seconds=args.wait_seconds,
        )
    except (ValueError, FileNotFoundError) as error:
        print(f"refusing to withdraw: {error}", file=sys.stderr)
        return 1
    print(json.dumps(outcome.as_json(), indent=2))
    print(outcome.explanation(), file=sys.stderr)
    return 0 if outcome.withdrawn else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microcosm-publish-release-build",
        description=(
            "Publish a build of a release whose builds publish into their own "
            "folders (the UK national, dense and local-area releases and the US "
            "releases) through the emitter, or make a published build current. "
            + TAGGED_RELEASE_ADVICE
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
        help=(
            "Make the build the release's current build when it is published "
            "(refused for a compatibility recertification)."
        ),
    )
    action.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Make this already published build the release's current build "
            "(refused for a compatibility recertification)."
        ),
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
        "--parent-h5",
        type=Path,
        help="A source enrichment's exact parent H5, which its contract replays.",
    )
    parser.add_argument(
        "--compatibility-wheel",
        action="append",
        type=Path,
        default=[],
        help=(
            "An exact tested wheel a source enrichment's or recertification's "
            "contract replays its loader under; repeat for each package."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-reform-validation",
        action="store_true",
        help=(
            "Publish even if reform_validation.json records that out-of-sample "
            "reforms were skipped (--skip-out-of-sample-reforms)."
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


def _claim_records(release_dir: Path) -> dict[str, Any]:
    """The compatibility records a build carries for whoever publishes it.

    ``narrowed_claims`` is a compatibility range an earlier certification
    narrowed back to an exact pin; ``claim_measurement`` is the model
    versions a declared compatibility range was measured against (a source
    enrichment's, or a recertification's with its baseline and drift
    summary). Only records the build has are returned. Reading them refuses
    nothing: the release contract has already accepted or refused them.
    """
    from microcosm.data import compatibility_recertification, source_enrichment

    narrowed = source_enrichment.recorded_narrowed_claims(release_dir)
    measurement = source_enrichment.recorded_claim_measurement(
        release_dir
    ) or compatibility_recertification.recorded_claim_measurement(release_dir)
    records: dict[str, Any] = {}
    if narrowed:
        records["narrowed_claims"] = narrowed
    if measurement:
        records["claim_measurement"] = measurement
    return records


def _print_claim_records(records: Mapping[str, Any]) -> None:
    """Say a build's compatibility records on stderr, as the tagged command
    did, on the preflight and on publication, which needs no preflight
    first; the preflight's verdict on stdout carries them too."""
    if "narrowed_claims" in records:
        print(
            "note: this build records a compatibility narrowing from an "
            f"earlier certification: {json.dumps(records['narrowed_claims'])}",
            file=sys.stderr,
        )
    if "claim_measurement" in records:
        print(
            "note: this build declares a publisher compatibility range that "
            "certification measured against only the versions recorded here: "
            f"{json.dumps(records['claim_measurement'], sort_keys=True)}",
            file=sys.stderr,
        )


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
            release_dir,
            allow_incomplete_reform_validation=args.allow_incomplete_reform_validation,
            allow_missing_staging=args.allow_missing_staging,
        )
        if refusal is not None:
            print(refusal, file=sys.stderr)
            return 1
        claim_records = _claim_records(release_dir)
        _print_claim_records(claim_records)
        if args.preflight_only:
            prepared = prepare_build_folder_release(
                release_dir,
                artifact_root=args.artifact_root,
                repository=args.repo_id,
                parent_h5=args.parent_h5,
                compatibility_wheels=tuple(args.compatibility_wheel),
            )
            verdict: dict[str, Any] = {
                "valid": True,
                "published": False,
                "release": prepared.release,
                "build": prepared.build,
                "repository": prepared.repository,
                "release_role": prepared.release_role,
                **claim_records,
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
            parent_h5=args.parent_h5,
            compatibility_wheels=tuple(args.compatibility_wheel),
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
