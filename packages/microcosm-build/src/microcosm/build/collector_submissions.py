"""What becomes of a staged bundle's or release change's collector submissions.

Every emitter a build starts has its Hugging Face module's collector
submissions on (``telemetry_emitter.COLLECTOR_SUBMISSIONS``). A staged bundle
or release publication that names its build (``CollectorProvenance``) is then
also submitted to the collector, from the emitter's own worker, once its
Hugging Face commit lands; a promotion or withdrawal that names its build
submits the ``releases.json`` its commit wrote. An emitter that is closing
starts no new submission, so a build or command that closes its emitter
straight after the commit would leave them queued for the next emitter
started on the machine. :func:`follow_collector_submissions` waits, after
the commit, until the collector has accepted them, and says what became of
them. Nothing here raises into the build or fails a staging or a release
change.
"""

from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any, Final, Protocol

from microcosm_emitter_interface.submissions import CollectorSubmissionReceipt

__all__ = [
    "COLLECTOR_RECEIPT_POLL_SECONDS",
    "COLLECTOR_SUBMISSION_WAIT_SECONDS",
    "UK_COLLECTOR_COUNTRY_CODE",
    "US_COLLECTOR_COUNTRY_CODE",
    "CollectorReceipts",
    "follow_collector_submissions",
]

#: The country code a UK build's collector provenance names. Staging
#: telemetry registers UK runs as ``GB``, but the collector's calibration
#: tree handler knows the country as ``UK`` and builds no tree from a
#: submission naming another code. Its Orrery graph handler records whatever
#: code it is given.
UK_COLLECTOR_COUNTRY_CODE: Final = "UK"
#: The country code a US build's collector provenance names.
US_COLLECTOR_COUNTRY_CODE: Final = "US"
#: How long a build or command waits, once a described bundle or publication
#: has landed on Hugging Face, for the collector to accept its submissions.
#: The emitter's collector row may have retried every five minutes while it
#: waited for the commit, and its next attempt then uploads the files, so ten
#: minutes covers one attempt after the commit. The wait also ends at that
#: attempt when it fails.
COLLECTOR_SUBMISSION_WAIT_SECONDS: Final = 600.0
#: How often the receipt is asked for meanwhile.
COLLECTOR_RECEIPT_POLL_SECONDS: Final = 1.0
#: A receipt's ``pending`` code while the Hugging Face commit has not landed.
_WAITING_FOR_HUGGING_FACE: Final = "WAITING_FOR_HUGGING_FACE"
#: What each ``skipped`` code means.
_NOT_ON_HUGGING_FACE: Final = "its Hugging Face upload failed or was refused"
_SKIPPED_REASONS: Final[Mapping[str, str]] = {
    "COLLECTOR_SETTINGS_REFUSED": (
        "the emitter refused its collector settings or the collector-token "
        "variables it inherits"
    ),
    "COLLECTOR_FOLDER_TOO_LONG": (
        "its Hugging Face folder is longer than the collector accepts"
    ),
    "NOT_STAGED_ON_HUGGING_FACE": _NOT_ON_HUGGING_FACE,
    "NOT_PUBLISHED_ON_HUGGING_FACE": _NOT_ON_HUGGING_FACE,
    # A promotion or withdrawal submits releases.json only when it leaves the
    # release a current build to name.
    "NO_CURRENT_BUILD": "the release has no current build after the change",
}
#: Which credential collector submissions use. An emitter given a collector
#: token in the environment uses it instead of the Hugging Face credential,
#: and the token only sends telemetry, so its submissions never have a
#: credential: the variables are to be unset, not completed.
_CREDENTIAL_ADVICE: Final = (
    "Collector submissions use the Hugging Face credential, and an emitter"
    " given a collector token uses that token instead, which only sends"
    " telemetry: unset MICROCOSM_COLLECTOR_TOKEN,"
    " MICROCOSM_COLLECTOR_TOKEN_EXPIRES_AT and MICROCOSM_COLLECTOR_ORIGIN for a"
    " staging or publication that should reach the collector."
)
#: Added to a ``COLLECTOR_SETTINGS_REFUSED`` warning.
_SETTINGS_REFUSED_ADVICE: Final = (
    " The emitter's own warning gives the error type: a collector origin or"
    " development collector it does not trust, or collector-token variables"
    f" it cannot use. {_CREDENTIAL_ADVICE}"
)
#: A ``pending`` code: the emitter has no credential for the submissions,
#: because it was given a telemetry-only collector token or finds no Hugging
#: Face credential. A later emitter sends them only once it has one.
_CREDENTIAL_MISSING: Final = "CREDENTIAL_MISSING"
#: A receipt's code, printed only when it has this shape: the emitter's codes,
#: or the collector's own, which are lower case (``content_mismatch``).
_CODE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


class CollectorReceipts(Protocol):
    """The emitter calls :func:`follow_collector_submissions` makes."""

    @property
    def available(self) -> bool: ...

    def collector_submission_receipt(
        self, handover_id: str
    ) -> CollectorSubmissionReceipt | None: ...


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr, flush=True)


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:g} minutes"


def _attempted_after_commit(
    receipt: Mapping[str, Any], earlier_attempts: object
) -> bool:
    """Whether an attempt that found the Hugging Face commit has finished."""
    attempts = receipt.get("attempts")
    return (
        isinstance(attempts, int)
        and isinstance(earlier_attempts, int)
        and attempts > earlier_attempts
        and receipt.get("error_code") != _WAITING_FOR_HUGGING_FACE
    )


def _collector_warning(receipt: Mapping[str, Any], subject: str) -> str | None:
    """What the build says about a collector receipt, or None."""
    status = receipt.get("status")
    raw_code = receipt.get("error_code")
    code = raw_code if isinstance(raw_code, str) and _CODE.fullmatch(raw_code) else None
    shown = code or "no code"
    if status == "skipped":
        reason = _SKIPPED_REASONS.get(shown, "the emitter skipped them")
        advice = (
            _SETTINGS_REFUSED_ADVICE if code == "COLLECTOR_SETTINGS_REFUSED" else ""
        )
        return (
            f"the collector receives nothing of {subject}: {reason} ({shown}).{advice}"
        )
    if status == "rejected":
        return f"the collector rejected the submissions of {subject} ({shown})."
    if status == "failed":
        return f"the submissions of {subject} to the collector failed ({shown})."
    if status == "timed_out":
        return (
            f"the collector gave no final status for the submissions of {subject} "
            f"within a day of accepting them ({shown})."
        )
    if status == "pending" and code == _CREDENTIAL_MISSING:
        return (
            f"the collector had not accepted the submissions of {subject} when "
            f"the wait ended ({code}): the emitter has no credential to submit "
            f"them with. {_CREDENTIAL_ADVICE} A build with no Hugging Face "
            "credential needs HF_TOKEN or `hf auth login`. The emitter keeps them "
            "queued on this machine for a later emitter that has the credential."
        )
    if status == "pending":
        detail = f" ({code})" if code else ""
        return (
            f"the collector had not accepted the submissions of {subject} when "
            f"the wait ended{detail}; the emitter keeps them queued on this "
            "machine, and an emitter running here later sends them. Leave "
            "unchanged any of their files the emitter did not copy at hand-over "
            "until the hand-over settles."
        )
    return None


def follow_collector_submissions(
    emitter: CollectorReceipts,
    handover_id: str,
    *,
    subject: str,
    wait_seconds: float | None,
    poll_seconds: float = COLLECTOR_RECEIPT_POLL_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> CollectorSubmissionReceipt | None:
    """The hand-over's collector receipt, once the collector has taken it.

    ``wait_seconds`` is None for a hand-over whose Hugging Face commit has
    not landed: its submissions wait for the commit, so this asks once, and
    warns only when the collector will receive nothing or turned them down.
    Otherwise it says, on stderr, that ``subject`` is on Hugging Face and
    waits, at most ``wait_seconds``, while the receipt is ``pending``, until
    an attempt that found the commit has finished: one that the collector
    accepted ends ``pending``, and one that failed is retried by the
    emitter. Any answer but an accepted or succeeded one is a warning naming
    ``subject``. None, without a warning, when the emitter has no collector
    row for the hand-over (it named no build, or no file the collector
    takes) or does not answer.
    """

    if not emitter.available:
        return None
    receipt = emitter.collector_submission_receipt(handover_id)
    if receipt is None:
        return None
    if wait_seconds is not None:
        if receipt.get("status") == "pending":
            print(
                f"{subject} is on Hugging Face; waiting up to "
                f"{_minutes(wait_seconds)} for the collector to accept its "
                "submissions...",
                file=sys.stderr,
                flush=True,
            )
        earlier_attempts = receipt.get("attempts")
        deadline = clock() + max(0.0, float(wait_seconds))
        while (
            receipt.get("status") == "pending"
            and not _attempted_after_commit(receipt, earlier_attempts)
            and clock() < deadline
        ):
            sleep(poll_seconds)
            latest = emitter.collector_submission_receipt(handover_id)
            if latest is None:
                break
            receipt = latest
    elif receipt.get("status") == "pending":
        return receipt
    warning = _collector_warning(receipt, subject)
    if warning is not None:
        _warn(warning)
    return receipt
