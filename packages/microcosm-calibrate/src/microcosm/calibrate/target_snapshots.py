"""Per-target estimate snapshots from an in-progress calibration (microcosm#908).

The solver already materializes the complete target-estimate vector once per
epoch (``estimate = A @ w`` inside :func:`~microcosm.calibrate.solve._optimize`)
and then drops it: only the scalar loss reaches ``progress_callback``, and the
complete per-target picture appears only after the run finishes, in
``calibration_diagnostics.json``. This module is the shared, aggregate-only
codec that lets a run publish that vector *while it is still running*, without
changing what the optimizer computes.

Three promises the codec is built to keep.

**Aggregate-only.** A snapshot carries one row per calibration *target* and
nothing else. There is no weight vector, no record index, and no household,
person, benefit-unit or source-record identifier. That is enforced by a closed
contract rather than by a denylist over arbitrary JSON: the payload's keys are
a closed set, its identifiers (``run_id``, ``candidate_id``, ``phase``) are
strings, and *every* metadata seam — ``context``, ``search``, ``selection`` and
``best_retained`` — is a flat mapping of string keys to JSON **scalars**
(``str``, ``int``, ``float``, ``bool``, ``null``), bounded in entry count, key
length and string length (:data:`MAX_METADATA_ENTRIES`,
:data:`MAX_METADATA_KEY_LENGTH`, :data:`MAX_METADATA_STRING_LENGTH`). A list is
not a scalar, so a record-level vector has no shape to travel in, at any depth;
a nested mapping is refused for the same reason. The record-level key-name rule
(:data:`RECORD_LEVEL_KEY_NAMES`) is kept on top of that, because a *scalar*
``household_id`` is still record-level identity, and because those are the
names the version-2 staging content policy refuses on upload (microcosm#896).

The contract is the shape every real caller already emits:
:mod:`~microcosm.calibrate.solve` emits ``search`` as
``{budget_iteration, budget_iters, l0_lambda}``, ``selection`` as the iterate
selection receipt (``rule``, ``selected_epoch``, ``epochs_executed``,
``epoch_convention``, ``selected_loss_float32``,
``closing_iterate_loss_float32``) and ``best_retained`` as the closed triple
``{available, epoch, loss}``. ``context`` is caller-supplied run labelling and
has no in-tree producer, so it takes the same flat scalar contract rather than
a wider one invented for a caller that does not exist.

Because every supported metadata value is an immutable scalar, the mapping the
sink receives is a freshly built container of its own: a sink that mutates a
delivered ``context``, ``search``, ``selection``, ``best_retained`` or target
row cannot reach the caller's mapping, the observer, or the next snapshot.

**Honest iterate labelling.** ``current`` means "the weights the optimizer held
at this epoch". ``best_retained`` means "these values are the incumbent best
iterate's, and the optimizer is actually retaining a best" — it is never
attached to an iterate merely because it is the newest. ``selected`` means "the
weights the solver actually returned". A run whose optimizer retains no best at
all (gated, mass-conserved, or L2-penalized runs; see ``retain_best`` in
:mod:`~microcosm.calibrate.solve`) reports ``best_retained.available: false``
rather than nominating one.

**Stable identity.** ``targets_sha256`` digests the ordered ``(row name, target
value)`` list, so a consumer can tell that two snapshots describe the same
ordered target list, and reordering or revaluing the targets breaks the digest.

Values sampled inside the optimization loop come off a ``float32`` tensor and
say so (``precision: "float32"``); the closing ``selected`` snapshot is taken
from the same ``float64`` path the final diagnostics use
(``precision: "float64"``). The solver's existing receipt already makes this
distinction (``selected_loss_float32``), and a snapshot must not quietly imply
more precision than it has.

The store keeps a ``latest.json`` replaced atomically (temporary file in the
same directory, ``fsync``, ``os.replace``, parent ``fsync`` — the idiom
``microcosm.build.logbook_adoption.atomic_write_json`` already uses, reproduced
here because the calibrate shard may not depend on the build shard) so a
concurrent dashboard read never sees a partially written document. History
chunks under ``history/`` are published the same way and for the same reason:
the bytes go to a hidden temporary file which is written, flushed and
``fsync``-ed first, and only then does the chunk acquire its immutable
``history/<sequence>.json`` name, through :func:`os.link` — atomic, and it
refuses rather than overwrites when that name already exists. So a polling
reader sees a chunk only once its bytes are complete, and a write that fails
part-way leaves neither a partial chunk nor a stray temporary behind. Retention
is bounded and its drops are recorded rather than silent.

This module is a leaf: it imports only the standard library and numpy, so
:mod:`microcosm.calibrate.solve` can import it without a cycle.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "EVERY_EPOCH",
    "ITERATE_BEST_RETAINED",
    "ITERATE_CURRENT",
    "ITERATE_SELECTED",
    "LATEST_SNAPSHOT_FILENAME",
    "MAX_METADATA_ENTRIES",
    "MAX_METADATA_KEY_LENGTH",
    "MAX_METADATA_STRING_LENGTH",
    "METADATA_LOCATIONS",
    "RECORD_LEVEL_KEY_NAMES",
    "SNAPSHOT_HISTORY_DIRNAME",
    "SNAPSHOT_HISTORY_INDEX_FILENAME",
    "TARGET_SNAPSHOT_ITERATES",
    "TARGET_SNAPSHOT_SCHEMA",
    "TARGET_SNAPSHOT_SCHEMA_VERSION",
    "BoundTargetSnapshots",
    "TargetSnapshotCadence",
    "TargetSnapshotError",
    "TargetSnapshotObserver",
    "TargetSnapshotWriter",
    "signed_relative_error",
    "target_identity_digest",
    "validate_target_snapshot",
]

#: Schema identity. Consumers key their readers on the pair; bump the version
#: with any shape change, exactly as the diagnostics payload does.
TARGET_SNAPSHOT_SCHEMA = "microcosm.calibration.target-snapshot"
TARGET_SNAPSHOT_SCHEMA_VERSION = 1

#: Which weights a snapshot's values were read off.
ITERATE_CURRENT = "current"
ITERATE_BEST_RETAINED = "best_retained"
ITERATE_SELECTED = "selected"
TARGET_SNAPSHOT_ITERATES = (
    ITERATE_CURRENT,
    ITERATE_BEST_RETAINED,
    ITERATE_SELECTED,
)

#: The cadence that emits on every epoch.
EVERY_EPOCH = 1

LATEST_SNAPSHOT_FILENAME = "latest.json"
SNAPSHOT_HISTORY_DIRNAME = "history"
SNAPSHOT_HISTORY_INDEX_FILENAME = "history_index.json"

#: Record-level key names a snapshot may never carry, at any depth. The names
#: match the prohibited content keys the version-2 staging content policy
#: enforces on upload (microcosm#896), so a payload that passes here is not
#: rejected later at the publication boundary; the check is repeated in the
#: shared codec so a purely local snapshot store is bound by the same rule.
RECORD_LEVEL_KEY_NAMES = frozenset(
    {
        "benunit_id",
        "household_id",
        "person_id",
        "raw_record",
        "raw_records",
        "row_id",
        "source_value",
        "source_values",
    }
)

#: Every metadata seam the schema has. The same contract applies to all of
#: them, so a seam added later is covered by construction rather than by
#: someone remembering to extend a list.
METADATA_LOCATIONS = ("context", "search", "selection", "best_retained")

#: Bounds on a metadata mapping. Aggregate metadata labels a run; it is not a
#: payload channel, and a bound is what makes that difference enforceable.
MAX_METADATA_ENTRIES = 32
MAX_METADATA_KEY_LENGTH = 64
MAX_METADATA_STRING_LENGTH = 256

#: JSON integers stay exactly representable in an IEEE-754 double, so every
#: consumer (including a browser) reads back what was written.
_MAX_EXACT_JSON_INTEGER = 2**53

#: The closed shape of ``best_retained``. ``available`` says whether the
#: optimizer is running a retain-best rule at all; ``epoch``/``loss`` describe
#: the incumbent when there is one, and are null when there is not (including
#: on a retaining run that has not recorded one yet, and on a run whose closing
#: iterate won).
_BEST_RETAINED_KEYS = frozenset({"available", "epoch", "loss"})

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "run_id",
        "candidate_id",
        "created_at",
        "sequence",
        "phase",
        "epoch",
        "epochs",
        "search",
        "iterate",
        "precision",
        "loss",
        "non_finite_rows",
        "best_retained",
        "selection",
        "targets_sha256",
        "n_targets",
        "context",
        "targets",
    }
)
_TARGET_ROW_KEYS = frozenset({"index", "name", "target", "estimate", "relative_error"})

_PRECISIONS = ("float32", "float64")

#: Tolerance for the relative-error consistency check. The stored value is a
#: float64 recomputation of a float32-derived estimate on the in-loop path, so
#: the check is a consistency guard, not a bit-equality assertion.
_RELATIVE_ERROR_RTOL = 1e-9


def _strict_number(value: object, *, what: str) -> float | None:
    """A JSON number or null. A bool or a numeric string is neither."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TargetSnapshotError(
            f"{what} must be a JSON number or null, got {value!r}."
        )
    result = float(value)
    if not math.isfinite(result):
        raise TargetSnapshotError(f"{what} must be finite when present.")
    return result


class TargetSnapshotError(ValueError):
    """A snapshot payload violates the versioned contract."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _finite_or_none(value: object) -> float | None:
    """A float, or ``None`` when it is not finite.

    JSON has no NaN or Infinity, and a diverging optimizer can legitimately
    hold a non-finite estimate mid-run — the capped loss absorbs it and the
    run still returns weights. Following
    :func:`microcosm.calibrate.diagnostics._finite`, such a value serializes
    as null rather than aborting the calibration that an unobserved run would
    have completed.
    """
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_float(value: object, *, what: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TargetSnapshotError(
            f"{what} must be a real number, got {value!r}."
        ) from (error)
    if not math.isfinite(result):
        raise TargetSnapshotError(f"{what} must be finite, got {value!r}.")
    return result


def _canonical_json_bytes(value: object) -> bytes:
    """Canonical bytes for a digest: sorted keys, no spaces, no NaN."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def signed_relative_error(estimate: float | None, target: float | None) -> float | None:
    """The signed miss of ``estimate`` against ``target``.

    ``(estimate - target) / target``, except that a zero target has no
    relative form, so the signed *absolute* miss ``estimate - target`` is
    reported instead. This is exactly the convention
    :class:`~microcosm.calibrate.solve.TargetDiagnostic` documents and
    ``_target_diagnostics`` implements, so a mid-run snapshot and the run's
    final diagnostics never disagree about what a target's error means.
    """
    estimate = _finite_or_none(estimate)
    target = _finite_or_none(target)
    if estimate is None or target is None:
        # No honest signed error exists; the row reports null rather than
        # inventing one or aborting the run.
        return None
    if target == 0.0:
        return estimate - target
    result = (estimate - target) / target
    return _finite_or_none(result)


def target_identity_digest(
    names: Sequence[str], targets: Sequence[float] | np.ndarray
) -> str:
    """A sha256 over the ordered ``(row index, name, target value)`` list.

    Order-sensitive by construction: the digest covers the list, not a set, so
    a consumer comparing two snapshots' digests is comparing the exact ordered
    target identity the rows are aligned to. The row index is part of each
    entry, so identity stays unambiguous even where a compiled problem happens
    to produce two rows with the same ``name@period`` label — a snapshot must
    never be the thing that aborts a calibration that would otherwise return
    weights. A non-finite target digests as null, for the same reason.
    """
    names = tuple(str(name) for name in names)
    values = [_finite_or_none(value) for value in np.asarray(targets)]
    if len(names) != len(values):
        raise TargetSnapshotError(
            f"names and targets must align: {len(names)} names, {len(values)} targets."
        )
    if not names:
        raise TargetSnapshotError("a calibration snapshot needs at least one target.")
    if any(not name for name in names):
        raise TargetSnapshotError("target row names must be non-empty.")
    return hashlib.sha256(
        _canonical_json_bytes(
            [
                [index, name, value]
                for index, (name, value) in enumerate(zip(names, values, strict=True))
            ]
        )
    ).hexdigest()


def _identifier(value: object, *, what: str, required: bool) -> str | None:
    """A string identifier, or ``None`` where one is optional.

    Coercion is deliberately absent: ``str({"tax_unit_id": [101, 102]})``
    is a perfectly good Python string, and stringifying a mapping is exactly
    how record-level content reached a supposedly aggregate-only identifier.
    """
    if value is None:
        if required:
            raise TargetSnapshotError(f"{what} must be a non-empty string.")
        return None
    if not isinstance(value, str):
        raise TargetSnapshotError(
            f"{what} must be a string, got {type(value).__name__}: {value!r}."
        )
    if not value:
        raise TargetSnapshotError(f"{what} must be a non-empty string.")
    if len(value) > MAX_METADATA_STRING_LENGTH:
        raise TargetSnapshotError(
            f"{what} must be at most {MAX_METADATA_STRING_LENGTH} characters."
        )
    return value


def _metadata_key(key: object, *, where: str) -> str:
    if not isinstance(key, str) or not key:
        raise TargetSnapshotError(
            f"{where} keys must be non-empty strings, got {key!r}."
        )
    if len(key) > MAX_METADATA_KEY_LENGTH:
        raise TargetSnapshotError(
            f"{where} key {key!r} exceeds {MAX_METADATA_KEY_LENGTH} characters."
        )
    if key.lower() in RECORD_LEVEL_KEY_NAMES:
        raise TargetSnapshotError(
            f"{where} carries the record-level key {key!r}; calibration target "
            "snapshots are aggregate-only."
        )
    return key


def _metadata_value(value: object, *, where: str, key: str, coerce: bool) -> object:
    """One aggregate metadata value: a JSON scalar, and nothing else.

    ``coerce`` separates the two edges this contract has. At the *emitting*
    edge a numpy scalar becomes its Python equivalent and a non-finite float
    becomes ``None``, for the same reason a non-finite estimate does: a
    diverging run legitimately holds one, JSON cannot carry it, and the
    observer must never be the thing that ends a calibration an unobserved run
    would have finished. In the *public codec* neither is accepted, so a
    payload that validates is a payload that serializes — the strictness is
    what makes the finite-or-null claim true, and the emitter's normalization
    is what keeps it from costing a run.
    """
    if isinstance(value, np.generic):
        if not coerce:
            raise TargetSnapshotError(
                f"{where}[{key!r}] is a {type(value).__name__}; a validated "
                "snapshot carries plain JSON scalars."
            )
        value = value.item()
    if value is None or isinstance(value, (bool, str)):
        if isinstance(value, str) and len(value) > MAX_METADATA_STRING_LENGTH:
            raise TargetSnapshotError(
                f"{where}[{key!r}] exceeds {MAX_METADATA_STRING_LENGTH} characters; "
                "aggregate metadata labels a run, it is not a payload channel."
            )
        return value
    if isinstance(value, int):
        if abs(value) >= _MAX_EXACT_JSON_INTEGER:
            raise TargetSnapshotError(
                f"{where}[{key!r}] is not exactly representable as a JSON number."
            )
        return value
    if isinstance(value, float):
        if coerce:
            return _finite_or_none(value)
        if not math.isfinite(value):
            raise TargetSnapshotError(
                f"{where}[{key!r}] must be finite or null, got {value!r}."
            )
        return value
    raise TargetSnapshotError(
        f"{where}[{key!r}] must be a JSON scalar (string, number, boolean or null), "
        f"got {type(value).__name__}. Calibration target snapshots carry aggregate "
        "metadata only: no record-level vectors and no nested payloads."
    )


def normalize_metadata(
    value: object, *, where: str, allow_none: bool = False, coerce: bool = True
) -> dict[str, object] | None:
    """Validate ``value`` against the metadata contract and return a fresh copy.

    The returned mapping is newly built and holds only immutable scalars, so
    the caller's mapping and every later snapshot are detached from whatever a
    sink does to the one it was handed.
    """
    if value is None and allow_none:
        return None
    if not isinstance(value, Mapping):
        raise TargetSnapshotError(
            f"{where} must be a mapping of aggregate metadata, got "
            f"{type(value).__name__}."
        )
    if len(value) > MAX_METADATA_ENTRIES:
        raise TargetSnapshotError(
            f"{where} carries {len(value)} entries; at most {MAX_METADATA_ENTRIES} "
            "aggregate metadata entries are supported."
        )
    normalized: dict[str, object] = {}
    for key, item in value.items():
        name = _metadata_key(key, where=where)
        normalized[name] = _metadata_value(item, where=where, key=name, coerce=coerce)
    return normalized


def normalize_best_retained(
    value: object, *, epochs: int | None = None, coerce: bool = True
) -> dict[str, object]:
    """The closed ``{available, epoch, loss}`` triple, freshly built.

    ``coerce`` splits the emitting edge from the public codec exactly as
    :func:`normalize_metadata` does, and ``coerce=False`` additionally
    requires the triple to be complete rather than defaulted.
    """
    if value is None and coerce:
        return {"available": False, "epoch": None, "loss": None}
    if not isinstance(value, Mapping):
        raise TargetSnapshotError(
            f"best_retained must be a mapping, got {type(value).__name__}."
        )
    unknown = sorted(set(map(str, value)) - _BEST_RETAINED_KEYS)
    if unknown:
        raise TargetSnapshotError(f"best_retained carries unknown keys: {unknown}.")
    if "available" not in value:
        raise TargetSnapshotError("best_retained must record whether one is available.")
    if not coerce:
        missing = sorted(_BEST_RETAINED_KEYS - set(map(str, value)))
        if missing:
            raise TargetSnapshotError(f"best_retained is missing keys: {missing}.")
    available = value["available"]
    if not isinstance(available, bool):
        raise TargetSnapshotError("best_retained.available must be a boolean.")
    epoch = value.get("epoch")
    if isinstance(epoch, np.generic):
        if not coerce:
            raise TargetSnapshotError(
                f"best_retained.epoch is a {type(epoch).__name__}; a validated "
                "snapshot carries plain JSON scalars."
            )
        epoch = epoch.item()
    if epoch is not None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise TargetSnapshotError(
                f"best_retained.epoch must be a non-negative integer or null, "
                f"got {epoch!r}."
            )
        if epochs is not None and epoch > epochs:
            raise TargetSnapshotError(
                f"best_retained.epoch is {epoch}, past the {epochs} epoch(s) this "
                "snapshot covers."
            )
    loss = value.get("loss")
    if isinstance(loss, np.generic):
        if not coerce:
            raise TargetSnapshotError(
                f"best_retained.loss is a {type(loss).__name__}; a validated "
                "snapshot carries plain JSON scalars."
            )
        loss = loss.item()
    if loss is not None:
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            raise TargetSnapshotError(
                f"best_retained.loss must be a number or null, got {loss!r}."
            )
        if coerce:
            loss = _finite_or_none(loss)
        elif not math.isfinite(loss):
            raise TargetSnapshotError(
                f"best_retained.loss must be finite or null, got {loss!r}."
            )
    if not available and (epoch is not None or loss is not None):
        raise TargetSnapshotError(
            "best_retained says no best iterate is available, so it cannot also "
            f"describe one (epoch={epoch!r}, loss={loss!r})."
        )
    return {"available": available, "epoch": epoch, "loss": loss}


def _isoformat_timestamp(moment: object) -> str:
    """An ISO-8601 timestamp that always carries a timezone.

    A clock that returns a naive datetime is a caller misconfiguration, not a
    reason to abort a calibration mid-run, so the naive case is read as UTC —
    which is what :func:`_utc_now`, the only default, produces anyway.
    """
    if not isinstance(moment, datetime):
        raise TargetSnapshotError(
            f"clock() must return a datetime, got {type(moment).__name__}."
        )
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.isoformat()


def _validate_timestamp(value: object) -> None:
    if not isinstance(value, str):
        raise TargetSnapshotError(
            f"created_at must be an ISO-8601 string, got {type(value).__name__}."
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise TargetSnapshotError(
            f"created_at must be an ISO-8601 timestamp, got {value!r}."
        ) from error
    if parsed.tzinfo is None:
        raise TargetSnapshotError(
            f"created_at must carry a timezone offset, got {value!r}."
        )


def _reject_record_level_keys(value: object, *, where: str) -> None:
    """Refuse a record-level key name anywhere in ``value``."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in RECORD_LEVEL_KEY_NAMES:
                raise TargetSnapshotError(
                    f"{where} carries the record-level key {key!r}; calibration "
                    "target snapshots are aggregate-only."
                )
            _reject_record_level_keys(item, where=where)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_record_level_keys(item, where=where)


@dataclass(frozen=True)
class TargetSnapshotCadence:
    """How often a run emits a per-target snapshot.

    ``every`` epochs, plus the first and last epoch of the stretch so a
    consumer always sees where a phase started and where it ended.
    :data:`EVERY_EPOCH` is the explicit every-epoch mode.
    """

    every: int = 25
    include_first: bool = True
    include_last: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.every, bool) or not isinstance(self.every, int):
            raise ValueError(f"every must be an integer, got {self.every!r}.")
        if self.every <= 0:
            raise ValueError(f"every must be positive, got {self.every!r}.")

    def emits(self, epoch: int, epochs: int) -> bool:
        """Whether the 1-based ``epoch`` of an ``epochs``-long stretch emits."""
        if epoch <= 0:
            return False
        if self.include_first and epoch == 1:
            return True
        if self.include_last and epoch == epochs:
            return True
        return epoch % self.every == 0


@dataclass(frozen=True)
class TargetSnapshotObserver:
    """Caller-side snapshot configuration: where snapshots go and how often.

    The observer knows nothing about the compiled target list — a caller
    cannot, before calibration compiles it. :meth:`bind` attaches the ordered
    target identity once the solver has it, producing the
    :class:`BoundTargetSnapshots` the solver emits through.

    ``sink`` follows the existing ``progress_callback`` contract: it receives a
    freshly built dictionary on every call (so mutating a delivered payload can
    never reach the next one or the optimizer), and an exception it raises
    propagates rather than being swallowed — the solver does not guard
    ``progress_callback`` either.
    """

    sink: Callable[[dict[str, object]], None]
    run_id: str
    candidate_id: str | None = None
    cadence: TargetSnapshotCadence = field(default_factory=TargetSnapshotCadence)
    context: Mapping[str, object] = field(default_factory=dict)
    clock: Callable[[], datetime] = _utc_now
    phase: str | None = None
    #: One monotone emission counter per observer, shared by every phase view
    #: and every bind. A multi-phase run's epoch numbers restart, so `sequence`
    #: is what recovers the true emission order across the whole run.
    sequence_counter: list[int] = field(
        default_factory=lambda: [0], repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Bind the metadata contract where a caller can still fix a breach.

        Failing at construction, rather than at the first emission, keeps a
        misconfigured observer from aborting a calibration mid-run.
        """
        object.__setattr__(
            self, "run_id", _identifier(self.run_id, what="run_id", required=True)
        )
        object.__setattr__(
            self,
            "candidate_id",
            _identifier(self.candidate_id, what="candidate_id", required=False),
        )
        object.__setattr__(
            self, "phase", _identifier(self.phase, what="phase", required=False)
        )
        object.__setattr__(
            self, "context", normalize_metadata(self.context, where="context")
        )
        if not callable(self.sink):
            raise TargetSnapshotError("sink must be callable.")
        if not callable(self.clock):
            raise TargetSnapshotError("clock must be callable.")

    def with_phase(self, phase: str) -> TargetSnapshotObserver:
        """A view labelling its snapshots ``phase``, sharing this counter."""
        return replace(self, phase=phase)

    def bind(
        self,
        *,
        names: Sequence[str],
        targets: Sequence[float] | np.ndarray,
    ) -> BoundTargetSnapshots:
        """Bind this observer to a compiled problem's ordered targets."""
        names = tuple(str(name) for name in names)
        values = tuple(_finite_or_none(value) for value in np.asarray(targets))
        return BoundTargetSnapshots(
            observer=self,
            names=names,
            targets=values,
            digest=target_identity_digest(names, values),
            counter=self.sequence_counter,
            phase=self.phase,
        )


@dataclass(frozen=True)
class BoundTargetSnapshots:
    """An observer bound to one compiled target list, with phase/search identity.

    ``phase`` and ``search`` are the identity a multi-phase run and a
    sparse-selection search need: ``(phase, search iteration, epoch, iterate)``
    identifies a snapshot even though epoch numbers restart in every phase and
    in every budget-search probe. ``sequence`` is a monotone counter shared by
    all derived views of one bound observer, so the emission order is
    recoverable even when the identity tuple is not ordered.
    """

    observer: TargetSnapshotObserver
    names: tuple[str, ...]
    targets: tuple[float | None, ...]
    digest: str
    counter: list[int]
    phase: str | None = None
    search: Mapping[str, object] | None = None

    def with_phase(self, phase: str) -> BoundTargetSnapshots:
        """A view whose snapshots are labelled with ``phase``."""
        return replace(self, phase=_identifier(phase, what="phase", required=True))

    def with_search(self, **search: object) -> BoundTargetSnapshots:
        """A view whose snapshots carry a sparse-selection search identity.

        The search identity takes the same closed metadata contract as every
        other seam, checked here so a breach surfaces where the caller set it.
        """
        return replace(self, search=normalize_metadata(search, where="search"))

    @property
    def cadence(self) -> TargetSnapshotCadence:
        return self.observer.cadence

    def emits(self, epoch: int, epochs: int) -> bool:
        return self.cadence.emits(epoch, epochs)

    def snapshot(
        self,
        estimates: np.ndarray,
        *,
        epoch: int,
        epochs: int,
        iterate: str,
        precision: str = "float32",
        loss: float | None = None,
        best_retained: Mapping[str, object] | None = None,
        selection: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Build (and validate) one snapshot payload. Does not call the sink."""
        if iterate not in TARGET_SNAPSHOT_ITERATES:
            raise TargetSnapshotError(
                f"iterate must be one of {TARGET_SNAPSHOT_ITERATES}, got {iterate!r}."
            )
        if precision not in _PRECISIONS:
            raise TargetSnapshotError(
                f"precision must be one of {_PRECISIONS}, got {precision!r}."
            )
        values = np.asarray(estimates, dtype=np.float64).reshape(-1)
        if values.shape[0] != len(self.names):
            raise TargetSnapshotError(
                "estimates must align with the compiled targets: got "
                f"{values.shape[0]}, expected {len(self.names)}."
            )
        rows: list[dict[str, object]] = []
        non_finite = 0
        for index, (name, target) in enumerate(
            zip(self.names, self.targets, strict=True)
        ):
            estimate = _finite_or_none(values[index])
            error = signed_relative_error(estimate, target)
            if error is None:
                non_finite += 1
            rows.append(
                {
                    "index": index,
                    "name": name,
                    "target": target,
                    "estimate": estimate,
                    "relative_error": error,
                }
            )
        self.counter[0] += 1
        payload: dict[str, object] = {
            "schema": TARGET_SNAPSHOT_SCHEMA,
            "schema_version": TARGET_SNAPSHOT_SCHEMA_VERSION,
            "run_id": self.observer.run_id,
            "candidate_id": self.observer.candidate_id,
            "created_at": _isoformat_timestamp(self.observer.clock()),
            "sequence": self.counter[0],
            "phase": self.phase,
            "epoch": int(epoch),
            "epochs": int(epochs),
            "search": normalize_metadata(self.search, where="search", allow_none=True),
            "iterate": iterate,
            "precision": precision,
            "loss": _finite_or_none(loss),
            "non_finite_rows": non_finite,
            "best_retained": normalize_best_retained(best_retained, epochs=int(epochs)),
            "selection": normalize_metadata(
                selection, where="selection", allow_none=True
            ),
            "targets_sha256": self.digest,
            "n_targets": len(self.names),
            "context": normalize_metadata(self.observer.context, where="context"),
            "targets": rows,
        }
        validate_target_snapshot(payload)
        return payload

    def emit(self, estimates: np.ndarray, **kwargs: Any) -> dict[str, object] | None:
        """Build a snapshot and hand it to the sink. Returns what was sent."""
        payload = self.snapshot(estimates, **kwargs)
        self.observer.sink(payload)
        return payload


def validate_target_snapshot(payload: Mapping[str, object]) -> None:
    """Raise :class:`TargetSnapshotError` unless ``payload`` obeys the schema."""
    if not isinstance(payload, Mapping):
        raise TargetSnapshotError(f"a snapshot must be a mapping, got {type(payload)}.")
    keys = set(payload)
    unknown = sorted(keys - _TOP_LEVEL_KEYS)
    if unknown:
        raise TargetSnapshotError(
            f"snapshot carries unknown top-level keys: {unknown}."
        )
    missing = sorted(_TOP_LEVEL_KEYS - keys)
    if missing:
        raise TargetSnapshotError(f"snapshot is missing required keys: {missing}.")
    if payload["schema"] != TARGET_SNAPSHOT_SCHEMA:
        raise TargetSnapshotError(
            f"schema must be {TARGET_SNAPSHOT_SCHEMA!r}, got {payload['schema']!r}."
        )
    if payload["schema_version"] != TARGET_SNAPSHOT_SCHEMA_VERSION:
        raise TargetSnapshotError(
            f"schema_version must be {TARGET_SNAPSHOT_SCHEMA_VERSION}, "
            f"got {payload['schema_version']!r}."
        )
    if payload["iterate"] not in TARGET_SNAPSHOT_ITERATES:
        raise TargetSnapshotError(
            f"iterate must be one of {TARGET_SNAPSHOT_ITERATES}, "
            f"got {payload['iterate']!r}."
        )
    if payload["precision"] not in _PRECISIONS:
        raise TargetSnapshotError(
            f"precision must be one of {_PRECISIONS}, got {payload['precision']!r}."
        )
    _identifier(payload["run_id"], what="run_id", required=True)
    _identifier(payload["candidate_id"], what="candidate_id", required=False)
    _identifier(payload["phase"], what="phase", required=False)
    _validate_timestamp(payload["created_at"])
    for key in ("sequence", "epoch", "epochs", "n_targets"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TargetSnapshotError(f"{key} must be a non-negative integer.")
    # A snapshot describes an epoch of the stretch it declares. The floor is
    # zero, not one: the solver's retain-best rule reports the epoch it
    # actually selected on the `selected` snapshot, and its epoch convention
    # ("completed_optimizer_updates; zero is start") makes zero reachable.
    if payload["epoch"] > payload["epochs"]:
        raise TargetSnapshotError(
            f"epoch is {payload['epoch']} but the snapshot declares "
            f"{payload['epochs']} epoch(s)."
        )
    if payload["sequence"] < 1:
        raise TargetSnapshotError("sequence counts emissions and starts at 1.")
    if payload["n_targets"] < 1:
        raise TargetSnapshotError("a calibration snapshot needs at least one target.")
    _strict_number(payload["loss"], what="loss")
    # Every metadata seam, one contract, checked uniformly.
    normalize_metadata(payload["context"], where="context", coerce=False)
    normalize_metadata(payload["search"], where="search", allow_none=True, coerce=False)
    normalize_metadata(
        payload["selection"], where="selection", allow_none=True, coerce=False
    )
    non_finite_rows = payload["non_finite_rows"]
    if (
        isinstance(non_finite_rows, bool)
        or not isinstance(non_finite_rows, int)
        or non_finite_rows < 0
    ):
        raise TargetSnapshotError("non_finite_rows must be a non-negative integer.")
    best = normalize_best_retained(
        payload["best_retained"], epochs=int(payload["epochs"]), coerce=False
    )
    selection = payload["selection"]
    if payload["iterate"] == ITERATE_SELECTED and isinstance(selection, Mapping):
        selected_epoch = selection.get("selected_epoch")
        if (
            isinstance(selected_epoch, int)
            and not isinstance(selected_epoch, bool)
            and selected_epoch != payload["epoch"]
        ):
            raise TargetSnapshotError(
                f"a selected snapshot is stamped epoch {payload['epoch']} but its "
                f"selection receipt selected epoch {selected_epoch}."
            )
    if not best["available"] and payload["iterate"] == ITERATE_BEST_RETAINED:
        raise TargetSnapshotError(
            "a snapshot cannot be labelled best_retained on a run that retains no "
            "best iterate."
        )

    rows = payload["targets"]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TargetSnapshotError("targets must be a sequence of rows.")
    if len(rows) != payload["n_targets"]:
        raise TargetSnapshotError(
            f"n_targets is {payload['n_targets']} but {len(rows)} rows are present."
        )
    names: list[str] = []
    values: list[float | None] = []
    observed_non_finite = 0
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TargetSnapshotError(f"target row {position} is not a mapping.")
        row_unknown = sorted(set(row) - _TARGET_ROW_KEYS)
        if row_unknown:
            raise TargetSnapshotError(
                f"target row {position} carries unknown keys: {row_unknown}."
            )
        row_missing = sorted(_TARGET_ROW_KEYS - set(row))
        if row_missing:
            raise TargetSnapshotError(
                f"target row {position} is missing keys: {row_missing}."
            )
        if row["index"] != position:
            raise TargetSnapshotError(
                f"target rows must be in compiled order: row at position {position} "
                f"declares index {row['index']!r}."
            )
        name = row["name"]
        if not isinstance(name, str) or not name:
            raise TargetSnapshotError(f"target row {position} needs a non-empty name.")
        target = _strict_number(row["target"], what=f"target for {name!r}")
        estimate = _strict_number(row["estimate"], what=f"estimate for {name!r}")
        stored = _strict_number(
            row["relative_error"], what=f"relative_error for {name!r}"
        )
        expected = signed_relative_error(estimate, target)
        if expected is None:
            observed_non_finite += 1
            if stored is not None:
                raise TargetSnapshotError(
                    f"relative_error for {name!r} must be null when its operands "
                    "or computed error are not finite."
                )
        else:
            if stored is None:
                raise TargetSnapshotError(
                    f"relative_error for {name!r} must be present when its "
                    "computed error is finite."
                )
            if not math.isclose(
                stored, expected, rel_tol=_RELATIVE_ERROR_RTOL, abs_tol=0.0
            ):
                raise TargetSnapshotError(
                    f"relative_error for {name!r} is {stored!r}; the signed relative "
                    f"error of {estimate!r} against {target!r} is {expected!r}."
                )
        names.append(name)
        values.append(target)
    if non_finite_rows > payload["n_targets"]:
        raise TargetSnapshotError(
            f"non_finite_rows is {non_finite_rows}, past this snapshot's "
            f"{payload['n_targets']} target row(s)."
        )
    if observed_non_finite != non_finite_rows:
        raise TargetSnapshotError(
            f"non_finite_rows says {non_finite_rows} but {observed_non_finite} rows "
            "carry a null estimate, target or computed error."
        )
    digest = target_identity_digest(names, values)
    if payload["targets_sha256"] != digest:
        raise TargetSnapshotError(
            "targets_sha256 does not digest this snapshot's ordered target "
            f"identity: declared {payload['targets_sha256']!r}, computed {digest!r}."
        )
    # Scan the WHOLE payload, not a chosen subset: a key added to the schema
    # later must be covered by the aggregate-only rule without anyone
    # remembering to extend a list here.
    _reject_record_level_keys(payload, where="snapshot")
    # A last strictness check: the payload must be strict JSON, so a consumer
    # never receives NaN/Infinity tokens no parser outside Python accepts.
    _canonical_json_bytes(payload)


def _fsync_parent_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Replace ``path`` atomically: a reader sees the old or the new document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent_directory(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class TargetSnapshotWriter:
    """A local snapshot store: atomic ``latest.json`` plus bounded history.

    ``latest.json`` is replaced atomically, so a dashboard polling it either
    reads the previous complete snapshot or the new one — never a truncated
    document. Each snapshot is also written once to
    ``history/<sequence>.json``; those chunks are immutable, so a reader that
    has opened one is never racing a rewrite of it.

    ``history_limit`` bounds what the store *retains* (the oldest chunks are
    dropped once the limit is exceeded); ``history_index.json`` records what is
    retained and how many were dropped, so a bounded history is visibly
    bounded rather than silently lossy. ``history_limit=None`` retains
    everything.
    """

    directory: Path
    history_limit: int | None = 256
    #: Adopt a directory that already holds history chunks. Off by default:
    #: chunk names come from the observer's sequence counter, which restarts
    #: at 1 for every new observer, so a second run sharing a directory would
    #: collide with — and prune — the first run's chunks. Failing at
    #: construction names that at the point a caller can fix it, instead of
    #: mid-calibration.
    allow_existing_history: bool = False
    _dropped: int = field(default=0, init=False, repr=False)
    _written: list[int] = field(default_factory=list, init=False, repr=False)
    _run_id: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.history_limit is not None and (
            isinstance(self.history_limit, bool)
            or not isinstance(self.history_limit, int)
            or self.history_limit <= 0
        ):
            raise ValueError(
                f"history_limit must be a positive integer or None, "
                f"got {self.history_limit!r}."
            )
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.history_directory.mkdir(parents=True, exist_ok=True)
        existing = self.retained()
        if existing and not self.allow_existing_history:
            raise TargetSnapshotError(
                f"{self.history_directory} already holds {len(existing)} snapshot "
                "chunk(s). Snapshot sequences restart at 1 for every observer, so "
                "writing a second run here would collide with and prune the first "
                "run's history. Use a per-run directory, or pass "
                "allow_existing_history=True if you have reconciled the sequences."
            )

    @property
    def history_directory(self) -> Path:
        return self.directory / SNAPSHOT_HISTORY_DIRNAME

    @property
    def latest_path(self) -> Path:
        return self.directory / LATEST_SNAPSHOT_FILENAME

    @property
    def index_path(self) -> Path:
        return self.directory / SNAPSHOT_HISTORY_INDEX_FILENAME

    def retained(self) -> tuple[str, ...]:
        """The retained history chunk filenames, oldest first."""
        return tuple(
            sorted(path.name for path in self.history_directory.glob("*.json"))
        )

    def _write_chunk(self, path: Path, payload: Mapping[str, object]) -> None:
        """Publish a history chunk atomically, exactly once.

        The bytes are written, flushed and ``fsync``-ed to a hidden temporary
        file first; only a *complete* file is then given the immutable
        ``history/<sequence>.json`` name, with :func:`os.link` — which is
        atomic and, unlike :func:`os.replace`, refuses when that name already
        exists rather than overwriting it. So a concurrent reader polling the
        history never opens a chunk mid-write, and a write that fails part-way
        (a full disk, a crashed process) publishes nothing and leaves no
        temporary behind. The temporary is hidden and does not end in
        ``.json``, so neither :meth:`retained` nor :func:`iter_history` can see
        it even while it exists.
        """
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(payload, indent=1, sort_keys=True, allow_nan=False)
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise TargetSnapshotError(
                    f"history chunks are immutable; {path.name} already exists."
                ) from error
        finally:
            temporary.unlink(missing_ok=True)
        _fsync_parent_directory(path)

    def __call__(self, payload: Mapping[str, object]) -> None:
        validate_target_snapshot(payload)
        run_id = str(payload["run_id"])
        if self._run_id is None:
            self._run_id = run_id
        elif run_id != self._run_id:
            raise TargetSnapshotError(
                f"this store belongs to run {self._run_id!r}; it will not accept a "
                f"snapshot from run {run_id!r}. Give each run its own directory."
            )
        sequence = int(payload["sequence"])
        self._write_chunk(self.history_directory / f"{sequence:08d}.json", payload)
        self._written.append(sequence)
        self._prune()
        _atomic_write_json(self.latest_path, dict(payload))
        retained = self.retained()
        _atomic_write_json(
            self.index_path,
            {
                "schema": TARGET_SNAPSHOT_SCHEMA,
                "schema_version": TARGET_SNAPSHOT_SCHEMA_VERSION,
                "run_id": payload["run_id"],
                "history_limit": self.history_limit,
                "retained": list(retained),
                "first_retained_sequence": (
                    int(Path(retained[0]).stem) if retained else None
                ),
                "last_sequence": sequence,
                "dropped": self._dropped,
            },
        )

    def _prune(self) -> int:
        """Drop the oldest chunks THIS writer wrote, past ``history_limit``.

        Pruning is scoped to this writer's own sequences: a directory it was
        told to adopt keeps whatever was already there, rather than this run
        silently deleting another one's evidence.
        """
        if self.history_limit is None:
            return 0
        excess = len(self._written) - self.history_limit
        if excess <= 0:
            return 0
        for sequence in self._written[:excess]:
            (self.history_directory / f"{sequence:08d}.json").unlink(missing_ok=True)
        del self._written[:excess]
        self._dropped += excess
        return excess


def bounded_cadence(every: int) -> TargetSnapshotCadence:
    """A bounded cadence emitting every ``every`` epochs (plus first and last)."""
    return TargetSnapshotCadence(every=every)


def iter_history(directory: Path | str) -> Iterable[dict[str, object]]:
    """Read a store's retained history chunks in sequence order."""
    history = Path(directory) / SNAPSHOT_HISTORY_DIRNAME
    for path in sorted(history.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_target_snapshot(payload)
        yield payload
