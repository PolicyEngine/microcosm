"""The shared gate battery in shadow beside today's US gates.

microcosm#1585 item 11 moves every US gate onto the country-agnostic battery
(:mod:`microcosm.build.gate_battery`, microcosm#611) in four pull requests.
This module is the second one's runtime. Each US build tool keeps gating
exactly as it does today, and beside each gate it evaluates the gate's
declared ``us/gates.json`` entry through one
:class:`~microcosm.build.gate_battery.GateBatteryRun` per tool scope, on the
same evidence, then compares the two verdicts.

Today's code stays authoritative. Its raises, written files, manifest keys
and exit do not change, and the shadow never fails a build: a difference, or
a fault in the shadow itself, is recorded. When the tool finishes, the
scope's battery report and the comparison are written beside its outputs as
shadow files. A verdict difference is then flagged in the comparison
(``verdict_difference``) and printed as a warning, so each build records on
its own inputs whether the battery agrees, while today's gates go on
deciding until the flip (PR 3) makes the battery authoritative. When today's
code raises, the files are written with what the battery saw up to that
point, and today's exception propagates unchanged. The phase it stopped in
is recorded only when every entry there without an outcome is one today's
code never reached, here or in a process whose observations were loaded.

The comparison follows the waiver proposal posted on microcosm#1609, which is
still for Max and María to settle:

- waivers stay the tools' ``--allow-*`` flags, and the battery evaluates every
  entry as if unwaived;
- verdicts are compared, not enforcement: passed or failed, with the same
  failure lines, so a waived gate that fails in both is a match;
- the shadow file records, by entry id, which waiver flags the build used;
- an exception in today's evaluation and a battery outcome carrying
  ``evaluation_error`` count as the same verdict.

Two cases have no verdict to compare. A gate today's code did not evaluate on
this build (a stage that did not run, a gate a flag skips) matches a battery
outcome of ``evidence_absent``. A gate whose threshold the build overrode
with a command-line flag is evaluated by the battery at the declared value,
so a difference there is recorded as explained by the override and is not
flagged as a verdict difference: the spec cannot hold a per-run threshold,
and that decision belongs to the flip (PR 3), not to the shadow.

Every real run arms its tool's shadow. The test suite sets
:data:`US_GATE_SHADOW_SWITCH` to ``off`` for every test, because its tests
stand stubs in for whole stages and a finished shadow would rightly report
the gate sites they skip; the shadow's own tests turn it back on.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import sys
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from microcosm.build.country_spec import GatesManifest
from microcosm.build.gate_battery import (
    EvidenceContext,
    GateBatteryRun,
    GateBinding,
    GateOutcome,
    GateStatus,
    assemble_phase_report,
    blocking_outcomes,
)
from microcosm.build.gates import GateResult

__all__ = [
    "GATE_BATTERY_SHADOW_COMPARISON_SUFFIX",
    "GATE_BATTERY_SHADOW_DIRECTORY",
    "GATE_BATTERY_SHADOW_PRODUCER",
    "GATE_BATTERY_SHADOW_REPORT_SUFFIX",
    "GATE_BATTERY_SHADOW_SCHEMA_VERSION",
    "US_GATE_SHADOW_SWITCH",
    "GateBatteryShadow",
    "LegacyStatus",
    "LegacyVerdict",
    "active_gate_shadow",
    "arm_gate_shadow",
    "compare_verdicts",
    "current_gate_shadow",
    "disarm_gate_shadow",
    "legacy_parameter_overrides",
    "shadow_check",
    "shadow_failures",
    "shadow_file_paths",
    "shadow_gate",
    "shadow_not_run",
    "shadow_value",
    "tool_gate_shadows_enabled",
]

GATE_BATTERY_SHADOW_SCHEMA_VERSION = 1
GATE_BATTERY_SHADOW_PRODUCER = "microcosm.build.us_runtime.gate_shadow"
#: The directory a tool writes its shadow files into, beside its outputs and
#: never inside a release directory, whose files the release manifests and
#: checksum ledgers may pick up.
GATE_BATTERY_SHADOW_DIRECTORY = "gate_battery_shadow"
_OBSERVATIONS_KIND = "us_gate_battery_shadow_observations"
#: ``<scope><suffix>``: the battery's own schema-4 report for the scope.
GATE_BATTERY_SHADOW_REPORT_SUFFIX = ".gate_battery_shadow_report.json"
#: ``<scope><suffix>``: the comparison with today's verdicts.
GATE_BATTERY_SHADOW_COMPARISON_SUFFIX = ".gate_battery_shadow_comparison.json"
#: Set to ``off`` and a build tool runs without its scope's shadow. Real
#: builds never set it; ``packages/microcosm-build/tests/conftest.py`` sets it
#: for every test.
US_GATE_SHADOW_SWITCH = "MICROCOSM_US_GATE_SHADOW"


def tool_gate_shadows_enabled() -> bool:
    """Whether a build tool arms its scope's shadow: unless the switch is ``off``.

    Only the tools ask; a shadow built and armed directly behaves the same
    either way.
    """

    return os.environ.get(US_GATE_SHADOW_SWITCH, "").strip().lower() != "off"


def shadow_file_paths(directory: Path | str, scope: str) -> tuple[Path, Path]:
    """The scope's shadow report and comparison paths inside ``directory``."""

    if not scope or not scope.replace("_", "").isalnum():
        raise ValueError(f"scope must be a plain identifier, got {scope!r}.")
    root = Path(directory)
    return (
        root / f"{scope}{GATE_BATTERY_SHADOW_REPORT_SUFFIX}",
        root / f"{scope}{GATE_BATTERY_SHADOW_COMPARISON_SUFFIX}",
    )


class LegacyStatus(enum.Enum):
    """What today's code concluded about one gate on this build."""

    PASSED = "passed"
    FAILED = "failed"
    #: Today's evaluation raised (a crash, not a measured failure).
    ERROR = "error"
    #: Today's code did not evaluate the gate on this build.
    NOT_RUN = "not_run"


@dataclass(frozen=True)
class LegacyVerdict:
    """Today's verdict for one gate, in the comparator's terms.

    Attributes:
        status: The :class:`LegacyStatus`.
        failures: Today's failure lines, in today's order; empty unless
            ``status`` is ``failed``.
        error: ``"<ExceptionType>: <message>"`` when today's evaluation
            raised.
        reason: Why today's code did not evaluate the gate.
    """

    status: LegacyStatus
    failures: tuple[str, ...] = ()
    error: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, LegacyStatus):
            raise TypeError("status must be a LegacyStatus.")
        if (self.status is LegacyStatus.FAILED) != bool(self.failures):
            raise ValueError("a failed verdict, and only one, names its failures.")
        if (self.status is LegacyStatus.ERROR) != bool(self.error):
            raise ValueError("an error verdict, and only one, names its error.")
        if (self.status is LegacyStatus.NOT_RUN) != bool(self.reason):
            raise ValueError("a gate that did not run, and only one, says why.")

    @classmethod
    def from_result(cls, result: object) -> LegacyVerdict:
        """Today's verdict object: a :class:`~microcosm.build.gates.GateResult`,
        or anything else with ``passed`` and ``failures``, as today's code
        reads it (tests stand in such objects for gates)."""

        passed = getattr(result, "passed", None)
        failures = getattr(result, "failures", None)
        if (
            not isinstance(passed, bool)
            or failures is None
            or isinstance(failures, str)
        ):
            raise TypeError(
                f"expected a gate verdict with passed and failures, got "
                f"{type(result).__name__}; wrap a failure list with "
                "LegacyVerdict.from_failures."
            )
        if passed:
            return cls(status=LegacyStatus.PASSED)
        return cls(
            status=LegacyStatus.FAILED,
            failures=tuple(str(line) for line in failures),
        )

    @classmethod
    def from_failures(cls, failures: Iterable[str]) -> LegacyVerdict:
        """Today's verdict from a gate that returns only failure lines."""

        if isinstance(failures, str):
            raise TypeError("failures must be a sequence of lines, not one string.")
        lines = tuple(str(line) for line in failures)
        if not lines:
            return cls(status=LegacyStatus.PASSED)
        return cls(status=LegacyStatus.FAILED, failures=lines)

    @classmethod
    def from_exception(cls, error: BaseException) -> LegacyVerdict:
        """Today's evaluation raised ``error``."""

        return cls(
            status=LegacyStatus.ERROR,
            error=f"{type(error).__name__}: {error}",
        )

    @classmethod
    def not_run(cls, reason: str) -> LegacyVerdict:
        """Today's code did not evaluate the gate, for ``reason``."""

        return cls(status=LegacyStatus.NOT_RUN, reason=str(reason))

    def to_payload(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "failures": list(self.failures),
            "error": self.error,
            "reason": self.reason,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> LegacyVerdict:
        """The inverse of :meth:`to_payload`."""

        failures = payload.get("failures") or ()
        if isinstance(failures, str) or not isinstance(failures, (list, tuple)):
            raise ValueError("a stored legacy verdict's failures must be a list.")
        return cls(
            status=LegacyStatus(payload["status"]),
            failures=tuple(str(line) for line in failures),
            error=payload.get("error"),  # type: ignore[arg-type]
            reason=payload.get("reason"),  # type: ignore[arg-type]
        )


def _evaluation_error(outcome: GateOutcome) -> Mapping[str, object] | None:
    if outcome.result is None or outcome.result.passed:
        return None
    error = outcome.result.details.get("evaluation_error")
    return error if isinstance(error, Mapping) else None


def compare_verdicts(legacy: LegacyVerdict, outcome: GateOutcome) -> str | None:
    """Whether the battery reached today's verdict; ``None`` when it did.

    The rule is the waiver proposal's (module docstring): verdicts, not
    enforcement. Otherwise the returned sentence says how they differ.
    """

    status = outcome.status
    if legacy.status is LegacyStatus.NOT_RUN:
        if status is GateStatus.EVIDENCE_ABSENT:
            return None
        return (
            f"today's code did not evaluate it ({legacy.reason}), but the battery "
            f"recorded {status.value}."
        )
    if outcome.result is None:
        return (
            f"today's code reached {legacy.status.value}, but the battery "
            f"recorded {status.value}: {outcome.reason}"
        )
    crashed = _evaluation_error(outcome)
    if legacy.status is LegacyStatus.ERROR:
        if crashed is not None:
            return None
        return (
            f"today's evaluation raised {legacy.error}, but the battery's "
            f"evaluation {status.value}."
        )
    if crashed is not None:
        return (
            f"the battery's evaluation raised {crashed.get('type')}: "
            f"{crashed.get('message')}, but today's code {legacy.status.value}."
        )
    if legacy.status is LegacyStatus.PASSED:
        if status is GateStatus.PASSED:
            return None
        first = outcome.result.failures[0] if outcome.result.failures else ""
        return f"today's code passed, but the battery failed: {first}"
    if status is GateStatus.PASSED:
        return f"today's code failed ({legacy.failures[0]}), but the battery passed."
    battery_failures = tuple(str(line) for line in outcome.result.failures)
    if battery_failures != legacy.failures:
        return (
            "both failed, with different failure lines: today "
            f"{list(legacy.failures)}, the battery {list(battery_failures)}."
        )
    return None


def _outcome_payload(outcome: GateOutcome) -> dict[str, object]:
    crashed = _evaluation_error(outcome)
    return {
        "status": outcome.status.value,
        "failures": (
            [str(line) for line in outcome.result.failures]
            if outcome.result is not None
            else []
        ),
        "evaluation_error": dict(crashed) if crashed is not None else None,
        "reason": outcome.reason,
    }


def _outcome_to_wire(outcome: GateOutcome) -> dict[str, object]:
    return {
        **outcome.to_payload(),
        "result_name": outcome.result.name if outcome.result is not None else None,
        "evidence_sha256": outcome.evidence_sha256,
    }


def _outcome_from_wire(
    gates: GatesManifest, entry_id: str, row: Mapping
) -> GateOutcome:
    """Rebuild a stored outcome against the current entry, as replay does."""

    declared = {entry.id: entry for entry in gates.gates}
    entry = declared.get(entry_id)
    if entry is None:
        raise ValueError(f"stored outcome names undeclared entry {entry_id!r}.")
    if any(
        row.get(key) != getattr(entry, key) for key in ("gate", "phase", "criticality")
    ):
        raise ValueError(f"stored outcome for {entry_id!r} differs from its entry.")
    status = GateStatus(row["status"])
    result = None
    if status in (GateStatus.PASSED, GateStatus.FAILED):
        result = GateResult(
            name=str(row["result_name"]),
            passed=status is GateStatus.PASSED,
            failures=tuple(str(line) for line in row.get("failures") or ()),
            details=dict(row.get("details") or {}),
        )
    return GateOutcome(
        entry=entry,
        status=status,
        result=result,
        reason=row.get("reason"),
        evidence_sha256=row.get("evidence_sha256"),
    )


@dataclass(frozen=True)
class _Observation:
    entry_id: str
    site: str | None
    legacy: LegacyVerdict
    outcome: GateOutcome | None
    difference: str | None
    explained_by: tuple[str, ...]


def _disagreement_warning(scope: str, differences: Sequence[str], path: Path) -> str:
    """The warning a finished run prints when the battery disagrees.

    Printed only after both shadow files are on disk. Today's gates still
    decide, so the build's exit is the one they gave.
    """

    lines = "\n".join(f"  - {line}" for line in differences)
    return (
        f"warning: US gate battery shadow for scope {scope!r} disagrees with "
        f"today's gates (comparison: {path}). Today's gates still decide, so "
        f"the build's exit is unchanged:\n{lines}"
    )


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


class GateBatteryShadow:
    """One tool scope's battery, evaluated beside today's gates.

    Construct it before the tool's first gate, observe each gate where it
    runs today (:meth:`gate`, :meth:`failures`, :meth:`check`,
    :meth:`not_run`), and :meth:`finish` once at the end, or use the instance
    as a context manager around the tool's body, which does both ends.

    No observation raises: an unknown entry, a context that cannot be built,
    or any other fault in the shadow is recorded as a shadow error, which
    :meth:`finish` flags and reports like a verdict difference. The battery is
    run with the record-without-enforce posture (``release_candidate`` false,
    never ``enforce``), so every phase the tool reaches is recorded and the
    shadow report is never shippable: it is evidence, not an authority.

    Args:
        gates: The scope's manifest (``us_scoped_gate_manifest``).
        scope: The scope's name, written into both files.
        release_id: The build the report describes.
        report_path: Where the battery writes its schema-4 report.
        comparison_path: Where the comparison is written.
        registry: The scope's bindings.
        waivers: Entry id to the ``--allow-*`` flags this build used that
            waive it today.
        overrides: Entry id to the declared parameters this build's flags
            overrode, each ``{"declared": ..., "runtime": ...}``.
        hold_files: Touch no file until :meth:`release_files`, for a tool
            that must refuse before its first side effect: the shadow neither
            clears an earlier run's files nor writes its own, and a run that
            stops while they are held leaves the directory as it found it.
            A finished run writes them regardless.
    """

    def __init__(
        self,
        gates: GatesManifest,
        *,
        scope: str,
        release_id: str,
        report_path: Path | str,
        comparison_path: Path | str,
        registry: Mapping[str, GateBinding],
        waivers: Mapping[str, Sequence[str]] | None = None,
        overrides: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
        hold_files: bool = False,
    ) -> None:
        declared = {entry.id for entry in gates.gates}
        self._scope = str(scope)
        self._release_id = release_id
        self._gates = gates
        self._comparison_path = Path(comparison_path)
        self._run = GateBatteryRun(
            gates,
            release_id=release_id,
            report_path=report_path,
            release_candidate=False,
            registry=registry,
        )
        self._waivers = {
            str(entry_id): tuple(str(flag) for flag in flags)
            for entry_id, flags in sorted((waivers or {}).items())
            if flags
        }
        self._overrides = {
            str(entry_id): {
                str(name): dict(values) for name, values in sorted(params.items())
            }
            for entry_id, params in sorted((overrides or {}).items())
            if params
        }
        unknown = sorted((set(self._waivers) | set(self._overrides)) - declared)
        if unknown:
            raise ValueError(
                f"waivers or overrides name entries outside scope {scope!r}: {unknown}."
            )
        self._declared = frozenset(declared)
        self._lock = threading.RLock()
        self._observations: dict[str, _Observation] = {}
        # Entries today's code reached, here or in a process whose saved
        # observations were loaded, whatever the shadow made of them: a
        # stopped phase never marks one of them unreached.
        self._reached: set[str] = set()
        # Observations of this scope that are missing: a saved file that could
        # not be read, or a stage that ran and saved none.
        self._observations_missing: list[str] = []
        self._shadow_errors: list[str] = []
        self._finished = False
        self._files_held = bool(hold_files)
        if not self._files_held:
            self._clear_stale_files()

    def _clear_stale_files(self) -> None:
        # A rerun into the same output directory must not leave an earlier
        # attempt's shadow files looking like this run's when it stops before
        # writing its own. Never raises, so arming the shadow cannot stop a
        # build: a fault is recorded as a shadow error.
        try:
            for stale in (self._run.report_path, self._comparison_path):
                stale.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - the shadow never fails a build
            self._shadow_errors.append(
                f"shadow could not clear an earlier run's files: "
                f"{type(exc).__name__}: {exc}"
            )

    def release_files(self) -> None:
        """Let a shadow built with ``hold_files`` clear and write its files.

        Never raises: a fault is recorded as a shadow error.
        """

        with self._lock:
            if not self._files_held:
                return
            self._files_held = False
            self._clear_stale_files()

    # -- properties ---------------------------------------------------------

    @property
    def scope(self) -> str:
        return self._scope

    @property
    def gates(self) -> GatesManifest:
        return self._gates

    @property
    def run(self) -> GateBatteryRun:
        return self._run

    @property
    def report_path(self) -> Path:
        return self._run.report_path

    @property
    def comparison_path(self) -> Path:
        return self._comparison_path

    def declares(self, entry_id: str) -> bool:
        """Whether ``entry_id`` belongs to this scope."""

        return entry_id in self._declared

    def outcome(self, entry_id: str) -> GateOutcome | None:
        """The battery's outcome for an observed entry, if it has one."""

        observation = self._observations.get(entry_id)
        return observation.outcome if observation is not None else None

    def differences(self) -> tuple[str, ...]:
        """Unexplained verdict differences and shadow errors, so far."""

        lines = [
            f"[{observation.entry_id}] {observation.difference}"
            for observation in self._observations.values()
            if observation.difference is not None and not observation.explained_by
        ]
        lines.extend(self._shadow_errors)
        return tuple(lines)

    # -- observation --------------------------------------------------------

    def check(
        self,
        entry_id: str,
        context: EvidenceContext | Callable[[], EvidenceContext],
        legacy: LegacyVerdict,
        *,
        site: str | None = None,
    ) -> GateOutcome | None:
        """Evaluate ``entry_id`` beside today's ``legacy`` verdict.

        ``context`` may be a callable, so building the evidence runs inside
        the shadow's own guard. Returns the battery's outcome, or ``None``
        when the shadow could not evaluate the entry (recorded as a shadow
        error). Never raises.
        """

        with self._lock:
            return self._check(entry_id, context, legacy, site=site)

    def _check(
        self,
        entry_id: str,
        context: EvidenceContext | Callable[[], EvidenceContext],
        legacy: LegacyVerdict,
        *,
        site: str | None,
    ) -> GateOutcome | None:
        try:
            # Today's code reached this entry, whatever the shadow makes of it
            # (as text, so a miswired id cannot stop the stage's save).
            self._reached.add(str(entry_id))
            if self._finished:
                raise RuntimeError("the shadow has already finished.")
            if entry_id in self._observations:
                raise RuntimeError("entry observed more than once.")
            if not isinstance(legacy, LegacyVerdict):
                raise TypeError("legacy must be a LegacyVerdict.")
            evidence = context() if callable(context) else context
            if not isinstance(evidence, EvidenceContext):
                raise TypeError("context must be an EvidenceContext.")
            (outcome,) = self._run.evaluate_entries((entry_id,), evidence)
        except Exception as exc:  # noqa: BLE001 - the shadow never stops the tool
            self._shadow_errors.append(
                f"[{entry_id}] shadow could not evaluate the entry: "
                f"{type(exc).__name__}: {exc}"
            )
            return None
        difference = compare_verdicts(legacy, outcome)
        explained_by: tuple[str, ...] = ()
        if difference is not None and entry_id in self._overrides:
            explained_by = tuple(
                f"runtime override of {name}" for name in self._overrides[entry_id]
            )
        self._observations[entry_id] = _Observation(
            entry_id=entry_id,
            site=site,
            legacy=legacy,
            outcome=outcome,
            difference=difference,
            explained_by=explained_by,
        )
        return outcome

    def gate(
        self,
        entry_id: str,
        evaluate: Callable[[], GateResult],
        context: EvidenceContext | Callable[[], EvidenceContext],
        *,
        site: str | None = None,
    ) -> GateResult:
        """Run today's ``evaluate`` and observe it; return its result as is.

        Today's exception, if it raises, is re-raised unchanged after the
        observation, so the call site behaves exactly as before.
        """

        try:
            result = evaluate()
        except Exception as exc:
            self.check(entry_id, context, LegacyVerdict.from_exception(exc), site=site)
            raise
        try:
            legacy = LegacyVerdict.from_result(result)
        except Exception as exc:  # noqa: BLE001 - recorded, never raised
            self.record_error(
                entry_id,
                f"today's verdict is not comparable: {type(exc).__name__}: {exc}",
            )
            return result
        self.check(entry_id, context, legacy, site=site)
        return result

    def failures(
        self,
        entry_id: str,
        evaluate: Callable[[], Sequence[str]],
        context: EvidenceContext | Callable[[], EvidenceContext],
        *,
        site: str | None = None,
    ) -> Sequence[str]:
        """:meth:`gate` for a gate that returns only failure lines."""

        try:
            lines = evaluate()
        except Exception as exc:
            self.check(entry_id, context, LegacyVerdict.from_exception(exc), site=site)
            raise
        try:
            legacy = LegacyVerdict.from_failures(lines)
        except Exception as exc:  # noqa: BLE001 - recorded, never raised
            self.record_error(
                entry_id,
                f"today's verdict is not comparable: {type(exc).__name__}: {exc}",
            )
            return lines
        self.check(entry_id, context, legacy, site=site)
        return lines

    def record_error(self, entry_id: str, message: str) -> None:
        """Record a fault in the shadow's own wiring for ``entry_id``.

        A finished run flags and reports it like a verdict difference; it
        never raises.
        """

        with self._lock:
            self._shadow_errors.append(f"[{entry_id}] {message}")

    def value(
        self,
        entry_id: str,
        evaluate: Callable[[], Any],
        context: EvidenceContext | Callable[[], EvidenceContext],
        *,
        verdict: Callable[[Any], LegacyVerdict],
        refusals: tuple[type[BaseException], ...] = (),
        site: str | None = None,
    ) -> Any:
        """:meth:`gate` for a check that returns something else.

        ``verdict`` reads today's verdict from the value; an exception of a
        ``refusals`` type is the check's own failure, its message the line.
        Either way today's value, or exception, comes back unchanged.
        """

        try:
            result = evaluate()
        except refusals as refusal:
            self.check(
                entry_id,
                context,
                LegacyVerdict.from_failures([str(refusal)]),
                site=site,
            )
            raise
        except Exception as exc:
            self.check(entry_id, context, LegacyVerdict.from_exception(exc), site=site)
            raise
        try:
            legacy = verdict(result)
        except Exception as exc:  # noqa: BLE001 - recorded, never raised
            self.record_error(
                entry_id,
                f"today's verdict is not comparable: {type(exc).__name__}: {exc}",
            )
            return result
        self.check(entry_id, context, legacy, site=site)
        return result

    def not_run(self, entry_id: str, reason: str, *, site: str | None = None) -> None:
        """Record that today's code did not evaluate ``entry_id`` on this build.

        The battery evaluates it on no evidence, which must leave it
        ``evidence_absent``.
        """

        self.check(
            entry_id,
            EvidenceContext(),
            LegacyVerdict.not_run(reason),
            site=site,
        )

    # -- carrying observations between processes -----------------------------

    def save_observations(self, path: Path | str) -> None:
        """Write this process's observations for a later process to load.

        A tool whose stages run as separate processes (the base builder's
        checkpointed stages, ACS local's ``--stage`` runs) observes each
        stage's gates in its own process and leaves them here; the process
        that finishes the scope loads every stage's file. The file is bound
        to the scope's manifest digest, so a later process with another
        policy refuses it instead of mixing two. It also names every entry
        today's code reached here, including those the shadow could not
        evaluate, so a later process never marks one of them unreached.
        """

        with self._lock:
            payload = {
                "kind": _OBSERVATIONS_KIND,
                "schema_version": GATE_BATTERY_SHADOW_SCHEMA_VERSION,
                "producer": GATE_BATTERY_SHADOW_PRODUCER,
                "scope": self._scope,
                "gates_manifest_sha256": self._run.gates_manifest_sha256,
                "reached": sorted(self._reached),
                "observations": [
                    {
                        "id": observation.entry_id,
                        "site": observation.site,
                        "legacy": observation.legacy.to_payload(),
                        "outcome": (
                            _outcome_to_wire(observation.outcome)
                            if observation.outcome is not None
                            else None
                        ),
                    }
                    for observation in self._observations.values()
                ],
                "shadow_errors": list(self._shadow_errors),
            }
        _atomic_write_json(Path(path), payload)

    def load_observations(self, path: Path | str) -> None:
        """Add the observations another process saved. Never raises.

        Each stored observation is compared again under this shadow's own
        rule and overrides. A file for another scope or manifest, an entry
        already observed, or an unreadable file is a shadow error; a file
        not read whole also counts as missing observations (see
        :meth:`note_missing_observations`).
        """

        source = Path(path)
        with self._lock:
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
                if (
                    payload.get("kind") != _OBSERVATIONS_KIND
                    or payload.get("schema_version")
                    != GATE_BATTERY_SHADOW_SCHEMA_VERSION
                ):
                    raise ValueError("not a shadow observations file.")
                if payload.get("scope") != self._scope:
                    raise ValueError(f"observations of scope {payload.get('scope')!r}.")
                if (
                    payload.get("gates_manifest_sha256")
                    != self._run.gates_manifest_sha256
                ):
                    raise ValueError("observations made under another gate manifest.")
                reached = payload.get("reached")
                if not isinstance(reached, list) or not all(
                    isinstance(entry_id, str) for entry_id in reached
                ):
                    raise ValueError("reached must be a list of entry ids.")
                self._reached.update(reached)
                rows = payload.get("observations")
                if not isinstance(rows, list):
                    raise ValueError("observations must be a list.")
                for row in rows:
                    self._load_row(row, source)
                for line in payload.get("shadow_errors") or ():
                    self._shadow_errors.append(f"{source.name}: {line}")
            except Exception as exc:  # noqa: BLE001 - recorded, never raised
                self._shadow_errors.append(
                    f"could not load shadow observations {source}: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._observations_missing.append(f"{source.name} was not read whole")

    def note_missing_observations(self, source: str) -> None:
        """Note that a process which may have reached this scope's entries left
        no observations here, such as a stage that ran and saved none.

        It is not a difference by itself: an entry that process observed is
        then unobserved, which a finished run flags. A stopped run cannot tell
        that process's entries from the ones never reached, so it leaves the
        phase it stopped in unrecorded. Never raises.
        """

        with self._lock:
            self._observations_missing.append(str(source))

    def _load_row(self, row: Mapping, source: Path) -> None:
        entry_id = str(row["id"])
        # Another process observed it, so today's code reached it there.
        self._reached.add(entry_id)
        if entry_id in self._observations:
            self._shadow_errors.append(
                f"[{entry_id}] observed again in {source.name}; kept the first."
            )
            return
        legacy = LegacyVerdict.from_payload(row["legacy"])
        stored = row.get("outcome")
        outcome = (
            _outcome_from_wire(self._gates, entry_id, stored)
            if stored is not None
            else None
        )
        if outcome is None:
            # The process that made it could not evaluate it; its own shadow
            # error travels with the file.
            return
        difference = compare_verdicts(legacy, outcome)
        explained_by: tuple[str, ...] = ()
        if difference is not None and entry_id in self._overrides:
            explained_by = tuple(
                f"runtime override of {name}" for name in self._overrides[entry_id]
            )
        self._observations[entry_id] = _Observation(
            entry_id=entry_id,
            site=row.get("site"),
            legacy=legacy,
            outcome=outcome,
            difference=difference,
            explained_by=explained_by,
        )

    # -- the end --------------------------------------------------------------

    def __enter__(self) -> GateBatteryShadow:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._finished:
            # The tool finished the shadow itself (a dry run's deliberate
            # stop), so its own exit stands.
            return
        if exc is None:
            self.finish()
            return
        self.stop(exc)

    def stop(self, error: BaseException) -> None:
        """Finish a run today's code stopped with ``error``; never raises.

        Today's exception is the one that must propagate, so a fault while
        writing the shadow files only warns.
        """

        try:
            self.finish(stopped_by=error)
        except Exception as fault:  # noqa: BLE001 - never replace today's error
            print(
                f"warning: US gate battery shadow for scope {self._scope!r} "
                f"could not finish: {type(fault).__name__}: {fault}",
                file=sys.stderr,
            )

    def finish(
        self,
        *,
        stopped_by: BaseException | None = None,
        stopped_at: str | None = None,
    ) -> dict[str, object]:
        """Record the phases, write both shadow files, and report.

        It never fails the tool: today's gates decide until the flip, so the
        tool's own exit stands whatever the battery found. A finished run
        (``stopped_by`` and ``stopped_at`` both ``None``) must have observed
        every entry of its scope. An unexplained difference, an unobserved
        entry or a shadow error is flagged in the comparison
        (``verdict_difference``) and printed as a warning after the files are
        written. A run today's code stopped early (``stopped_by``: its
        exception) or that stops on purpose before its last phase
        (``stopped_at``: why, as the release's dry run does) records the
        phases it completed, plus the phase it stopped in when a battery
        outcome there blocks. That phase names unreached the entries without
        an outcome, so it is recorded only when today's code reached none of
        them and no observations are missing; otherwise it stays unrecorded
        rather than mark a gate that ran as never reached. A shadow file that
        cannot be written is a warning too.

        Returns:
            The comparison payload written to disk, or an empty mapping when a
            run stopped while its files were held, which writes nothing.
        """

        with self._lock:
            return self._finish(stopped_by=stopped_by, stopped_at=stopped_at)

    def _finish(
        self,
        *,
        stopped_by: BaseException | None,
        stopped_at: str | None,
    ) -> dict[str, object]:
        if self._finished:
            raise RuntimeError("the shadow has already finished.")
        self._finished = True
        completed = stopped_by is None and stopped_at is None
        if self._files_held:
            if not completed:
                # The tool stopped before it let the shadow write, as on a
                # refusal ahead of its first side effect: an earlier run's
                # files stay exactly as they were.
                print(
                    f"note: US gate battery shadow for scope {self._scope!r} wrote "
                    "no files: the run stopped before the tool let it write.",
                    file=sys.stderr,
                )
                return {}
            self._files_held = False
            self._clear_stale_files()
        recorded: list[str] = []
        not_observed: list[str] = []
        try:
            not_observed = self._not_observed()
            # Phases recorded before a fault stay listed: the run has already
            # written them to its report.
            self._record_phases(recorded, completed=completed)
        except Exception as exc:  # noqa: BLE001 - written down, decided below
            self._shadow_errors.append(
                f"shadow could not record its phases: {type(exc).__name__}: {exc}"
            )
        if completed:
            # A process that saved no observations (a stage run with the
            # shadow off, or before it existed) explains an unobserved entry
            # better than its site does, so it is named instead.
            cause = (
                "; ".join(self._observations_missing)
                + ", so it may have run there unrecorded"
                if self._observations_missing
                else "its site is not wired"
            )
            for entry_id in not_observed:
                self._shadow_errors.append(
                    f"[{entry_id}] today's run finished without observing this "
                    f"entry; {cause}."
                )
        payload = self._comparison_payload(
            recorded=recorded,
            not_observed=not_observed,
            completed=completed,
            stopped_by=stopped_by,
            stopped_at=stopped_at,
        )
        try:
            _atomic_write_json(self._comparison_path, payload)
        except Exception as exc:  # noqa: BLE001 - the shadow never fails a build
            # A shadow file that cannot be written must neither fail the build
            # nor replace today's own exception (or deliberate stop).
            print(
                f"warning: US gate battery shadow for scope {self._scope!r} "
                f"could not write {self._comparison_path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return payload
        differences = self.differences()
        if completed and differences:
            print(
                _disagreement_warning(self._scope, differences, self._comparison_path),
                file=sys.stderr,
            )
        return payload

    def _not_observed(self) -> list[str]:
        """Applicable entries without a battery outcome, phase by phase."""

        return [
            entry.id
            for phase in self._gates.phases
            for entry in self._gates.gates
            if entry.phase == phase
            and entry.not_applicable is None
            and self.outcome(entry.id) is None
        ]

    def _record_phases(self, recorded: list[str], *, completed: bool) -> None:
        """Assemble and record each phase the tool reached, in order.

        Each recorded phase is appended to ``recorded`` as soon as the run
        has written it, so a later fault leaves the list true to the report.
        """

        for phase in self._gates.phases:
            entries = [entry for entry in self._gates.gates if entry.phase == phase]
            ran = [
                entry
                for entry in entries
                if entry.not_applicable is None and self.outcome(entry.id) is not None
            ]
            missing = [
                entry.id
                for entry in entries
                if entry.not_applicable is None and self.outcome(entry.id) is None
            ]
            outcomes = [self.outcome(entry.id) for entry in ran]
            if not missing:
                self._run.record_phase(
                    assemble_phase_report(self._gates, phase, outcomes)
                )
                recorded.append(phase)
                continue
            if completed or not ran:
                return
            # The tool stopped inside this phase, and its entries without an
            # outcome would be named unreached. That is untrue of one today's
            # code reached but the shadow could not evaluate (here or in the
            # process that saved it), and may be untrue when a stage's
            # observations are missing, so then the phase stays unrecorded.
            # The run itself refuses a phase that marks unreached an entry it
            # evaluated, but it sees only this process's stages.
            if self._observations_missing or any(
                entry_id in self._reached for entry_id in missing
            ):
                return
            # Otherwise name the last observed entry that blocks as the
            # blocker, as the battery would have stopped there; without one
            # the phase stays unrecorded.
            observed_order = [
                observation.entry_id
                for observation in self._observations.values()
                if observation.outcome is not None
                and observation.outcome.entry.phase == phase
            ]
            blocking = {
                outcome.entry.id
                for outcome in blocking_outcomes(outcomes, release_candidate=False)
            }
            blockers = [entry_id for entry_id in observed_order if entry_id in blocking]
            if not blockers:
                return
            self._run.record_phase(
                assemble_phase_report(
                    self._gates,
                    phase,
                    outcomes,
                    blocked_by=blockers[-1],
                    unreached=missing,
                    release_candidate=False,
                )
            )
            recorded.append(phase)
            return

    def _comparison_payload(
        self,
        *,
        recorded: list[str],
        not_observed: list[str],
        completed: bool,
        stopped_by: BaseException | None,
        stopped_at: str | None,
    ) -> dict[str, object]:
        report_path = self._run.report_path
        report_written = bool(recorded) and report_path.is_file()
        entries = []
        for observation in self._observations.values():
            outcome = observation.outcome
            entry = outcome.entry if outcome is not None else None
            entries.append(
                {
                    "id": observation.entry_id,
                    "gate": entry.gate if entry is not None else None,
                    "phase": entry.phase if entry is not None else None,
                    "criticality": entry.criticality if entry is not None else None,
                    "site": observation.site,
                    "legacy": observation.legacy.to_payload(),
                    "battery": (
                        _outcome_payload(outcome) if outcome is not None else None
                    ),
                    "waived_by": list(self._waivers.get(observation.entry_id, ())),
                    "match": observation.difference is None,
                    "difference": observation.difference,
                    "explained_by": list(observation.explained_by),
                }
            )
        unexplained = [
            {"id": observation.entry_id, "difference": observation.difference}
            for observation in self._observations.values()
            if observation.difference is not None and not observation.explained_by
        ]
        explained = [
            {
                "id": observation.entry_id,
                "difference": observation.difference,
                "explained_by": list(observation.explained_by),
            }
            for observation in self._observations.values()
            if observation.difference is not None and observation.explained_by
        ]
        return {
            "schema_version": GATE_BATTERY_SHADOW_SCHEMA_VERSION,
            "producer": GATE_BATTERY_SHADOW_PRODUCER,
            "country": self._gates.country,
            "scope": self._scope,
            "authority": "legacy",
            "release_id": self._release_id,
            "gates_manifest_sha256": self._run.gates_manifest_sha256,
            "policy_sha256": self._run.policy_sha256,
            "spec_fingerprint": self._run.spec_fingerprint,
            "report_file": report_path.name if report_written else None,
            "report_sha256": _sha256_file(report_path) if report_written else None,
            "phases_recorded": recorded,
            "blocked_at_phase": self._run.blocked_at_phase,
            "run_completed": completed,
            "stopped_by": (
                {"type": type(stopped_by).__name__, "message": str(stopped_by)}
                if stopped_by is not None
                else None
            ),
            "stopped_at": stopped_at,
            "waivers": {key: list(value) for key, value in self._waivers.items()},
            "overrides": self._overrides,
            "entries": entries,
            "not_observed": not_observed,
            "observations_missing": list(self._observations_missing),
            "differences": unexplained,
            "explained_differences": explained,
            "shadow_errors": list(self._shadow_errors),
            "verdict_difference": bool(unexplained or self._shadow_errors),
        }


# ---------------------------------------------------------------------------
# The shadow a tool has armed, for the gate sites to find
# ---------------------------------------------------------------------------

_ACTIVE_LOCK = threading.Lock()
_ACTIVE: list[GateBatteryShadow] = []


def arm_gate_shadow(shadow: GateBatteryShadow) -> None:
    """Arm ``shadow`` until :func:`disarm_gate_shadow`; prefer the context
    manager, :func:`active_gate_shadow`, where one block holds the run."""

    with _ACTIVE_LOCK:
        _ACTIVE.append(shadow)


def disarm_gate_shadow(shadow: GateBatteryShadow) -> None:
    """Disarm a shadow :func:`arm_gate_shadow` armed; a no-op otherwise."""

    with _ACTIVE_LOCK:
        if shadow in _ACTIVE:
            _ACTIVE.remove(shadow)


@contextmanager
def active_gate_shadow(shadow: GateBatteryShadow) -> Iterator[GateBatteryShadow]:
    """Arm ``shadow`` for the gate sites while the block runs.

    The sites call :func:`shadow_gate` and its siblings, which observe through
    the innermost armed shadow that declares the entry. Nothing is armed for
    a tool that imports another tool's gate functions to re-run them (the
    release's scorers, probe and dry-run grading), so those calls run exactly
    as before.
    """

    with _ACTIVE_LOCK:
        _ACTIVE.append(shadow)
    try:
        yield shadow
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE.remove(shadow)


def current_gate_shadow(entry_id: str | None = None) -> GateBatteryShadow | None:
    """The innermost armed shadow, or the innermost declaring ``entry_id``."""

    with _ACTIVE_LOCK:
        armed = list(_ACTIVE)
    for shadow in reversed(armed):
        if entry_id is None or shadow.declares(entry_id):
            return shadow
    return None


def shadow_gate(
    entry_id: str,
    evaluate: Callable[[], GateResult],
    context: EvidenceContext | Callable[[], EvidenceContext],
    *,
    site: str | None = None,
) -> GateResult:
    """Run today's gate, observing it when a shadow declaring it is armed.

    Returns today's result unchanged, and re-raises today's exception
    unchanged; with no shadow armed this is ``evaluate()``.
    """

    shadow = current_gate_shadow(entry_id)
    if shadow is None:
        return evaluate()
    return shadow.gate(entry_id, evaluate, context, site=site)


def shadow_failures(
    entry_id: str,
    evaluate: Callable[[], Sequence[str]],
    context: EvidenceContext | Callable[[], EvidenceContext],
    *,
    site: str | None = None,
) -> Sequence[str]:
    """:func:`shadow_gate` for a gate that returns failure lines."""

    shadow = current_gate_shadow(entry_id)
    if shadow is None:
        return evaluate()
    return shadow.failures(entry_id, evaluate, context, site=site)


def shadow_value(
    entry_id: str,
    evaluate: Callable[[], Any],
    context: EvidenceContext | Callable[[], EvidenceContext],
    *,
    verdict: Callable[[Any], LegacyVerdict],
    refusals: tuple[type[BaseException], ...] = (),
    site: str | None = None,
) -> Any:
    """:func:`shadow_gate` for a check that returns another value."""

    shadow = current_gate_shadow(entry_id)
    if shadow is None:
        return evaluate()
    return shadow.value(
        entry_id, evaluate, context, verdict=verdict, refusals=refusals, site=site
    )


def shadow_check(
    entry_id: str,
    context: EvidenceContext | Callable[[], EvidenceContext],
    legacy: LegacyVerdict,
    *,
    site: str | None = None,
) -> None:
    """Observe a verdict today's code reached by itself, when armed."""

    shadow = current_gate_shadow(entry_id)
    if shadow is not None:
        shadow.check(entry_id, context, legacy, site=site)


def shadow_not_run(entry_id: str, reason: str, *, site: str | None = None) -> None:
    """Record, when armed, that today's code did not evaluate ``entry_id``."""

    shadow = current_gate_shadow(entry_id)
    if shadow is not None:
        shadow.not_run(entry_id, reason, site=site)


def legacy_parameter_overrides(
    declared: Mapping[str, Mapping[str, Any]],
    runtime: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, object]]]:
    """The declared parameters a build's flags overrode, by entry id.

    ``declared`` and ``runtime`` map entry ids to parameter values; a
    parameter whose runtime value differs from the declared one is an
    override. Values compare as JSON would see them.
    """

    overrides: dict[str, dict[str, dict[str, object]]] = {}
    for entry_id, values in runtime.items():
        for name, value in values.items():
            expected = declared.get(entry_id, {}).get(name)
            if json.dumps(expected, sort_keys=True) != json.dumps(
                value, sort_keys=True
            ):
                overrides.setdefault(entry_id, {})[name] = {
                    "declared": expected,
                    "runtime": value,
                }
    return overrides
