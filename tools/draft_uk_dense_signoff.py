"""Draft the gate-only sign-off between the two dense UK builds (microcosm#1301).

A dense release candidate (``--release-role dense --release-candidate``)
blocked only by failures a reviewer can sign without touching the solve is
released by a second build at the sign-off's head. Three gate-only registers
take those sign-offs:

* ``uk_target_fit``: national target rows (``name@2025``), in
  ``target_fit_reviewed_exclusions.json``;
* ``uk_local_target_fit``: local target cells (``family/area_code/metric``),
  in ``local_target_fit_reviewed_exclusions.json`` (microcosm#1301).
  A checkout whose ``gates.json`` wires no such register cannot sign them;
* ``uk_local_area_support``: areas below the support floor, in the
  ``exclusions`` block of ``local_area_support_exclusions.json``.

Since the gate-only area-support change (microcosm#1301), the local reference
generator keeps its own support-floor constants, lockstep-tested against the
area-support register:
``SUPPORT_FLOOR_BOUND_FAMILIES`` against the ``bound_despite_support_floor``
block, and ``SUPPORT_FLOOR_DEFERRED_AREA_IDS`` against the exclusions of the
areas whose cells it defers. So the tool never edits that block, and never
retires or renews those areas' exclusions: retiring one moves the generator,
and so the solve, and a renewal keeps the entry's reviewed wording and lines,
so a reviewer signs it by hand. Nor does it touch the degenerate register,
whose digest the release contract pins.

Draft, after build 1::

    python tools/draft_uk_dense_signoff.py --run-dir <build-1> \\
      --approved-by <reviewer> --approved-on <YYYY-MM-DD> [--gate-clock <date>]

It reads the build's full gate report, its signed local gate report, its
``*.targets.csv`` and ``*.area_support.csv`` and its Logbook row, and checks
them against one another and against this checkout's ``gates.json``. It then
drafts a ``release_lines: ["local"]`` entry for every signable failure, the
removal of every entry build 2 would fail (stale, unknown or out of its
window), and the test pins those registers move (``test_uk_terminal_gates.py``
for the two target-fit registers, ``test_uk_weighted_integrity.py`` for the
area exclusions). A national target-fit entry that applies on every line is
never removed or replaced, since the national line may still stand on it:
when build 2 would fail it on the local line, it is scoped to the national
line, unchanged otherwise. An expired one also goes to review, for the
national line to renew or retire, and a row build 1 still breaches under such
an entry needs a hand sign-off, since one key holds one entry. Any other
release-blocking failure makes the build unsignable: nothing is drafted, and
a third build is needed.

The receipt and the PR body give the assembly deadline: the earliest expiry
among the measure exclusions build 1's solve ran under (as its run directory
receipts them), the local line's degenerate-register entries and the in-line
gate-only entries after the sign-off. Build 2's gate-clock range ends there,
and a ``--gate-clock`` past the solve-scoped expiries is refused: renewing
either kind of entry changes the solve.

Verify, after build 2::

    python tools/draft_uk_dense_signoff.py --verify <build-2> --build-1 <build-1> \\
      [--draft-receipt <out>/signoff_draft.json]

It checks that build 2 stood on every in-line entry of the three registers at
this checkout and that none was stale, dormant, premature, expired or unknown.
It checks that no release-blocking gate failed and that the signed local report
passes the dense release contract, its disclosure walk included (the signing
key must be exported). It fails once the assembly deadline has passed and
warns from two days before it. With
``--build-1`` it also reports whether build 2 reused build 1's solve and
reproduced its per-row errors.

Outputs go to ``--out``, which must lie outside every git work tree:

* ``signoff.patch``: the registers, the pins and a changelog fragment. Apply it
  with ``git apply`` on a branch from build 1's commit.
* ``signoff_draft.json``: the receipt.
* ``PR_BODY.md``: entry counts, gate ids and policy dates.
* ``PRIVATE_review.csv``: mode 0600, never committed or posted.

Nothing is committed, pushed or posted.

Disclosure (UKDS): the patch, the receipt and the PR body carry target and
area identifiers, counts, policy constants, the reviewer's login and their
free text. Reasons are templated and never quote an error, an estimate or a
support figure, and the reports' failure lines are never copied.
``--approved-by`` must be a GitHub login. The free text (``--class-note``,
``--area-name``, ``--adjudication``) may carry digits only in issue references
and ISO dates, and no ``%``; the drafted entries then get the release
contract's disclosure walk. Of the stale-risk and near-bound counts, the
receipt and the PR body carry the national register's only: the local
registers' bracket named sub-GOR estimates, so they go to standard output
and, with per-row errors and per-area support figures, to the private review
file. The area-support table's ``max_weight`` column is never read.
"""

from __future__ import annotations

import argparse
import ast
import calendar
import difflib
import fnmatch
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.country_spec import load_country_spec
from microcosm.build.gates import area_support_below_floor, area_support_gate
from microcosm.build.uk_runtime.graph_national import UK_DENSE_UNBOUND_RELEASE_ID
from microcosm.build.uk_runtime.graph_terminal import decode_full_gate_report
from microcosm.build.uk_runtime.terminal_gates import uk_target_fit_gate
from microcosm.build.uk_runtime.weighted_integrity import (
    UKReviewedExclusion,
    coerce_reviewed_exclusions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
UK_RESOURCES = Path("packages/microcosm-build/src/microcosm/build/uk")
TERMINAL_GATES_TEST = Path(
    "packages/microcosm-build/tests/engine_free/uk/test_uk_terminal_gates.py"
)
BATTERY_BINDINGS_TEST = Path(
    "packages/microcosm-build/tests/engine_free/uk/test_uk_battery_bindings.py"
)
WEIGHTED_INTEGRITY_TEST = Path(
    "packages/microcosm-build/tests/engine_free/uk/test_uk_weighted_integrity.py"
)
#: The test files searched for a register's key-set pin
#: (``assert register == {}`` or ``assert set(register) == {...}``).
PIN_SEARCH = (TERMINAL_GATES_TEST, BATTERY_BINDINGS_TEST, WEIGHTED_INTEGRITY_TEST)
#: The local reference generator, whose own support-floor deferrals
#: (microcosm#1301) keep their area-support exclusions.
LOCAL_REFERENCE_GENERATOR = Path("tools/generate_uk_local_target_references.py")
DEFERRED_AREAS_CONSTANT = "SUPPORT_FLOOR_DEFERRED_AREA_IDS"
#: The repository's ruff line length, which rewritten pins respect.
LINE_LENGTH = 88

#: ``full_build_cli.FULL_GATE_REPORT_FILENAME`` (lockstep-tested).
FULL_GATE_REPORT_FILENAME = "uk.full.gates.calibrated.gate_report.json"
LOCAL_REPORT_SUFFIX = ".local_gates.json"
#: ``rowwise_cli.MANIFEST_FILENAME`` (lockstep-tested): the candidate manifest,
#: whose measure exclusions assembly and upload validate on the day they run.
CANDIDATE_MANIFEST_FILENAME = "rowwise_candidate_manifest.json"
#: The target compilation's surface as the build materializes its evidence
#: (``<node>.<artifact>.json``, lockstep-tested). The manifest's measure
#: exclusions are projected from it, and a build that its gates stopped
#: before the export leaves only this.
TARGET_SURFACE_FILENAME = "uk.full.target_compilation.surface.json"
#: The degenerate register: solve-scoped, since the release contract pins its
#: digest, yet the calibrated gates evaluate it on build 2's gate clock.
DEGENERATE_REGISTER = UK_RESOURCES / "degenerate_reviewed_exclusions.json"
LOCAL_LINE = "local"
NATIONAL_LINE = "national"
#: The geography levels whose targets the local surface materializes
#: (``full_gates._LOCAL_LEVELS``); every other row is a national row.
LOCAL_LEVELS = frozenset({"constituency", "local_authority", "la"})

FIT_WINDOW_DAYS = 28
AREA_WINDOW_MONTHS = 3
MAX_WINDOW_DAYS = 92
STALE_RISK_MARGIN = 0.01
#: ``--verify`` warns from this many days before the assembly deadline.
DEADLINE_WARNING_DAYS = 2

#: The only digits reviewer free text may carry (:func:`figure_free`): issue
#: references (``#1301``, ``microcosm#1301``, ``PolicyEngine/microcosm#1301``,
#: whose owner and repository names hold no digits) and ISO dates of real days.
_ISSUE_REFERENCE_RE = re.compile(
    r"(?:[A-Za-z][A-Za-z_.-]*/)?(?:[A-Za-z][A-Za-z_.-]*)?#[0-9]+"
)
_ISO_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
#: A GitHub login, the only form ``--approved-by`` takes (:func:`github_login`).
_GITHUB_LOGIN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}")

#: Release-blocking gates whose entries are signed by hand in the sign-off PR.
HAND_SIGNED_GATES = {
    "uk_input_mass_parity": (
        "input-mass entries are signed by hand, with the contract's "
        "_UK_GATE_BATTERY_INPUT_MASS_EVIDENCE_SHA256 re-pin"
    ),
}
#: The full gate report's artifact keys that identify build 1's solve.
SOLVE_ARTIFACTS = ("problem", "solution", "result")
IDENTITY_ARTIFACTS = (*SOLVE_ARTIFACTS, "selection", "holdout")
_TARGET_KEY_COLUMNS = ("name", "family", "geography_level", "area_code", "metric")
_SUPPORT_COLUMNS = (
    "geography_level",
    "area_code",
    "nonzero_households",
    "nonzero_source_households",
    "effective_sample_size",
)


@dataclass(frozen=True)
class RegisterSpec:
    """One gate-only register the tool may draft into.

    ``loader`` is the function, in ``loader_module``, whose result the
    register's key-set pin asserts on; ``pin_block`` is the block the pin
    subscripts when the loader returns the whole register.
    """

    gate_id: str
    path: Path
    parameter: str
    kind: str
    loader: str | None = None
    loader_module: str = "microcosm.build.uk_runtime.terminal_gates"
    pin_block: str | None = None
    block: str = "exclusions"

    @property
    def resource(self) -> str:
        return self.path.name


#: The local target-fit register (microcosm#1301). Older checkouts
#: have no such file, and their ``gates.json`` wires none, so local cells are
#: unsignable there.
LOCAL_TARGET_FIT_REGISTER = UK_RESOURCES / "local_target_fit_reviewed_exclusions.json"

REGISTERS = (
    RegisterSpec(
        gate_id="uk_target_fit",
        path=UK_RESOURCES / "target_fit_reviewed_exclusions.json",
        parameter="reviewed_exclusions_resource",
        kind="national_row",
        loader="uk_default_target_fit_reviewed_exclusions",
    ),
    RegisterSpec(
        gate_id="uk_local_target_fit",
        path=LOCAL_TARGET_FIT_REGISTER,
        parameter="reviewed_exclusions_resource",
        kind="local_cell",
        loader="uk_default_local_target_fit_reviewed_exclusions",
    ),
    RegisterSpec(
        gate_id="uk_local_area_support",
        path=UK_RESOURCES / "local_area_support_exclusions.json",
        parameter="exclusions_resource",
        kind="area",
        loader="load_uk_local_area_support_exclusion_register",
        loader_module="microcosm.build.uk_runtime.weighted_integrity",
        pin_block="exclusions",
    ),
)
SIGNABLE_GATES = frozenset(spec.gate_id for spec in REGISTERS)


class InputError(Exception):
    """An input the tool cannot draft from or verify against (exit 2)."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InputError(f"{label} is not readable JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise InputError(f"{label} must be a JSON object: {path}")
    return payload


def _iso(value: str, label: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise InputError(f"{label} must be an ISO date, got {value!r}.") from error
    if parsed.isoformat() != value:
        raise InputError(f"{label} must be an ISO date, got {value!r}.")
    return parsed


def _add_months(day: date, months: int) -> date:
    month = day.month - 1 + months
    year = day.year + month // 12
    month = month % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def register_wiring(gates=None) -> dict[str, str | None]:
    """Per register, ``None`` when this checkout's gates read it, else why not."""

    gates = load_country_spec("uk").gates if gates is None else gates
    entries = {entry.id: entry for entry in gates.gates}
    wiring: dict[str, str | None] = {}
    for spec in REGISTERS:
        entry = entries.get(spec.gate_id)
        if entry is None:
            wiring[spec.gate_id] = f"gates.json declares no {spec.gate_id}"
        elif dict(entry.parameters or {}).get(spec.parameter) != spec.resource:
            wiring[spec.gate_id] = (
                f"this checkout's gates.json wires no {spec.resource} into "
                f"{spec.gate_id} ({spec.parameter}); its failures need a solve "
                "change or a checkout that reads the register"
            )
        else:
            wiring[spec.gate_id] = None
    return wiring


def generator_deferred_areas(repo_root: Path) -> frozenset[str] | None:
    """The area keys whose own cells the checkout's local generator defers.

    ``SUPPORT_FLOOR_DEFERRED_AREA_IDS`` of the local reference generator at
    ``repo_root``, as ``local_authority/<code>`` keys, read from its source
    rather than imported. ``None`` when the generator declares no such
    constant: before the gate-only area-support change (microcosm#1301) it
    derived its deferrals from the area-support register itself, so signing
    an area changed the solve.
    """

    path = repo_root / LOCAL_REFERENCE_GENERATOR
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise InputError(
            f"cannot read the local reference generator {LOCAL_REFERENCE_GENERATOR}."
        ) from error
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        if node.value is None or not any(
            isinstance(target, ast.Name) and target.id == DEFERRED_AREAS_CONSTANT
            for target in targets
        ):
            continue
        try:
            codes = ast.literal_eval(node.value)
        except ValueError as error:
            raise InputError(
                f"{LOCAL_REFERENCE_GENERATOR} {DEFERRED_AREAS_CONSTANT} is not a "
                "literal."
            ) from error
        return frozenset(f"local_authority/{code}" for code in codes)
    return None


# ---------------------------------------------------------------------------
# Build evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildEvidence:
    """One run directory, its four inputs checked against one another, and the
    measure exclusions its solve ran under."""

    run_dir: Path
    attempt_id: str
    document: Mapping[str, Any]
    enforcement: Mapping[str, Any]
    outcomes: Mapping[str, Mapping[str, Any]]
    parameters: Mapping[str, Mapping[str, Any]]
    local_report: Mapping[str, Any]
    national_errors: Mapping[str, float]
    local_errors: Mapping[str, float]
    area_support: pd.DataFrame
    sha256: Mapping[str, str]
    clock: date
    dev_build: bool
    #: The measure exclusions the build's solve ran under, name to expiry, and
    #: the file that receipts them (:func:`receipted_measure_exclusions`).
    measure_exclusions: Mapping[str, date]
    measure_exclusions_from: str

    def details(self, gate_id: str) -> Mapping[str, Any]:
        return self.outcomes.get(gate_id, {}).get("details") or {}

    def status(self, gate_id: str) -> str | None:
        return self.outcomes.get(gate_id, {}).get("status")

    @property
    def area_floors(self) -> dict[str, float | int]:
        parameters = self.parameters["uk_local_area_support"]
        return {
            "minimum_rows": int(parameters["minimum_rows"]),
            "minimum_effective_sample_size": float(
                parameters["minimum_effective_sample_size"]
            ),
            "minimum_distinct_sources": int(parameters["minimum_distinct_sources"]),
        }

    def fit_bound(self, gate_id: str) -> float:
        return float(self.parameters[gate_id]["max_abs_relative_error"])

    def errors_for(self, spec: RegisterSpec) -> Mapping[str, float]:
        return (
            self.national_errors if spec.kind == "national_row" else self.local_errors
        )

    def identity(self) -> dict[str, Any]:
        artifacts = self.document.get("artifacts") or {}
        return {
            "attempt_id": self.attempt_id,
            "sha256": dict(self.sha256),
            "gates_manifest_sha256": self.document["report"]["gates_manifest_sha256"],
            "exclusions_evaluated_on": self.clock.isoformat(),
            "release_candidate": bool(self.document.get("release_candidate")),
            "sample_fraction": float(self.document.get("sample_fraction", 0.0)),
            "artifacts": {name: artifacts.get(name) for name in IDENTITY_ARTIFACTS},
            "measure_exclusions_from": self.measure_exclusions_from,
        }


def receipted_measure_exclusions(run_dir: Path) -> tuple[dict[str, date], str]:
    """The measure exclusions a dense run's solve ran under: expiries and source.

    Assembly and upload validate the candidate manifest's
    ``measure_exclusions`` on the day they run, so their earliest expiry is
    the last day a release of this solve can be assembled; renewing one
    changes the solve. The manifest is read when the build exported one.
    Otherwise the target compilation's materialized surface, from which the
    manifest projects them, is: every run that reached the calibrated gates
    leaves it, blocked or not.
    """

    for filename in (CANDIDATE_MANIFEST_FILENAME, TARGET_SURFACE_FILENAME):
        path = run_dir / filename
        if not path.is_file():
            continue
        records = _read_json(path, filename).get("measure_exclusions")
        if not isinstance(records, Mapping):
            raise InputError(f"{filename} receipts no measure_exclusions mapping.")
        expiries = {}
        for name, record in records.items():
            if not isinstance(record, Mapping):
                raise InputError(
                    f"{filename} measure exclusion {name!r} is not a record."
                )
            expiries[str(name)] = _iso(
                record.get("expires_on"), f"{filename} {name!r} expires_on"
            )
        return expiries, filename
    raise InputError(
        f"the run directory has neither {CANDIDATE_MANIFEST_FILENAME} nor "
        f"{TARGET_SURFACE_FILENAME}, so the measure exclusions its solve ran "
        "under are not receipted."
    )


def _one(paths: Sequence[Path], label: str) -> Path:
    if len(paths) != 1:
        raise InputError(f"expected exactly one {label}, found {len(paths)}.")
    return paths[0]


def _logbook_attempt(run_dir: Path) -> str:
    rows = sorted((run_dir / "logbook-spool").glob("*.json"))
    if not rows:
        raise InputError("the run directory carries no Logbook row (logbook-spool).")
    ids = {str(_read_json(path, "Logbook row").get("build_id") or "") for path in rows}
    ids.discard("")
    if len(ids) != 1:
        raise InputError(
            f"the Logbook spool names {len(ids)} attempts; expected exactly one."
        )
    return ids.pop()


def _phase_outcomes(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    outcomes: dict[str, Mapping[str, Any]] = {}
    phases = [*(document.get("upstream_phase_reports") or {}).values()]
    phases.append(document["report"])
    for phase in phases:
        for row in phase.get("outcomes") or ():
            outcomes[str(row["id"])] = row
    return outcomes


def _read_targets(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=[*_TARGET_KEY_COLUMNS, "relative_error"],
        dtype={column: str for column in _TARGET_KEY_COLUMNS},
        keep_default_na=False,
        float_precision="round_trip",
    )
    frame["relative_error"] = pd.to_numeric(frame["relative_error"], errors="raise")
    return frame


def _read_support(path: Path) -> pd.DataFrame:
    # Only the floor columns: the table's max_weight is a single record's
    # weight and is never read.
    header = pd.read_csv(path, nrows=0).columns
    columns = [*_SUPPORT_COLUMNS]
    if "support_below_floor" in header:
        columns.append("support_below_floor")
    return pd.read_csv(
        path,
        usecols=columns,
        dtype={"geography_level": str, "area_code": str},
        keep_default_na=False,
        float_precision="round_trip",
    )


def _check_targets(frame: pd.DataFrame, embedded: object) -> None:
    if not isinstance(embedded, list) or not embedded:
        raise InputError("the full gate report embeds no target diagnostics.")
    expected = pd.DataFrame(embedded)
    if len(expected) != len(frame) or any(
        column not in expected for column in (*_TARGET_KEY_COLUMNS, "relative_error")
    ):
        raise InputError("targets.csv does not match the full gate report's rows.")
    for column in _TARGET_KEY_COLUMNS:
        if list(expected[column].astype(str)) != list(frame[column].astype(str)):
            raise InputError(
                f"targets.csv column {column!r} does not match the full gate report."
            )
    if not np.array_equal(
        expected["relative_error"].to_numpy(dtype=float),
        frame["relative_error"].to_numpy(dtype=float),
    ):
        raise InputError(
            "targets.csv relative errors do not match the full gate report."
        )


def _check_support(frame: pd.DataFrame, embedded: object) -> None:
    if not isinstance(embedded, list) or not embedded:
        raise InputError("the full gate report embeds no area-support table.")
    expected = pd.DataFrame(embedded)
    columns = [column for column in frame.columns if column in expected]
    if len(expected) != len(frame) or set(_SUPPORT_COLUMNS) - set(columns):
        raise InputError("area_support.csv does not match the full gate report.")
    for column in columns:
        left, right = expected[column], frame[column]
        if column in ("geography_level", "area_code"):
            same = list(left.astype(str)) == list(right.astype(str))
        else:
            same = np.array_equal(
                left.to_numpy(dtype=float), right.to_numpy(dtype=float)
            )
        if not same:
            raise InputError(
                f"area_support.csv column {column!r} does not match the full "
                "gate report."
            )


def _split_errors(
    frame: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    national: dict[str, float] = {}
    local: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        error = float(row.relative_error)
        if not math.isfinite(error):
            raise InputError("targets.csv carries a non-finite relative error.")
        if row.geography_level in LOCAL_LEVELS:
            key = f"{row.family.strip()}/{row.area_code.strip()}/{row.metric.strip()}"
            target = local
        else:
            key = row.name
            target = national
        if key in target:
            raise InputError(f"targets.csv repeats the gate key {key!r}.")
        target[key] = error
    return national, local


def _check_fit_details(
    gate_id: str,
    outcome: Mapping[str, Any],
    errors: Mapping[str, float],
    bound: float,
) -> None:
    if outcome.get("status") not in ("passed", "failed"):
        return
    details = outcome.get("details") or {}
    if details.get("targets_checked") != len(errors):
        raise InputError(
            f"{gate_id} checked a different number of rows than targets.csv holds."
        )
    breaching = {key for key, error in errors.items() if abs(error) > bound}
    failing = set(details.get("failing_targets") or {})
    excused = set(details.get("reviewed_exclusions") or {})
    if failing | excused != breaching:
        raise InputError(
            f"{gate_id}'s failing and excused rows differ from targets.csv's "
            "rows beyond the bound."
        )


def load_build(run_dir: Path, *, allow_dev_build: bool = False) -> BuildEvidence:
    """Read and cross-check one dense run directory."""

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise InputError(f"run directory not found: {run_dir}")
    report_path = run_dir / FULL_GATE_REPORT_FILENAME
    if not report_path.is_file():
        raise InputError(f"the run directory has no {FULL_GATE_REPORT_FILENAME}.")
    payload = report_path.read_bytes()
    try:
        report, enforcement = decode_full_gate_report(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise InputError(
            "the full gate report does not decode against this checkout's "
            "gates.json; run the tool from a checkout whose gates.json matches "
            f"the build ({type(error).__name__}: {error})"
        ) from error
    document = json.loads(payload)
    if report.phase != "terminal":
        raise InputError("the full gate report is not the terminal phase's.")
    local_path = _one(
        sorted(run_dir.glob(f"*{LOCAL_REPORT_SUFFIX}")), f"*{LOCAL_REPORT_SUFFIX}"
    )
    stem = local_path.name[: -len(LOCAL_REPORT_SUFFIX)]
    targets_path = run_dir / f"{stem}.targets.csv"
    support_path = run_dir / f"{stem}.area_support.csv"
    for path in (targets_path, support_path):
        if not path.is_file():
            raise InputError(f"the run directory has no {path.name}.")
    local_report = _read_json(local_path, "local gate report")
    attempt_id = _logbook_attempt(run_dir)
    release_id = local_report.get("release_id")
    if release_id == UK_DENSE_UNBOUND_RELEASE_ID:
        raise InputError("the local gate report was bound to no Logbook attempt.")
    if release_id != attempt_id:
        raise InputError(
            "the local gate report's release_id is not the Logbook attempt id."
        )
    dev_build = not (
        document.get("release_candidate") is True
        and float(document.get("sample_fraction", 0.0)) == 1.0
        and local_report.get("release_candidate") is True
        and local_report.get("posture") == "local_candidate"
    )
    if dev_build and not allow_dev_build:
        raise InputError(
            "the build is not a full-sample local release candidate; pass "
            "--allow-dev-build for a dry run (nothing is signable)."
        )
    outcomes = _phase_outcomes(document)
    local_gates = local_report.get("gates")
    if not isinstance(local_gates, Mapping) or not local_gates:
        raise InputError("the local gate report carries no gates.")
    for gate_id, entry in local_gates.items():
        full = outcomes.get(str(gate_id))
        if full is None or any(
            entry.get(key) != full.get(key) for key in ("status", "failures", "details")
        ):
            raise InputError(
                f"the local gate report's {gate_id} differs from the full gate report."
            )
    targets = _read_targets(targets_path)
    _check_targets(targets, document.get("target_diagnostics"))
    support = _read_support(support_path)
    _check_support(support, document.get("area_support"))
    parameters = {
        outcome.entry.id: dict(outcome.entry.parameters or {})
        for outcome in report.outcomes
    }
    missing = sorted(SIGNABLE_GATES - set(parameters))
    if missing:
        raise InputError(f"the terminal report has no outcome for {missing}.")
    national, local = _split_errors(targets)
    for gate_id, errors in (
        ("uk_target_fit", national),
        ("uk_local_target_fit", local),
    ):
        outcome = outcomes[gate_id]
        if (outcome.get("details") or {}).get("evaluation_error") is not None:
            raise InputError(f"{gate_id} failed closed; there is nothing to sign.")
        bound = float(parameters[gate_id]["max_abs_relative_error"])
        _check_fit_details(gate_id, outcome, errors, bound)
    if (outcomes["uk_local_area_support"].get("details") or {}).get(
        "evaluation_error"
    ) is not None:
        raise InputError(
            "uk_local_area_support failed closed; there is nothing to sign."
        )
    clock_text = (outcomes["uk_target_fit"].get("details") or {}).get(
        "exclusions_evaluated_on"
    )
    if not isinstance(clock_text, str):
        raise InputError("uk_target_fit recorded no exclusions_evaluated_on clock.")
    measure_exclusions, measure_exclusions_from = receipted_measure_exclusions(run_dir)
    evidence = BuildEvidence(
        run_dir=run_dir,
        attempt_id=attempt_id,
        document=document,
        enforcement=enforcement,
        outcomes=outcomes,
        parameters=parameters,
        local_report=local_report,
        national_errors=national,
        local_errors=local,
        area_support=support,
        sha256={
            "full_gate_report": hashlib.sha256(payload).hexdigest(),
            "local_gate_report": _sha256(local_path),
            "targets_csv": _sha256(targets_path),
            "area_support_csv": _sha256(support_path),
        },
        clock=_iso(clock_text, "uk_target_fit exclusions_evaluated_on"),
        dev_build=dev_build,
        measure_exclusions=measure_exclusions,
        measure_exclusions_from=measure_exclusions_from,
    )
    below = area_support_below_floor(support, **evidence.area_floors)
    if "support_below_floor" in support and not np.array_equal(
        below.to_numpy(dtype=bool), support["support_below_floor"].to_numpy(dtype=bool)
    ):
        raise InputError(
            "area_support.csv's support_below_floor disagrees with the gate's floors."
        )
    return evidence


def failing_release_gates(evidence: BuildEvidence) -> list[str]:
    """Release-blocking gates that did not pass, in every evaluated phase."""

    enforcement = evidence.enforcement
    return list(
        dict.fromkeys(
            [
                *enforcement.get("enforced_blocking", ()),
                *enforcement.get("unenforced_release_failures", ()),
            ]
        )
    )


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """The approval window every drafted entry carries."""

    approved_by: str
    adjudication: str
    approved_on: date
    expires_on: date
    area_expires_on: date
    gate_clock: date

    def entry(self, reason: str, *, area: bool) -> dict[str, Any]:
        expires = self.area_expires_on if area else self.expires_on
        payload = {
            "reason": reason,
            "approved_by": self.approved_by,
            "adjudication": self.adjudication,
            "approved_on": self.approved_on.isoformat(),
            "expires_on": expires.isoformat(),
            "release_lines": [LOCAL_LINE],
        }
        UKReviewedExclusion(
            **{**payload, "release_lines": tuple(payload["release_lines"])}
        )
        return payload


@dataclass
class RegisterDraft:
    """What the sign-off does to one register."""

    spec: RegisterSpec
    before: dict[str, Any]
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    add: list[str] = field(default_factory=list)
    renew: list[str] = field(default_factory=list)
    keep: list[str] = field(default_factory=list)
    remove: list[str] = field(default_factory=list)
    #: Unscoped national entries scoped to the national line rather than
    #: removed (:func:`draft_target_fit`).
    rescope: list[str] = field(default_factory=list)
    review: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    held: list[str] = field(default_factory=list)
    stale_risk: list[str] = field(default_factory=list)
    near_bound: list[str] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)

    def after(self) -> dict[str, Any]:
        block = {
            key: value for key, value in self.before.items() if key not in self.remove
        }
        for key in self.rescope:
            block[key] = {**self.before[key], "release_lines": [NATIONAL_LINE]}
        for key in self.renew:
            block[key] = self.entries[key]
        for key in sorted(self.add):
            block[key] = self.entries[key]
        return block

    @property
    def changed(self) -> bool:
        return bool(self.add or self.renew or self.remove or self.rescope)

    def summary(self) -> dict[str, Any]:
        """The receipt's record of the register: keys and actions.

        Only the national register's stale-risk and near-bound counts are
        recorded. A local cell signed just beyond the bound has its key in the
        patch, so a count of such cells next to it brackets a named sub-GOR
        estimate (microcosm#1301); the local counts go to the private review
        file and the tool's standard output (:meth:`fit_bands`) only.
        """

        summary: dict[str, Any] = {
            "register": self.spec.path.as_posix(),
            "add": sorted(self.add),
            "renew": sorted(self.renew),
            "keep": sorted(self.keep),
            "remove": sorted(self.remove),
            "rescope": sorted(self.rescope),
            "review": sorted(self.review),
            "conflicts": sorted(self.conflicts),
            "held": sorted(self.held),
            "notes": dict(sorted(self.notes.items())),
        }
        if self.spec.kind == "national_row":
            summary["stale_risk_count"] = len(self.stale_risk)
            summary["near_bound_count"] = len(self.near_bound)
        return summary

    def fit_bands(self) -> dict[str, int]:
        """Stale-risk and near-bound counts, for the operator's terminal."""

        return {"stale_risk": len(self.stale_risk), "near_bound": len(self.near_bound)}


def _in_force(record: UKReviewedExclusion, clock: date) -> bool:
    return not record.expired(clock) and not record.premature(clock)


def _rescope(
    draft: RegisterDraft,
    key: str,
    record: UKReviewedExclusion,
    clock: date,
    *,
    breaching: bool,
) -> None:
    """Keep an entry the national line may stand on, scoped to that line.

    The national target-fit register is shared with the national line, so the
    dense sign-off never removes or replaces an entry that applies there: it
    scopes it to the national line, unchanged otherwise. An expired one also
    goes to review, since only the national line can renew or retire it. A
    row build 1 still breaches is then left without a local entry, because
    one key holds one entry, and needs a hand sign-off.
    """

    draft.rescope.append(key)
    if record.expired(clock):
        state = f"it expired {record.expires_on}"
    elif record.premature(clock):
        state = f"it takes force {record.approved_on}, after the gate clock"
    else:
        state = (
            "dense build 1 fits the row inside the bound, so build 2 would "
            "fail it as stale"
        )
    note = f"kept for the national line and scoped to it, unchanged otherwise: {state}"
    if record.expired(clock):
        draft.review.append(key)
        note += "; the national line must renew or retire it"
    if breaching:
        draft.conflicts.append(key)
        note += (
            "; build 1 breaches the bound on the row, and one key holds one "
            "entry, so no local-only entry can be drafted beside it: sign the "
            "row by hand"
        )
        if record.premature(clock):
            note += f", or choose a gate clock on or after {record.approved_on}"
    draft.notes[key] = note


def _classify_failing(
    draft: RegisterDraft,
    failing: Iterable[str],
    records: Mapping[str, UKReviewedExclusion],
    *,
    clock: date,
    entry: Callable[[str], dict[str, Any]],
    shared: bool = False,
) -> None:
    """Sign the failing keys; ``shared`` marks the national register."""

    for key in sorted(failing):
        record = records.get(key)
        if record is None:
            draft.add.append(key)
            draft.entries[key] = entry(key)
        elif not record.applies_on(LOCAL_LINE):
            draft.conflicts.append(key)
            draft.notes[key] = (
                "the key holds an entry for another release line; sign it by hand"
            )
        elif _in_force(record, clock):
            draft.keep.append(key)
        elif shared and record.applies_on(NATIONAL_LINE):
            _rescope(draft, key, record, clock, breaching=True)
        else:
            draft.renew.append(key)
            draft.entries[key] = entry(key)
            if record.release_lines is None:
                draft.notes[key] = "renewed for the local line only"


def draft_target_fit(
    spec: RegisterSpec,
    errors: Mapping[str, float],
    block: Mapping[str, Any],
    *,
    bound: float,
    window: Window,
    reason: Callable[[str], str],
    margin: float = STALE_RISK_MARGIN,
) -> RegisterDraft:
    """Sign every row beyond the bound; retire what build 2 would fail.

    The national register is shared with the national line, so an entry there
    that also applies on that line is never removed or replaced
    (:func:`_rescope`): the national line reads it as before, and the local
    line sets it aside.
    """

    records = coerce_reviewed_exclusions(block, label=spec.resource)
    draft = RegisterDraft(spec=spec, before=dict(block))
    breaching = {key for key, error in errors.items() if abs(error) > bound}
    clock = window.gate_clock
    shared = spec.kind == "national_row"
    _classify_failing(
        draft,
        breaching,
        records,
        clock=clock,
        entry=lambda key: window.entry(reason(key), area=False),
        shared=shared,
    )
    for key, record in records.items():
        if key in breaching or not record.applies_on(LOCAL_LINE):
            continue
        local_only = record.release_lines == (LOCAL_LINE,)
        if not (key in errors or not _in_force(record, clock) or local_only):
            # An unscoped entry no local row earns may still serve the
            # national line; a reviewer decides.
            draft.review.append(key)
            if shared:
                draft.notes[key] = (
                    "no local row earns it, and the national line may still stand on it"
                )
        elif shared and not local_only:
            _rescope(draft, key, record, clock, breaching=False)
        else:
            # Stale, out of its window, or a local entry no row earns: build 2
            # fails the first two and the third is dead on its own line. Of
            # the national register only local entries get here, and the
            # national line reads no local cell.
            draft.remove.append(key)
    signed = {*draft.add, *draft.renew, *draft.keep}
    draft.stale_risk = sorted(
        key for key in signed if abs(errors[key]) <= bound + margin
    )
    draft.near_bound = sorted(
        key
        for key, error in errors.items()
        if key not in breaching and abs(error) > bound - margin
    )
    return draft


def _missed_floors(row: Any, floors: Mapping[str, float | int]) -> tuple[str, ...]:
    missed = []
    if row.nonzero_households < floors["minimum_rows"]:
        missed.append(f"{floors['minimum_rows']}-row")
    if row.effective_sample_size < floors["minimum_effective_sample_size"]:
        missed.append(f"ESS-{float(floors['minimum_effective_sample_size']):g}")
    if row.nonzero_source_households < floors["minimum_distinct_sources"]:
        missed.append(f"{floors['minimum_distinct_sources']}-source")
    return tuple(missed)


def area_shortfalls(
    support: pd.DataFrame, floors: Mapping[str, float | int]
) -> dict[str, tuple[str, ...]]:
    """Area key -> the floors it misses (names only), for areas below a floor."""

    below: dict[str, tuple[str, ...]] = {}
    for row in support.itertuples(index=False):
        missed = _missed_floors(row, floors)
        if missed:
            below[f"{row.geography_level}/{row.area_code}"] = missed
    return below


def draft_area_support(
    spec: RegisterSpec,
    support: pd.DataFrame,
    register: Mapping[str, Any],
    *,
    floors: Mapping[str, float | int],
    window: Window,
    reason: Callable[[str, tuple[str, ...]], str],
    deferred: frozenset[str] = frozenset(),
) -> tuple[RegisterDraft, list[str]]:
    """Sign every area below a floor; retire stale and unknown entries.

    The entries of ``deferred`` areas, whose cells the local reference
    generator defers on its own constants, are never retired or renewed: one
    that build 2 would fail is held (``draft.held``) and the gate is
    unsignable until a reviewer acts on it.

    Returns the draft and the ``bound_despite_support_floor`` entries out of
    their window at the gate clock: the gate fails on them, and the tool never
    edits that block, whose families are the generator's constants.
    """

    block = register[spec.block]
    records = coerce_reviewed_exclusions(block, label=spec.resource)
    draft = RegisterDraft(spec=spec, before=dict(block))
    below = area_shortfalls(support, floors)
    roster = {
        f"{level}/{code}"
        for level, code in zip(
            support["geography_level"], support["area_code"], strict=True
        )
    }
    clock = window.gate_clock
    _classify_failing(
        draft,
        below,
        records,
        clock=clock,
        entry=lambda key: window.entry(reason(key, below[key]), area=True),
    )
    for key in [key for key in draft.renew if key in deferred]:
        draft.renew.remove(key)
        draft.entries.pop(key)
        draft.held.append(key)
        draft.notes[key] = (
            "out of its window at the gate clock; the local reference generator "
            "defers this area's cells, so renew the entry by hand, keeping its "
            "reviewed wording and lines, and draft again"
        )
    for key, record in records.items():
        if key in below or not record.applies_on(LOCAL_LINE):
            continue
        if key in deferred:
            draft.held.append(key)
            draft.notes[key] = (
                "stale: the area meets every floor, but the local reference "
                f"generator defers its cells ({DEFERRED_AREAS_CONSTANT}); "
                "retiring the entry moves the generator, and so the solve"
            )
            continue
        draft.remove.append(key)
        draft.notes[key] = (
            "stale: the area meets every floor"
            if key in roster
            else "unknown to the area roster"
        )
    bound = coerce_reviewed_exclusions(
        register.get("bound_despite_support_floor") or {},
        label=f"{spec.resource} bound_despite_support_floor",
    )
    blockers = sorted(
        name for name, record in bound.items() if not _in_force(record, clock)
    )
    return draft, blockers


def _scoped(block: Mapping[str, Any], label: str) -> dict[str, UKReviewedExclusion]:
    return {
        key: record
        for key, record in coerce_reviewed_exclusions(block, label=label).items()
        if record.applies_on(LOCAL_LINE)
    }


def predict(
    evidence: BuildEvidence,
    drafts: Mapping[str, RegisterDraft],
    *,
    clock: date,
    bound_despite: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Re-run the three gates on build 1's evidence with the drafted registers.

    The verdict build 2 earns if it reproduces build 1's rows exactly.
    """

    verdicts: dict[str, dict[str, Any]] = {}
    for gate_id, draft in drafts.items():
        spec = draft.spec
        records = _scoped(draft.after(), spec.resource)
        if spec.kind == "area":
            invalid = [
                key
                for key, record in {
                    **records,
                    **coerce_reviewed_exclusions(
                        bound_despite or {}, label="bound_despite_support_floor"
                    ),
                }.items()
                if not _in_force(record, clock)
            ]
            if invalid:
                verdicts[gate_id] = {"passes": False, "failure_count": len(invalid)}
                continue
            support = evidence.area_support
            parameters = evidence.parameters[gate_id]
            levels = [str(level) for level in parameters["geography_levels"]]
            result = area_support_gate(
                support,
                area_roster={
                    level: support.loc[
                        support["geography_level"] == level, "area_code"
                    ].tolist()
                    for level in levels
                },
                geography_levels=levels,
                reviewed_exclusions={
                    key: record.policy_payload() for key, record in records.items()
                },
                **evidence.area_floors,
            )
        else:
            result = uk_target_fit_gate(
                evidence.errors_for(spec),
                reviewed_exclusions=records,
                now=clock,
                max_abs_relative_error=evidence.fit_bound(gate_id),
            )
        verdicts[gate_id] = {
            "passes": bool(result.passed),
            "failure_count": len(result.failures),
        }
    return verdicts


# ---------------------------------------------------------------------------
# Deadline
# ---------------------------------------------------------------------------


def _expiries(records: Mapping[str, UKReviewedExclusion]) -> dict[str, date]:
    return {
        key: date.fromisoformat(record.expires_on) for key, record in records.items()
    }


def degenerate_expiries(repo_root: Path) -> dict[str, date]:
    """The local line's degenerate-register entries at ``repo_root``, by expiry.

    The register is solve-scoped, so build 2 stands on build 1's entries, and
    its calibrated gates evaluate them on the gate clock: one out of its
    window there fails the gate, and renewing it changes the solve.
    """

    payload = _read_json(repo_root / DEGENERATE_REGISTER, DEGENERATE_REGISTER.name)
    try:
        records = _scoped(payload.get("exclusions") or {}, DEGENERATE_REGISTER.name)
    except (TypeError, ValueError) as error:
        raise InputError(
            f"{DEGENERATE_REGISTER.name} does not load: {error}"
        ) from error
    return _expiries(records)


def assembly_deadline(
    sources: Mapping[str, Mapping[str, date]],
) -> dict[str, Any] | None:
    """The earliest expiry a release of build 1's solve depends on.

    ``sources`` maps a source (``measure_exclusion``, ``degenerate``, a
    gate-only register's gate id or block) to its entries' expiries. Build
    2's gate clock may not be later, and assembly and upload, which check
    the measure exclusions on the day they run, may not be either. Returns
    the date and the ``source:name`` of every entry expiring on it.
    """

    dated = [
        (day, f"{source}:{name}")
        for source, entries in sources.items()
        for name, day in entries.items()
    ]
    if not dated:
        return None
    earliest = min(day for day, _ in dated)
    return {
        "date": earliest.isoformat(),
        "set_by": sorted(label for day, label in dated if day == earliest),
    }


# ---------------------------------------------------------------------------
# Reasons
# ---------------------------------------------------------------------------


def _citation(evidence: BuildEvidence) -> str:
    return (
        f"dense build-1 attempt {evidence.attempt_id} (full gate report sha256 "
        f"{evidence.sha256['full_gate_report']})"
    )


def _class_note(key: str, notes: Sequence[tuple[str, str]]) -> str:
    return "".join(
        f" {text}" for pattern, text in notes if fnmatch.fnmatch(key, pattern)
    )


def fit_reason(
    spec: RegisterSpec,
    key: str,
    evidence: BuildEvidence,
    notes: Sequence[tuple[str, str]] = (),
) -> str:
    subject = "national target row" if spec.kind == "national_row" else "local cell"
    bound = evidence.fit_bound(spec.gate_id)
    return (
        f"Gate-only sign-off of the {subject} {key} on the local line "
        f"(microcosm#1301). On {_citation(evidence)} it breached the target-fit "
        f"release bound (max_abs_relative_error {bound:g}). It stays bound in "
        "the solve: this entry suppresses only the release fence on the local "
        "line, and a build in which it is back inside the bound fails the entry "
        "as stale." + _class_note(key, notes)
    )


_LEVEL_LABELS = {
    "local_authority": "local-authority",
    "la": "local-authority",
    "constituency": "constituency",
}


def area_reason(
    key: str,
    missed: Sequence[str],
    evidence: BuildEvidence,
    *,
    names: Mapping[str, str],
    notes: Sequence[tuple[str, str]] = (),
) -> str:
    level, _, code = key.partition("/")
    name = names.get(code)
    area = f"{name} ({code})" if name else code
    floors = (
        missed[0] if len(missed) == 1 else ", ".join(missed[:-1]) + f" and {missed[-1]}"
    )
    plural = "s" if len(missed) > 1 else ""
    return (
        f"Gate-only sign-off of the {_LEVEL_LABELS.get(level, level)} support "
        f"floor for {area} on the local line (microcosm#1301). On "
        f"{_citation(evidence)} the area missed the {floors} floor{plural}. It "
        "stays in the solve with its measured support and is flagged "
        "support_below_floor in the build's area-support table; consumers must "
        "treat its estimates as unsupported. Re-measured on every candidate: a "
        "build in which the area meets every floor fails the entry as stale."
        + _class_note(key, notes)
    )


def _area_names(overrides: Mapping[str, str]) -> dict[str, str]:
    from importlib.resources import files

    payload = json.loads(
        files("microcosm.build.uk").joinpath("local_authority_names.json").read_text()
    )
    names = {
        str(code): str(row["name"])
        for code, row in (payload.get("areas") or {}).items()
    }
    names.update(overrides)
    return names


# ---------------------------------------------------------------------------
# Rendering: registers, pins, patch
# ---------------------------------------------------------------------------


def _indent(text: str) -> int:
    lines = text.splitlines()
    if len(lines) < 2:
        raise InputError("a register must be indented JSON.")
    width = len(lines[1]) - len(lines[1].lstrip(" "))
    if width < 1:
        raise InputError("a register must be indented JSON.")
    return width


def _dump(payload: Mapping[str, Any], indent: int) -> str:
    return json.dumps(payload, indent=indent, ensure_ascii=False) + "\n"


def render_register(text: str, block: str, entries: Mapping[str, Any]) -> str:
    """The register with ``block`` replaced, byte-stable everywhere else.

    The file must already be what ``json.dumps`` writes at its own indent, so
    the diff holds only the entries that change.
    """

    indent = _indent(text)
    payload = json.loads(text)
    if _dump(payload, indent) != text:
        raise InputError(
            "a register is not in its canonical form (json.dumps, indent "
            f"{indent}, ensure_ascii=False, trailing newline); refusing to "
            "rewrite it."
        )
    payload[block] = dict(entries)
    return _dump(payload, indent)


def _is_loader_call(node: ast.AST, loader: str) -> bool:
    """A call of ``loader`` with no arguments but ``None`` placeholders."""

    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == loader
        and all(
            isinstance(argument, ast.Constant) and argument.value is None
            for argument in node.args
        )
        and not node.keywords
    )


def _pin_subject(
    test: ast.AST,
    names: set[str],
    loader: str,
    block: str | None,
    source: str,
) -> str | None:
    """The pinned expression's source, when ``test`` pins the register's keys.

    With ``block``, the loader returns the whole register and the pin
    subscripts that block (``set(loaded["exclusions"]) == {...}``).
    """

    if not (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and (
            (isinstance(test.comparators[0], ast.Dict) and not test.comparators[0].keys)
            or isinstance(test.comparators[0], ast.Set)
        )
    ):
        return None
    left = test.left
    if (
        isinstance(left, ast.Call)
        and isinstance(left.func, ast.Name)
        and left.func.id == "set"
        and len(left.args) == 1
    ):
        left = left.args[0]
    base = left
    if block is not None:
        if not (
            isinstance(left, ast.Subscript)
            and isinstance(left.slice, ast.Constant)
            and left.slice.value == block
        ):
            return None
        base = left.value
    if (isinstance(base, ast.Name) and base.id in names) or _is_loader_call(
        base, loader
    ):
        return ast.get_source_segment(source, left)
    return None


def _is_scope_pin(stmt: ast.AST, subject: str) -> bool:
    if not (
        isinstance(stmt, ast.Assert)
        and isinstance(stmt.test, ast.Compare)
        and isinstance(stmt.test.left, ast.SetComp)
        and isinstance(stmt.test.left.elt, ast.Attribute)
        and stmt.test.left.elt.attr == "release_lines"
    ):
        return False
    expected = ast.parse(f"({subject}).values()", mode="eval").body
    return ast.dump(stmt.test.left.generators[0].iter) == ast.dump(expected)


def find_pins(
    source: str, loader: str, block: str | None = None
) -> list[tuple[str, ast.Assert, ast.AST | None, str]]:
    """Test functions pinning the register ``loader`` returns, last pin each."""

    sites = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        names = {
            target.id
            for statement in ast.walk(node)
            if isinstance(statement, ast.Assign)
            and _is_loader_call(statement.value, loader)
            for target in statement.targets
            if isinstance(target, ast.Name)
        }
        found = None
        for index, statement in enumerate(node.body):
            if not isinstance(statement, ast.Assert):
                continue
            subject = _pin_subject(statement.test, names, loader, block, source)
            if subject is None:
                continue
            following = node.body[index + 1] if index + 1 < len(node.body) else None
            scope = following if _is_scope_pin(following, subject) else None
            found = (node.name, statement, scope, subject)
        if found is not None:
            sites.append(found)
    return sites


def rewrite_pin(
    source: str,
    loader: str,
    keys: Iterable[str],
    *,
    local_only: bool,
    block: str | None = None,
) -> tuple[str, str] | None:
    """Rewrite the one pin on ``loader``'s register (or its ``block``) to ``keys``.

    Returns the new source and the test's name, or ``None`` when the source
    holds no single pin to rewrite.
    """

    sites = find_pins(source, loader, block)
    if len(sites) != 1:
        return None
    name, statement, scope, subject = sites[0]
    indent = " " * statement.col_offset
    keys = sorted(keys)
    lines_out = ""
    if subject.endswith(")"):
        lines_out += f"{indent}register = {subject}\n"
        subject = "register"
    if not keys:
        lines_out += f"{indent}assert {subject} == {{}}\n"
    else:
        lines_out += f"{indent}assert set({subject}) == {{\n"
        lines_out += "".join(
            f"{indent}    {json.dumps(key, ensure_ascii=False)},\n" for key in keys
        )
        lines_out += f"{indent}}}\n"
        if local_only:
            head = (
                f"{indent}assert {{record.release_lines for record in "
                f"{subject}.values()}} == {{"
            )
            line = f'("{LOCAL_LINE}",)'
            # As ruff format writes it: one line when it fits, else the set
            # literal's element on its own line.
            if len(head) + len(line) + 1 <= LINE_LENGTH:
                lines_out += f"{head}{line}}}\n"
            else:
                lines_out += f"{head}\n{indent}    {line}\n{indent}}}\n"
    end = (scope or statement).end_lineno
    lines = source.splitlines(keepends=True)
    rewritten = (
        "".join(lines[: statement.lineno - 1]) + lines_out + "".join(lines[end:])
    )
    ast.parse(rewritten)
    return rewritten, name


def unified_patch(changes: Sequence[tuple[str, str | None, str]]) -> str:
    """A ``git apply`` patch; paths are repository-relative, old ``None`` is new."""

    chunks = []
    for path, old, new in changes:
        if old == new:
            continue
        header = [f"diff --git a/{path} b/{path}\n"]
        if old is None:
            header.append("new file mode 100644\n")
        chunks.extend(header)
        chunks.extend(
            difflib.unified_diff(
                [] if old is None else old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile="/dev/null" if old is None else f"a/{path}",
                tofile=f"b/{path}",
            )
        )
    return "".join(chunks)


def tests_naming(repo_root: Path, needles: Iterable[str]) -> list[str]:
    """Test and test-support files whose text names any of ``needles``."""

    needles = sorted(set(needles))
    if not needles:
        return []
    paths = [
        *(
            path
            for path in (repo_root / "packages").rglob("*.py")
            if "tests" in path.relative_to(repo_root).parts
        ),
        *(repo_root / "test_support").rglob("*.py"),
    ]
    hits = []
    for path in sorted(paths):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(needle in text for needle in needles):
            hits.append(path.relative_to(repo_root).as_posix())
    return hits


def _review_registers() -> list[object]:
    """The registers besides target fit that the national graph test reviews."""

    from microcosm.build.uk_runtime.measure_simulation import (
        load_uk_calibration_measure_exclusions,
    )
    from microcosm.build.uk_runtime.support_floor import (
        load_uk_support_floor_declaration,
    )
    from microcosm.build.uk_runtime.terminal_gates import (
        uk_default_degenerate_reviewed_exclusions,
    )
    from microcosm.build.uk_runtime.weighted_integrity import (
        uk_default_input_mass_reviewed_exclusions,
    )

    return [
        dict(uk_default_degenerate_reviewed_exclusions()),
        {
            name: dict(block)
            for name, block in uk_default_input_mass_reviewed_exclusions().items()
        },
        list(load_uk_calibration_measure_exclusions()),
        [load_uk_support_floor_declaration() or {}],
    ]


def national_review_invariant(national_block: Mapping[str, Any]) -> str | None:
    """``test_uk_national_graph._review_date``'s invariant on the drafted register.

    That test reviews the national graph at the latest approval across the
    target-fit register (every entry, whatever its line) and
    :func:`_review_registers`, and requires it before the earliest expiry.
    """

    def records(value: object):
        if hasattr(value, "approved_on"):
            yield value
        elif isinstance(value, Mapping) and "approved_on" in value:
            yield value
        elif isinstance(value, Mapping):
            for item in value.values():
                yield from records(item)
        elif isinstance(value, list | tuple):
            for item in value:
                yield from records(item)

    def field_of(record: object, name: str) -> str:
        return str(
            record[name] if isinstance(record, Mapping) else getattr(record, name)
        )

    entries = [
        record
        for register in (national_block, *_review_registers())
        for record in records(register)
    ]
    if not entries:
        return None
    latest = max(field_of(record, "approved_on") for record in entries)
    earliest = min(field_of(record, "expires_on") for record in entries)
    if latest < earliest:
        return None
    return (
        f"the drafted national entries break the national graph test's review "
        f"invariant: the latest approval {latest} is not before the earliest "
        f"expiry {earliest}"
    )


def drafted_disclosure_failures(drafts: Mapping[str, RegisterDraft]) -> list[str]:
    """The dense contract's disclosure walk over the drafted entries.

    Build 2's signed local report, which a release ships, carries every entry
    its gates stand on, so a drafted reason (a ``--class-note`` included)
    must pass the walk the release directory gets (microcosm#1301).
    """

    from microcosm.data.contract import _uk_dense_disclosure_failures

    return _uk_dense_disclosure_failures(
        "the drafted entries",
        {
            gate_id: {
                "reviewed_exclusions": {
                    key: {"record": entry} for key, entry in draft.entries.items()
                }
            }
            for gate_id, draft in drafts.items()
        },
    )


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _refuse_committable(out: Path, repo_root: Path) -> None:
    """Refuse an ``--out`` that git could commit (microcosm#1301).

    The private review file must never be committable: not in this
    repository, nor in any other git work tree, such as another checkout or a
    sign-off worktree. ``git rev-parse --show-toplevel`` runs from ``--out``
    or its nearest existing ancestor, with git's own variables removed from
    the environment so that only the filesystem decides.
    """

    out = out.resolve()
    if out.is_relative_to(repo_root.resolve()):
        raise InputError(
            f"--out {out} lies inside the repository; the private review file "
            "must never be committable."
        )
    probe = out
    while not probe.is_dir():
        probe = probe.parent
    try:
        found = subprocess.run(
            ["git", "-C", str(probe), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
            env={
                name: value
                for name, value in os.environ.items()
                if not name.startswith("GIT_")
            },
        )
    except OSError as error:
        raise InputError(
            "git is needed to check that --out lies outside every git work tree."
        ) from error
    if found.returncode == 0:
        raise InputError(
            f"--out {out} lies inside the git work tree {found.stdout.strip()}; "
            "the private review file must never be committable, so write the "
            "draft outside every git work tree."
        )


def _write_private(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.chmod(path, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        handle.write(
            "# PRIVATE: licensed-data review values (UKDS). Do not commit, post "
            "or share.\n"
        )
        pd.DataFrame(
            rows,
            dtype=object,
            columns=[
                "gate",
                "key",
                "action",
                "relative_error",
                "stale_risk",
                "near_bound",
                "floors_missed",
                "rows",
                "ess",
                "sources",
            ],
        ).to_csv(handle, index=False)


def _private_rows(
    evidence: BuildEvidence, drafts: Mapping[str, RegisterDraft]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate_id, draft in drafts.items():
        actions = {
            **dict.fromkeys(draft.keep, "keep"),
            **dict.fromkeys(draft.add, "add"),
            **dict.fromkeys(draft.renew, "renew"),
            **dict.fromkeys(draft.remove, "remove"),
            **dict.fromkeys(draft.rescope, "rescope"),
            **dict.fromkeys(draft.review, "review"),
            **dict.fromkeys(draft.conflicts, "conflict"),
            **dict.fromkeys(draft.held, "held"),
        }
        if draft.spec.kind == "area":
            support = evidence.area_support.assign(
                key=evidence.area_support["geography_level"]
                + "/"
                + evidence.area_support["area_code"]
            ).set_index("key")
            floors = evidence.area_floors
            for key, action in sorted(actions.items()):
                row = support.loc[key] if key in support.index else None
                rows.append(
                    {
                        "gate": gate_id,
                        "key": key,
                        "action": action,
                        "floors_missed": None
                        if row is None
                        else " ".join(_missed_floors(row, floors)),
                        "rows": None if row is None else int(row.nonzero_households),
                        "ess": None
                        if row is None
                        else float(row.effective_sample_size),
                        "sources": None
                        if row is None
                        else int(row.nonzero_source_households),
                    }
                )
            continue
        errors = evidence.errors_for(draft.spec)
        for key in sorted({*actions, *draft.near_bound}):
            rows.append(
                {
                    "gate": gate_id,
                    "key": key,
                    "action": actions.get(key, "none"),
                    "relative_error": errors.get(key),
                    "stale_risk": key in draft.stale_risk,
                    "near_bound": key in draft.near_bound,
                }
            )
    return rows


_TABLE_LABELS = {
    "uk_target_fit": "National target-fit rows (`uk_target_fit`)",
    "uk_local_target_fit": "Local target-fit cells (`uk_local_target_fit`)",
    "uk_local_area_support": "Areas below the support floor (`uk_local_area_support`)",
}


def _named(labels: Sequence[str], *, quote: str = "`", shown: int = 3) -> str:
    """Up to ``shown`` labels and a count of the rest, as a sentence list."""

    parts = [f"{quote}{label}{quote}" for label in labels[:shown]]
    if len(labels) > shown:
        parts.append(f"{len(labels) - shown} more")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _deadline_lines(receipt: Mapping[str, Any]) -> list[str]:
    deadline = receipt.get("assembly_deadline")
    if deadline is None:
        return []
    set_by = deadline["set_by"]
    return [
        f"**Assembly deadline: {deadline['date']}**, when {_named(set_by)} "
        f"{'expires' if len(set_by) == 1 else 'expire'}: the earliest expiry "
        "among the measure exclusions build 1's solve ran under, the local "
        "line's degenerate-register entries and the gate-only entries the "
        "local line stands on. Build 2, its `--verify` and the release "
        "assembly must all run by then. Assembly and upload refuse a measure "
        "exclusion past its expiry, and renewing one, or a degenerate-register "
        "entry, changes the solve.",
        "",
    ]


def pr_body(receipt: Mapping[str, Any]) -> str:
    """The sign-off PR's body: counts, gate ids and policy dates only.

    Of the stale-risk and near-bound counts, only the national register's are
    printed (:meth:`RegisterDraft.summary`).
    """

    build = receipt["build_1"]
    window = receipt["window"]
    lines = [
        "## Gate-only sign-off from dense build 1 (microcosm#1301)",
        "",
        f"Build-1 attempt `{build['attempt_id']}`, full gate report sha256 "
        f"`{build['sha256']['full_gate_report']}`, gates evaluated on "
        f"{build['exclusions_evaluated_on']}.",
        "",
    ]
    if not receipt["signable"]:
        lines += [
            "**Not signable: build 1 needs a third build, or a hand fix and a "
            "new draft where a reason below says so.**",
            "",
            *(
                f"- `{gate}`: {reason}"
                for gate, reason in sorted(receipt["not_signable"].items())
            ),
            "",
        ]
        if receipt["dev_build"]:
            lines += ["(Dry run on a development build: nothing is signable.)", ""]
        return "\n".join(lines)
    if receipt["patch"] is None:
        lines += [
            "Nothing to sign: build 1 passed every gate this tool signs, and no "
            "register entry would fail build 2.",
            "",
            *_deadline_lines(receipt),
        ]
        return "\n".join(lines)
    earliest, latest = receipt["gate_clock_range"]
    lines += [
        "Register data and test pins only. Every signed row stays bound in the "
        "solve: each entry is scoped to the local line and suppresses only its "
        "release fence. A build in which a signed row passes fails its entry "
        "as stale.",
        "",
        "| Register | Added | Renewed | Kept | Retired | Rescoped | For review |",
        "|---|---|---|---|---|---|---|",
    ]
    for gate_id, summary in receipt["registers"].items():
        lines.append(
            f"| {_TABLE_LABELS[gate_id]} | {len(summary['add'])} | "
            f"{len(summary['renew'])} | {len(summary['keep'])} | "
            f"{len(summary['remove'])} | {len(summary['rescope'])} | "
            f"{len(summary['review'])} |"
        )
    if any(summary["rescope"] for summary in receipt["registers"].values()):
        lines += [
            "",
            "Rescoped entries are national target-fit entries that applied on "
            "every line and that build 2 would fail on the local line, stale or "
            'out of their window at the gate clock. They now carry `"release_lines": '
            '["national"]` and are otherwise unchanged, so the national line '
            "still stands on them as before.",
        ]
    national = receipt["registers"].get("uk_target_fit")
    if national is not None and national["review"]:
        lines += [
            "",
            f"National target-fit entries for review: {_named(national['review'])}. "
            "The receipt's notes give each one's reason; an expired one must be "
            "renewed or retired on the national line.",
        ]
    if national is not None:
        lines += [
            "",
            f"National target-fit rows: {national['stale_risk_count']} "
            f"stale-risk, {national['near_bound_count']} near-bound. Stale-risk "
            f"counts signed rows within {receipt['stale_risk_margin']:g} above "
            "the bound, which build 2 fails as stale if it does not reproduce "
            "build 1 exactly; near-bound counts unsigned rows within the same "
            "margin below it. The local registers' counts are not printed here.",
        ]
    lines += [
        "",
        f"Window: approved {window['approved_on']} by {window['approved_by']}, "
        f"adjudication {window['adjudication']}; fit entries expire "
        f"{window['expires_on']}, "
        f"area entries {window['area_support_expires_on']}. Build 2 needs a gate "
        f"clock (`--gate-review-date`) on or after {earliest}"
        + (f" and on or before {latest}." if latest else "."),
        "",
        *_deadline_lines(receipt),
        "Predicted build-2 verdicts on build 1's rows: "
        + ", ".join(
            f"`{gate}` {'passes' if verdict['passes'] else 'fails'}"
            for gate, verdict in receipt["predicted_build_2"].items()
        )
        + ".",
    ]
    if receipt["hand_sign"]:
        lines += [
            "",
            "Signed by hand in this PR: "
            + ", ".join(f"`{gate}`" for gate in receipt["hand_sign"])
            + ".",
        ]
    if receipt["pins"]["manual"]:
        lines += [
            "",
            "Pins to edit by hand: "
            + ", ".join(f"`{gate}`" for gate in receipt["pins"]["manual"])
            + ".",
        ]
    tests = sorted({*receipt["pins"]["files"], *receipt["tests_naming_keys"]})
    if tests:
        lines += [
            "",
            "### Tests to run",
            "",
            "```",
            "uv run --no-sync pytest " + " ".join(tests),
            "```",
        ]
    lines += [
        "",
        "Disclosure: this body carries gate ids and their predicted verdicts, "
        "build 1's attempt id and report digest, register entry counts, the "
        "national register's stale-risk and near-bound counts, the reviewer, "
        "the adjudication, policy dates, the names of the national entries for "
        "review and of the entries that set the assembly deadline, and test "
        "paths. Per-row errors, the local registers' stale-risk and near-bound "
        "counts and per-area support figures stay in the private review file "
        "outside every git work tree.",
        "",
    ]
    return "\n".join(lines)


def _changelog(receipt: Mapping[str, Any]) -> str:
    registers = receipt["registers"]

    def count(gate_id: str, *actions: str) -> int:
        summary = registers.get(gate_id)
        return 0 if summary is None else sum(len(summary[a]) for a in actions)

    def counted(number: int, noun: str, plural: str | None = None) -> str:
        return f"{number} {noun if number == 1 else plural or noun + 's'}"

    window = receipt["window"]
    retired = sum(len(summary["remove"]) for summary in registers.values())
    rescoped = count("uk_target_fit", "rescope")
    return (
        "Sign dense build 1's gate-only failures for the local line "
        f"(microcosm#1301, attempt {receipt['build_1']['attempt_id']}): "
        f"{counted(count('uk_target_fit', 'add', 'renew'), 'national target-fit row')}, "
        f"{counted(count('uk_local_target_fit', 'add', 'renew'), 'local target-fit cell')} "
        f"and {counted(count('uk_local_area_support', 'add', 'renew'), 'area')} "
        f"below the support floor, approved {window['approved_on']}; "
        f"{counted(retired, 'entry', 'entries')} that build 2 would fail "
        f"{'is' if retired == 1 else 'are'} retired"
        + (
            f", and {counted(rescoped, 'national target-fit entry', 'national target-fit entries')} "
            f"that applied on every line {'is' if rescoped == 1 else 'are'} "
            "scoped to the national line, which still stands on "
            f"{'it' if rescoped == 1 else 'them'}"
            if rescoped
            else ""
        )
        + ". Every signed row stays "
        "bound in the solve, each entry suppresses only the local line's release "
        "fence, and a build in which a signed row passes fails its entry as "
        "stale.\n"
    )


# ---------------------------------------------------------------------------
# Draft
# ---------------------------------------------------------------------------


def _real_day(text: str) -> bool:
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def figure_free(text: str, label: str) -> str:
    """``text`` unchanged, or refused when it may state a figure (microcosm#1301).

    Reviewer free text lands in the committed registers and the PR body,
    where nothing else checks it: the contract's disclosure walk catches only
    an area's support worded with its measured value. So the text may carry
    a digit only inside an issue reference (``#1301``, ``microcosm#1301``,
    ``PolicyEngine/microcosm#1301``) or an ISO date of a real day, and never a
    ``%``. A figure written in words passes; the manual scan covers that.
    """

    if "%" in text:
        raise InputError(f"{label} must carry no figures ('%'): {text!r}.")
    rest = _ISO_DATE_RE.sub(
        lambda match: "" if _real_day(match.group()) else match.group(), text
    )
    rest = _ISSUE_REFERENCE_RE.sub("", rest)
    if any(character.isnumeric() for character in rest):
        raise InputError(
            f"{label} may carry digits only in issue references (#1301, "
            "microcosm#1301, PolicyEngine/microcosm#1301) and ISO dates "
            f"(YYYY-MM-DD), so that it states no figure: {text!r}."
        )
    return text


def github_login(value: str | None, label: str) -> str:
    """``value`` unchanged when it is a GitHub login, else refused (microcosm#1301).

    The reviewer lands in every drafted entry and the PR body, so it is a
    login and nothing else: no spaces, no other text, and so no figure.
    """

    if value is None or _GITHUB_LOGIN_RE.fullmatch(value) is None:
        raise InputError(
            f"{label} must be a GitHub login (letters, digits and hyphens, at "
            f"most 39 characters, starting with a letter or digit), got {value!r}."
        )
    return value


def _pairs(values: Sequence[str], label: str) -> list[tuple[str, str]]:
    """``KEY=TEXT`` arguments; only the text, which is published, is checked."""

    pairs = []
    for value in values:
        key, separator, text = value.partition("=")
        if not separator or not key.strip() or not text.strip():
            raise InputError(f"{label} must be KEY=TEXT, got {value!r}.")
        pairs.append((key.strip(), figure_free(text.strip(), label)))
    return pairs


def _window(
    args: argparse.Namespace, evidence: BuildEvidence, *, adjudication: str
) -> Window:
    """The approval window; ``adjudication`` is the checked ``--adjudication``."""

    approved_on = _iso(args.approved_on, "--approved-on")
    expires_on = (
        _iso(args.expires_on, "--expires-on")
        if args.expires_on
        else date.fromordinal(approved_on.toordinal() + FIT_WINDOW_DAYS)
    )
    area_expires_on = (
        _iso(args.area_support_expires_on, "--area-support-expires-on")
        if args.area_support_expires_on
        else _add_months(approved_on, AREA_WINDOW_MONTHS)
    )
    gate_clock = (
        _iso(args.gate_clock, "--gate-clock") if args.gate_clock else approved_on
    )
    approved_by = (args.approved_by or "").strip()
    if not approved_by:
        raise InputError("--approved-by is required.")
    adjudication = adjudication or (
        f"microcosm#1301 (build-1 attempt {evidence.attempt_id}; sign-off "
        f"{approved_on.isoformat()})"
    )
    if gate_clock < approved_on:
        raise InputError(
            "--gate-clock precedes --approved-on: every drafted entry would be "
            "premature."
        )
    if gate_clock < evidence.clock:
        raise InputError(
            "--gate-clock precedes build 1's clock; build 2's gate clock may not "
            "be earlier than its solve clock."
        )
    for label, expiry in (
        ("--expires-on", expires_on),
        ("--area-support-expires-on", area_expires_on),
    ):
        if expiry <= gate_clock:
            raise InputError(f"{label} must be after the gate clock.")
        if (expiry - approved_on).days > args.max_window_days:
            raise InputError(
                f"{label} gives a window longer than {args.max_window_days} days."
            )
    return Window(
        approved_by=approved_by,
        adjudication=adjudication,
        approved_on=approved_on,
        expires_on=expires_on,
        area_expires_on=area_expires_on,
        gate_clock=gate_clock,
    )


def run_draft(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    out = (args.out or Path(args.run_dir) / "signoff-draft").resolve()
    _refuse_committable(out, repo_root)
    # The reviewer's login and free text are checked before the build is read.
    github_login(args.approved_by, "--approved-by")
    notes = _pairs(args.class_note, "--class-note")
    name_overrides = dict(_pairs(args.area_name, "--area-name"))
    adjudication = figure_free((args.adjudication or "").strip(), "--adjudication")
    evidence = load_build(args.run_dir, allow_dev_build=args.allow_dev_build)
    window = _window(args, evidence, adjudication=adjudication)
    # The solve-scoped expiries no sign-off can move: renewing either kind of
    # entry changes the solve, and build 2 must reuse build 1's.
    solve_scoped = {
        "measure_exclusion": evidence.measure_exclusions,
        "degenerate": degenerate_expiries(repo_root),
    }
    solve_deadline = assembly_deadline(solve_scoped)
    if solve_deadline is not None and window.gate_clock > date.fromisoformat(
        solve_deadline["date"]
    ):
        set_by = solve_deadline["set_by"]
        raise InputError(
            f"--gate-clock {window.gate_clock.isoformat()} is after "
            f"{solve_deadline['date']}, when {_named(set_by, quote='')} "
            f"{'expires' if len(set_by) == 1 else 'expire'}: build 2's gates "
            "fail an expired degenerate-register entry, and assembly refuses an "
            "expired measure exclusion. Choose a gate clock on or before that "
            "date; renewing either changes the solve and needs a third build."
        )
    names = _area_names(name_overrides)
    wiring = register_wiring()
    deferred: frozenset[str] = frozenset()
    if wiring["uk_local_area_support"] is None:
        found = generator_deferred_areas(repo_root)
        if found is None:
            wiring["uk_local_area_support"] = (
                f"this checkout's {LOCAL_REFERENCE_GENERATOR} declares no "
                f"{DEFERRED_AREAS_CONSTANT}: it takes its support-floor deferrals "
                "from the area-support register, so signing an area there "
                "changes the solve"
            )
        else:
            deferred = found

    not_signable: dict[str, str] = {}
    hand_sign: list[str] = []
    for gate_id in failing_release_gates(evidence):
        status = evidence.status(gate_id)
        if gate_id in SIGNABLE_GATES:
            if wiring[gate_id] is not None:
                not_signable[gate_id] = wiring[gate_id]
            elif status != "failed":
                not_signable[gate_id] = f"{status}: only a measured failure is signable"
        elif gate_id in HAND_SIGNED_GATES:
            hand_sign.append(gate_id)
        else:
            not_signable[gate_id] = (
                f"{status}: no gate-only register; a failure here needs a solve change"
            )

    drafts: dict[str, RegisterDraft] = {}
    texts: dict[str, str] = {}
    bound_despite: Mapping[str, Any] = {}
    for spec in REGISTERS:
        if wiring[spec.gate_id] is not None:
            continue
        path = repo_root / spec.path
        if not path.is_file():
            raise InputError(f"{spec.path} is wired into gates.json but missing.")
        texts[spec.gate_id] = path.read_text(encoding="utf-8")
        register = json.loads(texts[spec.gate_id])
        if spec.kind == "area":
            draft, blockers = draft_area_support(
                spec,
                evidence.area_support,
                register,
                floors=evidence.area_floors,
                window=window,
                reason=lambda key, missed: area_reason(
                    key, missed, evidence, names=names, notes=notes
                ),
                deferred=deferred,
            )
            bound_despite = register.get("bound_despite_support_floor") or {}
            problems = []
            if blockers:
                problems.append(
                    "bound_despite_support_floor entries are outside their window "
                    f"at the gate clock ({', '.join(blockers)}); the tool never "
                    "edits that block, whose families are the local reference "
                    "generator's SUPPORT_FLOOR_BOUND_FAMILIES, so renew them by "
                    "hand and draft again"
                )
            problems.extend(
                f"{key} is held: {draft.notes[key]}" for key in sorted(draft.held)
            )
            if problems:
                not_signable[spec.gate_id] = "; ".join(problems)
        else:
            draft = draft_target_fit(
                spec,
                evidence.errors_for(spec),
                register[spec.block],
                bound=evidence.fit_bound(spec.gate_id),
                window=window,
                reason=lambda key, spec=spec: fit_reason(spec, key, evidence, notes),
                margin=args.stale_risk_margin,
            )
        if draft.conflicts:
            not_signable.setdefault(
                spec.gate_id,
                "failing keys hold entries the tool may not replace, and a key "
                "holds one entry; sign them by hand (the receipt's conflicts "
                "and notes name them)",
            )
        drafts[spec.gate_id] = draft

    leaks = drafted_disclosure_failures(drafts)
    if leaks:
        raise InputError(" ".join(leaks))
    predicted = predict(
        evidence, drafts, clock=window.gate_clock, bound_despite=bound_despite
    )
    for gate_id, verdict in predicted.items():
        if not verdict["passes"] and gate_id not in not_signable:
            not_signable[gate_id] = "the drafted register does not make the gate pass"
    if "uk_target_fit" in drafts and drafts["uk_target_fit"].changed:
        national = drafts["uk_target_fit"]
        after = national.after()
        problem = national_review_invariant(after)
        # An expired entry kept for the national line breaks the national
        # graph test's review invariant beside newer approvals: only the
        # national line can renew or retire it, and then the draft is redone.
        expired = sorted(set(national.rescope) & set(national.review))
        if problem is not None and not (
            expired
            and national_review_invariant(
                {key: value for key, value in after.items() if key not in expired}
            )
            is None
        ):
            raise InputError(problem)
        if problem is not None:
            reason = (
                f"{problem}, because {_named(expired, quote='')} expired and "
                f"{'is' if len(expired) == 1 else 'are'} kept for the national "
                f"line: renew or retire {'it' if len(expired) == 1 else 'them'} "
                "there by hand, then draft again"
            )
            earlier = not_signable.get("uk_target_fit")
            not_signable["uk_target_fit"] = (
                reason if earlier is None else f"{earlier}; {reason}"
            )
    if evidence.dev_build:
        not_signable.setdefault(
            "release_candidate", "a development build is never signable"
        )
    signable = not not_signable

    changes: list[tuple[str, str | None, str]] = []
    pin_files: list[str] = []
    manual_pins: list[str] = []
    if signable:
        sources = {
            relative: (repo_root / relative).read_text(encoding="utf-8")
            for relative in PIN_SEARCH
            if (repo_root / relative).is_file()
        }
        originals = dict(sources)
        for gate_id, draft in drafts.items():
            spec = draft.spec
            if not draft.changed:
                continue
            new_block = draft.after()
            changes.append(
                (
                    spec.path.as_posix(),
                    texts[gate_id],
                    render_register(texts[gate_id], spec.block, new_block),
                )
            )
            if spec.loader is None:
                continue
            records = coerce_reviewed_exclusions(new_block, label=spec.resource)
            local_only = all(
                record.release_lines == (LOCAL_LINE,) for record in records.values()
            )
            rewrote = False
            for relative, text in sources.items():
                rewritten = rewrite_pin(
                    text,
                    spec.loader,
                    new_block,
                    local_only=local_only,
                    block=spec.pin_block,
                )
                if rewritten is not None:
                    sources[relative] = rewritten[0]
                    rewrote = True
            if not rewrote:
                manual_pins.append(gate_id)
        for relative, text in sources.items():
            if text != originals[relative]:
                changes.append((relative.as_posix(), originals[relative], text))
                pin_files.append(relative.as_posix())

    touched = {
        draft.spec.kind: [*draft.add, *draft.renew, *draft.remove, *draft.rescope]
        for draft in drafts.values()
    }
    needles = {
        key.partition("/")[2] if kind == "area" else key
        for kind, keys in touched.items()
        for key in keys
    }
    # Every gate-only entry build 2's calibrated gates evaluate on the local
    # line, after the sign-off.
    in_line_after = {
        draft.spec.gate_id: _scoped(draft.after(), draft.spec.resource)
        for draft in drafts.values()
    }
    if "uk_local_area_support" in drafts:
        in_line_after["bound_despite_support_floor"] = _scoped(
            bound_despite, "bound_despite_support_floor"
        )
    deadline = assembly_deadline(
        {
            **solve_scoped,
            **{source: _expiries(records) for source, records in in_line_after.items()},
        }
    )
    receipt: dict[str, Any] = {
        "schema": "microcosm.uk.dense-signoff-draft.v1",
        "build_1": {"run_dir": str(evidence.run_dir), **evidence.identity()},
        "window": {
            "approved_by": window.approved_by,
            "adjudication": window.adjudication,
            "approved_on": window.approved_on.isoformat(),
            "expires_on": window.expires_on.isoformat(),
            "area_support_expires_on": window.area_expires_on.isoformat(),
            "gate_clock": window.gate_clock.isoformat(),
        },
        # Build 2's gate clock must put every in-line entry in force, may not
        # precede its solve clock (build 1's) and may not pass the deadline.
        "gate_clock_range": [
            max(
                [evidence.clock.isoformat()]
                + [
                    record.approved_on
                    for records in in_line_after.values()
                    for record in records.values()
                ]
            ),
            None if deadline is None else deadline["date"],
        ],
        "assembly_deadline": deadline,
        "stale_risk_margin": args.stale_risk_margin,
        "signable": signable,
        "dev_build": evidence.dev_build,
        "not_signable": dict(sorted(not_signable.items())),
        "hand_sign": hand_sign,
        "diagnostic_failures": list(
            evidence.enforcement.get("diagnostic_failures", ())
        ),
        "unwired_registers": {
            gate: reason for gate, reason in wiring.items() if reason is not None
        },
        "registers": {gate: draft.summary() for gate, draft in drafts.items()},
        "predicted_build_2": predicted,
        "pins": {"files": pin_files, "manual": manual_pins},
        "tests_naming_keys": tests_naming(repo_root, needles),
    }
    if signable and changes:
        changes.append(
            (
                f"changelog.d/1301-uk-dense-signoff-{window.approved_on.isoformat()}"
                ".changed.md",
                None,
                _changelog(receipt),
            )
        )
    out.mkdir(parents=True, exist_ok=True)
    patch_path = out / "signoff.patch"
    if patch_path.exists():
        patch_path.unlink()
    if signable and changes:
        patch_path.write_text(unified_patch(changes), encoding="utf-8")
    receipt["patch"] = patch_path.name if patch_path.exists() else None
    (out / "signoff_draft.json").write_text(
        json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "PR_BODY.md").write_text(pr_body(receipt), encoding="utf-8")
    _write_private(out / "PRIVATE_review.csv", _private_rows(evidence, drafts))
    print(
        json.dumps(
            {
                "out": str(out),
                "signable": signable,
                "patch": receipt["patch"],
                "not_signable": sorted(not_signable),
                "hand_sign": hand_sign,
                "counts": {
                    gate: {
                        action: len(summary[action])
                        for action in (
                            "add",
                            "renew",
                            "keep",
                            "remove",
                            "rescope",
                            "review",
                        )
                    }
                    for gate, summary in receipt["registers"].items()
                },
                # For the operator only: the local counts stay out of the
                # receipt and the PR body (RegisterDraft.summary).
                "fit_band_counts": {
                    gate: draft.fit_bands()
                    for gate, draft in drafts.items()
                    if draft.spec.kind != "area"
                },
                "gate_clock_range": receipt["gate_clock_range"],
                "assembly_deadline": deadline,
            },
            indent=1,
        )
    )
    return 0 if signable else 1


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def _receipt_matches(entry: Mapping[str, Any], record: UKReviewedExclusion) -> bool:
    return all(
        entry.get(name) == getattr(record, name)
        for name in ("reason", "approved_by", "adjudication", "expires_on")
        if name in entry
    )


def _today() -> date:
    """The day ``--verify`` runs on; the tests pin it."""

    return date.today()


def verify(
    build_2: BuildEvidence,
    *,
    repo_root: Path,
    build_1: BuildEvidence | None = None,
    draft_receipt: Mapping[str, Any] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Whether build 2 stood on exactly the committed sign-off and passed.

    It also fails once the assembly deadline (:func:`assembly_deadline`, on
    build 2's measure exclusions and this checkout's registers) has passed,
    and warns from :data:`DEADLINE_WARNING_DAYS` days before it.
    """

    from microcosm.data.contract import (
        _check_uk_dense_gate_report,
        _uk_dense_disclosure_failures,
    )

    today = _today() if today is None else today
    failures: list[str] = []
    warnings: list[str] = []
    enforcement = build_2.enforcement
    blocking = failing_release_gates(build_2)
    if blocking:
        failures.append(f"release-blocking gates did not pass: {blocking}")
    if not enforcement.get("artifact_permitted"):
        failures.append("the build was not permitted an artifact")
    if not enforcement.get("release_blocking_gates_passed"):
        failures.append("release_blocking_gates_passed is false")
    if enforcement.get("diagnostic_failures"):
        warnings.append(
            f"diagnostic gates failed: {list(enforcement['diagnostic_failures'])}"
        )
    wiring = register_wiring()
    stood_on: dict[str, int] = {}
    in_line: dict[str, Mapping[str, UKReviewedExclusion]] = {}
    for spec in REGISTERS:
        if wiring[spec.gate_id] is not None:
            continue
        details = build_2.details(spec.gate_id)
        payload = _read_json(repo_root / spec.path, spec.resource)
        records = _scoped(payload[spec.block], spec.resource)
        in_line[spec.gate_id] = records
        if spec.kind == "area":
            in_line["bound_despite_support_floor"] = _scoped(
                payload.get("bound_despite_support_floor") or {},
                "bound_despite_support_floor",
            )
        used = details.get("reviewed_exclusions") or {}
        for name in (
            "stale_exclusions",
            "expired_exclusions",
            "premature_exclusions",
            "unknown_exclusions",
            "invalid_reviewed_exclusions",
        ):
            if details.get(name):
                failures.append(f"{spec.gate_id} {name}: {sorted(details[name])}")
        if details.get("failing_targets"):
            failures.append(
                f"{spec.gate_id}: {len(details['failing_targets'])} rows still fail"
            )
        dormant = set(details.get("dormant_exclusions") or ())
        unscoped_dormant = {
            key
            for key in dormant
            if key in records and records[key].release_lines is None
        }
        if unscoped_dormant:
            warnings.append(
                f"{spec.gate_id}: unscoped entries no local row earns: "
                f"{sorted(unscoped_dormant)}"
            )
        unused = sorted(set(records) - set(used) - unscoped_dormant)
        if unused:
            failures.append(
                f"{spec.gate_id}: build 2 did not stand on committed entries {unused}"
            )
        foreign = sorted(set(used) - set(records))
        if foreign:
            failures.append(
                f"{spec.gate_id}: build 2 stood on entries this checkout does not "
                f"hold {foreign}; verify from the sign-off's head"
            )
        for key in sorted(set(used) & set(records)):
            entry = used[key]
            entry = entry.get("record", entry) if spec.kind == "area" else entry
            if not isinstance(entry, Mapping) or not _receipt_matches(
                entry, records[key]
            ):
                failures.append(
                    f"{spec.gate_id}: build 2's receipt for {key} differs from the "
                    "committed entry"
                )
        stood_on[spec.gate_id] = len(set(used) & set(records))
    contract_failures: list[str] = []
    _check_uk_dense_gate_report(
        build_2.local_report, failures=contract_failures, attempt_id=build_2.attempt_id
    )
    # The signed local report ships in the release, so it gets the release
    # directory's disclosure walk (microcosm#1301) here, before assembly.
    contract_failures.extend(
        _uk_dense_disclosure_failures(
            "the signed local gate report", build_2.local_report
        )
    )
    failures.extend(f"contract: {line}" for line in contract_failures)

    comparison: dict[str, Any] = {}
    if build_1 is not None:
        if build_1.attempt_id == build_2.attempt_id:
            failures.append("build 2 carries build 1's attempt id")
        if (
            build_1.document["report"]["gates_manifest_sha256"]
            != build_2.document["report"]["gates_manifest_sha256"]
        ):
            failures.append("the builds evaluated different gate manifests")
        if build_1.document.get("selection_receipt") != build_2.document.get(
            "selection_receipt"
        ):
            failures.append("the builds selected different targets")
        one = build_1.document.get("artifacts") or {}
        two = build_2.document.get("artifacts") or {}
        reused = all(
            one.get(name) is not None and one.get(name) == two.get(name)
            for name in SOLVE_ARTIFACTS
        )
        if not reused:
            warnings.append("build 2 did not reuse build 1's solve")
        differing = sum(
            1
            for errors_1, errors_2 in (
                (build_1.national_errors, build_2.national_errors),
                (build_1.local_errors, build_2.local_errors),
            )
            for key in set(errors_1) | set(errors_2)
            if errors_1.get(key) != errors_2.get(key)
        )
        if differing:
            warnings.append(f"{differing} target rows differ from build 1")
        comparison = {
            "build_1_attempt_id": build_1.attempt_id,
            "solve_reused": reused,
            "rows_differing": differing,
        }
    if draft_receipt is not None:
        recorded = draft_receipt.get("build_1") or {}
        if build_1 is not None and (
            recorded.get("attempt_id") != build_1.attempt_id
            or (recorded.get("sha256") or {}).get("full_gate_report")
            != build_1.sha256["full_gate_report"]
        ):
            failures.append("the draft receipt was drafted from another build 1")
        for gate_id, summary in (draft_receipt.get("registers") or {}).items():
            spec = next((s for s in REGISTERS if s.gate_id == gate_id), None)
            if spec is None or wiring[gate_id] is not None:
                continue
            block = _read_json(repo_root / spec.path, spec.resource)[spec.block]
            committed = set(block)
            missing = sorted(set(summary.get("add", ())) - committed) + sorted(
                set(summary.get("renew", ())) - committed
            )
            kept = sorted(set(summary.get("remove", ())) & committed)
            records = coerce_reviewed_exclusions(block, label=spec.resource)
            unscoped = sorted(
                key
                for key in summary.get("rescope", ())
                if key not in records or records[key].applies_on(LOCAL_LINE)
            )
            if missing:
                failures.append(f"{gate_id}: drafted entries not committed {missing}")
            if kept:
                failures.append(f"{gate_id}: retired entries still committed {kept}")
            if unscoped:
                failures.append(
                    f"{gate_id}: entries drafted for the national line only are "
                    f"not committed so {unscoped}"
                )
    deadline = assembly_deadline(
        {
            "measure_exclusion": build_2.measure_exclusions,
            "degenerate": degenerate_expiries(repo_root),
            **{source: _expiries(records) for source, records in in_line.items()},
        }
    )
    if deadline is not None:
        days_left = (date.fromisoformat(deadline["date"]) - today).days
        named = _named(deadline["set_by"], quote="")
        if days_left < 0:
            failures.append(
                f"the assembly deadline {deadline['date']} has passed ({named})"
            )
        elif days_left <= DEADLINE_WARNING_DAYS:
            warnings.append(
                f"the assembly deadline {deadline['date']} is {days_left} day(s) "
                f"away ({named}): assemble and upload by then"
            )
    return {
        "schema": "microcosm.uk.dense-signoff-verify.v1",
        "ok": not failures,
        "build_2": {"run_dir": str(build_2.run_dir), **build_2.identity()},
        "stood_on": stood_on,
        "comparison": comparison,
        "assembly_deadline": deadline,
        "failures": failures,
        "warnings": warnings,
    }


def run_verify(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    out = (args.out or Path(args.verify) / "signoff-verify").resolve()
    _refuse_committable(out, repo_root)
    build_2 = load_build(args.verify, allow_dev_build=args.allow_dev_build)
    build_1 = (
        load_build(args.build_1, allow_dev_build=args.allow_dev_build)
        if args.build_1 is not None
        else None
    )
    receipt = (
        _read_json(args.draft_receipt, "--draft-receipt")
        if args.draft_receipt is not None
        else None
    )
    result = verify(
        build_2, repo_root=repo_root, build_1=build_1, draft_receipt=receipt
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "signoff_verify.json").write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if result["ok"] else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", type=Path, help="build 1's run directory")
    mode.add_argument(
        "--verify", type=Path, metavar="BUILD_2", help="build 2's run directory"
    )
    parser.add_argument("--build-1", type=Path, help="with --verify: build 1")
    parser.add_argument(
        "--draft-receipt", type=Path, help="with --verify: the draft's receipt"
    )
    parser.add_argument("--approved-by", help="the reviewer's GitHub login")
    parser.add_argument("--approved-on", help="YYYY-MM-DD")
    parser.add_argument(
        "--expires-on",
        help=f"fit entries' expiry (default --approved-on + {FIT_WINDOW_DAYS} days)",
    )
    parser.add_argument(
        "--area-support-expires-on",
        help=f"area entries' expiry (default --approved-on + {AREA_WINDOW_MONTHS} months)",
    )
    parser.add_argument(
        "--adjudication",
        help="the entries' adjudication (digits only in issue references and "
        "ISO dates)",
    )
    parser.add_argument(
        "--gate-clock",
        help="build 2's planned --gate-review-date (default --approved-on)",
    )
    parser.add_argument("--max-window-days", type=int, default=MAX_WINDOW_DAYS)
    parser.add_argument("--stale-risk-margin", type=float, default=STALE_RISK_MARGIN)
    parser.add_argument(
        "--class-note",
        action="append",
        default=[],
        metavar="GLOB=TEXT",
        help="a sentence appended to the reasons of keys matching GLOB (digits "
        "only in issue references and ISO dates)",
    )
    parser.add_argument(
        "--area-name",
        action="append",
        default=[],
        metavar="CODE=NAME",
        help="an area's name in its reasons (the --class-note digit check)",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--out", type=Path, help="output directory, outside every git work tree"
    )
    parser.add_argument(
        "--allow-dev-build",
        action="store_true",
        help="dry run on a development build; nothing is signable",
    )
    args = parser.parse_args(argv)
    if args.run_dir is not None:
        if args.build_1 is not None or args.draft_receipt is not None:
            parser.error("--build-1 and --draft-receipt go with --verify")
        if not args.approved_by or not args.approved_on:
            parser.error("--run-dir needs --approved-by and --approved-on")
    if not 0.0 < args.stale_risk_margin < 1.0:
        parser.error("--stale-risk-margin must be in (0, 1)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_verify(args) if args.verify is not None else run_draft(args)
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
