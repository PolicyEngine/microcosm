"""Statically-knowable US release-gate failures, computed without a solve.

Build M release attempts 9/10/11 each burned ~2h of calibration to surface ONE
gate-group failure that was already determined by the *inputs* — the base pool
values, the frozen selection, and the target/coverage registry — and needed no
weight solve to see. This module recovers those signals in minutes so a release
launch is not the first place a knowable defect is caught.

Four checks, each mirroring a live release gate but evaluated pre-solve:

1. **Selection carryover** — does the frozen selection-source manifest map
   cleanly onto the base pool? (Reuses
   :meth:`~microcosm.build.us_runtime.warm_start_selection.SelectionSource.base_selection_mask`
   in ``frozen_support`` mode: an unmapped/ambiguous/non-unique key is the same
   hard failure the release refuses on, found in seconds instead of after a
   base rebuild.)

2. **Zero-support preview** — for every compiled *positive* fiscal target, its
   materialized support under the selection at base weights. A positive target
   whose estimate is ~0 pre-solve has no records to move and stays a structural
   zero after calibration (the release ``zero_support`` gate's exact condition,
   ``initial_estimate ~ final_estimate ~ 0``). Direct-column targets are checked
   by reusing the calibrate constraint compiler
   (:func:`microcosm.calibrate.matrix.build_constraint_matrix`); targets whose
   measure is engine-derived (needs a microsim materialization pass) are marked
   explicitly as *not statically checkable* rather than silently passed.

3. **Export-mass parity risk** — for every column the export-mass parity gate
   checks, the pool mass at *base* weights against the reference band
   (``reference ± relative_tolerance``, honoring the release tool's
   ``US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS`` register and the
   ``minimum_reference_total`` floor). A column already outside its band at base
   weights is an AT-RISK signal: the solve optimizes the target surface, not
   this untargeted column, so a pre-solve out-of-band mass usually stays out.
   The band math is reused verbatim via
   :func:`microcosm.build.gates.input_mass_parity_gate`.

4. **Smoke-probe support audit** — for every reform-coverage probe, for every
   ``binding_inputs`` leaf: pool nonzero support, *selected* nonzero support
   (via the frame's declared entity->household linkage), selected weighted mass,
   and the pool sign-leg decomposition (positive/negative weighted). A leaf with
   pool support but **zero selected support** is the ``keogh_distributions``
   restoration-vs-frozen-selection failure (microcosm#434): the smoke gate's only
   teeth, computed as a set intersection instead of after a solve. A thin
   selection (<5 records) or a signed leaf whose net sign contradicts the
   probe's declared ``expected_sign`` (the ``farm_operations_income`` net-+$10.4B
   Schedule-F-loss-heavy defect, microcosm#432) is AT-RISK.

The pure check functions take frames and already-loaded registry material so they
unit-test on tiny synthetic fixtures with no policyengine-us and no large H5.
:func:`run_preflight` is the runbook entry point that loads the real artifacts
and reuses the release tool's own register and engine input-variable surface.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from microcosm.build.gates import input_mass_parity_gate
from microcosm.build.us_runtime.h5_io import (
    identify_us_multispine_pool_manifest,
    load_authenticated_us_multispine_pool_for_release,
    refuse_denied_pool_h5_digest,
    require_authenticated_us_multispine_pool_h5,
    us_multispine_pool_release_receipt,
)
from microcosm.build.us_runtime.input_mass import us_input_mass_totals
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    assert_puf_capital_gains_tail_survives_selection,
)
from microcosm.build.us_runtime.release_input_coverage import (
    ReformCoverageProbe,
    us_release_reform_coverage_probes,
)
from microcosm.build.us_runtime.warm_start_selection import (
    SelectionSource,
    load_selection_source_from_manifest,
)
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.calibrate.registry import TargetRegistry, TargetSpec
from microcosm.frame import Frame

__all__ = [
    "CheckResult",
    "PreflightReport",
    "check_export_mass_parity_risk",
    "check_selection_carryover",
    "check_smoke_probe_support",
    "check_zero_support_preview",
    "run_preflight",
    "selected_household_ids",
]

#: A check's verdict.
PreflightStatus = Literal["PASS", "FAIL", "AT_RISK", "SKIPPED"]

#: An estimate at or below this absolute value is treated as zero support —
#: the same magnitude the release ``zero_support`` gate uses for a target's
#: initial and final estimate.
_ZERO_ESTIMATE_ATOL = 1e-9

#: A probe leaf with fewer than this many selected nonzero records is AT-RISK:
#: the selection can express the input, but so thinly that a tail-heavy column's
#: mass is one record away from a structural zero (microcosm#434's 4-of-13 case).
_MIN_SELECTED_RECORDS = 5

#: A column is "materially signed" — a genuine positive/negative column rather
#: than a one-signed amount that merely has a stray outlier — when its minority
#: leg carries at least this share of its majority leg's weighted mass. Only
#: materially-signed columns are eligible for the sign-contradiction check, so a
#: one-signed deduction leaf (e.g. a refund amount) is never a false positive.
_CANCELLATION_LEG_RATIO = 0.15

#: The export-mass parity gate defaults, matching the release tool CLI
#: (``--input-mass-relative-tolerance`` / ``--input-mass-minimum-reference-total``).
_DEFAULT_RELATIVE_TOLERANCE = 0.5
_DEFAULT_MINIMUM_REFERENCE_TOTAL = 1e9

_STATUS_RANK = {"PASS": 0, "SKIPPED": 0, "AT_RISK": 1, "FAIL": 2}


@dataclass(frozen=True)
class CheckResult:
    """One preflight check's verdict and the numbers behind it.

    Attributes:
        name: Stable check id.
        status: ``PASS`` / ``FAIL`` / ``AT_RISK`` / ``SKIPPED``.
        summary: One-line human summary with the headline number.
        failures: Hard-failure lines (drive exit 1).
        at_risks: Advisory lines (drive exit 2 when no failure).
        rows: Per-item measured numbers, JSON-ready, for the report table.
        details: Extra machine-readable context.
    """

    name: str
    status: PreflightStatus
    summary: str
    failures: tuple[str, ...] = ()
    at_risks: tuple[str, ...] = ()
    rows: tuple[Mapping[str, Any], ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "failures": list(self.failures),
            "at_risks": list(self.at_risks),
            "rows": [dict(row) for row in self.rows],
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class PreflightReport:
    """Every check's verdict plus the run's inputs, and the process exit code."""

    checks: tuple[CheckResult, ...]
    inputs: Mapping[str, Any] = field(default_factory=dict)
    base_pool: Mapping[str, Any] | None = None

    @property
    def status(self) -> PreflightStatus:
        """The worst check status: FAIL > AT_RISK > (PASS/SKIPPED)."""
        worst = max((_STATUS_RANK[c.status] for c in self.checks), default=0)
        for status, rank in (("FAIL", 2), ("AT_RISK", 1)):
            if worst == rank:
                return status  # type: ignore[return-value]
        return "PASS"

    @property
    def exit_code(self) -> int:
        """1 on any FAIL, 2 on AT-RISK only, 0 clean."""
        if any(c.status == "FAIL" for c in self.checks):
            return 1
        if any(c.status == "AT_RISK" for c in self.checks):
            return 2
        return 0

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "exit_code": self.exit_code,
            "inputs": dict(self.inputs),
            "checks": [check.to_dict() for check in self.checks],
        }
        if self.base_pool is not None:
            payload["base_pool"] = dict(self.base_pool)
        return payload

    def human_table(self) -> str:
        """A compact, terminal-friendly rendering of every check."""
        lines: list[str] = []
        if self.base_pool is not None:
            agreement = self.base_pool.get("agreement_gate_reference")
            if (
                isinstance(agreement, Mapping)
                and agreement.get("battery_status") == "red"
            ):
                failure_count = agreement.get("failure_count")
                lines.extend(
                    [
                        "!" * 72,
                        f"BASE POOL BATTERY: RED — {failure_count} FAILURES",
                        "Authenticated opt-in: --allow-gate-failed-base-pool",
                        f"Gates JSON SHA-256: {agreement.get('gates_json_sha256')}",
                        "Human publication decision required; this evidence "
                        "does not determine the preflight exit code.",
                    ]
                )
                failures = agreement.get("failures")
                if isinstance(failures, list):
                    for failure in failures:
                        if isinstance(failure, Mapping):
                            lines.append(
                                f"  [{failure.get('gate')}] {failure.get('message')}"
                            )
                lines.append("!" * 72)
        badge = {
            "PASS": "PASS   ",
            "FAIL": "FAIL   ",
            "AT_RISK": "AT-RISK",
            "SKIPPED": "SKIPPED",
        }
        lines.append("US release-gate preflight (no calibration solve)")
        lines.append("=" * 72)
        for check in self.checks:
            lines.append(f"[{badge[check.status]}] {check.name}")
            lines.append(f"          {check.summary}")
            for failure in check.failures:
                lines.append(f"    FAIL: {failure}")
            for at_risk in check.at_risks:
                lines.append(f"    RISK: {at_risk}")
        lines.append("-" * 72)
        lines.append(
            f"Overall: {self.status} (exit {self.exit_code})  "
            f"[FAIL=1, AT-RISK=2, clean=0]"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared linkage helpers
# ---------------------------------------------------------------------------


def selected_household_ids(base_frame: Frame, household_mask: np.ndarray) -> np.ndarray:
    """The household ids the selection mask marks, in household-table order."""
    household = base_frame.table("household")
    id_column = base_frame.schema.id_column("household")
    return household[id_column].to_numpy()[np.asarray(household_mask, dtype=bool)]


def _selected_entity_mask(
    frame: Frame, entity: str, selected_ids: np.ndarray
) -> np.ndarray:
    """Boolean mask over ``entity``'s rows: is each row in a selected household?

    The selection is at household granularity and every microcosm US entity nests
    inside exactly one household, so a group entity is selected iff its household
    is. The mapping uses only the schema's declared id / membership columns
    (``household_id``, ``person_household_id``, ``{group}_id``,
    ``person_{group}_id``) — never a guessed name.
    """
    schema = frame.schema
    selected = np.asarray(selected_ids)
    if entity == "household":
        ids = frame.table("household")[schema.id_column("household")].to_numpy()
        return np.isin(ids, selected)
    if entity == schema.person_entity:
        person_hh = frame.table("person")[
            schema.membership_column("household")
        ].to_numpy()
        return np.isin(person_hh, selected)
    # A group entity: map each group id to its household through the persons
    # (every group nests in exactly one household), then test membership.
    person = frame.table("person")
    group_to_household = (
        pd.DataFrame(
            {
                "group": person[schema.membership_column(entity)].to_numpy(),
                "household": person[schema.membership_column("household")].to_numpy(),
            }
        )
        .drop_duplicates("group")
        .set_index("group")["household"]
    )
    group_ids = frame.table(entity)[schema.id_column(entity)].to_numpy()
    group_household = group_to_household.reindex(group_ids).to_numpy()
    return np.isin(group_household, selected)


def _numeric_column(series: pd.Series) -> np.ndarray:
    """A leaf column as float64: booleans become their 1.0/0.0 mass."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=np.float64)
    return pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Check 1: selection carryover
# ---------------------------------------------------------------------------


def check_selection_carryover(
    base_frame: Frame,
    selection_source: SelectionSource,
    *,
    require_capital_gains_tail: bool = True,
) -> tuple[CheckResult, np.ndarray | None]:
    """Does the frozen selection map cleanly onto the base pool (no solve)?

    Reuses the frozen-support recovery contract directly: any unmapped,
    ambiguous, or non-unique-key identity raises the same hard error the release
    refuses on, surfaced here in seconds. Returns the household selection mask
    (or ``None`` on failure) so the smoke-probe audit can reuse it.
    """
    try:
        mask, report = selection_source.base_selection_mask(
            base_frame, mode="frozen_support"
        )
        selected_frame = base_frame.select(_household_person_mask(base_frame, mask))
        tail_retention = assert_puf_capital_gains_tail_survives_selection(
            base_frame,
            selected_frame,
            require_present=require_capital_gains_tail,
        )
    except ValueError as exc:
        return (
            CheckResult(
                name="selection_carryover",
                status="FAIL",
                summary=(
                    "selection-source manifest does NOT map cleanly onto the base "
                    "pool (the release would refuse this base)"
                ),
                failures=(str(exc),),
                details={"mode": "frozen_support"},
            ),
            None,
        )

    n_selected = int(report.n_selected)
    n_source = int(report.n_source)
    clean = (
        n_selected == n_source and report.n_unmapped == 0 and report.n_ambiguous == 0
    )
    row = {
        "n_source_identities": n_source,
        "n_selected": n_selected,
        "n_base_candidates": int(report.n_base_candidates),
        "n_unmapped": int(report.n_unmapped),
        "n_ambiguous": int(report.n_ambiguous),
    }
    return (
        CheckResult(
            name="selection_carryover",
            status="PASS" if clean else "FAIL",
            summary=(
                f"{n_selected}/{n_source} selection identities map onto "
                f"{report.n_base_candidates} base households (frozen_support)"
            ),
            failures=()
            if clean
            else (
                f"{report.n_unmapped} unmapped / {report.n_ambiguous} ambiguous "
                "identities under the join key.",
            ),
            rows=(row,),
            details={
                "join_key": list(selection_source.join_key),
                "puf_capital_gains_tail_retention": tail_retention,
                **row,
            },
        ),
        mask,
    )


# ---------------------------------------------------------------------------
# Check 2: zero-support preview
# ---------------------------------------------------------------------------


def _positive_specs(target_specs: Iterable[TargetSpec]) -> list[TargetSpec]:
    return [spec for spec in target_specs if float(spec.value) > 0.0]


def check_zero_support_preview(
    selected_frame: Frame,
    target_specs: Iterable[TargetSpec],
    *,
    weight_entity: str = "household",
    country: str = "us",
) -> CheckResult:
    """Compiled positive targets whose materialized support is ~0 pre-solve.

    A positive target whose estimate at base weights is ~0 has no records for
    calibration to move, so it stays a structural zero after the solve — exactly
    the release ``zero_support`` gate's condition, knowable now. Targets whose
    measure/filter are direct columns of ``selected_frame`` are checked by
    compiling the calibrate constraint system (the release's own materialization
    algebra); targets whose measure is engine-derived (needs a microsim pass) are
    reported as *not statically checkable* rather than silently passed.
    """
    positive = _positive_specs(target_specs)
    if not positive:
        return CheckResult(
            name="zero_support_preview",
            status="PASS",
            summary="no positive fiscal targets to preview",
            details={"positive_targets": 0},
        )

    registry = TargetRegistry(positive, country=country)
    try:
        problem = build_constraint_matrix(
            selected_frame, registry.to_target_set(), weight_entity=weight_entity
        )
        compiled_targets = problem.targets
        skipped = problem.skipped
        estimates = problem.estimates(problem.initial_weights.values)
        row_nnz = np.diff(problem.matrix.indptr)
    except ValueError:
        # Every positive target's measure is engine-derived on this raw frame:
        # nothing is statically checkable without a materialization pass.
        compiled_targets = ()
        skipped = tuple(_SkippedLike(target=spec) for spec in registry.to_target_set())
        estimates = np.zeros(0)
        row_nnz = np.zeros(0, dtype=int)

    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(compiled_targets):
        estimate = float(estimates[index])
        support = int(row_nnz[index])
        zero_support = support == 0 or abs(estimate) <= _ZERO_ESTIMATE_ATOL
        rows.append(
            {
                "target": getattr(target, "name", str(target)),
                "value": float(getattr(target, "value", 0.0)),
                "checkable": True,
                "support_records": support,
                "base_weight_estimate": estimate,
                "verdict": "zero_support" if zero_support else "supported",
            }
        )
        if zero_support:
            failures.append(
                f"{getattr(target, 'name', target)}: positive target "
                f"(value {float(getattr(target, 'value', 0.0)):,.0f}) materializes "
                f"{support} support record(s) and a base-weight estimate of "
                f"{estimate:,.2f} under the selection — a structural zero the "
                "solve cannot move."
            )

    not_checkable = 0
    for skip in skipped:
        target = getattr(skip, "target", skip)
        not_checkable += 1
        rows.append(
            {
                "target": getattr(target, "name", str(target)),
                "value": float(getattr(target, "value", 0.0)),
                "checkable": False,
                "verdict": "not_statically_checkable",
            }
        )

    status: PreflightStatus = "FAIL" if failures else "PASS"
    checkable = len(compiled_targets)
    summary = (
        f"{len(failures)} zero-support of {checkable} statically-checkable "
        f"positive targets ({not_checkable} derived/reform-materialized, not "
        "statically checkable)"
    )
    return CheckResult(
        name="zero_support_preview",
        status=status,
        summary=summary,
        failures=tuple(failures),
        rows=tuple(rows),
        details={
            "positive_targets": len(positive),
            "statically_checkable": checkable,
            "not_statically_checkable": not_checkable,
            "zero_support": len(failures),
        },
    )


@dataclass(frozen=True)
class _SkippedLike:
    """Uniform ``.target`` holder for the all-skipped fallback path."""

    target: Any


# ---------------------------------------------------------------------------
# Check 3: export-mass parity risk (at base weights)
# ---------------------------------------------------------------------------


def check_export_mass_parity_risk(
    pool_totals: Mapping[str, float],
    reference_totals: Mapping[str, float],
    *,
    relative_tolerance: float = _DEFAULT_RELATIVE_TOLERANCE,
    minimum_reference_total: float = _DEFAULT_MINIMUM_REFERENCE_TOTAL,
    reviewed_exclusions: Mapping[str, str] | None = None,
) -> CheckResult:
    """Pool mass at base weights vs the export-mass reference band.

    Runs the release export-mass parity gate's *own* band math
    (:func:`microcosm.build.gates.input_mass_parity_gate`) with the pre-solve pool
    totals as the candidate. A column outside its ``reference ± tolerance`` band
    at base weights is AT-RISK — the solve targets the calibration surface, not
    this untargeted column, so a pre-solve out-of-band mass is unlikely to land
    in-band. Columns in ``reviewed_exclusions`` are reported as excluded and
    never escalate (they are the documented, base-rebuild-tracked drifts).
    """
    reviewed_exclusions = dict(reviewed_exclusions or {})
    gate = input_mass_parity_gate(
        pool_totals,
        reference_totals,
        candidate_name="base_pool_at_base_weights",
        reference_name="export_mass_reference",
        relative_tolerance=relative_tolerance,
        minimum_reference_total=minimum_reference_total,
        reviewed_exclusions=reviewed_exclusions,
    )

    # gate.failures name the checked columns whose pre-solve pool mass is out of
    # band. Pre-solve these are risks, not release failures.
    out_of_band = {failure.split(":", 1)[0] for failure in gate.failures}

    rows: list[dict[str, Any]] = []
    at_risks: list[str] = []
    in_band = 0
    below_floor = 0
    excluded = 0
    for name in sorted(reference_totals):
        reference = float(reference_totals[name])
        pool = float(pool_totals.get(name, 0.0))
        drift = (
            (pool - reference) / abs(reference) if reference != 0.0 else float("inf")
        )
        low = reference - relative_tolerance * abs(reference)
        high = reference + relative_tolerance * abs(reference)
        if abs(reference) <= minimum_reference_total:
            verdict = "below_reference_floor"
            below_floor += 1
        elif name in reviewed_exclusions:
            verdict = "excluded"
            excluded += 1
        elif name in out_of_band:
            verdict = "out_of_band_at_base_weights"
            at_risks.append(
                f"{name}: pool mass {pool:,.0f} at base weights is outside the "
                f"band [{low:,.0f}, {high:,.0f}] (reference {reference:,.0f}, "
                f"drift {drift:+.1%}); confirm the calibration surface targets "
                "this column — a targeted column is lifted into band by the "
                "solve, an untargeted one fails the export-mass parity gate "
                "(the rental_income/microcosm#432 class)."
            )
        else:
            verdict = "in_band"
            in_band += 1
        rows.append(
            {
                "column": name,
                "reference_mass": reference,
                "pool_mass_at_base_weights": pool,
                "band_low": low,
                "band_high": high,
                "drift": drift,
                "verdict": verdict,
            }
        )

    status: PreflightStatus = "AT_RISK" if at_risks else "PASS"
    summary = (
        f"{len(at_risks)} column(s) out of band at base weights, {in_band} "
        f"in-band, {excluded} excluded, {below_floor} below reference floor"
    )
    return CheckResult(
        name="export_mass_parity_risk",
        status=status,
        summary=summary,
        at_risks=tuple(at_risks),
        rows=tuple(rows),
        details={
            "relative_tolerance": float(relative_tolerance),
            "minimum_reference_total": float(minimum_reference_total),
            "in_band": in_band,
            "out_of_band_at_base_weights": len(at_risks),
            "excluded": excluded,
            "below_reference_floor": below_floor,
        },
    )


# ---------------------------------------------------------------------------
# Check 4: smoke-probe support audit
# ---------------------------------------------------------------------------


def _leaf_entity(frame: Frame, leaf: str) -> str | None:
    try:
        return frame.column_entity(leaf)
    except ValueError:
        return None


def _sign_legs(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Weighted (positive_leg, negative_leg) of a signed column; N is <= 0."""
    positive = float(values[values > 0] @ weights[values > 0])
    negative = float(values[values < 0] @ weights[values < 0])
    return positive, negative


def check_smoke_probe_support(
    base_frame: Frame,
    selected_ids: np.ndarray,
    probes: Iterable[ReformCoverageProbe],
    *,
    min_selected_records: int = _MIN_SELECTED_RECORDS,
    cancellation_leg_ratio: float = _CANCELLATION_LEG_RATIO,
) -> CheckResult:
    """Reform-coverage probe leaves without selection support, or wrong-signed.

    For every probe leaf: pool nonzero support, *selected* nonzero support (via
    the frame's declared entity->household linkage), selected weighted mass, and
    the pool sign-leg decomposition.

    - **FAIL** when a leaf has pool support but zero selected support — an input
      restored after the selection froze, invisible to every count-based gate
      (microcosm#434, ``keogh_distributions``).
    - **AT-RISK** when selected support is thin (<``min_selected_records``), or
      when a materially-signed leaf's net weighted sign contradicts its probe's
      declared ``expected_sign`` (microcosm#432, ``farm_operations_income``
      imputed net-positive where the Schedule-F instrument is loss-heavy).
      Probes declaring ``expected_sign="either"`` assert no direction and are
      exempt from the sign check.
    """
    probes = tuple(probes)
    selected_ids = np.asarray(selected_ids)

    entity_selected: dict[str, np.ndarray] = {}
    entity_weights: dict[str, np.ndarray] = {}

    failures: list[str] = []
    at_risks: list[str] = []
    rows: list[dict[str, Any]] = []

    for probe in probes:
        # An "either"-sign probe (microcosm#437: net direction is a frame
        # property, not a coverage property) declares no direction, so the
        # sign-contradiction check below never applies to it.
        directional = probe.expected_sign in ("positive", "negative")
        expected_positive = probe.expected_sign == "positive"
        for leaf in probe.binding_inputs:
            entity = _leaf_entity(base_frame, leaf)
            if entity is None:
                rows.append(
                    {
                        "probe": probe.id,
                        "leaf": leaf,
                        "verdict": "absent_from_base_pool",
                    }
                )
                continue
            if entity not in entity_selected:
                entity_selected[entity] = _selected_entity_mask(
                    base_frame, entity, selected_ids
                )
                entity_weights[entity] = np.asarray(
                    base_frame.resolve_weights(entity).values, dtype=np.float64
                )
            selected_mask = entity_selected[entity]
            weights = entity_weights[entity]
            values = _numeric_column(base_frame.table(entity)[leaf])

            nonzero = values != 0.0
            pool_support = int(nonzero.sum())
            selected_nonzero = nonzero & selected_mask
            selected_support = int(selected_nonzero.sum())
            selected_mass = float(values[selected_nonzero] @ weights[selected_nonzero])
            positive_leg, negative_leg = _sign_legs(values, weights)
            net = positive_leg + negative_leg

            verdict = "supported"
            if pool_support > 0 and selected_support == 0:
                verdict = "structural_zero_in_selection"
                failures.append(
                    f"{probe.id}/{leaf}: {pool_support} pool carrier(s) but 0 in "
                    "the selection — an input the frozen selection cannot express; "
                    "the reform-coverage smoke probe will score a structural zero "
                    f"(issue {probe.issue})."
                )
            elif 0 < selected_support < min_selected_records:
                verdict = "thin_selection"
                at_risks.append(
                    f"{probe.id}/{leaf}: only {selected_support} selected carrier(s) "
                    f"(pool has {pool_support}); a tail-heavy column is one record "
                    "from a structural zero under this selection."
                )

            # Sign-structure contradiction, for genuinely-signed columns only.
            majority = max(positive_leg, -negative_leg)
            minority = min(positive_leg, -negative_leg)
            materially_signed = majority > 0 and minority / majority >= (
                cancellation_leg_ratio
            )
            net_positive = net > 0
            if (
                directional
                and materially_signed
                and pool_support > 0
                and net != 0.0
                and net_positive != expected_positive
            ):
                if verdict == "supported":
                    verdict = "sign_structure_contradicts_probe"
                at_risks.append(
                    f"{probe.id}/{leaf}: pool sign legs +{positive_leg:,.0f} / "
                    f"{negative_leg:,.0f} net {net:+,.0f} — a "
                    f"{'positive' if net_positive else 'negative'} net that "
                    f"contradicts the probe's expected_sign="
                    f"{probe.expected_sign!r} (issue {probe.issue})."
                )

            rows.append(
                {
                    "probe": probe.id,
                    "leaf": leaf,
                    "entity": entity,
                    "pool_support": pool_support,
                    "selected_support": selected_support,
                    "selected_weighted_mass": selected_mass,
                    "pool_positive_leg": positive_leg,
                    "pool_negative_leg": negative_leg,
                    "pool_net": net,
                    "expected_sign": probe.expected_sign,
                    "verdict": verdict,
                }
            )

    if failures:
        status = "FAIL"
    elif at_risks:
        status = "AT_RISK"
    else:
        status = "PASS"
    summary = (
        f"{len(failures)} leaf(s) with pool support but zero selected support, "
        f"{len(at_risks)} at-risk (thin selection / sign structure)"
    )
    return CheckResult(
        name="smoke_probe_support",
        status=status,
        summary=summary,
        failures=tuple(failures),
        at_risks=tuple(at_risks),
        rows=tuple(rows),
        details={
            "probes": len(probes),
            "min_selected_records": int(min_selected_records),
            "cancellation_leg_ratio": float(cancellation_leg_ratio),
        },
    )


# ---------------------------------------------------------------------------
# Runbook orchestrator (loads real artifacts; reuses the release tool register)
# ---------------------------------------------------------------------------


def _release_tool_module() -> Any:
    """Import ``build_us_fiscal_refresh_release`` (the release tool) by locating
    the repo ``tools/`` directory relative to this file.

    Keeps the export-mass register and the engine input-variable surface single-
    sourced in the release tool — this preflight never re-declares them.
    """
    repo_root = Path(__file__).resolve()
    for parent in repo_root.parents:
        candidate = parent / "tools" / "build_us_fiscal_refresh_release.py"
        if candidate.exists():
            if str(candidate.parent) not in sys.path:
                sys.path.insert(0, str(candidate.parent))
            import build_us_fiscal_refresh_release as release  # type: ignore

            return release
    raise ModuleNotFoundError(
        "Could not locate tools/build_us_fiscal_refresh_release.py to reuse the "
        "export-mass register; pass engine_input_variables and "
        "export_mass_reviewed_exclusions explicitly."
    )


def _load_ledger_target_specs(
    ledger_facts: Path,
    *,
    target_period: int | str,
    ledger_facts_sha256: str | None,
) -> tuple[TargetSpec, ...]:
    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from microcosm.build.us_runtime.fiscal_targets import (
        compile_us_fiscal_target_registry,
    )

    artifact = load_ledger_consumer_artifact(
        ledger_facts, expected_facts_sha256=ledger_facts_sha256
    )
    # age_targets mirrors the release tool's default (--age-targets on): the
    # feed's cross-period dollar facts fail the period contract un-aged, and
    # the preview must compile the registry the release run will calibrate.
    registry = compile_us_fiscal_target_registry(
        artifact.facts, target_period=target_period, age_targets=True
    )
    return registry.specs


def run_preflight(
    *,
    base_h5: str | Path,
    selection_source_manifest: str | Path,
    export_input_mass_reference_h5: str | Path | None = None,
    ledger_facts: str | Path | None = None,
    ledger_facts_sha256: str | None = None,
    target_period: int | str = 2024,
    relative_tolerance: float = _DEFAULT_RELATIVE_TOLERANCE,
    minimum_reference_total: float = _DEFAULT_MINIMUM_REFERENCE_TOTAL,
    probes: Iterable[ReformCoverageProbe] | None = None,
    engine_input_variables: Sequence[str] | None = None,
    export_mass_reviewed_exclusions: Mapping[str, str] | None = None,
    allow_gate_failed_base_pool: bool = False,
) -> PreflightReport:
    """Load the real artifacts and run every preflight check (no solve).

    Read-only against ``base_h5`` / ``export_input_mass_reference_h5``. The
    export-mass register and engine input-variable surface default to the
    release tool's own definitions (single-sourced, never re-declared here).
    """
    from microcosm.build.us_runtime.l0_refit_export import load_us_frame

    base_h5 = Path(base_h5)
    base_frame, base_pool, base_pool_manifest = _load_preflight_base(
        base_h5,
        allow_gate_failed_base_pool=allow_gate_failed_base_pool,
    )
    selection_source = load_selection_source_from_manifest(selection_source_manifest)

    checks: list[CheckResult] = []

    carryover, mask = check_selection_carryover(base_frame, selection_source)
    checks.append(carryover)

    selected = (
        selected_household_ids(base_frame, mask) if mask is not None else np.array([])
    )
    selected_frame = (
        base_frame.select(_household_person_mask(base_frame, mask))
        if mask is not None
        else None
    )

    # Check 2: zero-support preview (requires the compiled fiscal-target surface).
    if ledger_facts is not None and selected_frame is not None:
        target_specs = _load_ledger_target_specs(
            Path(ledger_facts),
            target_period=target_period,
            ledger_facts_sha256=ledger_facts_sha256,
        )
        checks.append(check_zero_support_preview(selected_frame, target_specs))
    else:
        checks.append(
            CheckResult(
                name="zero_support_preview",
                status="SKIPPED",
                summary=(
                    "no --ledger-facts feed provided; the compiled fiscal-target "
                    "surface is unavailable so zero-support cannot be previewed"
                ),
                details={"reason": "ledger_facts_not_provided"},
            )
        )

    # Check 3: export-mass parity risk (requires a reference H5).
    if export_input_mass_reference_h5 is not None:
        if engine_input_variables is None or export_mass_reviewed_exclusions is None:
            release = _release_tool_module()
            if engine_input_variables is None:
                engine_input_variables = release._engine_input_variables()
            if export_mass_reviewed_exclusions is None:
                export_mass_reviewed_exclusions = (
                    release.US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS
                )
        reference_frame = load_us_frame(Path(export_input_mass_reference_h5))
        pool_totals = us_input_mass_totals(base_frame, columns=engine_input_variables)
        reference_totals = us_input_mass_totals(
            reference_frame, columns=engine_input_variables
        )
        checks.append(
            check_export_mass_parity_risk(
                pool_totals,
                reference_totals,
                relative_tolerance=relative_tolerance,
                minimum_reference_total=minimum_reference_total,
                reviewed_exclusions=export_mass_reviewed_exclusions,
            )
        )
    else:
        checks.append(
            CheckResult(
                name="export_mass_parity_risk",
                status="SKIPPED",
                summary="no --export-input-mass-reference-h5 provided",
                details={"reason": "reference_h5_not_provided"},
            )
        )

    # Check 4: smoke-probe support audit.
    if mask is not None:
        probe_set = tuple(
            probes if probes is not None else us_release_reform_coverage_probes()
        )
        checks.append(check_smoke_probe_support(base_frame, selected, probe_set))
    else:
        checks.append(
            CheckResult(
                name="smoke_probe_support",
                status="SKIPPED",
                summary="selection did not map onto the base; cannot audit probes",
                details={"reason": "selection_carryover_failed"},
            )
        )

    inputs = {
        "base_h5": str(base_h5),
        "base_pool_manifest": (
            str(base_pool_manifest) if base_pool_manifest is not None else None
        ),
        "allow_gate_failed_base_pool": bool(allow_gate_failed_base_pool),
        "selection_source_manifest": str(selection_source_manifest),
        "export_input_mass_reference_h5": (
            str(export_input_mass_reference_h5)
            if export_input_mass_reference_h5 is not None
            else None
        ),
        "ledger_facts": str(ledger_facts) if ledger_facts is not None else None,
        "target_period": target_period,
        "relative_tolerance": float(relative_tolerance),
        "minimum_reference_total": float(minimum_reference_total),
    }
    return PreflightReport(
        checks=tuple(checks),
        inputs=inputs,
        base_pool=base_pool,
    )


def _file_sha256(path: Path) -> str:
    """Stream a file's SHA-256 (a pool H5 is gigabytes)."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_preflight_base(
    base_h5: Path,
    *,
    allow_gate_failed_base_pool: bool,
) -> tuple[Frame, dict[str, object] | None, Path | None]:
    """Load a generic base or authenticate a positively identified pool."""

    from microcosm.build.us_runtime.l0_refit_export import load_us_frame

    manifest_path = identify_us_multispine_pool_manifest(base_h5)
    if manifest_path is None:
        if allow_gate_failed_base_pool:
            raise ValueError(
                "--allow-gate-failed-base-pool was set, but --base-h5 does not "
                "identify as a US multispine pool."
            )
        # A denied pool whose sidecar and metadata row were stripped lands
        # here; its bytes are still its identity.
        refuse_denied_pool_h5_digest(
            _file_sha256(base_h5),
            consumer="US release-gate preflight --base-h5 (generic path)",
        )
        return load_us_frame(base_h5), None, None

    frame, manifest, authenticated_pool_h5 = (
        load_authenticated_us_multispine_pool_for_release(
            manifest_path,
            allow_terminal_gate_failure=allow_gate_failed_base_pool,
        )
    )
    require_authenticated_us_multispine_pool_h5(
        base_h5,
        authenticated_pool_h5,
        consumer="US release-gate preflight --base-h5",
    )
    receipt = us_multispine_pool_release_receipt(
        manifest,
        authenticated_pool_h5,
        allow_gate_failed_base_pool=allow_gate_failed_base_pool,
    )
    return frame, receipt, manifest_path


def _household_person_mask(base_frame: Frame, household_mask: np.ndarray) -> np.ndarray:
    """Person mask for the households in ``household_mask`` (for ``Frame.select``)."""
    schema = base_frame.schema
    household = base_frame.table("household")
    selected = household[schema.id_column("household")].to_numpy()[
        np.asarray(household_mask, dtype=bool)
    ]
    person_hh = base_frame.table("person")[
        schema.membership_column("household")
    ].to_numpy()
    return np.isin(person_hh, selected)
