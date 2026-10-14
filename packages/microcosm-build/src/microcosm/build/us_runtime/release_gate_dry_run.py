"""The US release dry run's certainty model, register diffs and report.

Route A release run ``310842b986d7`` (2026-09-26) ran 13,707 s and failed on
one input it had carried from the start: the per-run QRF tail-concentration
waiver register (``--qrf-tail-concentration-exclusions``). After
microcosm#1033 changed the QRF draws, four columns crossed the 0.75 top-100
share unwaived (``bond_assets``, ``domestic_production_ald``,
``estate_income``, ``w2_wages_from_qualified_business``), two entries went
stale (``alimony_expense``, ``qualified_bdc_income``) and one went thin
(``farm_income``). The release grades that register on the calibrated export
frame, and so it grades the export input-mass register and the stored-input,
SPM-composition and input-coverage gates. The run's own ``build.timing``
(``calibration_diagnostics.json``) puts that frame after at most 2,049 s of
base load, input stages and pre-solve gates, 9,951 s of target compilation
and 1,668 s of calibration.

``tools/preflight_us_release_gates.py`` grades the raw base in minutes. But the
register surface belongs to the *staged* frame. Five QRF stages run inside the
release (``scf_wealth``, ``org_wages``, ``ssi_disability_criteria``,
``sipp_head_start`` and ``voluntary_filing_input``), and ``bond_assets`` exists
only after them.

``tools/build_us_fiscal_refresh_release.py <release args> --dry-run-gates-report
PATH`` runs the release's own code path. That means the same argv, base load,
input stages and pre-solve gates. It stops where the staged frame is handed to
target materialization. At that point it grades the staged frame at its base
weights, using the release tool's own gate functions. It writes the report
built here and exits 1 on any certain failure, 2 on AT-RISK only, and 0 when
clean. ``tools/dry_run_us_release_gates.py`` wraps it for a saved release
config.

This module holds only what the release tool does not: the certainty model,
the register-diff rows and the report. Gate semantics stay in the release tool.

Certainty model
---------------
A base-weight verdict is *certain* when no solve inside the stated margins can
change it. The solve can move three kinds of thing:

* **Record weights.** The tail and mass gates weigh records at the calibrated
  weights. The full-pool solve parametrizes every weight as ``exp(log w)``
  (``microcosm.calibrate.solve``). It therefore never zeroes a positive weight
  and never adds a record, so carrier counts and nonzero shares are exact at
  base weights. A top-``k`` share is not exact. The tail margins bound how far
  calibration may raise it (:data:`DEFAULT_TAIL_SHARE_RISE_MARGIN`) or lower
  it (:data:`DEFAULT_TAIL_SHARE_FALL_MARGIN`).
* **Record support**, on the L0 path only. A sparse release keeps the
  households its L0 selection picks and refits their weights. Nonzero shares
  and carrier counts move, and top-k shares move by an amount nobody has
  measured: the tail margins come from full-pool runs, so on this path the L0
  tail margins apply instead, and by default they admit any share. Carriers
  can only fall, since records are dropped and never added, so a thin column
  stays thin. A value-only verdict the release re-grades on the selected
  export stays certain only while every signal it rests on keeps a record
  under the carrier-retention margin. The support margins are conservative
  defaults, not measurements.
* **Nothing else.** The staged frame fixes which columns exist, their dtypes,
  whether a column is a QRF output, and every value of every record.

For each graded item the dry run lists the release classes the margins allow
and maps each class to the release's verdict. If every class gives one
verdict, that verdict is certain (FAIL if it fails). If the classes give
several verdicts and some of them fail, the item is AT-RISK. With every margin
at zero on the full-pool path, each item allows exactly its base-weight class.
The dry run's verdict is then the release gate's own verdict at base weights,
the differential property the tests pin.

Evidence for the default margins
--------------------------------
See ``experiments/us-release-dry-run-margin-evidence.md``.

* **Tail shares, run 310842b986d7.** The release's own tail gate was
  recomputed at base and at calibrated weights on 32 of the 33 columns it
  checked, read from the raw base (``bond_assets`` exists only after the
  release's stages). The calibrated side reproduces the release's recorded
  ``qrf_tail_concentration.json`` exactly for 31 of them. The exception, off
  by 0.0013, is a column a release-time stage rewrites. The shifts
  (calibrated minus base) run from -0.033 to +0.299, median +0.105.
* **The dry run itself, on that run's config** (at commit ``21c1f9ba3``, the
  tool's first revision). At the stop point, the staged
  frame with the release's saved final weights attached reproduces the
  release's recorded tail surface exactly: every share, carrier count, refusal
  and register-mismatch entry. ``bond_assets`` moves from 0.520 at base weights
  to 0.766, inside the margins.
* **Tail shares, route A d177 register.** Its six initial-weight versus
  calibrated pairs give +0.06 to +0.29.
* **Export input mass.** Between base and calibrated weights, the drift of the
  20 worst-drift columns moved from -0.30 to +0.84. That is larger than the
  +/-0.5 band itself, so no nonzero-mass verdict is certain at base weights.
  The mass margin only widens the AT-RISK net.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

from microcosm.build.us_runtime.release_gate_preflight import PreflightReport
from microcosm.build.us_runtime.spm_composition import CheckResult, PreflightStatus

__all__ = [
    "DEFAULT_L0_TAIL_SHARE_MARGIN",
    "DEFAULT_MASS_DRIFT_MARGIN",
    "DEFAULT_SUPPORT_CARRIER_RETENTION",
    "DEFAULT_SUPPORT_NONZERO_SHARE_MARGIN",
    "DEFAULT_TAIL_SHARE_FALL_MARGIN",
    "DEFAULT_TAIL_SHARE_RISE_MARGIN",
    "FAILING_TAIL_VERDICTS",
    "MARGIN_EVIDENCE",
    "NOT_PREVIEWABLE_GATES",
    "DryRunMargins",
    "ReleaseDryRunReport",
    "TailClass",
    "TailColumnDryRun",
    "TailVerdict",
    "base_tail_classes",
    "certain_lines_check",
    "apply_evidence_ownership",
    "bound_zero_support_on_l0",
    "classify_tail_column",
    "degenerate_input_register_check",
    "ecps_parity_register_check",
    "export_input_mass_check",
    "export_signal_regrades_check",
    "input_coverage_register_check",
    "not_previewable_check",
    "possible_tail_classes",
    "post_stop_error_report",
    "pre_solve_refusal_report",
    "qrf_tail_register_check",
    "qrf_tail_register_unloadable_check",
    "tail_register_verdict",
    "tail_share_classes",
]

#: How far calibration may raise a column's top-k weighted-mass share. The
#: largest rise measured on route A run 310842b986d7 was +0.299
#: (``alimony_income``), and the d177 register's largest was +0.29. 0.35
#: covers both.
DEFAULT_TAIL_SHARE_RISE_MARGIN = 0.35

#: How far calibration may lower a top-k share. Run 310842b986d7's only fall
#: was -0.033 (``partnership_income``). 0.05 covers it.
DEFAULT_TAIL_SHARE_FALL_MARGIN = 0.05

#: How close to the export input-mass band edge (in relative-drift units) an
#: in-band column must sit at base weights to be flagged. Measured drift moves
#: were larger than the band (see the module docstring), so this is not a
#: bound. It only sets which in-band columns are called out.
DEFAULT_MASS_DRIFT_MARGIN = 0.10

#: L0 path only: how far an L0 selection and refit may raise or lower a top-k
#: share. Nothing has measured it (every measured shift is from full-pool runs),
#: so the default admits any share: on the L0 path no checked column's share
#: verdict is certain until an operator with L0 evidence narrows it.
DEFAULT_L0_TAIL_SHARE_MARGIN = 1.0

#: L0 path only: how far a column's record nonzero share may move under the
#: solve's household selection. A conservative default, not a measurement.
DEFAULT_SUPPORT_NONZERO_SHARE_MARGIN = 0.02

#: L0 path only: the smallest fraction of the records carrying one signal (a
#: column's carriers, the records holding one value, a vintage's reporters) that
#: the solve's household selection may keep. A conservative default, not a
#: measurement.
DEFAULT_SUPPORT_CARRIER_RETENTION = 0.25

#: Where the default margins' evidence lives.
MARGIN_EVIDENCE = "experiments/us-release-dry-run-margin-evidence.md"

_MARGIN_NAMES = (
    "tail_share_rise",
    "tail_share_fall",
    "mass_drift",
    "support_nonzero_share",
    "l0_tail_share_rise",
    "l0_tail_share_fall",
)


@dataclass(frozen=True)
class DryRunMargins:
    """The stated bounds on what the solve may move (see the module docstring).

    Attributes:
        tail_share_rise: Largest rise of a top-k share under calibration.
        tail_share_fall: Largest fall of a top-k share under calibration.
        mass_drift: Distance from the export input-mass band edge inside which
            an in-band column is flagged.
        support_nonzero_share: L0 path only. Largest move of a column's record
            nonzero share.
        support_carrier_retention: L0 path only. Smallest kept fraction of the
            records carrying one signal, in ``(0, 1]``.
        l0_tail_share_rise: L0 path only. Largest rise of a top-k share under
            an L0 selection and refit (unmeasured; the default admits any).
        l0_tail_share_fall: L0 path only. Largest fall of a top-k share under
            an L0 selection and refit (unmeasured; the default admits any).
    """

    tail_share_rise: float = DEFAULT_TAIL_SHARE_RISE_MARGIN
    tail_share_fall: float = DEFAULT_TAIL_SHARE_FALL_MARGIN
    mass_drift: float = DEFAULT_MASS_DRIFT_MARGIN
    support_nonzero_share: float = DEFAULT_SUPPORT_NONZERO_SHARE_MARGIN
    support_carrier_retention: float = DEFAULT_SUPPORT_CARRIER_RETENTION
    l0_tail_share_rise: float = DEFAULT_L0_TAIL_SHARE_MARGIN
    l0_tail_share_fall: float = DEFAULT_L0_TAIL_SHARE_MARGIN

    def __post_init__(self) -> None:
        for name in _MARGIN_NAMES:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"dry-run margin {name} must be a finite number >= 0, got "
                    f"{value!r}."
                )
        retention = self.support_carrier_retention
        if (
            isinstance(retention, bool)
            or not isinstance(retention, (int, float))
            or not math.isfinite(retention)
            or not 0.0 < retention <= 1.0
        ):
            raise ValueError(
                "dry-run margin support_carrier_retention must be in (0, 1], "
                f"got {retention!r}."
            )

    @classmethod
    def exact(cls) -> DryRunMargins:
        """Margins under which every item keeps its base-weight class."""

        return cls(
            tail_share_rise=0.0,
            tail_share_fall=0.0,
            mass_drift=0.0,
            support_nonzero_share=0.0,
            support_carrier_retention=1.0,
            l0_tail_share_rise=0.0,
            l0_tail_share_fall=0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tail_share_rise": float(self.tail_share_rise),
            "tail_share_fall": float(self.tail_share_fall),
            "mass_drift": float(self.mass_drift),
            "support_nonzero_share": float(self.support_nonzero_share),
            "support_carrier_retention": float(self.support_carrier_retention),
            "l0_tail_share_rise": float(self.l0_tail_share_rise),
            "l0_tail_share_fall": float(self.l0_tail_share_fall),
            "evidence": MARGIN_EVIDENCE,
        }

    def keeps_a_record(self, signal_records: int) -> bool:
        """Whether a selection inside the margins keeps one of these records.

        On the L0 path the selection keeps at least
        ``signal_records * support_carrier_retention`` of the records carrying
        one signal, so at least one survives iff that is at least one.
        """

        return signal_records * self.support_carrier_retention >= 1.0


# ---------------------------------------------------------------------------
# QRF tail-concentration register
# ---------------------------------------------------------------------------

#: Where the release's tail gate puts one column:
#:
#: * ``not_qrf_output``: a register entry that no ``fit_weighted_qrf`` stage
#:   produces.
#: * ``absent``: no entity carries the column.
#: * ``non_numeric``: a boolean column.
#: * ``dense``: record nonzero share above the sparse maximum.
#: * ``thin``: fewer weighted carriers than the gate's minimum.
#: * ``over``: the top-k share exceeds the threshold.
#: * ``at_or_under``: the top-k share is at or under the threshold.
TailClass = Literal[
    "not_qrf_output",
    "absent",
    "non_numeric",
    "dense",
    "thin",
    "over",
    "at_or_under",
]

#: The release's verdict on one column:
#:
#: * ``used``: a register entry for a concentrated column.
#: * ``stale``: a register entry for a checked column at or under the
#:   threshold.
#: * ``unused``: any other register entry.
#: * ``unwaived``: a concentrated column with no entry.
#: * ``waived``: an unwaived column under ``--allow-qrf-tail-concentration``.
#: * ``owned``: an unwaived column under ``--evidence-release`` whose release
#:   line an owner pattern matches, so the evidence tier ships it as a known
#:   failure instead of refusing.
#: * ``ok``: anything else.
TailVerdict = Literal["ok", "used", "unwaived", "waived", "owned", "stale", "unused"]

#: Verdicts the release refuses. ``stale`` and ``unused`` ride the
#: register-mismatch line, which is appended whatever
#: ``--allow-qrf-tail-concentration`` says and which ``--evidence-release``
#: always refuses. ``unwaived`` rides the gate's own line, which that flag
#: suppresses (into ``waived``) and an evidence owner can own (into ``owned``).
FAILING_TAIL_VERDICTS: frozenset[str] = frozenset({"unwaived", "stale", "unused"})

_TAIL_CLASSES: tuple[str, ...] = get_args(TailClass)
_STRUCTURAL_TAIL_CLASSES = frozenset({"not_qrf_output", "absent", "non_numeric"})


def tail_register_verdict(
    tail_class: TailClass,
    *,
    in_register: bool,
    allow_concentration: bool = False,
    evidence_owned: bool = False,
) -> TailVerdict:
    """The release's verdict on a column in ``tail_class``.

    This mirrors ``_record_qrf_tail_concentration_gate`` in the release tool:
    ``tail_concentration_gate`` plus ``_qrf_tail_register_mismatch``, and
    under ``--evidence-release`` the owner check of its terminal batch.

    Args:
        tail_class: The column's class under the release's tail gate.
        in_register: Whether the per-run register names the column.
        allow_concentration: ``--allow-qrf-tail-concentration``.
        evidence_owned: ``--evidence-release`` with an owner pattern matching
            this column's ``QRF tail concentration failed:`` line.

    Raises:
        ValueError: For an unknown class.
    """

    if tail_class not in _TAIL_CLASSES:
        raise ValueError(f"Unknown QRF tail class {tail_class!r}.")
    if in_register:
        if tail_class == "over":
            return "used"
        if tail_class == "at_or_under":
            return "stale"
        return "unused"
    if tail_class == "over":
        if allow_concentration:
            return "waived"
        return "owned" if evidence_owned else "unwaived"
    return "ok"


def tail_share_classes(
    top_share: float,
    *,
    max_top_share: float,
    rise_margin: float,
    fall_margin: float,
) -> frozenset[TailClass]:
    """Which side of the threshold a top-k share can land on after calibration.

    The release calls a checked column concentrated iff its share is strictly
    above ``max_top_share``. Under the margins the calibrated share lies in
    ``[top_share - fall_margin, top_share + rise_margin]``.
    """

    classes: set[TailClass] = set()
    if top_share + rise_margin > max_top_share:
        classes.add("over")
    if top_share - fall_margin <= max_top_share:
        classes.add("at_or_under")
    return frozenset(classes)


def possible_tail_classes(
    base_class: TailClass,
    *,
    top_share: float | None,
    nonzero_share: float | None,
    carriers: int | None,
    support_fixed: bool,
    margins: DryRunMargins,
    max_top_share: float,
    sparse_nonzero_share_max: float,
    min_nonzero_records: int,
) -> frozenset[TailClass]:
    """Every class the release can put a column in, given its base class.

    Args:
        base_class: The column's class at base weights on the staged support.
        top_share: Its top-k share at base weights (checked columns only).
        nonzero_share: Its record nonzero share on the staged support.
        carriers: Its carrier count at base weights. For a column the gate did
            not weigh, this is its finite nonzero record count.
        support_fixed: True on the full-pool path, where the export keeps
            every staged record and every weight stays positive.
        margins: The stated bounds.
        max_top_share: The gate's blocking share threshold.
        sparse_nonzero_share_max: The release's dense/sparse cut.
        min_nonzero_records: The gate's thin cut.

    Raises:
        ValueError: If a checked column comes without its share.
    """

    if base_class not in _TAIL_CLASSES:
        raise ValueError(f"Unknown QRF tail class {base_class!r}.")
    if base_class in _STRUCTURAL_TAIL_CLASSES:
        return frozenset({base_class})
    if base_class == "thin":
        # Carriers are records with finite |value| * weight > 0. No solve adds
        # a record, and the full-pool solve keeps every weight positive, so the
        # count cannot rise.
        return frozenset({"thin"})
    if base_class in ("over", "at_or_under") and top_share is None:
        raise ValueError(f"A checked column needs its top share ({base_class}).")

    def share_band() -> frozenset[TailClass]:
        assert top_share is not None
        # The tail margins were measured on full-pool runs only; an L0 selection
        # and refit get their own (unmeasured) margins.
        return tail_share_classes(
            top_share,
            max_top_share=max_top_share,
            rise_margin=(
                margins.tail_share_rise if support_fixed else margins.l0_tail_share_rise
            ),
            fall_margin=(
                margins.tail_share_fall if support_fixed else margins.l0_tail_share_fall
            ),
        )

    if support_fixed:
        if base_class == "dense":
            return frozenset({"dense"})
        return share_band()

    # L0 path: the solve chooses which households the export keeps.
    classes: set[TailClass] = set()
    thin_possible = (
        carriers is not None
        and carriers * margins.support_carrier_retention < min_nonzero_records
    )
    if base_class == "dense":
        classes.add("dense")
        if (
            nonzero_share is not None
            and nonzero_share - margins.support_nonzero_share
            <= sparse_nonzero_share_max
        ):
            # It may turn sparse. The gate never weighed it, so its share is
            # unknown and both sides of the threshold stay possible.
            classes.update(("over", "at_or_under"))
            if thin_possible:
                classes.add("thin")
        return frozenset(classes)
    classes.update(share_band())
    if (
        nonzero_share is not None
        and nonzero_share + margins.support_nonzero_share > sparse_nonzero_share_max
    ):
        classes.add("dense")
    if thin_possible:
        classes.add("thin")
    return frozenset(classes)


@dataclass(frozen=True)
class TailColumnDryRun:
    """One column's dry-run verdict under the QRF tail register."""

    column: str
    in_register: bool
    base_class: TailClass
    possible_classes: frozenset[TailClass]
    base_verdict: TailVerdict
    possible_verdicts: frozenset[TailVerdict]
    top_share: float | None = None
    nonzero_share: float | None = None
    carriers: int | None = None

    @property
    def status(self) -> PreflightStatus:
        # An owned failure is not a refusal, but the release ships it as a
        # known failure, so it needs a human's eye: AT-RISK, never PASS.
        failing = self.possible_verdicts & FAILING_TAIL_VERDICTS
        if failing and failing == self.possible_verdicts:
            return "FAIL"
        if failing or "owned" in self.possible_verdicts:
            return "AT_RISK"
        return "PASS"

    @property
    def certain(self) -> bool:
        return len(self.possible_verdicts) == 1

    def to_row(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "in_register": self.in_register,
            "base_class": self.base_class,
            "possible_classes": sorted(self.possible_classes),
            "verdict_at_base_weights": self.base_verdict,
            "possible_verdicts": sorted(self.possible_verdicts),
            "certain": self.certain,
            "status": self.status,
            "top_share_at_base_weights": self.top_share,
            "nonzero_share": self.nonzero_share,
            "carriers_at_base_weights": self.carriers,
        }


def classify_tail_column(
    column: str,
    *,
    base_class: TailClass,
    in_register: bool,
    allow_concentration: bool,
    top_share: float | None,
    nonzero_share: float | None,
    carriers: int | None,
    support_fixed: bool,
    margins: DryRunMargins,
    max_top_share: float,
    sparse_nonzero_share_max: float,
    min_nonzero_records: int,
    evidence_owned: bool = False,
) -> TailColumnDryRun:
    """Grade one column: its base verdict and every verdict the solve allows."""

    possible = possible_tail_classes(
        base_class,
        top_share=top_share,
        nonzero_share=nonzero_share,
        carriers=carriers,
        support_fixed=support_fixed,
        margins=margins,
        max_top_share=max_top_share,
        sparse_nonzero_share_max=sparse_nonzero_share_max,
        min_nonzero_records=min_nonzero_records,
    )
    return TailColumnDryRun(
        column=column,
        in_register=in_register,
        base_class=base_class,
        possible_classes=possible,
        base_verdict=tail_register_verdict(
            base_class,
            in_register=in_register,
            allow_concentration=allow_concentration,
            evidence_owned=evidence_owned,
        ),
        possible_verdicts=frozenset(
            tail_register_verdict(
                tail_class,
                in_register=in_register,
                allow_concentration=allow_concentration,
                evidence_owned=evidence_owned,
            )
            for tail_class in possible
        ),
        top_share=top_share,
        nonzero_share=nonzero_share,
        carriers=carriers,
    )


def base_tail_classes(
    *,
    qrf_outputs: Iterable[str],
    register: Iterable[str],
    surface: Mapping[str, Any],
    gate_details: Mapping[str, Any],
) -> dict[str, TailClass]:
    """Each QRF output's and register entry's class, read off the release gate.

    ``surface`` is the second value ``_qrf_tail_concentration_gate`` returns,
    and ``gate_details`` is its gate's ``details``. Nothing is recomputed here.

    Raises:
        ValueError: If the surface does not partition the QRF outputs. That
            would mean the release's surface changed shape under this reader.
    """

    max_top_share = float(gate_details["max_top_share"])
    top_share = {str(k): float(v) for k, v in gate_details["top_share"].items()}
    thin = {str(k) for k in gate_details["thin_columns"]}
    by_bucket: dict[str, set[str]] = {
        "absent": {str(c) for c in surface["absent_columns"]},
        "non_numeric": {str(c) for c in surface["non_numeric_columns"]},
        "dense": {str(c) for c in surface["dense_columns"]},
    }
    checked = {str(c) for c in surface["checked_sparse_columns"]}
    classes: dict[str, TailClass] = {}
    for column in sorted({str(c) for c in qrf_outputs}):
        buckets = [name for name, members in by_bucket.items() if column in members]
        if column in checked:
            buckets.append("checked")
        if len(buckets) != 1:
            raise ValueError(
                f"QRF output {column!r} is in {buckets or 'no'} surface bucket(s); "
                "the release's tail surface must place each output exactly once."
            )
        bucket = buckets[0]
        if bucket != "checked":
            classes[column] = bucket  # type: ignore[assignment]
        elif column in thin:
            classes[column] = "thin"
        elif column in top_share:
            classes[column] = (
                "over" if top_share[column] > max_top_share else "at_or_under"
            )
        else:
            raise ValueError(
                f"Checked QRF output {column!r} has neither a top share nor a "
                "thin carrier count in the gate details."
            )
    for column in sorted({str(c) for c in register}):
        classes.setdefault(column, "not_qrf_output")
    return classes


def _share(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _describe_tail_class(row: TailColumnDryRun, gate: Mapping[str, Any]) -> str:
    top_k = int(gate["top_k"])
    max_top_share = float(gate["max_top_share"])
    base = row.base_class
    if base == "not_qrf_output":
        return "no fit_weighted_qrf stage produces it"
    if base == "absent":
        return "no entity of the staged frame carries it"
    if base == "non_numeric":
        return "it is a boolean column"
    if base == "dense":
        return (
            f"it is dense (nonzero on {_share(row.nonzero_share)} of its records, "
            f"above {float(gate['sparse_nonzero_share_max']):.2f})"
        )
    if base == "thin":
        return (
            f"it is thin ({row.carriers} weighted carriers, under "
            f"{int(gate['min_nonzero_records'])})"
        )
    relation = "over" if base == "over" else "at or under"
    return (
        f"its top-{top_k} weighted-mass share is {_share(row.top_share)} at base "
        f"weights, {relation} {max_top_share:.2f}"
    )


_VERDICT_PHRASES: Mapping[str, str] = {
    "used": "a used register entry",
    "stale": "a stale register entry",
    "unused": "an unused register entry",
    "unwaived": "an unwaived concentrated column",
    "waived": "a concentrated column waived by --allow-qrf-tail-concentration",
    "owned": (
        "a concentrated column the evidence tier would ship as an owned known failure"
    ),
    "ok": "not a register matter",
}

_REMEDIES: Mapping[str, str] = {
    "stale": "remove the entry",
    "unused": "remove the entry",
    "unwaived": (
        "fix the imputation or add a reviewed register entry naming the tracked defect"
    ),
}


def _tail_line(
    row: TailColumnDryRun,
    gate: Mapping[str, Any],
    margins: DryRunMargins,
    *,
    support_fixed: bool,
) -> str:
    description = _describe_tail_class(row, gate)
    if row.status == "FAIL":
        verdicts = sorted(row.possible_verdicts)
        verdict = verdicts[0] if len(verdicts) == 1 else " or ".join(verdicts)
        remedy = "; ".join(
            dict.fromkeys(_REMEDIES[v] for v in verdicts if v in _REMEDIES)
        )
        why = (
            "no solve inside the stated margins moves it across a cut"
            if row.base_class not in _STRUCTURAL_TAIL_CLASSES
            else "that is fixed by the staged frame"
        )
        return (
            f"{row.column}: {_VERDICT_PHRASES.get(verdict, verdict)} — "
            f"{description}; {why}. The release refuses it: {remedy}."
        )
    if support_fixed:
        margin_note = (
            f"calibration may raise a share by up to {margins.tail_share_rise:.2f} "
            f"or lower it by up to {margins.tail_share_fall:.2f}"
        )
    else:
        margin_note = (
            "on the L0 path the solve picks the export's records and refits their "
            "weights: a share may rise by up to "
            f"{margins.l0_tail_share_rise:.2f} or fall by up to "
            f"{margins.l0_tail_share_fall:.2f} (unmeasured), the nonzero share may "
            f"move by {margins.support_nonzero_share:.2f}, and carrier retention is "
            f"at least {margins.support_carrier_retention:.2f}"
        )
    return (
        f"{row.column}: {_VERDICT_PHRASES.get(row.base_verdict, row.base_verdict)} "
        f"at base weights — {description}. Within the stated margins "
        f"({margin_note}) the release may find it "
        f"{' or '.join(sorted(row.possible_verdicts))}."
    )


def qrf_tail_register_check(
    *,
    qrf_outputs: Iterable[str],
    register: Mapping[str, str],
    surface: Mapping[str, Any],
    gate_details: Mapping[str, Any],
    release_lines_at_base_weights: Sequence[str],
    support_fixed: bool,
    margins: DryRunMargins,
    allow_concentration: bool,
    register_source: Mapping[str, Any],
    evidence_owner: Callable[[str], str | None] | None = None,
) -> CheckResult:
    """The per-run QRF tail register against the staged frame at base weights.

    Every QRF output and every register entry is graded. Rows carry the base
    class, the classes and verdicts the margins allow, and the measured
    numbers. Failure lines are the certain refusals and at-risk lines the
    solve-dependent ones. ``release_lines_at_base_weights`` holds the release's
    own failure lines at base weights, for the record.

    Args:
        qrf_outputs: ``_qrf_imputed_source_outputs()``.
        register: The loaded per-run register (column -> reason).
        surface: The surface dict from ``_qrf_tail_concentration_gate``, which
            must carry ``nonzero_shares`` and ``nonzero_records``.
        gate_details: That call's gate details.
        release_lines_at_base_weights: The lines the release would append at
            these weights.
        support_fixed: True on the full-pool path.
        margins: The stated bounds.
        allow_concentration: ``--allow-qrf-tail-concentration``.
        register_source: The register's path, sha256 and entry count.
        evidence_owner: Under ``--evidence-release``, the owner (or ``None``) of
            a column's ``QRF tail concentration failed:`` line; ``None`` outside
            the evidence tier. The register-mismatch line is never ownable.
    """

    gate = {
        "top_k": gate_details["top_k"],
        "max_top_share": gate_details["max_top_share"],
        "min_nonzero_records": gate_details["min_nonzero_records"],
        "sparse_nonzero_share_max": surface["sparse_nonzero_share_max"],
    }
    classes = base_tail_classes(
        qrf_outputs=qrf_outputs,
        register=register,
        surface=surface,
        gate_details=gate_details,
    )
    nonzero_shares = {
        str(k): float(v) for k, v in dict(surface.get("nonzero_shares", {})).items()
    }
    nonzero_records = {
        str(k): int(v) for k, v in dict(surface.get("nonzero_records", {})).items()
    }
    carrier_counts = {
        str(k): int(v) for k, v in dict(gate_details["carrier_counts"]).items()
    }
    thin_counts = {
        str(k): int(v) for k, v in dict(gate_details["thin_columns"]).items()
    }
    top_share = {str(k): float(v) for k, v in gate_details["top_share"].items()}

    rows: list[TailColumnDryRun] = []
    owners: dict[str, str] = {}
    for column, base_class in classes.items():
        carriers = carrier_counts.get(
            column, thin_counts.get(column, nonzero_records.get(column))
        )
        owner = evidence_owner(column) if evidence_owner is not None else None
        if owner is not None:
            owners[column] = owner
        rows.append(
            classify_tail_column(
                column,
                base_class=base_class,
                in_register=column in register,
                allow_concentration=allow_concentration,
                top_share=top_share.get(column),
                nonzero_share=nonzero_shares.get(column),
                carriers=carriers,
                support_fixed=support_fixed,
                margins=margins,
                max_top_share=float(gate["max_top_share"]),
                sparse_nonzero_share_max=float(gate["sparse_nonzero_share_max"]),
                min_nonzero_records=int(gate["min_nonzero_records"]),
                evidence_owned=owner is not None,
            )
        )

    failures = tuple(
        _tail_line(row, gate, margins, support_fixed=support_fixed)
        for row in rows
        if row.status == "FAIL"
    )
    at_risks = tuple(
        _tail_line(row, gate, margins, support_fixed=support_fixed)
        for row in rows
        if row.status == "AT_RISK"
    )
    status: PreflightStatus = "FAIL" if failures else "AT_RISK" if at_risks else "PASS"
    base_counts: dict[str, int] = {}
    for row in rows:
        if row.in_register or row.base_verdict in FAILING_TAIL_VERDICTS:
            base_counts[row.base_verdict] = base_counts.get(row.base_verdict, 0) + 1
    summary = (
        f"{len(register)} register entr{'y' if len(register) == 1 else 'ies'}; "
        f"at base weights: "
        + (
            ", ".join(
                f"{count} {verdict}" for verdict, count in sorted(base_counts.items())
            )
            or "nothing to report"
        )
        + f"; {len(failures)} certain refusal(s), {len(at_risks)} solve-dependent"
    )
    return CheckResult(
        name="qrf_tail_register",
        status=status,
        summary=summary,
        failures=failures,
        at_risks=at_risks,
        rows=tuple(row.to_row() for row in rows),
        details={
            "register": dict(register_source),
            "calibration_path": "full_pool" if support_fixed else "l0_selection",
            "allow_qrf_tail_concentration": bool(allow_concentration),
            "top_k": int(gate["top_k"]),
            "max_top_share": float(gate["max_top_share"]),
            "min_nonzero_records": int(gate["min_nonzero_records"]),
            "sparse_nonzero_share_max": float(gate["sparse_nonzero_share_max"]),
            "margins": margins.to_dict(),
            "release_lines_at_base_weights": list(release_lines_at_base_weights),
            **(
                {"evidence_owners": dict(sorted(owners.items()))}
                if evidence_owner is not None
                else {}
            ),
        },
    )


def qrf_tail_register_unloadable_check(
    register_source: Mapping[str, Any],
    error: BaseException,
    *,
    evidence_owner: str | None = None,
) -> CheckResult:
    """The QRF tail register does not load.

    The release reads ``--qrf-tail-concentration-exclusions`` only at its
    terminal gates, after target materialization and the solve
    (``_record_qrf_tail_concentration_gate``). On a certified build it fails
    there: a certain refusal. Under ``--evidence-release`` it refuses only when
    no other terminal failure is on record; otherwise it records a
    ``QRF tail concentration failed: evaluation error ...`` line, and if an
    owner pattern (``evidence_owner``) matches that line the evidence tier
    ships it. Which case applies depends on the solve, so it is AT-RISK.
    """

    if evidence_owner is not None:
        return CheckResult(
            name="qrf_tail_register",
            status="AT_RISK",
            summary=(
                "the per-run register does not load; under --evidence-release "
                "the release refuses it or ships it as an owned failure, "
                "depending on the other terminal failures"
            ),
            at_risks=(
                f"{register_source.get('path')}: {type(error).__name__}: {error}. "
                "The release loads this register at its terminal gates, after the "
                "solve. With no other terminal failure on record it refuses the "
                "run there; otherwise it records a 'QRF tail concentration "
                f"failed: evaluation error' line, which {evidence_owner} owns, "
                "and the evidence tier ships it. Fix the file (a JSON object of "
                "column -> non-empty reason).",
            ),
            details={
                "register": dict(register_source),
                "evidence_owner": evidence_owner,
            },
        )
    return CheckResult(
        name="qrf_tail_register",
        status="FAIL",
        summary=(
            "the per-run register does not load; the release reads it only "
            "after the solve and fails there"
        ),
        failures=(
            f"{register_source.get('path')}: {type(error).__name__}: {error}. The "
            "release loads this register at its terminal gates, after target "
            "materialization and the solve, and refuses the run there: fix the "
            "file (a JSON object of column -> non-empty reason).",
        ),
        details={"register": dict(register_source)},
    )


# ---------------------------------------------------------------------------
# Export input-mass register
# ---------------------------------------------------------------------------


def export_input_mass_check(
    *,
    gate_failures: Sequence[str],
    gate_details: Mapping[str, Any],
    candidate_totals: Mapping[str, float],
    reference_totals: Mapping[str, float],
    candidate_nonzero_shares: Mapping[str, float],
    reviewed_exclusions: Mapping[str, str],
    relative_tolerance: float,
    minimum_reference_total: float,
    margin: float,
    allow_drift: bool,
    reference_is_candidate: bool,
    reference_label: str,
) -> CheckResult:
    """The export input-mass gate and its register, previewed at base weights.

    Pass in the release's ``_export_input_mass_gate`` run with the staged frame
    as candidate. Its failure lines are the authority on which checked columns
    fail at base weights. The register is classified as the gate classifies
    it:

    * ``used``: in the reference above the floor. The gate skips the column.
    * ``below_reference_floor``: in the reference at or under the floor.
    * ``unused``: not in the reference. The gate reports it and never fails on
      it.

    Nonzero mass moves too far under calibration (module docstring) for any
    mass verdict to be certain. The certain refusals are only a checked column
    absent from the staged frame, and one with no nonzero record: no weight
    creates mass from zeros (``us_nonzero_shares``' record share is zero).
    Out-of-band columns, zero totals from cancelling
    signs, and in-band columns within ``margin`` of a band edge are AT-RISK.

    With no ``--export-input-mass-reference-h5`` the release compares the
    export with the staged frame itself. At base weights that drift is
    identically zero, so the check is SKIPPED and reports only the register.
    """

    exclusions = dict(reviewed_exclusions)
    failing = {str(line).split(":", 1)[0] for line in gate_failures}
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    at_risks: list[str] = []

    for name in sorted(exclusions):
        if name not in reference_totals:
            register_class = "unused"
        elif abs(float(reference_totals[name])) <= minimum_reference_total:
            register_class = "below_reference_floor"
        else:
            register_class = "used"
        row: dict[str, Any] = {
            "column": name,
            "in_register": True,
            "register_class": register_class,
        }
        if register_class == "used" and not reference_is_candidate:
            reference = float(reference_totals[name])
            candidate = float(candidate_totals.get(name, 0.0))
            drift = (candidate - reference) / abs(reference)
            row.update(
                {
                    "reference_mass": reference,
                    "mass_at_base_weights": candidate,
                    "drift_at_base_weights": drift,
                    "in_band_without_exclusion_at_base_weights": (
                        name in candidate_totals
                        and candidate != 0.0
                        and abs(drift) <= relative_tolerance
                    ),
                }
            )
        rows.append(row)

    register_rows = list(rows)
    if reference_is_candidate:
        return CheckResult(
            name="export_input_mass",
            status="SKIPPED",
            summary=(
                "no --export-input-mass-reference-h5: the release compares the "
                "export with its own staged frame, whose drift at base weights "
                "is identically zero; the verdict belongs to the solve"
            ),
            rows=tuple(register_rows),
            details={
                "reason": "reference_is_the_staged_frame",
                "reference": reference_label,
                "register_entries": len(exclusions),
            },
        )

    for name in sorted(reference_totals):
        reference = float(reference_totals[name])
        if abs(reference) <= minimum_reference_total or name in exclusions:
            continue
        low = reference - relative_tolerance * abs(reference)
        high = reference + relative_tolerance * abs(reference)
        present = name in candidate_totals
        candidate = float(candidate_totals.get(name, 0.0))
        drift = (candidate - reference) / abs(reference) if present else -1.0
        nonzero_share = candidate_nonzero_shares.get(name)
        gate_fails = name in failing
        if not present or candidate == 0.0 or abs(drift) > relative_tolerance:
            # The release's gate fails exactly these (absent, zero, beyond the
            # tolerance). A disagreement means its arithmetic changed under
            # this reader, which must not pass silently.
            if not gate_fails:
                raise ValueError(
                    f"{name}: the preview reads it as failing at base weights "
                    "but the release's gate passes it; the gate changed."
                )
        elif gate_fails:
            raise ValueError(
                f"{name}: the release's gate fails it at base weights but the "
                "preview reads it in band; the gate changed."
            )
        if not present:
            verdict, status = "absent", "FAIL"
        elif candidate == 0.0 and nonzero_share == 0.0:
            verdict, status = "zero_mass", "FAIL"
        elif candidate == 0.0:
            verdict, status = "zero_mass_by_cancellation", "AT_RISK"
        elif gate_fails:
            verdict, status = "out_of_band", "AT_RISK"
        elif abs(drift) > relative_tolerance - margin:
            verdict, status = "in_band_near_edge", "AT_RISK"
        else:
            verdict, status = "in_band", "PASS"
        if allow_drift and status != "PASS":
            status = "PASS"
            verdict = f"{verdict} (waived by --allow-input-mass-drift)"
        rows.append(
            {
                "column": name,
                "in_register": False,
                "reference_mass": reference,
                "mass_at_base_weights": candidate if present else None,
                "nonzero_share": nonzero_share,
                "band_low": low,
                "band_high": high,
                "drift_at_base_weights": drift,
                "verdict": verdict,
                "status": status,
                # Only a structural refusal (absent, no nonzero record) is
                # certain; calibration moves drift further than the band.
                "certain": status == "FAIL",
            }
        )
        if status == "FAIL":
            failures.append(
                f"{name}: the reference carries {reference:,.0f} but the staged "
                + (
                    "frame lacks the column"
                    if verdict == "absent"
                    else "frame has no nonzero record"
                )
                + "; no weights can create that mass, so the export input-mass "
                "gate refuses it (the microcosm#278 signature)."
            )
        elif status == "AT_RISK":
            reason = {
                "zero_mass_by_cancellation": (
                    "its signed records cancel to zero at base weights"
                ),
                "out_of_band": (
                    f"its mass {candidate:,.0f} at base weights is outside the band "
                    f"[{low:,.0f}, {high:,.0f}] (drift {drift:+.1%})"
                ),
                "in_band_near_edge": (
                    f"its drift {drift:+.1%} at base weights is within "
                    f"{margin:.0%} of the ±{relative_tolerance:.0%} band edge"
                ),
            }[verdict]
            at_risks.append(
                f"{name}: {reason}. Calibration moves this drift (by up to +0.84 on "
                "route A run 310842b986d7), so the export verdict belongs to the "
                "solve: a targeted column is pulled toward its target, an "
                "untargeted one wanders."
            )

    status: PreflightStatus = "FAIL" if failures else "AT_RISK" if at_risks else "PASS"
    summary = (
        f"{len(failures)} certain refusal(s), {len(at_risks)} at risk of "
        f"{sum(1 for r in rows if not r['in_register'])} checked column(s); "
        f"register: {len(exclusions)} entr{'y' if len(exclusions) == 1 else 'ies'} ("
        + ", ".join(
            f"{sum(1 for r in register_rows if r['register_class'] == c)} {c}"
            for c in ("used", "below_reference_floor", "unused")
        )
        + "); an in-band column is not certified at base weights (see "
        "not_previewable)"
    )
    return CheckResult(
        name="export_input_mass",
        status=status,
        summary=summary,
        failures=tuple(failures),
        at_risks=tuple(at_risks),
        rows=tuple(rows),
        details={
            "reference": reference_label,
            "relative_tolerance": float(relative_tolerance),
            "minimum_reference_total": float(minimum_reference_total),
            "margin": float(margin),
            "allow_input_mass_drift": bool(allow_drift),
            "gate_at_base_weights": {
                "failures": list(gate_failures),
                "unused_reviewed_exclusions": list(
                    gate_details.get("unused_reviewed_exclusions", ())
                ),
            },
        },
    )


# ---------------------------------------------------------------------------
# Registers the release grades on the staged frame itself
# ---------------------------------------------------------------------------


def degenerate_input_register_check(
    *,
    gate_passed: bool,
    gate_details: Mapping[str, Any],
    register: Mapping[str, str],
    release_lines: Sequence[str],
) -> CheckResult:
    """The degenerate-input register, as the release grades it.

    The release runs ``_degenerate_input_signal_gate`` on this staged frame
    before the solve and never re-grades it, so the verdict here is the
    release's own and is certain. The rows classify every register entry as
    ``used``, ``stale`` (fails) or ``dormant``.
    """

    used = set(dict(gate_details.get("reviewed_exclusions", {})))
    stale = set(gate_details.get("stale_exclusions", ()))
    dormant = set(gate_details.get("dormant_exclusions", ()))
    rows: list[dict[str, Any]] = [
        {
            "column": name,
            "in_register": True,
            "register_class": (
                "used"
                if name in used
                else "stale"
                if name in stale
                else "dormant"
                if name in dormant
                else "unclassified"
            ),
        }
        for name in sorted(register)
    ]
    for name, value in sorted(
        dict(gate_details.get("default_valued_columns", {})).items()
    ):
        rows.append(
            {
                "column": name,
                "in_register": False,
                "register_class": "degenerate",
                "value": value,
            }
        )
    unclassified = [
        row["column"] for row in rows if row["register_class"] == "unclassified"
    ]
    if unclassified:
        raise ValueError(
            f"Degenerate-input register entries {unclassified} are neither used, "
            "stale nor dormant in the gate details."
        )
    return CheckResult(
        name="degenerate_input_register",
        status="PASS" if gate_passed else "FAIL",
        summary=(
            f"{len(register)} register entries: {len(used)} used, {len(stale)} "
            f"stale, {len(dormant)} dormant; "
            f"{len(dict(gate_details.get('default_valued_columns', {})))} "
            "unexcused degenerate column(s) (graded on the staged frame, as the "
            "release does)"
        ),
        failures=tuple(release_lines),
        rows=tuple(rows),
        details={"columns_checked": gate_details.get("columns_checked")},
    )


def ecps_parity_register_check(
    *,
    gate_passed: bool,
    gate_details: Mapping[str, Any],
    release_lines: Sequence[str],
    waived: bool,
) -> CheckResult:
    """The eCPS parity known-gap register, as the release grades it.

    The release runs ``_ecps_parity_gate`` on this staged frame before the solve,
    so the verdict is certain. ``--allow-ecps-parity-gaps`` unenforces it and
    the rows still record it.
    """

    exempted = set(gate_details.get("exempted", ()))
    stale = set(gate_details.get("stale_exemptions", ()))
    dormant = set(gate_details.get("dormant_exemptions", ()))
    known = dict(gate_details.get("known_gaps", {}))
    rows = tuple(
        {
            "layer": name,
            "in_register": True,
            "register_class": (
                "stale"
                if name in stale
                else "dormant"
                if name in dormant
                else "used"
                if name in exempted
                else "unclassified"
            ),
            **{key: value for key, value in dict(entry).items() if key != "reason"},
        }
        for name, entry in sorted(known.items())
    )
    status: PreflightStatus = "PASS" if gate_passed or waived else "FAIL"
    return CheckResult(
        name="ecps_parity_register",
        status=status,
        summary=(
            f"{len(known)} known-gap entries: {len(stale)} stale, {len(dormant)} "
            f"dormant; {gate_details.get('gaps', 0)} unexempted gap(s)"
            + (
                " (unenforced: --allow-ecps-parity-gaps)"
                if waived and not gate_passed
                else ""
            )
        ),
        failures=() if waived else tuple(release_lines),
        rows=rows,
        details={
            "reference_populated_layers": gate_details.get(
                "reference_populated_layers"
            ),
            "candidate_populated_layers": gate_details.get(
                "candidate_populated_layers"
            ),
            "waived": bool(waived),
            "release_lines": list(release_lines),
        },
    )


def input_coverage_register_check(
    *,
    gate_passed: bool,
    gate_failures: Sequence[str],
    gate_details: Mapping[str, Any],
    support_fixed: bool,
    waived: bool,
    signal_counts: Mapping[str, int] | None = None,
    margins: DryRunMargins | None = None,
) -> CheckResult:
    """``us_release_input_coverage_gate`` on the staged frame.

    Values only, no weights. On the full-pool path the export keeps every
    staged record, so the verdict is the release's. On the L0 path the release
    re-grades the gate on the solve's household selection:

    * A missing or all-default required column stays failed under any
      selection, so those refusals are still certain.
    * A stale reviewed exclusion (the column carries signal) can be cured by a
      selection that drops its carriers, so it is AT-RISK.
    * A passing required column can go degenerate if the selection drops every
      record off the engine default. ``signal_counts``
      (``us_release_input_coverage_signal_counts``) gives those records per
      column; a column whose count does not keep a record under the
      carrier-retention margin is AT-RISK.
    """

    stale = list(gate_details.get("stale_exclusions", ()))
    stale_lines = [line for line in gate_failures if line.startswith("Stale reviewed")]
    other_lines = [
        line for line in gate_failures if not line.startswith("Stale reviewed")
    ]
    failing_columns = set(gate_details.get("missing", ())) | set(
        gate_details.get("degenerate_required", ())
    )
    fragile: dict[str, int] = {}
    if not support_fixed:
        if signal_counts is None or margins is None:
            raise ValueError(
                "The L0 path needs the required columns' signal counts and the "
                "margins to bound the release's re-grade on the selection."
            )
        fragile = {
            column: count
            for column, count in sorted(signal_counts.items())
            if column not in failing_columns and not margins.keeps_a_record(count)
        }
    if waived:
        failures: tuple[str, ...] = ()
        at_risks: tuple[str, ...] = ()
    elif support_fixed:
        failures = tuple(gate_failures)
        at_risks = ()
    else:
        failures = tuple(other_lines)
        at_risks = tuple(
            f"{line} (L0 path: a selection that drops every carrier would cure it)"
            for line in stale_lines
        ) + tuple(
            f"{column}: required input column with {count} record(s) off the "
            "engine default; on the L0 path the release re-grades input coverage "
            "on the solve's household selection, and at carrier retention "
            f"{margins.support_carrier_retention:.2f} a selection may drop them "
            "all, leaving the column degenerate on the export."
            for column, count in fragile.items()
            if margins is not None
        )
    status: PreflightStatus = "FAIL" if failures else "AT_RISK" if at_risks else "PASS"
    return CheckResult(
        name="input_coverage_register",
        status=status,
        summary=(
            f"{len(gate_details.get('missing', ()))} missing and "
            f"{len(gate_details.get('degenerate_required', ()))} degenerate required "
            f"column(s); {len(stale)} stale and "
            f"{len(gate_details.get('dormant_exclusions', ()))} dormant exclusion(s)"
            + (
                " (unenforced: --allow-input-coverage-gaps)"
                if waived and not gate_passed
                else ""
            )
        ),
        failures=failures,
        at_risks=at_risks,
        rows=tuple({"column": name, "register_class": "stale"} for name in stale)
        + tuple(
            {"column": name, "register_class": "missing"}
            for name in gate_details.get("missing", ())
        )
        + tuple(
            {"column": name, "register_class": "degenerate_required"}
            for name in gate_details.get("degenerate_required", ())
        ),
        details={
            "waived": bool(waived),
            "reviewed_exclusions": dict(gate_details.get("reviewed_exclusions", {})),
            "dormant_exclusions": list(gate_details.get("dormant_exclusions", ())),
            "calibration_path": "full_pool" if support_fixed else "l0_selection",
            **({"l0_fragile_required_columns": fragile} if fragile else {}),
        },
    )


def export_signal_regrades_check(
    *,
    health_value_counts: Mapping[str, Sequence[tuple[object, int]]],
    reported_coverage_details: Mapping[str, Any],
    support_fixed: bool,
    margins: DryRunMargins,
) -> CheckResult:
    """The two value-only signal gates the release re-grades on the export.

    The release grades the health-input and reported-coverage-vintage gates on
    the staged frame before the solve (their failures are in
    ``pre_solve_battery``). It then re-grades both on the export frame. Neither
    reads weights, so on the full-pool path the re-grade repeats the staged
    verdict. On the L0 path the solve's household selection can flip a pass:

    * A health-input column goes constant if the selection keeps only one of
      its observed values. It stays nonconstant while its second most common
      value keeps a record under the carrier-retention margin.
    * A reported-coverage vintage group with at least the gate's minimum rows
      fails if the selection drops all reporters of one input. A group below
      the minimum is never enforced, and selection only shrinks groups.

    Args:
        health_value_counts: Each ``US_HEALTH_INPUT_NONCONSTANT_COLUMNS``
            column's observed values and counts (``observed_value_counts``).
        reported_coverage_details: The staged-frame
            ``us_reported_coverage_vintage_signal_gate`` details.
        support_fixed: True on the full-pool path.
        margins: The stated bounds.
    """

    rows: list[dict[str, Any]] = []
    at_risks: list[str] = []
    for column, counts in sorted(health_value_counts.items()):
        runner_up = int(counts[1][1]) if len(counts) > 1 else 0
        stays = support_fixed or margins.keeps_a_record(runner_up)
        rows.append(
            {
                "gate": "health_input_signal",
                "column": column,
                "distinct_observed_values": len(counts),
                "second_value_records": runner_up,
                "certain_on_export": stays,
            }
        )
        if not stays and len(counts) > 1:
            at_risks.append(
                f"health_input_signal/{column}: its second most common value has "
                f"{runner_up} record(s); on the L0 path, at carrier retention "
                f"{margins.support_carrier_retention:.2f}, a selection may keep only "
                "one value, and the release's export re-grade fails a constant "
                "column."
            )
    vintages = dict(reported_coverage_details.get("vintages", {}))
    for label, vintage in sorted(vintages.items()):
        if not vintage.get("enforced"):
            continue
        for column, reporters in sorted(
            dict(vintage.get("reporter_counts", {})).items()
        ):
            stays = support_fixed or margins.keeps_a_record(int(reporters))
            if stays:
                continue
            rows.append(
                {
                    "gate": "reported_coverage_vintage_signal",
                    "vintage": label,
                    "column": column,
                    "reporters": int(reporters),
                    "certain_on_export": False,
                }
            )
            if int(reporters) > 0:
                at_risks.append(
                    f"reported_coverage_vintage_signal/{label}/{column}: "
                    f"{reporters} reporter(s) in an enforced vintage group; on the "
                    "L0 path, at carrier retention "
                    f"{margins.support_carrier_retention:.2f}, a selection may drop "
                    "them all, and the release's export re-grade fails a vintage "
                    "with no reporters."
                )
    status: PreflightStatus = "AT_RISK" if at_risks else "PASS"
    return CheckResult(
        name="export_signal_regrades",
        status=status,
        summary=(
            "the health-input and reported-coverage-vintage gates the release "
            "re-grades on the export "
            + (
                "repeat their staged-frame verdicts (no weights read, every record "
                "kept)"
                if support_fixed
                else "stay passed on any selection inside the margins"
                if not at_risks
                else f"can flip on the L0 selection: {len(at_risks)} fragile signal(s)"
            )
        ),
        at_risks=tuple(at_risks),
        rows=tuple(rows),
        details={
            "calibration_path": "full_pool" if support_fixed else "l0_selection",
            "enforced_vintages": sum(1 for v in vintages.values() if v.get("enforced")),
        },
    )


def bound_zero_support_on_l0(
    check: CheckResult,
    *,
    support_fixed: bool,
    margins: DryRunMargins,
) -> CheckResult:
    """``check_zero_support_preview`` with the L0 re-grade bounded.

    A zero-support target stays zero on any subset of records, so failures
    stay certain. On the L0 path the release grades zero support on the
    selected support, so a supported target whose support records may all be
    dropped under the carrier-retention margin is AT-RISK.
    """

    if support_fixed:
        return check
    fragile = [
        row
        for row in check.rows
        if row.get("checkable")
        and row.get("verdict") == "supported"
        and not margins.keeps_a_record(int(row.get("support_records", 0)))
    ]
    if not fragile:
        return check
    at_risks = tuple(check.at_risks) + tuple(
        f"{row['target']}: {row['support_records']} support record(s); on the L0 "
        "path, at carrier retention "
        f"{margins.support_carrier_retention:.2f}, a selection may drop them all "
        "and leave the target a structural zero."
        for row in fragile
    )
    return CheckResult(
        name=check.name,
        status="FAIL" if check.failures else "AT_RISK",
        summary=check.summary + f"; {len(fragile)} fragile on the L0 selection",
        failures=check.failures,
        at_risks=at_risks,
        rows=check.rows,
        details={**dict(check.details), "l0_fragile_targets": len(fragile)},
    )


def apply_evidence_ownership(
    check: CheckResult,
    *,
    release_lines: Sequence[str],
    owner_of: Callable[[str], str | None],
) -> CheckResult:
    """Grade a failing check's release lines against the evidence tier's owners.

    Under ``--evidence-release`` the release ships a failure whose line an
    owner pattern matches as a known failure, and refuses the run if any line
    is unowned. ``release_lines`` are the lines the release would append for
    this check's certain failures. If every one is owned, the failures become
    AT-RISK ("owned") lines; otherwise the check still certainly refuses.
    """

    if check.status != "FAIL" or not release_lines:
        return check
    owners = [owner_of(line) for line in release_lines]
    if any(owner is None for owner in owners):
        return CheckResult(
            name=check.name,
            status=check.status,
            summary=check.summary,
            failures=check.failures,
            at_risks=check.at_risks,
            rows=check.rows,
            details={
                **dict(check.details),
                "evidence_unowned_release_lines": [
                    line
                    for line, owner in zip(release_lines, owners, strict=True)
                    if owner is None
                ],
            },
        )
    named = ", ".join(dict.fromkeys(owner for owner in owners if owner))
    return CheckResult(
        name=check.name,
        status="AT_RISK",
        summary=check.summary + f" (owned under --evidence-release: {named})",
        at_risks=tuple(
            f"OWNED ({named}; the evidence tier ships it as a known failure): {line}"
            for line in check.failures
        )
        + tuple(check.at_risks),
        rows=check.rows,
        details={**dict(check.details), "evidence_owners": named},
    )


def certain_lines_check(
    name: str,
    *,
    lines: Sequence[str],
    passed_summary: str,
    failed_summary: str,
    certain: bool = True,
    details: Mapping[str, Any] | None = None,
) -> CheckResult:
    """A check whose release failure lines are known exactly at this point.

    ``certain=False`` turns the lines into AT-RISK ones. Use it where a
    selection of records may cure them, such as an SPM unit without a
    classified adult on the L0 path.
    """

    lines = tuple(lines)
    status: PreflightStatus = "PASS" if not lines else "FAIL" if certain else "AT_RISK"
    return CheckResult(
        name=name,
        status=status,
        summary=passed_summary if not lines else failed_summary,
        failures=lines if certain else (),
        at_risks=() if certain else lines,
        details=dict(details or {}),
    )


#: Pre-export gates no base-weight run can grade, with the reason. Listed so
#: the report never implies coverage it lacks.
NOT_PREVIEWABLE_GATES: tuple[tuple[str, str], ...] = (
    (
        "critical_target_fit",
        "compares calibrated final estimates with the critical-target requirements",
    ),
    (
        "soi_table_1_4_national_dollar_fit",
        "compares calibrated final estimates with SOI Table 1.4 dollar targets",
    ),
    ("calibration_loss", "needs the solve's initial and final loss"),
    (
        "dropped_and_skipped_targets",
        "needs target materialization (the release's engine pass over every "
        "target measure)",
    ),
    (
        "ssi_take_up_delivery",
        "compares SSI take-up deliveries at calibrated weights with the SSA "
        "band targets",
    ),
    (
        "final_ssi_take_up_law",
        "re-measures the frozen SSI flags on the export with an engine run",
    ),
    (
        "final_medicaid_diagnostics",
        "re-measures Medicaid take-up on the export with an engine run",
    ),
    (
        "other_health_insurance_export_shares",
        "re-grades weighted positive-share bands at calibrated weights",
    ),
    (
        "exact_k_frozen_register_fit",
        "compares the solved loss with the incumbent's (exact-k builds)",
    ),
    (
        "post_export_scoring_plan",
        "is built from the solved result's in-sample estimates",
    ),
    (
        "reform_coverage_smoke",
        "post-export: scores the written H5 with an engine "
        "(tools/preflight_us_release_gates.py check 4 previews probe support)",
    ),
    (
        "stale_count_calibrated_take_up",
        "post-export: reads take-up diagnostics of the written export",
    ),
    (
        "exact_k_puf_capital_gains_tail",
        "compares the exact-k ladder's selected support with the pool's PUF "
        "capital-gains tail (exact-k builds)",
    ),
    (
        "export_input_mass_nonzero_columns",
        "the export input-mass verdict on a column with nonzero mass: "
        "calibration moved drift by up to +0.84 on route A run 310842b986d7, "
        "past the band's own half-width, so an in-band column at base weights "
        "is not certified (export_input_mass certifies only structural "
        "refusals)",
    ),
)


def not_previewable_check() -> CheckResult:
    """The gates this dry run does not grade, and why."""

    return CheckResult(
        name="not_previewable",
        status="SKIPPED",
        summary=(
            f"{len(NOT_PREVIEWABLE_GATES)} pre- and post-export gates depend on "
            "the solve, target materialization or the written H5"
        ),
        rows=tuple(
            {"gate": gate, "reason": reason} for gate, reason in NOT_PREVIEWABLE_GATES
        ),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReleaseDryRunReport:
    """Every dry-run check, the binding of the run, and the exit code.

    Status and exit code are :class:`PreflightReport`'s: 1 on any FAIL, 2 on
    AT-RISK only, 0 clean.
    """

    checks: tuple[CheckResult, ...]
    inputs: Mapping[str, Any]

    @property
    def _preflight(self) -> PreflightReport:
        return PreflightReport(checks=self.checks, inputs=self.inputs)

    @property
    def status(self) -> PreflightStatus:
        return self._preflight.status

    @property
    def exit_code(self) -> int:
        return self._preflight.exit_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "us_release_dry_run",
            "schema_version": 1,
            "status": self.status,
            "exit_code": self.exit_code,
            "inputs": dict(self.inputs),
            "checks": [check.to_dict() for check in self.checks],
        }

    def human_table(self) -> str:
        badge = {
            "PASS": "PASS   ",
            "FAIL": "FAIL   ",
            "AT_RISK": "AT-RISK",
            "SKIPPED": "SKIPPED",
        }
        lines = [
            "US release dry run (staged frame at base weights; no target "
            "materialization, no solve)"
        ]
        inputs = self.inputs
        binding = [
            ("base H5", _nested(inputs, "base_h5", "path")),
            ("base sha256", _nested(inputs, "base_h5", "sha256")),
            ("build commit", inputs.get("build_commit")),
            ("staged frame sha256", inputs.get("staged_frame_sha256")),
            ("calibration path", inputs.get("calibration_path")),
            (
                "QRF tail register",
                _nested(inputs, "registers", "qrf_tail_concentration", "path"),
            ),
            (
                "target-frame checkpoint",
                _checkpoint_note(inputs.get("target_frame_checkpoint")),
            ),
        ]
        for label, value in binding:
            if value is not None:
                lines.append(f"  {label}: {value}")
        if inputs.get("git_dirty"):
            lines.append(
                "  NOTE: the worktree is dirty; the release itself refuses to "
                "build from it"
            )
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
            "[FAIL=1, AT-RISK=2, clean=0]"
        )
        return "\n".join(lines)


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _checkpoint_note(checkpoint: Any) -> str | None:
    if not isinstance(checkpoint, Mapping):
        return None
    if not checkpoint.get("enabled"):
        return "disabled"
    if not checkpoint.get("exists"):
        return f"{checkpoint.get('path')} (absent: the release materializes targets)"
    if checkpoint.get("identity_matches"):
        return (
            f"{checkpoint.get('path')} (identity matches: the release skips "
            "target materialization)"
        )
    return f"{checkpoint.get('path')} (identity differs: the release rematerializes)"


def pre_solve_refusal_report(
    error: BaseException,
    *,
    inputs: Mapping[str, Any],
) -> ReleaseDryRunReport:
    """The report for a dry run the release refused before the stop point.

    The release refuses too, at the same place and in the same time, so this
    is a certain failure. Nothing past the refusal was graded.
    """

    return ReleaseDryRunReport(
        checks=(
            CheckResult(
                name="pre_solve_refusal",
                status="FAIL",
                summary=(
                    "the release refuses before target materialization; the "
                    "dry run stopped where the release would, and graded "
                    "nothing after it"
                ),
                failures=(f"{type(error).__name__}: {error}",),
                details={"error_type": type(error).__name__},
            ),
        ),
        inputs=inputs,
    )


def post_stop_error_report(
    error: BaseException,
    *,
    inputs: Mapping[str, Any],
) -> ReleaseDryRunReport:
    """The report for a dry run that reached its stop point and then crashed.

    Every check is evaluated behind its own guard, so this means the report
    itself could not be assembled. It is not a release refusal: the dry run
    certifies nothing, and exits 1 so nobody reads the crash as clean.
    """

    return ReleaseDryRunReport(
        checks=(
            CheckResult(
                name="dry_run_evaluation_error",
                status="FAIL",
                summary=(
                    "the dry run reached its stop point but crashed while "
                    "grading; it certifies nothing about the release"
                ),
                failures=(f"{type(error).__name__}: {error}",),
                details={"error_type": type(error).__name__},
            ),
        ),
        inputs=inputs,
    )
