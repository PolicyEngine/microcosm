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
SPM-composition and input-coverage gates. In that run the export frame came
after about an hour of input stages, about 2 h 20 min of target
materialization, and the solve (measured from the run's supervisor series and
materialization-cache timestamps).

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
  households its L0 selection picks, so nonzero shares and carrier counts move
  too. Carriers can only fall, since records are dropped and never added, so a
  thin column stays thin. The support margins bound the rest. They are
  conservative defaults, not measurements.
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
  recomputed at base and at calibrated weights over all 32 columns it checked.
  The calibrated side reproduces the release's recorded
  ``qrf_tail_concentration.json`` exactly for 31 columns. The exception, off by
  0.0013, is a column a release-time stage rewrites. The shifts (calibrated
  minus base) run from -0.033 to +0.299, median +0.105.
* **Tail shares, route A d177 register.** Its six initial-weight versus
  calibrated pairs give +0.06 to +0.29.
* **Export input mass.** Between base and calibrated weights, the drift of the
  20 worst-drift columns moved from -0.30 to +0.84. That is larger than the
  +/-0.5 band itself, so no nonzero-mass verdict is certain at base weights.
  The mass margin only widens the AT-RISK net.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

from microcosm.build.us_runtime.release_gate_preflight import PreflightReport
from microcosm.build.us_runtime.spm_composition import CheckResult, PreflightStatus

__all__ = [
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
    "classify_tail_column",
    "degenerate_input_register_check",
    "ecps_parity_register_check",
    "export_input_mass_check",
    "input_coverage_register_check",
    "not_previewable_check",
    "possible_tail_classes",
    "pre_solve_refusal_report",
    "qrf_tail_register_check",
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

#: L0 path only: how far a column's record nonzero share may move under the
#: solve's household selection. A conservative default, not a measurement.
DEFAULT_SUPPORT_NONZERO_SHARE_MARGIN = 0.02

#: L0 path only: the smallest fraction of a column's base carriers the solve's
#: household selection may keep. A conservative default, not a measurement.
DEFAULT_SUPPORT_CARRIER_RETENTION = 0.25

#: Where the default margins' evidence lives.
MARGIN_EVIDENCE = "experiments/us-release-dry-run-margin-evidence.md"

_MARGIN_NAMES = (
    "tail_share_rise",
    "tail_share_fall",
    "mass_drift",
    "support_nonzero_share",
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
        support_carrier_retention: L0 path only. Smallest kept fraction of a
            column's carriers, in ``(0, 1]``.
    """

    tail_share_rise: float = DEFAULT_TAIL_SHARE_RISE_MARGIN
    tail_share_fall: float = DEFAULT_TAIL_SHARE_FALL_MARGIN
    mass_drift: float = DEFAULT_MASS_DRIFT_MARGIN
    support_nonzero_share: float = DEFAULT_SUPPORT_NONZERO_SHARE_MARGIN
    support_carrier_retention: float = DEFAULT_SUPPORT_CARRIER_RETENTION

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
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tail_share_rise": float(self.tail_share_rise),
            "tail_share_fall": float(self.tail_share_fall),
            "mass_drift": float(self.mass_drift),
            "support_nonzero_share": float(self.support_nonzero_share),
            "support_carrier_retention": float(self.support_carrier_retention),
            "evidence": MARGIN_EVIDENCE,
        }


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
#: * ``ok``: anything else.
TailVerdict = Literal["ok", "used", "unwaived", "waived", "stale", "unused"]

#: Verdicts the release refuses. ``stale`` and ``unused`` ride the
#: register-mismatch line, which is appended whatever
#: ``--allow-qrf-tail-concentration`` says. ``unwaived`` rides the gate's own
#: line, which that flag suppresses (into ``waived``).
FAILING_TAIL_VERDICTS: frozenset[str] = frozenset({"unwaived", "stale", "unused"})

_TAIL_CLASSES: tuple[str, ...] = get_args(TailClass)
_STRUCTURAL_TAIL_CLASSES = frozenset({"not_qrf_output", "absent", "non_numeric"})


def tail_register_verdict(
    tail_class: TailClass,
    *,
    in_register: bool,
    allow_concentration: bool = False,
) -> TailVerdict:
    """The release's verdict on a column in ``tail_class``.

    This mirrors ``_record_qrf_tail_concentration_gate`` in the release tool:
    ``tail_concentration_gate`` plus ``_qrf_tail_register_mismatch``.

    Args:
        tail_class: The column's class under the release's tail gate.
        in_register: Whether the per-run register names the column.
        allow_concentration: ``--allow-qrf-tail-concentration``.

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
        return "waived" if allow_concentration else "unwaived"
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
        return tail_share_classes(
            top_share,
            max_top_share=max_top_share,
            rise_margin=margins.tail_share_rise,
            fall_margin=margins.tail_share_fall,
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
        failing = self.possible_verdicts & FAILING_TAIL_VERDICTS
        if failing and failing == self.possible_verdicts:
            return "FAIL"
        if failing:
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
        ),
        possible_verdicts=frozenset(
            tail_register_verdict(
                tail_class,
                in_register=in_register,
                allow_concentration=allow_concentration,
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
    margin_note = (
        f"calibration may raise a share by up to {margins.tail_share_rise:.2f} or "
        f"lower it by up to {margins.tail_share_fall:.2f}"
    )
    if not support_fixed:
        margin_note += (
            "; on the L0 path the solve also picks the export's records "
            f"(nonzero-share margin {margins.support_nonzero_share:.2f}, carrier "
            f"retention {margins.support_carrier_retention:.2f})"
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
    for column, base_class in classes.items():
        carriers = carrier_counts.get(
            column, thin_counts.get(column, nonzero_records.get(column))
        )
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
        },
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
        + ")"
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
) -> CheckResult:
    """``us_release_input_coverage_gate`` on the staged frame.

    Values only, no weights. On the full-pool path the export keeps every
    staged record, so the verdict is the release's. On the L0 path a missing or
    all-default required column stays failed under any selection of records,
    so those refusals are still certain. A stale reviewed exclusion (the column
    carries signal) can be cured by a selection that drops its carriers, so
    there it is AT-RISK.
    """

    stale = list(gate_details.get("stale_exclusions", ()))
    stale_lines = [line for line in gate_failures if line.startswith("Stale reviewed")]
    other_lines = [
        line for line in gate_failures if not line.startswith("Stale reviewed")
    ]
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
        },
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
