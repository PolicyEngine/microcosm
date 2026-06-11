"""Dataset acceptance gates: what a build must prove before it ships.

Each gate is a pure function from evidence to a :class:`GateResult` — no gate
mutates anything, every failure names the exact variable/target involved, and
a :class:`GateReport` aggregates the suite into one publishable verdict.

The four gates encode the build lessons of 2026:

- :func:`parity_gate` — the incumbent-replacement contract: every variable
  layer the reference populates, the candidate populates. An all-zero layer
  the engine knows about silently masks the engine's own formulas.
- :func:`support_gate` — tail-faithful imputation must stay inside the
  donor's own realized support; a draw outside it is fabrication.
- :func:`aggregate_admin_gate` — weighted aggregates against administrative
  anchors (from the target registry), **with signs checked**. This is the
  permanent version of the check that would have caught short-term capital
  gains at −$3.9T and investment-interest at $33.5T.
- :func:`per_family_fit_gate` — calibration fit reported per source family,
  so a collapsed family cannot hide inside a good global average.

Scoring uses :func:`relative_error_loss` — the calibrator's own objective —
so there is no calibrator-vs-scorer objective mismatch: what the solver
minimizes is exactly what the gates and scorers measure.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from populace.calibrate.registry import TargetSpec
from populace.calibrate.solve import relative_error_loss

__all__ = [
    "GateResult",
    "GateReport",
    "parity_gate",
    "support_gate",
    "aggregate_admin_gate",
    "per_family_fit_gate",
    "relative_error_loss",
]


@dataclass(frozen=True)
class GateResult:
    """One gate's verdict.

    Attributes:
        name: The gate's name (``"parity"``, ``"support"``, ...).
        passed: Whether the gate passed.
        failures: One human-readable line per failure, naming the variable
            or target and the observed-vs-expected values. Empty iff passed
            (a gate that fails must say why; a gate that passes must not
            invent caveats).
        details: Gate-specific numbers worth recording in a release manifest
            (counts, shares, worst offenders).
    """

    name: str
    passed: bool
    failures: tuple[str, ...] = ()
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.passed and self.failures:
            raise ValueError(
                f"Gate {self.name!r} cannot pass with failures recorded."
            )
        if not self.passed and not self.failures:
            raise ValueError(
                f"Gate {self.name!r} cannot fail without naming a failure."
            )


@dataclass(frozen=True)
class GateReport:
    """The full acceptance suite: every gate, one verdict.

    Attributes:
        results: The individual gate results, in run order.
    """

    results: tuple[GateResult, ...]

    @property
    def passed(self) -> bool:
        """True iff every gate passed."""
        return all(result.passed for result in self.results)

    @property
    def failures(self) -> tuple[str, ...]:
        """Every failure line across the suite, gate-prefixed."""
        return tuple(
            f"[{result.name}] {line}"
            for result in self.results
            for line in result.failures
        )

    def to_manifest(self) -> dict[str, object]:
        """A JSON-ready summary for the release manifest."""
        return {
            "passed": self.passed,
            "gates": {
                result.name: {
                    "passed": result.passed,
                    "failures": list(result.failures),
                    "details": dict(result.details),
                }
                for result in self.results
            },
        }


def parity_gate(
    candidate_nonzero: Mapping[str, float],
    reference_nonzero: Mapping[str, float],
    *,
    known_gaps: Iterable[str] = (),
) -> GateResult:
    """Every layer the reference populates, the candidate populates.

    Args:
        candidate_nonzero: Variable -> share of records with a non-zero
            value in the candidate dataset.
        reference_nonzero: The same for the reference (incumbent) dataset.
            Variables only the candidate carries are ignored (extra layers
            are not a parity failure).
        known_gaps: Variables exempted *by name* — each must be a documented
            decision, and exemptions are recorded in the details.

    Returns:
        Pass iff no reference-populated variable is candidate-empty.
    """
    exempt = set(known_gaps)
    failures = []
    for name, ref_share in sorted(reference_nonzero.items()):
        if ref_share <= 0.0 or name in exempt:
            continue
        cand_share = candidate_nonzero.get(name, 0.0)
        if cand_share <= 0.0:
            failures.append(
                f"{name}: reference populates {ref_share:.1%} of records, "
                "candidate is all-zero (the layer would mask engine "
                "formulas or drop the variable entirely)."
            )
    populated = sum(
        1 for share in reference_nonzero.values() if share > 0.0
    )
    return GateResult(
        name="parity",
        passed=not failures,
        failures=tuple(failures),
        details={
            "reference_populated_layers": populated,
            "gaps": len(failures),
            "exempted": sorted(exempt),
        },
    )


def support_gate(
    values: Mapping[str, np.ndarray],
    donor_ranges: Mapping[str, tuple[float, float]],
    *,
    atol: float = 1e-9,
) -> GateResult:
    """Every imputed value lies inside its donor's realized range.

    Args:
        values: Imputed column -> the candidate's values for it.
        donor_ranges: Column -> ``(min, max)`` realized in the *donor's own*
            data (never a third dataset's range — anchoring to the incumbent
            is exactly the contamination the v3 build removed).
        atol: Absolute slack for float round-trips.

    Returns:
        Pass iff no column escapes its donor support. Columns in ``values``
        without a declared range fail — an imputation without a recorded
        support is unauditable.
    """
    failures = []
    checked = 0
    for name, column in sorted(values.items()):
        if name not in donor_ranges:
            failures.append(
                f"{name}: no donor range declared — record the donor's "
                "realized (min, max) at imputation time."
            )
            continue
        lo, hi = donor_ranges[name]
        arr = np.asarray(column, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        checked += 1
        if finite.size == 0:
            continue
        below, above = float(finite.min()), float(finite.max())
        if below < lo - atol or above > hi + atol:
            failures.append(
                f"{name}: values span [{below:.6g}, {above:.6g}] outside "
                f"donor support [{lo:.6g}, {hi:.6g}]."
            )
    return GateResult(
        name="support",
        passed=not failures,
        failures=tuple(failures),
        details={"columns_checked": checked},
    )


def aggregate_admin_gate(
    aggregates: Mapping[str, float],
    anchors: Iterable[TargetSpec],
    *,
    default_rtol: float = 0.5,
) -> GateResult:
    """Weighted aggregates against administrative anchors, signs checked.

    The permanent STCG/investment-interest catcher: every anchor is a
    registry :class:`~populace.calibrate.registry.TargetSpec` (value +
    citation), and the candidate's weighted aggregate must (1) carry the
    anchor's sign and (2) land within a relative tolerance of it.

    Args:
        aggregates: Anchor name -> the candidate's weighted aggregate.
        anchors: Registry specs to check against. A spec whose name is
            missing from ``aggregates`` fails — if a build declares an
            anchor, it must measure it.
        default_rtol: Relative tolerance when a spec declares no tolerance.
            Deliberately loose by default: this gate catches order-of-
            magnitude and sign disasters, not calibration slack.

    Returns:
        Pass iff every anchor is measured, sign-consistent, and within
        tolerance.
    """
    failures = []
    checked = 0
    for spec in anchors:
        if spec.name not in aggregates:
            failures.append(
                f"{spec.name}: declared as an admin anchor but the build "
                "did not measure it."
            )
            continue
        achieved = float(aggregates[spec.name])
        checked += 1
        if spec.value != 0 and achieved != 0 and (
            np.sign(achieved) != np.sign(spec.value)
        ):
            failures.append(
                f"{spec.name}: sign flip — achieved {achieved:.4g} vs "
                f"admin {spec.value:.4g} ({spec.source})."
            )
            continue
        scale = max(abs(spec.value), 1.0)
        rtol = (
            spec.tolerance / scale
            if spec.tolerance is not None
            else default_rtol
        )
        miss = abs(achieved - spec.value) / scale
        if miss > rtol:
            failures.append(
                f"{spec.name}: achieved {achieved:.4g} vs admin "
                f"{spec.value:.4g} — relative miss {miss:.2f} exceeds "
                f"{rtol:.2f} ({spec.source})."
            )
    return GateResult(
        name="aggregate_vs_admin",
        passed=not failures,
        failures=tuple(failures),
        details={"anchors_checked": checked},
    )


def per_family_fit_gate(
    target_names: Iterable[str],
    relative_errors: Iterable[float],
    *,
    within: float = 0.10,
    min_family_share: float = 0.5,
    min_family_size: int = 5,
) -> GateResult:
    """Calibration fit per source family: no family hides in the average.

    Args:
        target_names: Compiled target names (``"family/..."`` prefixes; an
            unprefixed name lands in family ``"unspecified"``).
        relative_errors: Per-target relative errors, aligned to the names.
        within: The hit threshold (default: within 10%).
        min_family_share: Minimum within-threshold share each family must
            reach.
        min_family_size: Families smaller than this are reported but not
            gated (a 2-target family's share is too noisy to gate on).

    Returns:
        Pass iff every family of at least ``min_family_size`` targets hits
        ``min_family_share``. Details carry every family's share either way.
    """
    names = list(target_names)
    errors = list(relative_errors)
    if len(names) != len(errors):
        raise ValueError(
            f"target_names and relative_errors must align: {len(names)} "
            f"names vs {len(errors)} errors."
        )
    by_family: dict[str, list[float]] = {}
    for name, error in zip(names, errors, strict=True):
        family = name.split("/", 1)[0] if "/" in name else "unspecified"
        by_family.setdefault(family, []).append(abs(float(error)))
    shares = {
        family: float(np.mean([e <= within for e in errs]))
        for family, errs in sorted(by_family.items())
    }
    failures = [
        f"{family}: only {share:.1%} of {len(by_family[family])} targets "
        f"within {within:.0%} (floor {min_family_share:.0%})."
        for family, share in shares.items()
        if len(by_family[family]) >= min_family_size
        and share < min_family_share
    ]
    return GateResult(
        name="per_family_fit",
        passed=not failures,
        failures=tuple(failures),
        details={
            "within": within,
            "family_within_shares": shares,
            "family_sizes": {f: len(e) for f, e in sorted(by_family.items())},
        },
    )
