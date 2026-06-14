"""Dataset acceptance gates: what a build must prove before it ships.

Each gate is a pure function from evidence to a :class:`GateResult` — no gate
mutates anything, every failure names the exact variable/target involved, and
a :class:`GateReport` aggregates the suite into one publishable verdict.

The gates encode the build lessons of 2026:

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
- :func:`source_coverage_gate` — hard-target source families must be active or
  explicitly excluded, while validation-only families must stay out of hard
  calibration.
- :func:`enum_domain_gate` — enum-typed engine inputs must carry engine enum
  member names, not raw source-system codes.
- :func:`export_surface_gate` and :func:`target_surface_gate` — replacement
  builds can prove they cover a reference artifact's export variables and
  calibration targets, e.g. UK Populace against eFRS.

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
    "enum_domain_gate",
    "export_surface_gate",
    "formula_owned_export_gate",
    "exported_nonzero_gate",
    "parity_gate",
    "support_gate",
    "aggregate_admin_gate",
    "per_family_fit_gate",
    "source_coverage_gate",
    "relative_error_loss",
    "target_surface_gate",
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
            raise ValueError(f"Gate {self.name!r} cannot pass with failures recorded.")
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


def exported_nonzero_gate(
    column_shares: Mapping[str, float],
    *,
    exemptions: Mapping[str, str] | None = None,
) -> GateResult:
    """Every exported column carries signal: no all-zero stored layers.

    Stronger than parity (which only checks layers the reference populates):
    if the artifact stores a column at all, it must be populated. An
    all-zero stored column is either a pipeline bug (real values lost on
    the way to export — the v3 head-carry incident) or dead scaffolding
    that masks the engine's own defaults/formulas; the fix is to populate
    it or remove it upstream, never to ship zeros.

    Args:
        column_shares: Stored column -> share of records with a non-zero
            (or True) value, over every exported column.
        exemptions: Column -> REASON for columns allowed to ship all-zero.
            Every exemption is a named, documented decision and is recorded
            in the gate details; an empty reason is refused.

    Returns:
        Pass iff every non-exempt column has a positive non-zero share.

    Raises:
        ValueError: If an exemption has an empty reason — an undocumented
            exemption is just a silent zero with extra steps.
    """
    exemptions = dict(exemptions or {})
    for column, reason in exemptions.items():
        if not reason:
            raise ValueError(
                f"Exemption for {column!r} needs a reason; an undocumented "
                "exemption is just a silent zero with extra steps."
            )
    failures = []
    for name, share in sorted(column_shares.items()):
        if share > 0.0 or name in exemptions:
            continue
        failures.append(
            f"{name}: stored but all-zero — populate it or remove it upstream "
            "(zeros mask engine defaults/formulas)."
        )
    unused = sorted(set(exemptions) - set(column_shares))
    return GateResult(
        name="exported_nonzero",
        passed=not failures,
        failures=tuple(failures),
        details={
            "columns_checked": len(column_shares),
            "exempted": {
                name: reason
                for name, reason in sorted(exemptions.items())
                if name in column_shares
            },
            "unused_exemptions": unused,
        },
    )


def formula_owned_export_gate(
    exported_columns: Iterable[str],
    formula_owned_columns: Iterable[str],
    *,
    structural_columns: Iterable[str] = (),
) -> GateResult:
    """Formula-owned engine outputs must not be persisted as inputs.

    A PolicyEngine-native HDF5 file turns every persisted variable column into
    a simulation input. Persisting a formula-owned variable such as ``ssi``
    therefore pins the baseline value and masks reforms; the artifact must
    arrive at export without it, so the engine computes it. Entity ids and
    memberships can be exempted via ``structural_columns`` because those are
    reconstruction scaffolding, not policy inputs.

    Args:
        exported_columns: Columns the artifact will persist.
        formula_owned_columns: Variables owned by engine formulas.
        structural_columns: Non-input structural columns allowed through even
            when their names overlap the engine's variable registry.

    Returns:
        Pass iff no non-structural exported column is formula-owned.
    """
    exported = set(exported_columns)
    structural = set(structural_columns)
    formula_owned = set(formula_owned_columns)
    offenders = sorted((exported & formula_owned) - structural)
    return GateResult(
        name="formula_owned_export",
        passed=not offenders,
        failures=tuple(
            f"{name}: formula-owned engine output exported as an input; "
            "remove it upstream before export."
            for name in offenders
        ),
        details={
            "columns_checked": len(exported),
            "formula_owned_columns": len(formula_owned),
            "structural_exemptions": sorted(structural & exported & formula_owned),
            "offenders": offenders,
        },
    )


def _reviewed_exclusion_reasons(
    reviewed_exclusions: Mapping[str, str] | None,
) -> dict[str, str]:
    if reviewed_exclusions is None:
        return {}
    if not isinstance(reviewed_exclusions, Mapping):
        raise TypeError("Reviewed exclusions must be a mapping from name to reason.")
    exclusions = {
        str(name): str(reason) for name, reason in reviewed_exclusions.items()
    }
    undocumented = sorted(name for name, reason in exclusions.items() if not reason)
    if undocumented:
        raise ValueError(
            f"Reviewed exclusions need reasons; missing reasons for {undocumented}."
        )
    return exclusions


def export_surface_gate(
    candidate_columns: Iterable[str],
    reference_columns: Iterable[str],
    *,
    candidate_name: str = "candidate",
    reference_name: str = "reference",
    allowed_extra_columns: Iterable[str] = (),
    reviewed_exclusions: Mapping[str, str] | None = None,
) -> GateResult:
    """Require a published export surface to match a reference dataset.

    This is stricter than :func:`parity_gate`: parity checks whether populated
    reference layers are also populated, while this gate checks the exported
    variable *surface* itself. It is intended for live release blocking where a
    country has a known incumbent-compatible artifact, such as UK Populace
    matching eFRS exported variables. Extra columns are refused unless the
    build declares them as structural/compatibility additions; missing
    reference columns require a named reviewed exclusion.
    """
    candidate = {str(name) for name in candidate_columns}
    reference = {str(name) for name in reference_columns}
    allowed_extras = {str(name) for name in allowed_extra_columns}
    exclusions = _reviewed_exclusion_reasons(reviewed_exclusions)

    missing = sorted((reference - candidate) - set(exclusions))
    unexpected = sorted((candidate - reference) - allowed_extras)
    failures: list[str] = []
    if missing:
        failures.append(
            f"{candidate_name}: missing {len(missing)} {reference_name} export "
            f"column(s): {missing[:20]}."
        )
    if unexpected:
        failures.append(
            f"{candidate_name}: exports {len(unexpected)} column(s) outside the "
            f"{reference_name} surface without an allow-list entry: "
            f"{unexpected[:20]}."
        )

    return GateResult(
        name="export_surface",
        passed=not failures,
        failures=tuple(failures),
        details={
            "candidate_name": candidate_name,
            "reference_name": reference_name,
            "candidate_columns": len(candidate),
            "reference_columns": len(reference),
            "allowed_extra_columns": sorted(candidate & allowed_extras),
            "unused_allowed_extra_columns": sorted(allowed_extras - candidate),
            "reviewed_exclusions": {
                name: reason
                for name, reason in sorted(exclusions.items())
                if name in reference
            },
            "unused_reviewed_exclusions": sorted(set(exclusions) - reference),
            "missing_reference_columns": missing,
            "unexpected_candidate_columns": unexpected,
        },
    )


def target_surface_gate(
    candidate_targets: Iterable[str],
    reference_targets: Iterable[str],
    *,
    candidate_name: str = "candidate",
    reference_name: str = "reference",
    reviewed_exclusions: Mapping[str, str] | None = None,
) -> GateResult:
    """Require calibration targets to cover a reference target surface.

    A replacement population should not publish with a narrower calibration
    surface than the dataset it is replacing. Candidate extras are allowed:
    newer source-backed targets are a strict improvement as long as every
    reference target is covered or has a reviewed exclusion.
    """
    candidate = {str(name) for name in candidate_targets}
    reference = {str(name) for name in reference_targets}
    exclusions = _reviewed_exclusion_reasons(reviewed_exclusions)

    missing = sorted((reference - candidate) - set(exclusions))
    extras = sorted(candidate - reference)
    failures: list[str] = []
    if missing:
        failures.append(
            f"{candidate_name}: missing {len(missing)} {reference_name} "
            f"calibration target(s): {missing[:20]}."
        )

    return GateResult(
        name="target_surface",
        passed=not failures,
        failures=tuple(failures),
        details={
            "candidate_name": candidate_name,
            "reference_name": reference_name,
            "candidate_targets": len(candidate),
            "reference_targets": len(reference),
            "extra_candidate_targets": extras,
            "reviewed_exclusions": {
                name: reason
                for name, reason in sorted(exclusions.items())
                if name in reference
            },
            "unused_reviewed_exclusions": sorted(set(exclusions) - reference),
            "missing_reference_targets": missing,
        },
    )


def _enum_member_name(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value)


def _enum_domain_names(domain: Iterable[object] | object) -> tuple[str, ...]:
    members = getattr(domain, "__members__", None)
    if isinstance(members, Mapping):
        return tuple(str(name) for name in members)
    return tuple(_enum_member_name(value) for value in domain)  # type: ignore[arg-type]


def enum_domain_gate(
    column_values: Mapping[str, Iterable[object]],
    enum_domains: Mapping[str, Iterable[object] | object],
) -> GateResult:
    """Validate exported enum inputs against their engine enum domains.

    A non-zero raw source code can pass parity and nonzero checks while still
    being impossible for the rules engine to interpret. This gate operates on
    exported columns whose corresponding engine variable declares enum
    ``possible_values`` and requires stored values to be enum member names
    such as ``"WHITE"`` rather than source codes such as ``"10"``.

    Args:
        column_values: Exported enum column -> stored values.
        enum_domains: Exported enum column -> valid enum members, member
            names, or an enum class exposing ``__members__``.

    Returns:
        Pass iff every provided enum column's non-missing values are inside
        its declared domain. Missing values are treated as invalid because a
        present enum input column should be fully interpretable by the engine;
        omit the column to let the engine default it.
    """
    failures: list[str] = []
    invalid_counts: dict[str, int] = {}
    invalid_examples: dict[str, list[str]] = {}
    allowed_values: dict[str, list[str]] = {}
    columns_checked = 0

    for column, values in sorted(column_values.items()):
        if column not in enum_domains:
            continue
        allowed = set(_enum_domain_names(enum_domains[column]))
        allowed_values[column] = sorted(allowed)
        columns_checked += 1
        invalid: list[str] = []
        total = 0
        for value in values:
            total += 1
            if value is None or (
                isinstance(value, (float, np.floating)) and np.isnan(value)
            ):
                invalid.append("<missing>")
                continue
            name = _enum_member_name(value)
            if name not in allowed:
                invalid.append(name)
        if not invalid:
            continue
        examples = sorted(set(invalid))[:8]
        invalid_counts[column] = len(invalid)
        invalid_examples[column] = examples
        failures.append(
            f"{column}: {len(invalid)}/{total} value(s) outside enum domain; "
            f"invalid examples {examples}; allowed values {sorted(allowed)[:8]}."
        )

    return GateResult(
        name="enum_domain",
        passed=not failures,
        failures=tuple(failures),
        details={
            "columns_checked": columns_checked,
            "invalid_counts": invalid_counts,
            "invalid_examples": invalid_examples,
            "allowed_values": allowed_values,
        },
    )


def _coverage_field(entry: object, name: str, default: object = None) -> object:
    if isinstance(entry, Mapping):
        return entry.get(name, default)
    return getattr(entry, name, default)


def source_coverage_gate(
    coverage_entries: Iterable[object],
    *,
    active_target_aliases: Iterable[str] = (),
    active_target_families: Iterable[str] = (),
    reviewed_exclusions: Mapping[str, str] | Iterable[str] = (),
) -> GateResult:
    """Gate source-family coverage for a release target profile.

    Hard-target source package aliases must either appear in the active target
    inventory or have an explicit reviewed exclusion. Validation-only families
    can appear in diagnostics, but fail the gate if activated as hard targets.
    Source gaps are reported in details without failing; they are facts about
    source availability, not evidence that the build covered the family.

    ``coverage_entries`` intentionally accepts either dict-like entries or the
    ``SourceCoverageEntry`` dataclass from ``populace.build.us.source_coverage``
    so callers can also pass a live Arch coverage contract.
    """
    active_aliases = set(active_target_aliases)
    active_families = set(active_target_families)
    if isinstance(reviewed_exclusions, Mapping):
        exclusion_reasons = {
            str(alias): str(reason) for alias, reason in reviewed_exclusions.items()
        }
    else:
        exclusion_reasons = {
            str(alias): "reviewed exclusion" for alias in reviewed_exclusions
        }

    failures: list[str] = []
    missing_hard_targets: list[str] = []
    reviewed: dict[str, str] = {}
    validation_misuse: list[str] = []
    source_gaps: dict[str, tuple[str, ...]] = {}

    for entry in coverage_entries:
        family = str(_coverage_field(entry, "family_id", ""))
        role = str(_coverage_field(entry, "role", ""))
        aliases = tuple(
            str(a) for a in (_coverage_field(entry, "package_aliases", ()) or ())
        )
        if role == "hard_target":
            for alias in aliases:
                if alias in active_aliases:
                    continue
                if alias in exclusion_reasons:
                    reviewed[alias] = exclusion_reasons[alias]
                    continue
                missing_hard_targets.append(alias)
                failures.append(
                    f"{family}/{alias}: hard-target source alias is not active "
                    "and has no reviewed exclusion."
                )
        elif role == "validation_only":
            if family in active_families or any(
                alias in active_aliases for alias in aliases
            ):
                validation_misuse.append(family)
                failures.append(
                    f"{family}: validation-only source family activated as a hard target."
                )
        elif role == "source_gap":
            source_gaps[family] = tuple(
                str(item)
                for item in (
                    _coverage_field(entry, "missing_source_packages", ()) or ()
                )
            )

    unused_exclusions = sorted(set(exclusion_reasons) - set(reviewed))
    if unused_exclusions:
        failures.append(
            f"Reviewed exclusions not in coverage contract: {unused_exclusions}."
        )

    return GateResult(
        name="source_coverage",
        passed=not failures,
        failures=tuple(failures),
        details={
            "active_target_aliases": sorted(active_aliases),
            "active_target_families": sorted(active_families),
            "missing_hard_targets": sorted(missing_hard_targets),
            "reviewed_exclusions": reviewed,
            "validation_only_activated": sorted(validation_misuse),
            "source_gaps": source_gaps,
        },
    )


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
    populated = sum(1 for share in reference_nonzero.values() if share > 0.0)
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
        if (
            spec.value != 0
            and achieved != 0
            and (np.sign(achieved) != np.sign(spec.value))
        ):
            failures.append(
                f"{spec.name}: sign flip — achieved {achieved:.4g} vs "
                f"admin {spec.value:.4g} ({spec.source})."
            )
            continue
        scale = max(abs(spec.value), 1.0)
        rtol = spec.tolerance / scale if spec.tolerance is not None else default_rtol
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
        if len(by_family[family]) >= min_family_size and share < min_family_share
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
