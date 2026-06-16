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
    "macro_realism_gate",
    "nonconstant_columns_gate",
    "nonnegative_columns_gate",
    "parity_gate",
    "support_gate",
    "aggregate_admin_gate",
    "per_family_fit_gate",
    "source_coverage_gate",
    "target_profile_coverage_gate",
    "TargetCoverageRequirement",
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


@dataclass(frozen=True)
class TargetCoverageRequirement:
    """One required concept in a calibration target profile.

    A source-coverage gate can prove the build *loaded* SOI, JCT, or Census
    packages; it cannot prove the compiled target registry actually contains
    the fiscal aggregates that replacement scoring depends on. This
    requirement names a target concept and the target names, name prefixes,
    substrings, measures, or families that satisfy it.
    """

    requirement_id: str
    label: str
    accepted_names: tuple[str, ...] = ()
    accepted_name_prefixes: tuple[str, ...] = ()
    accepted_name_substrings: tuple[str, ...] = ()
    accepted_measures: tuple[str, ...] = ()
    accepted_families: tuple[str, ...] = ()
    required_measures: tuple[str, ...] = ()
    required_metadata: tuple[tuple[str, str], ...] = ()
    min_matches: int = 1
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.requirement_id:
            raise ValueError("TargetCoverageRequirement.requirement_id is required.")
        if not self.label:
            raise ValueError(
                f"TargetCoverageRequirement {self.requirement_id!r}: label is required."
            )
        if self.min_matches < 1:
            raise ValueError(
                f"TargetCoverageRequirement {self.requirement_id!r}: "
                "min_matches must be at least 1."
            )
        if not (
            self.accepted_names
            or self.accepted_name_prefixes
            or self.accepted_name_substrings
            or self.accepted_measures
            or self.accepted_families
        ):
            raise ValueError(
                f"TargetCoverageRequirement {self.requirement_id!r}: at least "
                "one accepted selector is required."
            )
        bad_metadata = [
            (key, value)
            for key, value in self.required_metadata
            if not key or not value
        ]
        if bad_metadata:
            raise ValueError(
                f"TargetCoverageRequirement {self.requirement_id!r}: "
                f"metadata requirements need non-empty keys and values: {bad_metadata}."
            )


@dataclass(frozen=True)
class _TargetInventoryEntry:
    name: str
    measure: str
    family: str
    metadata: Mapping[str, str]


def _target_inventory_entry(target: object) -> _TargetInventoryEntry:
    if isinstance(target, str):
        name = target
        measure = ""
        family = name.split("/", 1)[0] if "/" in name else ""
        return _TargetInventoryEntry(
            name=name, measure=measure, family=family, metadata={}
        )
    metadata: dict[str, str] = {}
    if isinstance(target, Mapping):
        name = str(target.get("name", ""))
        measure = str(target.get("measure") or "")
        family = str(target.get("family") or "")
        metadata = {
            str(key): str(value)
            for key, value in target.items()
            if key not in {"name", "measure", "family", "metadata"}
            and value is not None
        }
        nested_metadata = target.get("metadata")
        if isinstance(nested_metadata, Mapping):
            metadata.update(
                {
                    str(key): str(value)
                    for key, value in nested_metadata.items()
                    if value is not None
                }
            )
    else:
        name = str(getattr(target, "name", "") or getattr(target, "target_name", ""))
        measure = str(getattr(target, "measure", "") or "")
        family = str(getattr(target, "family", "") or "")
        nested_metadata = getattr(target, "metadata", None)
        if isinstance(nested_metadata, Mapping):
            metadata.update(
                {
                    str(key): str(value)
                    for key, value in nested_metadata.items()
                    if value is not None
                }
            )
        for key in (
            "kind",
            "output_variable",
            "matrix_row",
            "neutralized_variable",
        ):
            value = getattr(target, key, None)
            if value is not None:
                metadata[key] = str(value)
    if not family and "/" in name:
        family = name.split("/", 1)[0]
    return _TargetInventoryEntry(
        name=name, measure=measure, family=family, metadata=metadata
    )


def _matches_target_requirement(
    entry: _TargetInventoryEntry,
    requirement: TargetCoverageRequirement,
) -> bool:
    name = entry.name
    lower_name = name.lower()
    selector_matched = (
        name in requirement.accepted_names
        or any(name.startswith(prefix) for prefix in requirement.accepted_name_prefixes)
        or any(
            substring.lower() in lower_name
            for substring in requirement.accepted_name_substrings
        )
        or entry.measure in requirement.accepted_measures
        or entry.family in requirement.accepted_families
    )
    if not selector_matched:
        return False
    if (
        requirement.required_measures
        and entry.measure not in requirement.required_measures
    ):
        return False
    return all(
        entry.metadata.get(key) == value for key, value in requirement.required_metadata
    )


def target_profile_coverage_gate(
    targets: Iterable[object],
    requirements: Iterable[TargetCoverageRequirement],
    *,
    reviewed_exclusions: Mapping[str, str] | None = None,
) -> GateResult:
    """Require a target profile to contain specified fiscal/admin concepts.

    This complements :func:`source_coverage_gate`: source coverage proves the
    administrative package is available, while this gate proves the compiled
    target registry actually activates rows for the concepts that must be hard
    calibration constraints.
    """

    inventory = tuple(_target_inventory_entry(target) for target in targets)
    requirements = tuple(requirements)
    exclusions = _reviewed_exclusion_reasons(reviewed_exclusions)

    failures: list[str] = []
    matches_by_requirement: dict[str, list[str]] = {}
    for requirement in requirements:
        matches = sorted(
            {
                entry.name
                for entry in inventory
                if entry.name and _matches_target_requirement(entry, requirement)
            }
        )
        matches_by_requirement[requirement.requirement_id] = matches
        if len(matches) >= requirement.min_matches:
            continue
        if requirement.requirement_id in exclusions:
            continue
        failures.append(
            f"{requirement.requirement_id}: target profile has {len(matches)} "
            f"match(es), needs {requirement.min_matches} for {requirement.label}."
        )

    requirement_ids = {requirement.requirement_id for requirement in requirements}
    unused_exclusions = sorted(set(exclusions) - requirement_ids)
    if unused_exclusions:
        failures.append(
            f"Reviewed exclusions not in target coverage requirements: "
            f"{unused_exclusions}."
        )

    return GateResult(
        name="target_profile_coverage",
        passed=not failures,
        failures=tuple(failures),
        details={
            "targets_checked": len(inventory),
            "requirements_checked": len(requirements),
            "matches_by_requirement": matches_by_requirement,
            "reviewed_exclusions": {
                requirement.requirement_id: exclusions[requirement.requirement_id]
                for requirement in requirements
                if requirement.requirement_id in exclusions
            },
            "unused_reviewed_exclusions": unused_exclusions,
        },
    )


def macro_realism_gate(
    metrics: Mapping[str, float],
    bands: Mapping[str, tuple[float, float]],
) -> GateResult:
    """Validate macro ratios against broad release-blocking bands.

    Bands are intentionally coarse backstops, not calibration objectives. They
    catch a published file whose aggregate fiscal profile is implausible even
    if the solver reports a low loss on the rows it was given.
    """

    failures: list[str] = []
    checked: dict[str, float] = {}
    for name, band in sorted(bands.items()):
        if len(band) != 2:
            raise ValueError(f"Band for {name!r} must be a (low, high) pair.")
        low, high = map(float, band)
        if low > high:
            raise ValueError(f"Band for {name!r} has low > high: {band!r}.")
        if name not in metrics:
            failures.append(f"{name}: macro metric missing.")
            continue
        value = float(metrics[name])
        checked[name] = value
        if not np.isfinite(value):
            failures.append(f"{name}: macro metric is not finite ({value!r}).")
        elif value < low or value > high:
            failures.append(
                f"{name}: {value:.6g} outside acceptable band [{low:.6g}, {high:.6g}]."
            )

    return GateResult(
        name="macro_realism",
        passed=not failures,
        failures=tuple(failures),
        details={
            "metrics_checked": checked,
            "bands": {name: tuple(map(float, band)) for name, band in bands.items()},
        },
    )


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


def _observed_column_values(values: Iterable[object]) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)
    if arr.dtype.kind == "f":
        return arr[np.isfinite(arr)]
    if arr.dtype.kind in {"b", "i", "u", "S", "U"}:
        return arr
    observed = []
    for value in arr.astype(object):
        if value is None:
            continue
        if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
            continue
        observed.append(value)
    return np.asarray(observed, dtype=object)


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _unique_observed_values(values: np.ndarray) -> np.ndarray:
    try:
        return np.unique(values)
    except TypeError:
        return np.asarray(sorted({repr(value) for value in values}), dtype=object)


def nonconstant_columns_gate(
    column_values: Mapping[str, Iterable[object]],
    required_nonconstant: Iterable[str],
    *,
    reviewed_exclusions: Mapping[str, str] | None = None,
) -> GateResult:
    """Require selected release-input columns to carry distributional signal.

    This catches a different failure mode than an all-zero check: a boolean
    take-up input can be populated but all ``True``, or a plan-choice scalar
    can be populated but exactly ``1.0`` for every record. Those columns pass
    presence and non-zero gates while still masking the intended stochastic or
    imputed source signal.
    """

    exclusions = _reviewed_exclusion_reasons(reviewed_exclusions)
    required = {str(name) for name in required_nonconstant}
    failures: list[str] = []
    unique_counts: dict[str, int] = {}
    observed_counts: dict[str, int] = {}
    unique_examples: dict[str, list[object]] = {}
    constant_values: dict[str, object] = {}

    for name in sorted(required):
        if name in exclusions:
            continue
        if name not in column_values:
            failures.append(f"{name}: required nonconstant column missing.")
            observed_counts[name] = 0
            unique_counts[name] = 0
            unique_examples[name] = []
            continue
        observed = _observed_column_values(column_values[name])
        observed_counts[name] = int(observed.size)
        if observed.size == 0:
            failures.append(f"{name}: required nonconstant column has no values.")
            unique_counts[name] = 0
            unique_examples[name] = []
            continue
        unique = _unique_observed_values(observed)
        unique_counts[name] = int(unique.size)
        examples = [_json_scalar(value) for value in unique[:5]]
        unique_examples[name] = examples
        if unique.size < 2:
            constant = examples[0] if examples else None
            constant_values[name] = constant
            failures.append(
                f"{name}: required nonconstant column has one observed value "
                f"({constant!r}); regenerate upstream source inputs before release."
            )

    unused = sorted(set(exclusions) - required)
    return GateResult(
        name="nonconstant_columns",
        passed=not failures,
        failures=tuple(failures),
        details={
            "columns_checked": len(required),
            "observed_counts": observed_counts,
            "unique_counts": unique_counts,
            "unique_examples": unique_examples,
            "constant_values": constant_values,
            "reviewed_exclusions": {
                name: reason
                for name, reason in sorted(exclusions.items())
                if name in required
            },
            "unused_exclusions": unused,
        },
    )


def _numeric_chunks(values: Iterable[float], chunk_size: int) -> Iterable[np.ndarray]:
    """Yield float chunks without forcing sliceable HDF5 columns into memory."""
    if isinstance(values, np.ndarray):
        yield values.astype(np.float64, copy=False).reshape(-1)
        return
    shape = getattr(values, "shape", None)
    if (
        shape
        and len(shape) >= 1
        and hasattr(values, "__getitem__")
        and not hasattr(values, "iloc")
    ):
        n_rows = int(shape[0])
        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            yield np.asarray(values[start:stop], dtype=np.float64).reshape(-1)
        return
    try:
        yield np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        yield np.fromiter(values, dtype=np.float64)


def nonnegative_columns_gate(
    column_values: Mapping[str, Iterable[float]],
    required_nonnegative: Iterable[str],
    *,
    reviewed_exclusions: Mapping[str, str] | None = None,
    atol: float = 0.0,
    chunk_size: int = 1_000_000,
) -> GateResult:
    """Require physically non-negative exported columns to have no negatives.

    Support guards only ensure imputations stay inside donor ranges. If a
    donor carries sentinel artifacts below zero for a physically non-negative
    quantity, support checks can preserve the bug. This gate is the release
    contract for columns such as loan interest amounts: all finite stored
    values must be at least zero, except for reviewed exclusions with reasons.
    """

    exclusions = _reviewed_exclusion_reasons(reviewed_exclusions)
    required = {str(name) for name in required_nonnegative}
    failures: list[str] = []
    negative_counts: dict[str, int] = {}
    minima: dict[str, float | None] = {}

    if atol < 0:
        raise ValueError(f"atol must be non-negative, got {atol!r}.")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size!r}.")

    for name in sorted(required):
        if name in exclusions:
            continue
        if name not in column_values:
            failures.append(f"{name}: required non-negative column missing.")
            minima[name] = None
            continue
        n_negative = 0
        minimum: float | None = None
        negative_minimum: float | None = None
        for values in _numeric_chunks(column_values[name], chunk_size):
            if values.size == 0:
                continue
            finite_mask = np.isfinite(values)
            if not finite_mask.any():
                continue
            finite = values[finite_mask]
            chunk_min = float(finite.min())
            minimum = chunk_min if minimum is None else min(minimum, chunk_min)
            negative = finite < -atol
            if negative.any():
                n_negative += int(negative.sum())
                chunk_negative_min = float(finite[negative].min())
                negative_minimum = (
                    chunk_negative_min
                    if negative_minimum is None
                    else min(negative_minimum, chunk_negative_min)
                )
        minima[name] = minimum
        negative_counts[name] = n_negative
        if n_negative:
            failures.append(
                f"{name}: {n_negative} finite value(s) below zero "
                f"(minimum {negative_minimum:.6g}); floor or fix "
                "the source before publishing."
            )

    unused_exclusions = sorted(set(exclusions) - required)
    if unused_exclusions:
        failures.append(
            f"Reviewed exclusions not in non-negative requirements: "
            f"{unused_exclusions}."
        )

    return GateResult(
        name="nonnegative_columns",
        passed=not failures,
        failures=tuple(failures),
        details={
            "columns_required": len(required),
            "columns_checked": sorted(required & set(column_values)),
            "negative_counts": negative_counts,
            "minima": minima,
            "reviewed_exclusions": {
                name: reason
                for name, reason in sorted(exclusions.items())
                if name in required
            },
            "unused_reviewed_exclusions": unused_exclusions,
            "atol": float(atol),
            "chunk_size": int(chunk_size),
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
    reviewed_exclusions: Mapping[str, str] | None = None,
    name: str = "source_coverage",
) -> GateResult:
    """Gate source-family coverage for a release target profile.

    Hard-target source package aliases must either appear in the active target
    inventory or have an explicit reviewed exclusion. Validation-only families
    can appear in diagnostics, but fail the gate if activated as hard targets.
    Source gaps are reported in details without failing; they are facts about
    source availability, not evidence that the build covered the family.

    ``coverage_entries`` intentionally accepts either dict-like entries or the
    ``SourceCoverageEntry`` dataclass from ``populace.build.us.source_coverage``
    so callers can also pass a live Ledger coverage contract.
    """
    active_aliases = set(active_target_aliases)
    active_families = set(active_target_families)
    if not name:
        raise ValueError("source coverage gate name must be non-empty.")

    reviewed_exclusions = reviewed_exclusions or {}
    if not isinstance(reviewed_exclusions, Mapping):
        raise TypeError("reviewed_exclusions must be a mapping from alias to reason.")
    exclusion_reasons: dict[str, str | object] = {
        str(alias): reason for alias, reason in reviewed_exclusions.items()
    }
    valid_exclusion_reasons: dict[str, str] = {}

    failures: list[str] = []
    missing_hard_targets: list[str] = []
    reviewed: dict[str, str] = {}
    validation_misuse: list[str] = []
    source_gaps: dict[str, tuple[str, ...]] = {}
    hard_target_families: dict[str, dict[str, object]] = {}
    validation_only_families: dict[str, dict[str, object]] = {}
    source_gap_families: dict[str, dict[str, object]] = {}

    for alias, reason in sorted(exclusion_reasons.items()):
        if isinstance(reason, str) and reason.strip():
            valid_exclusion_reasons[alias] = reason
        else:
            failures.append(
                f"{alias}: reviewed exclusion requires a non-empty string reason."
            )

    for entry in coverage_entries:
        family = str(_coverage_field(entry, "family_id", ""))
        role = str(_coverage_field(entry, "role", ""))
        label = str(_coverage_field(entry, "label", ""))
        aliases = tuple(
            str(a) for a in (_coverage_field(entry, "package_aliases", ()) or ())
        )
        if role == "hard_target":
            covered_aliases: list[str] = []
            missing_aliases: list[str] = []
            reviewed_aliases: dict[str, str] = {}
            for alias in aliases:
                if alias in active_aliases:
                    covered_aliases.append(alias)
                    continue
                if alias in valid_exclusion_reasons:
                    reviewed[alias] = valid_exclusion_reasons[alias]
                    reviewed_aliases[alias] = valid_exclusion_reasons[alias]
                    continue
                missing_hard_targets.append(alias)
                missing_aliases.append(alias)
                failures.append(
                    f"{family}/{alias}: hard-target source alias is not active "
                    "and has no reviewed exclusion."
                )
            hard_target_families[family] = {
                "label": label,
                "package_aliases": aliases,
                "covered_package_aliases": tuple(sorted(covered_aliases)),
                "missing_package_aliases": tuple(sorted(missing_aliases)),
                "reviewed_exclusions": dict(sorted(reviewed_aliases.items())),
            }
        elif role == "validation_only":
            activated = family in active_families or any(
                alias in active_aliases for alias in aliases
            )
            validation_only_families[family] = {
                "label": label,
                "package_aliases": aliases,
                "activated_as_hard_target": activated,
            }
            if family in active_families or any(
                alias in active_aliases for alias in aliases
            ):
                validation_misuse.append(family)
                failures.append(
                    f"{family}: validation-only source family activated as a hard target."
                )
        elif role == "source_gap":
            missing_sources = tuple(
                str(item)
                for item in (
                    _coverage_field(entry, "missing_source_packages", ()) or ()
                )
            )
            source_gaps[family] = missing_sources
            source_gap_families[family] = {
                "label": label,
                "missing_source_packages": missing_sources,
            }

    unused_exclusions = sorted(set(valid_exclusion_reasons) - set(reviewed))
    if unused_exclusions:
        failures.append(
            f"Reviewed exclusions not in coverage contract: {unused_exclusions}."
        )

    covered_hard_targets = sorted(
        alias
        for summary in hard_target_families.values()
        for alias in summary["covered_package_aliases"]
    )
    reviewed_hard_targets = sorted(reviewed)
    validation_family_count = len(validation_only_families)
    source_gap_package_count = sum(len(packages) for packages in source_gaps.values())

    return GateResult(
        name=name,
        passed=not failures,
        failures=tuple(failures),
        details={
            "active_target_aliases": sorted(active_aliases),
            "active_target_families": sorted(active_families),
            "hard_target_families": hard_target_families,
            "covered_hard_targets": covered_hard_targets,
            "missing_hard_targets": sorted(missing_hard_targets),
            "reviewed_exclusions": reviewed,
            "validation_only_families": validation_only_families,
            "validation_only_activated": sorted(validation_misuse),
            "source_gaps": source_gaps,
            "source_gap_families": source_gap_families,
            "coverage_summary": {
                "hard_target": {
                    "families": len(hard_target_families),
                    "package_aliases": (
                        len(covered_hard_targets)
                        + len(missing_hard_targets)
                        + len(reviewed_hard_targets)
                    ),
                    "covered_package_aliases": len(covered_hard_targets),
                    "missing_package_aliases": len(missing_hard_targets),
                    "reviewed_excluded_package_aliases": len(reviewed_hard_targets),
                },
                "validation_only": {
                    "families": validation_family_count,
                    "activated_families": len(validation_misuse),
                },
                "source_gap": {
                    "families": len(source_gaps),
                    "missing_source_packages": source_gap_package_count,
                },
            },
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
    hard_within: float | None = 0.25,
    min_hard_family_share: float = 0.5,
    min_family_size: int = 5,
) -> GateResult:
    """Calibration fit per source family: no family hides in the average.

    Args:
        target_names: Compiled target names (``"family/..."`` prefixes; an
            unprefixed name lands in family ``"unspecified"``).
        relative_errors: Per-target relative errors, aligned to the names.
        within: Diagnostic hit threshold (default: within 10%). This is
            reported, but is not itself a hard release threshold.
        min_family_share: Diagnostic floor for the within-threshold share.
        hard_within: Wider hit threshold used for hard failures. Set to
            ``None`` to make this gate report-only.
        min_hard_family_share: Minimum hard-within-threshold share each family
            must reach when ``hard_within`` is set.
        min_family_size: Families smaller than this are reported but not
            gated (a 2-target family's share is too noisy to gate on).

    Returns:
        Pass iff every family of at least ``min_family_size`` targets hits
        ``min_hard_family_share`` at ``hard_within``. Details carry the
        diagnostic and hard-fit shares either way.
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
    hard_shares = (
        {
            family: float(np.mean([e <= hard_within for e in errs]))
            for family, errs in sorted(by_family.items())
        }
        if hard_within is not None
        else {}
    )
    failures = (
        [
            f"{family}: only {share:.1%} of {len(by_family[family])} targets "
            f"within broad {hard_within:.0%} fit "
            f"(floor {min_hard_family_share:.0%})."
            for family, share in hard_shares.items()
            if len(by_family[family]) >= min_family_size
            and share < min_hard_family_share
        ]
        if hard_within is not None
        else []
    )
    return GateResult(
        name="per_family_fit",
        passed=not failures,
        failures=tuple(failures),
        details={
            "within": within,
            "min_family_share": min_family_share,
            "hard_within": hard_within,
            "min_hard_family_share": min_hard_family_share,
            "family_within_shares": shares,
            "family_hard_within_shares": hard_shares,
            "family_sizes": {f: len(e) for f, e in sorted(by_family.items())},
        },
    )
