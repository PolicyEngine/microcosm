"""Country-neutral gate bindings for transport countries' gate batteries.

``gates.battery@1`` (:mod:`~microcosm.build.transport.gate_kernels`) takes its
binding registry as a constructor argument; :data:`TRANSPORT_GATE_REGISTRY`
is that registry for a donor-based (transport) country. It is
:data:`~microcosm.build.gate_battery.DEFAULT_REGISTRY` plus one
:class:`~microcosm.build.gate_battery.FunctionBinding` per gate below. A
binding adapts evidence the transport graph already produces to a gate
comparison. The comparisons stay in :mod:`microcosm.build.gates` where one
exists; the two weight-concentration gates and the calibration-reference
coverage check have none there and are defined here.

Everything a binding reads arrives one of three ways, because the gate
kernel offers no spec object and binds no data a module reads from disk:

- **Thresholds and declared surfaces** come only from the entry's
  ``parameters`` in the country's ``gates.json``. No tunable threshold has a
  default here: an entry that omits one fails closed when the gate runs, and
  a composer that calls :func:`validate_required_gate_parameters` refuses
  such a manifest before any gate runs (the kernel itself runs only the
  battery's ``validate_gate_parameters``). The shared comparisons keep their
  own fixed semantics: non-negative means at least zero, and
  ``aggregate_admin_gate`` measures a miss relative to ``max(|value|, 1)``.
- **Artifacts** arrive under the aliases below, decoded by type
  (``gate_kernels.EVIDENCE_DECODERS``). A composer wires each alias to the
  output of the same name.
- **The frame** is the node's rebuilt population (``entities`` with
  ``weight_entity``) plus that entity's typed weights. The frame gates check
  the columns the node slices, so a node gating an export must slice what
  ``export.prepare@1`` slices; their details list the columns they read.

The evidence each bound gate reads:

- ``per_family_fit``: ``diagnostics`` (``diagnostics.calibration@1``).
- ``aggregate_admin``: ``surface`` (``targets.compile@1``) and
  ``diagnostics``.
- ``calibration_reference_coverage``: ``surface`` and ``problem``
  (``targets.problem@1``).
- ``target_profile_coverage``: ``problem``.
- ``nonnegative_columns``: the frame; ``columns`` from ``gates.json``.
- ``exported_nonzero``: the frame.
- ``formula_owned_export``: the frame; ``formula_owned_columns`` from
  ``gates.json``.
- ``weight_ess`` and ``weight_ratio``: the frame's one weighted entity.

Of the gates New Zealand declares, three have no binding here, because no
code in Microcosm yet produces their evidence for a transport build:
``support`` (realized donor-support bounds of transported columns),
``release_input_coverage`` (donor-artifact authentication and the per-module
Axiom input closure) and ``macro_realism`` (destination national-accounts
metrics and reviewed bands). A country declares them ``not_applicable`` with
the missing producer as its reason; an entry left applicable resolves to the
battery's ``evidence_absent`` gap. ``weights_audit`` keeps its
``DEFAULT_REGISTRY`` binding, but no transport kernel emits its
``fit_weight_records`` evidence yet, so it resolves to a named
``evidence_absent``.

Every binding is a frozen dataclass over top-level functions, so
``gates.battery@1`` can describe it exactly
(:mod:`~microcosm.build.transport.binding_identity`). Failure lines and
details are built in a fixed order, never from unordered set iteration.
"""

from __future__ import annotations

import inspect
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.country_spec import GatesManifest
from microcosm.build.gate_battery import (
    DEFAULT_REGISTRY,
    EvidenceContext,
    FunctionBinding,
    GateBinding,
)
from microcosm.build.gates import (
    GateResult,
    TargetCoverageRequirement,
    aggregate_admin_gate,
    exported_nonzero_gate,
    formula_owned_export_gate,
    nonnegative_columns_gate,
    per_family_fit_gate,
    target_profile_coverage_gate,
)
from microcosm.build.plan import _nonzero_share
from microcosm.calibrate.solve import effective_sample_size
from microcosm.frame import Frame

__all__ = [
    "DIAGNOSTICS",
    "PROBLEM",
    "SURFACE",
    "TRANSPORT_GATE_REGISTRY",
    "required_gate_parameters",
    "validate_required_gate_parameters",
    "weight_ess_gate",
    "weight_ratio_gate",
    "weight_summary",
]

#: Alias of the compiled target surface (``targets.compile@1`` ``surface``).
SURFACE = "surface"
#: Alias of the ordered calibration problem (``targets.problem@1`` ``problem``).
PROBLEM = "problem"
#: Alias of one calibration's schema-8 diagnostics
#: (``diagnostics.calibration@1`` ``diagnostics``).
DIAGNOSTICS = "diagnostics"


# ---------------------------------------------------------------------------
# Parameter and evidence readers
# ---------------------------------------------------------------------------


def _number(name: str, value: object) -> float:
    """A finite real number; a JSON boolean is refused, never read as 0 or 1."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number, got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return number


def _strings(name: str, value: object) -> tuple[str, ...]:
    """A non-empty sequence of distinct, non-empty strings."""

    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a list of strings, got {value!r}.")
    items = tuple(value)
    if not items:
        raise ValueError(f"{name} must name at least one item.")
    bad = [item for item in items if not isinstance(item, str) or not item]
    if bad:
        raise ValueError(f"{name} must hold non-empty strings, got {bad!r}.")
    if len(set(items)) != len(items):
        raise ValueError(f"{name} repeats {_repeated(items)}.")
    return items


def _optional_strings(name: str, value: object) -> tuple[str, ...] | None:
    return None if value is None else _strings(name, value)


def _diagnostics_rows(diagnostics: object) -> tuple[Mapping[str, Any], ...]:
    """The target rows of a decoded schema-8 calibration diagnostics model."""

    if not isinstance(diagnostics, Mapping):
        raise TypeError(
            "diagnostics must be a decoded calibration diagnostics mapping, got "
            f"{type(diagnostics).__name__}."
        )
    rows = diagnostics.get("targets")
    if not isinstance(rows, list | tuple) or not rows:
        raise ValueError("diagnostics must carry a non-empty list of target rows.")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"diagnostics target row {index} is not a mapping.")
    return tuple(rows)


def _row_family(row: Mapping[str, Any]) -> str:
    """The registry family of a diagnostics row, usable as a family label."""

    registry = row.get("registry")
    family = registry.get("family") if isinstance(registry, Mapping) else None
    if not isinstance(family, str) or not family or "/" in family:
        raise ValueError(
            f"diagnostics target {row.get('name')!r} has no usable registry "
            f"family ({family!r}); a target without one cannot be gated per "
            "family."
        )
    return family


def _row_name(row: Mapping[str, Any]) -> str:
    name = row.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"diagnostics target row has no name ({name!r}).")
    return name


def _row_number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"diagnostics target {row.get('name')!r} has no finite {key} ({value!r})."
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"diagnostics target {row.get('name')!r} has no finite {key} ({value!r})."
        )
    return number


def _missing_selection(
    gate: str, kind: str, declared: tuple[str, ...], found: Iterable[str]
) -> list[str]:
    present = set(found)
    return [
        f"{label}: declared {kind} has no calibrated target in the {gate} evidence."
        for label in declared
        if label not in present
    ]


def _with_failures(
    result: GateResult,
    *,
    failures: Sequence[str],
    details: Mapping[str, object],
    legacy: str | None = None,
    name: str | None = None,
) -> GateResult:
    """``result`` with extra failures (first) and details appended.

    A shared comparison that mints a legacy name is re-minted under ``name``
    only when it returns exactly ``legacy``; any other name passes through,
    so the battery's name check still fails a mismatch closed.
    """

    renamed = name if legacy is not None and result.name == legacy else result.name
    return GateResult(
        name=renamed,
        passed=result.passed and not failures,
        failures=(*failures, *result.failures),
        details={**dict(result.details), **dict(details)},
    )


def _repeated(items: Sequence[str]) -> list[str]:
    return sorted(item for item, count in Counter(items).items() if count > 1)


def _frame_columns(frame: Frame) -> dict[str, pd.Series]:
    """Every exported column: each entity table's columns, then the weights.

    Column names are unique across a frame's entity tables (a ``Frame``
    invariant). A weighted entity's typed weights stand in for its
    ``{entity}_weight`` column, which an export writes from them.
    """

    columns: dict[str, pd.Series] = {}
    for entity in frame.entities:
        table = frame.table(entity)
        for column in table.columns:
            columns[str(column)] = table[column]
    for entity in frame.weighted_entities:
        columns[f"{entity}_weight"] = pd.Series(
            frame.weights_for(entity).values, name=f"{entity}_weight"
        )
    return columns


def _structural_columns(frame: Frame) -> tuple[str, ...]:
    """Entity ids and the person table's memberships of the frame's groups.

    ``context_frame`` keeps memberships of groups outside the node's entities
    as ordinary person columns; those are measured like any other column.
    """

    schema = frame.schema
    return (
        schema.person_id_column,
        *(schema.membership_column(group) for group in schema.group_entities),
        *(schema.id_column(group) for group in schema.group_entities),
    )


def _weighted_entity(frame: Frame) -> tuple[str, np.ndarray]:
    weighted = frame.weighted_entities
    if len(weighted) != 1:
        raise ValueError(
            "The weight gates read one weighted entity; the frame carries "
            f"{list(weighted)}."
        )
    entity = weighted[0]
    return entity, frame.weights_for(entity).values


# ---------------------------------------------------------------------------
# Weight concentration (no shared comparison in microcosm.build.gates)
# ---------------------------------------------------------------------------


def weight_summary(weights: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """The concentration summary of one shipped weight vector.

    The ESS fraction divides the Kish effective sample size by every record,
    zero-weight rows included; the median is over positive weights only, so
    the max-to-median ratio does not depend on how many dead rows ship. An
    all-zero vector is reportable: its median and ratio are ``None``.
    """

    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Weights must be a non-empty vector.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Weights must be finite and non-negative.")
    positive = values[values > 0]
    ess = float(effective_sample_size(values))
    median = float(np.median(positive)) if positive.size else None
    maximum = float(values.max())
    return {
        "n_records": int(values.size),
        "positive_weight_records": int(positive.size),
        "zero_weight_records": int(values.size - positive.size),
        "total_weight": float(values.sum()),
        "effective_sample_size": ess,
        "ess_fraction": ess / values.size,
        "median_positive_weight": median,
        "max_weight": maximum,
        "max_to_median_positive_weight": (None if median is None else maximum / median),
    }


def weight_ess_gate(
    weights: Sequence[float] | np.ndarray, *, minimum_ess_fraction: float
) -> GateResult:
    """Require the shipped weights to retain an effective-sample-size floor."""

    minimum = _number("minimum_ess_fraction", minimum_ess_fraction)
    if not 0 < minimum <= 1:
        raise ValueError("minimum_ess_fraction must lie in (0, 1].")
    summary = weight_summary(weights)
    fraction = float(summary["ess_fraction"])
    failures = (
        (f"ESS fraction {fraction:.6g} is below the reviewed minimum {minimum:.6g}.",)
        if not fraction >= minimum
        else ()
    )
    return GateResult(
        name="weight_ess",
        passed=not failures,
        failures=failures,
        details={**summary, "minimum_ess_fraction": minimum},
    )


def weight_ratio_gate(
    weights: Sequence[float] | np.ndarray, *, maximum_max_to_median_ratio: float
) -> GateResult:
    """Backstop a shipped-weight max to positive-median concentration blowout."""

    maximum = _number("maximum_max_to_median_ratio", maximum_max_to_median_ratio)
    if not maximum > 0:
        raise ValueError("maximum_max_to_median_ratio must be strictly positive.")
    summary = weight_summary(weights)
    ratio = summary["max_to_median_positive_weight"]
    if ratio is None:
        failures: tuple[str, ...] = (
            "Max/positive-median weight ratio is undefined because no weight "
            "is positive.",
        )
    elif not ratio <= maximum:
        failures = (
            f"Max/positive-median weight ratio {ratio!r} exceeds the reviewed "
            f"maximum {maximum!r}.",
        )
    else:
        failures = ()
    return GateResult(
        name="weight_ratio",
        passed=not failures,
        failures=failures,
        details={**summary, "maximum_max_to_median_ratio": maximum},
    )


# ---------------------------------------------------------------------------
# Evaluators: one per bound gate (keyword-only; thresholds have no default)
# ---------------------------------------------------------------------------


def _per_family_fit(
    *,
    diagnostics: object,
    within: float,
    min_family_share: float,
    hard_within: float,
    min_hard_family_share: float,
    min_family_size: int,
    families: Sequence[str] | None = None,
) -> GateResult:
    """Per-family calibration fit over one calibration's diagnostics rows.

    Every threshold of :func:`~microcosm.build.gates.per_family_fit_gate` is
    required. ``hard_within`` must be a number: the shared gate treats
    ``None`` as report-only, which belongs in a ``diagnostic`` entry, not in
    a binding that can pass a release-blocking one. ``families`` restricts
    the gate to those registry families, so a country can declare a tighter
    entry for one family; each declared family must have at least one target.
    """

    if isinstance(min_family_size, bool) or not isinstance(min_family_size, int):
        raise TypeError(f"min_family_size must be an integer, got {min_family_size!r}.")
    if min_family_size < 0:
        raise ValueError("min_family_size must be non-negative.")
    thresholds = {
        "within": _number("within", within),
        "min_family_share": _number("min_family_share", min_family_share),
        "hard_within": _number("hard_within", hard_within),
        "min_hard_family_share": _number(
            "min_hard_family_share", min_hard_family_share
        ),
        "min_family_size": min_family_size,
    }
    for key in ("within", "hard_within"):
        if thresholds[key] < 0:
            raise ValueError(f"{key} must be non-negative.")
    for key in ("min_family_share", "min_hard_family_share"):
        if not 0 <= thresholds[key] <= 1:
            raise ValueError(f"{key} must lie in [0, 1].")
    declared = _optional_strings("families", families)
    rows = [
        (_row_family(row), _row_name(row), _row_number(row, "relative_error"))
        for row in _diagnostics_rows(diagnostics)
    ]
    repeated = _repeated([name for _, name, _ in rows])
    if repeated:
        raise ValueError(f"diagnostics repeat targets {repeated}.")
    if declared is not None:
        rows = [row for row in rows if row[0] in declared]
    missing = (
        []
        if declared is None
        else _missing_selection(
            "per_family_fit", "family", declared, (row[0] for row in rows)
        )
    )
    result = per_family_fit_gate(
        [f"{family}/{name}" for family, name, _ in rows],
        [error for _, _, error in rows],
        **thresholds,
    )
    return _with_failures(
        result,
        failures=missing,
        details={
            "min_family_size": min_family_size,
            "families": None if declared is None else list(declared),
        },
    )


def _aggregate_admin(
    *,
    surface: object,
    diagnostics: object,
    default_rtol: float,
    families: Sequence[str] | None = None,
    geography_levels: Sequence[str] | None = None,
) -> GateResult:
    """Calibrated aggregates against the compiled surface's anchors, signed.

    Anchors are the surface's compiled target specs (each with its source
    and any fact-specific tolerance); the achieved aggregate is the
    diagnostics' final estimate of the same target. The diagnostics must
    record this surface (``build.surface_sha256``, which
    ``diagnostics.calibration@1`` writes), so another calibration's
    estimates cannot be graded against it. ``default_rtol`` applies where a
    spec declares no tolerance. ``families`` and ``geography_levels``
    restrict the anchors; each declared value must select at least one.
    """

    rtol = _number("default_rtol", default_rtol)
    if rtol < 0:
        raise ValueError("default_rtol must be non-negative.")
    declared_families = _optional_strings("families", families)
    declared_levels = _optional_strings("geography_levels", geography_levels)
    specs = tuple(surface.registry.specs)
    if declared_families is not None:
        specs = tuple(spec for spec in specs if spec.family in declared_families)
    levels: list[str] = []
    if declared_levels is not None:
        selected = []
        for spec in specs:
            if spec.hierarchy is None:
                raise ValueError(
                    f"surface target {spec.name!r} has no hierarchy, so its "
                    "geography level cannot be selected."
                )
            levels.append(spec.hierarchy.geography.level)
            if spec.hierarchy.geography.level in declared_levels:
                selected.append(spec)
        specs = tuple(selected)
    missing = []
    build = diagnostics.get("build") if isinstance(diagnostics, Mapping) else None
    recorded = build.get("surface_sha256") if isinstance(build, Mapping) else None
    if recorded != surface.sha256:
        missing.append(
            f"The diagnostics record surface {recorded!r}, not {surface.sha256!r}."
        )
    if declared_families is not None:
        missing += _missing_selection(
            "aggregate_admin",
            "family",
            declared_families,
            (spec.family for spec in specs),
        )
    if declared_levels is not None:
        missing += _missing_selection(
            "aggregate_admin", "geography level", declared_levels, levels
        )
    achieved: dict[tuple[object, object], Mapping[str, Any]] = {}
    for row in _diagnostics_rows(diagnostics):
        key = (row.get("target_name"), row.get("period"))
        if key in achieved:
            raise ValueError(f"diagnostics repeat target {key[0]!r} @ {key[1]!r}.")
        achieved[key] = row
    aggregates = {
        spec.name: _row_number(achieved[(spec.name, spec.period)], "final_estimate")
        for spec in specs
        if (spec.name, spec.period) in achieved
    }
    if not specs:
        missing.append("aggregate_admin: no compiled anchor was selected.")
    result = aggregate_admin_gate(aggregates, specs, default_rtol=rtol)
    return _with_failures(
        result,
        legacy="aggregate_vs_admin",
        name="aggregate_admin",
        failures=missing,
        details={
            "default_rtol": rtol,
            "anchors": [spec.name for spec in specs],
            "families": None if declared_families is None else list(declared_families),
            "geography_levels": None
            if declared_levels is None
            else list(declared_levels),
        },
    )


def _calibration_reference_coverage(*, surface: object, problem: object) -> GateResult:
    """Every activated reference resolved to a target that entered the matrix.

    The surface traces each compiled target to the one executable reference
    that produced it; the problem's rows are the calibration matrix. The gate
    passes iff at least one reference is activated, the activated
    references, the resolved targets and the matrix rows are the same set
    with no repeats and no skipped target, and the problem was compiled from
    this surface.
    """

    trace = tuple(surface.trace)
    activated = [str(row.get("reference")) for row in trace]
    resolved = [spec.name for spec in surface.registry.specs]
    resolved_rows = [spec.to_target().row_name for spec in surface.registry.specs]
    matrix_rows = [str(name) for name in problem.problem.names]
    failures: list[str] = []
    if not activated:
        failures.append("No calibration reference is activated.")
    for label, items in (
        ("activated references", activated),
        ("resolved targets", resolved_rows),
        ("matrix rows", matrix_rows),
    ):
        repeated = _repeated(items)
        if repeated:
            failures.append(f"The {label} repeat {repeated}.")
    if sorted(activated) != sorted(resolved):
        failures.append(
            "Activated references and resolved targets differ: "
            f"{sorted(set(activated) ^ set(resolved))}."
        )
    not_in_matrix = sorted(set(resolved_rows) - set(matrix_rows))
    not_activated = sorted(set(matrix_rows) - set(resolved_rows))
    if not_in_matrix:
        failures.append(f"Activated targets missing from the matrix: {not_in_matrix}.")
    if not_activated:
        failures.append(f"Matrix rows with no activated reference: {not_activated}.")
    skipped = sorted(item.target.row_name for item in problem.problem.skipped)
    if skipped:
        failures.append(f"The problem skipped targets {skipped}.")
    bound = problem.bindings.get("surface_sha256")
    if bound != surface.sha256:
        failures.append(
            f"The problem was compiled from surface {bound!r}, not {surface.sha256!r}."
        )
    return GateResult(
        name="calibration_reference_coverage",
        passed=not failures,
        failures=tuple(failures),
        details={
            "activated": len(activated),
            "resolved": len(resolved_rows),
            "matrix": len(matrix_rows),
            "not_in_matrix": not_in_matrix,
            "not_activated": not_activated,
            "skipped": skipped,
            "surface_sha256": surface.sha256,
            "problem_surface_sha256": bound,
        },
    )


def _target_profile_coverage(
    *,
    problem: object,
    required_families: Sequence[str],
    reviewed_exclusions: Mapping[str, str] | None = None,
) -> GateResult:
    """The calibration matrix keeps at least one target of every declared family.

    Each declared family is one requirement matched on the family the
    problem records for its rows (``target_metadata``). A reviewed exclusion
    names a family and its reason.
    """

    declared = _strings("required_families", required_families)
    names = tuple(problem.problem.names)
    metadata = tuple(problem.target_metadata)
    if len(metadata) != len(names):
        raise ValueError("Problem target metadata does not cover its matrix rows.")
    targets = [
        {"name": str(name), "family": str(row.get("family") or "")}
        for name, row in zip(names, metadata, strict=True)
    ]
    requirements = [
        TargetCoverageRequirement(
            requirement_id=family,
            label=f"calibration family {family!r}",
            accepted_families=(family,),
        )
        for family in declared
    ]
    return target_profile_coverage_gate(
        targets,
        requirements,
        reviewed_exclusions=None
        if reviewed_exclusions is None
        else dict(reviewed_exclusions),
    )


def _nonnegative_columns(
    *,
    frame: Frame,
    columns: Sequence[str],
    reviewed_exclusions: Mapping[str, str] | None = None,
) -> GateResult:
    """The declared physically non-negative columns hold finite values >= 0.

    ``columns`` is the country's declared list (its source manifest's
    non-negative outputs); a declared column the frame lacks fails. The
    shared gate skips non-finite values, so this binding fails them itself:
    ``-inf`` is negative, and a missing value cannot be certified
    non-negative (exclude the column with a reason if that is intended).
    """

    required = _strings("columns", columns)
    exclusions = None if reviewed_exclusions is None else dict(reviewed_exclusions)
    values = _frame_columns(frame)
    result = nonnegative_columns_gate(values, required, reviewed_exclusions=exclusions)
    nonfinite: dict[str, int] = {}
    for column in required:
        if column in (exclusions or {}) or column not in values:
            continue
        array = values[column].to_numpy(dtype=np.float64, na_value=np.nan)
        count = int((~np.isfinite(array)).sum())
        if count:
            nonfinite[column] = count
    return _with_failures(
        result,
        failures=[
            f"{column}: {count} non-finite value(s); a declared non-negative "
            "column must hold finite values."
            for column, count in nonfinite.items()
        ],
        details={"nonfinite_counts": nonfinite},
    )


def _exported_nonzero(
    *, frame: Frame, exemptions: Mapping[str, str] | None = None
) -> GateResult:
    """No exported column is all zero, except reviewed exemptions.

    Every column of the frame is measured with the build plan's share
    definition (non-zero numbers, ``True`` booleans, non-empty text), entity
    ids and memberships aside.
    """

    structural = set(_structural_columns(frame))
    shares = {
        column: _nonzero_share(values)
        for column, values in _frame_columns(frame).items()
        if column not in structural
    }
    result = exported_nonzero_gate(
        shares, exemptions=None if exemptions is None else dict(exemptions)
    )
    return _with_failures(result, failures=(), details={"columns": sorted(shares)})


def _formula_owned_export(
    *, frame: Frame, formula_owned_columns: Sequence[str]
) -> GateResult:
    """No rule output declared formula-owned is exported as an input column."""

    owned = _strings("formula_owned_columns", formula_owned_columns)
    exported = list(_frame_columns(frame))
    result = formula_owned_export_gate(
        exported, owned, structural_columns=_structural_columns(frame)
    )
    return _with_failures(
        result, failures=(), details={"exported_columns": sorted(exported)}
    )


def _weight_ess(*, frame: Frame, minimum_ess_fraction: float) -> GateResult:
    entity, weights = _weighted_entity(frame)
    result = weight_ess_gate(weights, minimum_ess_fraction=minimum_ess_fraction)
    return _with_failures(result, failures=(), details={"weight_entity": entity})


def _weight_ratio(*, frame: Frame, maximum_max_to_median_ratio: float) -> GateResult:
    entity, weights = _weighted_entity(frame)
    result = weight_ratio_gate(
        weights, maximum_max_to_median_ratio=maximum_max_to_median_ratio
    )
    return _with_failures(result, failures=(), details={"weight_entity": entity})


# ---------------------------------------------------------------------------
# Evidence payloads (the report's evidence_sha256 lines)
# ---------------------------------------------------------------------------


def _diagnostics_evidence(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> object:
    return {"diagnostics": context.artifacts[DIAGNOSTICS]}


def _surface_diagnostics_evidence(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> object:
    return {
        "surface_sha256": context.artifacts[SURFACE].sha256,
        "diagnostics": context.artifacts[DIAGNOSTICS],
    }


def _surface_problem_evidence(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> object:
    return {
        "surface_sha256": context.artifacts[SURFACE].sha256,
        "problem_sha256": context.artifacts[PROBLEM].sha256,
    }


def _problem_evidence(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> object:
    return {"problem_sha256": context.artifacts[PROBLEM].sha256}


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def _parameters(gate: object) -> frozenset[str]:
    """Every keyword the evaluator takes that no artifact or frame supplies."""

    supplied = {"frame", SURFACE, PROBLEM, DIAGNOSTICS}
    return frozenset(
        name for name in inspect.signature(gate).parameters if name not in supplied
    )


#: The country-neutral bindings, merged over ``DEFAULT_REGISTRY``.
TRANSPORT_GATE_REGISTRY: Mapping[str, GateBinding] = MappingProxyType(
    {
        **DEFAULT_REGISTRY,
        "per_family_fit": FunctionBinding(
            name="per_family_fit",
            gate=_per_family_fit,
            parameter_keys=_parameters(_per_family_fit),
            artifact_arguments=MappingProxyType({"diagnostics": DIAGNOSTICS}),
            evidence=_diagnostics_evidence,
        ),
        "aggregate_admin": FunctionBinding(
            name="aggregate_admin",
            gate=_aggregate_admin,
            parameter_keys=_parameters(_aggregate_admin),
            artifact_arguments=MappingProxyType(
                {"surface": SURFACE, "diagnostics": DIAGNOSTICS}
            ),
            evidence=_surface_diagnostics_evidence,
        ),
        "calibration_reference_coverage": FunctionBinding(
            name="calibration_reference_coverage",
            gate=_calibration_reference_coverage,
            parameter_keys=_parameters(_calibration_reference_coverage),
            artifact_arguments=MappingProxyType(
                {"surface": SURFACE, "problem": PROBLEM}
            ),
            evidence=_surface_problem_evidence,
        ),
        "target_profile_coverage": FunctionBinding(
            name="target_profile_coverage",
            gate=_target_profile_coverage,
            parameter_keys=_parameters(_target_profile_coverage),
            artifact_arguments=MappingProxyType({"problem": PROBLEM}),
            evidence=_problem_evidence,
        ),
        "nonnegative_columns": FunctionBinding(
            name="nonnegative_columns",
            gate=_nonnegative_columns,
            parameter_keys=_parameters(_nonnegative_columns),
            frame_argument="frame",
        ),
        "exported_nonzero": FunctionBinding(
            name="exported_nonzero",
            gate=_exported_nonzero,
            parameter_keys=_parameters(_exported_nonzero),
            frame_argument="frame",
        ),
        "formula_owned_export": FunctionBinding(
            name="formula_owned_export",
            gate=_formula_owned_export,
            parameter_keys=_parameters(_formula_owned_export),
            frame_argument="frame",
        ),
        "weight_ess": FunctionBinding(
            name="weight_ess",
            gate=_weight_ess,
            parameter_keys=_parameters(_weight_ess),
            frame_argument="frame",
        ),
        "weight_ratio": FunctionBinding(
            name="weight_ratio",
            gate=_weight_ratio,
            parameter_keys=_parameters(_weight_ratio),
            frame_argument="frame",
        ),
    }
)


def required_gate_parameters(binding: GateBinding) -> frozenset[str]:
    """The ``gates.json`` parameters a binding cannot evaluate without.

    For a :class:`~microcosm.build.gate_battery.FunctionBinding`, these are
    its gate function's parameters that have no default and that neither an
    artifact nor the frame supplies. Any other binding declares none.
    """

    if not isinstance(binding, FunctionBinding):
        return frozenset()
    supplied = set(binding.artifact_arguments)
    if binding.frame_argument is not None:
        supplied.add(binding.frame_argument)
    return frozenset(
        parameter.name
        for parameter in inspect.signature(binding.gate).parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.name not in supplied
    )


def validate_required_gate_parameters(
    gates: GatesManifest,
    registry: Mapping[str, GateBinding] = TRANSPORT_GATE_REGISTRY,
) -> None:
    """Refuse an applicable entry that omits a parameter its binding requires.

    The battery's :func:`~microcosm.build.gate_battery.validate_gate_parameters`
    refuses parameters a binding cannot route; this refuses the converse, a
    threshold the entry never declares. Without it the omission surfaces
    only when the gate runs, as a failed result. Entries declared
    ``not_applicable`` and gates with no binding are skipped: neither runs.

    Raises:
        ValueError: Naming the first offending entry and its missing keys.
    """

    for entry in gates.gates:
        if entry.not_applicable is not None:
            continue
        binding = registry.get(entry.gate)
        if binding is None:
            continue
        missing = sorted(required_gate_parameters(binding) - set(entry.parameters))
        if missing:
            raise ValueError(
                f"gate entry {entry.id!r} omits parameters {missing} that the "
                f"{entry.gate!r} binding requires; thresholds come only from "
                "gates.json."
            )
