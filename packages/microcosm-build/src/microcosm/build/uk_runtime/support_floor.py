"""The zero-support floor (microcosm#1095; María's rulings of 2026-10-16 and 2026-10-21).

A target row that no source household carries cannot be fitted by any
weights: the solver's initialiser refuses a nonzero target with no support. A
row that some households carry stays bound however few they are: a cell that
rests on fewer than ten records is no reason to leave a row out, and only a
measured distortion of the calibration is (María's rule of 2026-10-21, which
replaced the floor of ten she signed on 2026-10-16). What the licence asks of
such a row is that its model figures stay in the build: the diagnostics, the
gate evidence and every written surface withhold the figures of a row that
fewer than ten distinct source households carry
(:func:`microcosm.build.uk_runtime.diagnostics.uk_withheld_rows`).

The floor is declared once, in the ``support_floor`` block of
``uk/calibration_measure_exclusions.json`` (schema 3), and signed there; its
``floor`` is 1, the only value it may take. On a built frame, a row in its
scope whose matrix row reaches no source household (``household_source_id``)
with a positive prior weight is dropped, not pooled; each target's dropped
published total is reported. The run refuses outside the block's approval window, when the
dropped rows pass either declared ceiling, and when a per-row exclusion names
a row the floor would drop (the floor covers it, so the entry must retire).
The dropped set joins the calibration's measure exclusions as
``kind: support_floor``, so the gates, the written register and the
certifier read it back rather than recompute support, which they cannot.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from microcosm.calibrate import TargetRegistry

UK_SUPPORT_FLOOR_KEY = "support_floor"
#: The one floor the register may sign: a row that any source household
#: carries stays bound (María's rule of 2026-10-21), so the floor leaves out
#: only the rows no weights can reach.
UK_SUPPORT_FLOOR_VALUE = 1
UK_SUPPORT_FLOOR_KIND = "support_floor"
UK_SUPPORT_FLOOR_METRIC = "distinct_source_households_with_positive_prior_weight"
#: The spine household each row was built from; every clone shares it.
UK_SUPPORT_LINEAGE_COLUMN = "household_source_id"
_FIELDS = (
    "contract_target_ids",
    "metric",
    "floor",
    "max_dropped_rows",
    "max_dropped_published_share",
    "reason",
    "tracking",
    "approved_by",
    "adjudication",
    "approved_on",
    "expires_on",
)
_TEXT_FIELDS = ("reason", "tracking", "approved_by", "adjudication")
#: The record fields a dropped row carries, as a per-row exclusion carries its
#: own (``measure_simulation._UK_MEASURE_EXCLUSION_FIELDS`` less the name).
_RECORD_FIELDS = (
    "reason",
    "tracking",
    "approved_by",
    "adjudication",
    "approved_on",
    "expires_on",
)


@dataclass(frozen=True)
class UKSupportFloor:
    """The signed floor: its scope, threshold, ceilings and approval window."""

    contract_target_ids: frozenset[str]
    floor: int
    max_dropped_rows: int
    max_dropped_published_share: float
    reason: str
    tracking: str
    approved_by: str
    adjudication: str
    approved_on: date
    expires_on: date

    def in_scope(self, spec: Any) -> bool:
        return spec.metadata.get("contract_target_id") in self.contract_target_ids

    def record(self, evaluated_on: date) -> dict[str, str]:
        """The measure-exclusion record a dropped row carries."""

        return {
            "kind": UK_SUPPORT_FLOOR_KIND,
            "reason": self.reason,
            "tracking": self.tracking,
            "approved_by": self.approved_by,
            "adjudication": self.adjudication,
            "approved_on": self.approved_on.isoformat(),
            "expires_on": self.expires_on.isoformat(),
            "evaluated_on": evaluated_on.isoformat(),
        }

    def assert_in_force(self, evaluated_on: date) -> None:
        if not self.approved_on <= evaluated_on <= self.expires_on:
            raise ValueError(
                "UK support floor is outside its reviewed window "
                f"({self.approved_on.isoformat()}..{self.expires_on.isoformat()}, "
                f"evaluated {evaluated_on.isoformat()}): renew the signature "
                f"({self.tracking})."
            )


def uk_support_floor_from_payload(block: Mapping[str, Any]) -> UKSupportFloor:
    """Validate the ``support_floor`` block closed-world."""

    if not isinstance(block, Mapping):
        raise ValueError("UK support floor must be an object.")
    unknown = sorted(set(block) - set(_FIELDS))
    missing = sorted(set(_FIELDS) - set(block))
    if unknown or missing:
        raise ValueError(
            f"UK support floor fields are closed-world: unknown {unknown}, "
            f"missing {missing}."
        )
    if block["metric"] != UK_SUPPORT_FLOOR_METRIC:
        raise ValueError(
            f"UK support floor metric must be {UK_SUPPORT_FLOOR_METRIC!r}, got "
            f"{block['metric']!r}."
        )
    targets = block["contract_target_ids"]
    if (
        not isinstance(targets, list)
        or not targets
        or len(set(targets)) != len(targets)
        or not all(isinstance(target, str) and target for target in targets)
    ):
        raise ValueError("UK support floor scope must list distinct target ids.")
    floor = block["floor"]
    rows = block["max_dropped_rows"]
    share = block["max_dropped_published_share"]
    if (
        not isinstance(floor, int)
        or isinstance(floor, bool)
        or floor != UK_SUPPORT_FLOOR_VALUE
    ):
        raise ValueError(
            f"UK support floor must be {UK_SUPPORT_FLOOR_VALUE}: a row that any source "
            "household carries stays bound however few they are (María's rule of "
            "2026-10-21), and the figures of a thin row are withheld, not the row."
        )
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
        raise ValueError("UK support floor max_dropped_rows must be a count.")
    if (
        not isinstance(share, (int, float))
        or isinstance(share, bool)
        or not 0.0 <= float(share) < 1.0
    ):
        raise ValueError(
            "UK support floor max_dropped_published_share must be in [0, 1)."
        )
    for field in _TEXT_FIELDS:
        if not isinstance(block[field], str) or not block[field].strip():
            raise ValueError(f"UK support floor {field} is empty.")
    dates = {}
    for field in ("approved_on", "expires_on"):
        try:
            parsed = date.fromisoformat(str(block[field]))
        except ValueError:
            parsed = None
        if parsed is None or parsed.isoformat() != block[field]:
            raise ValueError(
                f"UK support floor {field} must be canonical ISO (YYYY-MM-DD), "
                f"got {block[field]!r}."
            )
        dates[field] = parsed
    if dates["expires_on"] <= dates["approved_on"]:
        raise ValueError("UK support floor expires_on must be after approved_on.")
    return UKSupportFloor(
        contract_target_ids=frozenset(targets),
        floor=floor,
        max_dropped_rows=rows,
        max_dropped_published_share=float(share),
        reason=block["reason"],
        tracking=block["tracking"],
        approved_by=block["approved_by"],
        adjudication=block["adjudication"],
        approved_on=dates["approved_on"],
        expires_on=dates["expires_on"],
    )


def load_uk_support_floor_declaration(
    path: Path | None = None,
) -> dict[str, Any] | None:
    """The register's validated ``support_floor`` block, or ``None``.

    ``path`` is the measure-exclusion register the run reads its per-row
    exclusions from (the packaged one by default), so the floor and the
    per-row entries always come from one signed file.
    """

    if path is None:
        text = (
            importlib_resources.files("microcosm.build.uk")
            .joinpath("calibration_measure_exclusions.json")
            .read_text(encoding="utf-8")
        )
    else:
        text = Path(path).read_text(encoding="utf-8")
    block = json.loads(text).get(UK_SUPPORT_FLOOR_KEY)
    if block is None:
        return None
    uk_support_floor_from_payload(block)
    return dict(block)


def uk_household_source_ids(frame: Any) -> np.ndarray:
    """The source household of each household column, or refuse."""

    household = frame.table("household")
    if UK_SUPPORT_LINEAGE_COLUMN not in household:
        raise ValueError(
            f"The UK support floor counts source households, and the frame has "
            f"no household.{UK_SUPPORT_LINEAGE_COLUMN} column."
        )
    return household[UK_SUPPORT_LINEAGE_COLUMN].to_numpy()


def uk_row_support(
    matrix: Any, prior_weights: np.ndarray, source_ids: np.ndarray
) -> np.ndarray:
    """Distinct source households behind each row's positive-weight entries."""

    csr = sparse.csr_matrix(matrix)
    weights = np.asarray(prior_weights, dtype=np.float64)
    sources = np.asarray(source_ids)
    if csr.shape[1] != len(weights) or len(weights) != len(sources):
        raise ValueError(
            "Support needs one prior weight and one source id per matrix column."
        )
    support = np.zeros(csr.shape[0], dtype=np.int64)
    for row in range(csr.shape[0]):
        start, end = csr.indptr[row], csr.indptr[row + 1]
        columns = csr.indices[start:end]
        live = columns[(csr.data[start:end] != 0.0) & (weights[columns] > 0.0)]
        support[row] = len(np.unique(sources[live]))
    return support


@dataclass(frozen=True)
class UKSupportFloorResult:
    """The narrowed register and problem, the dropped rows and the receipt."""

    registry: TargetRegistry
    problem: Any
    exclusions: dict[str, dict[str, str]]
    receipt: dict[str, Any]


def _assert_within_ceilings(floor: UKSupportFloor, dropped: int, share: float) -> None:
    if dropped > floor.max_dropped_rows:
        raise ValueError(
            f"UK support floor drops {dropped} rows, more than its signed "
            f"ceiling of {floor.max_dropped_rows} ({floor.tracking})."
        )
    if share > floor.max_dropped_published_share:
        raise ValueError(
            f"UK support floor drops {share:.4f} of its scope's published "
            "total, more than its signed ceiling of "
            f"{floor.max_dropped_published_share} ({floor.tracking})."
        )


def _dropped_by_target(specs: Sequence[Any]) -> dict[str, dict[str, float | int]]:
    totals: dict[str, dict[str, float | int]] = {}
    for spec in specs:
        entry = totals.setdefault(
            str(spec.metadata["contract_target_id"]),
            {"rows": 0, "published_total": 0.0},
        )
        entry["rows"] = int(entry["rows"]) + 1
        entry["published_total"] = float(entry["published_total"]) + float(spec.value)
    return {key: totals[key] for key in sorted(totals)}


def apply_uk_support_floor(
    registry: TargetRegistry,
    problem: Any,
    *,
    source_ids: Sequence[object] | np.ndarray | Callable[[], Any],
    floor: UKSupportFloor,
    evaluated_on: date,
    row_excluded: Collection[str] = (),
) -> UKSupportFloorResult:
    """Drop the in-scope rows below the floor from a compiled problem.

    ``problem`` is the compiled constraint problem over ``registry``, row for
    row; its columns are the households ``source_ids`` names, in order (or a
    callable that returns them, read only when a row in scope needs them).
    ``row_excluded`` names the registry rows that carry a per-row exclusion
    and were compiled only so their support could be measured: one the floor
    would drop refuses the run, and all of them leave the problem.
    """

    floor.assert_in_force(evaluated_on)
    specs = list(registry.specs)
    names = tuple(problem.names)
    if names != tuple(spec.to_target().row_name for spec in specs):
        raise ValueError("UK support floor needs the problem's rows in register order.")
    probes = {
        position for position, spec in enumerate(specs) if spec.name in row_excluded
    }
    unknown = sorted(set(row_excluded) - {specs[position].name for position in probes})
    if unknown:
        raise ValueError(f"UK support floor probes name no compiled row: {unknown}.")
    in_scope = [
        position
        for position, spec in enumerate(specs)
        if position not in probes and floor.in_scope(spec)
    ]
    measured = sorted({*in_scope, *(p for p in probes if floor.in_scope(specs[p]))})
    support: dict[int, int] = {}
    if measured:
        lineage = source_ids() if callable(source_ids) else source_ids
        counts = uk_row_support(
            sparse.csr_matrix(problem.matrix)[measured],
            np.asarray(problem.initial_weights.values, dtype=np.float64),
            np.asarray(lineage),
        )
        support = dict(zip(measured, counts.tolist(), strict=True))
    redundant = sorted(
        specs[position].name
        for position in probes
        if position in support and support[position] < floor.floor
    )
    if redundant:
        raise ValueError(
            "UK support floor covers rows that also carry a per-row measure "
            f"exclusion; retire those entries ({floor.tracking}): {redundant}."
        )
    dropped = [position for position in in_scope if support[position] < floor.floor]
    scope_total = math.fsum(float(specs[position].value) for position in in_scope)
    dropped_total = math.fsum(float(specs[position].value) for position in dropped)
    share = dropped_total / scope_total if scope_total > 0.0 else 0.0
    _assert_within_ceilings(floor, len(dropped), share)
    record = floor.record(evaluated_on)
    exclusions = {specs[position].name: dict(record) for position in dropped}
    leaving = probes | set(dropped)
    kept_rows = np.asarray(
        [position for position in range(len(specs)) if position not in leaving],
        dtype=np.int64,
    )
    narrowed = TargetRegistry(
        [specs[position] for position in kept_rows], country=registry.country
    )
    receipt = {
        "metric": UK_SUPPORT_FLOOR_METRIC,
        "floor": floor.floor,
        "tracking": floor.tracking,
        "approved_on": floor.approved_on.isoformat(),
        "expires_on": floor.expires_on.isoformat(),
        "evaluated_on": evaluated_on.isoformat(),
        "in_scope_rows": len(in_scope),
        "probed_row_exclusions": len(probes),
        "dropped_rows": len(dropped),
        "dropped_published_share": share,
        "dropped_by_target": _dropped_by_target([specs[p] for p in dropped]),
        "dropped": sorted(exclusions),
        "solved_register_sha256": narrowed.version,
    }
    if not leaving:
        return UKSupportFloorResult(registry, problem, {}, receipt)
    narrowed_problem = replace(
        problem,
        matrix=sparse.csr_array(sparse.csr_matrix(problem.matrix)[kept_rows]),
        target_vector=np.asarray(problem.target_vector)[kept_rows],
        names=tuple(names[row] for row in kept_rows),
        targets=tuple(problem.targets[row] for row in kept_rows),
    )
    return UKSupportFloorResult(narrowed, narrowed_problem, exclusions, receipt)


def uk_support_floor_names(
    measure_exclusions: Mapping[str, Mapping[str, str]],
) -> frozenset[str]:
    """The rows a run's floor dropped, read back from its measure exclusions."""

    return frozenset(
        name
        for name, record in measure_exclusions.items()
        if record.get("kind") == UK_SUPPORT_FLOOR_KIND
    )


def narrow_uk_registry_by_support_floor(
    registry: TargetRegistry,
    measure_exclusions: Mapping[str, Mapping[str, str]],
    floor: UKSupportFloor | None,
) -> TargetRegistry:
    """The register less the rows a run's floor dropped, checked against it.

    For the readers that cannot recompute support (the gates, the written
    register, the certifier's recompiled register): each recorded drop must
    be a row of ``registry`` in the declared scope, under the declared
    signature, and the drops must sit within the declared row ceiling.
    """

    names = uk_support_floor_names(measure_exclusions)
    if not names:
        return registry
    if floor is None:
        raise ValueError(
            "The run's measure exclusions record support-floor drops, and the "
            "register declares no support floor."
        )
    by_name: dict[str, list[Any]] = {}
    for spec in registry.specs:
        by_name.setdefault(spec.name, []).append(spec)
    failures = []
    for name in sorted(names):
        specs = by_name.get(name, [])
        record = measure_exclusions[name]
        if not specs:
            failures.append(f"{name}: not a row of the register")
        elif not all(floor.in_scope(spec) for spec in specs):
            failures.append(f"{name}: outside the floor's scope")
        elif any(
            record.get(field) != value
            for field, value in floor.record(floor.approved_on).items()
            if field in _RECORD_FIELDS
        ):
            failures.append(f"{name}: not under the declared signature")
    if failures:
        raise ValueError(
            "The run's support-floor drops do not match the declared floor: "
            + "; ".join(failures)
        )
    if len(names) > floor.max_dropped_rows:
        raise ValueError(
            f"The run's support floor dropped {len(names)} rows, more than its "
            f"signed ceiling of {floor.max_dropped_rows} ({floor.tracking})."
        )
    return TargetRegistry(
        [spec for spec in registry.specs if spec.name not in names],
        country=registry.country,
    )
