"""Held-out calibration targets, drawn by hashed blocks, for any country.

A local build holds some of its targets out of the calibration and scores the
calibrated file on them. One held-out target is not a fair test when its
siblings and its parent train, because adding up pins it. So a hold-out draws
whole *units*, for example every district target of one concept family in one
state. A unit is held out when a salted hash of its key falls below the
fraction (:func:`microcosm.build.holdout.hash_holdout_unit`).

A country declares its passes as data, and :func:`load_target_holdouts` reads
them. Each pass declares:

- ``scope``: tests on a record's fields that pick the records the pass
  writes. With no test, the pass writes every record.
- ``eligible``: tests that pick, in scope, the records that get a unit. The
  other records in scope train.
- ``unit_fields``: the fields the unit key joins with ``|``. The name
  ``concept_family`` stands for the record's concept family
  (:class:`ConceptFamilyRule`).
- ``salt``, ``default_fraction`` and ``max_fraction``.
- ``receipt``: the receipt keys the pass writes, in order.
- ``scoring``: how :func:`score_holdout` scores the held rows against a
  pro-rata baseline, which is a parent target times the child's share of the
  parent's population.

A target record is a mapping with at least ``name``, ``value``, ``family``
and ``geography_level``, the field names the targets carry everywhere. The
role pass adds ``holdout_unit`` and ``role``. The training target set
(:func:`training_target_set`) holds the training rows only, so a held-out
target never reaches the solve. Nothing here reads a frame or country code
(microcosm#1728).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from microcosm.build.holdout import hash_holdout_unit

__all__ = [
    "CONCEPT_FAMILY_FIELD",
    "POPULATION_CHECKS",
    "RECEIPT_KEYS",
    "RECORD_TEST_KINDS",
    "ROLE_HOLDOUT",
    "ROLE_TRAIN",
    "UNIT_SEPARATOR",
    "ConceptFamilyRule",
    "HoldoutPass",
    "HoldoutScoring",
    "PopulationShares",
    "RecordTest",
    "TargetHoldoutError",
    "TargetHoldouts",
    "assign_holdout_roles",
    "attach_population_shares",
    "csr_row_measure",
    "error_summary",
    "holdout_rows",
    "load_target_holdouts",
    "missing_population_shares",
    "pro_rata_baseline",
    "score_holdout",
    "train_rows",
    "training_target_set",
]

#: The role a record trains under, and the role of a held-out record.
ROLE_TRAIN = "train"
ROLE_HOLDOUT = "holdout"
#: The separator of a unit key's fields.
UNIT_SEPARATOR = "|"
#: The unit field that stands for the record's concept family.
CONCEPT_FAMILY_FIELD = "concept_family"
#: ``equals`` compares a field with a value; ``truthy`` needs a field that is
#: present and not empty, zero or None.
RECORD_TEST_KINDS = ("equals", "truthy")
#: The receipt keys a pass may declare.
RECEIPT_KEYS = (
    "unit",
    "salt",
    "fraction",
    "eligible_units",
    "held_units",
    "held_unit_keys",
    "eligible_targets",
    "held_targets",
    "hash",
)
#: How the pro-rata baseline checks a row's populations: ``nonzero`` refuses
#: a missing or zero population; ``finite_positive`` also refuses a negative
#: or non-finite one.
POPULATION_CHECKS = ("nonzero", "finite_positive")

_SECTION_KEYS = frozenset({"concept_families", "passes"})
_FAMILY_KEYS = frozenset(
    {"source_field", "families", "merge_prefixes", "strip_suffixes", "rationale"}
)
_PASS_KEYS = frozenset(
    {
        "id",
        "label",
        "unit",
        "unit_fields",
        "salt",
        "default_fraction",
        "max_fraction",
        "scope",
        "eligible",
        "receipt",
        "hash_description",
        "scoring",
        "rationale",
    }
)
_SCORING_KEYS = frozenset(
    {
        "level",
        "parent_name_field",
        "child_population_field",
        "parent_population_field",
        "population_check",
        "populations",
        "by_family",
        "target_fields",
        "baseline",
    }
)
_SHARES_KEYS = frozenset({"families", "code_field", "parent_code_divisor"})


class TargetHoldoutError(ValueError):
    """A hold-out declaration or a hold-out input is not valid."""


@dataclass(frozen=True)
class RecordTest:
    """One test on one field of a target record."""

    key: str
    kind: str
    value: Any = None

    def accepts(self, record: Mapping[str, Any]) -> bool:
        if self.kind == "equals":
            return record.get(self.key) == self.value
        return bool(record.get(self.key))


@dataclass(frozen=True)
class ConceptFamilyRule:
    """The concept family of a record: what one held unit groups together.

    For a record of one of ``families``, the family is its ``source_field``
    with the first matching ``merge_prefixes`` entry as the whole family, or
    else without the first matching ``strip_suffixes`` entry. Measures that
    add up to each other then share a family. A record of another family is
    its own concept family.
    """

    source_field: str
    families: frozenset[str]
    merge_prefixes: tuple[str, ...]
    strip_suffixes: tuple[str, ...]
    rationale: str = ""

    def of_measure(self, measure: str) -> str:
        measure = str(measure)
        for prefix in self.merge_prefixes:
            if measure.startswith(prefix):
                return prefix
        for suffix in self.strip_suffixes:
            if measure.endswith(suffix):
                return measure[: -len(suffix)]
        return measure

    def of_record(self, record: Mapping[str, Any]) -> str:
        if record.get("family") in self.families:
            return self.of_measure(str(record.get(self.source_field)))
        return str(record.get("family"))


@dataclass(frozen=True)
class PopulationShares:
    """How a pass's rows get the populations of the pro-rata baseline.

    A row of one of ``families`` whose ``code_field`` is set gets its child
    population, and its parent's population: the sum of the child
    populations whose code divided by ``parent_code_divisor`` (integer
    division) is the same.
    """

    families: frozenset[str]
    code_field: str
    parent_code_divisor: int


@dataclass(frozen=True)
class HoldoutScoring:
    """How :func:`score_holdout` scores a pass's held rows."""

    level: str
    parent_name_field: str
    child_population_field: str
    parent_population_field: str
    population_check: str
    populations: PopulationShares | None
    by_family: bool
    target_fields: tuple[tuple[str, str], ...]
    baseline: str


@dataclass(frozen=True)
class HoldoutPass:
    """One declared hold-out pass (see the module docstring)."""

    pass_id: str
    label: str
    unit: str
    unit_fields: tuple[str, ...]
    salt: str
    default_fraction: float
    max_fraction: float
    scope: tuple[RecordTest, ...]
    eligible: tuple[RecordTest, ...]
    receipt: tuple[str, ...]
    hash_description: str | None
    scoring: HoldoutScoring | None
    rationale: str

    def in_scope(self, record: Mapping[str, Any]) -> bool:
        return all(test.accepts(record) for test in self.scope)

    def unit_key(
        self, record: Mapping[str, Any], concept_families: ConceptFamilyRule
    ) -> str | None:
        """The record's unit key, or None when the record is not eligible."""

        if not all(test.accepts(record) for test in self.eligible):
            return None
        parts = []
        for field in self.unit_fields:
            if field == CONCEPT_FAMILY_FIELD:
                parts.append(concept_families.of_record(record))
            else:
                parts.append(f"{record[field]}")
        return UNIT_SEPARATOR.join(parts)


@dataclass(frozen=True)
class TargetHoldouts:
    """A country's hold-out passes and its concept-family rule."""

    concept_families: ConceptFamilyRule
    passes: tuple[HoldoutPass, ...]

    def holdout_pass(self, pass_id: str) -> HoldoutPass:
        for holdout_pass in self.passes:
            if holdout_pass.pass_id == pass_id:
                return holdout_pass
        raise TargetHoldoutError(
            f"No hold-out pass {pass_id!r}; declared: "
            f"{[holdout_pass.pass_id for holdout_pass in self.passes]}."
        )


# ---------------------------------------------------------------------------
# Reading the declaration
# ---------------------------------------------------------------------------


def _mapping(value: Any, *, context: str, keys: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TargetHoldoutError(f"{context} must be an object.")
    unknown = sorted(set(value) - keys)
    missing = sorted(keys - set(value))
    if unknown or missing:
        raise TargetHoldoutError(
            f"{context}: unknown keys {unknown}, missing keys {missing}."
        )
    return value


def _text(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise TargetHoldoutError(f"{context} must be a non-empty string.")
    return value


def _texts(value: Any, *, context: str, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise TargetHoldoutError(f"{context} must be a list of strings.")
    items = tuple(_text(item, context=f"{context} entry") for item in value)
    if len(set(items)) != len(items):
        raise TargetHoldoutError(f"{context} repeats an entry.")
    return items


def _fraction(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TargetHoldoutError(f"{context} must be a number.")
    if not 0.0 <= float(value) <= 1.0:
        raise TargetHoldoutError(f"{context} must be in [0, 1].")
    return float(value)


def _record_tests(value: Any, *, context: str) -> tuple[RecordTest, ...]:
    if not isinstance(value, list):
        raise TargetHoldoutError(f"{context} must be a list of tests.")
    tests = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise TargetHoldoutError(f"{context}: a test must be an object.")
        kind = raw.get("kind")
        if kind not in RECORD_TEST_KINDS:
            raise TargetHoldoutError(
                f"{context}: test kind {kind!r} is not one of {RECORD_TEST_KINDS}."
            )
        expected = {"key", "kind", "value"} if kind == "equals" else {"key", "kind"}
        if set(raw) != expected:
            raise TargetHoldoutError(
                f"{context}: a {kind!r} test has the keys {sorted(expected)}."
            )
        tests.append(
            RecordTest(
                _text(raw["key"], context=f"{context} key"), kind, raw.get("value")
            )
        )
    return tuple(tests)


def _population_shares(value: Any, *, context: str) -> PopulationShares | None:
    if value is None:
        return None
    raw = _mapping(value, context=context, keys=_SHARES_KEYS)
    divisor = raw["parent_code_divisor"]
    if isinstance(divisor, bool) or not isinstance(divisor, int) or divisor < 1:
        raise TargetHoldoutError(
            f"{context} parent_code_divisor must be a positive integer."
        )
    return PopulationShares(
        families=frozenset(_texts(raw["families"], context=f"{context} families")),
        code_field=_text(raw["code_field"], context=f"{context} code_field"),
        parent_code_divisor=divisor,
    )


def _scoring(value: Any, *, context: str) -> HoldoutScoring | None:
    if value is None:
        return None
    raw = _mapping(value, context=context, keys=_SCORING_KEYS)
    check = raw["population_check"]
    if check not in POPULATION_CHECKS:
        raise TargetHoldoutError(
            f"{context} population_check must be one of {POPULATION_CHECKS}."
        )
    if not isinstance(raw["by_family"], bool):
        raise TargetHoldoutError(f"{context} by_family must be true or false.")
    fields = raw["target_fields"]
    if not isinstance(fields, Mapping) or not all(
        isinstance(key, str) and key and isinstance(field, str) and field
        for key, field in fields.items()
    ):
        raise TargetHoldoutError(
            f"{context} target_fields must map output keys to record fields."
        )
    return HoldoutScoring(
        level=_text(raw["level"], context=f"{context} level"),
        parent_name_field=_text(
            raw["parent_name_field"], context=f"{context} parent_name_field"
        ),
        child_population_field=_text(
            raw["child_population_field"], context=f"{context} child_population_field"
        ),
        parent_population_field=_text(
            raw["parent_population_field"],
            context=f"{context} parent_population_field",
        ),
        population_check=check,
        populations=_population_shares(
            raw["populations"], context=f"{context} populations"
        ),
        by_family=raw["by_family"],
        target_fields=tuple(fields.items()),
        baseline=_text(raw["baseline"], context=f"{context} baseline"),
    )


def _holdout_pass(raw: Any) -> HoldoutPass:
    raw = _mapping(raw, context="hold-out pass", keys=_PASS_KEYS)
    pass_id = _text(raw["id"], context="hold-out pass id")
    context = f"hold-out pass {pass_id!r}"
    receipt = _texts(raw["receipt"], context=f"{context} receipt")
    unknown = sorted(set(receipt) - set(RECEIPT_KEYS))
    if unknown:
        raise TargetHoldoutError(f"{context} receipt has unknown keys {unknown}.")
    hash_description = raw["hash_description"]
    if ("hash" in receipt) != (hash_description is not None):
        raise TargetHoldoutError(
            f"{context}: hash_description is set exactly when the receipt has 'hash'."
        )
    if hash_description is not None:
        _text(hash_description, context=f"{context} hash_description")
    default_fraction = _fraction(
        raw["default_fraction"], context=f"{context} default_fraction"
    )
    max_fraction = _fraction(raw["max_fraction"], context=f"{context} max_fraction")
    if default_fraction > max_fraction:
        raise TargetHoldoutError(f"{context}: default_fraction exceeds max_fraction.")
    return HoldoutPass(
        pass_id=pass_id,
        label=_text(raw["label"], context=f"{context} label"),
        unit=_text(raw["unit"], context=f"{context} unit"),
        unit_fields=_texts(raw["unit_fields"], context=f"{context} unit_fields"),
        salt=_text(raw["salt"], context=f"{context} salt"),
        default_fraction=default_fraction,
        max_fraction=max_fraction,
        scope=_record_tests(raw["scope"], context=f"{context} scope"),
        eligible=_record_tests(raw["eligible"], context=f"{context} eligible"),
        receipt=receipt,
        hash_description=hash_description,
        scoring=_scoring(raw["scoring"], context=f"{context} scoring"),
        rationale=_text(raw["rationale"], context=f"{context} rationale"),
    )


def load_target_holdouts(section: Mapping[str, Any]) -> TargetHoldouts:
    """Read a country's ``holdouts`` declaration; refuse anything unknown."""

    raw = _mapping(section, context="holdouts", keys=_SECTION_KEYS)
    families = _mapping(
        raw["concept_families"], context="holdouts concept_families", keys=_FAMILY_KEYS
    )
    concept_families = ConceptFamilyRule(
        source_field=_text(
            families["source_field"], context="concept_families source_field"
        ),
        families=frozenset(
            _texts(families["families"], context="concept_families families")
        ),
        merge_prefixes=_texts(
            families["merge_prefixes"],
            context="concept_families merge_prefixes",
            allow_empty=True,
        ),
        strip_suffixes=_texts(
            families["strip_suffixes"],
            context="concept_families strip_suffixes",
            allow_empty=True,
        ),
        rationale=_text(families["rationale"], context="concept_families rationale"),
    )
    if not isinstance(raw["passes"], list) or not raw["passes"]:
        raise TargetHoldoutError("holdouts passes must be a non-empty list.")
    passes = tuple(_holdout_pass(item) for item in raw["passes"])
    ids = [holdout_pass.pass_id for holdout_pass in passes]
    if len(set(ids)) != len(ids):
        raise TargetHoldoutError(f"holdouts passes repeat an id: {ids}.")
    return TargetHoldouts(concept_families=concept_families, passes=passes)


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


def assign_holdout_roles(
    records: Sequence[dict],
    holdout_pass: HoldoutPass,
    concept_families: ConceptFamilyRule,
    *,
    fraction: float | None = None,
    salt: str | None = None,
) -> dict:
    """Set ``holdout_unit`` and ``role`` on each record in scope; return the receipt.

    An eligible record gets its unit key and the unit's role: a unit is held
    out by :func:`microcosm.build.holdout.hash_holdout_unit` on its key, once.
    A record in scope that is not eligible gets no unit and trains. A record
    out of scope keeps what it had.
    """

    fraction = holdout_pass.default_fraction if fraction is None else fraction
    salt = holdout_pass.salt if salt is None else salt
    if not isinstance(salt, str) or not salt:
        raise TargetHoldoutError(
            f"{holdout_pass.label} holdout salt must be a non-empty string."
        )
    if not 0.0 <= float(fraction) <= holdout_pass.max_fraction:
        raise TargetHoldoutError(
            f"{holdout_pass.label} holdout fraction must be in "
            f"[0, {holdout_pass.max_fraction}], got {fraction!r}."
        )
    units: dict[str, bool] = {}
    eligible: list[dict] = []
    for record in records:
        if not holdout_pass.in_scope(record):
            continue
        unit = holdout_pass.unit_key(record, concept_families)
        record["holdout_unit"] = unit
        if unit is None:
            record["role"] = ROLE_TRAIN
            continue
        if unit not in units:
            units[unit] = hash_holdout_unit(unit, fraction=fraction, salt=salt)
        record["role"] = ROLE_HOLDOUT if units[unit] else ROLE_TRAIN
        eligible.append(record)
    held_unit_keys = sorted(unit for unit, held in units.items() if held)
    values = {
        "unit": holdout_pass.unit,
        "salt": salt,
        "fraction": float(fraction),
        "eligible_units": len(units),
        "held_units": len(held_unit_keys),
        "held_unit_keys": held_unit_keys,
        "eligible_targets": len(eligible),
        "held_targets": sum(1 for record in eligible if record["role"] == ROLE_HOLDOUT),
        "hash": holdout_pass.hash_description,
    }
    return {key: values[key] for key in holdout_pass.receipt}


def train_rows(records: Sequence[Mapping]) -> list[int]:
    return [
        index for index, record in enumerate(records) if record["role"] == ROLE_TRAIN
    ]


def holdout_rows(records: Sequence[Mapping]) -> list[int]:
    return [
        index for index, record in enumerate(records) if record["role"] == ROLE_HOLDOUT
    ]


# ---------------------------------------------------------------------------
# The training target set
# ---------------------------------------------------------------------------


def csr_row_measure(matrix: sparse.csr_array, row: int, n: int):
    """A callable measure returning one CSR row as a dense float64 vector.

    The calibrate kernel compiles targets through ``build_constraint_matrix``,
    which reads each target's row and keeps its nonzeros; a callable measure
    lets it read the row straight from the checkpoint CSR (the UK rowwise
    precedent, ``uk_runtime.local_rowwise._rowwise_target_set``). Only one
    dense row exists at a time. float32 values widen exactly to float64, as
    the dense checkpoint's float32 columns did.
    """

    start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
    indices = matrix.indices[start:stop]
    data = matrix.data[start:stop]

    def measure(frame) -> np.ndarray:
        values = np.zeros(n, dtype=np.float64)
        values[indices] = data
        return values

    # Calibration diagnostics name a callable measure by its qualified name;
    # name the matrix row rather than this closure.
    measure.__qualname__ = f"target_matrix_row[{row}]"
    return measure


def training_target_set(
    records: Sequence[Mapping],
    matrix: sparse.csr_array,
    n_households: int,
    *,
    specs: Sequence | None = None,
):
    """The TargetSet the calibrator sees: training targets only.

    Held-out targets are never constructed, so they cannot reach the solve.
    With ``specs`` (the checkpoint's registry specs, row-aligned with
    ``records``) each target keeps its spec's value, period, source,
    metadata and calibration hierarchy, as ``TargetSpec.to_target`` would,
    and only its measure becomes the CSR row.
    """

    from microcosm.calibrate.target import Target, TargetSet

    if matrix.shape != (len(records), n_households):
        raise ValueError(
            f"target matrix shape {matrix.shape} does not match "
            f"{len(records)} targets x {n_households} households."
        )
    if specs is not None and [spec.name for spec in specs] != [
        record["name"] for record in records
    ]:
        raise ValueError("target specs are not row-aligned with the roles.")
    targets = []
    for index in train_rows(records):
        record = records[index]
        if record["role"] != ROLE_TRAIN:  # pragma: no cover - train_rows filters
            raise AssertionError("a held-out target reached the calibration set")
        measure = csr_row_measure(matrix, index, n_households)
        if specs is None:
            targets.append(
                Target(
                    name=record["name"],
                    entity=record.get("entity", "household"),
                    measure=measure,
                    value=float(record["value"]),
                    period=record["period"],
                    source=record["source"],
                )
            )
            continue
        spec = specs[index]
        if spec.filter is not None:
            raise ValueError(
                f"{spec.name}: a filtered spec cannot take a matrix-row measure; "
                "the row already carries the filter."
            )
        targets.append(
            Target(
                name=spec.name,
                entity=spec.entity,
                measure=measure,
                value=spec.value,
                period=spec.period,
                tolerance=spec.tolerance,
                source=spec.source,
                metadata=spec.metadata,
                hierarchy=spec.hierarchy,
            )
        )
    held = {records[index]["name"] for index in holdout_rows(records)}
    leaked = held & {target.name for target in targets}
    if leaked:
        raise AssertionError(f"held-out targets reached the calibrator: {leaked}")
    return TargetSet(targets)


# ---------------------------------------------------------------------------
# Scoring against the pro-rata baseline
# ---------------------------------------------------------------------------


def error_summary(estimates, targets, *, cap: float) -> dict:
    from microcosm.calibrate import relative_error_loss

    estimates = np.asarray(estimates, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    scale = np.maximum(np.abs(targets), 1.0)
    errors = np.abs(estimates - targets) / scale
    return {
        "mean_abs_rel_error": float(errors.mean()),
        "median_abs_rel_error": float(np.median(errors)),
        "p90_abs_rel_error": float(np.quantile(errors, 0.9)),
        "max_abs_rel_error": float(errors.max()),
        "fraction_within_10pct": float((errors <= 0.10).mean()),
        "capped_loss": float(
            relative_error_loss(estimates, targets, target_loss_cap=cap)
        ),
    }


def attach_population_shares(
    records: Sequence[dict],
    child_populations: Mapping[int, float],
    scoring: HoldoutScoring,
) -> None:
    """Give each row its child and parent populations, for the baseline."""

    shares = scoring.populations
    if shares is None:
        raise TargetHoldoutError(
            f"The {scoring.level} scoring declares no population shares."
        )
    parent_population: dict[int, float] = {}
    for child, population in child_populations.items():
        parent = int(child) // shares.parent_code_divisor
        parent_population[parent] = parent_population.get(parent, 0.0) + float(
            population
        )
    for record in records:
        code = record.get(shares.code_field)
        if record.get("family") not in shares.families or not code:
            continue
        child = int(code)
        record[scoring.child_population_field] = float(
            child_populations.get(child, 0.0)
        )
        record[scoring.parent_population_field] = float(
            parent_population.get(child // shares.parent_code_divisor, 0.0)
        )


def missing_population_shares(
    records: Sequence[Mapping], scoring: HoldoutScoring
) -> list[str]:
    """The rows the baseline cannot score: no positive child or parent population."""

    shares = scoring.populations
    if shares is None:
        raise TargetHoldoutError(
            f"The {scoring.level} scoring declares no population shares."
        )
    return [
        record["name"]
        for record in records
        if record.get("family") in shares.families
        and record.get(shares.code_field)
        and not (
            record.get(scoring.child_population_field, 0) > 0
            and record.get(scoring.parent_population_field, 0) > 0
        )
    ]


def pro_rata_baseline(
    records: Sequence[Mapping], rows: Sequence[int], scoring: HoldoutScoring
) -> np.ndarray:
    """The parent target times the child's population share, for each row."""

    by_name = {record["name"]: record for record in records}
    values = []
    for index in rows:
        record = records[index]
        parent = by_name.get(record.get(scoring.parent_name_field))
        if scoring.population_check == "nonzero":
            if parent is None:
                raise TargetHoldoutError(
                    f"{record['name']} has no bound parent for the baseline."
                )
            child = record.get(scoring.child_population_field)
            parent_population = record.get(scoring.parent_population_field)
            if not child or not parent_population:
                raise TargetHoldoutError(
                    f"{record['name']} carries no child or parent population for "
                    "the pro-rata baseline."
                )
            values.append(
                float(parent["value"]) * float(child) / float(parent_population)
            )
            continue
        child = float(record.get(scoring.child_population_field, 0))
        parent_population = float(record.get(scoring.parent_population_field, 0))
        if parent is None or not (
            math.isfinite(child)
            and math.isfinite(parent_population)
            and child > 0
            and parent_population > 0
        ):
            raise TargetHoldoutError(
                f"{record['name']} has no parent or positive child and parent "
                "population for the pro-rata baseline."
            )
        values.append(float(parent["value"]) * child / parent_population)
    return np.asarray(values, dtype=np.float64)


def score_holdout(
    records: Sequence[Mapping],
    matrix: sparse.csr_array,
    scoring: HoldoutScoring,
    *,
    design_weights: np.ndarray,
    final_weights: np.ndarray,
    cap: float,
    receipt: Mapping | None = None,
) -> dict:
    """Held-out rows at the scoring level: calibrated vs design vs pro-rata."""

    rows = [
        row
        for row in holdout_rows(records)
        if records[row].get("geography_level") == scoring.level
    ]
    base = {"report_only": True, **dict(receipt or {}), "n_targets": len(rows)}
    if not rows:
        return base
    held = sparse.csr_array(matrix[rows]).astype(np.float64)
    targets = np.asarray([records[index]["value"] for index in rows], dtype=np.float64)
    final = held @ np.asarray(final_weights, dtype=np.float64)
    design = held @ np.asarray(design_weights, dtype=np.float64)
    baseline = pro_rata_baseline(records, rows, scoring)
    scale = np.maximum(np.abs(targets), 1.0)
    final_error = np.abs(final - targets) / scale
    baseline_error = np.abs(baseline - targets) / scale
    result = {
        **base,
        "calibrated": error_summary(final, targets, cap=cap),
        "design_weights": error_summary(design, targets, cap=cap),
        "pro_rata_baseline": error_summary(baseline, targets, cap=cap),
        "calibrated_beats_pro_rata_share": float((final_error < baseline_error).mean()),
        "baseline": scoring.baseline,
    }
    if scoring.by_family:
        families: dict[str, list[int]] = defaultdict(list)
        for position, index in enumerate(rows):
            family = str(records[index]["holdout_unit"]).split(UNIT_SEPARATOR, 1)[1]
            families[family].append(position)
        by_family = {}
        for family, positions in sorted(families.items()):
            idx = np.asarray(positions)
            by_family[family] = {
                "n_targets": len(positions),
                "calibrated_mean_abs_rel_error": float(final_error[idx].mean()),
                "pro_rata_mean_abs_rel_error": float(baseline_error[idx].mean()),
                "calibrated_beats_pro_rata": float(
                    (final_error[idx] < baseline_error[idx]).mean()
                ),
            }
        result["by_family"] = by_family
    result["targets"] = [
        {
            "name": records[index]["name"],
            **{key: records[index][field] for key, field in scoring.target_fields},
            "target": float(targets[position]),
            "calibrated": float(final[position]),
            "design": float(design[position]),
            "pro_rata": float(baseline[position]),
        }
        for position, index in enumerate(rows)
    ]
    return result
