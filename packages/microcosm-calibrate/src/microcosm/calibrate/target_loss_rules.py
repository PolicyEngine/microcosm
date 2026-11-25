"""Target-loss rules over plain row arrays, shared by every country.

A calibration minimizes the weighted mean of each target's capped, scaled
miss. Two per-target vectors shape that loss: the loss *weight* and the loss
*cap*. This module computes both from what each target is, never from a
hand-written vector:

- its **level** (the geography grain it is measured at: nation, state,
  district, local authority, ...),
- its **family** (the source or concept family it belongs to),
- its value **basis** (``"count"`` or ``"amount"``),
- its **value**, and
- for the concept-budget rule, its budget **group**.

A country says what each of its rows is (:class:`TargetLossRows`) and declares
the rule it uses by name with its parameters (:class:`TargetWeightRule`). The
arithmetic lives here once (microcosm#1729).

Weight rules (:data:`TARGET_WEIGHT_RULES`; ``L`` levels, ``F_l`` families in
level ``l``, ``n`` rows in a (level, family) cell):

``uniform``
    Every target weighs one.
``level_equal``
    Each level takes one equal share of the loss, rows inside a level equal
    shares of it: ``w = 1 / (L * n_l)``. The vector sums to one.
``level_family_equal``
    Each level takes one equal share, each family inside a level one equal
    share of it, rows inside a family equal shares:
    ``w = 1 / (L * F_l * n)``. The vector sums to one. With the parameter
    ``count_power`` (for example ``0.5``), the count rows of a cell share the
    cell's count budget in proportion to ``max(|value|, 1) ** count_power``;
    amount rows keep equal shares, and every level, family and cell budget is
    unchanged.
``sqrt_value_concept_budget``
    Value weight ``max(|value|, 1) ** value_power`` over its basis mean; each
    budget group rescaled to sum to its largest member's weight; each basis
    rescaled to an equal share of the total; normalized to mean one. It reads
    no level: a caller folds geography into the budget groups.

Any rule may carry ``family_multipliers``: each named family's weights are
multiplied, in sorted family order, and the result is renormalized to mean
one. A multiplier that names no row's family is an error.

Loss caps (:func:`target_loss_caps`): one number for the run by default, with
optional overrides for a level or a family, so that rows joining a solve do
not change the cap of the rows already in it. The solver takes one cap;
:func:`single_cap_equivalent` gives the one-cap loss that equals the loss
with a cap for each target, so a declared vector needs no solver change.

This module imports no solver code and no country code.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

__all__ = [
    "AMOUNT_BASIS",
    "COUNT_BASIS",
    "TARGET_WEIGHT_RULES",
    "SingleCapEquivalent",
    "TargetLossRows",
    "TargetWeightRule",
    "apply_family_multipliers",
    "budget_group_weights",
    "budget_groups",
    "single_cap_equivalent",
    "target_loss_caps",
    "target_loss_weight_shares",
    "target_loss_weights_for_rule",
    "value_basis_weights",
]

COUNT_BASIS = "count"
AMOUNT_BASIS = "amount"
_BASES = (COUNT_BASIS, AMOUNT_BASIS)


# ---------------------------------------------------------------------------
# What a country says about its rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetLossRows:
    """What each target is, row-aligned with the solve's target axis.

    Only ``values`` is always needed. A rule names the other fields it reads
    and refuses rows that lack one of them.

    Attributes:
        values: The target values, one finite number per row.
        levels: Each row's level label.
        families: Each row's family label (``None`` for a row with none).
        bases: Each row's value basis, ``"count"`` or ``"amount"``.
        groups: Each row's budget group (any hashable); rows that share a
            group share one budget under ``sqrt_value_concept_budget``.
    """

    values: Sequence[float] | np.ndarray
    levels: Sequence[str] | None = None
    families: Sequence[str | None] | None = None
    bases: Sequence[str] | None = None
    groups: Sequence[Hashable] | None = None

    def __post_init__(self) -> None:
        n_rows = len(self.values)
        for label in ("levels", "families", "bases", "groups"):
            column = getattr(self, label)
            if column is not None and len(column) != n_rows:
                raise ValueError(
                    f"target loss rows: {label} has {len(column)} entries for "
                    f"{n_rows} values."
                )

    def __len__(self) -> int:
        return len(self.values)

    def require(self, label: str, *, rule: str) -> Sequence:
        """The named column, refusing rows that do not carry it."""

        column = getattr(self, label)
        if column is None:
            raise ValueError(
                f"target weight rule {rule!r} reads each row's {label}; the "
                "rows carry none."
            )
        return column


# ---------------------------------------------------------------------------
# Building blocks (the arithmetic every rule shares)
# ---------------------------------------------------------------------------


def value_basis_weights(
    values: Sequence[float], bases: np.ndarray, *, power: float
) -> np.ndarray:
    """``max(|value|, 1) ** power``, divided by its mean over the row's basis."""

    weights = np.ones(len(bases), dtype=np.float64)
    magnitudes = np.asarray(
        [max(abs(float(value)), 1.0) for value in values],
        dtype=np.float64,
    )
    raw_weights = magnitudes**power
    for basis in sorted(set(bases.tolist())):
        mask = bases == basis
        mean_value = raw_weights[mask].mean()
        if mean_value > 0:
            weights[mask] = raw_weights[mask] / mean_value
    return weights


def budget_groups(groups: Sequence[Hashable]) -> list[list[int]]:
    """Row indices sharing each budget group, in first-row order."""

    group_indices: dict[Hashable, list[int]] = {}
    for index, key in enumerate(groups):
        group_indices.setdefault(key, []).append(index)
    return list(group_indices.values())


def budget_group_weights(weights: np.ndarray, groups: Sequence[Hashable]) -> np.ndarray:
    """Rescale each budget group, in place, to sum to its largest member."""

    for indices in budget_groups(groups):
        group_weights = weights[indices]
        group_total = float(group_weights.sum())
        group_budget = float(group_weights.max(initial=0.0))
        if group_total > 0 and group_budget > 0:
            weights[indices] *= group_budget / group_total
    return weights


def apply_family_multipliers(
    weights: np.ndarray,
    families: Sequence[str | None],
    family_multipliers: Mapping[str, float],
) -> np.ndarray:
    """Multiply each named family's weights, then renormalize to mean one.

    Families are applied in sorted order. A multiplier that names no row's
    family raises ``ValueError``. The multipliers are not checked here: a
    caller admits only positive finite ones.
    """

    weights = np.array(weights, dtype=np.float64)
    family_array = np.asarray(list(families), dtype=object)
    for family, multiplier in sorted(family_multipliers.items()):
        mask = family_array == family
        if not mask.any():
            raise ValueError(
                f"family multiplier for {family!r} matches no target family."
            )
        weights[mask] *= multiplier
    return weights / weights.mean()


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------


def _uniform(rows: TargetLossRows) -> np.ndarray:
    return np.ones(len(rows), dtype=np.float64)


def _level_equal(rows: TargetLossRows) -> np.ndarray:
    labels = [str(level) for level in rows.require("levels", rule="level_equal")]
    if any(not label for label in labels):
        raise ValueError("level_equal: every target must name its level.")
    counts = Counter(labels)
    level_count = len(counts)
    return np.asarray(
        [1.0 / (level_count * counts[label]) for label in labels],
        dtype=np.float64,
    )


def _level_family_equal(
    rows: TargetLossRows, *, count_power: float | None = None
) -> np.ndarray:
    rule = "level_family_equal"
    levels = np.asarray(list(rows.require("levels", rule=rule)), dtype=object)
    families = np.asarray(list(rows.require("families", rule=rule)), dtype=object)
    bases = None
    magnitudes = None
    if count_power is not None:
        bases = np.asarray(list(rows.require("bases", rule=rule)), dtype=object)
        unknown = sorted({str(basis) for basis in bases.tolist()} - set(_BASES))
        if unknown:
            raise ValueError(
                f"{rule}: count_power needs every row's basis to be one of "
                f"{_BASES}; got {unknown}."
            )
        magnitudes = (
            np.maximum(np.abs(np.asarray(rows.values, dtype=np.float64)), 1.0)
            ** count_power
        )
    level_order = list(dict.fromkeys(levels.tolist()))
    weights = np.zeros(len(rows), dtype=np.float64)
    for level in level_order:
        in_level = levels == level
        family_order = list(dict.fromkeys(families[in_level].tolist()))
        budget = 1.0 / (len(level_order) * len(family_order))
        for family in family_order:
            cell = np.flatnonzero(in_level & (families == family))
            if bases is None:
                weights[cell] = budget / cell.size
                continue
            counts = cell[bases[cell] == COUNT_BASIS]
            amounts = cell[bases[cell] == AMOUNT_BASIS]
            weights[amounts] = budget / cell.size
            if counts.size:
                share = magnitudes[counts]
                weights[counts] = (
                    (budget * counts.size / cell.size) * share / share.sum()
                )
    return weights


def _sqrt_value_concept_budget(
    rows: TargetLossRows, *, value_power: float = 0.5
) -> np.ndarray:
    rule = "sqrt_value_concept_budget"
    basis_array = np.asarray(list(rows.require("bases", rule=rule)), dtype=object)
    weights = budget_group_weights(
        value_basis_weights(rows.values, basis_array, power=value_power),
        rows.require("groups", rule=rule),
    )
    unique_bases = sorted(set(basis_array.tolist()))
    if not unique_bases:
        return weights
    basis_total = len(weights) / len(unique_bases)
    for basis in unique_bases:
        mask = basis_array == basis
        current_total = weights[mask].sum()
        if current_total > 0:
            weights[mask] *= basis_total / current_total
    return weights / weights.mean()


@dataclass(frozen=True)
class _RuleSpec:
    function: Callable[..., np.ndarray]
    #: Parameter name -> whether ``None`` is admitted.
    parameters: Mapping[str, bool]


#: The closed vocabulary of weight rules, by name.
TARGET_WEIGHT_RULES: Mapping[str, _RuleSpec] = MappingProxyType(
    {
        "uniform": _RuleSpec(_uniform, MappingProxyType({})),
        "level_equal": _RuleSpec(_level_equal, MappingProxyType({})),
        "level_family_equal": _RuleSpec(
            _level_family_equal, MappingProxyType({"count_power": True})
        ),
        "sqrt_value_concept_budget": _RuleSpec(
            _sqrt_value_concept_budget, MappingProxyType({"value_power": False})
        ),
    }
)

_FAMILY_MULTIPLIERS = "family_multipliers"


@dataclass(frozen=True)
class TargetWeightRule:
    """A named weight rule with its parameters: what a country declares.

    ``parameters`` holds the rule's own parameters (``count_power`` for
    ``level_family_equal``, ``value_power`` for ``sqrt_value_concept_budget``)
    and, for any rule, ``family_multipliers`` (family -> positive finite
    multiplier). An unknown rule or parameter is refused when the declaration
    is made, not when it is used.
    """

    name: str
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        spec = TARGET_WEIGHT_RULES.get(self.name)
        if spec is None:
            raise ValueError(
                f"target weight rule must be one of {sorted(TARGET_WEIGHT_RULES)}, "
                f"got {self.name!r}."
            )
        parameters = dict(self.parameters)
        unknown = sorted(set(parameters) - set(spec.parameters) - {_FAMILY_MULTIPLIERS})
        if unknown:
            raise ValueError(
                f"target weight rule {self.name!r} takes "
                f"{sorted([*spec.parameters, _FAMILY_MULTIPLIERS])}, got {unknown}."
            )
        for key, nullable in spec.parameters.items():
            if key not in parameters:
                continue
            value = parameters[key]
            if value is None:
                if not nullable:
                    raise ValueError(f"{self.name}: {key} must be a number.")
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"{self.name}: {key} must be a positive finite number, "
                    f"got {value!r}."
                )
            parameters[key] = float(value)
        multipliers = parameters.get(_FAMILY_MULTIPLIERS)
        if multipliers is not None:
            checked: dict[str, float] = {}
            for family, multiplier in dict(multipliers).items():
                if (
                    isinstance(multiplier, bool)
                    or not isinstance(multiplier, int | float)
                    or not math.isfinite(multiplier)
                    or multiplier <= 0
                ):
                    raise ValueError(
                        f"{self.name}: the family multiplier for {family!r} must "
                        f"be positive and finite, got {multiplier!r}."
                    )
                checked[str(family)] = float(multiplier)
            parameters[_FAMILY_MULTIPLIERS] = MappingProxyType(checked)
        object.__setattr__(self, "parameters", MappingProxyType(parameters))

    def with_family_multipliers(
        self, family_multipliers: Mapping[str, float] | None
    ) -> TargetWeightRule:
        """This rule with ``family_multipliers`` in place of its own."""

        parameters = {
            key: value
            for key, value in self.parameters.items()
            if key != _FAMILY_MULTIPLIERS
        }
        if family_multipliers:
            parameters[_FAMILY_MULTIPLIERS] = dict(family_multipliers)
        return TargetWeightRule(self.name, parameters)

    def to_json(self) -> dict[str, object]:
        """The declaration as JSON-serializable data, for a receipt."""

        parameters: dict[str, object] = {}
        for key, value in sorted(self.parameters.items()):
            parameters[key] = (
                dict(sorted(value.items())) if isinstance(value, Mapping) else value
            )
        return {"rule": self.name, "parameters": parameters}


def target_loss_weights_for_rule(
    rows: TargetLossRows, rule: TargetWeightRule
) -> np.ndarray:
    """The loss-weight vector ``rule`` gives ``rows``, row-aligned (float64).

    The solver normalizes loss weights by their sum, so a rule's own scale
    (sum one or mean one) does not change the objective.
    """

    if not isinstance(rows, TargetLossRows):
        raise TypeError("target loss weights need a TargetLossRows carrier.")
    if not isinstance(rule, TargetWeightRule):
        raise TypeError("target loss weights need a TargetWeightRule declaration.")
    spec = TARGET_WEIGHT_RULES[rule.name]
    parameters = {
        key: value for key, value in rule.parameters.items() if key in spec.parameters
    }
    weights = spec.function(rows, **parameters)
    multipliers = rule.parameters.get(_FAMILY_MULTIPLIERS)
    if not multipliers:
        return weights
    return apply_family_multipliers(
        weights, rows.require("families", rule=rule.name), multipliers
    )


def target_loss_weight_shares(
    rows: TargetLossRows, weights: np.ndarray
) -> list[dict[str, object]]:
    """Each (level, family) cell's share of the loss weight, for a receipt.

    ``loss_share`` is the cell's share of the total weight (the share of the
    loss it carries at equal misses); ``equal_share`` is what it would carry
    with every target weighted equally. Cells come in first-row order.
    """

    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(rows),):
        raise ValueError(
            f"{weights.shape} loss weights do not align with {len(rows)} rows."
        )
    levels = np.asarray(
        list(rows.require("levels", rule="target_loss_weight_shares")), dtype=object
    )
    families = np.asarray(
        list(rows.require("families", rule="target_loss_weight_shares")),
        dtype=object,
    )
    total = float(weights.sum())
    n_rows = len(rows)
    cells = []
    for level, family in dict.fromkeys(
        zip(levels.tolist(), families.tolist(), strict=True)
    ):
        mask = (levels == level) & (families == family)
        cells.append(
            {
                "level": level,
                "family": family,
                "n_targets": int(mask.sum()),
                "loss_share": float(weights[mask].sum()) / total if total else 0.0,
                "equal_share": float(mask.sum()) / n_rows if n_rows else 0.0,
            }
        )
    return cells


# ---------------------------------------------------------------------------
# Loss caps
# ---------------------------------------------------------------------------


def _checked_cap(label: str, cap: object) -> float:
    if (
        isinstance(cap, bool)
        or not isinstance(cap, int | float)
        or not math.isfinite(cap)
        or cap <= 0
    ):
        raise ValueError(f"{label} must be positive and finite, got {cap!r}.")
    return float(cap)


def target_loss_caps(
    default: float,
    *,
    levels: Sequence[str] | None = None,
    families: Sequence[str | None] | None = None,
    by_level: Mapping[str, float] | None = None,
    by_family: Mapping[str, float] | None = None,
) -> float | np.ndarray:
    """The loss cap of each target: one number, or a vector when declared.

    With no override the result is ``default`` itself, a plain number, so a
    run that declares none calls the solver exactly as before. With
    ``by_level`` or ``by_family`` it is a row-aligned float64 vector: every
    row starts at ``default``, a row whose level is in ``by_level`` takes that
    cap, and a row whose family is in ``by_family`` takes that cap (a family
    cap wins over a level cap). An override that matches no row is an error,
    so a misspelled level cannot silently leave a cap unchanged.
    """

    default = _checked_cap("the default target loss cap", default)
    by_level = dict(by_level or {})
    by_family = dict(by_family or {})
    if not by_level and not by_family:
        return default
    columns = {"level": (levels, by_level), "family": (families, by_family)}
    n_rows = None
    for label, (column, overrides) in columns.items():
        if not overrides:
            continue
        if column is None:
            raise ValueError(f"a target loss cap by {label} needs each row's {label}.")
        if n_rows is not None and len(column) != n_rows:
            raise ValueError("target loss caps: levels and families must align.")
        n_rows = len(column)
    caps = np.full(n_rows, default, dtype=np.float64)
    for label, (column, overrides) in columns.items():
        if not overrides:
            continue
        labels = np.asarray(list(column), dtype=object)
        for key, cap in sorted(overrides.items()):
            mask = labels == key
            if not mask.any():
                raise ValueError(
                    f"the target loss cap for {label} {key!r} matches no target."
                )
            caps[mask] = _checked_cap(f"the target loss cap for {label} {key!r}", cap)
    return caps


@dataclass(frozen=True)
class SingleCapEquivalent:
    """A one-cap loss that equals a loss with a cap for each target.

    Pass ``cap``, ``weights`` and ``scales`` to the solver as
    ``target_loss_cap``, ``target_loss_weights`` and ``target_loss_scales``.
    ``caps`` keeps the declared row caps for the receipt.
    """

    cap: float
    weights: np.ndarray
    scales: np.ndarray
    caps: np.ndarray

    def receipt(self) -> dict[str, object]:
        """How many rows carry each declared cap, and the equivalent cap."""

        values, counts = np.unique(self.caps, return_counts=True)
        return {
            "method": "single_cap_equivalent",
            "equivalent_cap": float(self.cap),
            "rows_by_cap": {
                repr(float(value)): int(count)
                for value, count in zip(values, counts, strict=True)
            },
        }


def single_cap_equivalent(
    caps: Sequence[float] | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray,
    scales: Sequence[float] | np.ndarray,
) -> SingleCapEquivalent:
    """The one-cap loss with exactly the value of a cap for each target.

    The solver's loss is ``sum(w * min(|miss| / s, cap)) / sum(w)`` with one
    ``cap``. For row caps ``c``, weights ``w`` and scales ``s``, take

    - ``cap0 = sum(w * c) / sum(w)`` (the weighted mean cap),
    - ``w' = w * c / cap0`` and ``s' = s * c / cap0``.

    Then ``w' * min(|miss| / s', cap0) == w * min(|miss| / s, c)`` for every
    row, and ``sum(w') == sum(w)``. So the loss, each row's contribution to
    it and its gradient are those of the loss with row caps. This is an
    identity, not an approximation: a row past its own cap contributes the
    constant ``w * c`` in both forms. It needs no change to the solver, whose
    cap stays one number.

    When every row has the same cap, the weights and scales come back
    unchanged and ``cap`` is that number, so equal caps call the solver
    exactly as one cap does.

    The transformed scales and weights are what the solver records. A
    diagnostic that reads the solver's ``target_loss_cap`` against the default
    scales (the past-cap census) therefore measures against ``cap0``, not
    against each row's own cap.
    """

    caps = np.array(caps, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    scales = np.array(scales, dtype=np.float64)
    if caps.ndim != 1 or not (caps.shape == weights.shape == scales.shape):
        raise ValueError(
            "caps, weights and scales must be aligned vectors, got shapes "
            f"{caps.shape}, {weights.shape} and {scales.shape}."
        )
    if not caps.size:
        raise ValueError("a loss needs at least one target.")
    for label, values, strict in (
        ("caps", caps, True),
        ("scales", scales, True),
        ("weights", weights, False),
    ):
        if not np.isfinite(values).all() or (
            (values <= 0).any() if strict else (values < 0).any()
        ):
            raise ValueError(
                f"target loss {label} must be finite and "
                f"{'positive' if strict else 'non-negative'}."
            )
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("target loss weights must have positive total weight.")
    if (caps == caps[0]).all():
        return SingleCapEquivalent(
            cap=float(caps[0]), weights=weights, scales=scales, caps=caps
        )
    cap = float((weights * caps).sum() / total)
    factor = caps / cap
    return SingleCapEquivalent(
        cap=cap, weights=weights * factor, scales=scales * factor, caps=caps
    )
