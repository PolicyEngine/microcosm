"""Which rows carry each target: a diagnostic for any pooled build (#1729).

A pooled file stacks several sets of rows ("spines"): for example a national
release's households beside the households of a larger survey. A target that
one spine carries almost alone can only be hit by moving weight onto that
spine's rows, which concentrates the weights and lowers the effective sample
size. This module measures that, target by target:

- **share**: the part of the target's weighted total that each spine's rows
  carry, and
- **mass share**: the part of the weight that the spine holds among the
  records the target is about (its *scope*: an area, or every record).

``share`` far above ``mass share`` marks a target that pulls weight onto that
spine. ``excess`` is their difference and ``lift`` their ratio.

The diagnostic reads a constraint matrix (targets x records), a weight vector
and each record's spine. It solves nothing and changes nothing, so it runs on
a saved calibration input or checkpoint, at the starting weights or at any
calibrated weights. It is report-only.

Shares are taken on absolute contributions, ``sum(|a_ij| * w_i)``, so they
lie in ``[0, 1]`` and sum to one for every target. For a measure that is
never negative that is the share of the estimate itself; ``mixed_sign`` marks
the targets where it is not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

__all__ = [
    "SpineTargetShares",
    "spine_target_shares",
    "spine_target_shares_for_problem",
]

Scope = tuple[str, str]


@dataclass(frozen=True)
class SpineTargetShares:
    """The share of each target that each spine carries (see the module).

    Arrays are target-aligned on axis 0 and spine-aligned on axis 1.

    Attributes:
        names: The target names.
        spines: The spine labels, sorted.
        estimates: Each target's weighted total, ``A @ w``.
        carried: The signed part of the estimate each spine carries.
        share: Each spine's share of the target's absolute contributions;
            NaN for a target no positive-weight record carries.
        mass_share: Each spine's share of the weight in the target's scope.
        carriers: The positive-weight records of the spine with a nonzero
            coefficient, counted as distinct sources when source ids were
            given.
        mixed_sign: Whether the target has positive and negative
            contributions.
        scopes: Each target's scope ``(kind, code)``, or ``None`` for every
            record.
        overall_mass_share: Each spine's share of the total weight.
        withheld: Whether the target is withheld under ``minimum_carriers``.
            A withheld target's ``estimates``, ``carried`` and ``share`` are
            NaN and its ``carriers`` are ``-1``.
        minimum_carriers: The floor targets were withheld under, if any.
    """

    names: tuple[str, ...]
    spines: tuple[str, ...]
    estimates: np.ndarray
    carried: np.ndarray
    share: np.ndarray
    mass_share: np.ndarray
    carriers: np.ndarray
    mixed_sign: np.ndarray
    scopes: tuple[Scope | None, ...]
    overall_mass_share: np.ndarray
    withheld: np.ndarray
    minimum_carriers: int | None = None

    @property
    def excess(self) -> np.ndarray:
        """``share - mass_share``: positive where the spine over-carries."""

        return self.share - self.mass_share

    @property
    def lift(self) -> np.ndarray:
        """``share / mass_share``; NaN where the spine holds no mass in scope."""

        with np.errstate(divide="ignore", invalid="ignore"):
            lift = self.share / self.mass_share
        return np.where(self.mass_share > 0, lift, np.nan)

    def _cell(self, value: float) -> float | None:
        return None if np.isnan(value) else float(value)

    def rows(self) -> list[dict[str, object]]:
        """One JSON-ready record for each target, in target order."""

        excess, lift = self.excess, self.lift
        rows: list[dict[str, object]] = []
        for index, name in enumerate(self.names):
            scope = self.scopes[index]
            row: dict[str, object] = {
                "name": name,
                "scope": None
                if scope is None
                else {"kind": scope[0], "code": scope[1]},
                "mass_share": {
                    spine: self._cell(self.mass_share[index, column])
                    for column, spine in enumerate(self.spines)
                },
            }
            if self.withheld[index]:
                row["withheld"] = (
                    f"a spine carries it with fewer than {self.minimum_carriers} "
                    "distinct carriers"
                )
                rows.append(row)
                continue
            shares = self.share[index]
            dominant = None if np.isnan(shares).all() else int(np.nanargmax(shares))
            row.update(
                {
                    "estimate": float(self.estimates[index]),
                    "mixed_sign": bool(self.mixed_sign[index]),
                    "dominant_spine": None
                    if dominant is None
                    else self.spines[dominant],
                    "dominant_share": (
                        None if dominant is None else float(shares[dominant])
                    ),
                    "share": {},
                    "excess": {},
                    "lift": {},
                    "carried": {},
                    "carriers": {},
                }
            )
            for column, spine in enumerate(self.spines):
                row["share"][spine] = self._cell(shares[column])
                row["excess"][spine] = self._cell(excess[index, column])
                row["lift"][spine] = self._cell(lift[index, column])
                row["carried"][spine] = float(self.carried[index, column])
                row["carriers"][spine] = int(self.carriers[index, column])
            rows.append(row)
        return rows

    def ranking(self, spine: str, *, top: int | None = None) -> list[dict[str, object]]:
        """Targets by how much more of them ``spine`` carries than its mass.

        Sorted by ``excess`` for ``spine``, largest first: the targets that
        pull weight onto that spine come first. Withheld targets and targets
        no record carries are left out.
        """

        if spine not in self.spines:
            raise ValueError(f"spine must be one of {self.spines}, got {spine!r}.")
        column = self.spines.index(spine)
        excess = self.excess[:, column]
        lift = self.lift[:, column]
        order = [
            int(index)
            for index in np.argsort(-np.nan_to_num(excess, nan=-np.inf), kind="stable")
            if not np.isnan(excess[index])
        ]
        if top is not None:
            order = order[:top]
        return [
            {
                "name": self.names[index],
                "share": float(self.share[index, column]),
                "mass_share": float(self.mass_share[index, column]),
                "excess": float(excess[index]),
                "lift": self._cell(lift[index]),
                "carriers": int(self.carriers[index, column]),
            }
            for index in order
        ]

    def summary(
        self, groups: Sequence[str], *, alone_share: float = 0.9
    ) -> list[dict[str, object]]:
        """Shares summarized over groups of targets (a family, a level, ...).

        ``groups`` labels each target. For each group and spine: the median
        share, the mean excess, and how many targets the spine carries almost
        alone (``share >= alone_share``). Withheld and uncarried targets do
        not enter a group's statistics; ``n_targets`` counts them.
        """

        if len(groups) != len(self.names):
            raise ValueError(
                f"{len(groups)} group labels do not align with "
                f"{len(self.names)} targets."
            )
        if not 0.0 < alone_share <= 1.0:
            raise ValueError("alone_share must be in (0, 1].")
        labels = np.asarray([str(group) for group in groups], dtype=object)
        measured = ~np.isnan(self.share).any(axis=1)
        excess = self.excess
        summary = []
        for label in dict.fromkeys(labels.tolist()):
            members = labels == label
            scored = members & measured
            entry: dict[str, object] = {
                "group": label,
                "n_targets": int(members.sum()),
                "n_measured": int(scored.sum()),
                "spines": {},
            }
            for column, spine in enumerate(self.spines):
                entry["spines"][spine] = (
                    {
                        "median_share": float(np.median(self.share[scored, column])),
                        "mean_excess": float(excess[scored, column].mean()),
                        "n_carried_almost_alone": int(
                            (self.share[scored, column] >= alone_share).sum()
                        ),
                    }
                    if scored.any()
                    else None
                )
            summary.append(entry)
        return summary

    def frame(self) -> pd.DataFrame:
        """Long form: one row for each (target, spine)."""

        n_targets, n_spines = len(self.names), len(self.spines)
        return pd.DataFrame(
            {
                "target": np.repeat(np.asarray(self.names, dtype=object), n_spines),
                "spine": np.tile(np.asarray(self.spines, dtype=object), n_targets),
                "share": self.share.ravel(),
                "mass_share": self.mass_share.ravel(),
                "excess": self.excess.ravel(),
                "lift": self.lift.ravel(),
                "carried": self.carried.ravel(),
                "carriers": self.carriers.ravel(),
                "withheld": np.repeat(self.withheld, n_spines),
            }
        )

    def report(self) -> dict[str, object]:
        """The whole diagnostic as JSON-serializable data."""

        return {
            "spines": list(self.spines),
            "overall_mass_share": {
                spine: float(self.overall_mass_share[column])
                for column, spine in enumerate(self.spines)
            },
            "n_targets": len(self.names),
            "n_withheld": int(self.withheld.sum()),
            "minimum_carriers": self.minimum_carriers,
            "share_basis": "sum(|coefficient| * weight) by spine over all spines",
            "mass_share_basis": "weight by spine among the records of the target's scope",
            "targets": self.rows(),
        }


def _checked_vector(values, n_records: int, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (n_records,):
        raise ValueError(
            f"{label} has shape {array.shape}; the matrix has {n_records} records."
        )
    return array


def _distinct_carriers(
    nonzero: sparse.csr_array,
    positive: np.ndarray,
    spine_index: np.ndarray,
    n_spines: int,
    source_ids: np.ndarray | None,
) -> np.ndarray:
    """Distinct positive-weight carriers of each target in each spine."""

    n_records = nonzero.shape[1]
    carriers = np.zeros((nonzero.shape[0], n_spines), dtype=np.int64)
    for spine in range(n_spines):
        members = np.flatnonzero(positive & (spine_index == spine))
        if not len(members):
            continue
        if source_ids is None:
            keys = np.arange(len(members))
        else:
            _, keys = np.unique(source_ids[members], return_inverse=True)
        to_source = sparse.csr_array(
            (np.ones(len(members)), (members, keys)),
            shape=(n_records, int(keys.max()) + 1),
        )
        # Stored entries of each row are the distinct sources that carry it.
        by_source = (nonzero @ to_source).tocsr()
        by_source.eliminate_zeros()
        carriers[:, spine] = np.diff(by_source.indptr)
    return carriers


def spine_target_shares(
    matrix,
    weights: Sequence[float] | np.ndarray,
    spines: Sequence | np.ndarray,
    *,
    names: Sequence[str] | None = None,
    target_scopes: Sequence[Scope | None] | None = None,
    scope_codes: Mapping[str, Sequence | np.ndarray] | None = None,
    source_ids: Sequence | np.ndarray | None = None,
    minimum_carriers: int | None = None,
) -> SpineTargetShares:
    """The share of each target each spine carries, against its share of mass.

    Args:
        matrix: The constraint matrix, targets x records (SciPy sparse or
            dense).
        weights: One finite, non-negative weight for each record: the
            starting weights, or any calibrated weights.
        spines: Each record's spine label.
        names: The target names (default ``target_0``, ``target_1``, ...).
        target_scopes: For each target, the records it is about, as
            ``(kind, code)`` (for example ``("state", "06")``), or ``None``
            for every record. Default: every record, for every target.
        scope_codes: For each scope kind, each record's code of that kind.
            Codes are compared as strings.
        source_ids: Each record's source household, to count a household's
            copies and clones once among a target's carriers. Two spines may
            reuse an id; carriers are counted within a spine.
        minimum_carriers: Withhold a target when a spine carries it with at
            least one and fewer than this many distinct carriers. A build
            under a disclosure rule passes its floor. A withheld target
            reports its scope and mass shares only, since its shares sum to
            one and one hidden cell could be read from the others.

    Returns:
        A :class:`SpineTargetShares`.
    """

    constraint = sparse.csr_array(matrix)
    n_targets, n_records = constraint.shape
    weight = np.asarray(weights, dtype=np.float64)
    if weight.shape != (n_records,):
        raise ValueError(
            f"weights have shape {weight.shape}; the matrix has {n_records} records."
        )
    if not np.isfinite(weight).all() or (weight < 0.0).any():
        raise ValueError("weights must be finite and non-negative.")
    total = float(weight.sum())
    if total <= 0.0:
        raise ValueError("weights must have a positive total.")
    labels = _checked_vector(spines, n_records, "spines").astype(str)
    spine_order, spine_index = np.unique(labels, return_inverse=True)
    n_spines = len(spine_order)
    if names is None:
        names = [f"target_{index}" for index in range(n_targets)]
    names = tuple(str(name) for name in names)
    if len(names) != n_targets:
        raise ValueError(f"{len(names)} names for {n_targets} targets.")
    if minimum_carriers is not None and (
        isinstance(minimum_carriers, bool)
        or not isinstance(minimum_carriers, int)
        or minimum_carriers < 1
    ):
        raise ValueError("minimum_carriers must be a positive integer.")
    sources = (
        None
        if source_ids is None
        else _checked_vector(source_ids, n_records, "source_ids")
    )

    records = np.arange(n_records)
    weight_by_spine = sparse.csr_array(
        (weight, (records, spine_index)), shape=(n_records, n_spines)
    )
    constraint = constraint.astype(np.float64)
    carried = (constraint @ weight_by_spine).toarray()
    absolute = (abs(constraint) @ weight_by_spine).toarray()
    row_total = absolute.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(row_total[:, None] > 0, absolute / row_total[:, None], np.nan)
    estimates = np.asarray(constraint @ weight, dtype=np.float64)
    mixed_sign = (np.asarray(constraint.maximum(0) @ weight) > 0) & (
        np.asarray(constraint.minimum(0) @ weight) < 0
    )

    nonzero = constraint.copy()
    nonzero.data = (nonzero.data != 0).astype(np.float64)
    nonzero.eliminate_zeros()
    carriers = _distinct_carriers(nonzero, weight > 0, spine_index, n_spines, sources)

    overall = np.asarray(weight_by_spine.sum(axis=0), dtype=np.float64).ravel() / total
    mass_share = np.tile(overall, (n_targets, 1))
    scopes: tuple[Scope | None, ...] = (None,) * n_targets
    if target_scopes is not None:
        if len(target_scopes) != n_targets:
            raise ValueError(f"{len(target_scopes)} scopes for {n_targets} targets.")
        scopes = tuple(
            None if scope is None else (str(scope[0]), str(scope[1]))
            for scope in target_scopes
        )
        kinds = {scope[0] for scope in scopes if scope is not None}
        missing = sorted(kinds - set(scope_codes or {}))
        if missing:
            raise ValueError(f"scope_codes has no record codes for {missing}.")
        mass_by_scope: dict[str, dict[str, np.ndarray]] = {}
        for kind in kinds:
            codes = _checked_vector(
                scope_codes[kind], n_records, f"scope_codes[{kind!r}]"
            )
            grouped = (
                pd.DataFrame(
                    {"code": codes.astype(str), "spine": spine_index, "weight": weight}
                )
                .groupby(["code", "spine"], sort=True)["weight"]
                .sum()
                .unstack("spine", fill_value=0.0)
                .reindex(columns=range(n_spines), fill_value=0.0)
            )
            mass_by_scope[kind] = {
                str(code): row.to_numpy(dtype=np.float64)
                for code, row in grouped.iterrows()
            }
        for index, scope in enumerate(scopes):
            if scope is None:
                continue
            mass = mass_by_scope[scope[0]].get(scope[1])
            mass_total = 0.0 if mass is None else float(mass.sum())
            mass_share[index] = mass / mass_total if mass_total > 0 else np.nan

    withheld = np.zeros(n_targets, dtype=bool)
    if minimum_carriers is not None:
        withheld = ((carriers > 0) & (carriers < minimum_carriers)).any(axis=1)
        estimates = np.where(withheld, np.nan, estimates)
        carried = np.where(withheld[:, None], np.nan, carried)
        share = np.where(withheld[:, None], np.nan, share)
        carriers = np.where(withheld[:, None], -1, carriers)
    return SpineTargetShares(
        names=names,
        spines=tuple(str(spine) for spine in spine_order),
        estimates=estimates,
        carried=carried,
        share=share,
        mass_share=mass_share,
        carriers=carriers,
        mixed_sign=mixed_sign,
        scopes=scopes,
        overall_mass_share=overall,
        withheld=withheld,
        minimum_carriers=minimum_carriers,
    )


def spine_target_shares_for_problem(
    problem,
    spines: Sequence | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    **options,
) -> SpineTargetShares:
    """:func:`spine_target_shares` for a compiled ``CalibrationProblem``.

    Reads the problem's matrix and target names. ``weights`` defaults to the
    problem's initial weights, so a saved calibration input needs no solve.
    """

    if weights is None:
        weights = problem.initial_weights.values
    return spine_target_shares(
        problem.matrix,
        weights,
        spines,
        names=options.pop("names", problem.names),
        **options,
    )
