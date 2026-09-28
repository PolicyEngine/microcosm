"""Support-mix bake-off primitives: CPS years and ACS rows at fixed row budgets.

The US calibration support can grow two ways. Pooling more CPS ASEC years adds
households at full CPS richness; adding ACS households adds real variation in
what the ACS observes (housing, local geography, composition) and only model
draws in what it does not (the CPS-only variables imputed onto ACS rows). A
third way, cloning CPS households to new locations, adds degrees of freedom
for geography alone. This module holds the pure, engine-free pieces of a
bake-off that compares those mixes at a fixed household budget:

- arm planning: how many CPS source households, ACS households and CPS
  location clones an arm holds (:func:`plan_arm_counts`);
- nested, identity-keyed ACS selection (:func:`acs_selection_order`), so a
  smaller arm's ACS households are always a subset of a larger arm's;
- deterministic clone allocation (:func:`clone_copies`);
- the arm's starting weights (:func:`arm_initial_weights`);
- the US target split mirrored from the pending ``target_split`` module
  (:func:`target_split_group_key`, :func:`target_split_role`);
- effective sample size over distinct households (:func:`distinct_unit_ess`).

A "source household" is one survey household record: one ASEC household-year
or one ACS household. A CPS source household keeps its PUF tax-detail clone
rows (the Route A support channels), so it can span two or three physical
rows; every physical row of one source household shares one unit key.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "HOLDOUT_FRACTION",
    "SEALED_FRACTION",
    "TARGET_SPLIT_HOLDOUT_SALT",
    "TARGET_SPLIT_SEALED_SALT",
    "ArmCounts",
    "SupportMixArm",
    "acs_selection_order",
    "arm_initial_weights",
    "assemble_target_matrix",
    "clone_copies",
    "distinct_unit_ess",
    "hash_uniform",
    "kish_ess",
    "plan_arm_counts",
    "target_split_group_key",
    "target_split_role",
]

#: The release split's salts and fractions, copied from the unmerged
#: ``us-target-roles-holdout`` branch (microcosm.build.us_runtime.target_split)
#: as of 2026-09-28. Its pins, which force whole groups to train, are not
#: mirrored: they were not yet written. Swap this block for that module once
#: it merges.
TARGET_SPLIT_SEALED_SALT = "microcosm.us.target_split.v1.sealed"
TARGET_SPLIT_HOLDOUT_SALT = "microcosm.us.target_split.v1.holdout"
SEALED_FRACTION = 0.05
HOLDOUT_FRACTION = 0.10

_HASH_UNIFORM_SCALE = float(2**64)
_VINTAGE_TOKEN = re.compile(r"^(?:(?:ty|cy|fy|v|oep)\d{4}|month\d{4}_\d{1,2})$")
_HEX16_TOKEN = re.compile(r"^[0-9a-f]{16}$")


def hash_uniform(key: str, *, salt: str) -> float:
    """A key's deterministic uniform draw on ``[0, 1)``.

    ``sha256(salt + "\\x1f" + key)``, first 8 bytes big-endian, divided by
    ``2**64`` — byte-identical to the pending
    ``microcosm.build.holdout.hash_holdout_uniform``. The draw depends only on
    the key and the salt, never on which other keys exist or their order.
    """
    if not isinstance(key, str) or not key:
        raise ValueError(f"hash key must be a non-empty string, got {key!r}.")
    if not isinstance(salt, str) or not salt:
        raise ValueError(f"hash salt must be a non-empty string, got {salt!r}.")
    digest = hashlib.sha256(f"{salt}\x1f{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / _HASH_UNIFORM_SCALE


def target_split_group_key(name: str, hierarchy_parent: str | None = None) -> str:
    """The unit a target's split role is drawn for.

    A congressional-district child takes its state parent's key, so a
    district is never held out while its parent and trained siblings pin it.
    The name is split on ``"."``; vintage tokens (``ty2023``, ``v2024``,
    ``month2024_7``, ...) are dropped so a restated vintage keeps its role;
    names with a ``current_cd`` token also drop a trailing 16-hex digest.
    Bare 4-digit tokens are district GEOIDs and are always kept.
    """
    source = hierarchy_parent if hierarchy_parent else name
    if not isinstance(source, str) or not source:
        raise ValueError(f"target name must be a non-empty string, got {source!r}.")
    tokens = [token for token in source.split(".") if not _VINTAGE_TOKEN.match(token)]
    if "current_cd" in tokens and tokens and _HEX16_TOKEN.match(tokens[-1]):
        tokens = tokens[:-1]
    return ".".join(tokens)


def target_split_role(group_key: str) -> str:
    """``"sealed"``, ``"holdout"`` or ``"train"`` for one group key.

    Sealed first, on its own salt, so the sealed tier rotates independently
    of the holdout; then holdout on the holdout salt. About 5% sealed, 9.5%
    holdout, 85.5% train.
    """
    if hash_uniform(group_key, salt=TARGET_SPLIT_SEALED_SALT) < SEALED_FRACTION:
        return "sealed"
    if hash_uniform(group_key, salt=TARGET_SPLIT_HOLDOUT_SALT) < HOLDOUT_FRACTION:
        return "holdout"
    return "train"


@dataclass(frozen=True)
class SupportMixArm:
    """One bake-off arm.

    Attributes:
        budget: Source households in the arm, or ``None`` for a CPS-only arm
            at its natural size (no fill).
        cps_income_years: ASEC income years whose households are all kept.
        acs_fill_share: Share of the fill (``budget`` minus the CPS source
            households) taken by ACS households; the rest are CPS location
            clones. ``1.0`` fills with ACS only, ``0.0`` with clones only.
        seed: Replicate index. The bake-off driver takes the ``seed``-th
            disjoint block of one fixed ACS order (so seed 0 arms are nested
            across budgets and replicates are independent draws) and salts
            the clone draws with it.
    """

    budget: int | None
    cps_income_years: tuple[int, ...]
    acs_fill_share: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.budget is not None and (
            isinstance(self.budget, bool) or int(self.budget) <= 0
        ):
            raise ValueError(f"budget must be positive or None, got {self.budget!r}.")
        years = tuple(int(year) for year in self.cps_income_years)
        if len(set(years)) != len(years):
            raise ValueError(f"cps_income_years repeats a year: {years}.")
        object.__setattr__(self, "cps_income_years", tuple(sorted(years)))
        share = float(self.acs_fill_share)
        if not (math.isfinite(share) and 0.0 <= share <= 1.0):
            raise ValueError(f"acs_fill_share must be in [0, 1], got {share!r}.")
        object.__setattr__(self, "acs_fill_share", share)

    @property
    def label(self) -> str:
        budget = "natural" if self.budget is None else f"b{self.budget // 1000}k"
        years = (
            "cps" + "-".join(str(year) for year in self.cps_income_years)
            if self.cps_income_years
            else "cps0"
        )
        return f"{budget}.{years}.acs{round(self.acs_fill_share * 100):03d}.s{self.seed}"


@dataclass(frozen=True)
class ArmCounts:
    """Source-household counts of one arm; they sum to its budget."""

    cps: int
    acs: int
    clones: int

    @property
    def total(self) -> int:
        return self.cps + self.acs + self.clones


def plan_arm_counts(
    arm: SupportMixArm,
    *,
    cps_households_by_year: Mapping[int, int],
    acs_households: int,
) -> ArmCounts:
    """How many CPS, ACS and clone source households ``arm`` holds.

    Raises:
        ValueError: If a year has no CPS households, the CPS years alone
            exceed the budget, clones are requested without CPS households,
            or the ACS fill exceeds the ACS households available.
    """
    missing = [year for year in arm.cps_income_years if year not in cps_households_by_year]
    if missing:
        raise ValueError(f"no CPS households for income year(s) {missing}.")
    cps = int(sum(int(cps_households_by_year[year]) for year in arm.cps_income_years))
    if arm.budget is None:
        return ArmCounts(cps=cps, acs=0, clones=0)
    fill = int(arm.budget) - cps
    if fill < 0:
        raise ValueError(
            f"{arm.label}: {cps:,} CPS source households exceed the "
            f"{arm.budget:,} budget."
        )
    acs = int(round(fill * arm.acs_fill_share))
    clones = fill - acs
    if clones and not cps:
        raise ValueError(f"{arm.label}: clones need at least one CPS household.")
    if acs > int(acs_households):
        raise ValueError(
            f"{arm.label}: the ACS fill of {acs:,} exceeds the "
            f"{int(acs_households):,} ACS households available."
        )
    return ArmCounts(cps=cps, acs=acs, clones=clones)


def acs_selection_order(unit_keys: Sequence[str], *, salt: str) -> np.ndarray:
    """Positions of ``unit_keys`` in selection order.

    An arm with ``n`` ACS households takes the first ``n`` positions, so
    selections are nested across budgets: every household in a smaller arm is
    in every larger arm with the same salt. Ties (identical draws) break by
    key, so the order is a pure function of the key set.
    """
    keys = [str(key) for key in unit_keys]
    draws = np.fromiter(
        (hash_uniform(key, salt=salt) for key in keys), dtype=np.float64, count=len(keys)
    )
    return np.lexsort((np.asarray(keys, dtype=object).astype(str), draws))


def clone_copies(unit_keys: Sequence[str], n_clones: int, *, salt: str) -> np.ndarray:
    """Extra location copies per CPS source household, summing to ``n_clones``.

    Every household gets ``n_clones // n`` copies and the remainder goes to
    the households first in hash order, so no household carries more than one
    copy above any other.
    """
    n = len(unit_keys)
    if n_clones < 0:
        raise ValueError(f"n_clones must be non-negative, got {n_clones!r}.")
    if n == 0:
        if n_clones:
            raise ValueError("clones need at least one CPS household.")
        return np.zeros(0, dtype=np.int64)
    copies = np.full(n, n_clones // n, dtype=np.int64)
    remainder = n_clones - int(copies.sum())
    if remainder:
        order = acs_selection_order(unit_keys, salt=salt)
        copies[order[:remainder]] += 1
    return copies


def arm_initial_weights(
    *,
    cps_design_weights: np.ndarray,
    cps_income_year: np.ndarray,
    cps_copies: np.ndarray,
    acs_design_weights: np.ndarray,
    counts: ArmCounts,
    total_mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Starting weights for an arm's CPS rows (with copies) and ACS rows.

    - Each CPS income year gets an equal share of the CPS mass, keeping the
      within-year design weights' shape.
    - A CPS row with ``k`` copies splits its weight over ``1 + k`` rows.
    - The ACS rows keep their design weights' shape.
    - The CPS side (source households plus clones) and the ACS side share
      ``total_mass`` in proportion to their source-household counts, so the
      mean starting weight per source household is the same on both sides.

    Args:
        cps_design_weights: Design weight per selected CPS physical row.
        cps_income_year: Income year per selected CPS physical row.
        cps_copies: Extra copies per selected CPS physical row (every row of
            one source household carries that household's count).
        acs_design_weights: Design weight per selected ACS row.
        counts: The arm's source-household counts.
        total_mass: The household weight the arm's weights sum to.

    Returns:
        ``(cps_row_weights, acs_row_weights)``: the CPS array is per physical
        row *per copy* (divide-by-``1 + k`` already applied), to be repeated
        ``1 + k`` times by the caller.
    """
    cps_w = np.asarray(cps_design_weights, dtype=np.float64)
    years = np.asarray(cps_income_year)
    copies = np.asarray(cps_copies, dtype=np.int64)
    acs_w = np.asarray(acs_design_weights, dtype=np.float64)
    if cps_w.shape != years.shape or cps_w.shape != copies.shape:
        raise ValueError("CPS weights, years and copies must align.")
    for name, values in (("CPS", cps_w), ("ACS", acs_w)):
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{name} design weights must be finite and positive.")
    if (copies < 0).any():
        raise ValueError("copies must be non-negative.")
    if not (math.isfinite(total_mass) and total_mass > 0):
        raise ValueError(f"total_mass must be positive, got {total_mass!r}.")
    source = counts.total
    if source <= 0:
        raise ValueError("an arm needs at least one source household.")
    cps_share = (counts.cps + counts.clones) / source
    acs_share = counts.acs / source
    if (counts.cps + counts.clones) and not len(cps_w):
        raise ValueError("counts declare CPS households but no CPS rows were given.")
    if counts.acs and not len(acs_w):
        raise ValueError("counts declare ACS households but no ACS rows were given.")

    cps_out = np.zeros_like(cps_w)
    if len(cps_w):
        unique_years = np.unique(years)
        per_year_mass = cps_share * total_mass / len(unique_years)
        for year in unique_years:
            in_year = years == year
            cps_out[in_year] = cps_w[in_year] * (per_year_mass / cps_w[in_year].sum())
        cps_out = cps_out / (1 + copies)
    acs_out = (
        acs_w * (acs_share * total_mass / acs_w.sum()) if len(acs_w) else acs_w.copy()
    )
    return cps_out, acs_out


def kish_ess(weights: np.ndarray) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2`` over rows."""
    w = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be finite and non-negative.")
    denominator = float(np.square(w).sum())
    return 0.0 if denominator == 0.0 else float(w.sum() ** 2 / denominator)


def distinct_unit_ess(weights: np.ndarray, unit_codes: Iterable) -> float:
    """Kish ESS after summing weights within each distinct unit.

    A repeated household (the same ASEC housing unit in two rotation years, a
    PUF clone row, a location clone) is one unit, so its rows count once.
    Never exceeds the row-level Kish ESS nor the number of distinct units.
    """
    w = np.asarray(weights, dtype=np.float64)
    codes = np.asarray(list(unit_codes) if not isinstance(unit_codes, np.ndarray) else unit_codes)
    if codes.shape != w.shape:
        raise ValueError("weights and unit codes must align.")
    _, inverse = np.unique(codes, return_inverse=True)
    unit_weight = np.bincount(inverse.ravel(), weights=w)
    return kish_ess(unit_weight)


def assemble_target_matrix(
    values,
    row_geography: Mapping[str, np.ndarray],
    target_concept: np.ndarray,
    target_level: Sequence[str],
    target_geography: np.ndarray,
):
    """Targets x rows: ``A[t, i] = values[i, concept(t)]`` where row ``i`` is
    in target ``t``'s geography, else 0.

    ``values`` is a rows x concepts sparse matrix. ``row_geography`` maps each
    level (``"state"``, ``"cd"``) to one integer code per row; ``"national"``
    needs none. Several targets may share a (concept, level, geography) cell
    (the same concept published by two sources); each receives the cell's
    values.
    """
    import scipy.sparse as sp

    coo = sp.coo_matrix(values)
    concept = np.asarray(target_concept, dtype=np.int64)
    level = np.asarray(target_level, dtype=object)
    geography = np.asarray(target_geography, dtype=np.int64)
    n_targets, n_rows = len(concept), coo.shape[0]
    rows_parts, cols_parts, data_parts = [], [], []
    for name in np.unique(level):
        sel = np.flatnonzero(level == name)
        if name == "national":
            row_geo = np.zeros(n_rows, dtype=np.int64)
            target_geo = np.zeros(len(sel), dtype=np.int64)
        else:
            row_geo = np.asarray(row_geography[name], dtype=np.int64)
            target_geo = geography[sel]
        span = int(max(row_geo.max(initial=0), target_geo.max(initial=0))) + 1
        cell = concept[sel] * span + target_geo
        order = np.argsort(cell, kind="stable")
        cells_sorted, targets_sorted = cell[order], sel[order]
        entry = coo.col.astype(np.int64) * span + row_geo[coo.row]
        low = np.searchsorted(cells_sorted, entry, "left")
        count = np.searchsorted(cells_sorted, entry, "right") - low
        keep = count > 0
        repeat = count[keep]
        starts = np.repeat(low[keep], repeat)
        offsets = np.arange(int(repeat.sum())) - np.repeat(np.cumsum(repeat) - repeat, repeat)
        rows_parts.append(targets_sorted[starts + offsets])
        cols_parts.append(np.repeat(coo.row[keep], repeat))
        data_parts.append(np.repeat(coo.data[keep], repeat))
    if not rows_parts:
        return sp.csr_matrix((n_targets, n_rows), dtype=np.float64)
    return sp.csr_matrix(
        (np.concatenate(data_parts), (np.concatenate(rows_parts), np.concatenate(cols_parts))),
        shape=(n_targets, n_rows),
    )
