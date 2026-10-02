"""Spread each normalized UK household's BRMA draw across its local clones.

The household distribution is the mean of its benefit units' census cells,
the exact marginal of ``frs_brma``'s unit draws and uniform household pick.
For clone c of K, invert that distribution at ``(u_h + c / K) mod 1`` in
increasing household-mean LHA-rate order. The uniform is keyed to lineage
before geographic expansion, including the SPI and CGT branches.

Invariants:
1. Exact marginals: over uniform u_h, every clone has the household's BRMA
   distribution, for any rate order.
2. Support: only positive-probability BRMAs in that household's region occur.
3. Spacing: K clones have K evenly spaced quantiles. For an outcome monotone
   in the walk order, their mean differs from its expectation by at most
   the outcome's range divided by K.
4. Determinism: the same lineage keys give the same draws regardless of row
   order; source-support branches have distinct keys.
5. Identity: K=1 retains the household distribution and supported clone 0.
6. Conservation: assignment returns only BRMA values; it changes no other
   column, weight, or row count and never mutates its input frame or table.

This ports the quantile logic from policyengine-uk-data without importing it.
The preparation uses one engine materialization per BRMA slot, avoiding a
frame stacked across all slots; the probability tables still use O(H * W)
memory. Applying the prepared distribution
after geographic expansion requires no country engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.frs_brma import (
    _benunit_household_map,
    _enum_name,
    load_brma_count_resource,
)
from microcosm.build.uk_runtime.national_frame import uk_time_period
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country


def _benunit_household_indices(frame: Frame) -> np.ndarray:
    household = frame.table("household")
    placements = _benunit_household_map(frame.table("person"))
    positions = pd.Series(np.arange(len(household)), index=household["household_id"])
    indices = frame.table("benunit")["benunit_id"].map(placements).map(positions)
    if indices.isna().any():
        raise ValueError("Every benefit unit needs a household for BRMA spreading.")
    return indices.to_numpy(dtype=np.int64)


def household_brma_probabilities(
    frame: Frame,
    *,
    lha_category: Sequence[object],
    count_resource: Mapping[str, Any],
) -> tuple[dict[str, list[str]], np.ndarray]:
    """Return regional alphabetical BRMAs and household-by-BRMA probabilities.

    Cells are the committed ``frs_brma`` census resource. Each unit's cell is
    normalized separately before averaging over its household's units. Rows
    are aligned to the household table, columns to its region's BRMA list;
    columns beyond the region's list have probability zero.
    """
    categories = np.asarray([_enum_name(value) for value in lha_category])
    if categories.shape != (frame.n("benunit"),):
        raise ValueError("LHA_category materialization must align to benunit rows.")
    cells = count_resource["cells"]
    brmas: dict[str, list[str]] = {}
    for region, region_cells in cells.items():
        names = set()
        for counts in region_cells.values():
            for name, count in counts.items():
                count = float(count)
                if not np.isfinite(count) or count < 0:
                    raise ValueError("BRMA counts must be finite and non-negative.")
                if count > 0:
                    names.add(str(name))
        if names:
            brmas[str(region)] = sorted(names)
    if not brmas:
        raise ValueError("BRMA count resource has no positive counts.")
    width = max(map(len, brmas.values()))
    household = frame.table("household")
    regions = np.asarray([_enum_name(value) for value in household["region"]])
    indices = _benunit_household_indices(frame)
    units = np.bincount(indices, minlength=len(household))
    if (units == 0).any():
        raise ValueError("Every household needs at least one benefit unit.")
    probabilities = np.zeros((len(household), width), dtype=float)
    unit_regions = regions[indices]
    for region, category in sorted(set(zip(unit_regions, categories, strict=True))):
        try:
            counts = cells[region][category]
        except KeyError as exc:
            raise KeyError(
                f"missing BRMA count-table cell for region={region!r}, "
                f"LHA_category={category!r}."
            ) from exc
        total = sum(float(count) for count in counts.values())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("BRMA count-table cell has no positive counts.")
        p = np.zeros(width)
        names = brmas[region]
        p[: len(names)] = [float(counts.get(name, 0)) / total for name in names]
        mask = (unit_regions == region) & (categories == category)
        np.add.at(probabilities, indices[mask], p)
    return brmas, probabilities / units[:, None]


def lha_rate_keys(
    frame: Frame,
    *,
    engine: object,
    brmas: Mapping[str, Sequence[str]],
    benunit_household: np.ndarray | None = None,
) -> np.ndarray:
    """Materialize each household's mean benefit-unit LHA rate per BRMA.

    Use ``uncapped_BRMA_LHA_rate`` where the engine defines it, otherwise
    ``BRMA_LHA_rate``. Engine ``variables()`` lists inputs only, so inspect
    variable metadata to detect this computed variable. Padded slots get inf.
    """
    assert_rules_engine_country(engine, "uk")
    variable = "uncapped_BRMA_LHA_rate"
    try:
        metadata = engine.variable_metadata(variable)
    except (KeyError, ValueError):
        variable = "BRMA_LHA_rate"
        metadata = engine.variable_metadata(variable)
    if metadata.entity != "benunit":
        raise ValueError(f"{variable} must be a benefit-unit variable.")
    household = frame.table("household")
    regions = np.asarray([_enum_name(value) for value in household["region"]])
    indices = (
        _benunit_household_indices(frame)
        if benunit_household is None
        else np.asarray(benunit_household)
    )
    if (
        indices.shape != (frame.n("benunit"),)
        or indices.dtype.kind not in "iu"
        or (indices < 0).any()
        or (indices >= len(household)).any()
    ):
        raise ValueError("Benefit-unit household indices must align to the frame.")
    units = np.bincount(indices, minlength=len(household))
    if (units == 0).any():
        raise ValueError("Every household needs at least one benefit unit.")
    if not brmas or any(not names for names in brmas.values()):
        raise ValueError("Every BRMA region needs a non-empty BRMA list.")
    try:
        count = np.asarray([len(brmas[region]) for region in regions])
    except KeyError as exc:
        raise ValueError("A household's region has no BRMA support.") from exc
    width = max(map(len, brmas.values()))
    keys = np.full((len(household), width), np.inf)
    for slot in range(width):
        moved = household.copy()
        moved["brma"] = [
            brmas[region][min(slot, size - 1)]
            for region, size in zip(regions, count, strict=True)
        ]
        # A fresh carrier preserves every stored input, weight and receipt.
        # Each materialize call starts its own simulation, so cached LHA
        # calculations from another BRMA cannot contaminate this slot.
        tables = {entity: frame.table(entity) for entity in frame.entities}
        tables.update({link: frame.link(link) for link in frame.links})
        tables["household"] = moved
        candidate = Frame(
            tables=tables,
            schema=frame.schema,
            weights={
                entity: frame.weights_for(entity) for entity in frame.weighted_entities
            },
            strata=frame.strata,
            mass_log=frame.mass_log,
            metadata=frame.metadata,
        )
        rate = np.asarray(
            engine.materialize(candidate, [variable], uk_time_period(frame))[variable],
            dtype=float,
        )
        if rate.shape != (frame.n("benunit"),) or not np.isfinite(rate).all():
            raise ValueError(
                f"{variable} must give finite rates aligned to benefit units."
            )
        mean = np.bincount(indices, weights=rate, minlength=len(household)) / units
        keys[:, slot] = np.where(slot < count, mean, np.inf)
        del candidate
    return keys


def _ordered_cdf(
    probabilities: np.ndarray, keys: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] == 0
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("Probabilities must be finite non-negative rows summing to 1.")
    rows, width = probabilities.shape
    if keys is None:
        order = np.broadcast_to(np.arange(width), probabilities.shape)
    else:
        keys = np.asarray(keys, dtype=float)
        if (
            keys.shape != probabilities.shape
            or not np.isfinite(keys[probabilities > 0]).all()
        ):
            raise ValueError("Keys must align and be finite on positive BRMA support.")
        order = np.argsort(
            np.where(probabilities > 0, keys, np.inf), axis=1, kind="stable"
        )
    p = np.take_along_axis(probabilities, order, axis=1)
    cumulative = np.cumsum(p, axis=1)
    last = width - 1 - np.argmax(p[:, ::-1] > 0, axis=1)
    cumulative[np.arange(width)[None, :] >= last[:, None]] = 1.0
    return order, cumulative


def _validate_quantiles(quantiles: np.ndarray) -> np.ndarray:
    quantiles = np.asarray(quantiles, dtype=float)
    if (
        quantiles.ndim != 1
        or not np.isfinite(quantiles).all()
        or np.any((quantiles < 0) | (quantiles >= 1))
    ):
        raise ValueError("Quantiles must be a vector in [0, 1).")
    return quantiles


def brmas_at_quantiles(
    probabilities: np.ndarray, keys: np.ndarray | None, quantiles: np.ndarray
) -> np.ndarray:
    """Return each row's positive-probability column at its quantile.

    Walk ascending keys, breaking ties by the alphabetical input column.
    The CDF is exactly 1 from the last positive column onwards, preventing
    floating-point rounding from selecting a zero-probability padded column.
    """
    quantiles = _validate_quantiles(quantiles)
    order, cumulative = _ordered_cdf(probabilities, keys)
    if quantiles.shape != (len(order),):
        raise ValueError("Quantiles must align to probability rows.")
    position = (cumulative <= quantiles[:, None]).sum(axis=1)
    return order[np.arange(len(order)), position]


def spaced_quantiles(uniforms: np.ndarray, k: int) -> np.ndarray:
    """Return ``(u_h + c / k) mod 1`` for each household and clone c."""
    if not isinstance(k, Integral) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")
    uniforms = _validate_quantiles(uniforms)
    return (uniforms[:, None] + np.arange(k)[None, :] / k) % 1.0


def prepare_brma_spread(
    frame: Frame,
    *,
    engine: object,
    lineage_keys: Sequence[str],
    count_resource: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare JSON-safe distributions before geographic expansion.

    The host supplies pre-geographic lineage keys from ``household_draw_key``;
    distinct SPI/CGT rows must have distinct keys, even with a shared source
    household. Names in zero-probability padding repeat a regional BRMA and
    cannot be selected. No engine work remains when applying this payload.
    """
    assert_rules_engine_country(engine, "uk")
    if len(lineage_keys) != frame.n("household") or any(
        not isinstance(key, str) or not key for key in lineage_keys
    ):
        raise ValueError(
            "Lineage keys must be non-empty strings aligned to households."
        )
    if len(set(lineage_keys)) != len(lineage_keys):
        raise ValueError("Every normalized household needs a distinct lineage key.")
    resource = load_brma_count_resource() if count_resource is None else count_resource
    category = engine.materialize(frame, ["LHA_category"], uk_time_period(frame))[
        "LHA_category"
    ]
    brmas, probabilities = household_brma_probabilities(
        frame, lha_category=category, count_resource=resource
    )
    keys = lha_rate_keys(frame, engine=engine, brmas=brmas)
    order, cumulative = _ordered_cdf(probabilities, keys)
    household = frame.table("household")
    width = probabilities.shape[1]
    names = np.asarray(
        [
            brmas[_enum_name(region)]
            + [brmas[_enum_name(region)][-1]] * (width - len(brmas[_enum_name(region)]))
            for region in household["region"]
        ],
        dtype=object,
    )
    return {
        "household_ids": household["household_id"].to_list(),
        "ordered_brmas": np.take_along_axis(names, order, axis=1).tolist(),
        "cumulative_probabilities": cumulative.tolist(),
        "uniforms": stable_identity_uniforms(
            lineage_keys, seed=0, salt="brma:clone_spread"
        ).tolist(),
    }


def assign_clone_brmas(
    household: pd.DataFrame,
    payload: Mapping[str, Any],
    *,
    n_clones: int,
    id_multiplier: int,
    clone_index_column: str,
) -> np.ndarray:
    """Return clone-aligned BRMAs, preserving the table and every other value.

    Geographic expansion owns ``household_id = pre_id + c * id_multiplier``
    and the clone index column. Recover its parent by that exact integer
    relationship, without assuming any row ordering. Process one clone at a
    time to avoid allocating an expanded-household-by-BRMA matrix.
    """
    if (
        not isinstance(id_multiplier, Integral)
        or isinstance(id_multiplier, bool)
        or id_multiplier <= 0
    ):
        raise ValueError("id_multiplier must be a positive integer.")
    uniforms = _validate_quantiles(payload["uniforms"])
    # Validate K without allocating the full household-by-clone grid.
    spaced_quantiles(np.empty(0), n_clones)
    ids = pd.Index(payload["household_ids"])
    if len(ids) != len(uniforms) or ids.has_duplicates or ids.dtype.kind not in "iu":
        raise ValueError(
            "Prepared household IDs must be unique integers aligned to draws."
        )
    names = np.asarray(payload["ordered_brmas"], dtype=object)
    cumulative = np.asarray(payload["cumulative_probabilities"], dtype=float)
    if (
        cumulative.ndim != 2
        or cumulative.shape[0] != len(ids)
        or cumulative.shape[1] == 0
        or names.shape != cumulative.shape
        or not np.isfinite(cumulative).all()
        or (cumulative < 0).any()
        or (cumulative > 1).any()
        or (np.diff(cumulative, axis=1) < 0).any()
        or (cumulative[:, -1] != 1).any()
    ):
        raise ValueError("Prepared BRMA names and cumulative probabilities must align.")
    clones = household[clone_index_column].to_numpy()
    household_ids = household["household_id"].to_numpy()
    if (
        clones.dtype.kind not in "iu"
        or household_ids.dtype.kind not in "iu"
        or (clones < 0).any()
        or (clones >= n_clones).any()
    ):
        raise ValueError("Geographic clone IDs and indices must be valid integers.")
    assigned = np.empty(len(household), dtype=object)
    for clone in np.unique(clones):
        rows = np.flatnonzero(clones == clone)
        parents = household_ids[rows] - int(clone) * int(id_multiplier)
        positions = ids.get_indexer(parents)
        if (positions < 0).any():
            raise ValueError("A geographic clone has no prepared parent household.")
        quantiles = (uniforms[positions] + int(clone) / n_clones) % 1.0
        columns = (cumulative[positions] <= quantiles[:, None]).sum(axis=1)
        assigned[rows] = names[positions, columns]
    return assigned
