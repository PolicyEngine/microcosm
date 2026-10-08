"""Country-neutral operators for transporting a declared donor population."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import blake2b, file_digest
from math import fsum
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from microcosm.frame.adapters.policyengine_us_concepts import (
    POLICYENGINE_US_CONCEPT_MAPPING,
)
from microcosm.frame.concepts import (
    HOUSEHOLD_ID_COLUMN,
    PERSON_HOUSEHOLD_ID_COLUMN,
    PERSON_ID_COLUMN,
    ContentBasis,
    Unit,
    concept_for_column,
    validate_concept_tables,
)
from microcosm.frame.materialize import read_frame_table
from microcosm.frame.scaling import apply_scale

__all__ = [
    "DonorBank",
    "currency_bridge",
    "derive_transport_seed",
    "quantile_map",
    "read_populace_us_donor",
]


@dataclass(frozen=True)
class _QuantileBand:
    lower: float
    upper: float | None
    share: float


def _quantile_bands(rows: Sequence[Mapping[str, object]]) -> tuple[_QuantileBand, ...]:
    if not rows:
        raise ValueError("Target bands must be nonempty.")
    bands = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"lower", "upper", "share"}:
            raise ValueError(
                "Each target band requires exactly lower, upper and share."
            )
        lower = float(row["lower"])
        upper = None if row["upper"] is None else float(row["upper"])
        share = float(row["share"])
        if not np.isfinite(lower) or lower < 0:
            raise ValueError("Band lower bounds must be finite and nonnegative.")
        if upper is not None and (not np.isfinite(upper) or upper <= lower):
            raise ValueError(
                "Band upper bounds must be finite and exceed lower bounds."
            )
        if not np.isfinite(share) or share <= 0:
            raise ValueError("Band shares must be finite and positive.")
        if bands and bands[-1].upper != lower:
            raise ValueError("Target bands must be ordered and contiguous.")
        bands.append(_QuantileBand(lower, upper, share))
    if not np.isclose(fsum(band.share for band in bands), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Target band shares must sum to one.")
    return tuple(bands)


def _quantile_interpolation(
    interpolation: str | Mapping[str, object],
) -> tuple[str, float | None]:
    if interpolation == "uniform":
        return "uniform", None
    if not isinstance(interpolation, Mapping) or set(interpolation) != {
        "method",
        "alpha",
    }:
        raise ValueError(
            "Interpolation must be uniform or {method: pareto, alpha: ...}."
        )
    if interpolation["method"] != "pareto":
        raise ValueError("The declared interpolation method must be pareto.")
    alpha = float(interpolation["alpha"])
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Pareto alpha must be finite and positive.")
    return "pareto", alpha


def _quantile_label_codes(labels: Sequence[object], size: int, name: str) -> np.ndarray:
    if isinstance(labels, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of labels.")
    labels = pd.Series(list(labels), dtype=object)
    if len(labels) != size or labels.isna().any():
        raise ValueError(f"{name} must have one nonmissing label per value.")
    try:
        codes, _ = pd.factorize(labels, sort=False)
    except TypeError as error:
        raise ValueError(f"{name} labels must be hashable.") from error
    return codes


def _pareto_inverse(
    fraction: np.ndarray, lower: float, upper: float | None, alpha: float
) -> np.ndarray:
    """Inverse CDF of a Pareto(alpha) band starting at ``lower``.

    A finite ``upper`` truncates the distribution to ``[lower, upper]``, whose
    CDF is ``(1 - (lower/x)**alpha) / (1 - (lower/upper)**alpha)``. Writing
    ``w = log(upper) - log(lower)``, the truncated mass is
    ``1 - (lower/upper)**alpha = -expm1(-alpha * w)`` and the inverse is
    ``lower * exp(-log1p(-fraction * mass) / alpha)``. ``expm1`` and ``log1p``
    keep both steps accurate when ``alpha * w`` is tiny, where subtracting
    ``(lower/upper)**alpha`` from 1 cancels: at ``alpha=1e-18`` on ``[10, 100]``
    the naive form returns 10 for the median instead of 31.62. As ``alpha``
    tends to 0 the truncated Pareto tends to the log-uniform distribution,
    ``lower * exp(fraction * w)``, which is used when ``alpha * w`` underflows.
    The result is ``lower * exp(offset)``; where ``exp(offset)`` alone would
    overflow inside a finite band, it is formed as ``exp(log(lower) + offset)``.
    """

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if upper is None:
            offset = -np.log1p(-fraction) / alpha
        else:
            width = np.log(upper) - np.log(lower)
            mass = -np.expm1(-alpha * width)
            if mass == 0:
                offset = fraction * width
            else:
                # offset <= width exactly; clamp the last-bit rounding above it.
                offset = np.minimum(-np.log1p(-fraction * mass) / alpha, width)
        result = lower * np.exp(offset)
        if upper is not None:
            result = np.where(
                np.isfinite(result), result, np.exp(np.log(lower) + offset)
            )
    return result


def _quantile_component(
    magnitudes: np.ndarray,
    weights: np.ndarray,
    bands: tuple[_QuantileBand, ...],
    method: str,
    alpha: float | None,
) -> np.ndarray:
    # Pool equal source values: splitting tied records must not change their rank.
    _, inverse = np.unique(magnitudes, return_inverse=True)
    scaled_weights = weights / weights.max()
    if (scaled_weights <= 0).any():
        raise ValueError("Weight ratios are too large for finite weighted ranks.")
    masses = np.bincount(inverse, weights=scaled_weights)
    total = fsum(masses)
    ranks = (np.cumsum(masses) - masses / 2) / total
    ranks = np.clip(ranks, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    shares = np.array([band.share for band in bands])
    boundaries = np.cumsum(shares / fsum(shares))
    boundaries[-1] = 1.0
    positions = np.searchsorted(boundaries, ranks, side="right")
    mapped = np.empty_like(ranks)
    for position, band in enumerate(bands):
        selected = positions == position
        start = 0.0 if position == 0 else boundaries[position - 1]
        fraction = (ranks[selected] - start) / (boundaries[position] - start)
        if method == "pareto" and position == len(bands) - 1:
            assert alpha is not None
            mapped[selected] = _pareto_inverse(fraction, band.lower, band.upper, alpha)
        else:
            assert band.upper is not None
            mapped[selected] = band.lower + fraction * (band.upper - band.lower)
    if not np.isfinite(mapped).all() or (mapped <= 0).any():
        raise ValueError("Mapped nonzero values must have finite positive magnitudes.")
    if (np.diff(mapped) <= 0).any():
        raise ValueError(
            "Mapped distinct ranks must remain distinct at float64 precision."
        )
    return mapped[inverse]


def _retain_exact_total(
    result: np.ndarray, selected: np.ndarray, anchor: int, total: float
) -> bool:
    """Try to correct ``result[anchor]`` in place so the members sum to total.

    The residual step uses the rounded current sum, which can miss by several
    units of a small anchor's spacing (a total at a power of two halves its
    spacing below it). If it and two one-unit steps fail, the anchor restarts
    from the correctly rounded difference between the total and the other
    members, then takes up to two more one-unit steps.
    """
    for attempt in range(3):
        current_total = fsum(result[selected])
        if current_total == total:
            return True
        if attempt == 0:
            result[anchor] += total - current_total
        else:
            direction = np.inf if current_total < total else -np.inf
            result[anchor] = np.nextafter(result[anchor], direction)
    if fsum(result[selected]) == total:
        return True
    result[anchor] = fsum([total, *(-result[selected[selected != anchor]])])
    for _ in range(2):
        current_total = fsum(result[selected])
        if current_total == total:
            return True
        direction = np.inf if current_total < total else -np.inf
        result[anchor] = np.nextafter(result[anchor], direction)
    return fsum(result[selected]) == total


def _quantile_map_components(
    values: np.ndarray,
    weights: np.ndarray,
    components: np.ndarray,
    sign_bands: Mapping[str, tuple[_QuantileBand, ...]],
    method: str,
    alpha: float | None,
) -> np.ndarray:
    result = values.copy()
    order = np.argsort(components, kind="stable")
    blocks = np.split(order, np.flatnonzero(np.diff(components[order])) + 1)
    for block in blocks:
        for sign, name in ((1, "positive"), (-1, "negative")):
            selected = block[values[block] * sign > 0]
            if len(selected):
                result[selected] = sign * _quantile_component(
                    np.abs(values[selected]),
                    weights[selected],
                    sign_bands[name],
                    method,
                    alpha,
                )
    return result


def quantile_map(
    values: ArrayLike,
    weights: ArrayLike,
    target_bands: Sequence[Mapping[str, object]]
    | Mapping[str, Sequence[Mapping[str, object]]],
    *,
    interpolation: str | Mapping[str, object],
    group: Sequence[object] | Mapping[str, Sequence[object]] | None = None,
) -> np.ndarray:
    """Map weighted midpoint ranks to declared bands, returning a new float64 array.

    ``target_bands`` is a sequence of ``{lower, upper, share}`` rows over
    positive *magnitudes*, or a mapping with exactly ``positive`` and
    ``negative`` sequences. Bands are contiguous and increasing; shares sum
    to one separately for each sign, conditional on being nonzero. Zero
    values retain their value and do not enter either rank distribution.
    Inputs must be finite one-dimensional arrays with strictly positive
    weights. Weights are never changed.

    ``interpolation="uniform"`` requires finite upper bounds. A declaration
    ``{"method": "pareto", "alpha": ...}`` applies Pareto interpolation to
    the top band only, with a positive lower bound. Its upper bound may be
    ``None`` (unbounded) or finite (truncated); lower bands remain uniform.

    A sequence passed as ``group`` gives one component label per value.
    Ranks are computed independently within each component and sign. Equal
    magnitudes share their pooled weighted midpoint rank. Consequently the
    error in a band's weighted share is bounded by the largest tied block's
    fraction of the component/sign weight: a tied block is indivisible.

    Alternatively ``group={"entity_ids": [...], "components": [...]}``
    declares member-to-entity memberships. ``components`` is optional.
    Members of an entity must repeat its weight and component label. Their
    values are summed, entity totals are mapped using each entity weight
    once, and members are rescaled pro rata by a positive factor. A floating
    rounding residual is assigned to the largest nonzero member so that
    tied entity totals remain exactly tied on repeated mapping. This also
    preserves member signs for mixed-sign assets. Where the largest member
    alone cannot reach the exact total (members ``[1, 2]`` whose total maps
    to 0.21), a member with finer float64 spacing first moves by one unit in
    the last place. A zero-total entity is unchanged, including cancelling
    nonzero members.
    The pro rata factor is the mapped total over the entity total and has no
    upper bound: members that nearly cancel are amplified with it, so
    members ``[1000, -999]`` whose total of 1 maps to 15 become ``[15000,
    -14985]``. Mixed-sign members should not be aggregated when the entity
    total is small relative to their magnitudes; ``liquid_financial_assets``
    (``lower=0``) cannot mix signs. Component labels and entity ids must be
    nonmissing hashable scalars. Mapping fails if float64 cannot retain
    distinct ranks, signs, pro rata precision, or the mapped aggregate totals
    (including severe mixed-sign cancellation). Same-sign members always
    reach the exact total. Mixed-sign members can still be refused when no
    tried one-unit move reaches it, as for members ``[-5, 1, 1]`` whose total
    of -3 maps to -20.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError(
            "Values and weights must have matching one-dimensional shapes."
        )
    if not np.isfinite(values).all():
        raise ValueError("Values must be finite.")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Weights must be finite and positive.")
    method, alpha = _quantile_interpolation(interpolation)
    if isinstance(target_bands, Mapping):
        if set(target_bands) != {"positive", "negative"}:
            raise ValueError(
                "Signed target bands require exactly positive and negative."
            )
        sign_bands = {
            sign: _quantile_bands(rows) for sign, rows in target_bands.items()
        }
    else:
        bands = _quantile_bands(target_bands)
        sign_bands = {"positive": bands, "negative": bands}
    for bands in sign_bands.values():
        if method == "uniform" and bands[-1].upper is None:
            raise ValueError("Uniform interpolation requires finite upper bounds.")
        if method == "pareto" and bands[-1].lower <= 0:
            raise ValueError("The Pareto top band requires a positive lower bound.")
    if not isinstance(group, Mapping):
        components = (
            np.zeros(len(values), dtype=np.int64)
            if group is None
            else _quantile_label_codes(group, len(values), "Components")
        )
        return _quantile_map_components(
            values, weights, components, sign_bands, method, alpha
        )
    if "entity_ids" not in group or not set(group) <= {"entity_ids", "components"}:
        raise ValueError("Aggregate groups require entity_ids and optional components.")
    entities = _quantile_label_codes(group["entity_ids"], len(values), "Entity ids")
    components = (
        np.zeros(len(values), dtype=np.int64)
        if "components" not in group
        else _quantile_label_codes(group["components"], len(values), "Components")
    )
    count = int(entities.max()) + 1 if len(entities) else 0
    totals = np.empty(count)
    entity_weights = np.empty(count)
    entity_components = np.empty(count, dtype=np.int64)
    order = np.argsort(entities, kind="stable")
    members = (
        np.split(order, np.flatnonzero(np.diff(entities[order])) + 1) if count else []
    )
    for entity, selected in enumerate(members):
        if (weights[selected] != weights[selected[0]]).any():
            raise ValueError("Members must repeat one entity weight.")
        if (components[selected] != components[selected[0]]).any():
            raise ValueError("Members must repeat one entity component label.")
        try:
            totals[entity] = fsum(values[selected])
        except OverflowError as error:
            raise ValueError("Entity totals must be finite.") from error
        entity_weights[entity] = weights[selected[0]]
        entity_components[entity] = components[selected[0]]
    mapped = _quantile_map_components(
        totals, entity_weights, entity_components, sign_bands, method, alpha
    )
    result = values.copy()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for entity, selected in enumerate(members):
            if totals[entity] != 0:
                result[selected] = (values[selected] / totals[entity]) * mapped[entity]
                if not np.isfinite(result[selected]).all():
                    raise ValueError(
                        "Pro rata mapping must preserve finite member values."
                    )
                nonzero = selected[values[selected] != 0]
                # Put the sum-rounding residual on the largest member so an
                # otherwise tiny holding is never materially changed. Severe
                # cancellation may make the exact total unrepresentable.
                anchor = nonzero[np.argmax(np.abs(result[nonzero]))]
                original_anchor = result[anchor]
                retained = _retain_exact_total(result, selected, anchor, mapped[entity])
                # Smaller members can leave the exact total halfway between
                # two sums the largest member reaches (members [1, 2] whose
                # total maps to 0.21). A member with finer float64 spacing
                # than that tie moving by one unit in the last place, a
                # relative change near 1e-16, breaks it. Up to four members
                # finer than the largest are tried, largest first, then the
                # smallest member: the tie can sit a binade above the largest
                # member, and the smallest has the finest spacing. Each moves
                # away from zero and then toward it.
                candidates = []
                if not retained:
                    others = nonzero[nonzero != anchor]
                    others = others[np.argsort(-np.abs(result[others]), kind="stable")]
                    finer = np.spacing(np.abs(result[others])) < np.spacing(
                        abs(original_anchor)
                    )
                    candidates = list(dict.fromkeys([*others[finer][:4], others[-1]]))
                for other, toward in [
                    (other, side * np.sign(result[other]))
                    for other in candidates
                    for side in (np.inf, 0.0)
                ]:
                    if retained:
                        break
                    kept = result[other]
                    result[anchor] = original_anchor
                    result[other] = np.nextafter(kept, toward)
                    retained = result[other] != 0 and _retain_exact_total(
                        result, selected, anchor, mapped[entity]
                    )
                    if not retained:
                        result[other] = kept
                if not retained:
                    raise ValueError(
                        "Pro rata mapping cannot retain exact totals at float64 precision."
                    )
                correction_bound = 4 * len(selected) * abs(np.spacing(original_anchor))
                if abs(result[anchor] - original_anchor) > correction_bound:
                    raise ValueError(
                        "Pro rata correction exceeds float64 rounding precision."
                    )
    if not np.isfinite(result).all() or (np.sign(result) != np.sign(values)).any():
        raise ValueError(
            "Pro rata mapping must preserve finite member values and signs."
        )
    return result


@dataclass(frozen=True)
class DonorBank:
    """US donor concepts and aligned provenance, never destination microdata.

    ``weights`` and ``support_strata`` follow the household table's row order;
    ``source_person_ids`` follows the person table. Weights are the untouched
    donor design weights. Explicit ``person_source_id`` values retain their
    exact integer/string identities, including repetitions on support clones.
    Otherwise full source lineage is encoded as a JSON string of the support
    stratum and typed year/household/person IDs, avoiding household-local ID
    collisions; files without lineage use their pinned donor ``person_id``.
    ``currency`` describes source
    amounts; the destination producer supplies its own frame declaration after
    transport. No destination country or year is inferred by this reader.
    ``dropped_parent_cycle_edges`` counts directed parent pointers removed by
    the reader's age-based two-person-cycle rule, including support copies.
    """

    tables: Mapping[str, pd.DataFrame]
    weights: np.ndarray
    support_strata: np.ndarray
    source_person_ids: np.ndarray
    content_basis: ContentBasis = ContentBasis.TRANSPORT
    donor_country: str = "us"
    currency: str = "USD"
    dropped_parent_cycle_edges: int = 0


#: Line numbers and pointers are validated as float64 (pointer columns can hold
#: NaN), which represents every integer exactly only below 2**53. A larger value
#: could round onto a different roster line, so it is refused before conversion.
_MAX_EXACT_LINE = 2**53


def _line_numbers(values: pd.Series, *, pointer: bool = False) -> np.ndarray:
    native = pd.to_numeric(values, errors="raise")
    numbers = native.to_numpy(dtype=np.float64, na_value=np.nan)
    # Compare in the column's own dtype; infinities fall to the integer check.
    if ((native.abs() >= _MAX_EXACT_LINE) & ~np.isinf(numbers)).any():
        raise ValueError(f"Donor {values.name} contains invalid line numbers.")
    if pointer:
        # A missing pointer is absent (rows from a vintage without PECOHAB
        # leave it NaN).
        numbers = np.where(np.isnan(numbers), 0, numbers)
    if not np.isfinite(numbers).all() or np.any(numbers != np.floor(numbers)):
        raise ValueError(f"Donor {values.name} must contain integer line numbers.")
    if pointer:
        # The pinned donor writes -1 for an absent parent or cohabiting
        # partner and 0 for an absent spouse. Every pointer <= 0 is absent,
        # as the US build's eligibility_inputs._parent_person_ids does for
        # PEPAR1/2.
        numbers = np.maximum(numbers, 0)
    if np.any(numbers < (0 if pointer else 1)) or np.any(numbers >= 2**63):
        raise ValueError(f"Donor {values.name} contains invalid line numbers.")
    return numbers.astype(np.int64)


def _pointed_rows(
    values: pd.Series, roster: pd.MultiIndex, household_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each line pointer's presence and its target row in the household roster."""
    pointed_line = _line_numbers(values, pointer=True)
    present = pointed_line != 0
    rows = roster.get_indexer(pd.MultiIndex.from_arrays([household_ids, pointed_line]))
    if np.any(present & (rows < 0)):
        raise ValueError(f"Donor {values.name} points to an unknown household line.")
    return present, rows


def _donor_extras(person: pd.DataFrame, household: pd.DataFrame) -> dict:
    """Translate donor-only columns before crossing the concept decode seam."""
    required = (
        "A_LINENO",
        "A_SPOUSE",
        "PEPAR1",
        "PEPAR2",
        "is_household_head",
        "bank_account_assets",
        "stock_assets",
        "bond_assets",
    )
    missing = set(required) - set(person.columns)
    if missing:
        raise ValueError(f"Donor person table lacks columns {sorted(missing)}.")
    if "household_weight" not in household:
        raise ValueError("Donor household table lacks household_weight.")
    # Validate the structural IDs without turning large integers into floats.
    validate_concept_tables(
        {
            "person": person[[PERSON_ID_COLUMN, PERSON_HOUSEHOLD_ID_COLUMN]],
            "household": household[[HOUSEHOLD_ID_COLUMN]],
        }
    )
    line = _line_numbers(person["A_LINENO"])
    household_ids = person[PERSON_HOUSEHOLD_ID_COLUMN].to_numpy()
    roster = pd.MultiIndex.from_arrays([household_ids, line])
    if not roster.is_unique:
        raise ValueError("Donor line numbers must be unique within a household.")
    if (person[PERSON_ID_COLUMN] > np.iinfo(np.int64).max).any():
        raise ValueError("Donor person IDs must fit the concept pointer's int64 dtype.")
    person_ids = person[PERSON_ID_COLUMN].to_numpy(dtype=np.int64)

    def pointer(present: np.ndarray, rows: np.ndarray) -> pd.arrays.IntegerArray:
        values = pd.array([pd.NA] * len(person), dtype="Int64")
        values[present] = person_ids[rows[present]]
        return values

    # Partners come only from the spouse and cohabiting-partner line pointers.
    # Relationship code 13 is never read: the Census recode uses it for
    # roommates and housemates as well as unmarried partners, and the
    # ACS-to-CPS crosswalk maps all three to it, so it does not identify a
    # partner. Rows from a vintage without PECOHAB leave it NaN, so their
    # cohabiting couples get no partner pointer.
    partner_present, partner_rows = _pointed_rows(
        person["A_SPOUSE"], roster, household_ids
    )
    if "PECOHAB" in person:
        cohabiting, cohabitant_rows = _pointed_rows(
            person["PECOHAB"], roster, household_ids
        )
        if np.any(partner_present & cohabiting & (partner_rows != cohabitant_rows)):
            raise ValueError("Donor A_SPOUSE and PECOHAB name different partners.")
        partner_rows = np.where(partner_present, partner_rows, cohabitant_rows)
        partner_present = partner_present | cohabiting
    pointers = {
        "partner_person_id": pointer(partner_present, partner_rows),
        "parent_1_person_id": pointer(
            *_pointed_rows(person["PEPAR1"], roster, household_ids)
        ),
        "parent_2_person_id": pointer(
            *_pointed_rows(person["PEPAR2"], roster, household_ids)
        ),
    }
    # A sole co-resident parent is always parent 1 in the concept contract.
    move = (
        pointers["parent_1_person_id"].isna() & ~pointers["parent_2_person_id"].isna()
    )
    pointers["parent_1_person_id"][move] = pointers["parent_2_person_id"][move]
    pointers["parent_2_person_id"][move] = pd.NA
    # The donor's US-engine head (P_SEQ == 1) is the reference person in
    # every vintage. A_EXPRRP is not read: older rows derive it from line 1.
    head_flags = person["is_household_head"]
    if not pd.api.types.is_bool_dtype(head_flags.dtype) or head_flags.isna().any():
        raise ValueError("Donor is_household_head must be boolean with no nulls.")
    head = head_flags.to_numpy(dtype=bool)
    if not pd.Index(household_ids[head]).is_unique:
        raise ValueError("Donor household has multiple is_household_head persons.")
    reference_rows = pd.Index(household_ids[head]).get_indexer(
        household[HOUSEHOLD_ID_COLUMN]
    )
    if np.any(reference_rows < 0):
        raise ValueError("Donor household has no is_household_head person.")
    reference_ids = person_ids[head][reference_rows]
    # The donor's three leaves are not invertible through an encoding mapping:
    # their sum is the reviewed donor-side stock, without inventing a split.
    leaves = person[["bank_account_assets", "stock_assets", "bond_assets"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(leaves).all() or np.any(leaves < 0):
        raise ValueError("Donor liquid-asset leaves must be finite and nonnegative.")
    assets = leaves.sum(axis=1)
    weights = household["household_weight"].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError(
            "Donor weights must be finite, nonnegative, with positive mass."
        )
    # Older Build P files have one undifferentiated US support stratum. Never
    # infer a survey channel from income or engine inputs when labels are absent.
    channel_column = "household_support_channel"
    if channel_column in household:
        channels = household[channel_column].to_numpy(dtype=object, copy=True)
        if any(not isinstance(value, str) or not value for value in channels):
            raise ValueError(
                "Donor household support channels must be nonempty strings."
            )
        strata = np.asarray([f"us:{value}" for value in channels], dtype=object)
    else:
        channels = None
        strata = np.full(len(household), "us", dtype=object)
    if "person_support_channel" in person:
        if channels is None:
            raise ValueError(
                "Person support channels require household support channels."
            )
        hh_rows = pd.Index(household[HOUSEHOLD_ID_COLUMN]).get_indexer(household_ids)
        if not np.array_equal(
            person["person_support_channel"].to_numpy(), channels[hh_rows]
        ):
            raise ValueError("Donor person and household support channels disagree.")
    lineage = ("source_year", "source_household_id", "source_person_id")
    if "person_source_id" in person:
        source_ids = person["person_source_id"].to_numpy(copy=True)
    elif any(name in person for name in lineage):
        if not all(name in person for name in lineage):
            raise ValueError(
                "Donor source lineage requires year, household and person IDs."
            )
        parts = [_source_id_payloads(person[name]) for name in lineage]
        hh_rows = pd.Index(household[HOUSEHOLD_ID_COLUMN]).get_indexer(household_ids)
        source_ids = np.asarray(
            [
                json.dumps(
                    [strata[hh_rows[index]], *items],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                for index, items in enumerate(zip(*parts, strict=True))
            ],
            dtype=object,
        )
    else:
        source_ids = person[PERSON_ID_COLUMN].to_numpy(copy=True)
    # Check exact identities with the same contract used by seed derivation.
    _source_id_payloads(source_ids)
    return {
        "pointers": pointers,
        "reference_person_id": pd.array(reference_ids, dtype="Int64"),
        "liquid_financial_assets": assets,
        "weights": weights,
        "support_strata": strata,
        "source_person_ids": source_ids,
    }


def _drop_two_person_parent_cycles(person: pd.DataFrame) -> int:
    """Repair reciprocal parent edges only after other concepts validate.

    Check the original graph before mutation: a reciprocal edge may also
    belong to a longer cycle, which must still refuse. Removing its direct
    reverse edge for a reachability search exposes any such longer cycle.
    Searches stay within a household and run only for reciprocal edges.
    """
    columns = ("parent_1_person_id", "parent_2_person_id")
    ids = pd.Index(person[PERSON_ID_COLUMN])
    parents = np.full((len(person), len(columns)), -1, dtype=np.int64)
    for slot, column in enumerate(columns):
        present = person[column].notna().to_numpy()
        parents[present, slot] = ids.get_indexer(
            person[column][present].to_numpy(dtype=np.int64)
        )
    rows = np.arange(len(person))
    reciprocal = np.zeros(parents.shape, dtype=bool)
    for slot in range(len(columns)):
        present = parents[:, slot] >= 0
        reciprocal[present, slot] = np.any(
            parents[parents[present, slot]] == rows[present, None], axis=1
        )
    edges = np.argwhere(reciprocal)
    if not len(edges):
        return 0
    for child, slot in edges:
        parent = parents[child, slot]
        pending = [parent]
        visited = {parent}
        while pending:
            current = pending.pop()
            for target in parents[current]:
                if target < 0 or (current == parent and target == child):
                    continue
                if target == child:
                    # Leave the original violations for the caller to report.
                    return 0
                if target not in visited:
                    visited.add(target)
                    pending.append(target)
    ages = (
        person["age"].to_numpy(dtype=np.float64, na_value=np.nan)
        if "age" in person
        else np.full(len(person), np.nan)
    )
    children, slots = edges.T
    targets = parents[children, slots]
    known = np.isfinite(ages[children]) & np.isfinite(ages[targets])
    drop = ~known | (ages[targets] <= ages[children])
    for slot, column in enumerate(columns):
        person.iloc[
            children[drop & (slots == slot)], person.columns.get_loc(column)
        ] = pd.NA
    move = person[columns[0]].isna() & person[columns[1]].notna()
    person.loc[move, columns[0]] = person.loc[move, columns[1]]
    person.loc[move, columns[1]] = pd.NA
    return int(drop.sum())


def read_populace_us_donor(path: str | Path, *, sha256: str, size: int) -> DonorBank:
    """Authenticate a local donor H5, then decode its person/household inputs.

    Size and SHA-256 are mandatory and verified before any HDF parsing. The
    file is hashed through one handle and then reopened by path, so a file
    swapped or rewritten in between would be read unverified; the reader
    assumes a local cache that nothing rewrites while it runs. (PyTables' core
    driver can parse the hashed bytes from memory, but only under a filename
    that does not exist on disk.) There is no download or alternate donor
    fallback.

    CPS line pointers are resolved by (household id, line number). A pointer
    that is missing or ``<= 0`` is absent: the donor the tests pin writes
    ``-1`` for an absent parent or cohabiting partner and ``0`` for an absent
    spouse. The partner pointer comes from ``A_SPOUSE``, or from ``PECOHAB``
    on rows whose source vintage carries it; the two must agree when both are
    set. In a donor that pools vintages, cohabiting partners are therefore
    recovered only for some source years (only the 2024 rows of the donor the
    tests pin carry ``PECOHAB``). Relationship code 13 is never read, because
    it also covers roommates and housemates.

    The reference person is the donor's ``is_household_head`` person in
    every household; exactly one boolean head flag is required. The US build
    sets it from ``P_SEQ == 1`` (``relationship_inputs.py``). On all 19,753
    households from 2024 in the pinned donor, that head agrees with the Census
    reference person, including 222 whose reference person is not line 1.
    The 2022/2023 ``A_EXPRRP`` recode instead derives from line 1
    (``asec_pool.py``), so ``A_EXPRRP`` is not used.

    When two people name each other as parents, drop the edge naming the
    younger parent and keep the other. Equal ages or an absent age column
    drop both edges. The bank's ``dropped_parent_cycle_edges`` counts removed
    directed pointers. A remaining sole parent is compacted into parent 1.
    Longer cycles, including ones sharing reciprocal edges, and every other
    concept inconsistency still refuse. A present age column must satisfy
    the concept schema's whole, nonmissing-age contract.

    Donor-only enrichment is staged before ``decode``; all work after that
    seam reads concept names. The result declares transport content with a
    US donor and unchanged source-currency amounts and design weights.
    """
    path = Path(path)
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("Donor size must be a nonnegative integer.")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(c not in "0123456789abcdef" for c in sha256)
    ):
        raise ValueError("Donor SHA-256 must be 64 lowercase hexadecimal characters.")
    with path.open("rb") as source:
        if path.stat().st_size != size:
            raise ValueError("Donor size mismatch.")
        if file_digest(source, "sha256").hexdigest() != sha256:
            raise ValueError("Donor SHA-256 mismatch.")
    with pd.HDFStore(path, mode="r") as store:
        raw = {
            entity: read_frame_table(store, entity)
            for entity in ("person", "household")
        }
    staged = _donor_extras(raw["person"], raw["household"])
    tables = POLICYENGINE_US_CONCEPT_MAPPING.decode(raw)
    del raw
    for name, values in staged["pointers"].items():
        tables["person"][name] = values
    tables["person"]["liquid_financial_assets"] = staged["liquid_financial_assets"]
    tables["household"]["reference_person_id"] = staged["reference_person_id"]
    violations = validate_concept_tables(tables)
    dropped = 0
    if violations and all(v.code == "parent_cycle" for v in violations):
        dropped = _drop_two_person_parent_cycles(tables["person"])
        if dropped:
            violations = validate_concept_tables(tables)
    if violations:
        details = "; ".join(f"{v.entity}.{v.column}: {v.message}" for v in violations)
        raise ValueError(f"Invalid donor concepts: {details}")
    return DonorBank(
        MappingProxyType(tables),
        staged["weights"],
        staged["support_strata"],
        staged["source_person_ids"],
        dropped_parent_cycle_edges=dropped,
    )


def _source_id_payloads(source_ids) -> list[tuple[str, str]]:
    values = np.asarray(source_ids, dtype=object)
    if values.ndim != 1:
        raise ValueError("Source person IDs must be one-dimensional.")
    payloads = []
    for value in values:
        if isinstance(value, str) and value:
            payloads.append(("str", value))
        elif isinstance(value, (int, np.integer)) and not isinstance(
            value, (bool, np.bool_)
        ):
            payloads.append(("int", str(value)))
        else:
            raise ValueError(
                "Source person IDs must be nonempty strings or exact integers."
            )
    return payloads


def derive_transport_seed(source_ids, stream: str) -> np.ndarray:
    """Return order-independent float64 seeds in [0, 1) from exact source IDs.

    A domain-separated BLAKE2b hash binds the declared stream and typed source
    identity. Repeated source IDs (support clones) share a seed. Keeping the
    top 53 hash bits makes the result exactly representable, including the
    largest possible draw, without a rounded value of 1.
    """
    if not isinstance(stream, str) or not stream:
        raise ValueError("A transport seed stream must be a nonempty string.")
    payloads = _source_id_payloads(source_ids)
    seeds = np.empty(len(payloads), dtype=np.float64)
    for index, (kind, identity) in enumerate(payloads):
        payload = json.dumps(
            ["microcosm.transport.seed.v1", stream, kind, identity],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        bits = int.from_bytes(blake2b(payload, digest_size=8).digest(), "big")
        seeds[index] = (bits >> 11) / 2**53
    return seeds


def currency_bridge(
    tables: Mapping[str, pd.DataFrame],
    columns: Mapping[str, Sequence[str]],
    rate: float,
) -> dict[str, pd.DataFrame]:
    """Copy tables and multiply each declared money column once, rounded to 2 dp.

    ``columns`` maps entities to their concept column names. Every selected
    column must be a currency amount; duplicates and absent columns fail.
    ``rate`` is supplied by the caller's specification, validated by the shared
    ``apply_scale`` operator. Rounding uses NumPy's round-to-nearest-even rule.
    Unselected columns and source tables are unchanged.
    """
    if isinstance(rate, (bool, np.bool_)) or not np.isscalar(rate):
        raise ValueError("A currency bridge requires one scalar rate.")
    rate = float(rate)
    apply_scale(np.empty(0), rate)
    out = {entity: table.copy() for entity, table in tables.items()}
    for entity, names in columns.items():
        if isinstance(names, str) or len(set(names)) != len(names):
            raise ValueError("Currency bridge columns must be distinct column names.")
        if entity not in out:
            raise ValueError(f"Unknown currency bridge entity {entity!r}.")
        for name in names:
            item = concept_for_column(entity, name)
            if item is None or item.unit is not Unit.BASE_CURRENCY:
                raise ValueError(
                    f"Currency bridge requires an amount concept: {entity}.{name}."
                )
            if name not in out[entity]:
                raise ValueError(f"Currency bridge column is absent: {entity}.{name}.")
            scaled = apply_scale(out[entity][name].to_numpy(dtype=np.float64), rate)
            with np.errstate(over="ignore", invalid="ignore"):
                rounded = np.round(scaled, 2)
            if not np.isfinite(rounded).all():
                raise ValueError("Rounded currency amounts must be finite.")
            out[entity][name] = rounded
    return out
