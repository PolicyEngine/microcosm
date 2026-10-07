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
            # A finite upper bound declares a truncated Pareto distribution.
            tail = 0.0 if band.upper is None else (band.lower / band.upper) ** alpha
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                mapped[selected] = band.lower * np.exp(
                    -np.log1p(-fraction * (1 - tail)) / alpha
                )
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
    preserves member signs for mixed-sign assets. A zero-total entity is
    unchanged, including cancelling nonzero members. Component labels and
    entity ids must be nonmissing hashable scalars. Mapping fails if float64
    cannot retain distinct ranks, signs, pro rata precision, or the mapped
    aggregate totals (including severe mixed-sign cancellation).
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
                for attempt in range(3):
                    current_total = fsum(result[selected])
                    if current_total == mapped[entity]:
                        break
                    if attempt == 0:
                        result[anchor] += mapped[entity] - current_total
                    else:
                        direction = (
                            np.inf if current_total < mapped[entity] else -np.inf
                        )
                        result[anchor] = np.nextafter(result[anchor], direction)
                if fsum(result[selected]) != mapped[entity]:
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
    """

    tables: Mapping[str, pd.DataFrame]
    weights: np.ndarray
    support_strata: np.ndarray
    source_person_ids: np.ndarray
    content_basis: ContentBasis = ContentBasis.TRANSPORT
    donor_country: str = "us"
    currency: str = "USD"


def _line_numbers(values: pd.Series, *, nullable: bool = False) -> np.ndarray:
    numbers = pd.to_numeric(values, errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    if nullable:
        numbers = np.where(np.isnan(numbers), 0, numbers)
    if not np.isfinite(numbers).all() or np.any(numbers != np.floor(numbers)):
        raise ValueError(f"Donor {values.name} must contain integer line numbers.")
    if np.any(numbers < (0 if nullable else 1)) or np.any(numbers >= 2**63):
        raise ValueError(f"Donor {values.name} contains invalid line numbers.")
    return numbers.astype(np.int64)


def _donor_extras(person: pd.DataFrame, household: pd.DataFrame) -> dict:
    """Translate donor-only columns before crossing the concept decode seam."""
    required = (
        "A_LINENO",
        "A_SPOUSE",
        "PEPAR1",
        "PEPAR2",
        "A_EXPRRP",
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
    pointers = {}
    for source, destination in (
        ("A_SPOUSE", "partner_person_id"),
        ("PEPAR1", "parent_1_person_id"),
        ("PEPAR2", "parent_2_person_id"),
    ):
        pointed_line = _line_numbers(person[source], nullable=True)
        present = pointed_line != 0
        rows = roster.get_indexer(
            pd.MultiIndex.from_arrays([household_ids, pointed_line])
        )
        if np.any(present & (rows < 0)):
            raise ValueError(f"Donor {source} points to an unknown household line.")
        pointer = pd.array([pd.NA] * len(person), dtype="Int64")
        pointer[present] = person_ids[rows[present]]
        pointers[destination] = pointer
    # A sole co-resident parent is always parent 1 in the concept contract.
    move = (
        pointers["parent_1_person_id"].isna() & ~pointers["parent_2_person_id"].isna()
    )
    pointers["parent_1_person_id"][move] = pointers["parent_2_person_id"][move]
    pointers["parent_2_person_id"][move] = pd.NA
    relationship = _line_numbers(person["A_EXPRRP"])
    reference = np.isin(relationship, (1, 2))
    reference_households = pd.Index(household_ids[reference])
    if not reference_households.is_unique:
        raise ValueError("Donor household has multiple reference persons.")
    reference_rows = reference_households.get_indexer(household[HOUSEHOLD_ID_COLUMN])
    if np.any(reference_rows < 0):
        raise ValueError("Donor household has no reference person.")
    reference_ids = person_ids[reference][reference_rows]
    # The CPS relationship recode identifies the reference person's unmarried
    # partner as 13 (the ACS-to-CPS crosswalk uses the same code). A_SPOUSE
    # alone does not capture these cohabiting couples.
    unmarried_rows = np.flatnonzero(relationship == 13)
    if not pd.Index(household_ids[unmarried_rows]).is_unique:
        raise ValueError("Donor household has multiple unmarried partners.")
    reference_positions = np.flatnonzero(reference)
    for partner_row in unmarried_rows:
        reference_row = reference_positions[
            reference_households.get_indexer([household_ids[partner_row]])[0]
        ]
        partners = pointers["partner_person_id"]
        for own_row, other_row in (
            (partner_row, reference_row),
            (reference_row, partner_row),
        ):
            if (
                not pd.isna(partners[own_row])
                and partners[own_row] != person_ids[other_row]
            ):
                raise ValueError(
                    "Donor unmarried-partner roster conflicts with spouse pointers."
                )
        partners[partner_row] = person_ids[reference_row]
        partners[reference_row] = person_ids[partner_row]
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


def read_populace_us_donor(path: str | Path, *, sha256: str, size: int) -> DonorBank:
    """Authenticate a local donor H5, then decode its person/household inputs.

    Size and SHA-256 are mandatory and verified before any HDF parsing. There
    is no download or alternate donor fallback. CPS pointers are resolved by
    (household id, line number), and invalid/asymmetric relations fail concept
    validation. Donor-only enrichment is staged before ``decode``; all work
    after that seam reads concept names. The result declares transport content
    with a US donor and unchanged source-currency amounts and design weights.
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
    # Use the validated roster as the input mapping's householder flag, so
    # stale engine flags cannot override the survey reference-person pointer.
    raw["person"]["is_household_head"] = raw["person"][PERSON_ID_COLUMN].isin(
        staged["reference_person_id"]
    )
    tables = POLICYENGINE_US_CONCEPT_MAPPING.decode(raw)
    del raw
    for name, values in staged["pointers"].items():
        tables["person"][name] = values
    tables["person"]["liquid_financial_assets"] = staged["liquid_financial_assets"]
    tables["household"]["reference_person_id"] = staged["reference_person_id"]
    violations = validate_concept_tables(tables)
    if violations:
        details = "; ".join(f"{v.entity}.{v.column}: {v.message}" for v in violations)
        raise ValueError(f"Invalid donor concepts: {details}")
    return DonorBank(
        MappingProxyType(tables),
        staged["weights"],
        staged["support_strata"],
        staged["source_person_ids"],
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
