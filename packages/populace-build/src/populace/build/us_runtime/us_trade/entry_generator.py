"""Synthetic import-entry generation from ledger value margins.

Every row this module emits is **synthetic** — generated, never observed.
No public entry-level import microdata exists; the customs-value margins
ingested in :mod:`populace.build.us_runtime.us_trade.census_imports` are the only
calibrated truth, and the generator reproduces them **exactly**: for every
(HTS-10, country, month) cell in the window, the weighted sum of entry
customs values equals the published margin to the integer dollar, by
construction. Everything below the margins is a documented assumption, and
the synthetic identity is stamped everywhere: entry ids carry a
``synthetic-`` prefix, the emitted table carries dataset-level synthetic
metadata, and the assumptions register ships beside the data.

Entry-size distribution (the assumption)
----------------------------------------

CBP publishes no entry-size distribution, only national totals
(:mod:`populace.build.us_runtime.us_trade.cbp_entry_stats`). The generator assumes
within-cell lognormal entry sizes, with both free parameters anchored to
those published totals:

- the national mean entry value pins per-cell entry counts
  (``count = clip(round(value / mean), 1, value)``), and
- the lognormal shape ``sigma`` is solved so the generated mixture's share
  of entries at or below $2,500 of customs value matches CBP's published
  informal-entry count share.

The sigma calibration is a **proxy**, not a statutory identification.
19 CFR 143.21 defines informal entry by entry-type rules, of which the
"not exceeding $2,500" clause is only the first: the section also admits
installment shipments and portions of consignments, personal and household
effects, certain institutional (books/libraries) imports, and some
returned US products up to $10,000, and carries Chapter 99 exclusions —
categories that are not a function of customs value. CBP's published
informal count is therefore not the CDF of entry value at $2,500; the
generator uses it as an *approximate* size-distribution moment because no
closer public moment exists, and the register discloses the mapping
limitation (``known_gaps.informal_share_proxy``).

Cells are stratified into equal-probability strata of that lognormal; each
stratum becomes at most two weighted rows (an integer largest-remainder
split), which keeps the file at a few rows per cell while preserving exact
integer margins and a nondegenerate size distribution for de minimis-style
threshold analyses. The scheme is fully deterministic — no random state,
no wall-clock input, and a canonical (period, hts10, country) cell order —
so the same margins and anchors reproduce the same rows and, under the
locked build environment (numpy/scipy/pyarrow pinned by ``uv.lock``), the
same serialized bytes. Cross-platform floating-point identity of the
lognormal stratum shapes is not claimed; the exact-margin guarantee holds
on every platform regardless, because all dollar allocation is integer.

Schema is the composed tariff spine's input surface (rulespec-us
``us:policies/cbp/us-tariff-duty/composition``): dotted ``hts_number``,
ISO-2 ``country_of_origin``, integer ``customs_value``,
``is_postal_shipment`` (``False`` — the bulk source's transport splits are
air/vessel/containerized only and its rate-provision codes classify duty
status, not entry type, so no public series grounds a postal flag;
documented gap), and a mid-month ``entry_date`` (sub-monthly timing is not
calibrated). ``shipment_value`` is set equal to customs value — the margins
carry no separate shipment-value series (documented assumption).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "DEFAULT_INFORMAL_VALUE_THRESHOLD",
    "DEFAULT_STRATA",
    "EntrySizeAssumption",
    "dotted_hts",
    "generate_entries",
    "solve_size_assumption",
    "validate_entries_against_margins",
]

#: The 19 CFR 143.21(a) value clause: informal entry covers shipments of
#: merchandise "not exceeding $2,500 in value". The section's other
#: clauses admit entries by type rather than value (see the module
#: docstring), so this threshold underpins a proxy calibration only.
DEFAULT_INFORMAL_VALUE_THRESHOLD = 2_500

DEFAULT_STRATA = 7

#: Margin cell values above this are outside the proven-exact allocation
#: domain: the largest-remainder residual split routes through float64
#: quotas, and integers above 2**53 are not exactly representable there.
#: Real cells are orders of magnitude below (the largest observed monthly
#: cell is ~$4e10); the gate makes the boundary explicit instead of
#: leaving exactness silently unproven on absurd inputs.
MAX_EXACT_CELL_VALUE = 2**53

_ENTRY_DAY_OF_MONTH = 15
_SIGMA_BOUNDS = (0.05, 6.0)
_SIGMA_ITERATIONS = 40
#: Bisection stops once the achieved share is within this of the target;
#: each evaluation runs the full exact stratification over every cell.
_SHARE_TOLERANCE = 1e-4


@dataclass(frozen=True)
class EntrySizeAssumption:
    """The documented entry-size assumption, with its anchor provenance.

    ``sigma`` is solved, not chosen: :func:`solve_size_assumption` bisects
    it so the generated mixture's informal count share matches
    ``informal_count_share_target`` on the actual margin cells. The
    achieved statistics are recorded so the register never overstates the
    fit.
    """

    mean_entry_value: float
    informal_count_share_target: float
    informal_value_threshold: int
    strata: int
    sigma: float
    achieved_informal_count_share: float
    achieved_mean_entry_value: float
    total_weighted_entries: int
    anchor_provenance: dict[str, Any]

    def register(self) -> dict[str, Any]:
        """The assumptions register, written beside every generated file."""
        return {
            "synthetic": True,
            "statement": (
                "Synthetic entries; only the (HTS-10 x country x month) "
                "customs-value margins are calibrated truth. Entry counts "
                "and sizes are an assumption: within-cell lognormal sizes, "
                "with per-cell counts pinned by CBP's published national "
                "mean entry value and the shape sigma proxy-calibrated so "
                "the share of entries at or below $2,500 of customs value "
                "matches CBP's published informal-entry count share. The "
                "informal share is an approximate size-distribution moment, "
                "not a statutory identification — see "
                "known_gaps.informal_share_proxy."
            ),
            "size_model": {
                **asdict(self),
                "sigma_calibration_class": "proxy_moment_match",
            },
            "sensitivity": (
                "Margins are invariant to this assumption by construction; "
                "entry-count and size-dependent statistics (for example "
                "de minimis-style threshold exposure) are not. The "
                "assumption concentrates in sigma and mean_entry_value; "
                "re-run the generator with overridden anchors to measure "
                "sensitivity."
            ),
            "known_gaps": {
                "informal_share_proxy": (
                    "Sigma equates CBP's 'Total Informal Entry Summaries' "
                    "count share with the share of entries at or below "
                    "$2,500 of customs value. 19 CFR 143.21 defines "
                    "informal entry by entry-type rules beyond that value "
                    "clause — installment shipments and portions of "
                    "consignments, personal and household effects, certain "
                    "institutional imports, some returned US products up "
                    "to $10,000 — with Chapter 99 exclusions, so the "
                    "published informal count is not the CDF of customs "
                    "value at $2,500. No public moment maps categories to "
                    "values, so the share is admitted as an approximate "
                    "moment; threshold-sensitive statistics inherit this "
                    "category-mapping error in addition to the lognormal "
                    "shape assumption."
                ),
                "is_postal_shipment": (
                    "All entries carry False. The bulk IMDB margins carry "
                    "air/vessel/containerized transport splits but no "
                    "postal marker, and Census rate-provision codes "
                    "classify free-vs-dutiable status only (verified "
                    "against census.gov/foreign-trade/reference/"
                    "rpcodes.html) — no public series grounds a postal "
                    "flag at any margin grain. Postal de minimis analyses "
                    "need a source before this flag is meaningful."
                ),
                "shipment_value": (
                    "Set equal to customs_value; no separate shipment-value "
                    "margin is published."
                ),
                "entry_date": (
                    f"Day {_ENTRY_DAY_OF_MONTH} of the margin month; "
                    "sub-monthly timing is not calibrated."
                ),
                "entry_counts": (
                    "Per-cell counts derive from the national mean entry "
                    "value; no public HTS- or country-level entry-count "
                    "split exists. The bulk IMDB detail's cards_mo column "
                    "is defined as 'Number of Detailed Records, Current "
                    "Month' — statistical detail records, not entry "
                    "summaries — so it is not admitted as an entry-count "
                    "anchor. Cells smaller than the mean are floored at "
                    "one entry, so the total weighted entry count exceeds "
                    "the value-implied count."
                ),
            },
        }


def dotted_hts(hts10: str) -> str:
    """``7202111000`` -> ``7202.11.10.00`` (the spine's input format)."""
    if len(hts10) != 10 or not hts10.isdigit():
        raise ValueError(f"HTS-10 code {hts10!r} is not ten digits.")
    return f"{hts10[0:4]}.{hts10[4:6]}.{hts10[6:8]}.{hts10[8:10]}"


def solve_size_assumption(
    cell_values: np.ndarray,
    *,
    mean_entry_value: float,
    informal_count_share: float,
    informal_value_threshold: int = DEFAULT_INFORMAL_VALUE_THRESHOLD,
    strata: int = DEFAULT_STRATA,
    anchor_provenance: dict[str, Any] | None = None,
) -> EntrySizeAssumption:
    """Solve the lognormal shape against the informal-share anchor.

    ``cell_values`` are the window's per-cell customs-value margins in
    integer dollars. Bisection drives the generated mixture's share of
    entries at or below the threshold to the target; if the target is
    outside the reachable range the boundary sigma is kept and the achieved
    share is recorded (never silently presented as matched).
    """
    values = _checked_cell_values(cell_values)
    if not 0.0 < informal_count_share < 1.0:
        raise ValueError(
            f"Informal count share {informal_count_share!r} must be in (0, 1)."
        )
    if mean_entry_value <= 1.0:
        raise ValueError(f"Mean entry value {mean_entry_value!r} must exceed $1.")
    counts = _cell_entry_counts(values, mean_entry_value)

    def share(sigma: float) -> float:
        weights, entry_values, _, _ = _stratify(values, counts, sigma, strata)
        informal = weights[entry_values <= informal_value_threshold].sum()
        return float(informal) / float(weights.sum())

    low, high = _SIGMA_BOUNDS
    share_low, share_high = share(low), share(high)
    if informal_count_share <= share_low:
        sigma = low
    elif informal_count_share >= share_high:
        sigma = high
    else:
        sigma = (low + high) / 2.0
        for _ in range(_SIGMA_ITERATIONS):
            sigma = (low + high) / 2.0
            mid_share = share(sigma)
            if abs(mid_share - informal_count_share) <= _SHARE_TOLERANCE:
                break
            if mid_share < informal_count_share:
                low = sigma
            else:
                high = sigma
    achieved = share(sigma)
    total_entries = int(counts.sum())
    return EntrySizeAssumption(
        mean_entry_value=float(mean_entry_value),
        informal_count_share_target=float(informal_count_share),
        informal_value_threshold=int(informal_value_threshold),
        strata=int(strata),
        sigma=float(sigma),
        achieved_informal_count_share=float(achieved),
        achieved_mean_entry_value=float(values.sum()) / float(total_entries),
        total_weighted_entries=total_entries,
        anchor_provenance=dict(anchor_provenance or {}),
    )


def generate_entries(
    margins: pd.DataFrame,
    assumption: EntrySizeAssumption,
) -> pd.DataFrame:
    """Generate the weighted synthetic entries table from margin cells.

    ``margins`` is the P1 tidy margins table (one row per nonzero
    HTS-10 x country x month cell); rows with zero imports-for-consumption
    value carry no consumption entries and are skipped. The result
    reproduces every cell's ``con_val_mo`` exactly:
    ``sum(weight * customs_value) == con_val_mo`` per cell, in integers.
    """
    required = {"period", "hts10", "cty_code", "iso2", "con_val_mo"}
    missing = sorted(required - set(margins.columns))
    if missing:
        raise ValueError(f"Margins table is missing column(s) {missing}.")
    # Canonical cell order: never inherit the input file's row order. Any
    # permutation of the same margin cells generates identical rows.
    cells = (
        margins.loc[margins["con_val_mo"] > 0]
        .sort_values(["period", "hts10", "cty_code"], kind="stable")
        .reset_index(drop=True)
    )
    if cells.empty:
        raise ValueError("No margin cells with positive consumption value.")
    duplicated = cells.duplicated(["period", "hts10", "cty_code"])
    if duplicated.any():
        first = cells.loc[duplicated].iloc[0]
        raise ValueError(
            "Margins table carries duplicate cells for "
            f"({first['period']}, {first['hts10']}, {first['cty_code']}); "
            "each (period, hts10, cty_code) must appear once."
        )
    values = _checked_cell_values(cells["con_val_mo"].to_numpy(dtype=np.int64))
    counts = _cell_entry_counts(values, assumption.mean_entry_value)
    weights, entry_values, cell_index, stratum = _stratify(
        values, counts, assumption.sigma, assumption.strata
    )
    plus_one = np.zeros(len(weights), dtype=bool)
    plus_one[1:] = (cell_index[1:] == cell_index[:-1]) & (stratum[1:] == stratum[:-1])

    periods = cells["period"].astype(str).to_numpy()[cell_index]
    hts10 = cells["hts10"].astype(str).to_numpy()[cell_index]
    cty = cells["cty_code"].astype(str).to_numpy()[cell_index]
    iso2 = cells["iso2"].astype(str).to_numpy()[cell_index]

    part = np.where(plus_one, "b", "a")
    entry_ids = [
        f"synthetic-{period}-{code}-{country}-s{layer}{suffix}"
        for period, code, country, layer, suffix in zip(
            periods, hts10, cty, stratum, part, strict=True
        )
    ]
    entry_dates = [
        date(int(period[:4]), int(period[5:7]), _ENTRY_DAY_OF_MONTH)
        for period in periods
    ]
    frame = pd.DataFrame(
        {
            "entry_id": pd.array(entry_ids, dtype="string"),
            "entry_date": entry_dates,
            "period": pd.array(periods, dtype="string"),
            "hts_number": pd.array(
                [dotted_hts(code) for code in hts10], dtype="string"
            ),
            "hts10": pd.array(hts10, dtype="string"),
            "chapter": pd.array([code[:2] for code in hts10], dtype="string"),
            "country_of_origin": pd.array(iso2, dtype="string"),
            "census_country_code": pd.array(cty, dtype="string"),
            "customs_value": entry_values.astype(np.int64),
            "shipment_value": entry_values.astype(np.int64),
            "is_postal_shipment": np.zeros(len(weights), dtype=bool),
            # Row-level synthetic marker: schema metadata and id prefixes do
            # not survive every conversion (CSV exports, database loads),
            # a column does.
            "is_synthetic": np.ones(len(weights), dtype=bool),
            "weight": weights.astype(np.int64),
            "size_stratum": stratum.astype(np.int8),
        }
    )
    # _stratify emits rows lexsorted by (cell, stratum, value) over the
    # canonically sorted cells, so the frame order is deterministic and
    # input-permutation-invariant.
    return frame


def validate_entries_against_margins(
    entries: pd.DataFrame,
    margins: pd.DataFrame,
) -> dict[str, Any]:
    """Exact-reproduction check of every calibrated cell, plus schema checks.

    Raises ``ValueError`` on any failure; returns a validation report on
    success. The margin comparison is exact integer equality on every
    (period, hts10, census country) cell with positive consumption value —
    no tolerances.
    """
    failures: list[str] = []
    expected = (
        margins.loc[margins["con_val_mo"] > 0]
        .set_index(["period", "hts10", "cty_code"])["con_val_mo"]
        .astype("int64")
        .sort_index()
    )
    weighted = entries["weight"].to_numpy() * entries["customs_value"].to_numpy()
    produced = (
        pd.Series(weighted, name="value")
        .groupby(
            [
                entries["period"].astype(str),
                entries["hts10"].astype(str),
                entries["census_country_code"].astype(str),
            ]
        )
        .sum()
        .astype("int64")
        .rename_axis(["period", "hts10", "cty_code"])
        .sort_index()
    )
    if len(produced) != len(expected):
        failures.append(
            f"Cell count mismatch: {len(produced)} generated vs "
            f"{len(expected)} margin cells."
        )
    else:
        if not produced.index.equals(expected.index):
            failures.append("Generated cells do not match margin cells.")
        else:
            mismatched = produced[produced.to_numpy() != expected.to_numpy()]
            for (period, hts10, cty), value in mismatched.head(20).items():
                failures.append(
                    f"{period} HTS {hts10} country {cty}: generated "
                    f"{int(value)} != margin {int(expected.loc[(period, hts10, cty)])}."
                )
            if len(mismatched) > 20:
                failures.append(f"...and {len(mismatched) - 20} more cell mismatches.")

    if (entries["customs_value"] <= 0).any():
        failures.append("Entries include non-positive customs values.")
    if (entries["weight"] <= 0).any():
        failures.append("Entries include non-positive weights.")
    if not (entries["shipment_value"] == entries["customs_value"]).all():
        failures.append("shipment_value must equal customs_value (documented rule).")
    if entries["is_postal_shipment"].any():
        failures.append("is_postal_shipment must be uniformly False (documented gap).")
    if "is_synthetic" not in entries.columns or not entries["is_synthetic"].all():
        failures.append("Every row must carry is_synthetic=True.")
    if not entries["entry_id"].str.startswith("synthetic-").all():
        failures.append("Every entry_id must carry the synthetic- prefix.")
    dotted = entries["hts_number"].str.fullmatch(r"\d{4}\.\d{2}\.\d{2}\.\d{2}")
    if not dotted.all():
        failures.append("hts_number must be dotted 4.2.2.2 form.")
    if failures:
        raise ValueError(
            "Synthetic entries failed validation:\n" + "\n".join(failures[:25])
        )
    weights = entries["weight"].to_numpy(dtype=np.int64)
    # Aggregate totals in Python ints: every per-cell sum is bounded by its
    # margin value (<= 2**53), but the window total is not bounded by
    # int64, so the grand totals must never route through fixed-width
    # accumulation.
    total_value = sum(int(value) for value in produced.to_numpy())
    total_weight = sum(int(weight) for weight in weights)
    return {
        "cells_checked": int(len(expected)),
        "cells_exact": int(len(expected)),
        "entry_rows": int(len(entries)),
        "weighted_entries": total_weight,
        "weighted_customs_value": total_value,
        "exact": True,
    }


def _checked_cell_values(cell_values: np.ndarray) -> np.ndarray:
    values = np.asarray(cell_values, dtype=np.int64)
    if values.size == 0:
        raise ValueError("No margin cells supplied.")
    if (values <= 0).any():
        raise ValueError("Margin cell values must be positive integers.")
    if (values > MAX_EXACT_CELL_VALUE).any():
        largest = int(values.max())
        raise ValueError(
            f"Margin cell value {largest} exceeds MAX_EXACT_CELL_VALUE "
            f"(2**53 = {MAX_EXACT_CELL_VALUE}): the largest-remainder "
            "residual split is only proven exact where integer dollars are "
            "exactly representable in float64. Real Census cells are "
            "orders of magnitude below this bound; refusing rather than "
            "allocating inexactly."
        )
    return values


def _cell_entry_counts(values: np.ndarray, mean_entry_value: float) -> np.ndarray:
    """Entry count per cell: value-implied, floored at one, capped at $1/entry."""
    implied = np.rint(values.astype(np.float64) / float(mean_entry_value))
    return np.clip(implied, 1, values).astype(np.int64)


def _stratum_shapes(strata: int, sigma: float) -> list[np.ndarray]:
    """Relative stratum mean sizes for each effective stratum count 1..K.

    For ``k`` equal-probability strata of ``LogNormal(0, sigma)``, the
    stratum-mean shape is ``k * (Phi(z_upper - sigma) - Phi(z_lower - sigma))``
    (mean one across strata), from the truncated-lognormal mean identity.
    """
    shapes: list[np.ndarray] = []
    for count in range(1, strata + 1):
        edges = norm.ppf(np.linspace(0.0, 1.0, count + 1))
        mass = np.diff(norm.cdf(edges - sigma))
        shapes.append(count * mass)
    return shapes


def _stratify(
    values: np.ndarray,
    counts: np.ndarray,
    sigma: float,
    strata: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split cells into weighted integer-valued rows, exactly per cell.

    Returns ``(weights, entry_values, cell_index, stratum)`` where per-cell
    ``sum(weights * entry_values)`` equals the cell value exactly. Each
    cell contributes ``min(strata, count)`` strata and each stratum at most
    two rows: the largest-remainder split of the stratum total into a base
    per-entry value and a +$1 remainder row.
    """
    if strata < 1:
        raise ValueError(f"Strata count {strata!r} must be at least one.")
    cell_count = len(values)
    effective = np.minimum(counts, strata).astype(np.int64)
    width = int(effective.max())
    columns = np.arange(width)
    active = columns[None, :] < effective[:, None]

    # Even split of each cell's entry count across its strata
    # (largest-remainder: lower strata take the excess).
    base = counts[:, None] // np.where(active, effective[:, None], 1)
    extra = columns[None, :] < (counts[:, None] % effective[:, None])
    stratum_counts = np.where(active, base + extra, 0)

    shapes = _stratum_shapes(strata, sigma)
    shape = np.zeros((cell_count, width))
    for count in range(1, width + 1):
        rows = effective == count
        if rows.any():
            shape[rows, :count] = shapes[count - 1]

    # Integer stratum value totals: $1-per-entry minimum, then the
    # remainder by largest-remainder on count-weighted lognormal shares.
    remainder = (values - counts).astype(np.float64)
    weight_shape = stratum_counts * shape
    share = weight_shape / np.where(
        weight_shape.sum(axis=1, keepdims=True) > 0,
        weight_shape.sum(axis=1, keepdims=True),
        1.0,
    )
    quota = remainder[:, None] * share
    floor = np.floor(quota).astype(np.int64)
    shortfall = (values - counts - floor.sum(axis=1)).astype(np.int64)
    # The +$1 top-up can add at most one dollar per active stratum, so
    # exactness requires 0 <= shortfall <= active strata per cell. Within
    # the MAX_EXACT_CELL_VALUE domain the float64 quotas are exact enough
    # to guarantee this; assert it anyway so any violation fails the build
    # loudly instead of shipping an inexact allocation.
    if (shortfall < 0).any() or (shortfall > effective).any():
        offender = int(np.argmax((shortfall < 0) | (shortfall > effective)))
        raise AssertionError(
            "Largest-remainder shortfall out of bounds for cell "
            f"{offender}: shortfall={int(shortfall[offender])}, active "
            f"strata={int(effective[offender])}, value="
            f"{int(values[offender])}, count={int(counts[offender])}. "
            "The allocation cannot be completed exactly; this is a bug "
            "or an out-of-domain input, never a rounding to absorb."
        )
    fraction = np.where(active, quota - floor, -1.0)
    order = np.argsort(-fraction, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, columns[None, :].repeat(cell_count, 0), axis=1)
    topped = rank < shortfall[:, None]
    totals = stratum_counts + floor + topped.astype(np.int64)

    safe_counts = np.where(stratum_counts > 0, stratum_counts, 1)
    per_entry = totals // safe_counts
    plus_one = totals - per_entry * safe_counts

    cell_grid = np.arange(cell_count)[:, None].repeat(width, 1)
    base_rows = stratum_counts - plus_one
    flat = {
        "cell": np.concatenate([cell_grid[active], cell_grid[active]]),
        "stratum": np.concatenate([columns[None, :].repeat(cell_count, 0)[active]] * 2),
        "weight": np.concatenate([base_rows[active], plus_one[active]]),
        "value": np.concatenate([per_entry[active], per_entry[active] + 1]),
    }
    keep = flat["weight"] > 0
    order_rows = np.lexsort(
        (flat["value"][keep], flat["stratum"][keep], flat["cell"][keep])
    )
    return (
        flat["weight"][keep][order_rows],
        flat["value"][keep][order_rows],
        flat["cell"][keep][order_rows],
        flat["stratum"][keep][order_rows],
    )


def assumption_to_json(assumption: EntrySizeAssumption) -> str:
    """Serialized assumptions register (stable key order)."""
    return json.dumps(assumption.register(), indent=2, sort_keys=True) + "\n"
