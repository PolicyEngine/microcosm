"""Engine-free county rung for the ACS local release.

County SOI facts supply *shares*, never a second state level. Children split
the stored state CSR row, including its return universe and engine values.
Population, splitting, holdout and reliability helpers also accept future ACS
county families. See docs/us-acs-local-county-rung.md for the input contract.

County facts come from their own pinned file (``--county-facts``, pinned by
``us/chronicle_county_facts.json``, microcosm#922), never from ``--feed``:
:func:`prepare_run` loads it once into ``surface.county_facts``; county steps
read ``surface.county_facts.rows`` and never reopen the file or ``--feed``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import shlex
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import us_acs_local_cd_surface as cd_surface
from scipy import sparse

COUNTY_COLUMN = "county_fips"
COUNTY_HOLDOUT_SALT = "acs-local-county-v1"
DEFAULT_COUNTY_HOLDOUT_FRACTION = 0.1
RELIABILITY_FILENAME = "county_reliability.json"
RELIABILITY_CSV_FILENAME = "county_reliability.csv"
SUPPORT_FILENAME = "county_support.json"
SURFACE_FILENAME = "county_surface.json"
#: Materialize-time validation inputs (microcosm#922): the public county
#: design and the household measures it is scored on. Neither is a target:
#: they never enter the registry, roles, matrix or loss.
VALIDATION_DESIGN_FILENAME = "county_validation.json"
VALIDATION_MEASURES_FILENAME = "county_validation_measures.npz"
SOI_MODE = "state_county"
COUNTY_PREFIX = "0500000US"
# The LWI release contract covers the 50 states and DC on planning-region
# county boundaries. A self-consistent but incomplete ladder is not complete.
REQUIRED_COUNTY_COUNT = 3_144
COUNTY_SOI_REQUIRED_MEASURES = ("return_count", "adjusted_gross_income")
#: Puerto Rico (state 72) is outside the County Files. The county facts loader
#: drops its municipios with a count; a row that still reaches this module is
#: a defect, never a county to fit.
PUERTO_RICO_STATE = 72

#: Counties absent from an IRS SOI county file, each merged into one declared
#: host county of the same state for the SOI families only. The IRS 2022
#: county users guide (https://www.irs.gov/pub/irs-soi/22incydocguide.docx)
#: says items with fewer than 20 returns for a county were excluded, and
#: 22incyallnoagi.csv has no row for Kalawao County, HI (15005). Its SOI value
#: is therefore missing, not zero; the guide does not say whether its returns
#: are counted in another county's row. Kalawao is merged into Maui (15009),
#: which surrounds it: the combined cell's share is Maui's published value
#: over the state's published county sum, and its households are those in
#: either county. Kalawao keeps its own ladder population, ACS household and
#: validation rows. Any other absent county is refused unless declared here.
DECLARED_SOI_COMBINED_CELLS = {15005: 15009}
SOI_COMBINED_CELL_BASIS = (
    "IRS SOI 2022 county users guide (22incydocguide.docx): items with fewer "
    "than 20 returns for a county were excluded; 22incyallnoagi.csv has no row "
    "for this county, so its SOI value is missing, not zero, and it is "
    "merged into its declared host county for the SOI families only."
)


def _concatenate(parts, dtype):
    return (
        np.concatenate(parts).astype(dtype, copy=False) if parts else np.empty(0, dtype)
    )


def _codes(values, *, column: str) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(float)
    if not np.isfinite(numeric).all() or (numeric != np.floor(numeric)).any():
        raise ValueError(f"{column} must contain finite integer geography codes.")
    return numeric.astype(np.int64)


def county_codes(households, *, county_column=COUNTY_COLUMN) -> np.ndarray:
    counties = _codes(households[county_column], column=county_column)
    states = _codes(households["state_fips"], column="state_fips")
    if ((counties < 1001) | (counties > 56999)).any():
        raise ValueError(f"{county_column} contains invalid five-digit county FIPS.")
    if not np.array_equal(counties // 1000, states):
        raise ValueError(f"{county_column} disagrees with household state_fips.")
    return counties


def with_county_populations(ladder, populations, geographies):
    """Aggregate ladder overlaps, then reconcile counties to state levels."""
    if "county" not in geographies:
        return populations
    result = {key: dict(value) for key, value in populations.items()}
    if "state" not in result:
        states = np.asarray(ladder.puma) // 100_000
        result["state"] = {
            int(state): float(np.asarray(ladder.puma_population)[states == state].sum())
            for state in np.unique(states)
        }
    counties = _codes(ladder.county_overlap_county, column="county_overlap_county")
    values = np.asarray(ladder.county_overlap_population, dtype=float)
    if (
        len(counties) != len(values)
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        raise ValueError("County overlap populations must be finite and nonnegative.")
    raw = {
        int(code): math.fsum(values[counties == code]) for code in np.unique(counties)
    }
    result["county"] = {}
    for state, target in sorted(result["state"].items()):
        block = sorted(code for code in raw if code // 1000 == int(state))
        total = math.fsum(raw[code] for code in block)
        if not block or total <= 0 or not math.isfinite(target) or target <= 0:
            raise ValueError(f"State {state} has no positive county/state population.")
        scaled = [float(target) * raw[code] / total for code in block]
        # Assign the rounding residual to the largest cell, never a small county.
        largest = max(range(len(block)), key=lambda index: scaled[index])
        scaled[largest] += float(target) - math.fsum(scaled)
        result["county"].update(zip(block, scaled, strict=True))
    unknown = sorted(set(raw) - set(result["county"]))
    if unknown:
        raise ValueError(f"County populations have no state parent: {unknown}.")
    return result


def population_rows(
    households,
    household_size,
    ladder_populations,
    geographies,
    *,
    county_column=COUNTY_COLUMN,
):
    """Sparse mixed-geography rows; unsupported county cells are named."""
    other = [geography for geography in geographies if geography != "county"]
    records, matrix, dropped = cd_surface.population_rows(
        households, household_size, ladder_populations, other
    )
    if "county" not in geographies:
        return records, matrix, dropped
    codes = county_codes(households, county_column=county_column)
    populations = ladder_populations["county"]
    unknown = sorted(set(codes) - set(populations))
    if unknown:
        raise ValueError(f"Households name counties absent from the ladder: {unknown}.")
    size = np.asarray(household_size, dtype=np.float32)
    if (
        size.shape != (len(households),)
        or not np.isfinite(size).all()
        or (size <= 0).any()
    ):
        raise ValueError("Every county household must have a positive finite size.")
    rows, cols, data, county_records = [], [], [], []
    for code, population in sorted(populations.items()):
        name = f"pop_county_{int(code):05d}"
        positions = np.flatnonzero(codes == int(code))
        if not len(positions):
            dropped.append(name)
            continue
        rows.append(np.full(len(positions), len(county_records), np.int32))
        cols.append(positions.astype(np.int32))
        data.append(size[positions])
        county_records.append(
            {
                "name": name,
                "value": float(population),
                "source": "us_puma_ladder_2020",
                "family": "census_population_ladder",
                "geography_level": "county",
                "county_fips": f"{int(code):05d}",
                "state_fips": f"{int(code) // 1000:02d}",
            }
        )
    county_matrix = sparse.csr_array(
        (
            _concatenate(data, np.float32),
            (_concatenate(rows, np.int32), _concatenate(cols, np.int32)),
        ),
        shape=(len(county_records), len(households)),
    )
    return (
        records + county_records,
        sparse.vstack([matrix, county_matrix], format="csr"),
        dropped,
    )


def _hierarchy(name, code, *, parent=None):
    from microcosm.calibrate import (
        CalibrationHierarchy,
        HierarchyCategory,
        HierarchyGeography,
        HierarchyNode,
    )

    # One deterministic label for a county in all families. Source labels are
    # retained in metadata; no mutable county-name registry is needed.
    geography = HierarchyGeography(f"{COUNTY_PREFIX}{code}", f"County {code}", "county")
    if parent is not None and parent.hierarchy is not None:
        return dataclasses.replace(
            parent.hierarchy,
            geography=geography,
            target=HierarchyNode(
                name, f"{parent.hierarchy.target.label}, county {code}"
            ),
        )
    provider = HierarchyNode("census_population", "Census population")
    return CalibrationHierarchy(
        provider=provider,
        category=HierarchyCategory(
            "census_population.resident_population", "Resident population", provider.id
        ),
        geography=geography,
        dimensions=(),
        target=HierarchyNode(name, f"County {code} resident population"),
    )


def population_spec(name, value, *, period=2024):
    from microcosm.calibrate import TargetSpec

    code = name.removeprefix("pop_county_")
    return TargetSpec(
        name=name,
        measure=name,
        entity="household",
        value=value,
        period=period,
        source="US Census Bureau 2020 PUMA county population overlap, state-normalized",
        family="census_population",
        metadata={"geography_level": "county", "county_fips": code},
        hierarchy=_hierarchy(name, code),
    )


def _broad_parent(spec):
    metadata = spec.metadata
    return (
        spec.family == "irs_soi"
        and metadata.get("ledger_geography_level") == "state"
        and str(metadata.get("filing_status", "all")).lower() == "all"
        and float(metadata.get("agi_lower_bound", "-inf")) == -math.inf
        and float(metadata.get("agi_upper_bound", "inf")) == math.inf
        and metadata.get("soi_return_universe", "all_returns") == "all_returns"
        and metadata.get("itemized_only", "false").lower() != "true"
        and metadata.get("taxable_only", "false").lower() != "true"
        and spec.filter is None
        and all(
            str(metadata.get(key, "all")).lower() == "all"
            for key in ("ledger_filter_filing_status", "ledger_filter_income_range")
        )
    )


@dataclasses.dataclass(frozen=True)
class CountySurface:
    specs: tuple
    receipt: dict


def county_code_of(fact, *, source):
    """The five-digit county code of a county fact, refusing bad geography.

    The row geography is authoritative. A layout group value, when present,
    must name the same county (a FIPS/state mismatch is refused), and a
    Puerto Rico row reaching the rung is refused explicitly.
    """
    geography = fact.get("geography", {})
    if geography.get("level") != "county":
        raise ValueError(f"{source} county facts must be county rows: {geography!r}.")
    geoid = str(geography.get("id", ""))
    digits = geoid[len(COUNTY_PREFIX) :]
    if (
        not geoid.startswith(COUNTY_PREFIX)
        or len(digits) != 5
        or not (digits.isascii() and digits.isdigit())
    ):
        raise ValueError(f"Invalid county {source} geography {geoid!r}.")
    code = int(digits)
    if code // 1000 == PUERTO_RICO_STATE:
        raise ValueError(
            f"Puerto Rico {source} county fact {geoid} reached the county rung; "
            "state 72 is outside the 3,144 County Files county-equivalents and "
            "the county facts loader must drop it."
        )
    if not 1 <= code // 1000 <= 56 or code % 1000 == 0:
        raise ValueError(f"Invalid county {source} geography {geoid!r}.")
    group = (fact.get("layout") or {}).get("groupby_value_id")
    if group is not None and str(group)[-5:] != digits:
        raise ValueError(
            f"{source} county fact {geoid} names layout county {group!r}; "
            "geography and layout disagree on the county FIPS."
        )
    return code


def _validated_combined_cells(combined_cells):
    cells = {int(absent): int(host) for absent, host in dict(combined_cells).items()}
    for absent, host in cells.items():
        if absent == host or absent // 1000 != host // 1000:
            raise ValueError(
                f"Declared SOI combined cell {absent:05d}->{host:05d} must join two "
                "distinct counties of one state."
            )
        if host in cells:
            raise ValueError(
                f"Declared SOI combined cell host {host:05d} is itself merged."
            )
    return cells


def state_county_soi_surface(
    state_specs,
    facts,
    county_populations,
    *,
    combined_cells=DECLARED_SOI_COMBINED_CELLS,
):
    """Rebase complete county fact blocks onto existing broad state totals.

    Every state must carry county return count and AGI with unsliced state
    parents. Other measures require both county facts and a state parent.
    A county absent from a block is refused unless ``combined_cells`` declares
    it, with its host present: then one child row covers both counties, its
    share is the host's published value over the state's published sum, and
    ``state_county_cell_members`` lists the member counties.
    """
    combined = _validated_combined_cells(combined_cells)
    blocks = defaultdict(dict)
    for fact in facts:
        observed = fact.get("observed_measure", {})
        if observed.get("source_name") != "irs_soi":
            continue
        code = county_code_of(fact, source="SOI")
        dimensions = fact.get("dimensions", {})
        if any(
            str(dimensions.get(key, "all")).lower() != "all"
            for key in ("filing_status", "income_range")
        ):
            raise ValueError(
                "County SOI requires all-return facts without AGI classes."
            )
        measure = observed.get("source_measure_id")
        period = str(fact.get("period", {}).get("value", ""))
        if fact.get("period", {}).get("type") != "tax_year":
            raise ValueError("County SOI facts must carry a tax_year period.")
        key = (code // 1000, measure, period)
        if code in blocks[key]:
            raise ValueError(f"Duplicate county SOI fact for {key}, {code:05d}.")
        value = float(fact["value"])
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"County SOI shares require finite nonnegative values: {code:05d}."
            )
        blocks[key][code] = (value, fact)
    if not blocks:
        raise ValueError(
            "County facts carry no IRS SOI county rows; pass the pinned "
            "--county-facts file (us/chronicle_county_facts.json, microcosm#922)."
        )
    parents = {}
    for spec in state_specs:
        if not _broad_parent(spec):
            continue
        metadata = spec.metadata
        state = int(metadata["state_fips"])
        key = (
            state,
            metadata.get("source_measure_id"),
            str(metadata.get("source_period", "")),
        )
        if key[1] not in COUNTY_SOI_REQUIRED_MEASURES and key[1] not in {
            block[1] for block in blocks
        }:
            continue
        if key in parents:
            raise ValueError(f"Two state SOI parents for county concept {key}.")
        parents[key] = spec
    if not parents:
        raise ValueError(
            "No county SOI concept has an already materialized broad state parent."
        )
    for state in sorted({int(code) // 1000 for code in county_populations}):
        for measure in COUNTY_SOI_REQUIRED_MEASURES:
            if not any(key[:2] == (state, measure) for key in parents):
                raise ValueError(
                    f"Missing broad state SOI parent state={state:02d}, concept={measure}; "
                    "state_county requires return count and AGI in every state."
                )
    specs, receipts, merges = [], [], defaultdict(list)
    for key, parent in sorted(parents.items()):
        state, measure, period = key
        expected = {
            int(code) for code in county_populations if int(code) // 1000 == state
        }
        block = blocks.get(key, {})
        absent = expected - block.keys()
        merged = {
            code: combined[code]
            for code in absent
            if code in combined
            and combined[code] in block
            and combined[code] in expected
        }
        missing, extra = (
            sorted(absent - merged.keys()),
            sorted(block.keys() - expected),
        )
        if not expected or missing or extra:
            raise ValueError(
                f"Incomplete county SOI block state={state:02d}, concept={measure}, period={period}; "
                f"missing={[f'{code:05d}' for code in missing]}, unexpected={[f'{code:05d}' for code in extra]}. "
                "County facts, ladder and county column must use matching boundaries; "
                "only a declared combined SOI cell may cover an absent county."
            )
        members = {code: [code] for code in sorted(block)}
        for code, host in sorted(merged.items()):
            members[host].append(code)
            merges[(host, code)].append(f"{measure}|{period}")
        total = math.fsum(value for value, _fact in block.values())
        if total <= 0:
            raise ValueError(
                f"County SOI block {key} has no positive share denominator."
            )
        codes = sorted(block)
        shares = [block[code][0] / total for code in codes]
        largest = max(range(len(codes)), key=lambda index: shares[index])
        shares[largest] += 1.0 - math.fsum(shares)
        targets = [parent.value * share for share in shares]
        targets[largest] += parent.value - math.fsum(targets)
        state_population = math.fsum(county_populations[code] for code in expected)
        for code, share, target in zip(codes, shares, targets, strict=True):
            raw, fact = block[code]
            unit = fact.get("observed_measure", {}).get("unit")
            if (
                parent.metadata.get("ledger_measure_unit") is not None
                and unit != parent.metadata["ledger_measure_unit"]
            ):
                raise ValueError(
                    f"County SOI units differ from state parent {parent.name}: {unit!r}."
                )
            county = f"{code:05d}"
            cell = sorted(members[code])
            name = f"county_{county}__{parent.name}"
            metadata = {
                **parent.metadata,
                "ledger_geography_level": "county",
                "ledger_geography_id": f"{COUNTY_PREFIX}{county}",
                "ledger_geography_vintage": str(
                    fact["geography"].get("vintage", period)
                ),
                "ledger_layout_record_set_spec_id": str(
                    fact.get("layout", {}).get("record_set_spec_id", "county_soi")
                ),
                "county_fips": county,
                "county_label": str(fact["geography"].get("name", county)),
                "state_county_parent_target_name": parent.name,
                "state_county_share": str(share),
                "state_county_source_value": str(raw),
                "county_population": str(
                    math.fsum(county_populations[member] for member in cell)
                ),
                "state_population": str(state_population),
            }
            if len(cell) > 1:
                metadata["state_county_cell_members"] = ",".join(
                    f"{member:05d}" for member in cell
                )
                metadata["state_county_cell_basis"] = SOI_COMBINED_CELL_BASIS
            specs.append(
                dataclasses.replace(
                    parent,
                    name=name,
                    measure=name,
                    value=target,
                    se=None,
                    metadata=metadata,
                    hierarchy=_hierarchy(name, county, parent=parent),
                    source=f"{parent.source}; IRS SOI county {period} within-state shares",
                )
            )
        receipts.append(
            {
                "state_fips": f"{state:02d}",
                "concept": measure,
                "parent": parent.name,
                "counties": len(expected),
                "cells": len(codes),
                "share_sum": math.fsum(shares),
                "target_sum": math.fsum(targets),
            }
        )
    # Counties without a state parent cannot silently disappear.
    covered = {int(member) for spec in specs for member in cell_members(spec.metadata)}
    missing = sorted(set(county_populations) - covered)
    if missing:
        raise ValueError(f"Counties have no materialized SOI state parent: {missing}.")
    return CountySurface(
        tuple(specs),
        {
            "soi_mode": SOI_MODE,
            "county_targets": len(specs),
            "concepts": sorted({key[1] for key in parents}),
            "blocks": receipts,
            "combined_cells": [
                {
                    "members": sorted((f"{host:05d}", f"{absent:05d}")),
                    "host": f"{host:05d}",
                    "absent_from_soi": f"{absent:05d}",
                    "concepts": sorted(concepts),
                    "basis": SOI_COMBINED_CELL_BASIS,
                    "source": "https://www.irs.gov/pub/irs-soi/22incydocguide.docx",
                    "own_rows_kept": "ladder population, ACS households and "
                    "ACS validation rows stay per county",
                }
                for (host, absent), concepts in sorted(merges.items())
            ],
            "new_engine_columns": 0,
            "basis": "state target x county fact / state sum of county facts",
        },
    )


def cell_members(metadata):
    """The county codes one county child row covers (one unless combined)."""
    members = metadata.get("state_county_cell_members")
    if members:
        return [int(member) for member in str(members).split(",")]
    return [int(metadata["county_fips"])]


def split_state_rows(
    state_specs, state_matrix, county_specs, households, *, county_column=COUNTY_COLUMN
):
    """Partition stored parent rows exactly; never materialize new columns.

    A child row covers its member set (``cell_members``): one county, or a
    declared combined SOI cell. Member sets of one parent never overlap and
    cover every household of the parent's state, so each household falls in
    exactly one child row of each parent.
    """
    codes = county_codes(households, county_column=county_column)
    matrix = sparse.csr_array(state_matrix)
    if matrix.shape != (len(state_specs), len(households)):
        raise ValueError(
            "State target matrix is not aligned with specs and households."
        )
    by_name = {spec.name: index for index, spec in enumerate(state_specs)}
    blocks = defaultdict(list)
    for index, spec in enumerate(county_specs):
        blocks[spec.metadata["state_county_parent_target_name"]].append(index)
    rows, cols, data = [], [], []
    for parent_name, indices in blocks.items():
        parent_index = by_name[parent_name]
        parent = state_specs[parent_index]
        child_members = [
            cell_members(county_specs[index].metadata) for index in indices
        ]
        flat = [code for members in child_members for code in members]
        if len(set(flat)) != len(flat):
            raise ValueError(f"County rows of {parent_name} overlap.")
        state = int(parent.metadata["state_fips"])
        if any(code // 1000 != state for code in flat):
            raise ValueError(f"County rows of {parent_name} leave its state.")
        present = codes // 1000 == state
        if not np.isin(codes[present], flat).all():
            raise ValueError(f"Households of {parent_name} fall in no county row.")
        start, end = matrix.indptr[parent_index : parent_index + 2]
        positions, values = matrix.indices[start:end], matrix.data[start:end]
        if np.any(codes[positions] // 1000 != state):
            raise ValueError(
                f"State row {parent_name} contains households outside its state."
            )
        for index, members in zip(indices, child_members, strict=True):
            selected = np.isin(codes[positions], members)
            rows.append(np.full(int(selected.sum()), index, np.int32))
            cols.append(positions[selected].astype(np.int32))
            data.append(values[selected])
    return sparse.csr_array(
        (
            _concatenate(data, matrix.dtype),
            (_concatenate(rows, np.int32), _concatenate(cols, np.int32)),
        ),
        shape=(len(county_specs), len(households)),
    )


# ---------------------------------------------------------------------------
# ACS county families (microcosm#922)
# ---------------------------------------------------------------------------

ACS_SOURCE_NAME = "census_acs"
ACS_TABLES = "ACS 2020-2024 5-year"
ACS_HOUSEHOLDS_FAMILY = "acs_households"
ACS_HOUSEHOLDS_MEASURE = "B11001_001E"
#: Spine tags of the staging household table (``household_spine``).
ACS_SPINE = "acs_2024_1yr"
DONOR_SPINE = "asec_puf"
#: ACS PUMS household record type: 1 housing unit, 2 institutional and 3
#: noninstitutional group quarters. ACS household tables (B11001, B19001,
#: B19025, B19051, B19061) exclude group quarters, so the household measures
#: do too, as the county pilot did
#: (experiments/us-county-pilot-20261024/county_targets.py,
#: ``household_measures`` and ``GROUP_QUARTERS_TYPES``). Donor (ASEC by PUF)
#: households are all in the household universe there too.
HOUSEHOLD_KIND_COLUMN = "TYPEHUGQ"
ACS_HOUSING_UNIT_KIND = 1
GROUP_QUARTERS_KINDS = (2, 3)
#: Earnings: wages plus all self-employment, ACS B19061's concept. These are
#: the pilot's EARNINGS_COLUMNS plus DONOR_ONLY_EARNINGS_COLUMNS, summed over
#: a household's people. The pilot zeroed the last two on ACS records because
#: staging then stacked donor draws on SEMP (microcosm#1380). Materialize now
#: refuses such staging (``acs_self_employment_identity_gate``, run by
#: ``_require_acs_self_employment_identity`` before this point), and that gate
#: lets an ACS row's SEMP sit wholly on the SSTB leaf after QBI reconciliation
#: with farm income zero, so summing all four on every record counts SEMP
#: exactly once; zeroing the SSTB leaf would drop it.
EARNINGS_COLUMNS = (
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "sstb_self_employment_income_before_lsr",
    "farm_operations_income",
)
#: Partnership self-employment earnings, imputed on donor records from PUF
#: partnership SE income; ACS SEMP counts them, and the identity gate keeps
#: them zero on ACS rows. They count toward earnings on donor records when
#: staging stores them. The pilot left them out because it built income as
#: earnings plus ``partnership_income``, which already holds most of them;
#: here income is built from :data:`EARNINGS_COLUMNS` instead, so neither
#: measure counts them twice.
DONOR_PARTNERSHIP_EARNINGS_COLUMNS = ("partnership_self_employment_net_earnings",)
#: The pilot's other person-level income inputs, carried when staging stores
#: them, so the household-income concept of ACS B19025/B19001 can be
#: approximated. Absent ones are listed in the receipt.
OTHER_INCOME_COLUMNS = (
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "rental_income",
    "estate_income",
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
    "ssi_reported",
    "tanf_reported",
    "public_assistance_income",
    "taxable_private_pension_income",
    "tax_exempt_private_pension_income",
    "taxable_ira_distributions",
    "unemployment_compensation",
    "veterans_benefits",
    "workers_compensation",
    "alimony_income",
    "child_support_received",
    "miscellaneous_income",
)
#: Partnership and S-corporation income from donor tax detail. ACS SEMP
#: already counts partnership income, so on ACS records these donor leaves
#: are zeroed in the income approximation (the pilot's rule).
DONOR_ONLY_OTHER_INCOME_COLUMNS = ("partnership_income", "s_corp_income")
#: ACS B19001 household income brackets: estimate id, its income_range
#: dimension value in the county facts, and [lower, upper) dollar bounds.
#: Income below zero falls in the first bracket, as in ACS.
B19001_BRACKETS = (
    ("B19001_002E", "under_10000", -math.inf, 10_000.0),
    ("B19001_003E", "10000_to_14999", 10_000.0, 15_000.0),
    ("B19001_004E", "15000_to_19999", 15_000.0, 20_000.0),
    ("B19001_005E", "20000_to_24999", 20_000.0, 25_000.0),
    ("B19001_006E", "25000_to_29999", 25_000.0, 30_000.0),
    ("B19001_007E", "30000_to_34999", 30_000.0, 35_000.0),
    ("B19001_008E", "35000_to_39999", 35_000.0, 40_000.0),
    ("B19001_009E", "40000_to_44999", 40_000.0, 45_000.0),
    ("B19001_010E", "45000_to_49999", 45_000.0, 50_000.0),
    ("B19001_011E", "50000_to_59999", 50_000.0, 60_000.0),
    ("B19001_012E", "60000_to_74999", 60_000.0, 75_000.0),
    ("B19001_013E", "75000_to_99999", 75_000.0, 100_000.0),
    ("B19001_014E", "100000_to_124999", 100_000.0, 125_000.0),
    ("B19001_015E", "125000_to_149999", 125_000.0, 150_000.0),
    ("B19001_016E", "150000_to_199999", 150_000.0, 200_000.0),
    ("B19001_017E", "200000_plus", 200_000.0, math.inf),
)


@dataclasses.dataclass(frozen=True)
class ValidationFamily:
    """A held-out ACS county family: scored for every county, never trained."""

    name: str
    measure_ids: tuple
    household_measure: str
    approximate: bool
    unit: str
    description: str


#: Validation families: within-state shares of each ACS county table, scored
#: after calibration against the final weights' state totals. They never
#: enter the loss, the target registry or the roles.
VALIDATION_FAMILIES = (
    ValidationFamily(
        "acs_earnings",
        ("B19061_001E",),
        "earnings",
        False,
        "usd",
        "aggregate household earnings: wages plus all self-employment",
    ),
    ValidationFamily(
        "acs_earning_households",
        ("B19051_002E",),
        "earning_households",
        False,
        "count",
        "households with earnings (household earnings not zero)",
    ),
    ValidationFamily(
        "acs_household_income",
        ("B19025_001E",),
        "income",
        True,
        "usd",
        "aggregate household income, approximated by earnings plus the stored "
        "income inputs (approximate)",
    ),
    ValidationFamily(
        "acs_household_income_distribution",
        tuple(bracket[0] for bracket in B19001_BRACKETS),
        "income_bracket",
        True,
        "count",
        "households by household income bracket, on the approximate income "
        "(approximate); scored per bracket and as the bracket L1 error over "
        "the county's households",
    ),
)
#: B19013 is a median: a diagnostic only, never a linear target or a share.
MEDIAN_INCOME_MEASURE = "B19013_001E"


def _moe_id(measure_id):
    return measure_id[:-1] + "M"


def acs_county_values(facts, measure_ids, *, required=None):
    """``{measure: {county: (estimate, moe or None, name)}}`` from ACS rows.

    Reads every record set (B01003 splits its counties across two). Trains
    and scores on estimates only: a row is an estimate exactly when its
    ``statistic`` is ``estimate`` and its id ends in ``E``; its margin of
    error (id ending ``M``) is carried when present and never required.
    Duplicate, negative or nonfinite estimates are refused. With
    ``required`` (county codes), every measure must cover exactly those
    counties.
    """
    wanted = set(measure_ids)
    moe_ids = {_moe_id(measure): measure for measure in wanted}
    estimates = {measure: {} for measure in wanted}
    margins = {measure: {} for measure in wanted}
    for fact in facts:
        observed = fact.get("observed_measure", {})
        if observed.get("source_name") != ACS_SOURCE_NAME:
            continue
        measure = observed.get("source_measure_id")
        if measure not in wanted and measure not in moe_ids:
            continue
        statistic = (fact.get("dimensions") or {}).get("statistic")
        is_estimate = measure in wanted
        if statistic != ("estimate" if is_estimate else "margin_of_error"):
            raise ValueError(
                f"ACS county fact {measure} carries statistic {statistic!r}; "
                "an id ending E is an estimate and one ending M its margin of error."
            )
        code = county_code_of(fact, source="ACS")
        value = float(fact["value"])
        target = estimates[measure] if is_estimate else margins[moe_ids[measure]]
        if code in target:
            raise ValueError(f"Duplicate ACS county fact {measure} for {code:05d}.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"ACS county fact {measure} for {code:05d} must be finite and "
                f"nonnegative, got {value!r}."
            )
        target[code] = (value, str((fact.get("geography") or {}).get("name", "")))
    result = {}
    for measure in sorted(wanted):
        values = estimates[measure]
        if required is not None:
            expected = {int(code) for code in required}
            missing = sorted(expected - values.keys())
            extra = sorted(values.keys() - expected)
            if missing or extra:
                raise ValueError(
                    f"Incomplete ACS county inventory for {measure}: "
                    f"missing={[f'{code:05d}' for code in missing][:20]} "
                    f"({len(missing)}), unexpected="
                    f"{[f'{code:05d}' for code in extra][:20]} ({len(extra)}). "
                    "County facts, ladder and county column must use matching "
                    "boundaries."
                )
        result[measure] = {
            code: (
                value,
                margins[measure][code][0] if code in margins[measure] else None,
                name,
            )
            for code, (value, name) in sorted(values.items())
        }
    return result


def _state_label(state):
    from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_NAME

    label = US_STATE_FIPS_TO_NAME.get(f"{int(state):02d}")
    if label is None:
        raise ValueError(f"State {int(state):02d} has no Census state name.")
    return label


def _acs_hierarchy(name, *, state=None, county=None):
    from microcosm.calibrate import (
        CalibrationHierarchy,
        HierarchyCategory,
        HierarchyGeography,
        HierarchyNode,
    )
    from microcosm.calibrate.provider_labels import calibration_provider_label

    provider = HierarchyNode(
        ACS_SOURCE_NAME, calibration_provider_label("us", ACS_SOURCE_NAME)
    )
    if county is not None:
        # The same label every county family gives this county (_hierarchy).
        geography = HierarchyGeography(
            f"{COUNTY_PREFIX}{county}", f"County {county}", "county"
        )
        label = f"County {county} households (ACS)"
    else:
        state_label = _state_label(state)
        geography = HierarchyGeography(
            f"0400000US{int(state):02d}", state_label, "state"
        )
        label = f"{state_label} households (ACS)"
    return CalibrationHierarchy(
        provider=provider,
        category=HierarchyCategory("census_acs.households", "Households", provider.id),
        geography=geography,
        dimensions=(),
        target=HierarchyNode(name, label),
    )


def acs_households_parent_name(state):
    return f"acs_households_state_{int(state):02d}"


def acs_household_surface(facts, county_populations, *, period=2024):
    """State households parents and their county children (trained, levels).

    Each state parent is the sum of its counties' ACS B11001_001E values, a
    new state target that moves state household counts to ACS 5-year levels;
    its county children split it by county share, so they sum to it. The
    measure counts households outside ACS group quarters
    (:func:`household_universe`). ACS margins of error ride in metadata.
    """
    from microcosm.calibrate import TargetSpec

    values = acs_county_values(
        facts, [ACS_HOUSEHOLDS_MEASURE], required=county_populations
    )[ACS_HOUSEHOLDS_MEASURE]
    source = (
        f"US Census Bureau {ACS_TABLES} B11001_001E (households), county; "
        "state = sum of its counties"
    )
    specs, receipts = [], []
    for state in sorted({int(code) // 1000 for code in county_populations}):
        codes = sorted(code for code in values if code // 1000 == state)
        public = [values[code][0] for code in codes]
        total = math.fsum(public)
        if not codes or total <= 0:
            raise ValueError(
                f"ACS county households for state {state:02d} have no positive "
                "share denominator."
            )
        parent_name = acs_households_parent_name(state)
        state_population = math.fsum(county_populations[code] for code in codes)
        specs.append(
            TargetSpec(
                name=parent_name,
                entity="household",
                measure=parent_name,
                value=total,
                period=period,
                source=source,
                family=ACS_HOUSEHOLDS_FAMILY,
                metadata={
                    "geography_level": "state",
                    "state_fips": f"{state:02d}",
                    "source_measure_id": ACS_HOUSEHOLDS_MEASURE,
                    "acs_state_parent_basis": "sum of county estimates",
                    "household_universe": "excludes ACS group quarters (TYPEHUGQ 2, 3)",
                    "new_state_target": "true",
                },
                hierarchy=_acs_hierarchy(parent_name, state=state),
            )
        )
        shares = [value / total for value in public]
        largest = max(range(len(codes)), key=lambda index: shares[index])
        shares[largest] += 1.0 - math.fsum(shares)
        targets = [total * share for share in shares]
        targets[largest] += total - math.fsum(targets)
        for code, share, target in zip(codes, shares, targets, strict=True):
            county = f"{code:05d}"
            name = f"county_{county}__{parent_name}"
            estimate, moe, label = values[code]
            metadata = {
                "geography_level": "county",
                "county_fips": county,
                "state_fips": f"{state:02d}",
                "source_measure_id": ACS_HOUSEHOLDS_MEASURE,
                "state_county_parent_target_name": parent_name,
                "state_county_share": str(share),
                "state_county_source_value": str(estimate),
                "county_population": str(county_populations[code]),
                "state_population": str(state_population),
                "household_universe": "excludes ACS group quarters (TYPEHUGQ 2, 3)",
            }
            if label:
                metadata["county_label"] = label
            if moe is not None:
                metadata["acs_margin_of_error"] = str(moe)
            specs.append(
                TargetSpec(
                    name=name,
                    entity="household",
                    measure=name,
                    value=target,
                    period=period,
                    source=f"US Census Bureau {ACS_TABLES} B11001_001E "
                    "(households), county share of the state sum",
                    family=ACS_HOUSEHOLDS_FAMILY,
                    metadata=metadata,
                    hierarchy=_acs_hierarchy(name, county=county),
                )
            )
        receipts.append(
            {
                "state_fips": f"{state:02d}",
                "state_target": parent_name,
                "state_households": total,
                "counties": len(codes),
                "share_sum": math.fsum(shares),
                "target_sum": math.fsum(targets),
                "counties_with_moe": sum(values[code][1] is not None for code in codes),
            }
        )
    return CountySurface(
        tuple(specs),
        {
            "family": ACS_HOUSEHOLDS_FAMILY,
            "source_measure_id": ACS_HOUSEHOLDS_MEASURE,
            "source": f"{ACS_TABLES} county B11001",
            "new_state_targets": len(receipts),
            "county_targets": len(specs) - len(receipts),
            "note": (
                "New state targets: each state's households target is the sum "
                "of its counties' ACS 5-year B11001_001E estimates, which moves "
                "state household counts to ACS 5-year levels (the county pilot "
                "found the release about 6% under ACS households). County rows "
                "split those state targets; group-quarters records count as "
                "no household."
            ),
            "blocks": receipts,
            "new_engine_columns": 0,
        },
    )


def _spines(households):
    if "household_spine" not in households.columns:
        raise ValueError(
            "ACS county household measures need household_spine to find ACS rows."
        )
    spine = households["household_spine"].astype(str).to_numpy()
    unknown = sorted(set(spine) - {ACS_SPINE, DONOR_SPINE})
    if unknown:
        raise ValueError(f"Unexpected household spines {unknown[:5]}.")
    return spine


def household_universe(households):
    """1.0 for households in ACS's household universe, 0.0 for group quarters.

    An ACS household must carry ``TYPEHUGQ`` 1, 2 or 3 (missing or other
    codes are refused); a donor household carries none and is a household.
    """
    spine = _spines(households)
    acs = spine == ACS_SPINE
    if HOUSEHOLD_KIND_COLUMN in households.columns:
        kind = pd.to_numeric(
            households[HOUSEHOLD_KIND_COLUMN], errors="coerce"
        ).to_numpy(float)
    elif acs.any():
        raise ValueError(
            f"ACS households lack {HOUSEHOLD_KIND_COLUMN}; the household measures "
            "cannot exclude group quarters."
        )
    else:
        kind = np.full(len(households), np.nan)
    allowed = (ACS_HOUSING_UNIT_KIND, *GROUP_QUARTERS_KINDS)
    bad = acs & ~np.isin(kind, allowed)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} ACS household(s) carry {HOUSEHOLD_KIND_COLUMN} "
            f"outside {list(allowed)} (missing counts as outside)."
        )
    off_arm = ~acs & ~np.isnan(kind)
    if off_arm.any():
        raise ValueError(
            f"{int(off_arm.sum())} donor household(s) carry an ACS "
            f"{HOUSEHOLD_KIND_COLUMN}; it does not describe them."
        )
    return (~(acs & np.isin(kind, GROUP_QUARTERS_KINDS))).astype(np.float64)


def acs_household_rows(households, specs, *, county_column=COUNTY_COLUMN):
    """Rows of the ACS households family: state or county x household universe."""
    codes = county_codes(households, county_column=county_column)
    universe = household_universe(households).astype(np.float32)
    rows, cols, data = [], [], []
    for index, spec in enumerate(specs):
        state = int(spec.metadata["state_fips"])
        if spec.metadata["geography_level"] == "state":
            selected = codes // 1000 == state
        else:
            selected = codes == int(spec.metadata["county_fips"])
        positions = np.flatnonzero(selected & (universe > 0))
        rows.append(np.full(len(positions), index, np.int32))
        cols.append(positions.astype(np.int32))
        data.append(universe[positions])
    return sparse.csr_array(
        (
            _concatenate(data, np.float32),
            (_concatenate(rows, np.int32), _concatenate(cols, np.int32)),
        ),
        shape=(len(specs), len(households)),
    )


def _bracket_bounds(label):
    if label.startswith("under_"):
        return -math.inf, float(label.removeprefix("under_"))
    if label.endswith("_plus"):
        return float(label.removesuffix("_plus")), math.inf
    lower, _separator, upper = label.partition("_to_")
    return float(lower), float(upper) + 1.0


def county_validation_design(facts, county_populations):
    """The public side of every validation family, refused if incomplete.

    Estimates must cover every ladder county, each state's county sum must
    be positive (a zero share denominator is refused), and the B19001
    brackets must be the income ranges the facts label. B19013 medians are a
    diagnostic, so a county without one (De Baca, 35011) is listed, not
    refused.
    """
    counties = sorted(int(code) for code in county_populations)
    measure_ids = [
        measure for family in VALIDATION_FAMILIES for measure in family.measure_ids
    ]
    values = acs_county_values(facts, measure_ids, required=counties)
    labels = {}
    for fact in facts:
        observed = fact.get("observed_measure", {})
        if observed.get("source_name") == ACS_SOURCE_NAME and str(
            observed.get("source_measure_id", "")
        ).startswith("B19001_0"):
            labels.setdefault(
                observed["source_measure_id"],
                (fact.get("dimensions") or {}).get("income_range"),
            )
    for measure, label, lower, upper in B19001_BRACKETS:
        if labels.get(measure) != label or _bracket_bounds(label) != (lower, upper):
            raise ValueError(
                f"ACS {measure} is labelled {labels.get(measure)!r}; the bracket "
                f"table expects {label!r} = [{lower}, {upper})."
            )
    states = sorted({code // 1000 for code in counties})
    for measure, public in values.items():
        for state in states:
            total = math.fsum(
                value
                for code, (value, _m, _n) in public.items()
                if code // 1000 == state
            )
            if total <= 0:
                raise ValueError(
                    f"ACS {measure} has no positive share denominator in state "
                    f"{state:02d}."
                )
    median = acs_county_values(facts, [MEDIAN_INCOME_MEASURE])[MEDIAN_INCOME_MEASURE]
    names = {}
    for measure in measure_ids:
        for code, (_value, _moe, name) in values[measure].items():
            if name:
                names.setdefault(code, name)
    return {
        "schema_version": 1,
        "source": f"US Census Bureau {ACS_TABLES} county tables",
        "counties": [f"{code:05d}" for code in counties],
        "county_populations": {
            f"{code:05d}": float(county_populations[code]) for code in counties
        },
        "county_names": {f"{code:05d}": names.get(code, "") for code in counties},
        "families": {
            family.name: {
                "approximate": family.approximate,
                "unit": family.unit,
                "description": family.description,
                "household_measure": family.household_measure,
                "source_measure_ids": list(family.measure_ids),
                "public": {
                    measure: {
                        f"{code:05d}": [value, moe]
                        for code, (value, moe, _name) in values[measure].items()
                    }
                    for measure in family.measure_ids
                },
            }
            for family in VALIDATION_FAMILIES
        },
        # [lower, upper) dollars; JSON has no infinity, so an open end is null.
        "brackets": [
            {
                "measure": measure,
                "income_range": label,
                "lower": None if math.isinf(lower) else lower,
                "upper": None if math.isinf(upper) else upper,
            }
            for measure, label, lower, upper in B19001_BRACKETS
        ],
        "median_income_diagnostic": {
            "source_measure_id": MEDIAN_INCOME_MEASURE,
            "role": "diagnostic only; never a linear target or a share",
            "public": {
                f"{code:05d}": [value, moe]
                for code, (value, moe, _name) in median.items()
                if code in set(counties)
            },
            "missing": [f"{code:05d}" for code in counties if code not in median],
        },
        "score": (
            "target(c) = M_s x public(c) / sum of public over the state's "
            "counties, M_s the state total under the final weights; pro-rata "
            "baseline M_s x ladder county population / state population"
        ),
    }


def household_validation_measures(frame):
    """Household-aligned measures the validation families are scored on.

    Returns ``(arrays, report)``: ``household_id``, ``in_household_universe``,
    ``earnings`` (:data:`EARNINGS_COLUMNS` summed over the household's
    people) and ``household_income_approx`` (earnings plus the carried
    :data:`OTHER_INCOME_COLUMNS`, plus :data:`DONOR_ONLY_OTHER_INCOME_COLUMNS`
    on donor records). Missing person values count as zero, as in the pilot.
    """
    households = frame.table("household")
    person = frame.table("person")
    index = pd.Index(households["household_id"].to_numpy())
    position = index.get_indexer(person["person_household_id"].to_numpy())
    if (position < 0).any():
        raise ValueError("Persons reference households missing from the table.")
    missing = [column for column in EARNINGS_COLUMNS if column not in person.columns]
    if missing:
        raise ValueError(
            f"County validation needs person earnings inputs {missing}; refusing "
            "to treat them as zero."
        )
    n = len(households)
    spine = _spines(households)
    donor_person = spine[position] == DONOR_SPINE
    universe = household_universe(households)
    weights = np.asarray(frame.weights_for("household").values, dtype=np.float64)

    def summed(column, *, donor_only=False):
        values = pd.to_numeric(person[column], errors="coerce").to_numpy(
            np.float64, na_value=np.nan
        )
        if np.isinf(values).any():
            raise ValueError(f"Person {column} carries an infinite value.")
        values = np.nan_to_num(values, nan=0.0)
        if donor_only:
            values = np.where(donor_person, values, 0.0)
        return np.bincount(position, weights=values, minlength=n)

    earnings = np.zeros(n)
    components = {}
    for column in EARNINGS_COLUMNS:
        part = summed(column)
        earnings += part
        components[column] = float((weights * part).sum())
    # Income starts from the base leaves: donor partnership SE earnings are
    # mostly inside partnership_income, which the income approximation adds.
    income = earnings.copy()
    donor_earnings = [
        column
        for column in DONOR_PARTNERSHIP_EARNINGS_COLUMNS
        if column in person.columns
    ]
    for column in donor_earnings:
        part = summed(column, donor_only=True)
        earnings += part
        components[column] = float((weights * part).sum())
    carried = [column for column in OTHER_INCOME_COLUMNS if column in person.columns]
    donor_carried = [
        column for column in DONOR_ONLY_OTHER_INCOME_COLUMNS if column in person.columns
    ]
    for column in carried:
        part = summed(column)
        income += part
        components[column] = float((weights * part).sum())
    for column in donor_carried:
        part = summed(column, donor_only=True)
        income += part
        components[column] = float((weights * part).sum())
    group_quarters = universe == 0
    report = {
        "earnings_columns": list(EARNINGS_COLUMNS),
        "earnings_rule": "all four leaves on every record; the ACS "
        "self-employment identity gate guarantees SEMP once (microcosm#1380); "
        "partnership SE earnings on donor records",
        "donor_only_earnings_columns_carried": donor_earnings,
        "income_rule": "the four earnings leaves plus the carried income "
        "columns; donor partnership SE earnings enter through "
        "partnership_income, not twice",
        "other_income_columns_carried": carried,
        "donor_only_income_columns_carried": donor_carried,
        "income_columns_absent": [
            column
            for column in (*OTHER_INCOME_COLUMNS, *DONOR_ONLY_OTHER_INCOME_COLUMNS)
            if column not in person.columns
        ],
        "design_weighted_component_totals": components,
        "group_quarters_households": int(group_quarters.sum()),
        "group_quarters_design_weight": float(weights[group_quarters].sum()),
        "households": int(n),
    }
    arrays = {
        "household_id": households["household_id"].to_numpy(),
        "in_household_universe": universe,
        "earnings": earnings,
        "household_income_approx": income,
    }
    return arrays, report


def _family_vectors(family, measures):
    universe = np.asarray(measures["in_household_universe"], dtype=np.float64)
    earnings = np.asarray(measures["earnings"], dtype=np.float64)
    income = np.asarray(measures["household_income_approx"], dtype=np.float64)
    if family.household_measure == "earnings":
        return {family.measure_ids[0]: earnings * universe}
    if family.household_measure == "earning_households":
        return {family.measure_ids[0]: (earnings != 0) * universe}
    if family.household_measure == "income":
        return {family.measure_ids[0]: income * universe}
    return {
        measure: ((income >= lower) & (income < upper)) * universe
        for measure, _label, lower, upper in B19001_BRACKETS
    }


def _abs_rel(estimate, target):
    return np.abs(estimate - target) / np.maximum(np.abs(target), 1.0)


def _weighted_medians(group, values, weights, n_groups):
    order = np.lexsort((values, group))
    group, values, weights = group[order], values[order], weights[order]
    result = np.full(n_groups, np.nan)
    starts = np.searchsorted(group, np.arange(n_groups), side="left")
    ends = np.searchsorted(group, np.arange(n_groups), side="right")
    for index in range(n_groups):
        start, end = starts[index], ends[index]
        cumulative = np.cumsum(weights[start:end])
        if end == start or cumulative[-1] <= 0:
            continue
        result[index] = values[
            start + int(np.searchsorted(cumulative, cumulative[-1] / 2))
        ]
    return result


def score_county_validation(
    design,
    measures,
    households,
    *,
    final_weights,
    design_weights,
    cap,
    county_column=COUNTY_COLUMN,
):
    """Score every validation family for every county, report only.

    For each ACS measure, ``target(c) = M_s * public(c) / sum_s public`` with
    ``M_s`` the state total of the household measure under the final
    weights, so a uniform concept gap between the measure and the ACS table
    cancels; the pro-rata baseline is ``M_s * pop(c) / pop(s)`` from the
    ladder. Nothing here can reach the loss: it reads final weights only.
    """
    counties = [int(code) for code in design["counties"]]
    position = {code: index for index, code in enumerate(counties)}
    codes = county_codes(households, county_column=county_column)
    ids = households["household_id"].to_numpy()
    if not np.array_equal(np.asarray(measures["household_id"]), ids):
        raise ValueError(
            "County validation measures are not aligned with the checkpoint households."
        )
    unknown = sorted(set(codes) - set(position))
    if unknown:
        raise ValueError(f"Households name counties outside the design: {unknown[:5]}.")
    county_index = np.asarray([position[int(code)] for code in codes], dtype=np.int64)
    n = len(counties)
    states = np.asarray([code // 1000 for code in counties])
    state_codes = np.unique(states)
    state_of = np.searchsorted(state_codes, states)
    population = np.asarray(
        [design["county_populations"][f"{code:05d}"] for code in counties], float
    )
    state_population = np.bincount(state_of, weights=population)[state_of]
    weight_sets = {
        "calibrated": np.asarray(final_weights, dtype=np.float64),
        "design": np.asarray(design_weights, dtype=np.float64),
    }
    per_county = {f"{code:05d}": {} for code in counties}
    families = {}
    for family in VALIDATION_FAMILIES:
        published = design["families"][family.name]["public"]
        vectors = _family_vectors(family, measures)
        family_rows = {}
        for measure, vector in vectors.items():
            public = np.asarray(
                [published[measure][f"{code:05d}"][0] for code in counties], float
            )
            share = public / np.bincount(state_of, weights=public)[state_of]
            row = {"public": public}
            for label, weights in weight_sets.items():
                estimate = np.bincount(
                    county_index, weights=weights * vector, minlength=n
                )
                state_total = np.bincount(state_of, weights=estimate)[state_of]
                row[label] = estimate
                row[f"{label}_target"] = state_total * share
                if label == "calibrated":
                    row["pro_rata"] = state_total * population / state_population
            family_rows[measure] = row
        summary = {
            "approximate": family.approximate,
            "unit": family.unit,
            "description": family.description,
            "source_measure_ids": list(family.measure_ids),
        }
        if family.household_measure == "income_bracket":
            l1 = {}
            for label, target_key in (
                ("calibrated", "calibrated_target"),
                ("design", "design_target"),
                ("pro_rata", "calibrated_target"),
            ):
                difference = sum(
                    np.abs(row[label] - row[target_key]) for row in family_rows.values()
                )
                total = sum(row[target_key] for row in family_rows.values())
                l1[label] = difference / np.maximum(total, 1.0)
            for label, key in (
                ("calibrated", "calibrated"),
                ("design", "design_weights"),
                ("pro_rata", "pro_rata_baseline"),
            ):
                values = l1[label]
                summary[key] = {
                    "mean_bracket_l1_error": float(values.mean()),
                    "median_bracket_l1_error": float(np.median(values)),
                    "p90_bracket_l1_error": float(np.quantile(values, 0.9)),
                    "max_bracket_l1_error": float(values.max()),
                }
            summary["calibrated_beats_pro_rata_share"] = float(
                (l1["calibrated"] < l1["pro_rata"]).mean()
            )
            summary["error"] = (
                "sum over brackets |estimate - target| / max(sum of targets, 1)"
            )
            for index, code in enumerate(counties):
                key = f"{code:05d}"
                per_county[key][family.name] = {
                    "approximate": True,
                    "abs_rel_error": float(l1["calibrated"][index]),
                    "pro_rata_abs_rel_error": float(l1["pro_rata"][index]),
                    "design_abs_rel_error": float(l1["design"][index]),
                    "brackets": {
                        measure: {
                            "public_value": float(row["public"][index]),
                            "public_moe": published[measure][key][1],
                            "target": float(row["calibrated_target"][index]),
                            "estimate": float(row["calibrated"][index]),
                        }
                        for measure, row in family_rows.items()
                    },
                }
        else:
            measure = family.measure_ids[0]
            row = family_rows[measure]
            errors = _abs_rel(row["calibrated"], row["calibrated_target"])
            design_errors = _abs_rel(row["design"], row["design_target"])
            baseline = _abs_rel(row["pro_rata"], row["calibrated_target"])
            summary["calibrated"] = cd_surface._error_summary(
                row["calibrated"], row["calibrated_target"], cap=cap
            )
            summary["design_weights"] = cd_surface._error_summary(
                row["design"], row["design_target"], cap=cap
            )
            summary["pro_rata_baseline"] = cd_surface._error_summary(
                row["pro_rata"], row["calibrated_target"], cap=cap
            )
            summary["calibrated_beats_pro_rata_share"] = float(
                (errors < baseline).mean()
            )
            for index, code in enumerate(counties):
                key = f"{code:05d}"
                per_county[key][family.name] = {
                    "approximate": family.approximate,
                    "public_value": float(row["public"][index]),
                    "public_moe": published[measure][key][1],
                    "target": float(row["calibrated_target"][index]),
                    "estimate": float(row["calibrated"][index]),
                    "abs_rel_error": float(errors[index]),
                    "pro_rata": float(row["pro_rata"][index]),
                    "pro_rata_abs_rel_error": float(baseline[index]),
                    "design_abs_rel_error": float(design_errors[index]),
                }
        families[family.name] = summary
    universe = np.asarray(measures["in_household_universe"], dtype=np.float64) > 0
    medians = _weighted_medians(
        county_index[universe],
        np.asarray(measures["household_income_approx"], dtype=np.float64)[universe],
        weight_sets["calibrated"][universe],
        n,
    )
    median_public = design["median_income_diagnostic"]["public"]
    for index, code in enumerate(counties):
        key = f"{code:05d}"
        public = median_public.get(key)
        estimate = None if math.isnan(medians[index]) else float(medians[index])
        per_county[key]["median_household_income"] = {
            "diagnostic_only": True,
            "approximate": True,
            "public_value": None if public is None else public[0],
            "public_moe": None if public is None else public[1],
            "estimate": estimate,
            "ratio": (
                estimate / public[0]
                if public is not None and estimate is not None and public[0] > 0
                else None
            ),
            "status": "scored" if public is not None else "no_public_value",
        }
    return {
        "report_only": True,
        "never_in_loss": True,
        "families": families,
        "median_income_diagnostic": {
            "source_measure_id": MEDIAN_INCOME_MEASURE,
            "role": "diagnostic only; never a linear target or a share",
            "missing_public": design["median_income_diagnostic"]["missing"],
        },
        "score": design["score"],
        "counties": per_county,
    }


#: The validation errors a county's grade reads: one per family (the income
#: distribution as its bracket L1 error). Medians never enter a grade.
GRADED_VALIDATION_FAMILIES = tuple(family.name for family in VALIDATION_FAMILIES)


def target_record(spec):
    record = cd_surface.target_record(spec)
    record["geography_level"] = record["geography_level"] or spec.metadata.get(
        "geography_level"
    )
    for key in (
        "county_fips",
        "state_county_parent_target_name",
        "state_county_share",
        "county_population",
        "state_population",
        "state_county_cell_members",
    ):
        if key in spec.metadata:
            value = spec.metadata[key]
            record[key] = (
                float(value)
                if key
                in {"state_county_share", "county_population", "state_population"}
                else value
            )
    return record


def assign_target_roles(
    records, *, fraction=DEFAULT_COUNTY_HOLDOUT_FRACTION, salt=COUNTY_HOLDOUT_SALT
):
    """Hold out whole state/concept-family blocks to prevent sibling leakage."""
    from microcosm.build.holdout import hash_holdout_unit

    if not isinstance(salt, str) or not salt:
        raise ValueError("County holdout salt must be a non-empty string.")
    if not 0 <= fraction <= cd_surface.MAX_CD_HOLDOUT_FRACTION:
        raise ValueError(
            f"County holdout fraction must be in [0, {cd_surface.MAX_CD_HOLDOUT_FRACTION}]."
        )
    units = {}
    eligible = []
    for record in records:
        if record.get("geography_level") != "county":
            continue
        unit = None
        if record.get("state_county_parent_target_name"):
            family = (
                cd_surface.soi_concept_family(str(record.get("source_measure_id")))
                if record.get("family") == "irs_soi"
                else str(record.get("family"))
            )
            unit = f"{record['state_fips']}|{family}"
        record["holdout_unit"] = unit
        record["role"] = cd_surface.ROLE_TRAIN
        if unit is not None:
            eligible.append(record)
            units.setdefault(
                unit, hash_holdout_unit(unit, fraction=fraction, salt=salt)
            )
            if units[unit]:
                record["role"] = cd_surface.ROLE_HOLDOUT
    return {
        "unit": "state x county concept family",
        "salt": salt,
        "fraction": float(fraction),
        "eligible_units": len(units),
        "eligible_targets": len(eligible),
        "held_units": sum(units.values()),
        "held_unit_keys": sorted(key for key, value in units.items() if value),
        "held_targets": sum(
            record["role"] == cd_surface.ROLE_HOLDOUT for record in eligible
        ),
    }


def pro_rata_baseline(records, rows):
    by_name = {record["name"]: record for record in records}
    values = []
    for index in rows:
        record = records[index]
        parent = by_name.get(record.get("state_county_parent_target_name"))
        county, state = (
            float(record.get("county_population", 0)),
            float(record.get("state_population", 0)),
        )
        if parent is None or not (
            math.isfinite(county) and math.isfinite(state) and county > 0 and state > 0
        ):
            raise ValueError(
                f"{record['name']} has no state parent or positive county/state population for pro-rata baseline."
            )
        values.append(float(parent["value"]) * county / state)
    return np.asarray(values, dtype=float)


def score_county_holdout(
    records, matrix, *, design_weights, final_weights, cap, receipt=None
):
    rows = [
        index
        for index, record in enumerate(records)
        if record.get("geography_level") == "county"
        and record.get("role") == cd_surface.ROLE_HOLDOUT
    ]
    base = {"report_only": True, **dict(receipt or {}), "n_targets": len(rows)}
    if not rows:
        return base
    held = sparse.csr_array(matrix[rows]).astype(float)
    targets = np.asarray([records[index]["value"] for index in rows], float)
    final, design = held @ np.asarray(final_weights), held @ np.asarray(design_weights)
    baseline = pro_rata_baseline(records, rows)
    scale = np.maximum(np.abs(targets), 1.0)
    return {
        **base,
        "calibrated": cd_surface._error_summary(final, targets, cap=cap),
        "design_weights": cd_surface._error_summary(design, targets, cap=cap),
        "pro_rata_baseline": cd_surface._error_summary(baseline, targets, cap=cap),
        "calibrated_beats_pro_rata_share": float(
            (
                np.abs(final - targets) / scale < np.abs(baseline - targets) / scale
            ).mean()
        ),
        "baseline": "state parent target x county population / state population",
        "targets": [
            {
                "name": records[index]["name"],
                "county_fips": records[index]["county_fips"],
                "target": float(targets[position]),
                "calibrated": float(final[position]),
                "design": float(design[position]),
                "pro_rata": float(baseline[position]),
            }
            for position, index in enumerate(rows)
        ],
    }


def reliability_grade(ess, error):
    """Provisional A/B/C: source ESS >=400/100/30, error <=10/20/50%; else D.

    U means no holdout, validation or source-identity evidence. ESS is a
    precision diagnostic, not a standard error or a guarantee of unbiased
    county estimates.
    """
    if ess is None or error is None or not math.isfinite(error):
        return "U"
    for grade, minimum, maximum in (("A", 400, 0.1), ("B", 100, 0.2), ("C", 30, 0.5)):
        if ess >= minimum and error <= maximum:
            return grade
    return "D"


def record_counties(record):
    """The counties a county record's error counts toward (combined cells: all)."""
    members = record.get("state_county_cell_members")
    if members:
        return [int(member) for member in str(members).split(",")]
    return [int(record["county_fips"])]


def _county_groups(codes):
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    unique, starts = np.unique(sorted_codes, return_index=True)
    ends = np.append(starts[1:], len(sorted_codes))
    return {
        int(code): order[start:end]
        for code, start, end in zip(unique, starts, ends, strict=True)
    }


def reliability_report(
    households,
    records,
    matrix,
    *,
    final_weights,
    county_populations=None,
    county_column=COUNTY_COLUMN,
    validation=None,
    minimum_support_counties=(),
    county_names=None,
):
    """One reliability cell per county: support, ESS, errors and a grade.

    ``validation`` (from :func:`score_county_validation`) supplies each
    county's held-out ACS family errors, which count as holdout evidence, so
    a supported county with source identity always gets A/B/C/D. A county is
    U only without any held-out evidence (no household support) or without
    source identity. ``minimum_support_counties`` mark counties whose only
    households came from the minimum-support step.
    """
    codes = county_codes(households, county_column=county_column)
    weights = np.asarray(final_weights, float)
    if (
        weights.shape != (len(households),)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
    ):
        raise ValueError(
            "County reliability weights must be aligned, finite and nonnegative."
        )
    estimates = sparse.csr_array(matrix).astype(float) @ weights
    by_county = defaultdict(list)
    for index, record in enumerate(records):
        if record.get("geography_level") == "county":
            for code in record_counties(record):
                by_county[code].append(index)
    validated = (validation or {}).get("counties", {})
    counties = sorted(
        set(int(code) for code in codes)
        | set(by_county)
        | {int(code) for code in (county_populations or {})}
        | {int(code) for code in validated}
    )
    groups = _county_groups(codes)
    minimum = {int(code) for code in minimum_support_counties}
    names = county_names or {}
    result = {}
    provenance = {"household_spine", "household_source_id"}.issubset(households.columns)
    for code in counties:
        rows = groups.get(code, np.empty(0, dtype=np.int64))
        source = (
            households.iloc[rows][["household_spine", "household_source_id"]]
            if provenance
            else None
        )
        if source is not None and not source.isna().any().any():
            source_weights = cd_surface.distinct_household_weights(
                weights[rows],
                source["household_spine"].to_numpy(),
                source["household_source_id"].to_numpy(),
            )
        elif not len(rows):
            source_weights = np.empty(0)
        else:
            source_weights = None
        key = f"{int(code):05d}"
        status = "no_support"
        if len(rows):
            status = "minimum_support" if code in minimum else "supported"
        cell = {
            "state_fips": f"{int(code) // 1000:02d}",
            "county_name": names.get(key) or None,
            "supporting_households": len(rows),
            "distinct_source_households": (
                len(source_weights) if source_weights is not None else None
            ),
            "kish_ess": (
                cd_surface.kish_ess(source_weights)
                if source_weights is not None
                else None
            ),
            "kish_ess_rows": cd_surface.kish_ess(weights[rows]),
            "status": status,
        }
        candidates = []
        for role, label in (
            (cd_surface.ROLE_TRAIN, "training"),
            (cd_surface.ROLE_HOLDOUT, "holdout"),
        ):
            indices = [
                index for index in by_county[code] if records[index].get("role") == role
            ]
            errors = [
                abs(estimates[index] - records[index]["value"])
                / max(abs(records[index]["value"]), 1)
                for index in indices
            ]
            cell[label] = {
                "n_targets": len(indices),
                "mean_abs_rel_error": float(np.mean(errors)) if errors else None,
                "max_abs_rel_error": float(max(errors)) if errors else None,
            }
            candidates += [
                (float(error), f"{label}:{records[index]['name']}")
                for error, index in zip(errors, indices, strict=True)
            ]
        families = validated.get(key) or {}
        # A county without household support has no held-out estimate to
        # judge: its validation estimates are zero by construction.
        graded = (
            {
                name: families[name]
                for name in GRADED_VALIDATION_FAMILIES
                if name in families
            }
            if len(rows)
            else {}
        )
        cell["validation"] = {
            "n_families": len(graded),
            "families": families,
        }
        candidates += [
            (float(entry["abs_rel_error"]), f"validation:{name}")
            for name, entry in graded.items()
        ]
        # A county without household support is U: a combined SOI cell's
        # rows (Kalawao's in Maui's) carry another county's error.
        evidence = len(rows) and (cell["holdout"]["n_targets"] > 0 or bool(graded))
        if evidence and candidates:
            error, driver = max(candidates, key=lambda item: (item[0], item[1]))
        else:
            error, driver = None, None
        cell["grade_error"] = error
        cell["grade_driver"] = driver
        cell["grade"] = reliability_grade(cell["kish_ess"], error)
        result[key] = cell
    return {
        "schema_version": 3,
        "county_column": county_column,
        "ess_basis": {
            "kish_ess": "final weights summed within county by (household_spine, household_source_id)",
            "kish_ess_rows": "final weights over physical household rows, including clones",
        },
        "counties": result,
        "validation_families": {
            name: {
                key: value
                for key, value in summary.items()
                if key in {"approximate", "unit", "description", "source_measure_ids"}
            }
            for name, summary in (validation or {}).get("families", {}).items()
        },
        "grade_rule": {
            "provisional": True,
            "A": "source ESS >=400 and max error <=0.10",
            "B": "source ESS >=100 and max error <=0.20",
            "C": "source ESS >=30 and max error <=0.50",
            "D": "otherwise with holdout or validation evidence",
            "U": "no county holdout or validation evidence (no household "
            "support) or source household identity unavailable",
            "error": "maximum absolute relative error over the county's training "
            "and holdout targets and its validation families (the income "
            "distribution as its bracket L1 error); denominator max(abs(target),1)",
            "validation_families": list(GRADED_VALIDATION_FAMILIES),
        },
    }


#: Columns of the flat per-county reliability CSV, in order.
def reliability_csv_columns():
    columns = [
        "county_fips",
        "state_fips",
        "county_name",
        "status",
        "grade",
        "grade_provisional",
        "grade_error",
        "grade_driver",
        "supporting_households",
        "distinct_source_households",
        "kish_ess",
        "kish_ess_rows",
        "training_targets",
        "training_max_abs_rel_error",
        "training_mean_abs_rel_error",
        "holdout_targets",
        "holdout_max_abs_rel_error",
    ]
    for name in GRADED_VALIDATION_FAMILIES:
        columns += [f"{name}_abs_rel_error", f"{name}_pro_rata_abs_rel_error"]
    columns += [
        "median_household_income_public",
        "median_household_income_estimate",
    ]
    return columns


def reliability_csv(report):
    """The flat CSV of a reliability report: one row per county, sorted by FIPS."""
    import csv
    import io

    provisional = bool(report.get("grade_rule", {}).get("provisional", True))
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream, fieldnames=reliability_csv_columns(), lineterminator="\n"
    )
    writer.writeheader()

    def cell_value(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return repr(value)
        return value

    for code, cell in sorted(report["counties"].items()):
        families = (cell.get("validation") or {}).get("families", {})
        median = families.get("median_household_income") or {}
        row = {
            "county_fips": code,
            "state_fips": cell.get("state_fips"),
            "county_name": cell.get("county_name"),
            "status": cell.get("status"),
            "grade": cell.get("grade"),
            "grade_provisional": "true" if provisional else "false",
            "grade_error": cell.get("grade_error"),
            "grade_driver": cell.get("grade_driver"),
            "supporting_households": cell.get("supporting_households"),
            "distinct_source_households": cell.get("distinct_source_households"),
            "kish_ess": cell.get("kish_ess"),
            "kish_ess_rows": cell.get("kish_ess_rows"),
            "training_targets": (cell.get("training") or {}).get("n_targets"),
            "training_max_abs_rel_error": (cell.get("training") or {}).get(
                "max_abs_rel_error"
            ),
            "training_mean_abs_rel_error": (cell.get("training") or {}).get(
                "mean_abs_rel_error"
            ),
            "holdout_targets": (cell.get("holdout") or {}).get("n_targets"),
            "holdout_max_abs_rel_error": (cell.get("holdout") or {}).get(
                "max_abs_rel_error"
            ),
            "median_household_income_public": median.get("public_value"),
            "median_household_income_estimate": median.get("estimate"),
        }
        for name in GRADED_VALIDATION_FAMILIES:
            entry = families.get(name) or {}
            row[f"{name}_abs_rel_error"] = entry.get("abs_rel_error")
            row[f"{name}_pro_rata_abs_rel_error"] = entry.get("pro_rata_abs_rel_error")
        writer.writerow({key: cell_value(value) for key, value in row.items()})
    return stream.getvalue()


# ---------------------------------------------------------------------------
# Minimum support for counties without households (microcosm#922)
# ---------------------------------------------------------------------------

DEFAULT_COUNTY_MIN_SUPPORT = 3
MAX_COUNTY_MIN_SUPPORT = 50
COUNTY_MIN_SUPPORT_KEY = "acs-local-county-support-v1"
#: The household county columns a move rewrites (each only if present, plus
#: the configured county column); nothing else changes.
COUNTY_FIPS_COLUMNS = ("county_fips", "county_fips_2023")
#: Location columns that fix a household's county. A household carrying one
#: is never moved: finalize's ladder and block gates check the stored county
#: against it, and a move rewrites county columns only.
COUNTY_DETERMINING_LOCATION_COLUMNS = ("tract_geoid", "block_geoid")


def county_puma_overlaps(ladder_path):
    """``{county: [(puma, population), ...]}``, largest overlap first."""
    from microcosm.build.us_runtime.puma_ladder import load_us_puma_ladder

    ladder = load_us_puma_ladder(ladder_path)
    return puma_overlaps(
        ladder.county_overlap_puma,
        ladder.county_overlap_county,
        ladder.county_overlap_population,
    )


def puma_overlaps(pumas, counties, populations):
    cells = defaultdict(lambda: defaultdict(float))
    for puma, county, population in zip(
        np.asarray(pumas), np.asarray(counties), np.asarray(populations), strict=True
    ):
        cells[int(county)][int(puma)] += float(population)
    return {
        county: sorted(
            (
                (puma, population)
                for puma, population in by_puma.items()
                if population > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )
        for county, by_puma in cells.items()
    }


def support_rank(key, county, household_id):
    """The keyed hash that orders a destination county's candidate donors."""
    return hashlib.sha256(
        f"{key}\x1f{int(county):05d}\x1f{household_id}".encode()
    ).hexdigest()


def _household_pumas(households):
    if "puma" not in households.columns:
        raise ValueError("Minimum county support needs the household puma column.")
    return _codes(households["puma"], column="puma")


def _blank(values):
    series = pd.Series(values)
    return (
        series.isna() | series.astype(str).str.strip().isin(["", "nan", "None"])
    ).to_numpy()


def _moves_digest(moves):
    return hashlib.sha256(
        json.dumps(moves, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def support_universe(households):
    """Households that count as county support: those in the household universe.

    An ACS group-quarters record supports no households-family target, so it
    neither makes a county supported nor counts toward a donor county's floor.
    Without the columns that identify group quarters every household counts.
    """
    if "household_spine" in households.columns and (
        HOUSEHOLD_KIND_COLUMN in households.columns
        or not (households["household_spine"].astype(str) == ACS_SPINE).any()
    ):
        return household_universe(households) > 0
    return np.ones(len(households), dtype=bool)


def unsupported_counties(households, county_populations, *, county_column):
    """Counties with positive ladder population and no supporting household."""
    codes = county_codes(households, county_column=county_column)
    present = {int(code) for code in np.unique(codes[support_universe(households)])}
    return sorted(
        int(code)
        for code, population in county_populations.items()
        if float(population) > 0 and int(code) not in present
    )


def plan_minimum_support(
    households,
    county_populations,
    overlaps,
    *,
    county_column=COUNTY_COLUMN,
    per_county=DEFAULT_COUNTY_MIN_SUPPORT,
    key=COUNTY_MIN_SUPPORT_KEY,
):
    """Choose ``per_county`` donor households for each unsupported county.

    A county is unsupported when its ladder population is positive and no
    household in the household universe names it (:func:`support_universe`). Its donors come from the PUMA that overlaps it most
    (then the next, by ladder population), ordered by
    :func:`support_rank` of ``(key, county, household_id)``. A donor must be
    in the household universe (not ACS group quarters), carry no tract or
    block geoid, have agreeing county columns, and leave its own county with
    at least ``per_county`` households. Exactly ``per_county`` households
    move or none do (the county is then still unsupported and named).
    Counties with support are never chosen; state and PUMA never change.
    Returns ``(moves, receipt)``.
    """
    if per_county < 0:
        raise ValueError("Minimum county support must be nonnegative.")
    codes = county_codes(households, county_column=county_column)
    ids = households["household_id"].to_numpy()
    if len(pd.unique(ids)) != len(ids):
        raise ValueError("Minimum county support needs unique household ids.")
    universe = support_universe(households)
    unsupported = unsupported_counties(
        households, county_populations, county_column=county_column
    )
    receipt = {
        "households_per_county": int(per_county),
        "key": key,
        "rank": "sha256(key + 0x1f + five-digit county + 0x1f + household_id)",
        "unsupported_before": [f"{code:05d}" for code in unsupported],
        "rewritten_columns": [],
        "eligibility": (
            "same PUMA as the county (largest ladder overlap first); not ACS "
            "group quarters; no tract_geoid or block_geoid; county columns "
            "agree; the donor county keeps at least households_per_county "
            "households in the household universe"
        ),
        "counties": {},
        "moves": [],
    }
    if per_county == 0 or not unsupported:
        receipt["moves_sha256"] = _moves_digest([])
        return [], receipt
    pumas = _household_pumas(households)
    if np.any(pumas // 100_000 != codes // 1000):
        raise ValueError("Household PUMA and county disagree on the state.")
    eligible = universe.copy()
    for column in COUNTY_DETERMINING_LOCATION_COLUMNS:
        if column in households.columns:
            eligible &= _blank(households[column])
    rewritten = [county_column] + [
        column
        for column in COUNTY_FIPS_COLUMNS
        if column in households.columns and column != county_column
    ]
    for column in rewritten[1:]:
        eligible &= _codes(households[column], column=column) == codes
    receipt["rewritten_columns"] = rewritten
    receipt["location_columns_present"] = [
        column
        for column in COUNTY_DETERMINING_LOCATION_COLUMNS
        if column in households.columns
    ]
    by_puma = _county_groups(pumas)
    # The donor floor counts supporting households only, so a move never
    # leaves a donor county with group-quarters records alone.
    counts = defaultdict(int)
    for code in codes[universe]:
        counts[int(code)] += 1
    moved = np.zeros(len(households), dtype=bool)
    moves = []
    for county in unsupported:
        order = overlaps.get(county, [])
        tried, taken = [], []
        for puma, _population in order:
            if puma // 100_000 != county // 1000:
                raise ValueError(
                    f"Ladder overlap puts county {county:05d} in PUMA {puma:07d} "
                    "of another state."
                )
            tried.append(f"{puma:07d}")
            positions = by_puma.get(puma, np.empty(0, dtype=np.int64))
            candidates = sorted(
                (
                    position
                    for position in positions.tolist()
                    if eligible[position] and not moved[position]
                ),
                key=lambda position: (
                    support_rank(key, county, ids[position]),
                    position,
                ),
            )
            for position in candidates:
                if len(taken) == per_county:
                    break
                donor = int(codes[position])
                pending = sum(
                    1 for _p, from_county, _r in taken if from_county == donor
                )
                if counts[donor] - pending - 1 < per_county:
                    continue
                taken.append((position, donor, puma))
            if len(taken) == per_county:
                break
        status = "moved" if len(taken) == per_county else "insufficient_eligible_donors"
        if status == "moved":
            for position, donor, puma in taken:
                counts[donor] -= 1
                counts[county] += 1
                moved[position] = True
                household_id = ids[position]
                moves.append(
                    {
                        "household_id": (
                            int(household_id)
                            if isinstance(household_id, (int, np.integer))
                            else str(household_id)
                        ),
                        "from_county": f"{donor:05d}",
                        "to_county": f"{county:05d}",
                        "puma": f"{puma:07d}",
                        "rank": support_rank(key, county, household_id),
                    }
                )
        receipt["counties"][f"{county:05d}"] = {
            "ladder_population": float(county_populations[county]),
            "pumas_by_overlap": [f"{puma:07d}" for puma, _population in order],
            "pumas_tried": tried,
            "moved": per_county if status == "moved" else 0,
            "status": status,
        }
    receipt["moves"] = moves
    receipt["moves_sha256"] = _moves_digest(moves)
    receipt["moved_counties"] = sorted({move["to_county"] for move in moves})
    return moves, receipt


def _county_value(template, code):
    if pd.api.types.is_numeric_dtype(template.dtype):
        return int(code)
    return f"{int(code):05d}"


def apply_support_moves(
    households, moves, *, county_column=COUNTY_COLUMN, columns=None
):
    """Rewrite the county columns of each moved household, nothing else.

    Each move must still find its household in ``from_county`` and its PUMA,
    so a recorded plan cannot land on another frame silently.
    """
    if not moves:
        return households
    result = households.copy()
    columns = (
        list(columns)
        if columns is not None
        else [county_column]
        + [
            column
            for column in COUNTY_FIPS_COLUMNS
            if column in households.columns and column != county_column
        ]
    )
    ids = pd.Index(result["household_id"].to_numpy())
    if not ids.is_unique:
        raise ValueError("Minimum county support needs unique household ids.")
    codes = county_codes(result, county_column=county_column)
    pumas = _household_pumas(result)
    keys = [move["household_id"] for move in moves]
    positions = ids.get_indexer(keys)
    if (positions < 0).any():
        raise ValueError("A recorded county support move names an absent household.")
    if len(set(positions.tolist())) != len(positions):
        raise ValueError("A household is moved twice by the county support plan.")
    for position, move in zip(positions.tolist(), moves, strict=True):
        if codes[position] != int(move["from_county"]) or pumas[position] != int(
            move["puma"]
        ):
            raise ValueError(
                f"County support move of household {move['household_id']} no longer "
                f"matches its county {codes[position]:05d} or PUMA "
                f"{pumas[position]:07d}."
            )
        if int(move["to_county"]) // 1000 != codes[position] // 1000:
            raise ValueError("A county support move would change the state.")
    for column in columns:
        location = result.columns.get_loc(column)
        for position, move in zip(positions.tolist(), moves, strict=True):
            result.iat[position, location] = _county_value(
                result[column], move["to_county"]
            )
    return result


def with_household_table(frame, households):
    """The same frame with its household table replaced."""
    from microcosm.frame import Frame

    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["household"] = households
    tables.update({name: frame.link(name) for name in frame.links})
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def apply_recorded_support(frame, identity, checkpoint_dir):
    """Re-apply the materialize-time county support moves to a staging frame.

    The consumer export reloads staging, so without this the published file
    would lose the moves calibration saw. The recorded moves must match the
    digest the run identity binds.
    """
    receipt = (identity or {}).get("county_support") or {}
    if not receipt.get("moves"):
        return frame
    report = json.loads((Path(checkpoint_dir) / SUPPORT_FILENAME).read_text())
    moves = (report.get("minimum_support") or {}).get("moves", [])
    if _moves_digest(moves) != receipt.get("moves_sha256") or len(moves) != receipt.get(
        "moves"
    ):
        raise SystemExit(
            f"{SUPPORT_FILENAME} no longer holds the county support moves the run "
            "identity binds; re-run --stage materialize."
        )
    households = apply_support_moves(
        frame.table("household"),
        moves,
        county_column=identity.get("county_column", COUNTY_COLUMN),
        columns=receipt.get("rewritten_columns"),
    )
    return with_household_table(frame, households)


def require_support(dropped, *, capped_smoke=False):
    if dropped and not capped_smoke:
        raise SystemExit(
            f"County cells have no supporting household: {list(dropped)}. Refusing outside a capped smoke."
        )


def add_arguments(parser):
    parser.add_argument("--county-column", default=COUNTY_COLUMN)
    parser.add_argument(
        "--county-holdout-fraction", type=float, default=DEFAULT_COUNTY_HOLDOUT_FRACTION
    )
    parser.add_argument("--county-holdout-salt", default=COUNTY_HOLDOUT_SALT)
    parser.add_argument(
        "--county-facts",
        type=Path,
        help=(
            "The pinned US county facts file (us/chronicle_county_facts.json, "
            "microcosm#922). Required by a --soi-mode state_county materialize."
        ),
    )
    parser.add_argument(
        "--county-facts-sha256",
        help="SHA-256 of --county-facts; must be the county pin's facts_sha256.",
    )
    parser.add_argument(
        "--county-min-support",
        type=int,
        default=DEFAULT_COUNTY_MIN_SUPPORT,
        help=(
            "Households moved, within the same PUMA, into each county whose "
            "ladder population is positive but which has no household "
            f"(default {DEFAULT_COUNTY_MIN_SUPPORT}; 0 disables the step). "
            "Every move is recorded in county_support.json."
        ),
    )


def _validate_county_facts(args, parser, geographies):
    county_facts = getattr(args, "county_facts", None)
    sha256 = getattr(args, "county_facts_sha256", None)
    if county_facts is None:
        if args.soi_mode == SOI_MODE and "materialize" in getattr(
            args, "stages", ["materialize"]
        ):
            parser.error(
                "--soi-mode state_county needs --county-facts and "
                "--county-facts-sha256 (the pinned US county facts, microcosm#922)."
            )
        if sha256 is not None:
            parser.error("--county-facts-sha256 needs --county-facts.")
        return
    if args.soi_mode != SOI_MODE or "county" not in geographies:
        parser.error(
            "--county-facts needs --soi-mode state_county and county in --geographies."
        )
    if sha256 is None:
        parser.error("--county-facts needs --county-facts-sha256.")
    from microcosm.build.us_runtime.chronicle_feed import (
        load_us_chronicle_county_facts_pin,
    )

    pinned = load_us_chronicle_county_facts_pin().facts_sha256
    if sha256 != pinned:
        parser.error(
            f"--county-facts-sha256 {sha256} is not the county pin's "
            f"facts_sha256 {pinned}."
        )
    if not getattr(args, "feed_sha256", None):
        parser.error("--county-facts needs --feed-sha256 for the main feed.")
    feed = getattr(args, "feed", None)
    if feed is not None and (feed_path(feed).resolve() == Path(county_facts).resolve()):
        parser.error("--county-facts must not be the --feed file.")


def _load_county_facts(args):
    """Read the pinned county facts once and record their receipt on args."""

    from microcosm.build.us_runtime.county_facts import load_us_county_facts

    if getattr(args, "county_facts", None) is None:
        raise ValueError(
            "--soi-mode state_county needs --county-facts (microcosm#922)."
        )
    facts = load_us_county_facts(Path(args.county_facts).resolve())
    args.county_facts_receipt = facts.receipt()
    return facts


def validate_arguments(args, parser):
    geographies = [item.strip() for item in args.geographies.split(",") if item.strip()]
    if len(set(geographies)) != len(geographies) or set(geographies) - {
        "state",
        "cd",
        "county",
    }:
        parser.error("--geographies accepts unique state,cd,county values.")
    if not 0 <= args.county_holdout_fraction <= cd_surface.MAX_CD_HOLDOUT_FRACTION:
        parser.error(
            f"--county-holdout-fraction must be in [0, {cd_surface.MAX_CD_HOLDOUT_FRACTION}]."
        )
    if not args.county_column.isidentifier():
        parser.error("--county-column must be a household column name.")
    if not args.county_holdout_salt:
        parser.error("--county-holdout-salt must be a non-empty string.")
    if not 0 <= args.county_min_support <= MAX_COUNTY_MIN_SUPPORT:
        parser.error(f"--county-min-support must be in [0, {MAX_COUNTY_MIN_SUPPORT}].")
    _validate_county_facts(args, parser, geographies)


def prepare_run(args, geographies, populations, *, surface=None):
    # Validation inputs describe one materialization; a new one replaces them.
    for name in (VALIDATION_DESIGN_FILENAME, VALIDATION_MEASURES_FILENAME):
        (Path(args.checkpoint_dir) / name).unlink(missing_ok=True)
    if (
        "county" in geographies
        and args.sample_fraction == 1.0
        and len(populations["county"]) != REQUIRED_COUNTY_COUNT
    ):
        message = f"A full county rung requires {REQUIRED_COUNTY_COUNT} county-equivalents; the ladder declares {len(populations['county'])}. Refusing an incomplete county inventory."
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (args.checkpoint_dir / "county_surface.json").write_text(
            json.dumps({"status": "refused", "error": message})
        )
        raise SystemExit(message)
    if args.soi_mode == SOI_MODE and not {"state", "county"}.issubset(geographies):
        raise SystemExit(
            "--soi-mode state_county requires --geographies state,county (cd is optional)."
        )
    if args.soi_mode == SOI_MODE and "soi" not in {
        item.strip() for item in args.families.split(",")
    }:
        raise SystemExit("--soi-mode state_county requires soi in --families.")
    if args.soi_mode == SOI_MODE and surface is not None:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        try:
            surface.county_facts = _load_county_facts(args)
            surface.county_surface = state_county_soi_surface(
                surface.registry.specs,
                surface.county_facts.rows,
                populations["county"],
            )
            # Refuse incomplete ACS inventories, duplicates, negatives and
            # zero denominators here, before the engine pass.
            surface.acs_households = acs_household_surface(
                surface.county_facts.rows, populations["county"]
            )
            surface.county_validation_design = county_validation_design(
                surface.county_facts.rows, populations["county"]
            )
        except (ValueError, OSError) as error:
            (args.checkpoint_dir / "county_surface.json").write_text(
                json.dumps({"status": "refused", "error": str(error)})
            )
            raise SystemExit(str(error)) from error


def check_support(frame, populations, args):
    """Apply minimum support, write county_support.json, refuse what remains.

    Returns the frame, with county columns rewritten for any moved household.
    """
    if "county" not in populations:
        return frame
    column = getattr(args, "county_column", COUNTY_COLUMN)
    per_county = int(getattr(args, "county_min_support", 0) or 0)
    households = frame.table("household")
    codes = county_codes(households, county_column=column)
    unknown = sorted(set(codes) - set(populations["county"]))
    support = None
    if per_county > 0 and not unknown:
        needed = unsupported_counties(
            households, populations["county"], county_column=column
        )
        overlaps = county_puma_overlaps(args.ladder) if needed else {}
        moves, support = plan_minimum_support(
            households,
            populations["county"],
            overlaps,
            county_column=column,
            per_county=per_county,
        )
        if moves:
            frame = with_household_table(
                frame,
                apply_support_moves(
                    households,
                    moves,
                    county_column=column,
                    columns=support["rewritten_columns"],
                ),
            )
            households = frame.table("household")
            codes = county_codes(households, county_column=column)
        args.county_support_receipt = {
            "file": SUPPORT_FILENAME,
            "households_per_county": per_county,
            "key": COUNTY_MIN_SUPPORT_KEY,
            "moves": len(moves),
            "moves_sha256": support["moves_sha256"],
            "moved_counties": support.get("moved_counties", []),
            "rewritten_columns": support["rewritten_columns"],
        }
    # A state_county run trains an ACS households row for every county, which
    # only household-universe records support; refuse a county with group
    # quarters alone here, before the engine pass, not after it.
    strict = getattr(args, "soi_mode", None) == SOI_MODE
    supporting = codes[support_universe(households)] if strict else codes
    missing = sorted(set(populations["county"]) - set(supporting))
    counts = defaultdict(int)
    for code in supporting:
        counts[int(code)] += 1
    report = {
        "county_column": column,
        "counties": {
            f"{int(code):05d}": {"supporting_households": counts.get(int(code), 0)}
            for code in sorted(populations["county"])
        },
        "unsupported_counties": [f"{int(code):05d}" for code in missing],
        "unknown_counties": [f"{int(code):05d}" for code in unknown],
    }
    if strict:
        report["support_basis"] = (
            "households in the household universe (ACS group-quarters records "
            "support no acs_households target)"
        )
    if support is not None:
        report["minimum_support"] = support
    (args.checkpoint_dir / SUPPORT_FILENAME).write_text(json.dumps(report, indent=2))
    if unknown:
        raise SystemExit(f"Households name counties outside the ladder: {unknown}.")
    smoke = bool(args.allow_partial_geography and args.sample_fraction < 1)
    require_support(report["unsupported_counties"], capped_smoke=smoke)
    return frame


def _write_validation_inputs(checkpoint_dir, design, arrays, report):
    checkpoint_dir = Path(checkpoint_dir)
    design_path = checkpoint_dir / VALIDATION_DESIGN_FILENAME
    measures_path = checkpoint_dir / VALIDATION_MEASURES_FILENAME
    design_path.write_text(
        json.dumps({**design, "household_measures": report}, indent=1, allow_nan=False)
    )
    np.savez(
        measures_path,
        household_id=np.asarray(arrays["household_id"]),
        in_household_universe=np.asarray(arrays["in_household_universe"], np.float64),
        earnings=np.asarray(arrays["earnings"], np.float64),
        household_income_approx=np.asarray(
            arrays["household_income_approx"], np.float64
        ),
    )

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "design_file": VALIDATION_DESIGN_FILENAME,
        "design_sha256": digest(design_path),
        "measures_file": VALIDATION_MEASURES_FILENAME,
        "measures_sha256": digest(measures_path),
        "families": [family.name for family in VALIDATION_FAMILIES],
        "role": "held out: scored after calibration, never in the loss, the "
        "target registry, the roles or the matrix",
    }


def load_validation_inputs(checkpoint_dir, receipt):
    """The validation design and household measures, refused if they changed."""
    checkpoint_dir = Path(checkpoint_dir)
    design_path = checkpoint_dir / receipt["design_file"]
    measures_path = checkpoint_dir / receipt["measures_file"]
    for path, recorded in (
        (design_path, receipt.get("design_sha256")),
        (measures_path, receipt.get("measures_sha256")),
    ):
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != recorded
        ):
            raise SystemExit(
                f"{path.name} changed since materialize (or is missing); the "
                "county validation inputs no longer match the run identity. "
                "Re-run --stage materialize."
            )
    design = json.loads(design_path.read_text())
    with np.load(measures_path, allow_pickle=False) as stored:
        measures = {key: stored[key] for key in stored.files}
    return design, measures


def attach_surface(surface, materialized, frame, populations, args):
    if args.soi_mode != SOI_MODE:
        return materialized.compiled_specs, materialized.matrix
    column = getattr(args, "county_column", COUNTY_COLUMN)
    facts = getattr(surface, "county_facts", None)
    if facts is None:
        message = (
            "prepare_run did not load the county facts; refusing to reopen "
            "the county facts file during attach_surface. Re-run --stage materialize."
        )
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (args.checkpoint_dir / SURFACE_FILENAME).write_text(
            json.dumps({"status": "refused", "error": message})
        )
        raise SystemExit(message)
    county = getattr(surface, "county_surface", None)
    if county is None:
        county = state_county_soi_surface(
            materialized.compiled_specs, facts.rows, populations["county"]
        )
    acs = getattr(surface, "acs_households", None)
    if acs is None:
        acs = acs_household_surface(facts.rows, populations["county"])
    design = getattr(surface, "county_validation_design", None)
    if design is None:
        design = county_validation_design(facts.rows, populations["county"])
    households = frame.table("household")
    matrix = split_state_rows(
        materialized.compiled_specs,
        materialized.matrix,
        county.specs,
        households,
        county_column=column,
    )
    acs_matrix = acs_household_rows(households, acs.specs, county_column=column)
    # A positive target with only zero household measures is unsupported too.
    empty = [
        spec.name
        for rows, specs in ((matrix, county.specs), (acs_matrix, acs.specs))
        for index, spec in enumerate(specs)
        if spec.value != 0 and rows.indptr[index] == rows.indptr[index + 1]
    ]
    (args.checkpoint_dir / SURFACE_FILENAME).write_text(
        json.dumps(
            {
                **county.receipt,
                "acs_households": acs.receipt,
                "unsupported_targets": empty,
            },
            indent=2,
        )
    )
    require_support(
        empty,
        capped_smoke=bool(args.allow_partial_geography and args.sample_fraction < 1),
    )
    county_specs = [*county.specs, *acs.specs]
    matrix = sparse.vstack([matrix, acs_matrix], format="csr")
    if empty:
        unsupported = set(empty)
        keep = [
            index
            for index, spec in enumerate(county_specs)
            if spec.name not in unsupported
        ]
        matrix = sparse.csr_array(matrix[keep])
        county_specs = [county_specs[index] for index in keep]
    args.county_targets_dropped = empty
    args.county_acs_households_receipt = {
        key: acs.receipt[key]
        for key in (
            "family",
            "source_measure_id",
            "new_state_targets",
            "county_targets",
        )
    }
    # The validation families' household measures, from the full frame: the
    # calibrate stage only has the lean checkpoint.
    arrays, measure_report = household_validation_measures(frame)
    args.county_validation_receipt = _write_validation_inputs(
        args.checkpoint_dir, design, arrays, measure_report
    )
    surface.soi_receipt.update(county.receipt)
    return [*materialized.compiled_specs, *county_specs], sparse.vstack(
        [materialized.matrix, matrix], format="csr"
    )


def assign_roles(roles, args):
    for record in roles:
        if record.get("geography_level") == "county":
            record["county_column"] = getattr(args, "county_column", COUNTY_COLUMN)
    return assign_target_roles(
        roles,
        fraction=getattr(
            args, "county_holdout_fraction", DEFAULT_COUNTY_HOLDOUT_FRACTION
        ),
        salt=getattr(args, "county_holdout_salt", COUNTY_HOLDOUT_SALT),
    )


def run_metadata(args, county_holdout):
    metadata = {
        "geographies": ",".join(
            item.strip() for item in args.geographies.split(",") if item.strip()
        ),
        "county_column": getattr(args, "county_column", COUNTY_COLUMN),
        "county_holdout": county_holdout,
        "county_targets_dropped": getattr(args, "county_targets_dropped", []),
    }
    # Only a run that read county facts (or ran the county support step)
    # records them, so a no-county run identity keeps its pre-county bytes.
    for key, attribute in (
        ("county_facts", "county_facts_receipt"),
        ("county_support", "county_support_receipt"),
        ("county_acs_households", "county_acs_households_receipt"),
        ("county_validation", "county_validation_receipt"),
    ):
        receipt = getattr(args, attribute, None)
        if receipt is not None:
            metadata[key] = receipt
    return metadata


def struct_columns(households, existing_columns, *, county_column=COUNTY_COLUMN):
    # Preserve configured county columns through the structure-only checkpoint.
    return [
        column
        for column in households.columns
        if (column.startswith("county_fips") or column == county_column)
        and column not in existing_columns
    ]


def calibration_evidence(
    frame,
    roles,
    matrix,
    design_weights,
    weights,
    target_loss_cap,
    *,
    checkpoint_dir=None,
    identity=None,
):
    """County holdout, validation and reliability evidence (report only).

    Reads the final weights; nothing here feeds back into the solve. With a
    run identity that records county validation inputs, every validation
    family is scored for every county and the reliability report grades on
    it.
    """
    county_roles = [
        record for record in roles if record.get("geography_level") == "county"
    ]
    if not county_roles:
        return {}
    # All county records bind the configured column in the run identity. It is
    # stamped on records by assign_roles so resumed evidence selects it too.
    column = next(
        (
            record.get("county_column")
            for record in county_roles
            if record.get("county_column")
        ),
        COUNTY_COLUMN,
    )
    households = frame.table("household")
    receipt = (identity or {}).get("county_validation")
    validation = design = None
    if receipt:
        if checkpoint_dir is None:
            raise SystemExit(
                "The run identity records county validation inputs; scoring "
                "them needs the checkpoint directory."
            )
        design, measures = load_validation_inputs(checkpoint_dir, receipt)
        validation = score_county_validation(
            design,
            measures,
            households,
            final_weights=weights,
            design_weights=design_weights,
            cap=target_loss_cap,
            county_column=column,
        )
    support = (identity or {}).get("county_support") or {}
    evidence = {
        "county_holdout": score_county_holdout(
            roles,
            matrix,
            design_weights=design_weights,
            final_weights=weights,
            cap=target_loss_cap,
        ),
        "county_reliability": reliability_report(
            households,
            roles,
            matrix,
            final_weights=weights,
            county_column=column,
            validation=validation,
            county_populations=None if design is None else design["county_populations"],
            minimum_support_counties=support.get("moved_counties", []),
            county_names=None if design is None else design["county_names"],
        ),
    }
    if validation is not None:
        evidence["county_validation"] = {
            key: value for key, value in validation.items() if key != "counties"
        }
    return evidence


def _bound_report(summary, identity):
    report = json.loads(json.dumps(summary["county_reliability"]))
    # Include unsupported smoke cells even though population_rows omits them.
    for name in identity.get("population_cells_dropped", []):
        if name.startswith("pop_county_"):
            code = name.removeprefix("pop_county_")
            report["counties"].setdefault(
                code,
                {
                    "state_fips": code[:2],
                    "county_name": None,
                    "supporting_households": 0,
                    "distinct_source_households": 0,
                    "kish_ess": 0.0,
                    "kish_ess_rows": 0.0,
                    "status": "no_support",
                    "training": {
                        "n_targets": 0,
                        "mean_abs_rel_error": None,
                        "max_abs_rel_error": None,
                    },
                    "holdout": {
                        "n_targets": 0,
                        "mean_abs_rel_error": None,
                        "max_abs_rel_error": None,
                    },
                    "validation": {"n_families": 0, "families": {}},
                    "grade_error": None,
                    "grade_driver": None,
                    "grade": "U",
                },
            )
    report.update(
        {key: summary.get(key) for key in ("run_identity_sha256", "weights_sha256")}
    )
    report["county_column"] = identity.get("county_column", COUNTY_COLUMN)
    report["unsupported_targets"] = identity.get("county_targets_dropped", [])
    if identity.get("county_support"):
        report["minimum_support"] = identity["county_support"]
    return report


def feed_path(feed):
    """The hashable facts file for either supported Ledger artifact form."""
    path = Path(feed)
    return path / "consumer_facts.jsonl" if path.is_dir() else path


def write_reports(checkpoint_dir, summary, *, identity):
    """Write county_reliability.json and its flat CSV into the checkpoint."""
    if "county_reliability" not in summary:
        return
    report = _bound_report(summary, identity)
    (Path(checkpoint_dir) / RELIABILITY_FILENAME).write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    (Path(checkpoint_dir) / RELIABILITY_CSV_FILENAME).write_text(
        reliability_csv(report)
    )


def package_reports(contract_files, checkpoint_dir, identity):
    """Add the bound reliability JSON; return the CSV files to package beside it.

    Both must match the current calibration identity, weights and evidence:
    the CSV must be exactly the flattening of the bound report.
    """
    if "county" not in {
        item.strip() for item in identity.get("geographies", "").split(",")
    }:
        return {}
    report = json.loads((Path(checkpoint_dir) / RELIABILITY_FILENAME).read_text())
    summary = json.loads(
        (Path(checkpoint_dir) / "calibration_summary.json").read_text()
    )
    stamp = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        report.get("run_identity_sha256") != stamp
        or not report.get("weights_sha256")
        or report != _bound_report(summary, identity)
    ):
        raise SystemExit(
            "County reliability report does not match the current calibration identity, weights or evidence."
        )
    csv_path = Path(checkpoint_dir) / RELIABILITY_CSV_FILENAME
    expected = reliability_csv(report)
    if not csv_path.is_file() or csv_path.read_text() != expected:
        raise SystemExit(
            f"{RELIABILITY_CSV_FILENAME} is missing or is not the flattening of "
            f"the bound {RELIABILITY_FILENAME}; re-run --stage calibrate."
        )
    contract_files[RELIABILITY_FILENAME] = report
    return {RELIABILITY_CSV_FILENAME: expected}


def refresh_recipe(command, materialize_rss):
    if "county" not in {
        item.strip() for item in materialize_rss.get("geographies", "").split(",")
    }:
        return command
    holdout = materialize_rss["county_holdout"]
    command = (
        command + f" --geographies {shlex.quote(materialize_rss['geographies'])}"
        f" --county-column {shlex.quote(materialize_rss.get('county_column', COUNTY_COLUMN))}"
        f" --county-holdout-fraction {float(holdout['fraction'])!r}"
        f" --county-holdout-salt {shlex.quote(holdout['salt'])}"
    )
    # Like --feed, the county facts file is a placeholder: the recipe names
    # the pinned digest, never the build machine's path.
    facts = materialize_rss.get("county_facts")
    if facts:
        command += (
            " --county-facts <county-facts.jsonl>"
            f" --county-facts-sha256 {facts['facts_sha256']}"
        )
    # A run without a support receipt moved no household (the step was off,
    # or the run predates it), so its replay must turn the step off too.
    support = materialize_rss.get("county_support") or {}
    command += f" --county-min-support {int(support.get('households_per_county', 0))}"
    return command


def _is_county(spec):
    return (
        spec.metadata.get(
            "ledger_geography_level", spec.metadata.get("geography_level")
        )
        == "county"
    )


def _is_acs_family(spec):
    return spec.family == ACS_HOUSEHOLDS_FAMILY


def loss_row_mapping(specs):
    """Extend the existing loss formula with county count/amount budgets.

    Each county concept shares one budget across its state's counties, so
    adding thousands of counties does not multiply its loss budget. ACS
    household rows are counts; each state households parent has its own
    budget, as each state population row does.
    """
    from microcosm.build.us_runtime import target_loss_weights as weighting

    base = weighting.US_ACS_LOCAL_TARGET_LOSS_ROW_MAPPING
    if not any(_is_county(spec) or _is_acs_family(spec) for spec in specs):
        return base

    def basis(spec):
        if _is_acs_family(spec) or (
            _is_county(spec) and spec.family == "census_population"
        ):
            return weighting.COUNT_BASIS
        return base.value_basis(spec)

    def key(spec):
        if _is_acs_family(spec) and not _is_county(spec):
            return (basis(spec), spec.entity, spec.period, spec.family, spec.name)
        if not _is_county(spec):
            return base.concept_budget_key(spec)
        code = spec.metadata["county_fips"]
        concept = spec.metadata.get(
            "state_county_parent_target_name", "resident_population"
        )
        return (
            basis(spec),
            spec.entity,
            spec.period,
            spec.family,
            "county",
            code[:2],
            concept,
        )

    def validate(rows, bases, keys):
        indices = [
            index
            for index, spec in enumerate(rows)
            if not (_is_county(spec) or _is_acs_family(spec))
        ]
        if base.validate is not None:
            base.validate(
                [rows[index] for index in indices],
                [bases[index] for index in indices],
                [keys[index] for index in indices],
            )

    return weighting.TargetLossRowMapping(
        "us_acs_local_county.v2", basis, key, validate
    )


def target_loss_weights(specs, multipliers):
    from microcosm.build.us_runtime import target_loss_weights as weighting

    return weighting.target_loss_weights(
        specs, multipliers, row_mapping=loss_row_mapping(specs)
    )
