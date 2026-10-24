"""Engine-free county rung for the ACS local release.

County SOI facts supply *shares*, never a second state level. Children split
the stored state CSR row, including its return universe and engine values.
Population, splitting, holdout and reliability helpers also accept future ACS
county families. See docs/us-acs-local-county-rung.md for the input contract.
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
SOI_MODE = "state_county"
COUNTY_PREFIX = "0500000US"
# The LWI release contract covers the 50 states and DC on planning-region
# county boundaries. A self-consistent but incomplete ladder is not complete.
REQUIRED_COUNTY_COUNT = 3_144


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


def state_county_soi_surface(state_specs, facts, county_populations):
    """Rebase complete county fact blocks onto existing broad state totals.

    Only measures actually carried by county facts and by an unsliced state
    parent are eligible. New ACS families can produce the same child metadata
    and use split_state_rows without extending the engine materializer.
    """
    blocks = defaultdict(dict)
    for fact in facts:
        geography = fact.get("geography", {})
        observed = fact.get("observed_measure", {})
        if (
            geography.get("level") != "county"
            or observed.get("source_name") != "irs_soi"
        ):
            continue
        geoid = str(geography.get("id", ""))
        if not geoid.startswith(COUNTY_PREFIX) or len(geoid) != len(COUNTY_PREFIX) + 5:
            raise ValueError(f"Invalid county SOI geography {geoid!r}.")
        code = int(geoid[-5:])
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
            "Ledger feed has no county SOI facts; export soi-county-2022 and repin the feed."
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
        if key[1] not in {block[1] for block in blocks}:
            continue
        if key in parents:
            raise ValueError(f"Two state SOI parents for county concept {key}.")
        parents[key] = spec
    if not parents:
        raise ValueError(
            "No county SOI concept has an already materialized broad state parent."
        )
    specs, receipts = [], []
    for key, parent in sorted(parents.items()):
        state, measure, period = key
        expected = {
            int(code) for code in county_populations if int(code) // 1000 == state
        }
        block = blocks.get(key, {})
        missing, extra = (
            sorted(expected - block.keys()),
            sorted(block.keys() - expected),
        )
        if not expected or missing or extra:
            raise ValueError(
                f"Incomplete county SOI block state={state:02d}, concept={measure}, period={period}; "
                f"missing={[f'{code:05d}' for code in missing]}, unexpected={[f'{code:05d}' for code in extra]}. "
                "County facts, ladder and county column must use matching boundaries."
            )
        total = math.fsum(value for value, _fact in block.values())
        if total <= 0:
            raise ValueError(
                f"County SOI block {key} has no positive share denominator."
            )
        codes = sorted(expected)
        shares = [block[code][0] / total for code in codes]
        largest = max(range(len(codes)), key=lambda index: shares[index])
        shares[largest] += 1.0 - math.fsum(shares)
        targets = [parent.value * share for share in shares]
        targets[largest] += parent.value - math.fsum(targets)
        state_population = math.fsum(county_populations[code] for code in codes)
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
                "county_population": str(county_populations[code]),
                "state_population": str(state_population),
            }
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
                "counties": len(codes),
                "share_sum": math.fsum(shares),
                "target_sum": math.fsum(targets),
            }
        )
    # Counties without a state parent cannot silently disappear.
    covered = {int(spec.metadata["county_fips"]) for spec in specs}
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
            "new_engine_columns": 0,
            "basis": "state target x county fact / state sum of county facts",
        },
    )


def split_state_rows(
    state_specs, state_matrix, county_specs, households, *, county_column=COUNTY_COLUMN
):
    """Partition stored parent rows exactly; never materialize new columns."""
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
        child_codes = [
            int(county_specs[index].metadata["county_fips"]) for index in indices
        ]
        if len(set(child_codes)) != len(child_codes):
            raise ValueError(f"County rows of {parent_name} overlap.")
        state = int(parent.metadata["state_fips"])
        present = codes // 1000 == state
        if not np.isin(codes[present], child_codes).all():
            raise ValueError(f"Households of {parent_name} fall in no county row.")
        start, end = matrix.indptr[parent_index : parent_index + 2]
        positions, values = matrix.indices[start:end], matrix.data[start:end]
        if np.any(codes[positions] // 1000 != state):
            raise ValueError(
                f"State row {parent_name} contains households outside its state."
            )
        for index, code in zip(indices, child_codes, strict=True):
            selected = codes[positions] == code
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
    """A/B/C require ESS >=400/100/30 and error <=10/20/50%; else D.

    U means no holdout evidence. ESS is a precision diagnostic, not a standard
    error or a guarantee of unbiased county estimates.
    """
    if error is None or not math.isfinite(error):
        return "U"
    for grade, minimum, maximum in (("A", 400, 0.1), ("B", 100, 0.2), ("C", 30, 0.5)):
        if ess >= minimum and error <= maximum:
            return grade
    return "D"


def reliability_report(
    households,
    records,
    matrix,
    *,
    final_weights,
    county_populations=None,
    county_column=COUNTY_COLUMN,
):
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
            by_county[int(record["county_fips"])].append(index)
    counties = sorted(set(codes) | set(by_county) | set(county_populations or {}))
    result = {}
    provenance = {"household_spine", "household_source_id"}.issubset(households.columns)
    for code in counties:
        mask = codes == code
        source = (
            households.loc[mask, ["household_spine", "household_source_id"]]
            if provenance
            else None
        )
        distinct = (
            len(source.drop_duplicates())
            if source is not None and not source.isna().any().any()
            else None
        )
        cell = {
            "state_fips": f"{int(code) // 1000:02d}",
            "supporting_households": int(mask.sum()),
            "distinct_source_households": distinct,
            "kish_ess": cd_surface.kish_ess(weights[mask]),
            "status": "supported" if mask.any() else "no_support",
        }
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
        errors = [
            cell[key]["max_abs_rel_error"]
            for key in ("training", "holdout")
            if cell[key]["max_abs_rel_error"] is not None
        ]
        error = max(errors) if errors and cell["holdout"]["n_targets"] else None
        cell["grade"] = reliability_grade(cell["kish_ess"], error)
        result[f"{int(code):05d}"] = cell
    return {
        "schema_version": 1,
        "county_column": county_column,
        "counties": result,
        "grade_rule": {
            "A": "ESS >=400 and max error <=0.10",
            "B": "ESS >=100 and max error <=0.20",
            "C": "ESS >=30 and max error <=0.50",
            "D": "otherwise with holdout evidence",
            "U": "no county holdout evidence",
            "error": "maximum training/holdout absolute relative error; denominator max(abs(target),1)",
        },
    }


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


def prepare_run(args, geographies, populations, *, surface=None):
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
        from microcosm.build.ledger_artifact import load_ledger_consumer_artifact

        artifact = load_ledger_consumer_artifact(str(args.feed))
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        try:
            surface.county_surface = state_county_soi_surface(
                surface.registry.specs, artifact.facts, populations["county"]
            )
        except ValueError as error:
            (args.checkpoint_dir / "county_surface.json").write_text(
                json.dumps({"status": "refused", "error": str(error)})
            )
            raise SystemExit(str(error)) from error


def check_support(frame, populations, args):
    if "county" not in populations:
        return
    households = frame.table("household")
    codes = county_codes(
        households, county_column=getattr(args, "county_column", COUNTY_COLUMN)
    )
    unknown = sorted(set(codes) - set(populations["county"]))
    missing = sorted(set(populations["county"]) - set(codes))
    report = {
        "county_column": getattr(args, "county_column", COUNTY_COLUMN),
        "counties": {
            f"{int(code):05d}": {"supporting_households": int((codes == code).sum())}
            for code in sorted(populations["county"])
        },
        "unsupported_counties": [f"{int(code):05d}" for code in missing],
        "unknown_counties": [f"{int(code):05d}" for code in unknown],
    }
    (args.checkpoint_dir / "county_support.json").write_text(
        json.dumps(report, indent=2)
    )
    if unknown:
        raise SystemExit(f"Households name counties outside the ladder: {unknown}.")
    smoke = bool(args.allow_partial_geography and args.sample_fraction < 1)
    require_support(report["unsupported_counties"], capped_smoke=smoke)


def attach_surface(surface, materialized, frame, populations, args):
    if args.soi_mode != SOI_MODE:
        return materialized.compiled_specs, materialized.matrix
    county = getattr(surface, "county_surface", None)
    if county is None:
        from microcosm.build.ledger_artifact import load_ledger_consumer_artifact

        artifact = load_ledger_consumer_artifact(str(args.feed))
        county = state_county_soi_surface(
            materialized.compiled_specs, artifact.facts, populations["county"]
        )
    matrix = split_state_rows(
        materialized.compiled_specs,
        materialized.matrix,
        county.specs,
        frame.table("household"),
        county_column=getattr(args, "county_column", COUNTY_COLUMN),
    )
    # A positive target with only zero household measures is unsupported too.
    empty = [
        spec.name
        for index, spec in enumerate(county.specs)
        if spec.value != 0 and matrix.indptr[index] == matrix.indptr[index + 1]
    ]
    (args.checkpoint_dir / "county_surface.json").write_text(
        json.dumps({**county.receipt, "unsupported_targets": empty}, indent=2)
    )
    require_support(
        empty,
        capped_smoke=bool(args.allow_partial_geography and args.sample_fraction < 1),
    )
    county_specs = list(county.specs)
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
    return {
        "geographies": ",".join(
            item.strip() for item in args.geographies.split(",") if item.strip()
        ),
        "county_column": getattr(args, "county_column", COUNTY_COLUMN),
        "county_holdout": county_holdout,
        "county_targets_dropped": getattr(args, "county_targets_dropped", []),
    }


def struct_columns(households, existing_columns, *, county_column=COUNTY_COLUMN):
    # Preserve configured county columns through the structure-only checkpoint.
    return [
        column
        for column in households.columns
        if (column.startswith("county_fips") or column == county_column)
        and column not in existing_columns
    ]


def calibration_evidence(
    frame, roles, matrix, design_weights, weights, target_loss_cap
):
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
    return {
        "county_holdout": score_county_holdout(
            roles,
            matrix,
            design_weights=design_weights,
            final_weights=weights,
            cap=target_loss_cap,
        ),
        "county_reliability": reliability_report(
            frame.table("household"),
            roles,
            matrix,
            final_weights=weights,
            county_column=column,
        ),
    }


def _bound_report(summary, identity):
    report = json.loads(json.dumps(summary["county_reliability"]))
    # Include unsupported smoke cells even though population_rows omits them.
    for name in identity.get("population_cells_dropped", []):
        if name.startswith("pop_county_"):
            code = name.removeprefix("pop_county_")
            report["counties"].setdefault(
                code,
                {
                    "supporting_households": 0,
                    "distinct_source_households": 0,
                    "kish_ess": 0.0,
                    "training": {"n_targets": 0, "mean_abs_rel_error": None},
                    "holdout": {"n_targets": 0, "mean_abs_rel_error": None},
                    "grade": "U",
                    "status": "no_support",
                },
            )
    report.update(
        {key: summary.get(key) for key in ("run_identity_sha256", "weights_sha256")}
    )
    report["county_column"] = identity.get("county_column", COUNTY_COLUMN)
    report["unsupported_targets"] = identity.get("county_targets_dropped", [])
    return report


def feed_path(feed):
    """The hashable facts file for either supported Ledger artifact form."""
    path = Path(feed)
    return path / "consumer_facts.jsonl" if path.is_dir() else path


def write_reports(checkpoint_dir, summary, *, identity):
    if "county_reliability" not in summary:
        return
    (Path(checkpoint_dir) / RELIABILITY_FILENAME).write_text(
        json.dumps(_bound_report(summary, identity), indent=2, allow_nan=False)
    )


def package_reports(contract_files, checkpoint_dir, identity):
    if "county" not in {
        item.strip() for item in identity.get("geographies", "").split(",")
    }:
        return
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
    contract_files[RELIABILITY_FILENAME] = report


def refresh_recipe(command, materialize_rss):
    if "county" not in {
        item.strip() for item in materialize_rss.get("geographies", "").split(",")
    }:
        return command
    holdout = materialize_rss["county_holdout"]
    return (
        command + f" --geographies {shlex.quote(materialize_rss['geographies'])}"
        f" --county-column {shlex.quote(materialize_rss.get('county_column', COUNTY_COLUMN))}"
        f" --county-holdout-fraction {float(holdout['fraction'])!r}"
        f" --county-holdout-salt {shlex.quote(holdout['salt'])}"
    )


def _is_county(spec):
    return (
        spec.metadata.get(
            "ledger_geography_level", spec.metadata.get("geography_level")
        )
        == "county"
    )


def loss_row_mapping(specs):
    """Extend the existing loss formula with county count/amount budgets.

    Each county concept shares one budget across its state's counties, so
    adding thousands of counties does not multiply its loss budget.
    """
    from microcosm.build.us_runtime import target_loss_weights as weighting

    base = weighting.US_ACS_LOCAL_TARGET_LOSS_ROW_MAPPING
    if not any(_is_county(spec) for spec in specs):
        return base

    def basis(spec):
        if _is_county(spec) and spec.family == "census_population":
            return weighting.COUNT_BASIS
        return base.value_basis(spec)

    def key(spec):
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
        indices = [index for index, spec in enumerate(rows) if not _is_county(spec)]
        if base.validate is not None:
            base.validate(
                [rows[index] for index in indices],
                [bases[index] for index in indices],
                [keys[index] for index in indices],
            )

    return weighting.TargetLossRowMapping(
        "us_acs_local_county.v1", basis, key, validate
    )


def target_loss_weights(specs, multipliers):
    from microcosm.build.us_runtime import target_loss_weights as weighting

    return weighting.target_loss_weights(
        specs, multipliers, row_mapping=loss_row_mapping(specs)
    )
