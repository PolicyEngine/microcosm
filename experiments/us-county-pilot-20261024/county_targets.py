"""County target rows, metrics and the pilot driver for the LWI county pilot.

Every county row is a **within-state share** target. For a county ``c`` in
state ``s`` and a measure ``m`` (households, persons, earnings):

    target(c) = M_s(w0) * public(c) / sum(public(c') for c' in s)

where ``M_s(w0)`` is the state total of ``m`` under the release weights
``w0``. Each state total stays where the release calibration put it; the county
rows only redistribute it across the state's counties in proportion to the
public county figures. The pilot therefore tests whether the microdata can
carry county distributions, not whether the release's state levels are right
(those were already calibrated to national and state targets).

Trained families:

* ``county_households``: ACS 2020-2024 5-year B11001_001E (households).
* ``county_population``: Census PEP Vintage 2024 county population, July 2024.
* ``county_earnings``: ACS B19061_001E (aggregate household earnings).
* ``cd_households`` / ``cd_population``: each congressional district's total
  under ``w0``, held so the county rows cannot undo the district fit.

Held-out families (never trained; scored before and after):

* ``county_earning_households``: ACS B19051_002E (households with earnings).
* ``county_soi_wages``: IRS SOI county TY2022 A00200 (wages and salaries).
* ``county_income``: ACS B19025_001E (aggregate household income), against an
  approximate household income (earnings plus the stored income inputs).
"""

from __future__ import annotations

import csv
import io
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

PERIOD = 2024
PR_STATE = "72"

ACS_CELLS = {
    "households": ("B11001", "B11001_001E"),
    "earning_households": ("B19051", "B19051_002E"),
    "aggregate_earnings": ("B19061", "B19061_001E"),
    "aggregate_income": ("B19025", "B19025_001E"),
    "median_income": ("B19013", "B19013_001E"),
    "population_acs": ("B01003", "B01003_001E"),
}
ACS_SOURCE = "census_acs_5yr_2020_2024_county"
PEP_SOURCE = "census_pep_v2024_county_population_2024"
SOI_SOURCE = "irs_soi_county_ty2022"
HOLD_SOURCE = f"release_w0:{'populace-us-2024-buildo-acs-local-767312d60'}"

#: Provisional reliability grades on a county's effective sample size
#: (Kish, from the calibrated weights). Lower bounds, best first.
RELIABILITY_GRADES = (("A", 300.0), ("B", 100.0), ("C", 30.0), ("D", 0.0))
SIZE_BANDS = ((0, 10_000), (10_000, 50_000), (50_000, 250_000), (250_000, np.inf))


# ---------------------------------------------------------------------------
# Public county sources
# ---------------------------------------------------------------------------


def read_acs_county(raw: bytes, variable: str) -> pd.Series:
    """One ACS county estimate, indexed by 5-digit FIPS, Puerto Rico dropped.

    The Census API encodes suppressed or unavailable cells as large negative
    sentinels (for example -666666666); those become NaN.
    """

    rows = json.loads(raw)
    header = rows[0]
    value_at = header.index(variable)
    state_at = header.index("state")
    county_at = header.index("county")
    out = {}
    for row in rows[1:]:
        if row[state_at] == PR_STATE:
            continue
        value = float(row[value_at]) if row[value_at] is not None else np.nan
        out[row[state_at] + row[county_at]] = value if value >= 0 else np.nan
    return pd.Series(out, name=variable, dtype=np.float64).sort_index()


def read_pep_county(raw: bytes, year: int = PERIOD) -> pd.Series:
    """PEP county population for ``year``, indexed by 5-digit FIPS."""

    table = pd.read_csv(
        io.BytesIO(raw), dtype=str, encoding="latin-1", keep_default_na=False
    )
    table = table[table["SUMLEV"] == "050"]
    fips = table["STATE"].str.zfill(2) + table["COUNTY"].str.zfill(3)
    values = pd.to_numeric(table[f"POPESTIMATE{year}"]).to_numpy(np.float64)
    return pd.Series(values, index=fips.to_numpy(), name="pep").sort_index()


def read_soi_county(raw: bytes, column: str) -> pd.Series:
    """One SOI county column (no AGI split), indexed by 5-digit FIPS."""

    table = pd.read_csv(
        io.BytesIO(raw), dtype=str, encoding="latin-1", keep_default_na=False
    )
    table = table[(table["COUNTYFIPS"] != "000") & (table["agi_stub"] == "0")]
    fips = table["STATEFIPS"].str.zfill(2) + table["COUNTYFIPS"].str.zfill(3)
    values = pd.to_numeric(table[column]).to_numpy(np.float64)
    return pd.Series(values, index=fips.to_numpy(), name=column).sort_index()


# ---------------------------------------------------------------------------
# Connecticut: 2020 counties -> 2022 planning regions
# ---------------------------------------------------------------------------


def ct_planning_region_tables(
    crosswalk_text: str,
    block_mcd_text: str,
    tract_puma_text: str,
    tract_population_json: str,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """CT lookups from public Census files.

    Returns ``(tract_region, cell_regions)``: each 2020 CT tract's planning
    region (tracts nest in towns, so each has one), and for each
    ``"<7-digit PUMA>|<5-digit 2020 county>"`` cell the 2020 census population
    of each planning region in it. The files are the Census CT county-to-town
    crosswalk, the 2020 block assignment file's town layer, the 2020 tract to
    2020 PUMA relationship file and 2020 PL tract populations.
    """

    rows = list(csv.reader(io.StringIO(crosswalk_text.lstrip("﻿")), delimiter="|"))
    header = [cell.split("\n")[0].strip() for cell in rows[0]]
    column = {name: index for index, name in enumerate(header)}
    town_region: dict[str, str] = {}
    for row in rows[1:]:
        if len(row) < len(header) or not row[0].strip():
            continue
        cousub = row[column["COUSUBFP"]].strip()
        if cousub == "00000":
            continue
        old_county = row[column["OLD_COUNTYFP"]].strip()
        new_county = row[column["NEW_COUNTYFP"]].strip()
        town_region["09" + old_county + cousub] = "09" + new_county
    lines = block_mcd_text.splitlines()
    if lines[0].strip() != "BLOCKID|COUNTYFP|COUSUBFP":
        raise ValueError(f"unexpected BAF MCD header {lines[0]!r}")
    tract_regions: dict[str, set[str]] = defaultdict(set)
    for line in lines[1:]:
        if not line.strip():
            continue
        block, county, cousub = line.strip().split("|")
        region = town_region.get("09" + county + cousub)
        if region is not None:
            tract_regions[block[:11]].add(region)
    split = {tract for tract, regions in tract_regions.items() if len(regions) > 1}
    if split:
        raise ValueError(f"CT tracts span planning regions: {sorted(split)[:5]}")
    tract_region = {
        tract: next(iter(regions)) for tract, regions in tract_regions.items()
    }
    population_rows = json.loads(tract_population_json)
    pop_header = population_rows[0]
    population = {
        "09" + row[pop_header.index("county")] + row[pop_header.index("tract")]: float(
            row[pop_header.index("P1_001N")]
        )
        for row in population_rows[1:]
    }
    cells: dict[str, Counter] = defaultdict(Counter)
    reader = csv.DictReader(io.StringIO(tract_puma_text.lstrip("﻿")))
    for row in reader:
        if row["STATEFP"] != "09":
            continue
        tract = "09" + row["COUNTYFP"] + row["TRACTCE"]
        people = population.get(tract, 0.0)
        region = tract_region.get(tract)
        if region is None:
            if people:
                raise ValueError(f"populated CT tract {tract} has no planning region")
            continue
        cells["09" + row["PUMA5CE"] + "|09" + row["COUNTYFP"]][region] += people
    return tract_region, {cell: dict(regions) for cell, regions in cells.items()}


def assign_ct_regions(
    puma: np.ndarray,
    county: np.ndarray,
    cell_regions: Mapping[str, Mapping[str, float]],
    *,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Planning region for each CT household from its PUMA and 2020 county.

    A cell inside one region maps directly; a cell spanning regions draws by
    the regions' 2020 population in that cell, with a seeded generator, the
    way the PUMA ladder draws a county inside a PUMA.
    """

    rng = np.random.default_rng(seed)
    keys = np.char.add(np.char.add(puma.astype(str), "|"), county.astype(str))
    out = np.empty(len(keys), dtype=object)
    report = {"households": int(len(keys)), "drawn": 0, "cells_drawn": {}}
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        regions = cell_regions.get(str(key))
        if not regions:
            raise ValueError(f"CT household cell {key} is not in the PUMA/county table")
        names = sorted(regions)
        if len(names) == 1:
            out[rows] = names[0]
            continue
        weights = np.array([regions[name] for name in names], dtype=np.float64)
        out[rows] = rng.choice(names, size=len(rows), p=weights / weights.sum())
        report["drawn"] += int(len(rows))
        report["cells_drawn"][str(key)] = {
            "households": int(len(rows)),
            "population_by_region": {name: regions[name] for name in names},
            "drawn_by_region": dict(Counter(out[rows].tolist())),
        }
    return out, report


# ---------------------------------------------------------------------------
# Target design
# ---------------------------------------------------------------------------


def within_state_share_values(
    public: pd.Series, state_totals: Mapping[str, float]
) -> pd.Series:
    """``state_total * public(c) / sum_state(public)`` for every county ``c``."""

    state = public.index.str[:2]
    state_sum = public.groupby(state).transform("sum")
    totals = pd.Series(state, index=public.index).map(state_totals).astype(np.float64)
    return totals * (public / state_sum)


def _indicator_matrix(codes: np.ndarray, values: np.ndarray, order: list[str]):
    """Rows = ``order`` codes; row ``k`` holds ``values`` where ``codes == order[k]``."""

    position = {code: index for index, code in enumerate(order)}
    rows = np.array([position.get(code, -1) for code in codes], dtype=np.int64)
    keep = (rows >= 0) & (values != 0)
    columns = np.flatnonzero(keep)
    return sparse.csr_array(
        (values[keep].astype(np.float64), (rows[keep], columns)),
        shape=(len(order), len(codes)),
    )


def build_design(households: pd.DataFrame, sources: Mapping[str, bytes]) -> dict:
    """Every pilot row: its family, geography, role, value and matrix row."""

    w0 = households["weight"].to_numpy(np.float64)
    county = households["county_fips_2023"].astype(str).str.zfill(5).to_numpy()
    state = np.array([code[:2] for code in county])
    district = (
        households["congressional_district_geoid"]
        .astype(np.int64)
        .astype(str)
        .str.zfill(4)
    ).to_numpy()
    earnings = households["earnings"].to_numpy(np.float64)
    persons = households["n_persons"].to_numpy(np.float64)
    ones = np.ones(len(households))
    income = households["household_income_approx"].to_numpy(np.float64)

    acs = {
        key: read_acs_county(sources[f"acs5_2024_{table}_county"], variable)
        for key, (table, variable) in ACS_CELLS.items()
    }
    pep = read_pep_county(sources["pep_county_2024"])
    soi_wages = read_soi_county(sources["soi_county_2022"], "A00200")
    counties = sorted(acs["households"].index)
    for name, series in {**acs, "pep": pep, "soi_wages": soi_wages}.items():
        extra = sorted(set(series.index) - set(counties))
        if extra:
            raise ValueError(f"{name}: counties outside the ACS set {extra[:5]}")
    households_counties = set(county)
    unknown = sorted(households_counties - set(counties))
    if unknown:
        raise ValueError(
            f"household counties not in the public county set: {unknown[:5]}"
        )

    def state_totals(values: np.ndarray) -> dict[str, float]:
        return pd.Series(w0 * values).groupby(state).sum().to_dict()

    measure_of = {
        "households": ones,
        "persons": persons,
        "earnings": earnings,
        "earning_households": (earnings != 0).astype(np.float64),
        "wages": households["employment_income_before_lsr"].to_numpy(np.float64),
        "income": income,
    }
    families = [
        # (family, role, measure, public series, source)
        ("county_households", "train", "households", acs["households"], ACS_SOURCE),
        ("county_population", "train", "persons", pep, PEP_SOURCE),
        ("county_earnings", "train", "earnings", acs["aggregate_earnings"], ACS_SOURCE),
        (
            "county_earning_households",
            "holdout",
            "earning_households",
            acs["earning_households"],
            ACS_SOURCE,
        ),
        ("county_soi_wages", "holdout", "wages", soi_wages, SOI_SOURCE),
        ("county_income", "holdout", "income", acs["aggregate_income"], ACS_SOURCE),
    ]
    records: list[dict] = []
    blocks = []
    dropped: dict[str, list[str]] = {}
    for family, role, measure, public, source in families:
        public = public.reindex(counties)
        if public.isna().any():
            # A trained family must cover every county; a held-out family is
            # scored on the counties its source publishes (SOI omits Kalawao),
            # with the state share taken over those counties.
            if role == "train":
                raise ValueError(
                    f"{family}: {int(public.isna().sum())} counties without a value"
                )
            dropped[family] = sorted(public.index[public.isna()])
            public = public.dropna()
        covered = list(public.index)
        values = within_state_share_values(public, state_totals(measure_of[measure]))
        blocks.append(_indicator_matrix(county, measure_of[measure], covered))
        for code in covered:
            records.append(
                {
                    "name": f"{family}:{code}",
                    "family": family,
                    "role": role,
                    "geography": code,
                    "state": code[:2],
                    "measure": measure,
                    "public_value": float(public[code]),
                    "value": float(values[code]),
                    "source": source,
                }
            )
    districts = sorted(set(district))
    for family, measure in (
        ("cd_households", "households"),
        ("cd_population", "persons"),
    ):
        block = _indicator_matrix(district, measure_of[measure], districts)
        current = block @ w0
        blocks.append(block)
        for index, code in enumerate(districts):
            records.append(
                {
                    "name": f"{family}:{code}",
                    "family": family,
                    "role": "train",
                    "geography": code,
                    "state": code[:2],
                    "measure": measure,
                    "public_value": float("nan"),
                    "value": float(current[index]),
                    "source": HOLD_SOURCE,
                }
            )
    matrix = sparse.csr_array(sparse.vstack(blocks, format="csr"))
    matrix.sort_indices()
    support = np.diff(matrix.indptr)
    for record, count in zip(records, support, strict=True):
        record["support_records"] = int(count)
        if record["role"] == "train" and count == 0:
            record["role"] = "unsupported"
    return {
        "records": records,
        "matrix": matrix,
        "counties": counties,
        "dropped": dropped,
    }


def target_set(records: list[dict], matrix: sparse.csr_array, n: int):
    """The calibrator's TargetSet: trained rows only, each a CSR-row measure."""

    from microcosm.calibrate.target import Target, TargetSet

    targets = []
    for index, record in enumerate(records):
        if record["role"] != "train":
            continue
        start, stop = int(matrix.indptr[index]), int(matrix.indptr[index + 1])
        indices = matrix.indices[start:stop]
        data = matrix.data[start:stop]

        def measure(frame, indices=indices, data=data) -> np.ndarray:
            values = np.zeros(n, dtype=np.float64)
            values[indices] = data
            return values

        measure.__qualname__ = f"county_pilot_row[{index}]"
        targets.append(
            Target(
                name=record["name"],
                entity="household",
                measure=measure,
                value=record["value"],
                period=PERIOD,
                source=record["source"],
            )
        )
    return TargetSet(targets)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def kish_ess(weights: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"g": groups, "w": weights, "w2": weights**2})
    grouped = frame.groupby("g").agg(
        records=("w", "size"), w=("w", "sum"), w2=("w2", "sum")
    )
    grouped["ess"] = grouped["w"] ** 2 / grouped["w2"]
    return grouped[["records", "ess"]]


def grade(ess: float) -> str:
    for label, floor in RELIABILITY_GRADES:
        if ess >= floor:
            return label
    return RELIABILITY_GRADES[-1][0]


def weighted_median_by(
    values: np.ndarray, weights: np.ndarray, groups: np.ndarray
) -> pd.Series:
    order = np.lexsort((values, groups))
    g, v, w = groups[order], values[order], weights[order]
    frame = pd.DataFrame({"g": g, "v": v, "w": w})
    frame["cum"] = frame.groupby("g")["w"].cumsum()
    frame["half"] = frame.groupby("g")["w"].transform("sum") / 2.0
    return frame[frame["cum"] >= frame["half"]].groupby("g")["v"].first()


def fit_table(records: list[dict], matrix: sparse.csr_array, w0, w1) -> pd.DataFrame:
    table = pd.DataFrame(records)
    table["before"] = matrix @ w0
    table["after"] = matrix @ w1
    for stage in ("before", "after"):
        with np.errstate(divide="ignore", invalid="ignore"):
            table[f"rel_error_{stage}"] = table[stage] / table["value"] - 1.0
    return table


def summarize_fit(table: pd.DataFrame, county_population: pd.Series) -> dict:
    out: dict = {}
    for family, rows in table.groupby("family"):
        entry = {"role": rows["role"].mode().iat[0], "rows": int(len(rows))}
        for stage in ("before", "after"):
            error = rows[f"rel_error_{stage}"].abs().replace([np.inf], np.nan).dropna()
            entry[stage] = {
                "median_abs_rel_error": float(error.median()),
                "p90_abs_rel_error": float(error.quantile(0.9)),
                "within_5pct": float((error <= 0.05).mean()),
                "within_10pct": float((error <= 0.10).mean()),
                "within_25pct": float((error <= 0.25).mean()),
            }
        if family.startswith("county_"):
            bands = {}
            population = rows["geography"].map(county_population)
            for low, high in SIZE_BANDS:
                band = rows[(population >= low) & (population < high)]
                label = f"{low:,}-{high:,.0f}" if np.isfinite(high) else f"{low:,}+"
                bands[label] = {
                    stage: float(
                        band[f"rel_error_{stage}"]
                        .abs()
                        .replace([np.inf], np.nan)
                        .median()
                    )
                    for stage in ("before", "after")
                } | {"counties": int(len(band))}
            entry["median_abs_rel_error_by_county_population"] = bands
        out[family] = entry
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_lean_frame(path: Path):
    from microcosm.frame import Frame, WeightKind, Weights, read_frame_table
    from microcosm.frame.units import US_SCHEMA

    tables = {}
    with pd.HDFStore(path, mode="r") as store:
        for key in (
            "household",
            "person",
            "tax_unit",
            "spm_unit",
            "family",
            "marital_unit",
        ):
            tables[key] = read_frame_table(store, key)
    weights = tables["household"].pop("household_weight").to_numpy(np.float64)
    return Frame(
        tables, US_SCHEMA, {"household": Weights(weights, WeightKind.CALIBRATED)}
    ), weights


def run_pilot(
    extract_dir: Path,
    out_dir: Path,
    sources: Mapping[str, bytes],
    settings: Mapping,
    *,
    commit: Callable[[], None] = lambda: None,
) -> dict:
    from microcosm.calibrate import calibrate

    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    households = pd.read_parquet(extract_dir / "households.parquet")
    frame, w0 = load_lean_frame(extract_dir / "lean.h5")
    if not np.array_equal(
        frame.table("household")["household_id"].to_numpy(),
        households["household_id"].to_numpy(),
    ):
        raise ValueError("lean frame and household measures are not row-aligned")
    if not np.allclose(w0, households["weight"].to_numpy(np.float64)):
        raise ValueError("lean frame weights differ from the extract weights")
    design = build_design(households, sources)
    records, matrix = design["records"], design["matrix"]
    targets = target_set(records, matrix, len(households))
    print(
        f"design: {matrix.shape[0]} rows ({len(targets)} trained), nnz {matrix.nnz:,}, "
        f"{time.time() - started:.0f}s",
        flush=True,
    )
    weights = w0.copy()
    done = 0
    trajectory = []
    while done < settings["epochs"]:
        batch = min(settings["epoch_batch"], settings["epochs"] - done)
        batch_started = time.time()
        result = calibrate(
            frame,
            targets,
            weight_entity="household",
            method="adam",
            epochs=batch,
            learning_rate=settings["learning_rate"],
            mass="conserve",
            max_weight_ratio=settings["max_weight_ratio"],
            target_loss_cap=settings["target_loss_cap"],
            l2_lambda=settings.get("l2_lambda", 0.0),
            l2_basis=settings.get("l2_basis", "record"),
            seed=settings["seed"],
            warm_start_weights=weights,
        )
        done += batch
        weights = np.asarray(result.weights, dtype=np.float64).copy()
        step = {
            "epochs": done,
            "seconds": round(time.time() - batch_started, 1),
            "loss": float(result.final_loss),
            "within_10pct": float(result.fraction_within_10pct),
            "ess": float(result.effective_sample_size),
        }
        trajectory.append(step)
        print(json.dumps(step), flush=True)
        np.save(out_dir / "weights.npy", weights)
        commit()

    table = fit_table(records, matrix, w0, weights)
    county = households["county_fips_2023"].astype(str).str.zfill(5).to_numpy()
    pep = read_pep_county(sources["pep_county_2024"])
    ess_before = kish_ess(w0, county).rename(columns={"ess": "ess_before"})
    ess_after = kish_ess(weights, county)[["ess"]].rename(columns={"ess": "ess_after"})
    income = households["household_income_approx"].to_numpy(np.float64)
    median_acs = read_acs_county(
        sources["acs5_2024_B19013_county"], ACS_CELLS["median_income"][1]
    )
    per_county = (
        ess_before.join(ess_after, how="outer")
        .reindex(design["counties"])
        .assign(
            pep_population=pep.reindex(design["counties"]),
            grade_after=lambda d: d["ess_after"].fillna(0.0).map(grade),
            median_income_acs=median_acs.reindex(design["counties"]),
            median_income_before=weighted_median_by(income, w0, county),
            median_income_after=weighted_median_by(income, weights, county),
        )
    )
    per_county.index.name = "county_fips"
    for family in (
        "county_households",
        "county_population",
        "county_earnings",
        "county_earning_households",
        "county_soi_wages",
        "county_income",
    ):
        rows = table[table["family"] == family].set_index("geography")
        short = family.removeprefix("county_")
        per_county[f"{short}_target"] = rows["value"]
        per_county[f"{short}_before"] = rows["before"]
        per_county[f"{short}_after"] = rows["after"]
    per_county.to_csv(out_dir / "per_county.csv")
    table.to_csv(out_dir / "target_fit.csv", index=False)

    ratio = weights / w0
    grades = per_county["grade_after"].value_counts().to_dict()
    population_by_grade = (
        per_county.groupby("grade_after")["pep_population"].sum() / pep.sum()
    ).to_dict()
    summary = {
        "settings": dict(settings),
        "households": int(len(households)),
        "rows": int(matrix.shape[0]),
        "trained_rows": int(len(targets)),
        "unsupported_rows": [r["name"] for r in records if r["role"] == "unsupported"],
        "counties": len(design["counties"]),
        "holdout_counties_without_source_value": design["dropped"],
        "counties_without_records": int((per_county["records"].fillna(0) == 0).sum()),
        "trajectory": trajectory,
        "fit": summarize_fit(table, pep),
        "weight_ratio_quantiles": {
            str(q): float(np.quantile(ratio, q))
            for q in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)
        },
        "ess_total_before": float(w0.sum() ** 2 / (w0**2).sum()),
        "ess_total_after": float(weights.sum() ** 2 / (weights**2).sum()),
        "county_ess_after_quantiles": {
            str(q): float(per_county["ess_after"].fillna(0).quantile(q))
            for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "reliability_grades": {
            "thresholds_ess": dict(RELIABILITY_GRADES),
            "counties": {k: int(v) for k, v in grades.items()},
            "population_share": {k: float(v) for k, v in population_by_grade.items()},
        },
        "median_income_vs_acs_b19013": {
            stage: float(
                (
                    per_county[f"median_income_{stage}"]
                    / per_county["median_income_acs"]
                    - 1
                )
                .abs()
                .median()
            )
            for stage in ("before", "after")
        },
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    commit()
    return summary
