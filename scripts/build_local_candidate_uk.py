"""Build the populace UK local-area candidate: stratified pool x 650-constituency stack.

A faithful UK port of the US replication-spine driver
(``populace-local/scripts/build_local_candidate.py``): a stratified
household subsample of the populace UK pool, cloned once per Westminster
constituency (the 650 2024 boundaries), solved with populace.calibrate.
It deliberately does NOT use pe-uk-data's "wide" dense areas x households
clone-and-assign approach — the spine keeps a small donor pool and stacks
it across the area set, exactly like the US flow.

Stages (each cached, resumable):
  subsample : stratified household subsample of the populace UK pool
              (keep ALL households above the total_income percentile,
              uniform sample below) — the analog of create_stratified_cps.
  matrix    : sparse target matrix over the constituency stack. ONE engine
              simulation per COUNTRY (England / Scotland / Wales / Northern
              Ireland) — the UK analog of the US "one simulation per state",
              because devolved policy (Scottish income tax, Welsh rates,
              devolved benefits) differs by country and constituencies within
              a country share it. Household metric columns and per-constituency
              targets are bound against the REAL pe-uk-data matrix machinery
              (calibration/matrix_builder.py:_compute_household_metrics and
              :_load_area_targets, which pull HMRC SPI 3.15 income,
              ONS subnational age, and DWP Stat-Xplore UC).
  solve     : populace.calibrate log-weight Adam (+ optional L0 budget),
              design-weight init tiled across the constituency stack — the
              identical _optimize call the US stage_solve makes.

Column order: constituency_index * n_households + household_index, with
the constituency order = sorted constituency codes (== the row order of
constituencies_2024.csv), matching the US stacked-exporter layout.

Usage:
  .venv/bin/python scripts/build_local_candidate_uk.py \
      --dataset /path/to/populace_uk_2023.h5 \
      --out out/uk-local-v0 [--base-households 12000] [--pilot-constituencies 5]

ENV: the venv at ~/.claude-worktrees/populace-uk-build/.venv (Python 3.14,
policyengine_uk + torch + h5py + scipy + pandas + populace-calibrate/-frame
editable). populace.calibrate must be importable.

PRIVATE DATA: the populace UK pool is enhanced-FRS-derived microdata under
UK Data Service licence. Artifacts stay local. Only aggregates are logged;
never individual rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("populace.local.uk")
# Importing policyengine_uk / policyengine_uk_data reconfigures the root logger
# (it can detach the basicConfig handler), which would silence our stage
# progress. Give this logger its own handler that propagation can't strip.
log.setLevel(logging.INFO)
log.propagate = False
_h = logging.StreamHandler(sys.stderr)
_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log.addHandler(_h)

# The populace UK pool's native survey period is 2023 (the time_period table
# stored in populace_uk_2023.h5 reads b"2023"). We hold to it so the engine
# uprating matches the build that produced the artifact. Overridable via
# --time-period if a re-based pool ships a different vintage.
DEFAULT_TIME_PERIOD = 2023
SEED = 20260614

# UK devolution blocks. Each Westminster constituency belongs to exactly one
# country (constituencies_2024.csv `country` column). country in
# policyengine_uk is DERIVED from `region` (variables/household/demographic/
# country.py: select(region == SCOTLAND -> SCOTLAND, ...)), so to force a
# country block we set the `region` input. England has no devolved income tax,
# so any English region serves for devolution purposes; the three devolved
# countries map to their dedicated region enum value.
#   pe-uk Region enum: variables/household/demographic/geography.py
COUNTRY_TO_REGION = {
    "England": "SOUTH_EAST",  # representative English region; no devolved tax
    "Scotland": "SCOTLAND",
    "Wales": "WALES",
    "Northern Ireland": "NORTHERN_IRELAND",
}

AREA_TYPE = "constituency"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        default=str(
            Path.home()
            / ".claude-worktrees/populace-uk-build/artifacts/populace_uk_2023.h5"
        ),
        help="populace UK pool .h5 (UKSingleYearDataset table format).",
    )
    p.add_argument("--out", required=True)
    p.add_argument(
        "--income-variable",
        default="total_income",
        help="Household-mappable income variable for stratification "
        "(US analog: adjusted_gross_income).",
    )
    p.add_argument("--base-households", type=int, default=12_000)
    p.add_argument("--high-income-percentile", type=float, default=99.0)
    p.add_argument("--time-period", type=int, default=DEFAULT_TIME_PERIOD)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.15)
    p.add_argument(
        "--ratio",
        type=float,
        default=100.0,
        help="Per-column cap as a multiple of the design init weight.",
    )
    p.add_argument(
        "--target-records",
        type=int,
        default=None,
        help="L0 budget (nonzero spine columns). Default: no L0 (dense positive).",
    )
    p.add_argument(
        "--pilot-constituencies",
        type=int,
        default=None,
        help="Limit to the first N constituencies (pilot e2e run). The UK "
        "analog of the US --pilot-states. Whole countries are simulated, so "
        "only the countries spanned by the kept constituencies are run.",
    )
    p.add_argument(
        "--stage", choices=["all", "subsample", "matrix", "solve"], default="all"
    )
    return p.parse_args(argv)


# ----------------------------------------------------------------------
# Dataset IO: the populace UK pool is a UKSingleYearDataset table-format
# H5. The US script reads it with pd.HDFStore, but (a) the UK engine needs a
# UKSingleYearDataset object, not a path string (Microsimulation(dataset=str)
# only accepts HuggingFace URLs), and (b) PyTables/blosc2 is broken on the
# Python 3.14 venv. So we read the entity tables directly with h5py (the
# tables are uncompressed compound datasets at /<entity>/table) and build the
# dataset in memory. This is the one structural divergence from the US IO path.
# ----------------------------------------------------------------------


def _read_entity_table(f, entity: str):
    """Read /<entity>/table (PyTables table-format compound dataset) to a
    DataFrame, decoding byte-string (enum) columns to str. Drops the PyTables
    `index` bookkeeping column."""
    import pandas as pd

    t = f[f"{entity}/table"]
    cols = {}
    for name in t.dtype.names:
        if name == "index":
            continue
        values = t[name][:]
        if values.dtype.kind == "S":
            values = np.char.decode(values, "utf-8")
        cols[name] = values
    return pd.DataFrame(cols)


def _load_pool(dataset_path: str, fiscal_year: int):
    """Load the populace UK pool into a UKSingleYearDataset (in-memory)."""
    import h5py

    from policyengine_uk.data import UKSingleYearDataset

    with h5py.File(dataset_path, "r") as f:
        person = _read_entity_table(f, "person")
        benunit = _read_entity_table(f, "benunit")
        household = _read_entity_table(f, "household")
        period = fiscal_year
        try:
            period = int(f["time_period/table"]["values"][0].decode())
        except Exception:
            pass
    return (
        UKSingleYearDataset(
            person=person,
            benunit=benunit,
            household=household,
            fiscal_year=period,
        ),
        person,
        benunit,
        household,
    )


# ----------------------------------------------------------------------
# Stage 1: stratified subsample of the populace UK pool
# ----------------------------------------------------------------------


def stage_subsample(args, out: Path) -> Path:
    """Stratified household subsample mirroring create_stratified_cps /
    the US stage_subsample: keep ALL households above the income percentile,
    uniform-sample below; design init weight = household_weight x inverse
    sampling rate."""
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    sub_path = out / "base_pool.h5"
    meta_path = out / "base_pool_meta.json"
    if sub_path.exists() and meta_path.exists():
        log.info("subsample: cached at %s", sub_path)
        return sub_path

    ds, person, benunit, household = _load_pool(args.dataset, args.time_period)
    log.info(
        "subsample: pool loaded (%d person / %d benunit / %d household, period %s)",
        len(person), len(benunit), len(household), ds.time_period,
    )

    sim = Microsimulation(dataset=ds)
    sim.default_calculation_period = ds.time_period
    income = sim.calculate(
        args.income_variable, args.time_period, map_to="household"
    ).values
    hh_ids = sim.calculate("household_id", map_to="household").values
    n = len(hh_ids)

    cut = np.percentile(income, args.high_income_percentile)
    top = income >= cut
    n_top = int(top.sum())
    n_rest_target = max(args.base_households - n_top, 0)
    rest_idx = np.flatnonzero(~top)
    rng = np.random.default_rng(args.seed)
    keep_rest = rng.choice(
        rest_idx, size=min(n_rest_target, len(rest_idx)), replace=False
    )
    keep = np.zeros(n, dtype=bool)
    keep[top] = True
    keep[keep_rest] = True
    log.info(
        "subsample: %d households (%d top-%s%% by %s + %d sampled of %d)",
        int(keep.sum()), n_top, args.high_income_percentile,
        args.income_variable, len(keep_rest), n,
    )

    # Membership preservation via the pool's own foreign keys, so group-entity
    # integrity is preserved by construction (engine household ordering matches
    # the stored table order). UK entity hierarchy: person -> benunit, household.
    kept_hh_ids = set(np.asarray(hh_ids)[keep].tolist())
    hh_keep = household["household_id"].isin(kept_hh_ids).to_numpy()
    person_keep = person["person_household_id"].isin(kept_hh_ids).to_numpy()
    kept_person = person.loc[person_keep]
    benunit_ids = set(kept_person["person_benunit_id"].tolist())
    benunit_keep = benunit["benunit_id"].isin(benunit_ids).to_numpy()

    household_s = household.loc[hh_keep].reset_index(drop=True)
    person_s = kept_person.reset_index(drop=True)
    benunit_s = benunit.loc[benunit_keep].reset_index(drop=True)
    lengths = {
        "/person": len(person_s),
        "/benunit": len(benunit_s),
        "/household": len(household_s),
    }
    log.info("subsample: kept entity counts %s", lengths)

    # Design-weight init: top stratum kept at full inverse rate (1.0); the
    # uniform-sampled rest scaled up by 1/sampling_rate so the subsample's
    # design mass tracks the pool's. (US stage_subsample logic, UK columns.)
    inv_rate_rest = len(rest_idx) / max(len(keep_rest), 1)
    top_hh_ids = set(np.asarray(hh_ids)[top].tolist())
    is_top_row = household_s["household_id"].isin(top_hh_ids).to_numpy()
    hh_weight = household_s["household_weight"].to_numpy(dtype=float)
    base_init = hh_weight * np.where(is_top_row, 1.0, inv_rate_rest)
    np.save(out / "base_init_weights.npy", base_init)

    # Persist the subsampled pool as a UKSingleYearDataset H5 so the matrix
    # stage reloads exactly what we kept (its own save path; PyTables on the
    # write side works because UKSingleYearDataset.save uses pd.HDFStore put).
    UKSingleYearDataset(
        person=person_s,
        benunit=benunit_s,
        household=household_s,
        fiscal_year=int(ds.time_period),
    ).save(str(sub_path))

    meta = {
        "source": str(args.dataset),
        "income_variable": args.income_variable,
        "time_period": int(ds.time_period),
        "n_households": int(keep.sum()),
        "n_top": n_top,
        "income_cut": float(cut),
        "seed": args.seed,
        "entity_counts": lengths,
        "rest_inverse_rate": float(inv_rate_rest),
        "base_init_weight_sum": float(base_init.sum()),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("subsample: written %s (init mass %.3e)", sub_path, base_init.sum())
    return sub_path


# ----------------------------------------------------------------------
# Stage 2: constituency-stacked sparse matrix (one simulation per country)
# ----------------------------------------------------------------------


def stage_matrix(args, out: Path, base_h5: Path):
    """Assemble (n_targets x n_households*n_constituencies) CSR.

    UK target surface (the analog of the US UnifiedMatrixBuilder querying
    policy_data.db): pe-uk-data's calibration/matrix_builder.py provides
      - _compute_household_metrics(sim, "constituency") -> the 17 household
        metric columns (HMRC SPI employment/self-employment income amount+count,
        ONS age bands 0_10..70_80, DWP UC households + UC-by-children), and
      - _load_area_targets("constituency", area_codes_df, dataset, sim) -> a
        650 x 17 target table, ONE row per (constituency, metric), with the
        HMRC national-consistency adjustment, ONS population scaling, and the
        2010->2024 boundary mapping already applied.
    The two share identical column order (verified), so target column j is
    metric column j broadcast into each constituency's column block.

    UK vs US row structure: there are NO separate national/region rows in the
    constituency surface — the national totals are folded INTO each
    constituency's target inside _load_area_targets (see matrix_builder.py
    national_target/adjustment logic). Every row is therefore a constituency
    row that hits only that constituency's column block — the UK analog of the
    US "district rows", with the US "national"/"state" broadcast rows absent.
    """
    from scipy import sparse

    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset
    from policyengine_uk_data.calibration.matrix_builder import (
        _compute_household_metrics,
        _load_area_codes,
        _load_area_targets,
    )

    matrix_path = out / "X_stacked.npz"
    targets_path = out / "targets.parquet"
    if matrix_path.exists() and targets_path.exists():
        log.info("matrix: cached at %s", matrix_path)
        return

    import pandas as pd

    # Area set: 650 Westminster constituencies, sorted-code order == CSV order.
    area_codes_df = _load_area_codes(AREA_TYPE).copy()
    # constituencies_2024.csv carries an authoritative `country` column; use it
    # (fall back to the ONS code prefix only if absent).
    if "country" in area_codes_df.columns:
        area_codes_df["__country"] = area_codes_df["country"]
    else:
        area_codes_df["__country"] = area_codes_df["code"].map(_country_of_code)
    area_codes_df = area_codes_df.sort_values("code").reset_index(drop=True)
    if args.pilot_constituencies:
        area_codes_df = area_codes_df.iloc[: args.pilot_constituencies].copy()
        log.info(
            "matrix: PILOT limited to first %d constituencies (%s)",
            len(area_codes_df),
            dict(area_codes_df["__country"].value_counts()),
        )
    codes = area_codes_df["code"].tolist()
    code_to_country = dict(zip(codes, area_codes_df["__country"].tolist()))
    n_areas = len(codes)

    # Base pool (the subsample). One sim per country sets the `region` input to
    # force devolution; metric columns are recomputed per country.
    ds = UKSingleYearDataset(file_path=str(base_h5))
    sim0 = Microsimulation(dataset=ds)
    sim0.default_calculation_period = ds.time_period
    n_hh = len(sim0.calculate("household_id", map_to="household").values)
    n_total = n_hh * n_areas
    del sim0

    # Targets: the FULL 650-row x 17-col surface (one sim suffices for targets;
    # they do not depend on the per-country region we assign the pool — they are
    # external HMRC/ONS/DWP values). Compute against the base pool once, then
    # restrict to the kept constituencies.
    sim_t = Microsimulation(dataset=ds)
    sim_t.default_calculation_period = ds.time_period
    full_targets = _load_area_targets(AREA_TYPE, _load_area_codes(AREA_TYPE), ds, sim_t)
    metric_columns = list(full_targets.columns)
    del sim_t

    # full_targets is in _load_area_codes (CSV) order; map to our sorted-code
    # area order via the CSV code column.
    csv_codes = _load_area_codes(AREA_TYPE)["code"].tolist()
    csv_pos = {c: i for i, c in enumerate(csv_codes)}
    target_rows = []  # one dict per (constituency, metric) target row
    for area_i, code in enumerate(codes):
        src = csv_pos[code]
        for col_j, metric in enumerate(metric_columns):
            value = float(full_targets.iloc[src, col_j])
            target_rows.append(
                {
                    "constituency": code,
                    "country": code_to_country[code],
                    "metric": metric,
                    "metric_index": col_j,
                    "area_index": area_i,
                    "value": value,
                }
            )
    # Explicit RangeIndex: the row's position IS its row index into the CSR.
    targets_df = pd.DataFrame(target_rows).reset_index(drop=True)
    targets_df["target_index"] = np.arange(len(targets_df), dtype=np.int64)
    n_targets = len(targets_df)
    log.info(
        "matrix: %d targets (%d constituencies x %d metrics) x %d cols "
        "(%d hh x %d constituencies)",
        n_targets, n_areas, len(metric_columns), n_total, n_hh, n_areas,
    )

    countries = sorted(area_codes_df["__country"].unique())
    cache_dir = out / "country_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for c_i, country in enumerate(countries):
        cpath = cache_dir / f"country_{country.replace(' ', '_')}.npz"
        if cpath.exists():
            log.info(
                "matrix: country %s cached (%d/%d)", country, c_i + 1, len(countries)
            )
            continue

        country_codes = [c for c in codes if code_to_country[c] == country]

        # One simulation per country: force the country's region so devolved
        # policy fires for the whole block. TODO(uk-local): if a finer
        # within-country geography becomes a calibration target (e.g. BRMA for
        # housing benefit, or local council tax bands), set those per-constituency
        # inputs here the way the US exporter re-sets state_fips/in_nyc/county_fips
        # per area — see populace-local/scripts/build_local_candidate.py:360-367.
        sim = Microsimulation(dataset=ds)
        sim.default_calculation_period = ds.time_period
        region_value = COUNTRY_TO_REGION[country]
        sim.set_input(
            "region", args.time_period, np.array([region_value] * n_hh)
        )

        metrics = _compute_household_metrics(sim, AREA_TYPE)
        # Defensive: column order must match the target surface exactly.
        assert list(metrics.columns) == metric_columns, (
            "metric/target column mismatch: "
            f"{list(metrics.columns)} vs {metric_columns}"
        )
        metric_values = metrics.to_numpy(dtype=np.float32)

        rows_l, cols_l, vals_l = [], [], []
        country_target = targets_df[targets_df["country"] == country]
        # Constituency rows hit only that constituency's column block (the UK
        # analog of the US district rows). Cache each metric's nonzero pattern
        # since every constituency in this country reuses the same column.
        nz_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for row in country_target.itertuples():
            mi = int(row.metric_index)
            if mi not in nz_cache:
                col = metric_values[:, mi]
                nz = np.flatnonzero(col)
                nz_cache[mi] = (nz, col[nz].astype(np.float32))
            nz, vals = nz_cache[mi]
            if not len(nz):
                continue
            col_start = int(row.area_index) * n_hh
            rows_l.append(np.full(len(nz), int(row.target_index), dtype=np.int32))
            cols_l.append((col_start + nz).astype(np.int64))
            vals_l.append(vals)

        np.savez_compressed(
            cpath,
            rows=np.concatenate(rows_l) if rows_l else np.array([], dtype=np.int32),
            cols=np.concatenate(cols_l) if cols_l else np.array([], dtype=np.int64),
            vals=np.concatenate(vals_l) if vals_l else np.array([], dtype=np.float32),
        )
        del sim
        log.info(
            "matrix: country %s done (%d/%d, %d constituencies, %.1f min elapsed)",
            country, c_i + 1, len(countries), len(country_codes),
            (time.time() - t0) / 60,
        )

    all_r, all_c, all_v = [], [], []
    for country in countries:
        cpath = cache_dir / f"country_{country.replace(' ', '_')}.npz"
        d = np.load(cpath)
        all_r.append(d["rows"])
        all_c.append(d["cols"])
        all_v.append(d["vals"])
    rows = np.concatenate(all_r)
    cols = np.concatenate(all_c)
    vals = np.concatenate(all_v)
    log.info("matrix: assembling CSR from %d nnz", len(vals))
    X = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_targets, n_total), dtype=np.float32
    )
    sparse.save_npz(matrix_path, X)
    targets_df.to_parquet(targets_path)
    (out / "target_names.json").write_text(
        json.dumps(
            [f"{r.constituency}/{r.metric}" for r in targets_df.itertuples()]
        )
    )
    (out / "stack_meta.json").write_text(
        json.dumps(
            {
                "constituencies": codes,
                "n_households": n_hh,
                "n_constituencies": n_areas,
                "metric_columns": metric_columns,
                "countries": countries,
            }
        )
    )
    log.info("matrix: %s nnz=%d saved", X.shape, X.nnz)


def _country_of_code(code: str) -> str:
    """Westminster ONS code prefix -> country. E14/E -> England, W -> Wales,
    S -> Scotland, N -> Northern Ireland (constituencies_2024.csv `country`
    column is authoritative and used directly; this is the fallback mapping)."""
    return {
        "E": "England",
        "W": "Wales",
        "S": "Scotland",
        "N": "Northern Ireland",
    }.get(code[0], "England")


# ----------------------------------------------------------------------
# Stage 3: solve (the identical populace.calibrate _optimize call the US makes)
# ----------------------------------------------------------------------


def stage_solve(args, out: Path):
    import pandas as pd
    import torch
    from scipy import sparse

    from populace.calibrate.solve import _optimize, _torch_constraint_matrix

    w_path = out / "weights.npy"
    if w_path.exists():
        log.info("solve: cached at %s", w_path)
        return

    X = sparse.load_npz(out / "X_stacked.npz")
    targets_df = pd.read_parquet(out / "targets.parquet")
    names = np.array(json.loads((out / "target_names.json").read_text()))
    y = targets_df["value"].values.astype(np.float64)

    row_sums = np.asarray(np.abs(X).sum(axis=1)).ravel()
    achievable = row_sums != 0
    log.info("solve: %d/%d achievable targets", int(achievable.sum()), len(y))
    Xa = X[achievable]
    ya = y[achievable]

    n_total = X.shape[1]
    meta = json.loads((out / "stack_meta.json").read_text())
    n_areas = int(meta["n_constituencies"])
    base_init = np.load(out / "base_init_weights.npy")
    # Design-weight init: each household's pool design mass (inflated by its
    # stratum's inverse sampling rate) split evenly across the constituency
    # stack, so tail records start near their tiny per-constituency weight
    # rather than many log-units away. The cap is a per-column ratio on this
    # init. (US stage_solve init logic, UK stack width.)
    init = np.tile(base_init / n_areas, n_areas)
    init = np.maximum(init, 1e-4)
    A = _torch_constraint_matrix(Xa)
    t0 = time.time()
    w, traj = _optimize(
        A,
        torch.tensor(ya, dtype=torch.float32),
        init,
        epochs=args.epochs,
        learning_rate=args.lr,
        conserve_mass=False,
        max_weight_ratio=args.ratio,
        l0_lambda=0.0,
        target_records=args.target_records,
        init_mean=0.999,
        temperature=0.25,
    )
    log.info(
        "solve: %.1f min, final loss %.5f", (time.time() - t0) / 60, float(traj[-1])
    )

    np.save(w_path, w)
    est = Xa @ w
    rel = np.abs(est - ya) / (np.abs(ya) + 1)
    pd.DataFrame(
        {"name": names[achievable], "target": ya, "estimate": est, "rel_err": rel}
    ).to_csv(out / "solve_diagnostics.csv", index=False)
    nz = w > 0
    summary = {
        "loss_final": float(traj[-1]),
        "n_targets_achievable": int(achievable.sum()),
        "n_cols": int(n_total),
        "n_nonzero": int(nz.sum()),
        "weight_sum": float(w.sum()),
        "weight_max": float(w.max()),
        "within_10pct": float((rel < 0.10).mean()),
        "within_25pct": float((rel < 0.25).mean()),
        "epochs": args.epochs,
        "lr": args.lr,
        "ratio": args.ratio,
        "target_records": args.target_records,
    }
    (out / "solve_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("solve: %s", json.dumps(summary))


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2)
    )
    if args.stage in ("all", "subsample"):
        base = stage_subsample(args, out)
    else:
        base = out / "base_pool.h5"
    if args.stage in ("all", "matrix"):
        stage_matrix(args, out, base)
    if args.stage in ("all", "solve"):
        stage_solve(args, out)
    log.info("DONE")


if __name__ == "__main__":
    main()
