"""Build the populace local-area candidate: stratified pool x 436-CD stack.

Replicates the incumbent local-area structure exactly — a stratified
~12k-household base cloned once per congressional district (5.23M spine
columns) — but on the populace national pool instead of the stratified
extended CPS, with populace-calibrate as the solver.

Stages (each cached, resumable):
  subsample : stratified household subsample of the populace pool
              (keep all top-1% AGI households, uniform sample below)
  matrix    : sparse target matrix over the CD stack. One engine
              simulation per STATE (CDs within a state share policy and,
              in v0, takeup draws — the incumbent's newer flow draws per
              block; refining to per-CD draws is a later knob), assembled
              into rows for national / state / district targets queried
              from policy_data.db via the incumbent's own
              UnifiedMatrixBuilder (identical target set + uprating).
  solve     : populace.calibrate log-weight Adam (+ optional L0 budget).

Usage:
  .venv/bin/python scripts/build_local_candidate.py \
      --dataset inputs/populace_us_2024.h5 \
      --db inputs/calibration/policy_data.db \
      --out out/local-v0 [--base-households 11999] [--pilot-states 2]
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
log = logging.getLogger("populace.local")

TIME_PERIOD = 2024
SEED = 20260612

SIMPLE_TAKEUP_VARS = [
    ("takes_up_snap_if_eligible", "spm_unit", "snap"),
    ("takes_up_aca_if_eligible", "tax_unit", "aca"),
    ("takes_up_dc_ptc", "tax_unit", "dc_ptc"),
    ("takes_up_head_start_if_eligible", "person", "head_start"),
    ("takes_up_early_head_start_if_eligible", "person", "early_head_start"),
    ("takes_up_ssi_if_eligible", "person", "ssi"),
    ("would_file_taxes_voluntarily", "tax_unit", "voluntary_filing"),
    ("takes_up_medicaid_if_eligible", "person", "medicaid"),
    ("takes_up_tanf_if_eligible", "spm_unit", "tanf"),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base-households", type=int, default=11_999)
    p.add_argument("--high-income-percentile", type=float, default=99.0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.15)
    p.add_argument(
        "--max-weight",
        type=float,
        default=2_500.0,
        help="Hard per-column weight cap (incumbent local max is 2,405).",
    )
    p.add_argument(
        "--target-records",
        type=int,
        default=None,
        help="L0 budget (nonzero spine columns). Default: no L0 (dense positive).",
    )
    p.add_argument(
        "--skip-national-rows",
        action="store_true",
        help="Drop national target rows (each spans all 5.2M columns; "
        "state rows already anchor mass).",
    )
    p.add_argument(
        "--pilot-states",
        type=int,
        default=None,
        help="Limit to the first N states (pilot e2e run).",
    )
    p.add_argument(
        "--stage", choices=["all", "subsample", "matrix", "solve"], default="all"
    )
    return p.parse_args(argv)


# ----------------------------------------------------------------------
# Stage 1: stratified subsample of the populace pool
# ----------------------------------------------------------------------


def stage_subsample(args, out: Path) -> Path:
    """Stratified household subsample mirroring create_stratified_cps:
    keep ALL households above the AGI percentile, uniform sample below."""
    import h5py

    sub_path = out / "base_pool.h5"
    meta_path = out / "base_pool_meta.json"
    if sub_path.exists() and meta_path.exists():
        log.info("subsample: cached at %s", sub_path)
        return sub_path

    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=args.dataset)
    agi = sim.calculate(
        "adjusted_gross_income", TIME_PERIOD, map_to="household"
    ).values
    hh_ids = sim.calculate("household_id", map_to="household").values
    n = len(hh_ids)
    cut = np.percentile(agi, args.high_income_percentile)
    top = agi >= cut
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
        "subsample: %d households (%d top-%s%% + %d sampled of %d)",
        int(keep.sum()), n_top, args.high_income_percentile, len(keep_rest), n,
    )

    import pandas as pd

    # Membership comes from the bundle's own foreign keys so group-entity
    # integrity is preserved by construction (engine ordering of household
    # ids matches the stored table order).
    with pd.HDFStore(args.dataset, "r") as src:
        tables = {key: src[key] for key in src.keys()}
    hh_table = tables["/household"]
    kept_hh_ids = set(np.asarray(hh_ids)[keep].tolist())
    hh_keep = hh_table["household_id"].isin(kept_hh_ids).to_numpy()
    person = tables["/person"]
    person_keep = person["person_household_id"].isin(kept_hh_ids).to_numpy()
    kept_person = person.loc[person_keep]

    entity_masks = {"/household": hh_keep, "/person": person_keep}
    for key, fk in [
        ("/tax_unit", "person_tax_unit_id"),
        ("/spm_unit", "person_spm_unit_id"),
        ("/family", "person_family_id"),
        ("/marital_unit", "person_marital_unit_id"),
    ]:
        ent = tables[key]
        id_col = key[1:] + "_id"
        member_ids = set(kept_person[fk].tolist())
        entity_masks[key] = ent[id_col].isin(member_ids).to_numpy()

    lengths = {k: int(m.sum()) for k, m in entity_masks.items()}
    log.info("subsample: kept entity counts %s", lengths)

    with pd.HDFStore(sub_path, "w", complevel=4, complib="zlib") as dst:
        for key, df in tables.items():
            mask = entity_masks.get(key)
            out_df = df if mask is None else df.loc[mask].reset_index(drop=True)
            dst.put(key, out_df)

    meta = {
        "source": str(args.dataset),
        "n_households": int(keep.sum()),
        "n_top": n_top,
        "agi_cut": float(cut),
        "seed": args.seed,
        "entity_counts": lengths,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("subsample: written %s", sub_path)
    return sub_path


# ----------------------------------------------------------------------
# Stage 2: CD-stacked sparse matrix (one simulation per state)
# ----------------------------------------------------------------------


def stage_matrix(args, out: Path, base_h5: Path):
    """Assemble (n_targets x n_households*n_cds) CSR.

    Column order: cd_index * n_households + household_index, with
    cds_to_calibrate (sorted CD GEOIDs from the db) defining cd_index —
    the layout the incumbent's stacked exporter expects.
    """
    from scipy import sparse

    from policyengine_us import Microsimulation
    from policyengine_us_data.calibration.unified_matrix_builder import (
        NEUTRALIZE_VARIABLE_REFORM_ID,
        UnifiedMatrixBuilder,
        _GEO_VARS,
    )
    from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
        get_all_cds_from_database,
        get_calculated_variables,
    )
    from policyengine_us_data.parameters import load_take_up_rate
    from policyengine_us_data.utils.randomness import seeded_rng

    matrix_path = out / "X_stacked.npz"
    targets_path = out / "targets.parquet"
    if matrix_path.exists() and targets_path.exists():
        log.info("matrix: cached at %s", matrix_path)
        return

    db_uri = f"sqlite:///{args.db}"
    cds = get_all_cds_from_database(db_uri)
    # CD GEOIDs are integer strings: state_fips * 100 + district.
    def cd_state(cd: str) -> str:
        return f"{int(cd) // 100:02d}"

    if args.pilot_states:
        states_allowed = sorted({cd_state(cd) for cd in cds})[: args.pilot_states]
        cds = [cd for cd in cds if cd_state(cd) in states_allowed]
        log.info(
            "matrix: PILOT limited to states %s (%d CDs)", states_allowed, len(cds)
        )
    cd_index = {cd: i for i, cd in enumerate(cds)}
    n_cds = len(cds)

    builder = UnifiedMatrixBuilder(
        db_uri=db_uri, time_period=TIME_PERIOD, dataset_path=str(base_h5)
    )

    sim0 = Microsimulation(dataset=str(base_h5))
    n_hh = len(sim0.calculate("household_id", map_to="household").values)
    n_total = n_hh * n_cds
    params = sim0.tax_benefit_system.parameters
    del sim0

    # Targets: identical query + uprating to the incumbent's default run.
    targets_df = builder._query_targets({})
    factors = builder._calculate_uprating_factors(params)
    targets_df["original_value"] = targets_df["value"].copy()
    targets_df["uprating_factor"] = targets_df.apply(
        lambda row: builder._get_uprating_info(
            row["variable"], row["period"], factors
        )[0],
        axis=1,
    )
    targets_df["value"] = targets_df["original_value"] * targets_df["uprating_factor"]

    if args.skip_national_rows:
        targets_df = targets_df[targets_df["geo_level"] != "national"]
    if args.pilot_states:
        states_set = {cd_state(cd) for cd in cds}
        keep = (
            (targets_df["geo_level"] == "national")
            | (
                (targets_df["geo_level"] == "state")
                & targets_df["geographic_id"]
                .astype(str)
                .str.zfill(2)
                .isin(states_set)
            )
            | (
                (targets_df["geo_level"] == "district")
                & targets_df["geographic_id"].astype(str).isin(set(cds))
            )
        )
        targets_df = targets_df[keep]
    targets_df = targets_df.sort_values(
        ["geo_level", "variable", "geographic_id"]
    ).reset_index(drop=True)
    n_targets = len(targets_df)
    log.info(
        "matrix: %d targets x %d cols (%d hh x %d cds)",
        n_targets, n_total, n_hh, n_cds,
    )

    constraint_cache: dict[int, list] = {}
    rows_meta = []
    for _, row in targets_df.iterrows():
        sid = int(row["stratum_id"])
        if sid not in constraint_cache:
            constraint_cache[sid] = builder._get_stratum_constraints(sid)
        constraints = constraint_cache[sid]
        non_geo = [c for c in constraints if c["variable"] not in _GEO_VARS]
        reform_id = row.get("reform_id", 0)
        if reform_id is None or (isinstance(reform_id, float) and np.isnan(reform_id)):
            reform_id = 0
        rows_meta.append(
            {
                "variable": str(row["variable"]),
                "geo_level": row["geo_level"],
                "geo_id": str(row["geographic_id"]),
                "non_geo": non_geo,
                "reform_id": int(reform_id),
                "name": builder._make_target_name(
                    str(row["variable"]), constraints, int(reform_id)
                ),
            }
        )
    value_vars = sorted(
        {
            m["variable"]
            for m in rows_meta
            if m["reform_id"] == 0 and not m["variable"].endswith("_count")
        }
    )
    reform_vars = sorted(
        {
            m["variable"]
            for m in rows_meta
            if m["reform_id"] == NEUTRALIZE_VARIABLE_REFORM_ID
        }
    )

    national_rows = [
        i for i, m in enumerate(rows_meta) if m["geo_level"] == "national"
    ]
    state_rows: dict[str, list[int]] = {}
    district_rows: dict[str, list[int]] = {}
    for i, m in enumerate(rows_meta):
        if m["geo_level"] == "state":
            state_rows.setdefault(m["geo_id"].zfill(2), []).append(i)
        elif m["geo_level"] == "district":
            district_rows.setdefault(m["geo_id"], []).append(i)

    states = sorted({cd_state(cd) for cd in cds})
    cache_dir = out / "state_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for s_i, state in enumerate(states):
        state_cds = [cd for cd in cds if cd_state(cd) == state]
        spath = cache_dir / f"state_{state}.npz"
        if spath.exists():
            log.info("matrix: state %s cached (%d/%d)", state, s_i + 1, len(states))
            continue

        sim = Microsimulation(dataset=str(base_h5))
        state_fips_arr = np.full(n_hh, int(state), dtype=np.int32)
        sim.set_input("state_fips", TIME_PERIOD, state_fips_arr)
        # Stored geography-scoped inputs from the national build must not
        # survive state reassignment (e.g. in_nyc fires NYC credits under
        # any state). The exporter re-sets them per assigned area.
        sim.set_input("in_nyc", TIME_PERIOD, np.zeros(n_hh, dtype=bool))
        sim.set_input("county_fips", TIME_PERIOD, np.zeros(n_hh, dtype=np.int32))
        # v0: takeup drawn per state (incumbent's newer flow draws per
        # block; per-CD draws are a later refinement).
        for var_name, entity, rate_key in SIMPLE_TAKEUP_VARS:
            rate_or_dict = load_take_up_rate(rate_key, TIME_PERIOD)
            n_entities = len(sim.calculate(f"{entity}_id", map_to=entity).values)
            rng = seeded_rng(var_name, salt=f"state_{state}")
            draws = rng.random(n_entities)
            if isinstance(rate_or_dict, dict):
                rate = rate_or_dict.get(state, rate_or_dict.get(int(state), 0.8))
            else:
                rate = rate_or_dict
            sim.set_input(var_name, TIME_PERIOD, draws < rate)
        for var in get_calculated_variables(sim):
            sim.delete_arrays(var)

        builder._entity_rel_cache = None
        var_values: dict[str, np.ndarray] = {}
        for var in value_vars:
            try:
                var_values[var] = sim.calculate(
                    var, TIME_PERIOD, map_to="household"
                ).values.astype(np.float32)
            except Exception as exc:
                log.warning("state %s: cannot calculate %s: %s", state, var, exc)

        reform_values: dict[str, np.ndarray] = {}
        if reform_vars:
            baseline_it = sim.calculate(
                "income_tax", TIME_PERIOD, map_to="household"
            ).values.astype(np.float32)
            for var in reform_vars:
                try:
                    reform_values[var] = (
                        builder._calculate_neutralized_expenditure_values(
                            state_fips_arr, var, baseline_it
                        )
                    )
                except Exception as exc:
                    log.warning("state %s: cannot neutralize %s: %s", state, var, exc)

        mask_cache: dict[tuple, np.ndarray] = {}
        count_cache: dict[tuple, np.ndarray] = {}

        def row_values(m) -> np.ndarray | None:
            ckey = tuple(
                sorted(
                    (c["variable"], c["operation"], c["value"]) for c in m["non_geo"]
                )
            )
            if m["reform_id"] == NEUTRALIZE_VARIABLE_REFORM_ID:
                if m["variable"] not in reform_values:
                    return None
                if ckey not in mask_cache:
                    mask_cache[ckey] = builder._evaluate_constraints_entity_aware(
                        sim, m["non_geo"], n_hh
                    )
                return reform_values[m["variable"]] * mask_cache[ckey]
            if m["reform_id"] > 0:
                return None
            if m["variable"].endswith("_count"):
                vkey = (m["variable"], ckey)
                if vkey not in count_cache:
                    count_cache[vkey] = builder._calculate_target_values(
                        sim, m["variable"], m["non_geo"], n_hh
                    )
                return count_cache[vkey]
            if m["variable"] not in var_values:
                return None
            if ckey not in mask_cache:
                mask_cache[ckey] = builder._evaluate_constraints_entity_aware(
                    sim, m["non_geo"], n_hh
                )
            return var_values[m["variable"]] * mask_cache[ckey]

        rows_l, cols_l, vals_l = [], [], []

        # National + state rows: identical household values in every CD of
        # this state; emit per CD column block.
        broadcast_rows = national_rows + state_rows.get(state, [])
        for row_idx in broadcast_rows:
            values = row_values(rows_meta[row_idx])
            if values is None:
                continue
            nz = np.flatnonzero(values)
            if not len(nz):
                continue
            v = values[nz].astype(np.float32)
            for cd in state_cds:
                col_start = cd_index[cd] * n_hh
                rows_l.append(np.full(len(nz), row_idx, dtype=np.int32))
                cols_l.append((col_start + nz).astype(np.int64))
                vals_l.append(v)

        # District rows: only that CD's column block.
        for cd in state_cds:
            col_start = cd_index[cd] * n_hh
            for row_idx in district_rows.get(cd, []):
                values = row_values(rows_meta[row_idx])
                if values is None:
                    continue
                nz = np.flatnonzero(values)
                if not len(nz):
                    continue
                rows_l.append(np.full(len(nz), row_idx, dtype=np.int32))
                cols_l.append((col_start + nz).astype(np.int64))
                vals_l.append(values[nz].astype(np.float32))

        np.savez_compressed(
            spath,
            rows=np.concatenate(rows_l) if rows_l else np.array([], dtype=np.int32),
            cols=np.concatenate(cols_l) if cols_l else np.array([], dtype=np.int64),
            vals=np.concatenate(vals_l) if vals_l else np.array([], dtype=np.float32),
        )
        del sim
        log.info(
            "matrix: state %s done (%d/%d, %d CDs, %.1f min elapsed)",
            state, s_i + 1, len(states), len(state_cds), (time.time() - t0) / 60,
        )

    all_r, all_c, all_v = [], [], []
    for state in states:
        d = np.load(cache_dir / f"state_{state}.npz")
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
        json.dumps([m["name"] for m in rows_meta])
    )
    (out / "stack_meta.json").write_text(
        json.dumps({"cds": cds, "n_households": n_hh, "n_cds": n_cds})
    )
    log.info("matrix: %s nnz=%d saved", X.shape, X.nnz)


# ----------------------------------------------------------------------
# Stage 3: solve
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
    # Flat init at the incumbent's nonzero-mean weight scale; the cap then
    # gives 50x headroom — the same ratio discipline as the national build.
    init = np.full(n_total, 50.0)
    A = _torch_constraint_matrix(Xa)
    t0 = time.time()
    w, traj = _optimize(
        A,
        torch.tensor(ya, dtype=torch.float32),
        init,
        epochs=args.epochs,
        learning_rate=args.lr,
        conserve_mass=False,
        max_weight_ratio=args.max_weight / float(init[0]),
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
        "max_weight": args.max_weight,
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
