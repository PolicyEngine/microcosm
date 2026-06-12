"""Per-area benchmark: populace local candidate vs incumbent district h5s.

For each congressional district, both sides are scored AS SHIPPED under a
matched-household symmetric refit:

  incumbent : the published districts/{CD}.h5 (usdata 1.115.5) — household
              values computed by simulating that artifact with its stored
              inputs (state, takeup draws baked in at publish time).
  candidate : the corresponding column block of our stacked matrix
              (out/<run>/X_stacked.npz), restricted to nonzero-weight
              columns from our solve — the households we would ship.

Per rotation seed, the district's target rows are split train/holdout,
both sides are refit from flat initial weights by the same solver
(populace.calibrate Adam, identical epochs/lr/cap), and train + holdout
losses are recorded. The verdict requires winning train AND holdout in
aggregate, with per-area win counts disclosed.

Usage:
  .venv/bin/python scripts/score_local.py --run out/pilot \
      --districts-dir inputs/districts [--areas AL-01,AL-02] \
      [--rotations 3] [--epochs 256]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("populace.local.score")

TIME_PERIOD = 2024
HOLDOUT_SHARE = 0.2
BASE_ROTATION_SEED = 20260529


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Build output dir (out/pilot)")
    p.add_argument("--districts-dir", required=True)
    p.add_argument("--areas", default=None, help="Comma-separated CD codes (AL-01)")
    p.add_argument("--rotations", type=int, default=3)
    p.add_argument("--epochs", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.15)
    p.add_argument("--max-weight", type=float, default=2_500.0)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def cd_geoid_to_code(geoid: str, fips_to_code: dict) -> str:
    """'101' -> 'AL-01' (GEOID = state_fips * 100 + district)."""
    state = fips_to_code[int(geoid) // 100]
    return f"{state}-{int(geoid) % 100:02d}"


def loss_fn(est: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(((est - y) / (np.abs(y) + 1.0)) ** 2))


def matched_subsample(
    M: np.ndarray, w0: np.ndarray, n_target: int, seed: int
) -> np.ndarray:
    """Mass-preserving seeded subsample of columns: pick n_target columns
    with probability proportional to w0, return their indices."""
    n = M.shape[1]
    if n <= n_target:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    p = w0 / w0.sum() if w0.sum() > 0 else np.full(n, 1.0 / n)
    return np.sort(rng.choice(n, size=n_target, replace=False, p=p))


def refit_loss(
    M: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    hold_idx: np.ndarray,
    *,
    epochs: int,
    lr: float,
    max_weight: float,
    init_total: float,
) -> tuple[float, float]:
    import torch

    from populace.calibrate.solve import _optimize, _torch_constraint_matrix

    from scipy import sparse

    n = M.shape[1]
    init = np.full(n, max(init_total / n, 1.0))
    A = _torch_constraint_matrix(sparse.csr_matrix(M[train_idx]))
    w, _ = _optimize(
        A,
        torch.tensor(y[train_idx], dtype=torch.float32),
        init,
        epochs=epochs,
        learning_rate=lr,
        conserve_mass=False,
        max_weight_ratio=max_weight / float(init[0]),
        l0_lambda=0.0,
        target_records=None,
        init_mean=0.999,
        temperature=0.25,
    )
    train = loss_fn(M[train_idx] @ w, y[train_idx])
    hold = loss_fn(M[hold_idx] @ w, y[hold_idx]) if len(hold_idx) else float("nan")
    return train, hold


def incumbent_matrix(h5_path: Path, rows_meta: list, builder) -> np.ndarray:
    """(n_rows x n_households) value matrix for the incumbent artifact,
    simulated with its stored inputs."""
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=str(h5_path))
    builder._entity_rel_cache = None
    n_hh = len(sim.calculate("household_id", map_to="household").values)

    var_cache: dict[str, np.ndarray] = {}
    mask_cache: dict[tuple, np.ndarray] = {}
    count_cache: dict[tuple, np.ndarray] = {}
    out = np.zeros((len(rows_meta), n_hh), dtype=np.float64)
    for i, m in enumerate(rows_meta):
        ckey = tuple(
            sorted((c["variable"], c["operation"], c["value"]) for c in m["non_geo"])
        )
        if m["variable"].endswith("_count"):
            vkey = (m["variable"], ckey)
            if vkey not in count_cache:
                count_cache[vkey] = builder._calculate_target_values(
                    sim, m["variable"], m["non_geo"], n_hh
                )
            out[i] = count_cache[vkey]
        else:
            if m["variable"] not in var_cache:
                try:
                    var_cache[m["variable"]] = sim.calculate(
                        m["variable"], TIME_PERIOD, map_to="household"
                    ).values.astype(np.float64)
                except Exception as exc:
                    log.warning("incumbent %s: %s", m["variable"], exc)
                    var_cache[m["variable"]] = np.zeros(n_hh)
            if ckey not in mask_cache:
                mask_cache[ckey] = builder._evaluate_constraints_entity_aware(
                    sim, m["non_geo"], n_hh
                )
            out[i] = var_cache[m["variable"]] * mask_cache[ckey]
    weights = sim.calculate("household_weight", TIME_PERIOD).values.astype(np.float64)
    del sim
    return out, weights


def main(argv=None):
    import pandas as pd
    from scipy import sparse

    from policyengine_us_data.calibration.unified_matrix_builder import (
        UnifiedMatrixBuilder,
        _GEO_VARS,
    )
    from policyengine_us_data.utils.census import STATE_NAME_TO_FIPS

    args = parse_args(argv)
    run = Path(args.run)
    out_path = Path(args.out) if args.out else run / "benchmark.json"

    # State FIPS -> postal code via the calibration utils mapping.
    from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
        STATE_FIPS_TO_CODE,
    )

    meta = json.loads((run / "stack_meta.json").read_text())
    cds = meta["cds"]
    n_hh = meta["n_households"]
    targets_df = pd.read_parquet(run / "targets.parquet")
    X = sparse.load_npz(run / "X_stacked.npz")
    w_full = np.load(run / "weights.npy")

    db_uri = f"sqlite:///{Path(args.run).parent.parent / 'inputs' / 'calibration' / 'policy_data.db'}"
    builder = UnifiedMatrixBuilder(db_uri=db_uri, time_period=TIME_PERIOD)

    constraint_cache: dict[int, list] = {}

    area_codes = None
    if args.areas:
        area_codes = set(args.areas.split(","))

    results = []
    for cd in cds:
        code = cd_geoid_to_code(cd, STATE_FIPS_TO_CODE)
        if area_codes and code not in area_codes:
            continue
        h5 = Path(args.districts_dir) / f"{code}.h5"
        if not h5.exists():
            log.warning("%s: incumbent h5 missing, skipping", code)
            continue

        rows = targets_df[
            (targets_df["geo_level"] == "district")
            & (targets_df["geographic_id"].astype(str) == cd)
        ]
        if rows.empty:
            continue
        rows_meta = []
        row_indices = []
        for idx, row in rows.iterrows():
            sid = int(row["stratum_id"])
            if sid not in constraint_cache:
                constraint_cache[sid] = builder._get_stratum_constraints(sid)
            non_geo = [
                c for c in constraint_cache[sid] if c["variable"] not in _GEO_VARS
            ]
            reform_id = row.get("reform_id", 0) or 0
            if int(reform_id) != 0:
                continue
            rows_meta.append(
                {"variable": str(row["variable"]), "non_geo": non_geo}
            )
            row_indices.append(idx)
        y = targets_df.loc[row_indices, "value"].values.astype(np.float64)

        # Candidate block: nonzero-weight columns of this CD.
        ci = cds.index(cd)
        block = slice(ci * n_hh, (ci + 1) * n_hh)
        w_block = w_full[block]
        nz = np.flatnonzero(w_block > 0)
        M_cand = np.asarray(X[row_indices, block].todense(), dtype=np.float64)[:, nz]
        w0_cand = w_block[nz]

        # Incumbent block.
        M_inc, w0_inc = incumbent_matrix(h5, rows_meta, builder)

        n_matched = min(M_cand.shape[1], M_inc.shape[1])
        area = {
            "area": code,
            "n_targets": len(y),
            "n_candidate": int(M_cand.shape[1]),
            "n_incumbent": int(M_inc.shape[1]),
            "n_matched": int(n_matched),
            "rotations": [],
        }
        init_total = float(w0_inc.sum())

        for r in range(args.rotations):
            seed = BASE_ROTATION_SEED + r
            rng = np.random.default_rng(seed)
            n_t = len(y)
            hold_n = max(1, int(round(HOLDOUT_SHARE * n_t)))
            perm = rng.permutation(n_t)
            hold_idx = np.sort(perm[:hold_n])
            train_idx = np.sort(perm[hold_n:])

            sub_c = matched_subsample(M_cand, w0_cand, n_matched, seed)
            sub_i = matched_subsample(M_inc, w0_inc, n_matched, seed)

            tc, hc = refit_loss(
                M_cand[:, sub_c], y, train_idx, hold_idx,
                epochs=args.epochs, lr=args.lr,
                max_weight=args.max_weight, init_total=init_total,
            )
            ti, hi = refit_loss(
                M_inc[:, sub_i], y, train_idx, hold_idx,
                epochs=args.epochs, lr=args.lr,
                max_weight=args.max_weight, init_total=init_total,
            )
            area["rotations"].append(
                {
                    "seed": seed,
                    "candidate": {"train": tc, "holdout": hc},
                    "incumbent": {"train": ti, "holdout": hi},
                }
            )
            log.info(
                "%s r%d: cand %.4f/%.4f vs inc %.4f/%.4f",
                code, r, tc, hc, ti, hi,
            )
        results.append(area)

    # Aggregate verdict.
    def agg(side, kind):
        vals = [
            rot[side][kind]
            for a in results
            for rot in a["rotations"]
            if np.isfinite(rot[side][kind])
        ]
        return float(np.mean(vals)) if vals else float("nan")

    wins_train = sum(
        1
        for a in results
        for rot in a["rotations"]
        if rot["candidate"]["train"] < rot["incumbent"]["train"]
    )
    wins_hold = sum(
        1
        for a in results
        for rot in a["rotations"]
        if rot["candidate"]["holdout"] < rot["incumbent"]["holdout"]
    )
    n_rot = sum(len(a["rotations"]) for a in results)
    summary = {
        "areas": len(results),
        "rotations_total": n_rot,
        "candidate_train_mean": agg("candidate", "train"),
        "incumbent_train_mean": agg("incumbent", "train"),
        "candidate_holdout_mean": agg("candidate", "holdout"),
        "incumbent_holdout_mean": agg("incumbent", "holdout"),
        "wins_train": wins_train,
        "wins_holdout": wins_hold,
        "verdict_train": agg("candidate", "train") < agg("incumbent", "train"),
        "verdict_holdout": agg("candidate", "holdout") < agg("incumbent", "holdout"),
    }
    out_path.write_text(json.dumps({"summary": summary, "areas": results}, indent=2))
    log.info("SUMMARY %s", json.dumps(summary))


if __name__ == "__main__":
    main()
