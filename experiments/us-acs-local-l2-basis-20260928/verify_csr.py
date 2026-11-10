"""Verify the copied sparse checkpoint against the dense 09-23 release checkpoint.

The L2-basis sweep calibrates from a CSR copy (4,459 targets x 1,588,854
households) that another session converted from the dense lean checkpoint of
the published ACS local release
(``populace-us-2024-buildo-acs-local-767312d60-20260923T074941Z``). Before any
sweep result can speak for that release, this script checks the copy against
the dense source:

1. A seeded random sample of 3,000 households: their dense
   ``household/block2_values`` rows (float32) must equal the corresponding CSR
   columns exactly, zeros included.
2. The dense ``block2_items`` order must equal the CSR row order recorded in
   the copied ``targets.json`` (``measure`` and ``name``), and the 09-23
   ``targets.json`` names/values must equal the copy's.
3. The design weights must agree three ways: the dense H5's
   ``household_weight``, the structure H5's ``household_weight``, and the 09-23
   ``weights_latest.npz`` ``initial_weights``. Household ids, state FIPS and
   district geoids must agree between the dense and structure H5 files.
4. Whole-matrix check (every CSR entry, not a sample): ``A @ d`` and
   ``A @ w_0923`` must reproduce the 09-23 ``calibration_diagnostics.json``
   per-target ``initial_estimate`` and ``final_estimate``, and the 09-23
   headline metrics (ESS, fraction within 10%, final loss) must be recovered
   from the CSR and the saved weights.

Reads only; writes ``results/csr_verification.json``. Exits non-zero on any
mismatch. RAM stays well under 2 GB: the dense matrix is read 500 rows at a
time.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
DENSE_DIR = Path(
    "/Users/maxghenis/PolicyEngine/_recovered/scratch-backup/893/overnight-20260923/"
    "run/release/checkpoints"
)
CHECKPOINT = Path(
    "/Users/maxghenis/PolicyEngine/_build_artifacts/acs-local-l2-basis-20260928/checkpoint"
)
SAMPLE_SIZE = 3_000
SEED = 20260928
READ_CHUNK = 500


def peak_rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**30 if sys.platform == "darwin" else peak / 2**20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(items) -> list[str]:
    return [item.decode() if isinstance(item, bytes) else str(item) for item in items]


def block_column(group: h5py.Group, name: str) -> np.ndarray:
    """One column of a pandas fixed-format frame group, found by item name."""

    for key in group.keys():
        if key.endswith("_items") and name in decode(group[key][:]):
            items = decode(group[key][:])
            values = group[key.replace("_items", "_values")]
            if values.ndim != 2:
                raise ValueError(f"{name} lives in an object block; read it via pandas")
            return values[:, items.index(name)]
    raise KeyError(name)


def kish_ess(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.square(weights).sum())


def main() -> int:
    started = time.time()
    checks: dict[str, object] = {}
    failures: list[str] = []

    def record(name: str, ok: bool, **detail) -> None:
        checks[name] = {"ok": bool(ok), **detail}
        if not ok:
            failures.append(name)
        print(
            f"[{'ok' if ok else 'FAIL'}] {name} {json.dumps(detail)[:300]}", flush=True
        )

    manifest_digests = {
        name: sha256(CHECKPOINT / name)
        for name in (
            "target_matrix.npz",
            "targets.json",
            "target_frame_lean.h5",
            "identity.json",
        )
    }
    identity = json.loads((CHECKPOINT / "identity.json").read_text())
    record(
        "copy_digests_match_identity",
        manifest_digests["target_matrix.npz"] == identity["target_matrix_sha256"]
        and manifest_digests["targets.json"] == identity["targets_sha256"],
        digests=manifest_digests,
    )

    matrix = sparse.load_npz(CHECKPOINT / "target_matrix.npz").tocsr()
    matrix.sort_indices()
    n_targets, n_households = matrix.shape
    record(
        "csr_shape_nnz",
        matrix.shape == (4459, 1_588_854)
        and matrix.nnz == identity["published_matrix_nnz"]
        and matrix.dtype == np.float32,
        shape=list(matrix.shape),
        nnz=int(matrix.nnz),
        dtype=str(matrix.dtype),
        has_canonical_format=bool(matrix.has_canonical_format),
    )
    record(
        "csr_finite_no_explicit_zeros",
        bool(np.isfinite(matrix.data).all()) and int((matrix.data == 0).sum()) == 0,
        explicit_zeros=int((matrix.data == 0).sum()),
        nonfinite=int((~np.isfinite(matrix.data)).sum()),
    )

    targets = json.loads((CHECKPOINT / "targets.json").read_text())
    dense_targets = json.loads((DENSE_DIR / "targets.json").read_text())
    names = [target["name"] for target in targets]
    measures = [target["measure"] for target in targets]

    rng = np.random.default_rng(SEED)
    sample = np.sort(rng.choice(n_households, size=SAMPLE_SIZE, replace=False))
    # household-major view of just the sampled columns
    csr_sample = matrix[:, sample].toarray().T  # (SAMPLE_SIZE, n_targets) float32

    dense_path = DENSE_DIR / "target_frame_lean.h5"
    with h5py.File(dense_path, "r") as handle:
        household = handle["household"]
        block2_items = decode(household["block2_items"][:])
        values = household["block2_values"]
        record(
            "dense_block2_shape_dtype",
            values.shape == (n_households, n_targets) and values.dtype == np.float32,
            shape=list(values.shape),
            dtype=str(values.dtype),
        )
        record(
            "block2_items_equal_csr_target_order",
            block2_items == measures and block2_items == names,
            n_items=len(block2_items),
            first_mismatch=next(
                (
                    i
                    for i, (a, b) in enumerate(
                        zip(block2_items, measures, strict=False)
                    )
                    if a != b
                ),
                None,
            ),
        )
        dense_rows = np.empty((SAMPLE_SIZE, n_targets), dtype=np.float32)
        for low in range(0, SAMPLE_SIZE, READ_CHUNK):
            chunk = sample[low : low + READ_CHUNK]
            dense_rows[low : low + len(chunk)] = values[chunk.tolist(), :]
        dense_ids = block_column(household, "household_id")
        dense_state = block_column(household, "state_fips")
        dense_cd = block_column(household, "congressional_district_geoid")
        dense_weight = block_column(household, "household_weight").astype(np.float64)
        weight_block = next(
            key
            for key in household.keys()
            if key.endswith("_items")
            and "household_weight" in decode(household[key][:])
        )

    nan_dense = int(np.isnan(dense_rows).sum())
    mismatch = dense_rows != csr_sample
    record(
        "sampled_rows_exact_float32_equal",
        nan_dense == 0 and not bool(mismatch.any()),
        sample_size=SAMPLE_SIZE,
        seed=SEED,
        cells_compared=int(dense_rows.size),
        nonzero_cells_dense=int(np.count_nonzero(dense_rows)),
        nonzero_cells_csr=int(np.count_nonzero(csr_sample)),
        mismatched_cells=int(mismatch.sum()),
        nan_cells_dense=nan_dense,
        sample_index_min=int(sample.min()),
        sample_index_max=int(sample.max()),
    )
    del dense_rows, csr_sample, mismatch

    record(
        "dense_and_copied_targets_json_agree",
        [t["name"] for t in dense_targets] == names
        and [float(t["value"]) for t in dense_targets]
        == [float(t["value"]) for t in targets]
        and [t["measure"] for t in dense_targets] == measures,
        n=len(dense_targets),
    )

    import pandas as pd

    struct = pd.read_hdf(CHECKPOINT / "target_frame_lean.h5", "household")
    struct_weight = struct["household_weight"].to_numpy(np.float64)
    saved = np.load(DENSE_DIR / "weights_latest.npz")
    initial_weights = np.asarray(saved["initial_weights"], dtype=np.float64)
    w_0923 = np.asarray(saved["weights"], dtype=np.float64)
    record(
        "design_weights_three_way_exact",
        np.array_equal(dense_weight, struct_weight)
        and np.array_equal(dense_weight, initial_weights),
        dense_block=weight_block,
        n=int(dense_weight.size),
        total=float(dense_weight.sum()),
        min=float(dense_weight.min()),
        max=float(dense_weight.max()),
        max_abs_diff_struct=float(np.abs(dense_weight - struct_weight).max()),
        max_abs_diff_npz=float(np.abs(dense_weight - initial_weights).max()),
    )
    record(
        "household_structure_equal",
        np.array_equal(dense_ids, struct["household_id"].to_numpy())
        and np.array_equal(dense_state, struct["state_fips"].to_numpy())
        and np.array_equal(dense_cd, struct["congressional_district_geoid"].to_numpy()),
        household_id_sorted_unique=bool(np.all(np.diff(dense_ids) > 0)),
    )
    record(
        "weights_latest_epochs",
        int(saved["epochs_done"]) == 800 and w_0923.shape == (n_households,),
        epochs_done=int(saved["epochs_done"]),
    )

    # Whole-matrix check against the 09-23 per-target diagnostics.
    diagnostics = json.loads((DENSE_DIR / "calibration_diagnostics.json").read_text())
    diag_targets = diagnostics["targets"]
    record(
        "diagnostics_target_order",
        [t["name"] for t in diag_targets] == names,
        n=len(diag_targets),
    )
    matrix64 = matrix.astype(np.float64)
    est_design = matrix64 @ dense_weight
    est_0923 = matrix64 @ w_0923
    diag_initial = np.asarray([t["initial_estimate"] for t in diag_targets])
    diag_final = np.asarray([t["final_estimate"] for t in diag_targets])
    target_values = np.asarray([float(t["value"]) for t in targets])
    scale = np.maximum(np.abs(diag_initial), 1.0)
    rel_initial = np.abs(est_design - diag_initial) / scale
    rel_final = np.abs(est_0923 - diag_final) / np.maximum(np.abs(diag_final), 1.0)
    record(
        "full_matrix_reproduces_0923_estimates",
        float(rel_initial.max()) < 1e-9 and float(rel_final.max()) < 1e-9,
        max_rel_diff_initial_estimate=float(rel_initial.max()),
        max_rel_diff_final_estimate=float(rel_final.max()),
    )
    rel_error = np.where(
        target_values != 0,
        (est_0923 - target_values) / np.where(target_values != 0, target_values, 1.0),
        est_0923 - target_values,
    )
    within = float(np.mean(np.abs(rel_error) <= 0.10))
    capped = np.minimum(
        np.abs(est_0923 - target_values) / np.maximum(np.abs(target_values), 1.0), 1.0
    )
    final_loss = float(capped.mean())
    ess_0923 = kish_ess(w_0923)
    ess_design = kish_ess(dense_weight)
    from microcosm.calibrate import chi_square_distance

    chi2 = chi_square_distance(w_0923, dense_weight)
    record(
        "full_matrix_reproduces_0923_headline",
        abs(within - diagnostics["fraction_within_10pct"]) < 5e-5
        and abs(final_loss - diagnostics["final_loss"]) < 5e-7
        and abs(ess_0923 - diagnostics["effective_sample_size"]) < 0.05,
        fraction_within_10pct=within,
        fraction_within_10pct_0923=diagnostics["fraction_within_10pct"],
        final_loss=final_loss,
        final_loss_0923=diagnostics["final_loss"],
        kish_ess=ess_0923,
        kish_ess_0923=diagnostics["effective_sample_size"],
        kish_ess_design=ess_design,
        chi_square_distance_0923_vs_design=chi2,
        mass_ratio=float(w_0923.sum() / dense_weight.sum()),
        realized_max_weight_ratio=float((w_0923 / dense_weight).max()),
    )

    result = {
        "ok": not failures,
        "failures": failures,
        "dense_checkpoint": str(DENSE_DIR),
        "sparse_checkpoint": str(CHECKPOINT),
        "sample": {"size": SAMPLE_SIZE, "seed": SEED, "read_chunk": READ_CHUNK},
        "checks": checks,
        "wall_seconds": round(time.time() - started, 1),
        "peak_rss_gb": round(peak_rss_gb(), 3),
    }
    out = HERE / "results" / "csr_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"wrote {out} ok={result['ok']} peak_rss_gb={result['peak_rss_gb']}")
    if not math.isfinite(result["peak_rss_gb"]):
        return 1
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
