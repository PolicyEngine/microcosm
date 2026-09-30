"""Per-run harness for the chi-square L2 basis / softmax mass sweep.

One run recalibrates the published ACS local release's 4,459-target surface
(``populace-us-2024-buildo-acs-local-767312d60-20260923T074941Z``) from its
verified sparse checkpoint with the release's settings — Adam, 800 epochs as
warm-started batches of 400, lr 0.02, ``mass="conserve"``,
``max_weight_ratio=5.0``, ``target_loss_cap=1.0``, seed 0, default target loss
weights and scales — plus the run's ``l2_lambda``, ``l2_basis`` and
``mass_parametrization``, optionally holding one rotated fold of targets out.
The batch loop mirrors ``tools/build_us_acs_local_release.py`` ``do_calibrate``.

Run spec (JSON or flags)::

    {"run_id": "...", "l2_lambda": 0.0, "l2_basis": "record",
     "mass_parametrization": "projection", "epochs": 800, "epoch_batch": 400,
     "holdout_fold": null, "acs_share": null}

``acs_share`` reseeds the prior: the ACS spine's rows are rescaled together to
that share of the household mass and the donor (ASEC-by-PUF) rows to the rest,
each spine keeping its internal proportions, as the staging's ``--acs-share``
does through ``base_pool._pooled_household_weights``. ``null`` keeps the
checkpoint's weights (the release's ``--acs-share 0.5``). The prior becomes the
frame's weights, so it is also the chi-square anchor, the base of the 5x ratio
cap, and the "design" every metric is measured against; the original design
weights stay recorded alongside.

Modes:

``calibrate`` (default)
    Calibrate, then write ``<out_root>/<run_id>/metrics.json``,
    ``weights.npz`` (float64 final weights, per-batch loss trajectories) and
    ``progress.log``. A per-batch ``resume.npz`` lets a restarted run continue
    from its last finished batch (``--resume``).
``evaluate``
    Score a given weight vector without calibrating: ``--weights design``
    (``w = d``, the design reference) or ``--weights <npz>[:key]``. With
    ``--weights design`` every holdout fold's design block is also written.

Metrics (``metrics.json``) are computed from the final weights ``w`` and the
design weights ``d``: national, per-spine, per-state, per-district and
Massachusetts concentration; fit on the training targets (overall, per family
and per subfamily); and, with a holdout fold, the same fit block on the
held-out targets for ``w`` and for ``d``.

All heavy inputs are verified against the checkpoint's ``MANIFEST.json``
sha256s before use.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
REPO = Path(
    os.environ.get("MICROCOSM_REPO", HERE.parents[1] if len(HERE.parents) > 1 else HERE)
)
DEFAULT_CHECKPOINT = Path(
    "/Users/maxghenis/PolicyEngine/_build_artifacts/acs-local-l2-basis-20260928/checkpoint"
)
DEFAULT_OUT_ROOT = Path(
    "/Users/maxghenis/PolicyEngine/_build_artifacts/acs-local-l2-basis-20260928/runs"
)
RELEASE_TAG = "populace-us-2024-buildo-acs-local-767312d60-20260923T074941Z"
#: The published release's calibration settings (calibration_diagnostics.json).
RELEASE_SETTINGS = {
    "method": "adam",
    "learning_rate": 0.02,
    "mass": "conserve",
    "max_weight_ratio": 5.0,
    "target_loss_cap": 1.0,
    "seed": 0,
    "weight_entity": "household",
}
#: What the published release recorded at l2_lambda=0 (800 epochs, 2 x 400).
RELEASE_0923_RESULT = {
    "final_loss": 0.015491,
    "fraction_within_10pct": 0.9758,
    "effective_sample_size": 13631.3,
    "initial_loss_second_batch_start": 0.017768,
}
SPINES = ("acs_2024_1yr", "asec_puf")
MASSACHUSETTS = "25"
WITHIN = 0.10
CAP = 1.0
INPUT_FILES = (
    "target_matrix.npz",
    "households.parquet",
    "targets_meta.parquet",
    "holdout_folds.npz",
)
KERNEL_MODULES = (
    "microcosm.calibrate.solve",
    "microcosm.calibrate.matrix",
    "microcosm.calibrate.target",
    "microcosm.frame.bundle",
    "microcosm.frame.weights",
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def peak_rss_gb() -> float:
    """Process peak RSS (ru_maxrss is bytes on macOS, KiB on Linux)."""

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**30 if sys.platform == "darwin" else peak / 2**20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    """Recursively convert numpy/pandas scalars and containers for json."""

    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "items"):
        return {str(k): jsonable(v) for k, v in value.items()}
    return str(value)


class Logger:
    def __init__(self, path: Path | None):
        self.path = path
        self.handle = open(path, "a") if path is not None else None

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        if self.handle is not None:
            self.handle.write(line + "\n")
            self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()


# ---------------------------------------------------------------------------
# Run spec
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunSpec:
    run_id: str
    l2_lambda: float = 0.0
    l2_basis: str = "record"
    mass_parametrization: str = "projection"
    epochs: int = 800
    epoch_batch: int = 400
    holdout_fold: int | None = None
    acs_share: float | None = None

    def __post_init__(self) -> None:
        if not self.run_id or "/" in self.run_id:
            raise ValueError(
                f"run_id must be a non-empty path segment: {self.run_id!r}"
            )
        if not (math.isfinite(self.l2_lambda) and self.l2_lambda >= 0.0):
            raise ValueError(f"l2_lambda must be finite and >= 0: {self.l2_lambda!r}")
        if self.l2_basis not in ("record", "chi_square"):
            raise ValueError(f"unknown l2_basis {self.l2_basis!r}")
        if self.mass_parametrization not in ("projection", "softmax"):
            raise ValueError(
                f"unknown mass_parametrization {self.mass_parametrization!r}"
            )
        if self.epochs <= 0 or self.epoch_batch <= 0:
            raise ValueError("epochs and epoch_batch must be positive")
        if self.holdout_fold is not None and not (0 <= self.holdout_fold < 5):
            raise ValueError(
                f"holdout_fold must be None or 0..4: {self.holdout_fold!r}"
            )
        if self.acs_share is not None and not (0.0 < self.acs_share < 1.0):
            raise ValueError(f"acs_share must be in (0, 1): {self.acs_share!r}")

    @classmethod
    def from_mapping(cls, mapping: dict) -> RunSpec:
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"unknown run spec keys {unknown}")
        values = dict(mapping)
        for key in ("l2_lambda",):
            if key in values:
                values[key] = float(values[key])
        if values.get("acs_share") is not None:
            values["acs_share"] = float(values["acs_share"])
        for key in ("epochs", "epoch_batch"):
            if key in values:
                values[key] = int(values[key])
        if values.get("holdout_fold") is not None:
            values["holdout_fold"] = int(values["holdout_fold"])
        return cls(**values)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Inputs:
    checkpoint: Path
    matrix: sparse.csr_array  # (targets, households) float32
    households: pd.DataFrame
    meta: pd.DataFrame
    folds: dict[int, np.ndarray]
    fold_of_target: np.ndarray
    manifest: dict
    input_sha256: dict[str, str]
    _matrix64: sparse.csr_array | None = None

    def matrix64(self) -> sparse.csr_array:
        """The CSR widened to float64 (exact), built once on first use."""

        if self._matrix64 is None:
            self._matrix64 = self.matrix.astype(np.float64)
        return self._matrix64

    @property
    def design(self) -> np.ndarray:
        return self.households["design_weight"].to_numpy(np.float64)

    @property
    def values(self) -> np.ndarray:
        return self.meta["value"].to_numpy(np.float64)

    def prior(self, acs_share: float | None) -> np.ndarray:
        """The run's starting weights: the design, reseeded to ``acs_share``."""

        return reseed_prior(
            self.design, self.households["spine"].astype(str).to_numpy(), acs_share
        )


def reseed_prior(
    design: np.ndarray, spine: np.ndarray, acs_share: float | None
) -> np.ndarray:
    """Rescale each spine to its share of the unchanged total.

    ACS rows together get ``acs_share`` of the mass and donor rows the rest;
    proportions within a spine are kept. ``None`` returns ``design`` itself.
    """

    design = np.asarray(design, dtype=np.float64)
    if acs_share is None:
        return design
    acs = spine == SPINES[0]
    donor = spine == SPINES[1]
    if not (acs | donor).all() or not acs.any() or not donor.any():
        raise SystemExit("every household must be on exactly one of the two spines")
    total = float(design.sum())
    prior = design.copy()
    prior[acs] *= acs_share * total / float(design[acs].sum())
    prior[donor] *= (1.0 - acs_share) * total / float(design[donor].sum())
    return prior


def load_inputs(
    checkpoint: Path, *, verify: bool = True, log: Callable = print
) -> Inputs:
    started = time.time()
    manifest = json.loads((checkpoint / "MANIFEST.json").read_text())
    if not manifest.get("ok"):
        raise SystemExit(f"{checkpoint}/MANIFEST.json is not ok")
    digests = {}
    if verify:
        for name in INPUT_FILES:
            actual = sha256(checkpoint / name)
            expected = manifest["files"][name]
            if actual != expected:
                raise SystemExit(f"{name}: sha256 {actual} != manifest {expected}")
            digests[name] = actual
    matrix = sparse.load_npz(checkpoint / "target_matrix.npz").tocsr()
    matrix.sort_indices()
    households = pd.read_parquet(checkpoint / "households.parquet")
    meta = pd.read_parquet(checkpoint / "targets_meta.parquet")
    folds_npz = np.load(checkpoint / "holdout_folds.npz")
    n_folds = int(folds_npz["n_folds"])
    folds = {
        i: np.asarray(folds_npz[f"fold_{i}"], dtype=np.int64) for i in range(n_folds)
    }
    fold_of_target = np.asarray(folds_npz["fold_of_target"], dtype=np.int64)
    n_targets, n_households = matrix.shape
    if len(households) != n_households or len(meta) != n_targets:
        raise SystemExit("households/targets metadata do not align with the CSR")
    if not np.array_equal(meta["row"].to_numpy(), np.arange(n_targets)):
        raise SystemExit("targets_meta rows are not in CSR row order")
    if not np.all(np.diff(households["household_id"].to_numpy()) > 0):
        raise SystemExit("household ids must be strictly increasing (Frame contract)")
    if not np.array_equal(meta["fold"].to_numpy(), fold_of_target):
        raise SystemExit("targets_meta fold column differs from holdout_folds.npz")
    log(
        f"inputs: {n_targets} targets x {n_households:,} households, nnz "
        f"{matrix.nnz:,}, verified={verify}, {time.time() - started:.1f}s"
    )
    return Inputs(
        checkpoint=checkpoint,
        matrix=matrix,
        households=households,
        meta=meta,
        folds=folds,
        fold_of_target=fold_of_target,
        manifest=manifest,
        input_sha256=digests,
    )


def build_frame(households: pd.DataFrame, weights: np.ndarray):
    """A minimal frame: households with the run's prior weights, one person each."""

    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    household_id = households["household_id"].to_numpy(np.int64)
    tables = {
        "household": pd.DataFrame({"household_id": household_id}),
        "person": pd.DataFrame(
            {
                "person_id": np.arange(len(household_id), dtype=np.int64),
                "person_household_id": household_id,
            }
        ),
    }
    return Frame(
        tables,
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.asarray(weights, np.float64), WeightKind.DESIGN)},
    )


def csr_row_measure(matrix: sparse.csr_array, row: int, n: int):
    """A callable measure returning CSR row ``row`` as a dense float64 vector.

    float32 values widen exactly to float64, as the dense checkpoint's float32
    columns did when the release compiled them.
    """

    start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
    indices = matrix.indices[start:stop]
    data = matrix.data[start:stop]

    def measure(frame) -> np.ndarray:
        values = np.zeros(n, dtype=np.float64)
        values[indices] = data
        return values

    measure.__qualname__ = f"target_matrix_row[{row}]"
    return measure


def build_target_set(inputs: Inputs, rows: Sequence[int]):
    from microcosm.calibrate import Target, TargetSet

    n_households = inputs.matrix.shape[1]
    names = inputs.meta["name"].to_numpy()
    values = inputs.values
    return TargetSet(
        Target(
            name=str(names[row]),
            entity="household",
            measure=csr_row_measure(inputs.matrix, int(row), n_households),
            value=float(values[row]),
            period=2024,
            source="acs_local_0923_checkpoint",
        )
        for row in rows
    )


def training_rows(inputs: Inputs, holdout_fold: int | None) -> np.ndarray:
    n_targets = inputs.matrix.shape[0]
    if holdout_fold is None:
        return np.arange(n_targets, dtype=np.int64)
    return np.flatnonzero(inputs.fold_of_target != holdout_fold).astype(np.int64)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def n_records_holding_half(weights: np.ndarray) -> int:
    ordered = np.sort(weights)[::-1]
    cumulative = np.cumsum(ordered)
    return int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left") + 1)


def top_share(weights: np.ndarray, fraction: float = 0.01) -> float:
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    k = max(1, math.ceil(fraction * weights.size))
    return float(
        np.partition(weights, weights.size - k)[weights.size - k :].sum() / total
    )


def kish(weights: np.ndarray) -> float:
    denominator = float(np.square(weights).sum())
    return float(weights.sum() ** 2 / denominator) if denominator > 0 else 0.0


def grouped_concentration(
    weights: np.ndarray, codes: np.ndarray, n_groups: int
) -> dict[str, np.ndarray]:
    """Per-group n, mass, Kish ESS, top-1% share and half-mass record count."""

    n = np.bincount(codes, minlength=n_groups).astype(np.int64)
    mass = np.bincount(codes, weights=weights, minlength=n_groups)
    square = np.bincount(codes, weights=np.square(weights), minlength=n_groups)
    ess = np.divide(mass**2, square, out=np.zeros(n_groups), where=square > 0)
    order = np.lexsort((-weights, codes))
    sorted_weights = weights[order]
    sorted_codes = codes[order]
    starts = np.concatenate([[0], np.cumsum(n)[:-1]])
    rank = np.arange(weights.size) - starts[sorted_codes]
    cumulative = np.cumsum(sorted_weights)
    before = np.concatenate([[0.0], cumulative])[starts]
    within = cumulative - before[sorted_codes]
    below_half = within < 0.5 * mass[sorted_codes]
    half = np.bincount(sorted_codes, weights=below_half, minlength=n_groups).astype(
        np.int64
    ) + (n > 0)
    k = np.maximum(1, np.ceil(0.01 * n)).astype(np.int64)
    top = np.bincount(
        sorted_codes,
        weights=sorted_weights * (rank < k[sorted_codes]),
        minlength=n_groups,
    )
    top_share_values = np.divide(top, mass, out=np.zeros(n_groups), where=mass > 0)
    return {
        "n": n,
        "mass": mass,
        "kish_ess": ess,
        "top_1pct_share": top_share_values,
        "n_records_holding_half_weight": half,
    }


def summary_stats(values: np.ndarray, labels: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "argmin": str(labels[int(values.argmin())]),
        "argmax": str(labels[int(values.argmax())]),
    }


def concentration_metrics(
    w: np.ndarray, d: np.ndarray, households: pd.DataFrame
) -> dict:
    from microcosm.calibrate import chi_square_distance

    n = int(w.size)
    total = float(w.sum())
    ratio = w / d
    ess = kish(w)
    national = {
        "households": n,
        "total_weight": total,
        "mass_conserved_ratio": total / float(d.sum()),
        "kish_ess": ess,
        "kish_ess_design": kish(d),
        "ess_fraction": ess / n,
        "top_1pct_weight_share": top_share(w),
        "chi_square_distance": chi_square_distance(w, d),
        "realized_max_weight_ratio": float(ratio.max()),
        "realized_min_weight_ratio": float(ratio.min()),
        "share_of_design_mass_with_ratio_below_0_1": float(
            d[ratio < 0.1].sum() / d.sum()
        ),
        "share_of_design_mass_with_ratio_below_0_01": float(
            d[ratio < 0.01].sum() / d.sum()
        ),
        "share_of_design_mass_at_cap": float(
            d[ratio >= 5.0 * (1 - 1e-9)].sum() / d.sum()
        ),
        "n_records_at_cap": int((ratio >= 5.0 * (1 - 1e-9)).sum()),
        "n_records_holding_half_weight": n_records_holding_half(w),
        "weight_ratio_quantiles": {
            q: float(np.quantile(ratio, float(q)))
            for q in ("0.01", "0.1", "0.25", "0.5", "0.75", "0.9", "0.99")
        },
        "design_weighted_mean_log_ratio": float(
            np.sum(d * np.log(np.maximum(ratio, 1e-300))) / d.sum()
        ),
    }

    spine = households["spine"].astype(str).to_numpy()
    per_spine = {}
    for label in SPINES:
        mask = spine == label
        per_spine[label] = {
            "n": int(mask.sum()),
            "mass_share": float(w[mask].sum() / total),
            "design_mass_share": float(d[mask].sum() / d.sum()),
            "kish_ess": kish(w[mask]),
            "kish_ess_design": kish(d[mask]),
            "n_records_holding_half_weight": n_records_holding_half(w[mask]),
        }

    def geography(column: str) -> tuple[dict, dict, dict]:
        codes_cat = households[column].astype("category")
        labels = np.asarray(codes_cat.cat.categories.astype(str))
        codes = codes_cat.cat.codes.to_numpy(np.int64)
        on_w = grouped_concentration(w, codes, len(labels))
        on_d = grouped_concentration(d, codes, len(labels))
        present = on_w["n"] > 0
        rows = {}
        for i, label in enumerate(labels):
            if not present[i]:
                continue
            rows[str(label)] = {
                "n": int(on_w["n"][i]),
                "kish_ess": float(on_w["kish_ess"][i]),
                "kish_ess_design": float(on_d["kish_ess"][i]),
                "kish_ess_ratio": float(on_w["kish_ess"][i] / on_d["kish_ess"][i]),
                "mass_ratio_vs_design": float(on_w["mass"][i] / on_d["mass"][i]),
                "top_1pct_share": float(on_w["top_1pct_share"][i]),
                "n_records_holding_half_weight": int(
                    on_w["n_records_holding_half_weight"][i]
                ),
                "n_records_holding_half_weight_design": int(
                    on_d["n_records_holding_half_weight"][i]
                ),
            }
        ess = on_w["kish_ess"][present]
        ratio_ess = ess / on_d["kish_ess"][present]
        summary = {
            "n_groups": int(present.sum()),
            "kish_ess": summary_stats(ess, labels[present]),
            "kish_ess_design": summary_stats(
                on_d["kish_ess"][present], labels[present]
            ),
            "kish_ess_ratio": summary_stats(ratio_ess, labels[present]),
            "n_records_holding_half_weight": summary_stats(
                on_w["n_records_holding_half_weight"][present], labels[present]
            ),
            "mass_ratio_vs_design": summary_stats(
                on_w["mass"][present] / on_d["mass"][present], labels[present]
            ),
        }
        return rows, summary, {"labels": labels}

    per_state, state_summary, _ = geography("state_fips")
    per_cd, cd_summary, _ = geography("congressional_district_geoid")
    massachusetts = {
        "state": per_state[MASSACHUSETTS],
        "districts": {
            code: row for code, row in per_cd.items() if code.startswith(MASSACHUSETTS)
        },
    }
    return {
        "national": national,
        "per_spine": per_spine,
        "state_summary": state_summary,
        "cd_summary": cd_summary,
        "massachusetts": massachusetts,
        "per_state": per_state,
        "per_cd": per_cd,
    }


def fit_rows(estimates: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    difference = estimates - targets
    relative = np.where(
        targets != 0, difference / np.where(targets != 0, targets, 1.0), difference
    )
    capped = np.minimum(np.abs(difference) / np.maximum(np.abs(targets), 1.0), CAP)
    return {"relative": relative, "abs_relative": np.abs(relative), "capped": capped}


def fit_summary(rows: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict:
    abs_rel = rows["abs_relative"] if mask is None else rows["abs_relative"][mask]
    capped = rows["capped"] if mask is None else rows["capped"][mask]
    if abs_rel.size == 0:
        return {"n_targets": 0}
    return {
        "n_targets": int(abs_rel.size),
        "fraction_within_10pct": float(np.mean(abs_rel <= WITHIN)),
        "mean_abs_rel_error": float(abs_rel.mean()),
        "median_abs_rel_error": float(np.median(abs_rel)),
        "p90_abs_rel_error": float(np.percentile(abs_rel, 90)),
        "max_abs_rel_error": float(abs_rel.max()),
        "mean_capped_scaled_error": float(capped.mean()),
    }


def fit_block(
    estimates: np.ndarray, meta: pd.DataFrame, rows: np.ndarray, worst: int = 10
) -> dict:
    """Fit of ``estimates`` (all targets) on the target subset ``rows``."""

    subset = meta.iloc[rows]
    values = subset["value"].to_numpy(np.float64)
    est = estimates[rows]
    errors = fit_rows(est, values)
    block = {"overall": fit_summary(errors), "per_family": {}, "per_subfamily": {}}
    for column, key in (("family", "per_family"), ("subfamily", "per_subfamily")):
        labels = subset[column].to_numpy()
        for label in sorted(set(labels.tolist())):
            block[key][label] = fit_summary(errors, labels == label)
    order = np.argsort(-errors["capped"])[:worst]
    block["worst_targets"] = [
        {
            "row": int(subset["row"].iloc[i]),
            "name": str(subset["name"].iloc[i]),
            "value": float(values[i]),
            "estimate": float(est[i]),
            "relative_error": float(errors["relative"][i]),
        }
        for i in order
    ]
    return block


def estimates_for(inputs: Inputs, weights: np.ndarray) -> np.ndarray:
    return np.asarray(inputs.matrix64() @ np.asarray(weights, np.float64))


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def provenance() -> dict:
    import importlib

    import torch

    info: dict = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "pandas": pd.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "env": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "nice": os.nice(0) if hasattr(os, "nice") else None,
        "kernel_module_sha256": {},
    }
    for name in KERNEL_MODULES:
        module = importlib.import_module(name)
        info["kernel_module_sha256"][name] = sha256(Path(module.__file__))
        if name == "microcosm.calibrate.solve":
            info["kernel_solve_path"] = module.__file__

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        info["git"] = {
            "repo": str(REPO),
            "head": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        }
        for package in ("microcosm-calibrate", "microcosm-frame"):
            path = f"packages/{package}"
            status = git("status", "--porcelain", "--", path).splitlines()
            src_status = git("status", "--porcelain", "--", f"{path}/src").splitlines()
            info["git"][package] = {
                "clean_vs_head": not status,
                "src_clean_vs_head": not src_status,
                "dirty_paths": status,
            }
    except (OSError, subprocess.CalledProcessError) as exc:
        info["git"] = {
            "unavailable": str(exc)[:200],
            "head_from_env": os.environ.get("MICROCOSM_GIT_SHA"),
        }
    return info


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _result_block(result) -> dict:
    return {
        "final_loss": float(result.final_loss),
        "fraction_within_10pct": float(result.fraction_within_10pct),
        "effective_sample_size": float(result.effective_sample_size),
        "chi_square_distance": float(result.chi_square_distance),
        "realized_max_weight_ratio": float(result.realized_max_weight_ratio),
        "top_1pct_weight_share": float(result.top_1pct_weight_share),
        "n_targets": int(result.problem.n_targets),
        "matrix_nnz": int(result.problem.matrix.nnz),
        "skipped": len(result.problem.skipped),
        "initial_loss_last_batch": float(result.initial_loss),
    }


def run_calibration(
    spec: RunSpec,
    inputs: Inputs,
    out_dir: Path,
    log: Logger,
    *,
    resume: bool,
    stop_after_batches: int | None = None,
) -> dict:
    import inspect

    import torch

    from microcosm.calibrate import calibrate

    parameters = inspect.signature(calibrate).parameters
    if "l2_basis" not in parameters or "mass_parametrization" not in parameters:
        raise SystemExit(
            "the imported microcosm.calibrate.calibrate lacks l2_basis / "
            "mass_parametrization; run from the l2-design-basis kernel"
        )

    frame = build_frame(inputs.households, inputs.prior(spec.acs_share))
    train = training_rows(inputs, spec.holdout_fold)
    targets = build_target_set(inputs, train)
    expected_names = [f"{name}@2024" for name in inputs.meta["name"].to_numpy()[train]]
    log(
        f"run {spec.run_id}: {len(train)} training targets"
        + (
            f" (fold {spec.holdout_fold} held out)"
            if spec.holdout_fold is not None
            else ""
        )
        + f"; l2_lambda={spec.l2_lambda} l2_basis={spec.l2_basis} "
        f"mass_parametrization={spec.mass_parametrization} epochs={spec.epochs} "
        f"batch={spec.epoch_batch} acs_share={spec.acs_share}; "
        f"torch threads={torch.get_num_threads()}"
    )

    resume_path = out_dir / "resume.npz"
    warm, done = None, 0
    batches: list[dict] = []
    trajectories: list[np.ndarray] = []
    options: dict | None = None
    result_block: dict | None = None
    spec_json = json.dumps(dataclasses.asdict(spec), sort_keys=True)
    if resume and resume_path.exists():
        saved = np.load(resume_path, allow_pickle=False)
        if str(saved["spec_json"]) != spec_json:
            raise SystemExit(f"{resume_path} belongs to a different run spec")
        warm = np.asarray(saved["weights"], dtype=np.float64)
        done = int(saved["epochs_done"])
        batches = json.loads(str(saved["batches_json"]))
        trajectories = [
            np.asarray(saved[f"trajectory_{i}"]) for i in range(len(batches))
        ]
        options = json.loads(str(saved["options_json"]))
        result_block = json.loads(str(saved["result_json"]))
        log(f"RESUME from {done} epochs ({len(batches)} batches)")

    result = None
    batches_this_process = 0
    log_every = max(1, min(50, spec.epoch_batch // 8))
    while done < spec.epochs:
        if (
            stop_after_batches is not None
            and batches_this_process >= stop_after_batches
        ):
            log(
                f"STOP after {batches_this_process} batch(es) at {done} epochs (resumable)"
            )
            return {"stopped": True, "epochs_done": done}
        this_batch = min(spec.epoch_batch, spec.epochs - done)
        stamps: dict[str, float] = {}

        def on_progress(event: dict, _stamps=stamps, _done=done) -> None:
            if event.get("kind") != "calibration_epoch":
                return
            now = time.time()
            epoch = int(event["epoch"])
            if epoch == 1:
                _stamps["first_epoch"] = now
            _stamps["last_epoch"] = now
            _stamps["last_epoch_index"] = epoch
            if epoch == 1 or epoch % log_every == 0 or epoch == event.get("epochs"):
                log(
                    f"  epoch {_done + epoch}/{spec.epochs} "
                    f"loss={float(event['loss']):.6f} rss_peak={peak_rss_gb():.2f}GB"
                )

        if result is not None:
            del result
            gc.collect()
        batch_started = time.time()
        result = calibrate(
            frame,
            targets,
            weight_entity=RELEASE_SETTINGS["weight_entity"],
            method=RELEASE_SETTINGS["method"],
            epochs=this_batch,
            learning_rate=RELEASE_SETTINGS["learning_rate"],
            mass=RELEASE_SETTINGS["mass"],
            max_weight_ratio=RELEASE_SETTINGS["max_weight_ratio"],
            target_loss_cap=RELEASE_SETTINGS["target_loss_cap"],
            l2_lambda=spec.l2_lambda,
            l2_basis=spec.l2_basis,
            mass_parametrization=spec.mass_parametrization,
            seed=RELEASE_SETTINGS["seed"],
            warm_start_weights=warm,
            progress_callback=on_progress,
        )
        finished = time.time()
        if result.problem.skipped:
            skipped = [s.target.name for s in result.problem.skipped]
            raise SystemExit(
                f"calibration skipped {len(skipped)} targets: {skipped[:5]}"
            )
        if list(result.problem.names) != expected_names:
            raise SystemExit(
                "compiled target rows differ from the requested training rows"
            )
        done += this_batch
        warm = np.asarray(result.weights, dtype=np.float64).copy()
        first = stamps.get("first_epoch", finished)
        last = stamps.get("last_epoch", finished)
        n_epoch_events = int(stamps.get("last_epoch_index", 0))
        batch = {
            "batch": len(batches),
            "epochs": this_batch,
            "epochs_done": done,
            "wall_seconds": finished - batch_started,
            "compile_seconds": first - batch_started,
            "optimize_seconds": last - first,
            "seconds_per_epoch": (
                (last - first) / (n_epoch_events - 1) if n_epoch_events > 1 else None
            ),
            "post_seconds": finished - last,
            "start_loss": float(result.loss_trajectory[0]),
            "last_trajectory_loss": float(result.loss_trajectory[-1]),
            "final_loss": float(result.final_loss),
            "fraction_within_10pct": float(result.fraction_within_10pct),
            "effective_sample_size": float(result.effective_sample_size),
            "chi_square_distance": float(result.chi_square_distance),
            "realized_max_weight_ratio": float(result.realized_max_weight_ratio),
            "peak_rss_gb": peak_rss_gb(),
        }
        batches.append(batch)
        batches_this_process += 1
        trajectories.append(np.asarray(result.loss_trajectory, dtype=np.float64))
        options = jsonable(dict(result.options))
        result_block = _result_block(result)
        np.savez(
            resume_path,
            weights=warm,
            epochs_done=np.int64(done),
            spec_json=np.str_(spec_json),
            batches_json=np.str_(json.dumps(jsonable(batches))),
            options_json=np.str_(json.dumps(options)),
            result_json=np.str_(json.dumps(jsonable(result_block))),
            **{f"trajectory_{i}": t for i, t in enumerate(trajectories)},
        )
        log(
            f"batch -> {done}/{spec.epochs} ep, {batch['wall_seconds']:.1f}s "
            f"(compile {batch['compile_seconds']:.1f}s, "
            f"{(batch['seconds_per_epoch'] or float('nan')):.3f}s/epoch), "
            f"loss={batch['final_loss']:.6f}, "
            f"within10%={batch['fraction_within_10pct']:.2%}, "
            f"ESS={batch['effective_sample_size']:,.0f}, "
            f"chi2={batch['chi_square_distance']:.4f}, "
            f"peak RSS {batch['peak_rss_gb']:.2f}GB"
        )

    if result is None:
        # Resumed an already finished run: the saved weights and last-batch
        # result/options are the run's; nothing was recalibrated.
        if warm is None or result_block is None:
            raise SystemExit("no batches ran and no finished resume state exists")
        weights = warm
        result_block = {**result_block, "restored_from_resume": True}
    else:
        weights = np.asarray(result.weights, dtype=np.float64)
        del result
        gc.collect()

    np.savez(
        out_dir / "weights.npz",
        weights=weights,
        epochs_done=np.int64(done),
        spec_json=np.str_(spec_json),
        **{f"trajectory_{i}": t for i, t in enumerate(trajectories)},
    )
    return {
        "stopped": False,
        "resumed": bool(resume and batches_this_process < len(batches)),
        "weights": weights,
        "train_rows": train,
        "batches": batches,
        "options": options,
        "result": result_block,
        "trajectory_first_loss": float(trajectories[0][0]) if trajectories else None,
    }


def score(
    inputs: Inputs,
    weights: np.ndarray,
    train: np.ndarray,
    holdout: np.ndarray | None,
    acs_share: float | None = None,
) -> dict:
    """Metrics against the run's prior (the "design" blocks) and the release's.

    With ``acs_share`` unset the prior is the release's design weights and the
    two coincide.
    """

    from microcosm.calibrate import chi_square_distance

    design = inputs.prior(acs_share)
    estimates = estimates_for(inputs, weights)
    design_estimates = estimates_for(inputs, design)
    original = inputs.design
    metrics = {
        "concentration": concentration_metrics(weights, design, inputs.households),
        "fit_train": fit_block(estimates, inputs.meta, train),
        "prior": {
            "acs_share": acs_share,
            "realized_acs_share": float(
                design[
                    inputs.households["spine"].astype(str).to_numpy() == SPINES[0]
                ].sum()
                / design.sum()
            ),
            "kish_ess_prior": kish(design),
            "kish_ess_release_design": kish(original),
            "chi_square_distance_prior_from_release_design": chi_square_distance(
                design, original
            ),
            "chi_square_distance_from_release_design": chi_square_distance(
                weights, original
            ),
        },
    }
    if holdout is not None:
        metrics["fit_holdout"] = fit_block(estimates, inputs.meta, holdout)
        metrics["fit_holdout_design"] = fit_block(
            design_estimates, inputs.meta, holdout
        )
    metrics["fit_train_design"] = {
        "overall": fit_block(design_estimates, inputs.meta, train)["overall"]
    }
    return metrics


def calibrate_mode(
    spec: RunSpec,
    checkpoint: Path,
    out_root: Path,
    *,
    resume: bool,
    verify: bool,
    stop_after_batches: int | None = None,
) -> dict:
    out_dir = out_root / spec.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "progress.log")
    started = time.time()
    try:
        (out_dir / "spec.json").write_text(
            json.dumps(dataclasses.asdict(spec), indent=2) + "\n"
        )
        inputs = load_inputs(checkpoint, verify=verify, log=log)
        prov = provenance()
        calibration = run_calibration(
            spec,
            inputs,
            out_dir,
            log,
            resume=resume,
            stop_after_batches=stop_after_batches,
        )
        if calibration["stopped"]:
            return calibration
        weights = calibration["weights"]
        train = calibration["train_rows"]
        holdout = (
            inputs.folds[spec.holdout_fold] if spec.holdout_fold is not None else None
        )
        scoring_started = time.time()
        metrics = score(inputs, weights, train, holdout, spec.acs_share)
        from microcosm.calibrate import relative_error_loss

        recomputed_loss = relative_error_loss(
            estimates_for(inputs, weights)[train],
            inputs.values[train],
            target_loss_cap=RELEASE_SETTINGS["target_loss_cap"],
        )
        consistency = {
            "recomputed_final_loss": recomputed_loss,
            "recomputed_minus_result_final_loss": recomputed_loss
            - calibration["result"]["final_loss"],
            "fit_train_fraction_within_minus_result": (
                metrics["fit_train"]["overall"]["fraction_within_10pct"]
                - calibration["result"].get(
                    "fraction_within_10pct",
                    metrics["fit_train"]["overall"]["fraction_within_10pct"],
                )
            ),
            "mass_conserved_ratio": float(weights.sum() / inputs.design.sum()),
            "max_ratio_to_prior": float((weights / inputs.prior(spec.acs_share)).max()),
        }
        payload = {
            "kind": "calibrate",
            "resumed": calibration["resumed"],
            "release_tag": RELEASE_TAG,
            "spec": dataclasses.asdict(spec),
            "settings": RELEASE_SETTINGS,
            "release_0923_reference": RELEASE_0923_RESULT,
            "result": calibration["result"],
            "design_loss_first_batch_start": calibration["trajectory_first_loss"],
            "options": calibration["options"],
            "batches": calibration["batches"],
            "consistency": consistency,
            "metrics": metrics,
            "inputs": {
                "checkpoint": str(checkpoint),
                "manifest_sha256": sha256(checkpoint / "MANIFEST.json"),
                "input_sha256": inputs.input_sha256,
                "n_train_targets": int(len(train)),
                "n_holdout_targets": 0 if holdout is None else int(len(holdout)),
            },
            "provenance": prov,
            "timing": {
                "total_wall_seconds": time.time() - started,
                "scoring_seconds": time.time() - scoring_started,
            },
            "peak_rss_gb": peak_rss_gb(),
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(jsonable(payload), indent=1) + "\n"
        )
        national = metrics["concentration"]["national"]
        log(
            f"DONE {spec.run_id}: loss={calibration['result']['final_loss']:.6f} "
            f"within10%={metrics['fit_train']['overall']['fraction_within_10pct']:.4f} "
            f"ESS={national['kish_ess']:,.1f} chi2={national['chi_square_distance']:.4f} "
            f"peak RSS {payload['peak_rss_gb']:.2f}GB, {payload['timing']['total_wall_seconds']:.0f}s"
        )
        return payload
    finally:
        log.close()


def evaluate_mode(
    label: str,
    weights_arg: str,
    checkpoint: Path,
    out_root: Path,
    *,
    verify: bool,
    acs_share: float | None = None,
) -> dict:
    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "progress.log")
    started = time.time()
    try:
        inputs = load_inputs(checkpoint, verify=verify, log=log)
        if weights_arg == "design":
            weights = inputs.prior(acs_share)
            source = {"kind": "design", "acs_share": acs_share}
        else:
            path, _, key = weights_arg.partition(":")
            key = key or "weights"
            weights = np.asarray(np.load(path)[key], dtype=np.float64)
            source = {
                "kind": "npz",
                "path": path,
                "key": key,
                "sha256": sha256(Path(path)),
            }
        if weights.shape != (inputs.matrix.shape[1],):
            raise SystemExit(
                f"weights shape {weights.shape} does not match the checkpoint"
            )
        all_rows = np.arange(inputs.matrix.shape[0], dtype=np.int64)
        metrics = score(inputs, weights, all_rows, None, acs_share)
        estimates = estimates_for(inputs, weights)
        metrics["fit_holdout_by_fold"] = {
            str(fold): fit_block(estimates, inputs.meta, rows)
            for fold, rows in inputs.folds.items()
        }
        payload = {
            "kind": "evaluate",
            "label": label,
            "release_tag": RELEASE_TAG,
            "weights_source": source,
            "metrics": metrics,
            "inputs": {
                "checkpoint": str(checkpoint),
                "manifest_sha256": sha256(checkpoint / "MANIFEST.json"),
                "input_sha256": inputs.input_sha256,
            },
            "provenance": provenance(),
            "timing": {"total_wall_seconds": time.time() - started},
            "peak_rss_gb": peak_rss_gb(),
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(jsonable(payload), indent=1) + "\n"
        )
        national = metrics["concentration"]["national"]
        log(
            f"DONE evaluate {label}: loss={metrics['fit_train']['overall']['mean_capped_scaled_error']:.6f} "
            f"within10%={metrics['fit_train']['overall']['fraction_within_10pct']:.4f} "
            f"ESS={national['kish_ess']:,.1f} chi2={national['chi_square_distance']:.4f}"
        )
        return payload
    finally:
        log.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--mode", choices=("calibrate", "evaluate"), default="calibrate"
    )
    parser.add_argument("--spec", help="run spec as a JSON object or a path to one")
    parser.add_argument("--run-id")
    parser.add_argument("--l2-lambda", type=float, default=0.0)
    parser.add_argument("--l2-basis", default="record")
    parser.add_argument("--mass-parametrization", default="projection")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--epoch-batch", type=int, default=400)
    parser.add_argument("--holdout-fold", type=int, default=None)
    parser.add_argument("--acs-share", type=float, default=None)
    parser.add_argument(
        "--weights", default="design", help="evaluate: 'design' or <npz>[:key]"
    )
    parser.add_argument(
        "--label", default="design", help="evaluate: output directory name"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-after-batches",
        type=int,
        default=None,
        help="calibrate: stop (resumably) after this many batches in this process",
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="skip input sha256 checks"
    )
    args = parser.parse_args(argv)

    if args.mode == "evaluate":
        evaluate_mode(
            args.label,
            args.weights,
            args.checkpoint,
            args.out_root,
            verify=not args.no_verify,
            acs_share=args.acs_share,
        )
        return 0
    if args.spec:
        text = Path(args.spec).read_text() if Path(args.spec).exists() else args.spec
        spec = RunSpec.from_mapping(json.loads(text))
    else:
        if not args.run_id:
            parser.error("--run-id or --spec is required in calibrate mode")
        spec = RunSpec(
            run_id=args.run_id,
            l2_lambda=args.l2_lambda,
            l2_basis=args.l2_basis,
            mass_parametrization=args.mass_parametrization,
            epochs=args.epochs,
            epoch_batch=args.epoch_batch,
            holdout_fold=args.holdout_fold,
            acs_share=args.acs_share,
        )
    calibrate_mode(
        spec,
        args.checkpoint,
        args.out_root,
        resume=args.resume,
        verify=not args.no_verify,
        stop_after_batches=args.stop_after_batches,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
