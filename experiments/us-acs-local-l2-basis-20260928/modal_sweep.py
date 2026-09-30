"""Run the ACS local L2-basis sweep grid on Modal.

Each grid point is one ``sweep.py`` calibration of the 2026-09-23 release's
4,459-target surface (1,588,854 households; ~8 GB peak) in its own container.
The kernel packages are installed from this worktree's committed tree; the
launch refuses a tree whose kernel sources differ from HEAD, and every run
records that HEAD.

    modal run --detach experiments/us-acs-local-l2-basis-20260928/modal_sweep.py::launch
    modal run experiments/us-acs-local-l2-basis-20260928/modal_sweep.py::collect

``launch`` uploads the verified checkpoint once (sha256-checked in the
container against ``MANIFEST.json``), runs ``release_repro`` first and stops
unless it reproduces the published release, then fans out the rest.
``collect`` copies every finished run's ``metrics.json`` into
``results/runs/`` and its ``weights.npz`` into the local artifacts directory.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
#: The worktree root when this file runs from the repository; ``None`` inside
#: a container, where Modal mounts the launcher alone at /root.
REPO = (
    HERE.parents[1]
    if len(HERE.parents) > 1 and (HERE.parents[1] / ".git").exists()
    else None
)
CHECKPOINT = Path(
    "/Users/maxghenis/PolicyEngine/_build_artifacts/acs-local-l2-basis-20260928/checkpoint"
)
LOCAL_RUNS = CHECKPOINT.parent / "runs"
UPLOAD = (
    "MANIFEST.json",
    "target_matrix.npz",
    "households.parquet",
    "targets_meta.parquet",
    "holdout_folds.npz",
)
KERNEL = (
    "microcosm-frame",
    "microcosm-graph",
    "microcosm-diagnostics",
    "microcosm-calibrate",
)
VOLUME_ROOT = "/sweep"
CPUS = 8


def _head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _kernel_clean() -> bool:
    status = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--"]
        + [f"packages/{name}/src" for name in KERNEL],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return not status


GIT_SHA = _head() if REPO is not None else "container"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        "torch==2.12.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .uv_pip_install(
        "numpy==2.4.6",
        "scipy==1.17.1",
        "pandas==3.0.3",
        "pyarrow==25.0.0",
        "pydantic>=2.10,<3",
    )
    .env(
        {
            "OMP_NUM_THREADS": str(CPUS),
            "MKL_NUM_THREADS": str(CPUS),
            "OPENBLAS_NUM_THREADS": str(CPUS),
            "MICROCOSM_GIT_SHA": GIT_SHA,
            "MICROCOSM_REPO": "/opt/none",
        }
    )
)
if REPO is not None:
    for name in KERNEL:
        image = image.add_local_dir(
            str(REPO / "packages" / name), f"/opt/kernel/{name}", copy=True
        )
    image = image.run_commands(
        "python -m pip install --no-deps "
        + " ".join(f"/opt/kernel/{name}" for name in KERNEL)
    ).add_local_file(str(HERE / "sweep.py"), "/opt/sweep/sweep.py")

app = modal.App("microcosm-acs-l2-basis-sweep", image=image)
volume = modal.Volume.from_name("microcosm-acs-l2-basis-sweep", create_if_missing=True)


def grid() -> list[dict]:
    """The sweep's run specs; ``release_repro`` must come first."""

    specs = [
        {"run_id": "release_repro", "mass_parametrization": "projection"},
    ]
    lambdas = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    for share in (None, 0.7, 0.9, 0.964):
        tag = "s050" if share is None else f"s{round(share * 1000):03d}"
        for lam in lambdas:
            specs.append(
                {
                    "run_id": f"soft_chi_{tag}_{lam:g}",
                    "l2_lambda": lam,
                    "l2_basis": "chi_square",
                    "mass_parametrization": "softmax",
                    "acs_share": share,
                }
            )
    for lam in (0.1, 1.0):
        specs.append(
            {
                "run_id": f"proj_chi_s050_{lam:g}",
                "l2_lambda": lam,
                "l2_basis": "chi_square",
                "mass_parametrization": "projection",
            }
        )
    specs.append(
        {
            "run_id": "proj_rec_s050_0.1",
            "l2_lambda": 0.1,
            "l2_basis": "record",
            "mass_parametrization": "projection",
        }
    )
    for share in (None, 0.964):
        tag = "s050" if share is None else "s964"
        for lam in (0.0, 0.1, 1.0):
            specs.append(
                {
                    "run_id": f"hold_soft_chi_{tag}_{lam:g}",
                    "l2_lambda": lam,
                    "l2_basis": "chi_square",
                    "mass_parametrization": "softmax",
                    "acs_share": share,
                    "holdout_fold": 0,
                }
            )
    specs.append(
        {
            "run_id": "hold_proj_s050_0",
            "mass_parametrization": "projection",
            "holdout_fold": 0,
        }
    )
    # Second pass (after the first frontier): holdout at the knee lambdas and
    # the intermediate priors, and projection at the recommended lambda.
    for share, lams in (
        (None, (0.01, 0.03)),
        (0.7, (0.0, 0.01, 0.1)),
        (0.9, (0.0, 0.01, 0.1)),
    ):
        tag = "s050" if share is None else f"s{round(share * 1000):03d}"
        for lam in lams:
            specs.append(
                {
                    "run_id": f"hold_soft_chi_{tag}_{lam:g}",
                    "l2_lambda": lam,
                    "l2_basis": "chi_square",
                    "mass_parametrization": "softmax",
                    "acs_share": share,
                    "holdout_fold": 0,
                }
            )
    # Third pass: projection at the held-out optimum, and a second holdout
    # fold for the configurations the recommendation compares.
    specs.append(
        {
            "run_id": "proj_chi_s050_0.03",
            "l2_lambda": 0.03,
            "l2_basis": "chi_square",
            "mass_parametrization": "projection",
        }
    )
    for fold in (0, 1):
        specs.append(
            {
                "run_id": f"hold{fold}_proj_chi_s050_0.03".replace("hold0_", "hold_"),
                "l2_lambda": 0.03,
                "l2_basis": "chi_square",
                "mass_parametrization": "projection",
                "holdout_fold": fold,
            }
        )
    specs.append(
        {
            "run_id": "hold1_proj_s050_0",
            "mass_parametrization": "projection",
            "holdout_fold": 1,
        }
    )
    for share, lam in (
        (None, 0.03),
        (None, 0.1),
        (0.7, 0.01),
        (0.9, 0.0),
        (0.964, 0.0),
    ):
        tag = "s050" if share is None else f"s{round(share * 1000):03d}"
        specs.append(
            {
                "run_id": f"hold1_soft_chi_{tag}_{lam:g}",
                "l2_lambda": lam,
                "l2_basis": "chi_square",
                "mass_parametrization": "softmax",
                "acs_share": share,
                "holdout_fold": 1,
            }
        )
    specs.append(
        {
            "run_id": "hold_proj_chi_s050_0.1",
            "l2_lambda": 0.1,
            "l2_basis": "chi_square",
            "mass_parametrization": "projection",
            "holdout_fold": 0,
        }
    )
    return specs


@app.function(
    cpu=CPUS,
    memory=24576,
    timeout=4 * 3600,
    volumes={VOLUME_ROOT: volume},
    retries=1,
)
def run_point(spec: dict) -> dict:
    import sys

    import torch

    torch.set_num_threads(CPUS)
    sys.path.insert(0, "/opt/sweep")
    import sweep

    volume.reload()
    out_root = Path(VOLUME_ROOT) / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    finished = out_root / spec["run_id"] / "metrics.json"
    if finished.exists():
        payload = json.loads(finished.read_text())
        if payload.get("spec", {}).get("run_id") == spec["run_id"]:
            return {"run_id": spec["run_id"], "cached": True}
    payload = sweep.calibrate_mode(
        sweep.RunSpec.from_mapping(spec),
        Path(VOLUME_ROOT) / "checkpoint",
        out_root,
        resume=True,
        verify=True,
    )
    volume.commit()
    national = payload["metrics"]["concentration"]["national"]
    return {
        "run_id": spec["run_id"],
        "final_loss": payload["result"]["final_loss"],
        "fraction_within_10pct": payload["metrics"]["fit_train"]["overall"][
            "fraction_within_10pct"
        ],
        "kish_ess": national["kish_ess"],
        "chi_square_distance": national["chi_square_distance"],
        "wall_seconds": payload["timing"]["total_wall_seconds"],
    }


def _upload_checkpoint() -> None:
    with volume.batch_upload(force=True) as batch:
        for name in UPLOAD:
            batch.put_file(str(CHECKPOINT / name), f"/checkpoint/{name}")


@app.local_entrypoint()
def launch(skip_repro: bool = False) -> None:
    if not _kernel_clean():
        raise SystemExit("kernel sources differ from HEAD; commit before launching")
    print(f"kernel HEAD {GIT_SHA}")
    _upload_checkpoint()
    specs = grid()
    if not skip_repro:
        repro = run_point.remote(specs[0])
        print("release_repro:", json.dumps(repro))
        if not repro.get("cached"):
            reference = {"kish_ess": 13631.3, "final_loss": 0.015491}
            for key, value in reference.items():
                if abs(repro[key] / value - 1.0) > 0.05:
                    raise SystemExit(
                        f"release_repro {key}={repro[key]} is not within 5% of the "
                        f"release's {value}; not launching the grid"
                    )
            if abs(repro["fraction_within_10pct"] - 0.9758) > 0.01:
                raise SystemExit("release_repro within-10% is not within 1 pp")
    for result in run_point.map(
        specs[1:], return_exceptions=True, wrap_returned_exceptions=False
    ):
        print(json.dumps(result, default=str), flush=True)


@app.local_entrypoint()
def collect() -> None:
    results = HERE / "results" / "runs"
    results.mkdir(parents=True, exist_ok=True)
    collected = 0
    for spec in grid():
        run_id = spec["run_id"]
        local = LOCAL_RUNS / run_id
        local.mkdir(parents=True, exist_ok=True)
        try:
            metrics = b"".join(volume.read_file(f"/runs/{run_id}/metrics.json"))
        except FileNotFoundError:
            print(f"{run_id}: not finished")
            continue
        (results / f"{run_id}.json").write_bytes(metrics)
        (local / "metrics.json").write_bytes(metrics)
        with open(local / "weights.npz", "wb") as handle:
            for chunk in volume.read_file(f"/runs/{run_id}/weights.npz"):
                handle.write(chunk)
        collected += 1
    print(f"collected {collected} of {len(grid())} runs")
