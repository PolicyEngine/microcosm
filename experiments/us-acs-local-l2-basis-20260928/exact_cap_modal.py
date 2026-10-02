"""Measure the softmax cap step at full scale, before and after the exact projection.

Each run is one ``sweep.py`` softmax calibration of the 2026-09-23 release's
surface (1,588,854 households, 4,459 targets, a 5x cap; 800 epochs as two
warm-started batches) on Modal, in one of two variants of the same committed
kernel:

- ``water_fill``: the kernel as committed, whose per-step cap is the exact
  log-space projection ``_project_softmax_log_weights_``.
- ``rounds_32``: the same kernel with that one function replaced by
  microcosm#1078's renormalize/clamp loop (32 rounds at most), copied here.

Both record what the kernel's receipt records, the largest in-loop weight
over its cap, plus this script's own instrumentation: the ratio at every
forward pass, the cap step's rounds and its wall time.

    modal run --detach experiments/us-acs-local-l2-basis-20260928/exact_cap_modal.py::launch
    modal run experiments/us-acs-local-l2-basis-20260928/exact_cap_modal.py::collect
    modal run experiments/us-acs-local-l2-basis-20260928/exact_cap_modal.py::float32

``launch`` reads the checkpoint the sweep already uploaded to the volume
(``sweep.py`` checks every file against ``MANIFEST.json``) and uploads
nothing. ``collect`` copies each finished run's metrics into
``results/exact_cap/`` and writes ``results/exact_cap.md``. Weights stay on
the volume under ``/exact_cap``. ``float32`` runs ``exact_cap_float32.py``
against the volume's checkpoint and the sweep's ``dup_soft_chi_s050_0.03``
weights and writes ``results/exact_cap_float32.json``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = (
    HERE.parents[1]
    if len(HERE.parents) > 1 and (HERE.parents[1] / ".git").exists()
    else None
)
KERNEL = (
    "microcosm-frame",
    "microcosm-graph",
    "microcosm-diagnostics",
    "microcosm-calibrate",
)
VOLUME_ROOT = "/sweep"
OUT_ROOT = f"{VOLUME_ROOT}/exact_cap"
CPUS = 8
VARIANTS = ("water_fill", "rounds_32")
#: The two configurations whose earlier runs ran out of cap rounds on every
#: epoch of their last batch: the recommended penalty at the release's
#: seeding, and the unpenalized solve at an ACS share of 0.9.
CONFIGURATIONS = {
    "soft_chi_s050_0.03": {
        "l2_lambda": 0.03,
        "l2_basis": "chi_square",
        "mass_parametrization": "softmax",
    },
    "soft_chi_s900_0": {
        "l2_lambda": 0.0,
        "l2_basis": "chi_square",
        "mass_parametrization": "softmax",
        "acs_share": 0.9,
    },
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


GIT_SHA = _git("rev-parse", "HEAD") if REPO is not None else "container"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("torch==2.12.0", index_url="https://download.pytorch.org/whl/cpu")
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
    image = (
        image.run_commands(
            "python -m pip install --no-deps "
            + " ".join(f"/opt/kernel/{name}" for name in KERNEL)
        )
        .add_local_file(str(HERE / "sweep.py"), "/opt/sweep/sweep.py")
        .add_local_file(
            str(HERE / "exact_cap_float32.py"), "/opt/sweep/exact_cap_float32.py"
        )
    )

app = modal.App("microcosm-acs-exact-cap", image=image)
volume = modal.Volume.from_name("microcosm-acs-l2-basis-sweep")


def run_ids() -> list[tuple[str, str, str]]:
    return [
        (f"{variant}_{name}", variant, name)
        for name in CONFIGURATIONS
        for variant in VARIANTS
    ]


def _instrument(variant: str) -> dict:
    """Swap in the variant's cap step and record what each step and pass did."""

    import math
    import time

    import torch

    from microcosm.calibrate import solve

    record = {"variant": variant, "seconds": 0.0, "rounds": [], "ratios": []}
    water_fill = solve._project_softmax_log_weights_
    float32_caps: dict[int, torch.Tensor] = {}

    def rounds_32(log_w, total, log_upper):
        """microcosm#1078's cap step (b115745d1), plus the round count.

        Shift so ``exp(log_w)`` sums to ``total``, stop if no record is over
        its float32 log cap, else clamp and go again. Returns the rounds
        taken, negated when all 32 ran out with a record still over.
        """
        if log_upper is not None and id(log_upper) not in float32_caps:
            float32_caps[id(log_upper)] = log_upper.to(torch.float32)
        upper = None if log_upper is None else float32_caps[id(log_upper)]
        for taken in range(1, 33):
            log_w.add_(math.log(total) - float(torch.logsumexp(log_w, dim=0).item()))
            if upper is None or not bool((log_w > upper).any().item()):
                return taken
            log_w.clamp_(max=upper)
        return -32

    step = water_fill if variant == "water_fill" else rounds_32

    def timed(log_w, total, log_upper):
        started = time.perf_counter()
        taken = step(log_w, total, log_upper)
        record["seconds"] += time.perf_counter() - started
        record["rounds"].append(int(taken))
        return taken

    max_cap_ratio = solve._max_cap_ratio

    def recorded(weights, cap):
        ratio = max_cap_ratio(weights, cap)
        record["ratios"].append(ratio)
        return ratio

    solve._project_softmax_log_weights_ = timed
    solve._max_cap_ratio = recorded
    return record


def _cap_step_summary(record: dict) -> dict:
    import numpy as np

    rounds = np.asarray(record["rounds"])
    ratios = np.asarray(record["ratios"], dtype=np.float64)
    values, counts = np.unique(rounds, return_counts=True)
    return {
        "variant": record["variant"],
        "cap_steps": int(rounds.size),
        "cap_step_seconds_total": record["seconds"],
        "cap_step_seconds_mean": record["seconds"] / max(1, rounds.size),
        "rounds_histogram": {
            str(int(v)): int(c) for v, c in zip(values, counts, strict=True)
        },
        "rounds_exhausted_steps": int((rounds < 0).sum()),
        "forward_passes": int(ratios.size),
        "in_loop_cap_ratio_minus_one": {
            "max": float(ratios.max() - 1.0),
            "p99": float(np.quantile(ratios, 0.99) - 1.0),
            "median": float(np.median(ratios) - 1.0),
            "min": float(ratios.min() - 1.0),
            "passes_above_cap": int((ratios > 1.0).sum()),
        },
        "in_loop_cap_ratio_by_pass": [float(r) for r in ratios],
    }


@app.function(
    cpu=CPUS, memory=24576, timeout=3 * 3600, volumes={VOLUME_ROOT: volume}, retries=0
)
def run_variant(run_id: str, variant: str, name: str) -> dict:
    import sys

    import torch

    torch.set_num_threads(CPUS)
    sys.path.insert(0, "/opt/sweep")
    import sweep

    volume.reload()
    out_root = Path(OUT_ROOT) / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    record = _instrument(variant)
    payload = sweep.calibrate_mode(
        sweep.RunSpec.from_mapping({"run_id": run_id, **CONFIGURATIONS[name]}),
        Path(VOLUME_ROOT) / "checkpoint",
        out_root,
        resume=False,
        verify=True,
    )
    cap_step = _cap_step_summary(record)
    (out_root / run_id / "cap_step.json").write_text(
        json.dumps(cap_step, indent=1) + "\n"
    )
    volume.commit()
    return {
        "run_id": run_id,
        "final_loss": payload["result"]["final_loss"],
        "kish_ess": payload["metrics"]["concentration"]["national"]["kish_ess"],
        "wall_seconds": payload["timing"]["total_wall_seconds"],
        **{k: v for k, v in cap_step.items() if k != "in_loop_cap_ratio_by_pass"},
    }


@app.local_entrypoint()
def launch() -> None:
    kernel = [f"packages/{name}/src" for name in KERNEL]
    if _git("status", "--porcelain", "--", *kernel):
        raise SystemExit("kernel sources differ from HEAD; commit before launching")
    print(f"kernel HEAD {GIT_SHA}")
    calls = []
    for run_id, variant, name in run_ids():
        calls.append(run_variant.spawn(run_id, variant, name))
        print(run_id, calls[-1].object_id, flush=True)
    # The runs outlive this process under --detach; waiting only prints them.
    for call in calls:
        print(json.dumps(call.get()), flush=True)


@app.function(cpu=CPUS, memory=8192, timeout=1800, volumes={VOLUME_ROOT: volume})
def measure_float32() -> dict:
    import sys

    import torch

    torch.set_num_threads(CPUS)
    sys.path.insert(0, "/opt/sweep")
    import exact_cap_float32

    return exact_cap_float32.measure(
        Path(VOLUME_ROOT) / "checkpoint",
        Path(VOLUME_ROOT) / "runs" / "dup_soft_chi_s050_0.03" / "weights.npz",
    )


@app.local_entrypoint()
def float32() -> None:
    import sys

    kernel = [f"packages/{name}/src" for name in KERNEL]
    if _git("status", "--porcelain", "--", *kernel):
        raise SystemExit("kernel sources differ from HEAD; commit before launching")
    out = measure_float32.remote()
    out["kernel_head"] = GIT_SHA
    sys.path.insert(0, str(HERE))
    import exact_cap_float32

    exact_cap_float32.write(out)


def _read(path: str) -> bytes | None:
    try:
        return b"".join(volume.read_file(path))
    except FileNotFoundError:
        return None


@app.local_entrypoint()
def collect() -> None:
    results = HERE / "results" / "exact_cap"
    results.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_id, variant, name in run_ids():
        metrics = _read(f"/exact_cap/runs/{run_id}/metrics.json")
        cap_step = _read(f"/exact_cap/runs/{run_id}/cap_step.json")
        if metrics is None or cap_step is None:
            print(f"{run_id}: not finished")
            continue
        (results / f"{run_id}.json").write_bytes(metrics)
        (results / f"{run_id}.cap_step.json").write_bytes(cap_step)
        rows.append((name, variant, json.loads(metrics), json.loads(cap_step)))
    lines = [
        "| Configuration | Cap step | Largest in-loop weight over its cap, minus 1 "
        "(max / median over passes) | Steps out of rounds | Rounds per step "
        "| Cap step ms | s/epoch | Receipt (batch 1 / 2) | Final loss | National ESS |",
        "|---|---|---:|---:|---|---:|---:|---|---:|---:|",
    ]
    for name, variant, metrics, cap_step in rows:
        ratio = cap_step["in_loop_cap_ratio_minus_one"]
        batches = metrics["batches"]
        receipts = " / ".join(
            f"{(b.get('iterate_selection_receipt') or {}).get('softmax_in_loop_max_cap_ratio', float('nan')) - 1.0:.2e}"
            for b in batches
        )
        per_epoch = " / ".join(f"{b['seconds_per_epoch']:.2f}" for b in batches)
        lines.append(
            f"| `{name}` | `{variant}` | {ratio['max']:.2e} / {ratio['median']:.2e} "
            f"| {cap_step['rounds_exhausted_steps']} of {cap_step['cap_steps']} "
            f"| {json.dumps(cap_step['rounds_histogram'])} "
            f"| {1000 * cap_step['cap_step_seconds_mean']:.1f} | {per_epoch} "
            f"| {receipts} | {metrics['result']['final_loss']:.6f} "
            f"| {metrics['metrics']['concentration']['national']['kish_ess']:,.0f} |"
        )
    (HERE / "results" / "exact_cap.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"collected {len(rows)} of {len(run_ids())} runs")
