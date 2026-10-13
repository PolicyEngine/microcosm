"""Run the five #1228 arms of the Route A re-solve on Modal.

Each arm is one ``resolve.py`` calibration of the release's 5,694-target
national_state surface (352,932 households) in its own container. The kernel
packages are installed from this worktree's committed tree; the launch refuses
a tree whose kernel sources or ``resolve.py`` differ from HEAD, and every run
records that HEAD. A finished arm is reused only when its HEAD, its inputs'
``MANIFEST.json`` digest and its ``resolve.py`` digest all match.

    uv run --no-sync modal run --detach experiments/1228-net-income-less-loss/modal_resolve.py::launch --inputs DIR
    uv run --no-sync modal run experiments/1228-net-income-less-loss/modal_resolve.py::collect --into DIR

``launch`` uploads ``prepare.py``'s outputs once and fans out the arms.
``collect`` copies each finished arm's ``metrics.json`` and ``weights.npz``.
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
UPLOAD = (
    "MANIFEST.json",
    "target_matrix.npz",
    "concept_rows.npz",
    "targets_meta.parquet",
    "weights.npz",
    "rebase_rows.npz",
    "rebase_meta.parquet",
)
KERNEL = (
    "microcosm-frame",
    "microcosm-graph",
    "microcosm-diagnostics",
    "microcosm-calibrate",
)
ARMS = (
    "release",
    "concept_exact",
    "concept_exact_no_business_count",
    "drop",
    "rebase_positive_part",
)
VOLUME_ROOT = "/resolve"
CPUS = 16


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
        + [f"packages/{name}/src" for name in KERNEL]
        + [str(HERE.relative_to(REPO) / "resolve.py")],
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
    ).add_local_file(str(HERE / "resolve.py"), "/opt/resolve/resolve.py")

app = modal.App("microcosm-1228-net-income-resolve", image=image)
volume = modal.Volume.from_name(
    "microcosm-1228-net-income-resolve", create_if_missing=True
)


@app.function(
    cpu=CPUS,
    memory=24576,
    timeout=6 * 3600,
    volumes={VOLUME_ROOT: volume},
    retries=1,
)
def run_arm(arm: str) -> dict:
    import sys

    import torch

    torch.set_num_threads(CPUS)
    volume.reload()
    out_root = Path(VOLUME_ROOT) / "runs"
    finished = out_root / arm / "metrics.json"
    if finished.exists():
        import hashlib

        payload = json.loads(finished.read_text())
        identity = {
            "git_sha": GIT_SHA,
            "inputs_manifest_sha256": hashlib.sha256(
                Path(f"{VOLUME_ROOT}/inputs/MANIFEST.json").read_bytes()
            ).hexdigest(),
            "resolve_sha256": hashlib.sha256(
                Path("/opt/resolve/resolve.py").read_bytes()
            ).hexdigest(),
        }
        if all(payload.get(key) == value for key, value in identity.items()):
            return {"arm": arm, "cached": True, "final_loss": payload["final_loss"]}
    sys.argv = [
        "resolve.py",
        "--inputs",
        f"{VOLUME_ROOT}/inputs",
        "--out",
        str(out_root),
        "--arm",
        arm,
    ]
    sys.path.insert(0, "/opt/resolve")
    import resolve

    resolve.main()
    volume.commit()
    payload = json.loads(finished.read_text())
    return {
        "arm": arm,
        "final_loss": payload["final_loss"],
        "seconds": payload["seconds"],
    }


@app.local_entrypoint()
def launch(inputs: str) -> None:
    if not _kernel_clean():
        raise SystemExit("kernel sources differ from HEAD; commit before launching")
    print(f"kernel HEAD {GIT_SHA}")
    with volume.batch_upload(force=True) as batch:
        for name in UPLOAD:
            batch.put_file(str(Path(inputs) / name), f"/inputs/{name}")
    for result in run_arm.map(ARMS, return_exceptions=True):
        print(json.dumps(result, default=str), flush=True)


@app.local_entrypoint()
def collect(into: str) -> None:
    target = Path(into)
    for arm in ARMS:
        for name in ("metrics.json", "weights.npz"):
            remote = f"/runs/{arm}/{name}"
            try:
                data = b"".join(volume.read_file(remote))
            except FileNotFoundError:
                print(f"missing {remote}")
                continue
            path = target / arm / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            print(f"wrote {path} ({len(data)} bytes)")
