"""Run the microcosm#1073 education-debt experiments on Modal instead of local RAM.

Each run clones a pushed commit of PolicyEngine/microcosm, syncs that tree's
uv.lock, fetches the inputs and verifies each sha256, runs the requested steps
in order, and saves every receipt the steps write plus a run log to the
`microcosm-1073-cache` volume under results/<run_id>/. `collect` copies a
run's files back into this checkout. Inputs:

- the published host release `populace-us-2024-spm-20260915` from Hugging Face
  (policyengine/populace-us at a pinned revision);
- the Federal Reserve's SCF 2025 public zips (summary extract and full file).

    export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
    modal run experiments/modal_1073.py::preflight --commit <sha>
    modal run --detach experiments/modal_1073.py::main --commit <sha> --steps tests,shift,explore
    modal run experiments/modal_1073.py::collect --run-id <run_id>

Steps (see STEPS; each runs one or more commands, stopping at the first that
fails):

- ``generate``: regenerate the US spec bundle mirrors (``spec/*.yaml``,
  ``country_package.json``, ``engine_abi.lock.json``) with
  ``tools/generate_us_bundle_from_constants.py`` after a hand edit of
  ``source_stages.json``;
- ``tests``: the stage's unit tests;
- ``spec``: the engine-free and engine-contract tests that read the spec
  bundle, the stored-input register and the SCF stages, plus the release
  tool's education-debt tests;
- ``hostfit``, ``shift``, ``explore``: the #1073 diagnostics;
- ``receipt``: the fit receipt, labelled with the run id.

After the steps the run saves every tracked file they changed
(``git status --porcelain``) and every untracked file matching OUTPUT_GLOBS,
so ``collect`` brings regenerated mirrors back as well as receipts. This is an
experiment runner, not the registered stage runner in tools/modal_us_stage.py;
it writes nothing to the Hub and publishes nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import modal

REPO_URL = "https://github.com/PolicyEngine/microcosm.git"
HOST = {
    "repo_id": "policyengine/populace-us",
    "filename": "populace_us_2024.h5",
    "revision": "8ab57ffc2ca41d8af631ff62ebbce95e966299f3",
    "sha256": "6496cc4393d4d3c6574f76eca231de5898c803b9067645591fd5c4d3e65aee84",
}
# zip name -> (url, zip sha256, member, member sha256)
SCF_ZIPS = {
    "scfp2025s.zip": (
        "https://www.federalreserve.gov/econres/files/scfp2025s.zip",
        "32dcf86be48b20fb5fd9e659f79fa22a4d9cccf456e44b6ca9b1329e9869bbeb",
        "rscfp2025.dta",
        "454a287f188bf13f94efc09820fa878482d897db1d2a16ac63bec14eb580d2dd",
    ),
    "scf2025s.zip": (
        "https://www.federalreserve.gov/econres/files/scf2025s.zip",
        "c4010bd5decaa13d5d027862ed4b6c41b4106925089290f879749f8f5a467741",
        "p25i6.dta",
        "38926275dfb8f00ea843b80ced7a9412ad4653d39c7fab8fe24c3bdf912902ad",
    ),
}
_TESTS = "packages/microcosm-build/tests"
#: step -> commands. A part "glob:<pattern>" expands to the sorted matches.
STEPS: dict[str, list[list[str]]] = {
    "generate": [
        [
            "{python}",
            "tools/generate_us_bundle_from_constants.py",
            "--engine-lock-only",
        ],
        ["{python}", "tools/generate_us_bundle_from_constants.py"],
    ],
    "tests": [
        [
            "{python}",
            "-m",
            "pytest",
            "-q",
            f"{_TESTS}/engine_free/us/test_us_scf_education_debt.py",
        ]
    ],
    "spec": [
        [
            "{python}",
            "-m",
            "pytest",
            "-q",
            f"glob:{_TESTS}/engine_free/shared/test_spec_engine_*.py",
            f"{_TESTS}/engine_free/us/test_us_bundle_core_contracts.py",
            f"{_TESTS}/engine_free/us/test_us_spec_bundle.py",
            f"{_TESTS}/engine_free/us/test_us_stored_input_register.py",
            f"{_TESTS}/engine_free/us/test_us_scf_auto_loans.py",
            f"{_TESTS}/engine_free/us/test_us_scf_wealth.py",
            f"{_TESTS}/engine_free/us/test_us_plan.py",
            f"{_TESTS}/engine_free/us/test_us_spine_blindness.py",
        ],
        [
            "{python}",
            "-m",
            "pytest",
            "-q",
            f"{_TESTS}/engine_free/us/test_us_fiscal_refresh_builder.py",
            "-k",
            "scf_education_debt",
        ],
        [
            "{python}",
            "-m",
            "pytest",
            "-q",
            f"glob:{_TESTS}/engine_contract/us/test_spec_engine_*.py",
            f"{_TESTS}/engine_contract/us/test_us_bundle_core_contracts.py",
            f"{_TESTS}/engine_contract/us/test_us_spec_bundle.py",
            f"{_TESTS}/engine_contract/us/test_us_stored_input_register.py",
            f"{_TESTS}/engine_contract/us/test_us_scf_wealth.py",
            f"{_TESTS}/engine_contract/us/test_imputation_lineage_spec.py",
            f"{_TESTS}/engine_contract/us/test_release_input_coverage.py",
        ],
    ],
    "hostfit": [
        [
            "{python}",
            "experiments/diagnose_1073_host_fit.py",
            "--h5",
            "{h5}",
            "--scf-cache-dir",
            "{scf}",
        ]
    ],
    "shift": [
        [
            "{python}",
            "experiments/diagnose_1073_predictor_shift.py",
            "--h5",
            "{h5}",
            "--scf-cache-dir",
            "{scf}",
        ]
    ],
    "explore": [
        [
            "{python}",
            "experiments/explore_1073_recipient_definitions.py",
            "--h5",
            "{h5}",
            "--scf-cache-dir",
            "{scf}",
        ]
    ],
    "receipt": [
        [
            "{python}",
            "experiments/build_1073_scf_education_debt_receipt.py",
            "--h5",
            "{h5}",
            "--scf-zip-dir",
            "{zips}",
            "--scf-cache-dir",
            "{scf}",
            "--run",
            "{run_id}",
        ]
    ],
}
OUTPUT_GLOBS = ("experiments/receipts/1073-*.json", "experiments/1073-*.md")
CPU = 8.0
MEMORY_MB = 32768

image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("git")
    .pip_install("uv==0.11.7", "huggingface_hub")
)
app = modal.App("microcosm-1073-experiments")
cache = modal.Volume.from_name("microcosm-1073-cache", create_if_missing=True)
CACHE = Path("/cache")
WORK = Path("/work/microcosm")
VENV = Path("/opt/venv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path, sha256: str) -> bool:
    return path.exists() and _sha256(path) == sha256


def _clean_env(**overrides: str) -> dict[str, str]:
    """The container environment without the Modal runtime's Python path.

    Modal puts its own client and dependencies (under /pkg) on PYTHONPATH. A
    subprocess that inherits it sees those packages beside the venv's, and the
    build's worker-identity check refuses the duplicates (e.g. idna).
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}
    }
    env.update(overrides)
    return env


def _checkout(commit: str) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    run = lambda *args: subprocess.run(args, cwd=WORK, check=True)  # noqa: E731
    run("git", "init", "-q")
    run("git", "remote", "add", "origin", REPO_URL)
    run("git", "fetch", "-q", "--depth", "1", "origin", commit)
    run("git", "checkout", "-q", "--detach", "FETCH_HEAD")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=WORK,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != commit:
        raise RuntimeError(f"checked out {head}, expected {commit}")
    env = _clean_env(
        **{
            "UV_PROJECT_ENVIRONMENT": str(VENV),
            # A Modal volume cannot host uv's cache: uv persists temporary files
            # by atomic rename, which the volume refuses.
            "UV_CACHE_DIR": "/root/.cache/uv",
            "UV_LINK_MODE": "copy",
        }
    )
    subprocess.run(
        ["uv", "sync", "--all-packages", "--extra", "us", "--frozen"],
        cwd=WORK,
        env=env,
        check=True,
    )


def _inputs() -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    host_dir = CACHE / "host" / HOST["revision"]
    host = host_dir / HOST["filename"]
    if not _verified(host, HOST["sha256"]):
        hf_hub_download(
            repo_id=HOST["repo_id"],
            filename=HOST["filename"],
            revision=HOST["revision"],
            repo_type="dataset",
            local_dir=host_dir,
        )
        if not _verified(host, HOST["sha256"]):
            raise RuntimeError("host H5 sha256 mismatch")
    zips = CACHE / "scf" / "zips"
    scf = CACHE / "scf" / "dta"
    zips.mkdir(parents=True, exist_ok=True)
    scf.mkdir(parents=True, exist_ok=True)
    for name, (url, zip_sha, member, member_sha) in SCF_ZIPS.items():
        archive = zips / name
        if not _verified(archive, zip_sha):
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=600) as response:
                archive.write_bytes(response.read())
            if not _verified(archive, zip_sha):
                raise RuntimeError(f"{name} sha256 mismatch")
        target = scf / member
        if not _verified(target, member_sha):
            with zipfile.ZipFile(archive) as bundle:
                target.write_bytes(bundle.read(member))
            if not _verified(target, member_sha):
                raise RuntimeError(f"{member} sha256 mismatch")
    cache.commit()
    return {"h5": host, "zips": zips, "scf": scf}


def _expand(command: list[str], fields: dict[str, str]) -> list[str]:
    parts: list[str] = []
    for part in command:
        if part.startswith("glob:"):
            matches = sorted(WORK.glob(part.removeprefix("glob:")))
            if not matches:
                raise RuntimeError(f"{part} matched nothing")
            parts.extend(str(path.relative_to(WORK)) for path in matches)
        else:
            parts.append(part.format(**fields))
    return parts


def _save_changed_files(target_root: Path) -> list[str]:
    """Copy tracked files the steps changed, and new outputs, under target_root."""

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=WORK,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    saved = []
    for line in status:
        code, rel = line[:2], line[3:]
        path = WORK / rel
        untracked = code == "??"
        if untracked and not any(path.match(p) for p in OUTPUT_GLOBS):
            continue
        if not path.is_file():
            continue
        target = target_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        saved.append(f"{code} {rel}")
    return saved


@app.function(
    image=image, cpu=2.0, memory=8192, timeout=3600, volumes={"/cache": cache}
)
def preflight_remote(commit: str) -> dict:
    """Clone, sync and stage the inputs without running a step."""
    started = time.time()
    _checkout(commit)
    paths = _inputs()
    probe = subprocess.run(
        [
            str(VENV / "bin" / "python"),
            "-c",
            "import microcosm.build.us_runtime.scf_education_debt as m; print(m.__file__)",
        ],
        cwd=WORK,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit,
        "inputs": {key: str(path) for key, path in paths.items()},
        "import_ok": probe.returncode == 0,
        "import_output": (probe.stdout + probe.stderr)[-2000:],
        "seconds": round(time.time() - started, 1),
    }


@app.function(
    image=image, cpu=CPU, memory=MEMORY_MB, timeout=4 * 3600, volumes={"/cache": cache}
)
def run_steps(
    commit: str, steps: list[str], run_id: str, script_text: str = ""
) -> dict:
    import resource  # Unix-only; the local entrypoint runs on Windows too

    started = datetime.now(UTC)
    _checkout(commit)
    paths = _inputs()
    fields = {key: str(path) for key, path in paths.items()} | {
        "python": str(VENV / "bin" / "python"),
        "run_id": run_id,
    }
    threads = str(int(CPU))
    env = _clean_env(
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS=threads,
        MKL_NUM_THREADS=threads,
        OPENBLAS_NUM_THREADS=threads,
    )
    out = CACHE / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    log = {
        "run_id": run_id,
        "commit": commit,
        "steps": [],
        "cpu": CPU,
        "memory_mb": MEMORY_MB,
        "inputs": {"host": HOST, "scf": {k: v[1] for k, v in SCF_ZIPS.items()}},
        "started_at": started.isoformat(),
    }
    failed = False
    for step in steps:
        if step == "script":
            # An uncommitted local probe passed by --script; it runs in the
            # clone like any step, and the run log records its text.
            (WORK / "_modal_probe.py").write_text(script_text, encoding="utf-8")
            log["script_sha256"] = hashlib.sha256(script_text.encode()).hexdigest()
            commands = [[fields["python"], "_modal_probe.py"]]
        else:
            commands = [_expand(command, fields) for command in STEPS[step]]
        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        t0 = time.time()
        returncodes = []
        with (out / f"{step}.log").open("w", encoding="utf-8") as handle:
            for command in commands:
                handle.write("$ " + " ".join(command) + "\n")
                handle.flush()
                result = subprocess.run(
                    command,
                    cwd=WORK,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                returncodes.append(result.returncode)
                if result.returncode != 0:
                    break
        peak_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        log["steps"].append(
            {
                "step": step,
                "commands": commands,
                "returncodes": returncodes,
                "returncode": next((code for code in returncodes if code), 0),
                "seconds": round(time.time() - t0, 1),
                "peak_child_rss_gb": round(max(peak_kb, before) / 1024 / 1024, 2),
            }
        )
        cache.commit()
        if any(returncodes):
            failed = True
            break
    log["failed"] = failed
    log["changed_files"] = _save_changed_files(out / "files")
    log["finished_at"] = datetime.now(UTC).isoformat()
    (out / "run.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    cache.commit()
    return log


@app.function(image=image, volumes={"/cache": cache})
def fetch_results(run_id: str) -> dict[str, bytes]:
    root = CACHE / "results" / run_id
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


def _check_commit(commit: str) -> None:
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise SystemExit("--commit must be a full 40-hex sha that is pushed to GitHub")


@app.local_entrypoint()
def preflight(commit: str) -> None:
    _check_commit(commit)
    print(json.dumps(preflight_remote.remote(commit), indent=2))


@app.local_entrypoint()
def main(commit: str, steps: str = "tests,shift,explore", script: str = "") -> None:
    _check_commit(commit)
    names = [s for s in steps.split(",") if s]
    script_text = Path(script).read_text(encoding="utf-8") if script else ""
    if script and "script" not in names:
        names.append("script")
    unknown = sorted(set(names) - set(STEPS) - {"script"})
    if unknown:
        raise SystemExit(f"unknown steps: {unknown}")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + commit[:9]
    print(f"run_id={run_id}")
    print(json.dumps(run_steps.remote(commit, names, run_id, script_text), indent=2))


@app.local_entrypoint()
def collect(run_id: str, logs_dir: str = "") -> None:
    """Write a run's changed files into this checkout and its logs to logs_dir."""
    repo = Path(__file__).resolve().parents[1]
    logs = (
        Path(logs_dir)
        if logs_dir
        else Path.home() / ".cache" / "microcosm" / "modal-1073" / run_id
    )
    logs.mkdir(parents=True, exist_ok=True)
    for rel, data in fetch_results.remote(run_id).items():
        if rel.startswith("files/"):
            target = repo / rel.removeprefix("files/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            print("file", target.relative_to(repo))
        else:
            (logs / rel).write_bytes(data)
            print("log", logs / rel)
