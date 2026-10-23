"""Run the microcosm#1074 FSA-target experiments on Modal instead of local RAM.

Each run clones a pushed commit of PolicyEngine/microcosm, syncs that tree's
uv.lock, fetches the inputs and verifies each sha256, runs the requested steps
in order, and saves every receipt the steps write plus a run log to the
`microcosm-1074-cache` volume under results/<run_id>/. `collect` copies a
run's files back into this checkout. Inputs:

- the published host release `populace-us-2024-spm-20260915` from Hugging Face
  (policyengine/populace-us at a pinned revision);
- the Federal Reserve's SCF 2025 public zips (summary extract and full file);
- for ``fit``: a candidate Chronicle feed, built in the container and cached
  on the volume. It is the pinned US feed (rebuilt byte for byte at Chronicle
  ``51aa40f`` with ``tools/build_us_chronicle_feed.py`` and checked against
  ``us/chronicle_feed.json``) plus the six FSA packages of
  PolicyEngine/chronicle#347 (``chronicle build-bundle --year 2026`` at the
  merge commit ``acbedb5`` on Chronicle main, which reads the raw files from
  R2). The candidate is a reviewed diagnostic input only; no pin in the
  repository names it.

    export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
    modal run experiments/modal_1074.py::preflight --commit <sha>
    modal run --detach experiments/modal_1074.py::launch --commit <sha> --steps tests,fit
    modal run experiments/modal_1074.py::collect --run-id <run_id>

Steps (see STEPS; each runs one or more commands, stopping at the first that
fails):

- ``tests``: the FSA target, fiscal-target, ledger, parity, chronicle-feed,
  label and release-tool filter tests (engine-free);
- ``contract``: the engine-contract fiscal-target and release-input tests;
- ``spec``: the bundle generator and the spec-engine coverage attestation in
  ``--check`` mode, and the engine-free spec-engine tests, which fail if the
  change moved a spec identity;
- ``feed``: build (or verify the cached) candidate feed only;
- ``fit``: the calibration fit and receipt
  (``experiments/fit_1074_fsa_calibration.py``), labelled with the run id.

After the steps the run saves every tracked file they changed
(``git status --porcelain``) and every untracked file matching OUTPUT_GLOBS,
so ``collect`` brings receipts back. This is an experiment runner, not the
registered stage runner in tools/modal_us_stage.py; it writes nothing to the
Hub and publishes nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
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
#: The pinned US feed (us/chronicle_feed.json) and where it is rebuilt from.
PINNED_FEED = {
    "repo": "https://github.com/PolicyEngine/chronicle.git",
    "commit": "51aa40fd28e2ca81937752515e773d123fecfb71",
    "facts_sha256": "965a29ac9458edb58a05184917ee4a19772fd3dc7b4588db6eb6da5803641aac",
    "rows": 40178,
}
#: PolicyEngine/chronicle#347: the six FSA portfolio packages.
FSA_PACKAGES = {
    "repo": "https://github.com/PolicyEngine/chronicle.git",
    # The merge commit of chronicle#347 on Chronicle main (2026-10-22). Its
    # packages/ed_fsa tree is the PR head's (81e3f98), whose FSA facts are the
    # first draft's (4448a7c), the export the test fixture holds. The digest
    # is of the LF export the container wrote at the PR head (run
    # 20261022T194252Z-c94516ee1); a Windows build writes the same rows with
    # CRLF endings. The run records whether the merge commit's export matches.
    "commit": "acbedb5fef490137219207cf40945d70c2db34ea",
    "facts_sha256": "58774e11cf63754f85be0a082c29b581033c304e5d26dc71b2b23eac0cc3f1d8",
    "pull_request": "https://github.com/PolicyEngine/chronicle/pull/347",
    "year": 2026,
    "packages": [
        "packages/ed_fsa/portfolio_summary_fy2026_q3",
        "packages/ed_fsa/portfolio_by_location_fy2026_q2",
        "packages/ed_fsa/portfolio_by_age_fy2026_q2",
        "packages/ed_fsa/portfolio_by_debt_size_fy2026_q2",
        "packages/ed_fsa/portfolio_by_loan_type_fy2026_q3",
        "packages/ed_fsa/dl_portfolio_by_repayment_plan_fy2026_q3",
    ],
    "rows": 461,
}
_TESTS = "packages/microcosm-build/tests"
_CALIBRATE_TESTS = "packages/microcosm-calibrate/tests"
#: step -> commands. A part "glob:<pattern>" expands to the sorted matches.
STEPS: dict[str, list[list[str]]] = {
    "tests": [
        [
            "{python}",
            "-m",
            "pytest",
            "-q",
            f"{_TESTS}/engine_free/us/test_us_fsa_student_loan_targets.py",
            f"{_TESTS}/engine_free/us/test_us_fiscal_targets.py",
            f"{_TESTS}/engine_free/us/test_us_chronicle_feed.py",
            f"{_TESTS}/engine_free/shared/test_ledger_targets.py",
            f"{_TESTS}/engine_free/shared/test_ledger_artifact.py",
            f"{_TESTS}/engine_free/shared/test_release_target_parity.py",
            f"{_CALIBRATE_TESTS}/engine_free/shared/test_provider_labels.py",
            f"{_CALIBRATE_TESTS}/engine_free/shared/test_variable_labels.py",
        ],
        [
            "{python}",
            "-m",
            "pytest",
            "-q",
            f"{_TESTS}/engine_free/us/test_us_fiscal_refresh_builder.py",
            "-k",
            "ledger_filter or restated or scf_education_debt or parity or materializ",
        ],
    ],
    "contract": [
        [
            "{python}",
            "-m",
            "pytest",
            "-q",
            f"{_TESTS}/engine_contract/us/test_us_fiscal_targets.py",
            f"{_TESTS}/engine_contract/us/test_release_input_coverage.py",
        ],
    ],
    "spec": [
        ["{python}", "tools/generate_us_bundle_from_constants.py", "--check"],
        ["{python}", "tools/spec_engine_coverage.py", "--check"],
        [
            "{python}",
            "-m",
            "pytest",
            "-q",
            f"glob:{_TESTS}/engine_free/shared/test_spec_engine_*.py",
            f"{_TESTS}/engine_free/us/test_us_spec_bundle.py",
            f"{_TESTS}/engine_free/us/test_us_bundle_core_contracts.py",
        ],
    ],
    "feed": [],
    "fit": [
        [
            "{python}",
            "experiments/fit_1074_fsa_calibration.py",
            "--h5",
            "{h5}",
            "--feed",
            "{feed}",
            "--feed-provenance",
            "{feed_provenance}",
            "--scf-cache-dir",
            "{scf}",
            "--diagnostics-dir",
            "{results}/diagnostics",
            "--materialization-cache-dir",
            "/cache/materialization",
            "--run",
            "{run_id}",
        ]
    ],
}
OUTPUT_GLOBS = ("experiments/receipts/1074-*.json", "experiments/1074-*.md")
CPU = 8.0
# The whole 57,240-household support in one engine peaks near 88 GB.
MEMORY_MB = 131072

image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("git")
    .pip_install("uv==0.11.7", "huggingface_hub")
)
app = modal.App("microcosm-1074-experiments")
cache = modal.Volume.from_name("microcosm-1074-cache", create_if_missing=True)
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


def _uv_env(**overrides: str) -> dict[str, str]:
    # A Modal volume cannot host uv's cache: uv persists temporary files by
    # atomic rename, which the volume refuses.
    return _clean_env(UV_CACHE_DIR="/root/.cache/uv", UV_LINK_MODE="copy", **overrides)


def _clone(url: str, commit: str, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    run = lambda *args: subprocess.run(args, cwd=target, check=True)  # noqa: E731
    run("git", "init", "-q")
    run("git", "remote", "add", "origin", url)
    run("git", "fetch", "-q", "--depth", "1", "origin", commit)
    run("git", "checkout", "-q", "--detach", "FETCH_HEAD")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=target,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != commit:
        raise RuntimeError(f"{url}: checked out {head}, expected {commit}")


def _checkout(commit: str) -> None:
    _clone(REPO_URL, commit, WORK)
    subprocess.run(
        ["uv", "sync", "--all-packages", "--extra", "us", "--frozen"],
        cwd=WORK,
        env=_uv_env(UV_PROJECT_ENVIRONMENT=str(VENV)),
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


def _pinned_feed(log) -> Path:
    """The pinned US feed, rebuilt at its Chronicle commit and sha-checked."""

    feeds = CACHE / "feeds"
    feeds.mkdir(parents=True, exist_ok=True)
    path = feeds / f"consumer_facts_us_{PINNED_FEED['commit'][:7]}.jsonl"
    if _verified(path, PINNED_FEED["facts_sha256"]):
        return path
    root = Path("/work/chronicle-pinned")
    _clone(PINNED_FEED["repo"], PINNED_FEED["commit"], root)
    subprocess.run(["uv", "sync", "--frozen"], cwd=root, env=_uv_env(), check=True)
    out = Path("/tmp/us-feed")
    t0 = time.time()
    subprocess.run(
        [
            str(VENV / "bin" / "python"),
            "tools/build_us_chronicle_feed.py",
            "--chronicle-root",
            str(root),
            "--out",
            str(out),
            "--replace",
            "--skip-artifact",
        ],
        cwd=WORK,
        env=_uv_env(),
        check=True,
    )
    log["pinned_feed_build_seconds"] = round(time.time() - t0, 1)
    built = out / "consumer_facts.jsonl"
    if not _verified(built, PINNED_FEED["facts_sha256"]):
        raise RuntimeError(
            f"rebuilt US feed sha256 {_sha256(built)} != pinned "
            f"{PINNED_FEED['facts_sha256']}"
        )
    shutil.copyfile(built, path)
    cache.commit()
    return path


def _fsa_rows(log) -> Path:
    """The FSA packages' consumer facts at chronicle#347's merge commit."""

    feeds = CACHE / "feeds"
    path = feeds / f"ed_fsa_{FSA_PACKAGES['commit'][:7]}.jsonl"
    receipt = feeds / f"ed_fsa_{FSA_PACKAGES['commit'][:7]}.json"
    if path.exists() and receipt.exists():
        if json.loads(receipt.read_text())["sha256"] == _sha256(path):
            return path
    root = Path("/work/chronicle-fsa")
    _clone(FSA_PACKAGES["repo"], FSA_PACKAGES["commit"], root)
    subprocess.run(["uv", "sync", "--frozen"], cwd=root, env=_uv_env(), check=True)
    out = Path("/tmp/fsa-bundle")
    command = [
        "uv",
        "run",
        "--frozen",
        "chronicle",
        "build-bundle",
        "--year",
        str(FSA_PACKAGES["year"]),
        *[
            part
            for package in FSA_PACKAGES["packages"]
            for part in ("--source", package)
        ],
        "--out",
        str(out),
        "--replace",
    ]
    subprocess.run(
        command, cwd=root, env=_uv_env(), check=True, stdout=subprocess.DEVNULL
    )
    built = out / "consumer_facts.jsonl"
    rows = [line for line in built.read_bytes().splitlines() if line.strip()]
    if len(rows) != FSA_PACKAGES["rows"]:
        raise RuntimeError(f"FSA bundle has {len(rows)} rows, expected 461")
    shutil.copyfile(built, path)
    sha256 = _sha256(path)
    receipt.write_text(
        json.dumps(
            {
                "sha256": sha256,
                "matches_facts_sha256": sha256 == FSA_PACKAGES["facts_sha256"],
                "command": command,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    log["fsa_command"] = command
    cache.commit()
    return path


def _candidate_feed(log) -> tuple[Path, Path]:
    """The pinned feed plus the FSA rows, sorted by aggregate_fact_key."""

    pinned = _pinned_feed(log)
    fsa = _fsa_rows(log)
    feeds = CACHE / "feeds"
    path = feeds / (
        f"candidate_us_{PINNED_FEED['commit'][:7]}_ed_fsa_"
        f"{FSA_PACKAGES['commit'][:7]}.jsonl"
    )
    provenance = path.with_suffix(".provenance.json")
    if path.exists() and provenance.exists():
        if json.loads(provenance.read_text())["candidate_sha256"] == _sha256(path):
            return path, provenance
    pinned_rows = [line for line in pinned.read_bytes().splitlines() if line.strip()]
    fsa_rows = [line for line in fsa.read_bytes().splitlines() if line.strip()]
    keys = {}
    ids = set()
    for line in pinned_rows:
        row = json.loads(line)
        keys[row["aggregate_fact_key"]] = line
        ids.add(row["lineage"]["source_record_id"])
    for line in fsa_rows:
        row = json.loads(line)
        if (
            row["aggregate_fact_key"] in keys
            or row["lineage"]["source_record_id"] in ids
        ):
            raise RuntimeError(
                f"FSA row collides with the pinned feed: {row['lineage']}"
            )
        keys[row["aggregate_fact_key"]] = line
    payload = b"".join(keys[key] + b"\n" for key in sorted(keys))
    path.write_bytes(payload)
    provenance.write_text(
        json.dumps(
            {
                "kind": "reviewed diagnostic candidate (not a pin)",
                "pinned_feed": {**PINNED_FEED, "path": pinned.name},
                "fsa_packages": {
                    **FSA_PACKAGES,
                    "path": fsa.name,
                    "sha256": _sha256(fsa),
                    "matches_facts_sha256": _sha256(fsa)
                    == FSA_PACKAGES["facts_sha256"],
                },
                "rule": (
                    "pinned feed rows plus the FSA rows, no shared "
                    "aggregate_fact_key or source_record_id, sorted by "
                    "aggregate_fact_key"
                ),
                "rows": len(keys),
                "candidate_sha256": _sha256(path),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    cache.commit()
    return path, provenance


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
            "import microcosm.build.us_runtime.fsa_student_loan_targets as m; "
            "print(m.__file__)",
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


# Non-preemptible, as tools/modal_us_stage.py runs its heavy stages: a
# preempted container restarts the whole run from the clone.
@app.function(
    image=image,
    cpu=CPU,
    memory=MEMORY_MB,
    timeout=8 * 3600,
    volumes={"/cache": cache},
    retries=0,
    nonpreemptible=True,
)
def run_steps(
    commit: str, steps: list[str], run_id: str, script_text: str = ""
) -> dict:
    import resource  # Unix-only; the local entrypoint runs on Windows too

    started = datetime.now(UTC)
    _checkout(commit)
    paths = _inputs()
    out = CACHE / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    fields = {key: str(path) for key, path in paths.items()} | {
        "python": str(VENV / "bin" / "python"),
        "run_id": run_id,
        "results": str(out),
    }
    log = {
        "run_id": run_id,
        "commit": commit,
        "steps": [],
        "cpu": CPU,
        "memory_mb": MEMORY_MB,
        "inputs": {"host": HOST, "scf": {k: v[1] for k, v in SCF_ZIPS.items()}},
        "started_at": started.isoformat(),
    }
    if {"feed", "fit"} & set(steps):
        feed_log: dict[str, object] = {}
        feed, provenance = _candidate_feed(feed_log)
        fields |= {"feed": str(feed), "feed_provenance": str(provenance)}
        log["feed"] = {
            **json.loads(provenance.read_text()),
            **feed_log,
            "path": str(feed),
        }
    threads = str(int(CPU))
    env = _clean_env(
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS=threads,
        MKL_NUM_THREADS=threads,
        OPENBLAS_NUM_THREADS=threads,
    )
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
        str(p.relative_to(root)): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and "diagnostics" not in p.parts
    }


def _check_commit(commit: str) -> None:
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise SystemExit("--commit must be a full 40-hex sha that is pushed to GitHub")


def _names(steps: str, script: str) -> tuple[list[str], str]:
    names = [s for s in steps.split(",") if s]
    script_text = Path(script).read_text(encoding="utf-8") if script else ""
    if script and "script" not in names:
        names.append("script")
    unknown = sorted(set(names) - set(STEPS) - {"script"})
    if unknown:
        raise SystemExit(f"unknown steps: {unknown}")
    return names, script_text


@app.local_entrypoint()
def preflight(commit: str) -> None:
    _check_commit(commit)
    print(json.dumps(preflight_remote.remote(commit), indent=2))


@app.local_entrypoint()
def launch(commit: str, steps: str = "tests", script: str = "") -> None:
    """Spawn the run and exit at once (use with ``modal run --detach``).

    A client attached for the whole run cancels the remote input when it is
    stopped (for example when the machine runs low on memory), even under
    --detach. ``launch`` returns as soon as the call is spawned, so nothing
    local can cancel it; ``collect`` fetches the results.
    """
    _check_commit(commit)
    names, script_text = _names(steps, script)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + commit[:9]
    call = run_steps.spawn(commit, names, run_id, script_text)
    print(f"run_id={run_id}")
    print(f"function_call_id={call.object_id}")


@app.local_entrypoint()
def collect(run_id: str, logs_dir: str = "") -> None:
    """Write a run's changed files into this checkout and its logs to logs_dir."""
    repo = Path(__file__).resolve().parents[1]
    logs = (
        Path(logs_dir)
        if logs_dir
        else Path.home() / ".cache" / "microcosm" / "modal-1074" / run_id
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
