"""Run the microcosm#297 educational-attainment work on Modal instead of local RAM.

Each run clones a pushed commit of PolicyEngine/microcosm, syncs that tree's
uv.lock, fetches the inputs and verifies each sha256, runs the requested steps
in order, and saves every receipt the steps write plus a run log to the
`microcosm-297-cache` volume under results/<run_id>/. `collect` copies a
run's files back into this checkout. Inputs:

- the published host release `populace-us-2024-spm-20260915` from Hugging Face
  (policyengine/populace-us at a pinned revision);
- the official Census CPS ASEC public-use person archives for the host's
  pooled income years (asecpub23/24/25csv.zip, the pins in
  ``microcosm.build.us_runtime.education_assistance_source``);
- the ACS 2024 1-year S1501 state attainment facts, built with
  ``chronicle build-bundle`` from the DRAFT Chronicle package
  ``packages/census/acs_s1501_state_2024``. TEMPORARY: PolicyEngine/chronicle#347
  is unmerged, so the package is read from the author's fork at a pinned
  commit (CHRONICLE_DRAFT below) and the facts are checked against the digest
  of a local build. Nothing in the repository pins the fork; replace this input
  with the pinned US feed once #347 merges and the feed is re-pinned.

    export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
    modal run experiments/modal_297.py::preflight --commit <sha>
    modal run --detach experiments/modal_297.py::launch --commit <sha> --steps tests,receipt
    modal run experiments/modal_297.py::collect --run-id <run_id>

Steps (see STEPS; each runs one or more commands, stopping at the first that
fails):

- ``tests``: the engine-free tests the attainment work touches;
- ``contract``: the engine-contract tests that read the stored-input register
  and the US spec bundle;
- ``receipt``: the build stage on the host release (the exact Census join that
  writes ``A_HGA`` and ``cps_educational_attainment``), the S1501 target
  compile, and the state-totals receipt, labelled with the run id.

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
import subprocess
import time
import urllib.request
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
#: income year -> (official archive url, archive sha256). The same pins as
#: microcosm.build.us_runtime.education_assistance_source.ASEC_EDUCATION_ASSISTANCE_ARCHIVES;
#: the build stage re-verifies the archive and its person member against them.
ASEC_ARCHIVES = {
    2022: (
        "https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asecpub23csv.zip",
        "d2e000250782adfbdd7f29c82b66d866591a30f0d330496698ec19f9c784ce11",
    ),
    2023: (
        "https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asecpub24csv.zip",
        "cdb39cdac34bef99dd0940ab28e306f692404c2eea44d85dfd634214872a0a09",
    ),
    2024: (
        "https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asecpub25csv.zip",
        "318845a2b5e0034eb2973898de1738f4df0025727de38499e7669cb9c0deef0b",
    ),
}
#: TEMPORARY (PolicyEngine/chronicle#347, unmerged): the draft S1501 package on
#: the author's fork, at the head that also carries the raw files on R2. The facts are rebuilt here with that commit's own lock
#: (`chronicle build-bundle --year 2024 --source <package>`), cached under
#: their digest, and the receipt checks every value against the raw Census
#: response the package pins.
CHRONICLE_DRAFT = {
    "repo_url": "https://github.com/DTrim99/chronicle.git",
    "commit": "81e3f986ccc47571a427a4adda8d4b49560d03bf",
    "package": "packages/census/acs_s1501_state_2024",
    "raw": "db/data/census/acs_s1501_state_2024/acs_S1501_state_2024.json",
    "raw_sha256": "84fff08c9d2793a90f2326eb68ae8ba4d5e18957899fc8cdda015d26520f73a4",
    "year": 2024,
    "label": "TEMPORARY draft input: PolicyEngine/chronicle#347 (fork DTrim99/chronicle)",
}
_TESTS = "packages/microcosm-build/tests"
#: step -> commands. A part "glob:<pattern>" expands to the sorted matches.
STEPS: dict[str, list[list[str]]] = {
    "tests": [
        [
            "{python}",
            "-m",
            "pytest",
            "-q",
            f"{_TESTS}/engine_free/us/test_us_educational_attainment.py",
            f"{_TESTS}/engine_free/us/test_us_fiscal_targets.py",
            f"{_TESTS}/engine_free/us/test_us_stored_input_register.py",
            f"{_TESTS}/engine_free/us/test_us_chronicle_feed.py",
            "packages/microcosm-calibrate/tests/engine_free/shared/test_variable_labels.py",
            "packages/microcosm-data/tests/engine_free/us/test_stored_inputs.py",
        ],
        [
            "{python}",
            "-m",
            "pytest",
            "-q",
            f"{_TESTS}/engine_free/us/test_us_fiscal_refresh_builder.py",
            "-k",
            "attainment or population_age or ledger_filter or restated",
        ],
        ["{python}", "tools/ci_test_plan.py", "verify"],
    ],
    "contract": [
        [
            "{python}",
            "-m",
            "pytest",
            "-q",
            f"{_TESTS}/engine_contract/us/test_us_stored_input_register.py",
            f"{_TESTS}/engine_contract/us/test_us_spec_bundle.py",
            f"{_TESTS}/engine_contract/us/test_us_bundle_core_contracts.py",
            "packages/microcosm-data/tests/engine_contract/us/test_stored_inputs.py",
        ],
    ],
    "receipt": [
        [
            "{python}",
            "experiments/build_297_attainment_receipt.py",
            "--h5",
            "{h5}",
            "--asec-archive",
            "2022={asec_2022}",
            "--asec-archive",
            "2023={asec_2023}",
            "--asec-archive",
            "2024={asec_2024}",
            "--s1501-facts",
            "{s1501_facts}",
            "--s1501-raw",
            "{s1501_raw}",
            "--s1501-facts-source",
            "{s1501_source}",
            "--sidecar-dir",
            "{results}",
            "--run",
            "{run_id}",
        ]
    ],
}
OUTPUT_GLOBS = ("experiments/receipts/297-*.json", "experiments/297-*.md")
CPU = 8.0
MEMORY_MB = 32768

image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("git")
    .pip_install("uv==0.11.7", "huggingface_hub")
)
app = modal.App("microcosm-297-experiments")
cache = modal.Volume.from_name("microcosm-297-cache", create_if_missing=True)
CACHE = Path("/cache")
WORK = Path("/work/microcosm")
CHRONICLE = Path("/work/chronicle")
VENV = Path("/opt/venv")
CHRONICLE_VENV = Path("/opt/chronicle-venv")


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


def _uv_env(venv: Path) -> dict[str, str]:
    return _clean_env(
        **{
            "UV_PROJECT_ENVIRONMENT": str(venv),
            # A Modal volume cannot host uv's cache: uv persists temporary files
            # by atomic rename, which the volume refuses.
            "UV_CACHE_DIR": "/root/.cache/uv",
            "UV_LINK_MODE": "copy",
        }
    )


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
        raise RuntimeError(f"checked out {head}, expected {commit}")


def _checkout(commit: str) -> None:
    _clone(REPO_URL, commit, WORK)
    subprocess.run(
        ["uv", "sync", "--all-packages", "--extra", "us", "--frozen"],
        cwd=WORK,
        env=_uv_env(VENV),
        check=True,
    )


def _download(url: str, sha256: str, target: Path) -> None:
    if _verified(target, sha256):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=900) as response:
        target.write_bytes(response.read())
    if not _verified(target, sha256):
        raise RuntimeError(f"{target.name} sha256 mismatch")


def _s1501_facts() -> tuple[Path, Path]:
    """Build the draft S1501 consumer facts once per commit; return facts and raw."""

    root = CACHE / "s1501" / CHRONICLE_DRAFT["commit"]
    facts = root / "consumer_facts.jsonl"
    raw = root / "acs_S1501_state_2024.json"
    if facts.exists() and _verified(raw, CHRONICLE_DRAFT["raw_sha256"]):
        return facts, raw
    _clone(CHRONICLE_DRAFT["repo_url"], CHRONICLE_DRAFT["commit"], CHRONICLE)
    env = _uv_env(CHRONICLE_VENV)
    subprocess.run(["uv", "sync", "--frozen"], cwd=CHRONICLE, env=env, check=True)
    out = Path("/tmp/s1501-bundle")
    subprocess.run(
        [
            str(CHRONICLE_VENV / "bin" / "chronicle"),
            "build-bundle",
            "--year",
            str(CHRONICLE_DRAFT["year"]),
            "--source",
            CHRONICLE_DRAFT["package"],
            "--out",
            str(out),
            "--replace",
        ],
        cwd=CHRONICLE,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    source_raw = CHRONICLE / CHRONICLE_DRAFT["raw"]
    if not _verified(source_raw, CHRONICLE_DRAFT["raw_sha256"]):
        raise RuntimeError("draft S1501 raw Census response sha256 mismatch")
    root.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(source_raw.read_bytes())
    facts.write_bytes((out / "consumer_facts.jsonl").read_bytes())
    return facts, raw


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
    paths: dict[str, Path] = {"h5": host}
    for year, (url, sha256) in ASEC_ARCHIVES.items():
        archive = CACHE / "asec" / url.rsplit("/", 1)[-1]
        _download(url, sha256, archive)
        paths[f"asec_{year}"] = archive
    paths["s1501_facts"], paths["s1501_raw"] = _s1501_facts()
    cache.commit()
    return paths


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
            "import microcosm.build.us_runtime.educational_attainment as m; "
            "print(m.__file__)",
        ],
        cwd=WORK,
        capture_output=True,
        text=True,
        env=_clean_env(),
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
    out = CACHE / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    fields = {key: str(path) for key, path in paths.items()} | {
        "python": str(VENV / "bin" / "python"),
        "run_id": run_id,
        "results": str(out),
        "s1501_source": (
            f"{CHRONICLE_DRAFT['repo_url']}@{CHRONICLE_DRAFT['commit']}:"
            f"{CHRONICLE_DRAFT['package']} ({CHRONICLE_DRAFT['label']})"
        ),
    }
    threads = str(int(CPU))
    env = _clean_env(
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS=threads,
        MKL_NUM_THREADS=threads,
        OPENBLAS_NUM_THREADS=threads,
    )
    (out / "s1501_consumer_facts.jsonl").write_bytes(paths["s1501_facts"].read_bytes())
    log = {
        "run_id": run_id,
        "commit": commit,
        "steps": [],
        "cpu": CPU,
        "memory_mb": MEMORY_MB,
        "inputs": {
            "host": HOST,
            "asec_archives": {str(k): v[1] for k, v in ASEC_ARCHIVES.items()},
            "s1501_draft": CHRONICLE_DRAFT,
            "s1501_facts_sha256": _sha256(paths["s1501_facts"]),
        },
        "started_at": started.isoformat(),
    }
    failed = False
    for step in steps:
        if step == "script":
            # An uncommitted local probe passed by --script; it runs in the
            # clone like any step, and the run log records its text.
            (WORK / "_modal_probe.py").write_text(script_text, encoding="utf-8")
            log["script_sha256"] = hashlib.sha256(script_text.encode()).hexdigest()
            commands = [
                [
                    fields["python"],
                    "_modal_probe.py",
                    json.dumps({key: value for key, value in fields.items()}),
                ]
            ]
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
        if p.is_file() and p.suffix != ".parquet"
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
def main(commit: str, steps: str = "tests,receipt", script: str = "") -> None:
    _check_commit(commit)
    names, script_text = _names(steps, script)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + commit[:9]
    print(f"run_id={run_id}")
    print(json.dumps(run_steps.remote(commit, names, run_id, script_text), indent=2))


@app.local_entrypoint()
def launch(commit: str, steps: str = "tests,receipt", script: str = "") -> None:
    """Spawn the run and exit at once (use with ``modal run --detach``).

    ``main`` keeps a local client attached for the whole run, and stopping that
    client (for example when the machine runs low on memory) cancels the
    remote input even under --detach. ``launch`` returns as soon as the call is
    spawned, so nothing local can cancel it; ``collect`` fetches the results.
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
        else Path.home() / ".cache" / "microcosm" / "modal-297" / run_id
    )
    logs.mkdir(parents=True, exist_ok=True)
    for rel, data in fetch_results.remote(run_id).items():
        if rel.startswith("files/"):
            target = repo / rel.removeprefix("files/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            print("file", target.relative_to(repo))
        else:
            path = logs / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            print("log", path)
