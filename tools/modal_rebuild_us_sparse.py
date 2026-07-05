"""Modal app: rebuild the corrected sparse populace-us release.

Background
----------
The certified sparse-57k default release (``populace-us-2024-sparse-l0-refit-57k
-71a0887-national-only-20260701``, bundle policyengine 4.18.8) zeroes ~84
material engine-input bases the dense parent populates — IRA/HSA/self-employed
pension contributions and childcare among them — so CDCC/ALD reforms silently
score ~$0 while the release still hits its own calibration target surface
(populace issue #278). PR #279 fixed the pipeline:

* the PUF donor stage now imputes ``traditional_ira_contributions_desired`` and
  ``self_employed_pension_contributions_desired`` and carries
  ``health_savings_account_ald`` (``us_runtime/puf_support.py``);
* the CPS-carried stage derives ``spm_unit_pre_subsidy_childcare_expenses`` from
  the replicated ASEC ``SPM_CHILDCAREXPNS`` column (``us_runtime/cps_carried.py``);
* ``input_mass_parity_gate`` fails a build whose material input columns collapse
  versus a certified reference (``build/gates.py``), and the fiscal-refresh
  builder gates base-vs-``--input-mass-reference-h5`` **before** calibration and
  export-vs-base before writing the H5.

This app stands the corrected rebuild up on Modal. Everything runs inside one
container image built from the repo at the pinned ``uv.lock`` (policyengine-us
1.755.4, policyengine-core 3.26.11 — the engine version consumers pin for this
release, one bump past the July-1 sparse build's 1.752.2). Two changes separate
this build from the broken July-1 release: the #278 data fix, and the engine
bump (NJ filing-threshold floor, CA non-MAGI asset limit, Head Start age-window
fixes). The heavy build tools
and the in-container inspection both run under the workspace venv
(``/root/populace/.venv/bin/python``) as subprocesses; the Modal function body
is pure stdlib orchestration, so there is no interpreter mismatch.

Two phases:

1. **base rebuild** — ``tools/build_us_puf_support_base.py`` layers the #278
   PUF/childcare/immigration inputs onto a base support frame;
2. **calibrate + write the corrected sparse release** —
   ``tools/build_us_fiscal_refresh_release.py`` recalibrates the household
   weights (default sparse L0+refit), gating base-vs the certified DENSE
   f0af251 reference and export-vs-base for input-mass parity, and writes the
   release contract files.

A ``--smoke`` (default) mode proves the plumbing cheaply before any real money
is spent: it wires up the HF secret, the volumes, and the gate machinery,
inspects the published base for the #278 columns, and exercises the fast
base-vs-reference input-mass gate (which legitimately *fails* on the broken
published base — that failure is the point of #278/#279 and demonstrates the
gate fires end to end).

Licensing
---------
PUF-derived artifacts (the base H5, the dense reference H5, any donor PUF) move
only between Hugging Face and Modal compute here; nothing PUF-derived is written
to the git repo or printed to logs. Only column *names*, weighted *totals*, and
gate verdicts are surfaced.

Usage
-----
Smoke (cheap wiring proof; safe to run):

    modal run tools/modal_rebuild_us_sparse.py

Full corrected build (heavy; only with the main session's go-ahead):

    modal run --detach tools/modal_rebuild_us_sparse.py --full

See ``FULL_RUN_NOTES`` for the resource/cost estimate and the exact release
publish steps that follow a successful run.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

APP_NAME = "populace-us-sparse-rebuild"

# The public populace-us dataset holds the published releases. The current
# default (main) is the broken sparse-57k; the DENSE f0af251 release is the
# input-mass reference the corrected base must match (issue #278).
POPULACE_US_REPO = "policyengine/populace-us"
DEFAULT_BASE_FILENAME = "populace_us_2024.h5"

# Certified DENSE reference release (task-supplied): build_manifest dataset
# sha256 16be6338f9d0b3c339883dae59949e995663b64cf145de6728b3dd0f916c5d5f.
DENSE_REFERENCE_REVISION = "populace-us-2024-f0af251-703bd81a565c-20260620T201958Z"
DENSE_REFERENCE_SHA256 = (
    "16be6338f9d0b3c339883dae59949e995663b64cf145de6728b3dd0f916c5d5f"
)

# Current broken sparse default (for the smoke gate demonstration).
SPARSE_DEFAULT_REVISION = (
    "populace-us-2024-sparse-l0-refit-57k-71a0887-national-only-20260701"
)

# Processed PUF donor lives in the public us-data MODEL repo. (PUF is licensed;
# it is fetched only onto Modal and never logged/committed.)
US_DATA_MODEL_REPO = "policyengine/policyengine-us-data"
US_DATA_DONOR_REVISION = "main"
PUF_2024_PATH = "staging/1.73.0_22f922eb_20260329_223332/datasets/puf_2024.h5"

REPO_ROOT = Path(__file__).resolve().parent.parent  # the populace worktree root

# Where the repo is baked inside the image, and the workspace venv uv sync
# creates there.
IMAGE_REPO_ROOT = "/root/populace"
VENV_PYTHON = f"{IMAGE_REPO_ROOT}/.venv/bin/python"

# Output + reference caches on a persistent Modal volume.
OUTPUT_VOLUME_NAME = "populace-us-sparse-rebuild"
OUTPUT_MOUNT = "/rebuild"

# Existing volume that already holds the PolicyEngine Ledger consumer facts
# (created 2026-07-02). Mounted for --ledger-facts.
FACTS_VOLUME_NAME = "populace-us-build-facts"
FACTS_MOUNT = "/facts"
CONSUMER_FACTS_FILENAME = "consumer_facts.jsonl"

# The four #278 leaves whose collapse the gate must catch.
ISSUE_278_COLUMNS = (
    "traditional_ira_contributions_desired",
    "self_employed_pension_contributions_desired",
    "health_savings_account_ald",
    "spm_unit_pre_subsidy_childcare_expenses",
)

FULL_RUN_NOTES = """\
Full run (heavy). Do NOT launch without the main session's go-ahead. Recommended:

  modal run --detach tools/modal_rebuild_us_sparse.py --full

which runs `run_full_build.remote(...)` on a 64 GB / 16-CPU container. It:
  1. fetches the certified DENSE f0af251 base (already carries every #278 input
     column — the smoke run proved it) and calibrates it to a sparse L0+refit
     selection: build_us_fiscal_refresh_release.py --base-h5 <dense f0af251>
     --ledger-facts <consumer_facts.jsonl> --input-mass-reference-h5 <dense
     f0af251>, default sparse L0+refit, staging ON;
  2. the #279 base-vs-reference gate is trivially satisfied (base == reference)
     and the export-vs-base gate ensures the L0 selection keeps the #278 mass;
  3. persists the release dir + all gate verdicts to the output volume.

Pass --rebuild-base to additionally run a from-source base rebuild
(build_us_puf_support_base.py) on the dense base first; unnecessary given the
dense base is already correct, and slower.

Post-run: `modal volume get populace-us-sparse-rebuild full/release <local>` and
`.../full/release/artifacts <local>`, review gates (input_mass_parity.json,
calibration_diagnostics.json), then publish from a machine with the release env:
  tools/publish_release.sh <release_dir> --repo-id policyengine/populace-us \\
    --artifact-root <artifact_root>
then certify by loading the published latest.json dataset in policyengine.py.
"""


# --------------------------------------------------------------------------- #
# Image                                                                        #
# --------------------------------------------------------------------------- #
# Build from the repo at its pinned uv.lock so the engine is exactly the one
# consumers pin for this release (policyengine-us 1.755.4, policyengine-core
# 3.26.11 — one bump past the July-1 sparse build's 1.752.2). The #278 data fix
# and that engine bump are the two changes from the broken July-1 release.
# `uv sync --all-packages --extra us --frozen`
# creates /root/populace/.venv; every populace/HF operation runs under that
# venv as a subprocess, so the Modal function interpreter never needs the deps.

REPO_CLONE_URL = "https://github.com/PolicyEngine/populace"


def _local_repo_commit() -> str:
    """Pinned commit for the image: the worktree's pushed HEAD.

    The release builder records git provenance (commit, dirty state) from
    inside the container, so the image must hold a real clone at the exact
    commit being run — not a .git-less file copy. Refuse to build from a
    dirty tree so the image always matches what's on the remote.
    """
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise RuntimeError(
            "Local worktree is dirty; commit and push before launching so the "
            "image clone matches the code being run:\n" + dirty
        )
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _image() -> modal.Image:
    # The image spec is only materialized from the local client; inside the
    # container this module is re-imported but the spec is never rebuilt, so
    # skip the git lookup there (no repo at REPO_ROOT in-container).
    commit = _local_repo_commit() if modal.is_local() else "unused-in-container"
    return (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("git", "build-essential")
        .pip_install("uv==0.11.7", "psutil==7.2.0")
        .env(
            {
                "HF_HUB_ENABLE_HF_TRANSFER": "0",
                "POPULACE_STAGING_REPO_ID": "policyengine/populace-us-staging",
            }
        )
        .run_commands(
            # Blobless clone keeps the image light; checkout materializes the
            # pinned commit with a real .git so the builder's provenance
            # (_git_dirty/_git_output) sees a clean tree at the right sha.
            f"git clone --filter=blob:none {REPO_CLONE_URL} {IMAGE_REPO_ROOT}"
            f" && cd {IMAGE_REPO_ROOT}"
            f" && git fetch origin {commit}"
            f" && git checkout --detach {commit}",
            f"cd {IMAGE_REPO_ROOT} && uv sync --all-packages --extra us --frozen",
        )
    )


image = _image()

app = modal.App(APP_NAME)

hf_secret = modal.Secret.from_name("huggingface-token")

output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
facts_volume = modal.Volume.from_name(FACTS_VOLUME_NAME, create_if_missing=True)


# --------------------------------------------------------------------------- #
# The in-container worker script (runs under the workspace venv).               #
# --------------------------------------------------------------------------- #
# Kept as a string so it executes under VENV_PYTHON (where huggingface_hub and
# populace.* resolve) rather than the Modal function interpreter.

_SMOKE_WORKER = r"""
import json, os, sys
from pathlib import Path

REPO = os.environ["POPULACE_REPO"]
sys.path.insert(0, f"{REPO}/packages/populace-build/src")

POPULACE_US_REPO = os.environ["POPULACE_US_REPO"]
DEFAULT_BASE_FILENAME = os.environ["DEFAULT_BASE_FILENAME"]
SPARSE_DEFAULT_REVISION = os.environ["SPARSE_DEFAULT_REVISION"]
DENSE_REFERENCE_REVISION = os.environ["DENSE_REFERENCE_REVISION"]
ISSUE_278_COLUMNS = os.environ["ISSUE_278_COLUMNS"].split(",")
OUT = Path(os.environ["SMOKE_OUT"])
OUT.mkdir(parents=True, exist_ok=True)

from huggingface_hub import HfApi, hf_hub_download

verdict = {"steps": {}}

who = HfApi().whoami()
verdict["steps"]["hf_whoami"] = {
    "name": who.get("name"),
    "type": who.get("type"),
    "orgs": [o.get("name") for o in who.get("orgs", [])],
    "hf_token_present": bool(os.environ.get("HF_TOKEN")),
}
print("hf whoami:", verdict["steps"]["hf_whoami"], flush=True)

base_path = hf_hub_download(
    repo_id=POPULACE_US_REPO, filename=DEFAULT_BASE_FILENAME,
    revision=SPARSE_DEFAULT_REVISION, repo_type="dataset",
)
reference_path = hf_hub_download(
    repo_id=POPULACE_US_REPO, filename=DEFAULT_BASE_FILENAME,
    revision=DENSE_REFERENCE_REVISION, repo_type="dataset",
)
verdict["steps"]["downloads"] = {
    "sparse_base_bytes": Path(base_path).stat().st_size,
    "dense_reference_bytes": Path(reference_path).stat().st_size,
}

from populace.build.gates import input_mass_parity_gate
from populace.build.us_runtime.input_mass import us_input_mass_totals
from populace.build.us_runtime.l0_refit_export import load_us_frame
from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

input_variables = tuple(PolicyEngineUSEngine().variables())
base_frame = load_us_frame(base_path)
reference_frame = load_us_frame(reference_path)
base_totals = us_input_mass_totals(base_frame, columns=input_variables)
reference_totals = us_input_mass_totals(reference_frame, columns=input_variables)

issue_278 = {}
for col in ISSUE_278_COLUMNS:
    issue_278[col] = {
        "sparse_base_total": float(base_totals.get(col, 0.0)),
        "dense_reference_total": float(reference_totals.get(col, 0.0)),
        "present_in_base": col in base_totals,
        "present_in_reference": col in reference_totals,
    }
verdict["steps"]["issue_278_columns"] = issue_278
print("issue #278 column totals:", json.dumps(issue_278, indent=2), flush=True)

gate = input_mass_parity_gate(
    base_totals, reference_totals,
    candidate_name="published_sparse_base",
    reference_name="dense_f0af251_reference",
    relative_tolerance=0.5, minimum_reference_total=1e9,
)
verdict["steps"]["input_mass_parity_gate"] = {
    "passed": gate.passed,
    "n_failures": len(gate.failures),
    "failures_head": list(gate.failures)[:12],
    "columns_checked": gate.details.get("columns_checked"),
    "worst_drifts": gate.details.get("worst_drifts"),
}
print(
    "input_mass_parity_gate on published sparse base: "
    f"passed={gate.passed} n_failures={len(gate.failures)}",
    flush=True,
)

# Does the published base retain the raw ASEC childcare column the base
# rebuild's CPS-carried derivation needs? Informs the base-rebuild strategy.
raw_asec_present = {}
for raw_col, table in (("SPM_CHILDCAREXPNS", "spm_unit"), ("SPM_CHILDCAREXPNS", "person")):
    try:
        df = base_frame.table(table)
        raw_asec_present[f"{table}.{raw_col}"] = raw_col in df.columns
    except Exception:
        raw_asec_present[f"{table}.{raw_col}"] = "no_such_table"
verdict["steps"]["raw_asec_columns_in_published_base"] = raw_asec_present

facts_path = Path(os.environ["FACTS_PATH"])
verdict["steps"]["ledger_facts"] = {
    "path": str(facts_path),
    "exists": facts_path.exists(),
    "bytes": facts_path.stat().st_size if facts_path.exists() else 0,
}

(OUT / "smoke_verdict.json").write_text(
    json.dumps(verdict, indent=2, sort_keys=True) + "\n"
)
print("SMOKE_VERDICT_JSON_BEGIN")
print(json.dumps(verdict, sort_keys=True))
print("SMOKE_VERDICT_JSON_END")
"""


# --------------------------------------------------------------------------- #
# In-container helpers                                                          #
# --------------------------------------------------------------------------- #


def _start_rss_logger(interval_seconds: int = 60) -> None:
    """Print container memory usage periodically (daemon thread).

    The 64/128 GB runs died silently in target compilation; this trail
    turns the next death into a peak-memory measurement.
    """
    import threading

    import psutil

    def _loop() -> None:
        peak = 0.0
        # Log to the volume, not stdout: `modal app logs` streaming has been
        # unreliable from the build machine, while volume reads work mid-run
        # (the staging telemetry proves it).
        log_path = Path(OUTPUT_MOUNT) / "full" / "rss.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            used = psutil.virtual_memory().used / 2**30
            peak = max(peak, used)
            line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} used={used:.1f}GiB peak={peak:.1f}GiB"
            print(f"[rss-logger] {line}", flush=True)
            try:
                with log_path.open("a") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass
            time.sleep(interval_seconds)

    threading.Thread(target=_loop, daemon=True).start()


def _run(
    cmd: list[str],
    *,
    cwd: str = IMAGE_REPO_ROOT,
    env: dict | None = None,
    tee_path: Path | None = None,
) -> dict:
    """Run a subprocess, returning a small result dict.

    Never prints file *contents* — only the command and its stdout/stderr tail
    (which the build tools keep to column names / totals / gate verdicts).
    With ``tee_path``, combined output streams line-by-line to that file so
    crash forensics never depend on `modal app logs` streaming.
    """
    print(f"$ {' '.join(cmd)}", flush=True)
    run_env = {**os.environ, **(env or {})}
    if tee_path is None:
        proc = subprocess.run(cmd, cwd=cwd, env=run_env, text=True, capture_output=True)
        stdout, stderr = proc.stdout or "", proc.stderr or ""
        returncode = proc.returncode
    else:
        tee_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        with tee_path.open("a") as fh:
            popen = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=run_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert popen.stdout is not None
            for line in popen.stdout:
                lines.append(line)
                fh.write(line)
                fh.flush()
            popen.wait()
        stdout, stderr = "".join(lines), ""
        returncode = popen.returncode
    tail = "\n".join(stdout.splitlines()[-60:])
    err_tail = "\n".join(stderr.splitlines()[-40:])
    if tail:
        print(tail, flush=True)
    if returncode != 0:
        print(f"[exit {returncode}] stderr tail:\n{err_tail}", flush=True)
    return {
        "cmd": cmd,
        "returncode": returncode,
        "stdout": stdout,
        "stdout_tail": tail,
        "stderr_tail": err_tail or tail[-2000:],
    }


# --------------------------------------------------------------------------- #
# Smoke: prove the plumbing                                                     #
# --------------------------------------------------------------------------- #


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={OUTPUT_MOUNT: output_volume, FACTS_MOUNT: facts_volume},
    cpu=4.0,
    memory=16 * 1024,
    timeout=60 * 30,
)
def smoke() -> dict:
    """Prove the wiring end-to-end without a real (expensive) calibration.

    All populace/HF work runs under the workspace venv (VENV_PYTHON) via the
    _SMOKE_WORKER script. The function body only orchestrates + parses the
    verdict, so it needs no third-party imports itself.
    """
    smoke_out = Path(OUTPUT_MOUNT) / "smoke"
    facts_path = Path(FACTS_MOUNT) / CONSUMER_FACTS_FILENAME

    worker_env = {
        "POPULACE_REPO": IMAGE_REPO_ROOT,
        "POPULACE_US_REPO": POPULACE_US_REPO,
        "DEFAULT_BASE_FILENAME": DEFAULT_BASE_FILENAME,
        "SPARSE_DEFAULT_REVISION": SPARSE_DEFAULT_REVISION,
        "DENSE_REFERENCE_REVISION": DENSE_REFERENCE_REVISION,
        "ISSUE_278_COLUMNS": ",".join(ISSUE_278_COLUMNS),
        "SMOKE_OUT": str(smoke_out),
        "FACTS_PATH": str(facts_path),
    }
    worker = _run(
        [VENV_PYTHON, "-c", _SMOKE_WORKER],
        env=worker_env,
    )

    # Also confirm the build tools import + parse args under the venv.
    base_help = _run([VENV_PYTHON, "tools/build_us_puf_support_base.py", "--help"])
    refresh_help = _run(
        [VENV_PYTHON, "tools/build_us_fiscal_refresh_release.py", "--help"]
    )

    output_volume.commit()

    verdict: dict = {"worker_returncode": worker["returncode"]}
    # Extract the machine-readable verdict the worker printed.
    stdout = worker["stdout"]
    if "SMOKE_VERDICT_JSON_BEGIN" in stdout and "SMOKE_VERDICT_JSON_END" in stdout:
        payload = stdout.split("SMOKE_VERDICT_JSON_BEGIN", 1)[1]
        payload = payload.split("SMOKE_VERDICT_JSON_END", 1)[0].strip()
        try:
            verdict["worker"] = json.loads(payload)
        except json.JSONDecodeError:
            verdict["worker_parse_error"] = True
    if worker["returncode"] != 0:
        verdict["worker_stderr_tail"] = worker["stderr_tail"]
    verdict["base_builder_help_ok"] = base_help["returncode"] == 0
    verdict["fiscal_refresh_help_ok"] = refresh_help["returncode"] == 0
    verdict["verdict_path"] = str(smoke_out / "smoke_verdict.json")
    return verdict


# --------------------------------------------------------------------------- #
# Full build (heavy — gated on go-ahead)                                        #
# --------------------------------------------------------------------------- #

_HF_FETCH = r"""
import os, sys
from huggingface_hub import hf_hub_download
kind, repo, filename, revision = sys.argv[1:5]
print(hf_hub_download(repo_id=repo, filename=filename, revision=revision, repo_type=kind))
"""


def _fetch(kind: str, repo: str, filename: str, revision: str) -> str:
    res = _run([VENV_PYTHON, "-c", _HF_FETCH, kind, repo, filename, revision])
    if res["returncode"] != 0:
        raise RuntimeError(f"HF fetch failed for {repo}/{filename}@{revision}")
    return res["stdout"].strip().splitlines()[-1]


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={OUTPUT_MOUNT: output_volume, FACTS_MOUNT: facts_volume},
    cpu=16.0,
    # Measured: the full pipeline peaked at 85.5 GiB (r9's rss.log, which
    # ran calibration + reform targets end to end). 128 GiB gives ~50%
    # headroom. The earlier "OOM" deaths were client teardown, not memory.
    memory=128 * 1024,
    timeout=60 * 60 * 12,
)
def run_full_build(
    *,
    rebuild_base: bool = False,
    base_h5_override: str | None = None,
    epochs: int = 1500,
    skip_out_of_sample_reforms: bool = False,
    allow_input_mass_drift: bool = False,
    no_staging: bool = False,
    release_id: str | None = None,
    lambda_share: float | None = None,
    out_subdir: str = "full",
) -> dict:
    """Calibrate + write the corrected sparse release, gated against the dense
    f0af251 reference.

    The smoke run proved the DENSE f0af251 base already carries every #278 input
    column (IRA $16.1B, HSA $10.5B, SE-pension $1.0B, childcare $74.7B) plus 68
    other material bases the broken sparse base dropped. The corrected sparse
    release is therefore that correct dense base, recalibrated to a sparse
    L0+refit selection:

    * ``rebuild_base=False`` (default, recommended): feed the dense f0af251 base
      straight to the fiscal-refresh builder. base-vs-reference parity is
      trivially satisfied (base == reference) and export-vs-base parity ensures
      the L0 selection keeps the #278 mass.
    * ``rebuild_base=True``: additionally re-run ``build_us_puf_support_base.py``
      (PUF/childcare/immigration) on the dense base first — a genuine from-source
      base rebuild. Use only if a from-scratch base is specifically wanted; it
      re-clones the PUF support channel and is slower.

    NOTE: this is the expensive path (target compilation + L0+refit + reform
    validation). Do not launch without the main session's go-ahead.
    """
    _start_rss_logger()
    work = Path(OUTPUT_MOUNT) / out_subdir
    work.mkdir(parents=True, exist_ok=True)
    result: dict = {}

    reference_path = _fetch(
        "dataset", POPULACE_US_REPO, DEFAULT_BASE_FILENAME, DENSE_REFERENCE_REVISION
    )
    result["reference_h5"] = reference_path
    result["rebuild_base"] = rebuild_base

    # Phase 1 — choose the calibration base.
    base_support_h5 = base_h5_override or reference_path
    if not rebuild_base:
        # Recommended: the dense base is already correct; calibrate it to sparse.
        calibration_base_h5 = base_support_h5
    else:
        # From-source base rebuild: layer the #278 PUF/childcare/immigration
        # inputs onto the dense base. The processed PUF donor is fetched from the
        # us-data model repo (licensed; never logged/committed).
        puf_h5 = _fetch(
            "model", US_DATA_MODEL_REPO, PUF_2024_PATH, US_DATA_DONOR_REVISION
        )
        base_out = work / "base"
        base_rebuild = _run(
            [
                VENV_PYTHON,
                "tools/build_us_puf_support_base.py",
                "--base-h5",
                base_support_h5,
                "--puf-h5",
                puf_h5,
                "--out",
                str(base_out),
            ]
        )
        result["base_rebuild"] = {"returncode": base_rebuild["returncode"]}
        if base_rebuild["returncode"] != 0:
            result["base_rebuild"]["stderr_tail"] = base_rebuild["stderr_tail"]
            output_volume.commit()
            return result
        calibration_base_h5 = base_out / "base_populace_us_2024_puf_support.h5"

    # Phase 2 — calibrate + write the corrected sparse release.
    facts_path = Path(FACTS_MOUNT) / CONSUMER_FACTS_FILENAME
    release_out = work / "release"
    cmd = [
        VENV_PYTHON,
        "tools/build_us_fiscal_refresh_release.py",
        "--base-h5",
        str(calibration_base_h5),
        "--ledger-facts",
        str(facts_path),
        "--input-mass-reference-h5",
        reference_path,
        "--out",
        str(release_out),
        "--epochs",
        str(epochs),
        # The dense f0af251 base predates the #266 immigration channel (raw
        # PRCITSHP absent; persisted constant-CITIZEN outputs), so the
        # composition gate cannot pass on it. Approved escape hatch
        # (Max, 2026-07-03): record the verdict, keep building — parity with
        # every certified release to date; real fix tracked in populace#225.
        "--allow-immigration-composition-drift",
        # f0af251 is a RELEASE artifact: it persists engine-computed
        # aggregates (employment_income, social_security, ...) alongside
        # their leaves. The pipeline requires leaf inputs only; dropping the
        # aggregates is lossless (verified: every aggregate's leaves are
        # present; pension aggregates equal their private leaves exactly).
        "--drop-formula-owned-base-columns",
    ]
    # Staging telemetry is ON by default so the candidate appears on the
    # populace-us-staging dashboard and `populace-publish-release` does not warn
    # about a missing staging block. The HF secret carries the write token.
    if no_staging:
        cmd += ["--no-staging"]
    if release_id:
        cmd += ["--release-id", release_id]
    if skip_out_of_sample_reforms:
        cmd += ["--skip-out-of-sample-reforms"]
    if allow_input_mass_drift:
        cmd += ["--allow-input-mass-drift"]
    if lambda_share is not None:
        cmd += ["--l0-refit-lambda-share", str(lambda_share)]
    # Shared across probes and the final run: the ~40-minute target
    # materialization is identical for every lambda, so cache it once.
    cmd += [
        "--target-materialization-cache-dir",
        str(Path(OUTPUT_MOUNT) / "target_cache"),
    ]
    build = _run(cmd, tee_path=work / "build_stdout.log")
    result["fiscal_refresh"] = {"returncode": build["returncode"]}
    output_volume.commit()
    if build["returncode"] != 0:
        result["fiscal_refresh"]["stderr_tail"] = build["stderr_tail"]
    return result


# --------------------------------------------------------------------------- #
# Local entrypoint                                                              #
# --------------------------------------------------------------------------- #


@app.local_entrypoint()
def main(full: bool = False, rebuild_base: bool = False, no_staging: bool = False):
    """Default: run the smoke path. Pass --full only with the go-ahead."""
    if full:
        # .spawn() + detach: the client only has to survive the submission
        # round-trip. Every earlier full attempt died ~5 minutes into target
        # compilation regardless of memory size (64/128/336 GB) — the common
        # factor was the local client dying (session restarts, flaky wifi)
        # before/while streaming, tearing the app down. .remote() kept the
        # client on the hook for the whole build; .spawn() does not.
        print("Spawning FULL corrected build (heavy).")
        call = run_full_build.spawn(rebuild_base=rebuild_base, no_staging=no_staging)
        print(f"spawned function call: {call.object_id}")
        return
    print("Running smoke (wiring proof) ...")
    result = smoke.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
