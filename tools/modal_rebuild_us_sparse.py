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
1.752.2, policyengine-core 3.26.11 — exactly the July-1 sparse build engine, so
the only change from that release is the #278 data fix). The heavy build tools
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
# Build from the repo at its pinned uv.lock so the engine is exactly the
# July-1 sparse build's (policyengine-us 1.752.2, policyengine-core 3.26.11).
# Only the #278 data fix differs. `uv sync --all-packages --extra us --frozen`
# creates /root/populace/.venv; every populace/HF operation runs under that
# venv as a subprocess, so the Modal function interpreter never needs the deps.

def _image() -> modal.Image:
    ignore = [
        ".venv",
        "**/__pycache__",
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        "**/*.pyc",
        ".claude",
    ]
    return (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("git", "build-essential")
        .pip_install("uv==0.11.7")
        .env(
            {
                "HF_HUB_ENABLE_HF_TRANSFER": "0",
                "POPULACE_STAGING_REPO_ID": "policyengine/populace-us-staging",
            }
        )
        .add_local_dir(
            str(REPO_ROOT),
            IMAGE_REPO_ROOT,
            ignore=ignore,
            copy=True,
        )
        .run_commands(
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

_SMOKE_WORKER = r'''
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
'''


# --------------------------------------------------------------------------- #
# In-container helpers                                                          #
# --------------------------------------------------------------------------- #

def _run(cmd: list[str], *, cwd: str = IMAGE_REPO_ROOT, env: dict | None = None) -> dict:
    """Run a subprocess, returning a small result dict.

    Never prints file *contents* — only the command and its stdout/stderr tail
    (which the build tools keep to column names / totals / gate verdicts).
    """
    print(f"$ {' '.join(cmd)}", flush=True)
    run_env = {**os.environ, **(env or {})}
    proc = subprocess.run(cmd, cwd=cwd, env=run_env, text=True, capture_output=True)
    tail = "\n".join((proc.stdout or "").splitlines()[-60:])
    err_tail = "\n".join((proc.stderr or "").splitlines()[-40:])
    if tail:
        print(tail, flush=True)
    if proc.returncode != 0:
        print(f"[exit {proc.returncode}] stderr tail:\n{err_tail}", flush=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout or "",
        "stdout_tail": tail,
        "stderr_tail": err_tail,
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

_HF_FETCH = r'''
import os, sys
from huggingface_hub import hf_hub_download
kind, repo, filename, revision = sys.argv[1:5]
print(hf_hub_download(repo_id=repo, filename=filename, revision=revision, repo_type=kind))
'''


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
    memory=64 * 1024,
    timeout=60 * 60 * 8,
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
    work = Path(OUTPUT_MOUNT) / "full"
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
    build = _run(cmd)
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
        print("Launching FULL corrected build (heavy).")
        print(
            json.dumps(
                run_full_build.remote(
                    rebuild_base=rebuild_base, no_staging=no_staging
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    print("Running smoke (wiring proof) ...")
    result = smoke.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
