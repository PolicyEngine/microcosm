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

This app stands the corrected rebuild up on Modal. Two phases run inside one
container image built from the repo at the pinned ``uv.lock`` (policyengine-us
1.752.2, policyengine-core 3.26.11 — exactly the July-1 sparse build engine, so
the only change from that release is the #278 data fix):

1. **base rebuild** — ``tools/build_us_puf_support_base.py`` layers the #278
   PUF/childcare/immigration inputs onto a base support frame;
2. **calibrate + write the corrected sparse release** —
   ``tools/build_us_fiscal_refresh_release.py`` recalibrates the household
   weights (default sparse L0+refit), gating base-vs the certified DENSE
   f0af251 reference and export-vs-base for input-mass parity, and writes the
   release contract files.

A ``--smoke`` mode proves the plumbing cheaply before any real money is spent:
it wires up the HF secret, the volumes, and the gate machinery, inspects the
published base for the #278 columns, and exercises the fast base-vs-reference
input-mass gate (which legitimately *fails* on the broken published base — that
failure is the point of #278/#279 and demonstrates the gate fires end-to-end).

Licensing
---------
PUF-derived artifacts (the base H5, the dense reference H5, any donor PUF) move
only between Hugging Face and Modal compute here; nothing PUF-derived is written
to the git repo or printed to logs. Only column *names*, weighted *totals*, and
gate verdicts are surfaced.

Usage
-----
Smoke (cheap wiring proof; safe to run):

    modal run tools/modal_rebuild_us_sparse.py --smoke

Full corrected build (heavy; only with the main session's go-ahead):

    modal run --detach tools/modal_rebuild_us_sparse.py --full

See ``FULL_RUN_NOTES`` for the resource/cost estimate and the exact release
publish steps that follow a successful run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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

# Where the repo is baked inside the image.
IMAGE_REPO_ROOT = "/root/populace"
VENV_PYTHON = "/opt/venv/bin/python"

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
  1. rebuilds the base carrying the #278 columns (build_us_puf_support_base.py);
  2. recalibrates + writes the corrected sparse release
     (build_us_fiscal_refresh_release.py --base-h5 <rebuilt>
      --input-mass-reference-h5 <dense f0af251>), default L0+refit;
  3. persists the release dir + all gate verdicts to the output volume.

Post-run: review gates (input_mass_parity.json, calibration_diagnostics.json),
then publish from a machine with the release env via tools/publish_release.sh,
then certify with policyengine.py.
"""


# --------------------------------------------------------------------------- #
# Image                                                                        #
# --------------------------------------------------------------------------- #
# Build from the repo at its pinned uv.lock so the engine is exactly the
# July-1 sparse build's (policyengine-us 1.752.2, policyengine-core 3.26.11).
# Only the #278 data fix differs. h5py/tables/microunit come from the `us`
# extra.

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
                "UV_PROJECT_ENVIRONMENT": "/opt/venv",
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
    tail = "\n".join((proc.stdout or "").splitlines()[-40:])
    err_tail = "\n".join((proc.stderr or "").splitlines()[-40:])
    if proc.returncode != 0:
        print(f"[exit {proc.returncode}] stdout tail:\n{tail}", flush=True)
        print(f"[exit {proc.returncode}] stderr tail:\n{err_tail}", flush=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": tail,
        "stderr_tail": err_tail,
    }


def _hf_download(repo: str, filename: str, *, revision: str, repo_type: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo,
        filename=filename,
        revision=revision,
        repo_type=repo_type,
    )


def _prepend_build_src() -> None:
    src = f"{IMAGE_REPO_ROOT}/packages/populace-build/src"
    if src not in sys.path:
        sys.path.insert(0, src)


def _load_us_frame(path: str):
    _prepend_build_src()
    from populace.build.us_runtime.l0_refit_export import load_us_frame

    return load_us_frame(path)


def _engine_input_variables() -> tuple[str, ...]:
    _prepend_build_src()
    from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

    return tuple(PolicyEngineUSEngine().variables())


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

    Steps, each recorded in the returned verdict dict and staged to the output
    volume under ``smoke/``:

    1. HF secret works: authenticate + report the token's whoami (name only).
    2. Fetch the published sparse base and the DENSE f0af251 reference from HF
       (PUF-derived — fetched only here, never logged/committed).
    3. Inspect the published base for the four #278 input columns and confirm
       the DENSE reference carries their weighted mass.
    4. Run the fast base-vs-reference input-mass parity gate on the published
       sparse base. On the *broken* published base this legitimately FAILS —
       demonstrating the #279 gate fires end to end (its output shape is the
       deliverable, not a pass).
    5. Confirm the consumer-facts ledger artifact is present for --ledger-facts.
    6. Exercise the base-rebuild + fiscal-refresh entrypoints' import + arg
       plumbing under the built env.
    """
    out_dir = Path(OUTPUT_MOUNT) / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict: dict = {"steps": {}}

    # 1. HF secret / whoami --------------------------------------------------- #
    from huggingface_hub import HfApi

    who = HfApi().whoami()
    verdict["steps"]["hf_whoami"] = {
        "name": who.get("name"),
        "type": who.get("type"),
        "orgs": [o.get("name") for o in who.get("orgs", [])],
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
    }
    print("hf whoami:", verdict["steps"]["hf_whoami"], flush=True)

    # 2. Fetch base + dense reference ---------------------------------------- #
    base_path = _hf_download(
        POPULACE_US_REPO,
        DEFAULT_BASE_FILENAME,
        revision=SPARSE_DEFAULT_REVISION,
        repo_type="dataset",
    )
    reference_path = _hf_download(
        POPULACE_US_REPO,
        DEFAULT_BASE_FILENAME,
        revision=DENSE_REFERENCE_REVISION,
        repo_type="dataset",
    )
    verdict["steps"]["downloads"] = {
        "sparse_base_bytes": Path(base_path).stat().st_size,
        "dense_reference_bytes": Path(reference_path).stat().st_size,
    }

    # 3+4. Column inspection + input-mass parity gate ------------------------ #
    _prepend_build_src()
    from populace.build.gates import input_mass_parity_gate
    from populace.build.us_runtime.input_mass import us_input_mass_totals

    input_variables = _engine_input_variables()
    base_frame = _load_us_frame(base_path)
    reference_frame = _load_us_frame(reference_path)

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
        base_totals,
        reference_totals,
        candidate_name="published_sparse_base",
        reference_name="dense_f0af251_reference",
        relative_tolerance=0.5,
        minimum_reference_total=1e9,
    )
    verdict["steps"]["input_mass_parity_gate"] = {
        "passed": gate.passed,
        "n_failures": len(gate.failures),
        "failures_head": list(gate.failures)[:12],
        "columns_checked": gate.details.get("columns_checked"),
        "worst_drifts": gate.details.get("worst_drifts"),
    }
    print(
        f"input_mass_parity_gate on published sparse base: passed={gate.passed} "
        f"n_failures={len(gate.failures)}",
        flush=True,
    )

    # 5. Ledger facts present? ----------------------------------------------- #
    facts_path = Path(FACTS_MOUNT) / CONSUMER_FACTS_FILENAME
    verdict["steps"]["ledger_facts"] = {
        "path": str(facts_path),
        "exists": facts_path.exists(),
        "bytes": facts_path.stat().st_size if facts_path.exists() else 0,
    }

    # 6. Entrypoint plumbing -------------------------------------------------- #
    help_probe = _run([VENV_PYTHON, "tools/build_us_puf_support_base.py", "--help"])
    verdict["steps"]["base_builder_help"] = {
        "returncode": help_probe["returncode"],
        "ok": help_probe["returncode"] == 0,
    }
    refresh_help = _run(
        [VENV_PYTHON, "tools/build_us_fiscal_refresh_release.py", "--help"]
    )
    verdict["steps"]["fiscal_refresh_help"] = {
        "returncode": refresh_help["returncode"],
        "ok": refresh_help["returncode"] == 0,
    }

    verdict_path = out_dir / "smoke_verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    output_volume.commit()
    verdict["verdict_path"] = str(verdict_path)
    return verdict


# --------------------------------------------------------------------------- #
# Full build (heavy — gated on go-ahead)                                        #
# --------------------------------------------------------------------------- #

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
    base_h5_override: str | None = None,
    epochs: int = 1500,
    skip_out_of_sample_reforms: bool = False,
    allow_input_mass_drift: bool = False,
    release_id: str | None = None,
) -> dict:
    """Rebuild the base carrying the #278 inputs, then calibrate + write the
    corrected sparse release, gated against the dense f0af251 reference.

    NOTE: this is the expensive path (target compilation + L0+refit + reform
    validation). Do not launch without the main session's go-ahead.
    """
    work = Path(OUTPUT_MOUNT) / "full"
    work.mkdir(parents=True, exist_ok=True)
    result: dict = {}

    reference_path = _hf_download(
        POPULACE_US_REPO,
        DEFAULT_BASE_FILENAME,
        revision=DENSE_REFERENCE_REVISION,
        repo_type="dataset",
    )
    result["reference_h5"] = reference_path

    # Phase 1 — base rebuild carrying the #278 columns.
    # Start from the published DENSE base support frame (it carries the full
    # donor surface — SCF/SIPP/ORG/MEPS/ACS — and, if it retains the raw ASEC
    # scratch columns, lets build_us_puf_support_base.py derive childcare), and
    # layer the #278 PUF/childcare/immigration inputs onto it. The processed PUF
    # donor is fetched from the us-data model repo (licensed; never logged).
    base_support_h5 = base_h5_override or reference_path
    puf_h5 = _hf_download(
        US_DATA_MODEL_REPO,
        PUF_2024_PATH,
        revision=US_DATA_DONOR_REVISION,
        repo_type="model",
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
    rebuilt_base_h5 = base_out / "base_populace_us_2024_puf_support.h5"

    # Phase 2 — calibrate + write the corrected sparse release.
    facts_path = Path(FACTS_MOUNT) / CONSUMER_FACTS_FILENAME
    release_out = work / "release"
    cmd = [
        VENV_PYTHON,
        "tools/build_us_fiscal_refresh_release.py",
        "--base-h5",
        str(rebuilt_base_h5),
        "--ledger-facts",
        str(facts_path),
        "--input-mass-reference-h5",
        reference_path,
        "--out",
        str(release_out),
        "--epochs",
        str(epochs),
        "--no-staging",
    ]
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
def main(full: bool = False):
    """Default: run the smoke path. Pass --full only with the go-ahead."""
    if full:
        print("Launching FULL corrected build (heavy).")
        print(json.dumps(run_full_build.remote(), indent=2, sort_keys=True))
        return
    print("Running smoke (wiring proof) ...")
    result = smoke.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
