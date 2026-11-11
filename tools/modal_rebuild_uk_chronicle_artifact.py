"""Modal app: rebuild the pinned Chronicle UK consumer artifact off this machine.

The UK national build pins one Chronicle consumer artifact by digest
(``packages/microcosm-build/src/microcosm/build/uk/chronicle_feed.json``):
``build-bundle --suite uk`` then ``build-consumer-artifact`` at one Chronicle
commit. Chronicle is public and its UK suite builds from committed source
files plus two DESNZ workbooks fetched from the publisher by pinned sha256
(``LEDGER_SOURCE_ARTIFACT_FETCH=1``), so anyone can rebuild it. The bundle
takes tens of gigabytes of disk, more than a busy laptop has spare, so this
runs it in a Modal container and keeps only the artifact.

The container clones Chronicle at the pinned commit, syncs its own
``uv.lock``, builds the bundle and the artifact, and hashes both artifact
files against the pin. Results go to the ``microcosm-uk-chronicle`` Modal
volume under ``<commit>/<facts sha256>/``, with a receipt beside them.

Check the pin and print the plan (plain Python; contacts no Modal service
and sends nothing)::

    python tools/modal_rebuild_uk_chronicle_artifact.py

Rebuild (paid, a few hours; no credential goes to Modal)::

    modal run --detach tools/modal_rebuild_uk_chronicle_artifact.py --build

Rebuild and upload to the private Hugging Face repo when the digests match
the pin. ``modal run`` creates the app before its entrypoint runs, so the
token is attached only when ``MICROCOSM_UK_CHRONICLE_MODAL_UPLOAD=1`` says
the upload is deliberate. ``HF_TOKEN`` needs write access to
``policyengine/microcosm-uk-build-inputs``; give it to this one command. It
goes to Modal as an ephemeral secret for this run only: Modal holds it for
the run, but no named workspace secret keeps it::

    HF_TOKEN=… MICROCOSM_UK_CHRONICLE_MODAL_UPLOAD=1 modal run --detach tools/modal_rebuild_uk_chronicle_artifact.py --build --upload

An artifact whose digests differ from the pin is never uploaded under the
pinned path; with ``--upload`` its receipt and bundle reports go to
``diagnostic/`` so the difference can be traced.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

APP_NAME = "microcosm-uk-chronicle-rebuild"
VOLUME_NAME = "microcosm-uk-chronicle"
VOLUME_MOUNT = "/artifacts"
HF_REPO = "policyengine/microcosm-uk-build-inputs"
CHRONICLE_URL = "https://github.com/PolicyEngine/chronicle.git"
PYTHON_VERSION = "3.14"
UV_VERSION = "0.11.7"
HF_HUB_VERSION = "1.18.0"
FEED_PIN = Path(__file__).resolve().parents[1] / (
    "packages/microcosm-build/src/microcosm/build/uk/chronicle_feed.json"
)


def _load_pin() -> dict:
    pin = json.loads(FEED_PIN.read_text())
    if pin.get("country") != "uk" or pin.get("source_repo") != "PolicyEngine/chronicle":
        raise SystemExit(f"{FEED_PIN} is not the UK Chronicle feed pin.")
    return pin


PIN = _load_pin() if modal.is_local() else json.loads(os.environ["CHRONICLE_PIN"])


#: The launcher's opt-in for attaching HF_TOKEN to a ``modal run``.
UPLOAD_OPT_IN = "MICROCOSM_UK_CHRONICLE_MODAL_UPLOAD"


def _secrets() -> list[modal.Secret]:
    # An ephemeral secret for one deliberate upload run only: modal run sends
    # the app's secrets before the entrypoint can refuse. The container must
    # declare the same dependencies as the launcher, and there the secret
    # itself supplies HF_TOKEN, so both sides agree.
    if modal.is_local() and os.environ.get(UPLOAD_OPT_IN) != "1":
        return []
    token = os.environ.get("HF_TOKEN")
    return [modal.Secret.from_dict({"HF_TOKEN": token})] if token else []


image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git", "curl", "time")
    .pip_install(f"uv=={UV_VERSION}", f"huggingface_hub=={HF_HUB_VERSION}")
    .run_commands(f"uv python install {PYTHON_VERSION}")
    .env(
        {
            "CHRONICLE_PIN": json.dumps(PIN, sort_keys=True),
            "UV_LINK_MODE": "copy",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
)
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(argv: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> dict:
    started = time.monotonic()
    with log.open("ab") as handle:
        handle.write(f"\n$ {' '.join(argv)}\n".encode())
        handle.flush()
        completed = subprocess.run(
            ["/usr/bin/time", "-v", *argv],
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        tail = log.read_text(errors="replace")[-4000:]
        raise RuntimeError(f"{argv[:4]} exited {completed.returncode}:\n{tail}")
    return {"argv": argv, "seconds": round(time.monotonic() - started, 1)}


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=_secrets(),
    cpu=8.0,
    memory=98304,
    ephemeral_disk=512 * 1024,  # Modal's floor; the bundle needs tens of GB
    timeout=8 * 3600,
)
def rebuild(upload: bool) -> dict:
    """Clone, build, hash and (optionally) upload; return the receipt."""

    commit = PIN["source_commit"]
    work = Path("/work")
    repo = work / "chronicle"
    bundle = work / "bundle"
    artifact = work / "artifact"
    work.mkdir(parents=True, exist_ok=True)
    log = work / "build.log"
    receipt: dict = {
        "schema_version": 1,
        "kind": "microcosm.uk_chronicle_artifact_rebuild",
        "started_at": _now(),
        "pin": PIN,
        "chronicle_commit": commit,
        "steps": [],
    }
    env = {
        **os.environ,
        "LEDGER_SOURCE_ARTIFACT_FETCH": "1",
        "UV_PYTHON": PYTHON_VERSION,
        "UV_PROJECT_ENVIRONMENT": str(work / "venv"),
    }
    env.pop("HF_TOKEN", None)  # the build itself never needs the Hub
    subprocess.run(["git", "clone", "--quiet", CHRONICLE_URL, str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", commit], check=True)
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != commit:
        raise RuntimeError(f"checked out {head}, not the pinned {commit}")
    for argv in (
        ["uv", "sync", "--locked"],
        [
            "uv",
            "run",
            "--no-sync",
            "chronicle",
            "build-bundle",
            "--suite",
            "uk",
            "--out",
            str(bundle),
            "--replace",
        ],
        [
            "uv",
            "run",
            "--no-sync",
            "chronicle",
            "build-consumer-artifact",
            "--facts",
            str(bundle),
            "--out",
            str(artifact),
            "--replace",
        ],
    ):
        receipt["steps"].append(_run(argv, cwd=repo, env=env, log=log))
    receipt["environment"] = {
        "python": subprocess.check_output(
            ["uv", "run", "--no-sync", "python", "--version"],
            cwd=repo,
            env=env,
            text=True,
        ).strip(),
        "uv_lock_sha256": _sha256(repo / "uv.lock"),
        "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    }
    facts = artifact / "consumer_facts.jsonl"
    manifest = artifact / "manifest.json"
    observed = {
        "facts_sha256": _sha256(facts),
        "manifest_sha256": _sha256(manifest),
        "facts_size_bytes": facts.stat().st_size,
        "manifest": json.loads(manifest.read_text()),
    }
    matches = {
        "facts_sha256": observed["facts_sha256"] == PIN["facts_sha256"],
        "manifest_sha256": observed["manifest_sha256"] == PIN["manifest_sha256"],
        "fact_row_count": observed["manifest"].get("fact_row_count")
        == PIN["fact_row_count"],
    }
    receipt.update(observed=observed, matches=matches, reproduced=all(matches.values()))

    # Keep the artifact and the small bundle reports; drop the bundle.
    out = Path(VOLUME_MOUNT) / commit / observed["facts_sha256"]
    out.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "consumer_facts.jsonl"):
        shutil.copyfile(artifact / name, out / name)
    reports = out / "bundle-reports"
    reports.mkdir(exist_ok=True)
    for name in ("coverage.json", "source_packages.json", "reports/build_bundle.json"):
        source = bundle / name
        if source.exists():
            shutil.copyfile(source, reports / Path(name).name)
    shutil.copyfile(log, out / "build.log")
    receipt["volume_path"] = f"{VOLUME_NAME}:{out.relative_to(VOLUME_MOUNT)}"

    if upload:
        receipt["upload"] = _upload(out, receipt)
    receipt["finished_at"] = _now()
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    volume.commit()
    return receipt


def _upload(out: Path, receipt: dict) -> dict:
    from huggingface_hub import HfApi

    if not os.environ.get("HF_TOKEN"):
        return {"uploaded": False, "reason": "HF_TOKEN was not exported locally"}
    api = HfApi(token=os.environ["HF_TOKEN"])
    commit = receipt["chronicle_commit"]
    if receipt["reproduced"]:
        prefix = f"chronicle-uk/{commit}"
        allow = ["manifest.json", "consumer_facts.jsonl"]
        message = f"Add the Chronicle UK consumer artifact rebuilt at {commit}"
    else:
        # Never under the pinned path: only the evidence for a diff.
        prefix = (
            f"diagnostic/chronicle-uk/{commit}/{receipt['observed']['facts_sha256']}"
        )
        allow = ["manifest.json", "bundle-reports/*", "build.log"]
        message = (
            f"Diagnostic: Chronicle UK rebuild at {commit} did not reproduce the pin"
        )
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    allow.append("receipt.json")
    result = api.upload_folder(
        repo_id=HF_REPO,
        repo_type="dataset",
        folder_path=str(out),
        path_in_repo=prefix,
        allow_patterns=allow,
        commit_message=message,
    )
    return {"uploaded": True, "path_in_repo": prefix, "commit": result.oid}


def plan(*, upload: bool = False) -> dict:
    """Print what a rebuild would do; contacts no Modal service."""

    summary = {
        "chronicle_commit": PIN["source_commit"],
        "expected_facts_sha256": PIN["facts_sha256"],
        "expected_manifest_sha256": PIN["manifest_sha256"],
        "expected_rows": PIN["fact_row_count"],
        "volume": VOLUME_NAME,
        "upload_repo": HF_REPO if upload else None,
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
    }
    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(build: bool = False, upload: bool = False) -> None:
    sent = os.environ.get(UPLOAD_OPT_IN) == "1" and bool(os.environ.get("HF_TOKEN"))
    if not build:
        raise SystemExit(
            "modal run is only for a paid rebuild: pass --build. To plan, run "
            "`python tools/modal_rebuild_uk_chronicle_artifact.py`, which "
            "contacts no Modal service."
            + (
                f" HF_TOKEN was already sent to Modal ({UPLOAD_OPT_IN}=1)."
                if sent
                else ""
            )
        )
    if upload and not sent:
        raise SystemExit(
            f"--upload needs {UPLOAD_OPT_IN}=1 and HF_TOKEN given to this command "
            "(HF_TOKEN=… VAR=1 modal run …), not exported to the shell."
        )
    if sent and not upload:
        print(
            f"HF_TOKEN was sent to Modal ({UPLOAD_OPT_IN}=1) but this run does "
            "not upload; drop the variable when not uploading."
        )
    plan(upload=upload)
    receipt = rebuild.remote(upload)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    plan(upload="--upload" in sys.argv[1:])
    print(
        "Plan only. To rebuild (paid): modal run --detach "
        "tools/modal_rebuild_uk_chronicle_artifact.py --build [--upload]"
    )
