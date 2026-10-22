"""Modal app: run ``tools/run_uk_cut.py`` off this machine.

For when the build Mac is busy, short of disk or asleep. The container
clones microcosm at a pushed commit on ``main``, syncs that commit's own
lock with the ``uk`` extra, and runs the same one command with its work
directory on the ``microcosm-uk-cuts`` Modal volume, so fetched inputs and
every step's outputs survive the container and a rerun resumes.

Processing UK Data Service microdata on Modal needs the data owners' OK
before first use (``docs/uk-cut-anyone.md``). Nothing runs without
``--run``.

Plan only (free; builds nothing)::

    modal run tools/modal_uk_cut.py --commit <sha on main> --label sat24

Run (paid; about three hours; ``HF_TOKEN`` and the registered
``MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY`` must be exported locally; both
are passed to this run only and never stored as Modal secrets)::

    modal run --detach tools/modal_uk_cut.py --commit <sha> --label sat24 --run
    modal run --detach tools/modal_uk_cut.py --commit <sha> --label sat24 --run --publish

Read a finished run's receipt::

    modal volume get microcosm-uk-cuts cuts/<label>-<sha9>/cut_receipt.json -
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import modal

APP_NAME = "microcosm-uk-cut"
VOLUME_NAME = "microcosm-uk-cuts"
WORK = "/work"
REPO_URL = "https://github.com/PolicyEngine/microcosm.git"
CHECKOUT = "/microcosm"
PYTHON_VERSION = "3.13"  # the version PR CI tests the lock with
UV_VERSION = "0.11.7"
SIGNING_KEY_ENV = "MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY"
SHA = re.compile(r"^[0-9a-f]{40}$")


def _secrets() -> list[modal.Secret]:
    # Identical on both sides: the container's environment carries the same
    # two values the launcher passed, so the dependency lists agree.
    values = {
        name: os.environ[name]
        for name in ("HF_TOKEN", SIGNING_KEY_ENV)
        if os.environ.get(name)
    }
    return [modal.Secret.from_dict(values)] if values else []


image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git", "build-essential", "libhdf5-dev")
    .pip_install(f"uv=={UV_VERSION}")
    .env({"UV_LINK_MODE": "copy", "PYTHONDONTWRITEBYTECODE": "1"})
)
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _sh(argv: list[str], **kwargs) -> None:
    print("$", " ".join(argv), flush=True)
    subprocess.run(argv, check=True, **kwargs)


@app.function(
    image=image,
    volumes={WORK: volume},
    secrets=_secrets(),
    cpu=16.0,
    memory=98304,
    timeout=10 * 3600,
)
def cut(commit: str, label: str, publish: bool, extra: list[str]) -> int:
    """Clone at ``commit``, sync, and run the cut with its state on the volume."""

    _sh(["git", "clone", "--quiet", REPO_URL, CHECKOUT])
    _sh(["git", "-C", CHECKOUT, "checkout", "--quiet", commit])
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": f"{CHECKOUT}/.venv"}
    _sh(
        ["uv", "sync", "--all-packages", "--locked", "--extra", "uk"],
        cwd=CHECKOUT,
        env=env,
    )
    argv = [
        f"{CHECKOUT}/.venv/bin/python",
        "tools/run_uk_cut.py",
        "all",
        "--label",
        label,
        "--work-dir",
        WORK,
        # A volume reports no meaningful free space, and the Fleet ops
        # reservation is a property of the build Mac, not of this container.
        "--min-free-gb",
        "0",
        "--reservation-caller",
        "",
        *extra,
    ]
    if publish:
        argv.append("--publish")
    try:
        completed = subprocess.run(argv, cwd=CHECKOUT, env=env, check=False)
    finally:
        volume.commit()
    return completed.returncode


@app.local_entrypoint()
def main(
    commit: str,
    label: str,
    run: bool = False,
    publish: bool = False,
    retry_national: bool = False,
) -> None:
    if not SHA.match(commit):
        raise SystemExit("--commit must be a full 40-character sha on main.")
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,40}$", label):
        raise SystemExit("--label must be a short name (letters, digits, . _ -).")
    on_main = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    if on_main.returncode != 0:
        raise SystemExit(
            f"{commit} is not on origin/main here (git fetch first); Modal cuts "
            "build only from main."
        )
    missing = [n for n in ("HF_TOKEN", SIGNING_KEY_ENV) if not os.environ.get(n)]
    print(
        f"cut {label} at {commit[:9]} on Modal; publish={publish}; "
        f"work dir {VOLUME_NAME}:{WORK}; missing credentials: {missing or 'none'}"
    )
    if not run:
        print("Plan only; pass --run to build (paid).")
        return
    if missing:
        raise SystemExit(f"export {', '.join(missing)} before --run.")
    extra = ["--retry-national"] if retry_national else []
    code = cut.remote(commit, label, publish, extra)
    raise SystemExit(code)
