"""Modal app: run ``tools/run_uk_cut.py`` off this machine.

For when the build Mac is busy, short of disk or asleep. The container
clones microcosm at a pushed commit on ``main``, syncs that commit's own
lock with the ``uk`` extra, and runs the same one command with its work
directory on the ``microcosm-uk-cuts`` Modal volume, so fetched inputs and
every step's outputs survive the container and a rerun resumes.

The build hands its uploads to the separately installed ``microcosm-emitter``
(microcosm#1410, #1460), which the runner's preflight starts to check write
access, and the workspace lock pins only its interface. So the image installs
the emitter on its own, at or above the release
``minimum_emitter_version.json`` names in the checkout that launches the app,
as the US Modal image does. The build enforces the minimum of the commit it
clones, so the plan refuses a commit whose minimum is higher than the
launching checkout's; launch from a checkout at or after the commit. As in the
US image, the image first drops the compromised eMudhra certificate
authorities (an upgraded CA store, a hash-pinned pip, uv 0.13.4) before it
downloads anything else.

Processing UK Data Service microdata on Modal needs the data owners' OK
before first use (``docs/uk-cut-anyone.md``). The volume keeps UKDS inputs
and outputs, and anyone in the Modal workspace can read it.

Plan only (plain Python; contacts no Modal service and sends nothing)::

    python tools/modal_uk_cut.py --commit <sha on main> --label sat24

``modal run`` creates the app before its entrypoint runs, so it attaches the
credentials only when ``MICROCOSM_UK_CUT_MODAL_RUN=1`` says the run is
deliberate, and the entrypoint refuses anything but ``--run``. Run (paid;
about three hours). ``HF_TOKEN`` and the registered
``MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY`` go to Modal as an ephemeral secret
for this run only: Modal holds both for the run, but no named workspace
secret keeps them. A key used here is then also held on Modal, which its
registry entry's ``held_in`` must say::

    MICROCOSM_UK_CUT_MODAL_RUN=1 modal run --detach tools/modal_uk_cut.py --commit <sha> --label sat24 --run
    MICROCOSM_UK_CUT_MODAL_RUN=1 modal run --detach tools/modal_uk_cut.py --commit <sha> --label sat24 --run --publish

Read a finished run's receipt::

    modal volume get microcosm-uk-cuts cuts/<label>-<sha9>/cut_receipt.json -
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import modal

APP_NAME = "microcosm-uk-cut"
VOLUME_NAME = "microcosm-uk-cuts"
WORK = "/work"
REPO_URL = "https://github.com/PolicyEngine/microcosm.git"
CHECKOUT = "/microcosm"
PYTHON_VERSION = "3.13"  # the version PR CI tests the lock with
# The compromised eMudhra (emSign) roots, whose intermediate keys were
# disclosed on 2026-10-20, are dropped as in the US Modal image
# (tools/modal_us_stage_plan.py, whose values the tests compare with these).
# uv verifies HTTPS against a CA list compiled into it, and 0.13.4 is the
# first release whose list drops them.
UV_VERSION = "0.13.4"
# Modal's cached debian_slim base carries pip 25.3, whose bundled certifi
# still contains emSign ECC Root CA - G3, so the image replaces it with this
# pip before downloading anything else. The wheel's sha256, as PyPI publishes
# it, not the connection, makes that one download authentic.
PIP_VERSION = "26.2.2"
PIP_WHEEL_SHA256 = "fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
PIP_BOOTSTRAP_REQUIREMENTS = "/tmp/pip-bootstrap.txt"
SHA = re.compile(r"^[0-9a-f]{40}$")
#: The launcher's opt-in for attaching credentials to a ``modal run``.
RUN_OPT_IN = "MICROCOSM_UK_CUT_MODAL_RUN"
#: The signing-key registry's source, and where the image carries a copy
#: (beside this module, which Modal places in /root).
REGISTRY_SOURCE = Path(__file__).resolve().parents[1] / (
    "packages/microcosm-data/src/microcosm/data/gate_signing_keys.py"
)
REGISTRY_REMOTE = "/root/gate_signing_keys.py"
#: The oldest emitter release a build starts, written once in microcosm-build's
#: package data (``MINIMUM_EMITTER_VERSION`` in
#: ``microcosm.build.telemetry_emitter``); read by its path, since the launcher
#: cannot import microcosm. ``plan`` also reads the cloned commit's copy.
MINIMUM_EMITTER_VERSION_PATH = (
    "packages/microcosm-build/src/microcosm/build/minimum_emitter_version.json"
)
MINIMUM_EMITTER_VERSION_FILE = (
    Path(__file__).resolve().parents[1] / MINIMUM_EMITTER_VERSION_PATH
)
EMITTER_RELEASE = re.compile(r"\d+\.\d+\.\d+")
#: Where the image installs the emitter command, outside the environment the
#: container syncs, and the variable through which the build's interface
#: finds it (the interface also searches PATH).
EMITTER_BIN_DIR = "/usr/local/bin"
EMITTER_COMMAND = f"{EMITTER_BIN_DIR}/microcosm-emitter"
EMITTER_COMMAND_ENV = "MICROCOSM_EMITTER_COMMAND"


def load_gate_signing_keys(path: Path | None = None) -> ModuleType:
    """``microcosm.data.gate_signing_keys``, loaded from its source file.

    The launcher runs under the Modal CLI's own interpreter, and the container
    imports this module before it clones microcosm, so neither can import
    microcosm.data. The registry module needs only the standard library:
    locally it is read from this checkout, in the container from the image's
    copy.
    """

    if path is None:
        path = REGISTRY_SOURCE if modal.is_local() else Path(REGISTRY_REMOTE)
    spec = importlib.util.spec_from_file_location("uk_cut_gate_signing_keys", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the signing-key registry from {path}.")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


registry = load_gate_signing_keys()


def pip_bootstrap_command() -> str:
    """The shell step that replaces the base's pip before any package download.

    ``--require-hashes`` refuses any file whose sha256 is not
    ``PIP_WHEEL_SHA256``, whatever index or certificate answers the request.
    """

    requirement = f"pip=={PIP_VERSION} --hash=sha256:{PIP_WHEEL_SHA256}"
    path = shlex.quote(PIP_BOOTSTRAP_REQUIREMENTS)
    return (
        f"printf '%s\\n' {shlex.quote(requirement)} > {path}"
        f" && python -m pip install --require-hashes --no-deps -r {path}"
        f" && rm {path}"
    )


def _minimum_release(text: str, source: str) -> str:
    """The release a ``minimum_emitter_version.json`` text names; ``source`` names it."""

    try:
        data = json.loads(text)
    except ValueError as error:
        raise SystemExit(
            f"cannot read the emitter's minimum release from {source} "
            f"({type(error).__name__})."
        ) from None
    version = data.get("minimum_emitter_version") if isinstance(data, dict) else None
    if not isinstance(version, str) or EMITTER_RELEASE.fullmatch(version) is None:
        raise SystemExit(f'{source} must hold {{"minimum_emitter_version": "X.Y.Z"}}.')
    return version


def minimum_emitter_version() -> str:
    """The oldest emitter release a build starts, as the launching checkout says.

    Refuses a ``MINIMUM_EMITTER_VERSION_FILE`` that is missing or does not
    hold ``{"minimum_emitter_version": "X.Y.Z"}``.
    """

    path = MINIMUM_EMITTER_VERSION_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as error:
        raise SystemExit(
            f"cannot read the emitter's minimum release from {path} "
            f"({type(error).__name__})."
        ) from None
    return _minimum_release(text, str(path))


def emitter_requirement() -> str:
    """What the image installs: at least the oldest emitter release a build starts."""

    return f"microcosm-emitter>={minimum_emitter_version()}"


def commit_minimum_emitter_version(commit: str) -> str | None:
    """The oldest emitter release the build at ``commit`` starts.

    ``None`` when the commit has no ``minimum_emitter_version.json``: commits
    before microcosm#1522 kept an older minimum in code. ``commit`` must be
    in the launching checkout, as ``plan`` has checked.
    """

    shown = subprocess.run(
        ["git", "show", f"{commit}:{MINIMUM_EMITTER_VERSION_PATH}"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    if shown.returncode != 0:
        return None
    return _minimum_release(
        shown.stdout, f"{MINIMUM_EMITTER_VERSION_PATH} at {commit[:9]}"
    )


def _release_order(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _secrets() -> list[modal.Secret]:
    # The launcher attaches the credentials only to a deliberate run: modal
    # run sends the app's secrets before the entrypoint can refuse. In the
    # container the environment carries the same values the launcher
    # passed, so the dependency lists agree.
    if modal.is_local() and os.environ.get(RUN_OPT_IN) != "1":
        return []
    values = {
        name: os.environ[name]
        for name in ("HF_TOKEN", registry.UK_GATE_SIGNING_KEY_ENV)
        if os.environ.get(name)
    }
    return [modal.Secret.from_dict(values)] if values else []


def _image() -> modal.Image:
    image = (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        # Naming ca-certificates upgrades the base's CA store, which git and
        # the standard library's ssl module read, to the distribution's
        # current release, without the eMudhra roots.
        .apt_install("git", "build-essential", "libhdf5-dev", "ca-certificates")
        # Before the base's pip downloads anything else.
        .run_commands(pip_bootstrap_command())
        .pip_install(f"uv=={UV_VERSION}")
        .env(
            {
                "UV_LINK_MODE": "copy",
                "PYTHONDONTWRITEBYTECODE": "1",
                # The image's Python meets the lock and the emitter.
                "UV_PYTHON_DOWNLOADS": "never",
                EMITTER_COMMAND_ENV: EMITTER_COMMAND,
            }
        )
    )
    if modal.is_local():
        # The newest release at or above the minimum when Modal first builds
        # this step; Modal reuses the layer while the minimum is unchanged.
        # Running its version fails the image build if the command cannot run.
        image = image.run_commands(
            f"UV_TOOL_BIN_DIR={EMITTER_BIN_DIR} uv tool install "
            f"{shlex.quote(emitter_requirement())}",
            f"{EMITTER_COMMAND} version",
        )
        # Last, because Modal refuses build steps after a local file.
        image = image.add_local_file(str(REGISTRY_SOURCE), REGISTRY_REMOTE)
    return image


image = _image()
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
    # The sync installs third-party packages and needs no credential; the
    # runner itself scopes the signing key to the steps that use it.
    credentials = {"HF_TOKEN", registry.UK_GATE_SIGNING_KEY_ENV}
    _sh(
        ["uv", "sync", "--all-packages", "--locked", "--extra", "uk"],
        cwd=CHECKOUT,
        env={k: v for k, v in env.items() if k not in credentials},
    )
    argv = [
        f"{CHECKOUT}/.venv/bin/python",
        "tools/run_uk_cut.py",
        "all",
        "--label",
        label,
        "--work-dir",
        WORK,
        # A volume reports no meaningful free space.
        "--min-free-gb",
        "0",
        *extra,
    ]
    if publish:
        argv.append("--publish")
    try:
        completed = subprocess.run(argv, cwd=CHECKOUT, env=env, check=False)
    finally:
        volume.commit()
    return completed.returncode


def plan(commit: str, label: str, *, publish: bool = False) -> list[str]:
    """Check a cut's commit and label, print the plan, and name missing credentials.

    Contacts no Modal service, so planning sends nothing anywhere.
    """

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
    # The image installs this checkout's minimum and the build enforces the
    # commit's, so a higher one at the commit would stop preflight in the
    # paid container, where the emitter's upgrade advice cannot be followed.
    image_minimum = minimum_emitter_version()
    commit_minimum = commit_minimum_emitter_version(commit)
    if commit_minimum is not None and (
        _release_order(commit_minimum) > _release_order(image_minimum)
    ):
        raise SystemExit(
            f"{commit[:9]} needs microcosm-emitter>={commit_minimum}, but this "
            f"checkout's image installs microcosm-emitter>={image_minimum}, which "
            "the build's preflight would refuse; launch from a checkout at or "
            f"after {commit[:9]}."
        )
    missing = [
        name
        for name in ("HF_TOKEN", registry.UK_GATE_SIGNING_KEY_ENV)
        if not os.environ.get(name)
    ]
    print(
        f"cut {label} at {commit[:9]} on Modal; publish={publish}; "
        f"work dir {VOLUME_NAME}:{WORK}; emitter microcosm-emitter>={image_minimum}; "
        f"missing credentials: {missing or 'none'}"
    )
    return missing


@app.local_entrypoint()
def main(
    commit: str,
    label: str,
    run: bool = False,
    publish: bool = False,
    retry_national: bool = False,
) -> None:
    opted_in = os.environ.get(RUN_OPT_IN) == "1"
    if not run or not opted_in:
        raise SystemExit(
            "modal run is only for a deliberate, paid run: set "
            f"{RUN_OPT_IN}=1 and pass --run. To plan, run "
            "`python tools/modal_uk_cut.py --commit <sha> --label <label>`, "
            "which contacts no Modal service."
            + (
                f" The credentials in this environment were already sent to "
                f"Modal ({RUN_OPT_IN}=1); nothing was built."
                if opted_in
                else ""
            )
        )
    missing = plan(commit, label, publish=publish)
    if missing:
        raise SystemExit(
            f"give {', '.join(missing)} to this command (VAR=… modal run …), "
            "not exported to the shell, then --run again."
        )
    extra = ["--retry-national"] if retry_national else []
    code = cut.remote(commit, label, publish, extra)
    raise SystemExit(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plan a UK cut on Modal.")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--publish", action="store_true")
    options = parser.parse_args()
    plan(options.commit, options.label, publish=options.publish)
    print(
        f"Plan only. To build (paid): {RUN_OPT_IN}=1 modal run --detach "
        f"tools/modal_uk_cut.py --commit {options.commit} --label "
        f"{options.label} --run" + (" --publish" if options.publish else "")
    )
