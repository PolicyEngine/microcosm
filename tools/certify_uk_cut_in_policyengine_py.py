"""Certify a published Microcosm UK build into policyengine.py, on a fresh branch.

policyengine.py serves the UK build its bundle manifest certifies; moving
that pin is a policyengine.py pull request and release. This script does the
mechanical part of that pull request the way policyengine.py#685 and #766 did
it by hand (``docs/engineering/skills/data-certification.md`` there), in a new
worktree branched from policyengine.py's ``origin/main``:

1. refuse if someone is already certifying this build (one writer per build);
2. read the national (and local-area) build from its own folder,
   ``releases/<release>/<build>/``, and check, before anything is written,
   that the regional route can certify;
3. ``scripts/bundle.py certify-data`` with the national build's folder
   manifest and the chosen regional route, at the build's own policyengine-uk
   and core;
4. ``uv lock`` and ``scripts/release_lock.py``;
5. ``bundle.py generate --include-tros``, the certified-constant rewrite of
   policyengine.py's own qualification library, the household snapshots,
   then ``bundle.py check --published-spm --include-tros --strict-tros``;
6. the UK Universal Credit floor (weighted national totals only) and the
   test suite;
7. a commit, and a pull request body with every judgment left to a person
   (the UC floor acceptance, dropped dataset overlays, failing tests).

It pushes and opens a draft pull request only with ``--push`` and
``--open-pr``. It never merges and never releases.

A build is named by its name in ``releases.json`` (``--national-build``,
``--dense-build``), which must list it at ``main`` and not have withdrawn it,
as policyengine.py's certification requires (policyengine.py#683). The
repository publishes no tags, and policyengine.py refuses a tagged manifest
from a repository with ``releases.json``. Each build is pinned at the commit
that published its folder: the last commit of its ``release_manifest.json`` on
``main``, since a build folder never changes. ``--national-data-commit`` and
``--dense-data-commit`` pin another commit that holds the folder instead.

Regional routes (pick one):

``--dense-build BUILD``       a Microcosm local-area build (ship together)
``--keep-regional-release``   the bundle's current regional release
``--area-coded-national-uri`` an enhanced-FRS release that claims the version
``--drop-regional-release``   no area routing (sim-api's UK default then
                              lacks constituency and local-authority runs)

A regional release built on another policyengine-uk is refused by
certification unless policyengine.py accepts a recorded mismatch:
``--accept-regional-model-mismatch REASON`` with ``--audited-by WHO`` and at
least one ``--regional-limitation TEXT``. The regional microdata is downloaded
from its build's folder at the build's commit, policyengine.py's own
``scripts/audit_regional_input_surface.py`` (policyengine.py#699) audits its
stored columns against the build and target models, and ``certify-data``
takes that evidence; any dropped column refuses (see
``docs/uk-cut-anyone.md``).

Needs ``HF_TOKEN`` (read ``policyengine/populace-uk-private``), PyPI, and
``gh`` for the writer check and the pull request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from microcosm.data.build_folder_manifests import require_offered_build
from microcosm.data.gate_signing_keys import UK_GATE_SIGNING_KEY_ENV
from microcosm.data.releases_record import build_folder

PEPY_REPO = "PolicyEngine/policyengine.py"
BUILD_REPO = "policyengine/populace-uk-private"
NATIONAL_RELEASE = "microcosm-uk-2024-25-national"
DENSE_RELEASE = "microcosm-uk-2024-25-dense"
PEPY_PYTHON = "3.13"  # certify-uk-release.yaml's interpreter
RELEASE_MANIFEST = "release_manifest.json"
#: A release or build name the emitter accepts for a folder, the rule
#: ``microcosm.data.releases_record`` and policyengine.py repeat.
BUILD_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
#: A full Hub commit ID. A branch moves, and a short ID can be ambiguous.
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
#: A build's folder manifest in the build repository, as certify-data takes it.
FOLDER_MANIFEST_URI = re.compile(
    rf"^hf://dataset/{re.escape(BUILD_REPO)}@(?P<commit>[^/@]+)"
    rf"/releases/(?P<release>[^/]+)/(?P<build>[^/]+)/{re.escape(RELEASE_MANIFEST)}$"
)
CONSTANT_PATTERN = r'(?P<pre>["@/-]){old}(?P<post>["/])'
#: policyengine.py's bundle manifest, which certify-data rewrites.
BUNDLE_MANIFEST = "src/policyengine/data/bundle/manifest.json"
CONSTANT_FILES = (
    "tests/test_models.py",
    "tests/test_release_manifests.py",
    "tests/test_uk_regions.py",
)


class CertifyError(RuntimeError):
    """A refusal the operator must act on."""


def folder_manifest_path(release: str, build: str) -> str:
    """The build's manifest in the build repository, inside its own folder."""

    return f"{build_folder(release, build)}/{RELEASE_MANIFEST}"


def manifest_uri(release: str, build: str, commit: str) -> str:
    """The build's folder manifest at ``commit``, as policyengine.py certifies it."""

    path = folder_manifest_path(release, build)
    return f"hf://dataset/{BUILD_REPO}@{commit}/{path}"


class BuildPin(NamedTuple):
    """A build in its own folder, and the commit it is read at."""

    release: str
    build: str
    commit: str


def folder_reference(uri: str | None) -> BuildPin | None:
    """The build a folder manifest URI in the build repository names, or
    ``None`` for any other URI (a tagged manifest, another repository)."""

    match = FOLDER_MANIFEST_URI.match(uri or "")
    if match is None:
        return None
    return BuildPin(match["release"], match["build"], match["commit"])


@dataclass(frozen=True)
class PublishedBuild:
    """A build read from its own folder, and what it was built with."""

    release: str
    build: str
    commit: str
    model: str
    core: str
    claims: tuple[str, ...]

    @property
    def folder(self) -> str:
        return build_folder(self.release, self.build)

    @property
    def manifest_path(self) -> str:
        return folder_manifest_path(self.release, self.build)

    @property
    def pin(self) -> BuildPin:
        return BuildPin(self.release, self.build, self.commit)

    @property
    def manifest_uri(self) -> str:
        return manifest_uri(self.release, self.build, self.commit)


def hub_token() -> str | None:
    return os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")


def hub_api() -> Any:
    """A Hub client that reads with this process's token."""

    # Set before importing the hub: its HTTP timeout constants are import-time.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    from huggingface_hub import HfApi

    return HfApi(token=hub_token())


def publishing_commit(api: Any, release: str, build: str) -> str:
    """The commit that published the build's folder.

    A build folder never changes after it is published, so the last commit
    of its manifest on ``main`` is the one that published it.
    """

    path = folder_manifest_path(release, build)
    try:
        infos = api.get_paths_info(
            BUILD_REPO, [path], expand=True, revision="main", repo_type="dataset"
        )
    except Exception as error:
        raise CertifyError(
            f"could not ask the Hub which commit published {path}: {error}"
        ) from error
    for info in infos:
        commit = getattr(getattr(info, "last_commit", None), "oid", None)
        if getattr(info, "path", None) == path and COMMIT_PATTERN.match(
            str(commit or "")
        ):
            return str(commit)
    raise CertifyError(
        f"{BUILD_REPO} has no {path} on main, so build {build!r} of {release!r} "
        "has no folder to certify from."
    )


def require_on_main(api: Any, commit: str) -> None:
    """Refuse a commit outside ``main``'s history.

    A build folder never changes on ``main``, but a commit on another ref,
    such as a Hub pull request's, may hold an edited copy of it.
    """

    try:
        history = api.list_repo_commits(
            BUILD_REPO, repo_type="dataset", revision="main"
        )
    except Exception as error:
        raise CertifyError(
            f"could not list main's history in {BUILD_REPO}: {error}"
        ) from error
    if not any(entry.commit_id == commit for entry in history):
        raise CertifyError(
            f"commit {commit} is not in main's history in {BUILD_REPO}; pin a "
            "commit on main that holds the build's folder."
        )


def read_published_build(
    release: str, build: str, *, commit: str | None = None, api: Any = None
) -> PublishedBuild:
    """Build ``build`` of ``release``, read from its own folder.

    ``releases.json`` at ``main``, downloaded afresh, must list the build and
    not have withdrawn it. The build is pinned at ``commit``, which must be in
    ``main``'s history, or else at the commit that published its folder, and
    its manifest there must name the release and the build and record the
    packages it was built with. A Hub failure is a refusal.
    """

    api = hub_api() if api is None else api
    try:
        require_offered_build(
            api,
            repo_id=BUILD_REPO,
            release=release,
            build=build,
            label="build",
            fresh=True,
        )
    except ValueError as error:
        # Name releases.json where the refusal does not already.
        message = str(error)
        if "releases.json" not in message:
            message = f"releases.json at main in {BUILD_REPO}: {message}"
        raise CertifyError(message) from error
    except Exception as error:
        # Anything but an absent record: the Hub could not be asked.
        raise CertifyError(
            f"could not read releases.json at main in {BUILD_REPO}: {error}"
        ) from error
    if commit is None:
        commit = publishing_commit(api, release, build)
    else:
        require_on_main(api, commit)
    path = folder_manifest_path(release, build)
    try:
        local = api.hf_hub_download(
            repo_id=BUILD_REPO,
            filename=path,
            repo_type="dataset",
            revision=commit,
            etag_timeout=60,
        )
        manifest = json.loads(Path(local).read_text())
    except Exception as error:
        raise CertifyError(f"could not read {path} at {commit}: {error}") from error
    named = (manifest.get("release"), manifest.get("build"))
    if named != (release, build):
        raise CertifyError(
            f"{path} at {commit} names release {named[0]!r} and build {named[1]!r}."
        )
    versions = {}
    for field in ("built_with_model_package", "built_with_core_package"):
        version = (manifest.get(field) or {}).get("version")
        if not isinstance(version, str) or not version:
            raise CertifyError(f"{path} at {commit} records no {field} version.")
        versions[field] = version
    return PublishedBuild(
        release=release,
        build=build,
        commit=commit,
        model=versions["built_with_model_package"],
        core=versions["built_with_core_package"],
        claims=tuple(
            claim["specifier"]
            for claim in manifest.get("compatible_model_packages", ())
            if claim.get("name") == "policyengine-uk"
        ),
    )


def covers(build: PublishedBuild, version: str) -> bool:
    """policyengine.py's rule: built with the version, or an exact claim of it."""

    return build.model == version or f"=={version}" in build.claims


def regional_flags(
    args: argparse.Namespace, bundle_record: dict, dense: PublishedBuild | None
) -> list[str]:
    """certify-data's regional flags for the route the operator chose.

    ``dense`` is the local-area build ``--dense-build`` names, as
    :func:`read_published_build` read it; the other routes ignore it.
    """

    if args.dense_build:
        if dense is None:
            raise CertifyError(f"local-area build {args.dense_build!r} was not read.")
        return ["--regional-manifest-uri", dense.manifest_uri]
    if args.keep_regional_release:
        uri = bundle_record.get("regional_source_manifest_uri")
        if not uri:
            raise CertifyError("the bundle routes no regional release to keep.")
        return ["--regional-manifest-uri", uri]
    if args.area_coded_national_uri:
        return [
            "--regional-manifest-uri",
            args.area_coded_national_uri,
            "--regional-release-kind",
            "area_coded_national",
        ]
    return ["--drop-regional-release"]


def kept_regional_pin(bundle_record: dict) -> BuildPin | None:
    """The local-area build ``--keep-regional-release`` keeps, or ``None``
    when the bundle routes regions to another repository's release.

    policyengine.py refuses a tagged manifest from the build repository, which
    has ``releases.json``, so keeping one is refused here first.
    """

    uri = bundle_record.get("regional_source_manifest_uri")
    if not uri:
        raise CertifyError("the bundle routes no regional release to keep.")
    pin = folder_reference(uri)
    if pin is None and f"/{BUILD_REPO}@" in uri:
        raise CertifyError(
            f"the bundle's regional release {uri} is not a build in its own "
            f"folder, and policyengine.py certifies only those from {BUILD_REPO}; "
            "name a local-area build with --dense-build."
        )
    return pin


#: Free disk each heavy step needs by default: the certified environment,
#: the snapshots and the test suite's fixtures (decimal bytes).
DEFAULT_MIN_FREE_BYTES = 55_000_000_000


def resource_guard(
    root: Path,
    *,
    max_load1: float | None = None,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
) -> dict[str, float | int]:
    """Refuse work before spawning a process on an overloaded or full host.

    A load limit depends on the machine, so there is none unless the operator
    sets one (``--max-load1``).
    """

    load1 = os.getloadavg()[0]
    while not root.exists():
        root = root.parent
    free = shutil.disk_usage(root).free
    overloaded = max_load1 is not None and load1 > max_load1
    if overloaded or free < min_free_bytes:
        maximum = "none" if max_load1 is None else f"{max_load1:g}"
        raise CertifyError(
            f"resource guard: load1={load1:.2f} (maximum {maximum}), "
            f"free={free / 1e9:.2f} GB (minimum {min_free_bytes / 1e9:g} decimal GB)."
        )
    return {"load1": load1, "free_bytes": free}


def check_test_temp_directory(directory: Path | None = None) -> Path:
    """Refuse a full suite whose temporary fixtures inherit an ancestor Git repo."""
    directory = Path(directory or tempfile.gettempdir()).expanduser().resolve()
    for ancestor in (directory, *directory.parents):
        if (ancestor / ".git").exists():
            raise CertifyError(
                f"Full-suite temporary directory {directory} is inside Git "
                f"repository {ancestor}. Version fixtures would inherit its tags "
                "and private disclosure audit fixtures correctly refuse Git paths. "
                "Set TMPDIR to a private writable directory outside every Git "
                "repository before certification. If workspace-only execution "
                "prevents that, report full-suite validation blocked; --skip-tests "
                "records a mechanical handoff only and does not qualify the cut."
            )
    return directory


def certification_path(relative: str, national_build: str) -> bool:
    """Files generated by this handoff, excluding local data and operator files."""
    snapshots = {
        "uk_couple_two_kids",
        "uk_model_surface",
        "uk_single_adult_employment_income",
        "uk_single_adult_no_income",
        "uk_single_parent_one_child",
        "us_married_two_kids_high_income",
        "us_model_surface",
        "us_single_adult_employment_income",
        "us_single_adult_no_income",
        "us_single_parent_one_child",
    }
    return relative in {
        "pyproject.toml",
        "uv.lock",
        *CONSTANT_FILES,
        f"changelog.d/certify-uk-{national_build}.changed.md",
        BUNDLE_MANIFEST,
        "src/policyengine/data/bundle/uk.trace.tro.jsonld",
        "src/policyengine/data/bundle/us.trace.tro.jsonld",
        *(
            f"tests/fixtures/household_calculator_snapshots/{name}.json"
            for name in snapshots
        ),
    }


def certification_changes(worktree: Path, national_build: str) -> list[str]:
    """Preserve operator files and refuse unrelated tracked/index changes."""
    tracked = subprocess.check_output(
        ["git", "diff", "--name-only", "-z", "HEAD"], cwd=worktree, text=True
    ).split("\0")
    unexpected = [
        path
        for path in tracked
        if path and not certification_path(path, national_build)
    ]
    if unexpected:
        raise CertifyError(
            "unrelated tracked changes are preserved; refusing to replay or commit: "
            + ", ".join(unexpected)
        )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=worktree,
        text=True,
    ).split("\0")
    changelog = f"changelog.d/certify-uk-{national_build}.changed.md"
    return sorted(
        {path for path in tracked if path}
        | {path for path in untracked if path == changelog}
    )


def validate_resume_worktree(
    repo: Path, worktree: Path, branch: str, base_commit: str, national_build: str
) -> list[str]:
    """Adopt only an unfinished, registered handoff at the exact resolved base."""
    if not worktree.is_dir():
        raise CertifyError(f"resume worktree {worktree} does not exist.")
    records, record = [], {}
    output = subprocess.check_output(
        ["git", "worktree", "list", "--porcelain", "-z"], cwd=repo, text=True
    )
    for field in output.split("\0"):
        if not field:
            if record:
                records.append(record)
                record = {}
        else:
            key, _, value = field.partition(" ")
            record[key] = value
    if record:
        records.append(record)
    entry = next(
        (
            item
            for item in records
            if "worktree" in item
            and Path(item["worktree"]).resolve() == worktree.resolve()
        ),
        None,
    )
    if entry is None or entry.get("branch") != f"refs/heads/{branch}":
        raise CertifyError(
            "resume worktree is not registered on the expected certification branch."
        )
    common_dirs = []
    for directory in (repo, worktree):
        common = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--git-common-dir"],
                cwd=directory,
                text=True,
            ).strip()
        )
        common_dirs.append((directory / common).resolve())
    if common_dirs[0] != common_dirs[1]:
        raise CertifyError("resume worktree belongs to another Git repository.")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        text=True,
    ).strip()
    if head != base_commit or entry.get("HEAD") != base_commit:
        raise CertifyError(
            "resume requires HEAD at the exact resolved base commit; committed "
            "descendants/history are preserved. Use a fresh worktree for another run."
        )
    if subprocess.check_output(
        ["git", "ls-files", "--unmerged", "-z"],
        cwd=worktree,
        text=True,
    ):
        raise CertifyError(
            "resume worktree has unresolved merges; no files were changed."
        )
    paths = certification_changes(worktree, national_build)
    manifest_path = BUNDLE_MANIFEST
    if manifest_path in paths:
        record = json.loads((worktree / manifest_path).read_text())["data_releases"][
            "uk"
        ]
        certified = (record.get("data_package") or {}).get("release_manifest_path")
        expected = folder_manifest_path(NATIONAL_RELEASE, national_build)
        if certified != expected:
            raise CertifyError(
                "resume worktree's modified manifest certifies another build."
            )
    return paths


def check_resumed_pins(
    worktree: Path,
    replayed: Sequence[str],
    national: PublishedBuild,
    regional_uri: str | None,
) -> None:
    """Refuse a resumed worktree whose modified bundle pins other builds.

    ``regional_uri`` is the regional manifest URI this run passes to
    certify-data, which writes it as the bundle's
    ``regional_source_manifest_uri``, or ``None`` for no regional routing.
    The first attempt's constant rewrite moved the certified constants to its
    own national commit and local-area build, so a resume that certifies
    others would leave them behind.
    """

    if BUNDLE_MANIFEST not in replayed:
        return
    record = json.loads((worktree / BUNDLE_MANIFEST).read_text())["data_releases"]["uk"]
    revision = (record.get("data_package") or {}).get("release_manifest_revision")
    if revision != national.commit:
        raise CertifyError(
            "resume worktree's modified manifest pins the national build at "
            f"{revision!r}, not {national.commit}; use a fresh worktree to "
            "certify another commit."
        )
    regional = record.get("regional_source_manifest_uri")
    if regional != regional_uri:
        raise CertifyError(
            f"resume worktree's modified manifest routes regions to {regional!r}, "
            f"not {regional_uri!r} as this run would; use a fresh worktree to "
            "change the regional route."
        )


def record_timing(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def pytest_command(
    python: str, arguments: list[str], *, startup_timeout: float = 300.0
) -> list[str]:
    """Diagnose imports before pytest's session/per-test clocks can start."""
    if not math.isfinite(startup_timeout) or startup_timeout <= 0:
        raise ValueError("pytest startup timeout must be positive and finite")
    bootstrap = (
        "import faulthandler, os, sys\n"
        "diagnostic_stream = os.fdopen(os.dup(2), 'w')\n"
        f"faulthandler.dump_traceback_later({startup_timeout!r}, "
        "repeat=True, file=diagnostic_stream)\n"
        "try:\n"
        "    import pytest\n"
        "    raise SystemExit(pytest.main(sys.argv[1:]))\n"
        "finally:\n"
        "    faulthandler.cancel_dump_traceback_later()\n"
        "    diagnostic_stream.close()\n"
    )
    return [python, "-c", bootstrap, *arguments]


@contextmanager
def timed_step(name: str, path: Path):
    """Make network preflight and local Python work visible, including failures."""
    started = time.monotonic()
    record = {"step": name, "started_at": datetime.now(UTC).isoformat()}
    print(f"[start] {name}", flush=True)
    try:
        yield record
        record.setdefault("returncode", 0)
    except Exception as error:
        record["returncode"] = getattr(error, "returncode", None)
        record["error"] = str(error)
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        record_timing(path, record)
        print(
            f"[done] {name}: {record['elapsed_seconds']:.3f}s "
            f"(exit {record.get('returncode')})",
            flush=True,
        )


def run_logged(argv: list[str], *, cwd: Path, env: dict[str, str], handle):
    """Stream child output into the diagnostic log while retaining its result."""
    output = []
    with subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            output.append(line)
        returncode = process.wait()
    return subprocess.CompletedProcess(argv, returncode, "".join(output), "")


#: What policyengine.py's commands never get from this process: the caller's
#: environment selectors (uv would sync, install into or run another
#: project's venv) and Microcosm's gate signing key, which no policyengine.py
#: step signs or verifies with, while its install and test steps run
#: third-party code.
WITHHELD_ENVIRONMENT = frozenset(
    {"UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV", UK_GATE_SIGNING_KEY_ENV}
)


#: Hugging Face tokens: policyengine.py's scripts and tests read the private
#: cut repositories with them, but package installs and syncs never need one.
HF_TOKEN_ENVIRONMENT = frozenset(
    {
        "HF_TOKEN",
        "HUGGING_FACE_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    }
)


def policyengine_py_environment(*, credentials: bool = True) -> dict[str, str]:
    """This process's environment, less what policyengine.py must not see.

    ``credentials=False`` also withholds the Hugging Face tokens, for steps
    that only install or resolve packages.
    """

    withheld = WITHHELD_ENVIRONMENT | (
        frozenset() if credentials else HF_TOKEN_ENVIRONMENT
    )
    return {k: v for k, v in os.environ.items() if k not in withheld}


class Shell:
    """Runs commands in the policyengine.py worktree and records each one."""

    def __init__(
        self,
        cwd: Path,
        log: Path,
        *,
        dry_run: bool = False,
        timings: Path | None = None,
        max_load1: float | None = None,
        min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    ) -> None:
        self.cwd = cwd
        self.log = log
        self.dry_run = dry_run
        self.timings = timings or log.with_suffix(".timings.jsonl")
        self.limits = {"max_load1": max_load1, "min_free_bytes": min_free_bytes}
        self.commands: list[str] = []

    def __call__(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        check=True,
        step: str | None = None,
        credentials: bool = True,
    ) -> subprocess.CompletedProcess:
        line = shlex.join(argv)
        self.commands.append(line)
        print(f"$ {line}", flush=True)
        if self.dry_run:
            return subprocess.CompletedProcess(argv, 0, "", "")
        started = time.monotonic()
        timing = {
            "step": step or line,
            "command": line,
            "started_at": datetime.now(UTC).isoformat(),
            "returncode": None,
        }
        try:
            timing.update(resource_guard(self.cwd, **self.limits))
            with self.log.open("a") as handle:
                handle.write(f"\n# {datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} $ {line}\n")
                handle.flush()
                result = run_logged(
                    argv,
                    cwd=self.cwd,
                    env={
                        **policyengine_py_environment(credentials=credentials),
                        **(env or {}),
                    },
                    handle=handle,
                )
            timing["returncode"] = result.returncode
        except Exception as error:
            timing["error"] = str(error)
            raise
        finally:
            timing["elapsed_seconds"] = time.monotonic() - started
            record_timing(self.timings, timing)
            print(
                f"[done] {step or line}: {timing['elapsed_seconds']:.3f}s "
                f"(exit {timing.get('returncode')})",
                flush=True,
            )
        if check and result.returncode != 0:
            tail = "\n".join(result.stdout.splitlines()[-25:])
            raise CertifyError(f"{line} exited {result.returncode}:\n{tail}")
        return result


def writer_check(national_build: str, pepy_repo: Path) -> list[str]:
    """Open pull requests or branches already certifying this build.

    A writer is an open pull request that names the build, or any open pull
    request or remote branch whose name contains ``certify-uk-<id>``: this
    handoff's own branch, a suffixed one such as
    ``certify-uk-a133f37a-national``, or one under a namespace. A listing
    that cannot be read refuses rather than passing.
    """

    marker = f"certify-uk-{national_build.rsplit('-', 1)[-1]}"
    found = []
    result = subprocess.run(
        [
            "gh", "pr", "list", "--repo", PEPY_REPO, "--state", "open",
            "--limit", "1000", "--json", "number,title,body,headRefName",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )  # fmt: skip
    if result.returncode != 0:
        raise CertifyError(
            "could not list policyengine.py pull requests to check for another "
            f"writer ({result.stderr.strip()[:200]}); retry, or ask the UK hub."
        )
    for pr in json.loads(result.stdout or "[]"):
        text = f"{pr.get('title') or ''}\n{pr.get('body') or ''}"
        if national_build in text or marker in (pr.get("headRefName") or ""):
            found.append(f"#{pr['number']} {pr['title']} ({pr['headRefName']})")
    remote = subprocess.run(
        ["git", "ls-remote", "--heads", "origin"],
        cwd=pepy_repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if remote.returncode != 0:
        raise CertifyError(
            "could not list policyengine.py branches to check for another "
            f"writer ({remote.stderr.strip()[:200]}); retry, or ask the UK hub."
        )
    for line in remote.stdout.splitlines():
        branch = line.split("\t", 1)[-1].removeprefix("refs/heads/")
        if marker in branch:
            found.append(f"branch {branch} exists on origin")
    return found


def rewrite_regional_constants(
    root: Path, old: BuildPin | None, new: BuildPin | None
) -> list[str]:
    """Replace the old local-area build's name and commit with the new one's
    in the certified-constant files, where each is a quoted literal."""

    if old is None or new is None:
        return []
    replacements = [
        (before, after)
        for before, after in ((old.build, new.build), (old.commit, new.commit))
        if before != after
    ]
    changed = []
    for relative in CONSTANT_FILES if replacements else ():
        path = root / relative
        text = updated = path.read_text()
        for before, after in replacements:
            pattern = re.compile(CONSTANT_PATTERN.format(old=re.escape(before)))
            updated = pattern.sub(
                lambda m, after=after: m.group("pre") + after + m.group("post"),
                updated,
            )
        if updated != text:
            path.write_text(updated)
            changed.append(relative)
    return changed


#: policyengine.py's audit of a regional file's stored columns (policyengine.py#699).
#: An accepted model mismatch certifies only with the evidence it writes.
AUDIT_SCRIPT = "scripts/audit_regional_input_surface.py"
#: The evidence version that measures drops against the regional build model.
REGIONAL_EVIDENCE_SCHEMA = 2


#: The policyengine.py module that reads a build in its own folder (policyengine.py#683).
PEPY_MANIFEST_MODULE = "src/policyengine/provenance/manifest.py"


def build_folder_support(repo: Path, base_commit: str) -> str | None:
    """Why the base cannot certify a build in its own folder, or ``None`` if
    it can: the base needs .py's ``read_build_folder``."""

    found = subprocess.run(
        ["git", "grep", "-q", "-e", "def read_build_folder", base_commit, "--",
         PEPY_MANIFEST_MODULE],
        cwd=repo,
        capture_output=True,
        check=False,
    )  # fmt: skip
    if found.returncode != 0:
        return f"its {PEPY_MANIFEST_MODULE} has no read_build_folder"
    return None


def regional_audit_support(repo: Path, base_commit: str) -> str | None:
    """Why the base cannot certify an accepted mismatch, or ``None`` if it can.

    The base needs the audit script and a ``bundle.py`` that forwards its
    evidence (``--regional-input-surface-evidence``).
    """

    script = subprocess.run(
        ["git", "cat-file", "-e", f"{base_commit}:{AUDIT_SCRIPT}"],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    if script.returncode != 0:
        return f"it has no {AUDIT_SCRIPT}"
    flag = subprocess.run(
        ["git", "grep", "-q", "-e", "--regional-input-surface-evidence", base_commit,
         "--", "scripts/bundle.py"],
        cwd=repo,
        capture_output=True,
        check=False,
    )  # fmt: skip
    if flag.returncode != 0:
        return "its scripts/bundle.py does not take --regional-input-surface-evidence"
    return None


def artifact_pin(
    build: PublishedBuild, name: str, artifact: Mapping[str, Any]
) -> tuple[str, str, str]:
    """Where one of a build's artifacts is read, as policyengine.py pins it:
    the repository, the path and the revision.

    A file in the build repository is a bare file name, alone or after the
    build's folder, read in the folder at the build's commit. A file in
    another repository keeps its own path and must name its own revision.
    """

    repo = artifact.get("repo_id") or BUILD_REPO
    path = str(artifact.get("path") or "")
    if repo != BUILD_REPO:
        revision = artifact.get("revision")
        if not isinstance(revision, str) or not revision:
            raise CertifyError(
                f"{build.build}: {name} is in {repo} and names no revision."
            )
        return repo, path, revision
    file_name = path.removeprefix(f"{build.folder}/")
    if (
        not file_name
        or file_name in {".", ".."}
        or "/" in file_name
        or "\\" in file_name
    ):
        raise CertifyError(
            f"{build.build}: {name} ({path}) is not a file directly inside "
            f"{build.folder}/."
        )
    return repo, f"{build.folder}/{file_name}", build.commit


def fetch_regional_microdata(
    build: PublishedBuild, *, timings: Path, api: Any = None
) -> dict[str, Path]:
    """Every microdata H5 of a local-area build, by manifest name, checked by
    sha256.

    The manifest is read from the build's folder at its commit, and each file
    where policyengine.py pins it (:func:`artifact_pin`). These are the files
    policyengine.py's audit reads; it re-checks each digest against the
    manifest itself.
    """

    api = hub_api() if api is None else api

    def download(repo: str, file_path: str, revision: str) -> Path:
        try:
            return Path(
                api.hf_hub_download(
                    repo_id=repo,
                    filename=file_path,
                    repo_type="dataset",
                    revision=revision,
                )
            )
        except Exception as error:
            raise CertifyError(
                f"could not download {file_path} at {revision} from {repo}: {error}"
            ) from error

    with timed_step("regional-microdata", timings) as record:
        manifest = json.loads(
            download(BUILD_REPO, build.manifest_path, build.commit).read_text()
        )
        microdata = {
            name: artifact
            for name, artifact in (manifest.get("artifacts") or {}).items()
            if artifact.get("kind") == "microdata"
            and str(artifact.get("path", "")).lower().endswith(".h5")
        }
        if not microdata:
            raise CertifyError(f"{build.build}: the manifest lists no microdata H5.")
        pins = {}
        for name, artifact in sorted(microdata.items()):
            if not artifact.get("sha256"):
                raise CertifyError(f"{build.build}: {name} has no sha256.")
            pins[name] = artifact_pin(build, name, artifact)
        paths: dict[str, Path] = {}
        for name, (repo, file_path, revision) in pins.items():
            path = download(repo, file_path, revision)
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(chunk)
            expected = microdata[name]["sha256"]
            if digest.hexdigest() != expected:
                raise CertifyError(
                    f"{file_path} at {revision} hashes to {digest.hexdigest()}, "
                    f"not the manifest's {expected}."
                )
            paths[name] = path
        record.update(artifacts=sorted(paths))
    return paths


def regional_audit_argv(
    python: str,
    *,
    manifest_uri: str,
    artifacts: dict[str, Path],
    model: str,
    audited_by: str,
    limitations: Sequence[str],
    output: Path,
) -> list[str]:
    """The audit command for these regional files under the target model."""

    argv = [python, AUDIT_SCRIPT, "--regional-manifest-uri", manifest_uri]
    for name, path in sorted(artifacts.items()):
        argv += ["--artifact", f"{name}={path}"]
    argv += ["--model-version", model, "--audited-by", audited_by]
    for limitation in limitations:
        argv += ["--limitation", limitation]
    return [*argv, "--output", str(output)]


def undeclared_flags(help_text: str, argv: Sequence[str]) -> list[str]:
    """The ``--flags`` of ``argv`` that a tool's ``--help`` does not declare."""

    declared = set(re.findall(r"(--[a-z][a-z0-9-]*)", help_text))
    return sorted({a for a in argv if a.startswith("--") and a not in declared})


def read_regional_evidence(path: Path, *, model: str, build_model: str) -> dict:
    """The audit's evidence, refused unless it is build-relative and drop-free."""

    try:
        evidence = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise CertifyError(f"the regional audit wrote no readable {path}.") from error
    if not isinstance(evidence, dict):
        raise CertifyError(f"{path} is not an evidence object.")
    if evidence.get("schema_version") != REGIONAL_EVIDENCE_SCHEMA:
        raise CertifyError(
            f"{path} is evidence schema {evidence.get('schema_version')!r}, not "
            f"{REGIONAL_EVIDENCE_SCHEMA}: this policyengine.py measures drops "
            "against the target model only, so it refuses Microcosm's own "
            "columns. Certify from a base with policyengine.py#699's "
            "build-model baseline."
        )
    for field, expected in (
        ("model_version", model),
        ("build_model_version", build_model),
    ):
        if evidence.get(field) != expected:
            raise CertifyError(
                f"{path} audits {field} {evidence.get(field)!r}, not {expected!r}."
            )
    if evidence.get("dropped_columns"):
        raise CertifyError(
            f"policyengine-uk {model} no longer defines these stored columns of "
            f"the {build_model} local-area file, so area runs would lose them: "
            f"{', '.join(evidence['dropped_columns'])}. Build a dense cut on {model}."
        )
    return evidence


def overlay_diff(old_bundle: dict, new_bundle: dict) -> list[str]:
    old = set((old_bundle.get("dataset_overlays") or {}).get("uk", {}) or {})
    new = set((new_bundle.get("dataset_overlays") or {}).get("uk", {}) or {})
    return sorted(old - new)


def uc_floor_summary(receipt: object) -> tuple[bool, str]:
    """The measured floor's verdict and a summary of its ratios.

    Only a complete receipt is accepted: a mapping for the ``uk_uc_floor``
    check with a boolean verdict, and a list of measures that includes
    ``benefit_units`` and ``spend``. Each measure has a unique name, a
    boolean verdict and a finite numeric ratio, and the overall verdict is
    the conjunction of theirs. Any other value, whatever its shape, is the
    floor failing operationally, never a crash.
    """

    incomplete = CertifyError(
        "UC floor failed operationally: incomplete measured receipt."
    )
    if not isinstance(receipt, dict) or receipt.get("check") != "uk_uc_floor":
        raise incomplete
    passed, measures = receipt.get("passed"), receipt.get("measures")
    if not isinstance(passed, bool) or not isinstance(measures, list):
        raise incomplete
    rows: list[tuple[str, float, bool]] = []
    for measure in measures:
        if not isinstance(measure, dict):
            raise incomplete
        name, ratio, verdict = (measure.get(k) for k in ("name", "ratio", "passed"))
        if not isinstance(name, str) or not name or not isinstance(verdict, bool):
            raise incomplete
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise incomplete
        try:
            ratio = float(ratio)
        except OverflowError:
            raise incomplete from None
        if not math.isfinite(ratio):
            raise incomplete
        rows.append((name, ratio, verdict))
    names = [name for name, _, _ in rows]
    if len(set(names)) != len(names) or not {"benefit_units", "spend"} <= set(names):
        raise incomplete
    if passed != all(verdict for _, _, verdict in rows):
        raise incomplete
    ratios = ", ".join(f"{name}={ratio:.6f}" for name, ratio, _ in rows)
    return (
        passed,
        f"{'passed' if passed else 'FAILED'}; measured/benchmark ratios: {ratios}",
    )


def pr_body(context: dict[str, Any]) -> str:
    lines = [
        f"Certifies Microcosm UK national build `{context['national_build']}` "
        f"(policyengine-uk {context['model']}, core {context['core']}), from its "
        f"folder at commit `{context['national_commit']}`.",
        "",
        "**Mechanical handoff; wrapper qualification pending** unless reviewed "
        "receipts cover this exact candidate pair, commit and bundle. Microcosm's "
        "19 producer release-cut gates and passing pytest do not establish "
        "wrapper qualification.",
        "",
        f"- **Regional route:** {context['regional_summary']}",
        f"- **Compatibility basis:** `{context.get('basis')}`",
        f"- **Made by** `tools/certify_uk_cut_in_policyengine_py.py` in "
        f"PolicyEngine/microcosm at `{context['microcosm_commit'][:9]}`.",
        "",
        "## How it was made",
        "",
        "```bash",
        *context["commands"],
        "```",
        "",
        "## Receipts",
        "",
        f"- UC floor (weighted national totals, `scripts/check_uk_uc_floor.py`): "
        f"{context.get('uc_floor', 'not run')}",
        f"- Tests: {context.get('tests', 'not run')}",
        f"- Certified-constant files rewritten: {context.get('constants') or 'none'}",
        "",
        "## Needs a person before this leaves draft",
        "",
        "- [ ] **Wrapper qualification.** Attach exact-pair receipts for every "
        "required candidate gate, including the full reform contract, metrics "
        "review and reform responsiveness; attach and review scorecard replay "
        "evidence. Keep qualification pending until the evidence is reviewed. "
        "Record any explicit reviewer-approved exception; this script waives "
        "no gate.",
    ]
    if context.get("uc_floor_passed") is False:
        lines.append(
            "- [ ] **UC floor acceptance.** This build has a measured floor failure "
            "and needs explicit acceptance in its pull request "
            "(data-certification.md, recorded exceptions). Accepted by: ___"
        )
    if context.get("shared_old_commit"):
        regional = context.get("regional_commit")
        lines.append(
            "- [ ] **Local-area revision constants.** The previous local-area "
            f"build shared commit `{context['shared_old_commit']}` with a national "
            "build, so rewriting quoted literals cannot tell the two builds' "
            "constants apart, and this tool left the local-area commit to a "
            "person. Check that each national revision constant names "
            f"`{context['national_commit']}` and each local-area one names "
            + (f"`{regional}`." if regional else "the local-area build's commit.")
        )
    if context.get("dropped_overlays"):
        lines.append(
            "- [ ] **Dataset overlays dropped by certification:** "
            f"{', '.join(context['dropped_overlays'])}. Re-add any that must stay "
            "loadable from the previous bundle (as #685 did)."
        )
    evidence = context.get("regional_evidence")
    if evidence is not None:
        lines.append(
            "- [ ] **Area runs on the older local-area file** (accepted mismatch, "
            f"built on {evidence['build_model_version']}, audited by "
            f"{evidence['audited_by']}): no stored column is dropped; inputs that "
            f"take defaults: {', '.join(evidence['defaulted_inputs']) or 'none'}; "
            f"{len(evidence.get('never_modelled_columns') or [])} stored columns "
            "neither model defines are recorded only. Limitations recorded: "
            f"{'; '.join(evidence['limitations'])}. Confirm none matters for a "
            "reform the Budget dashboard shows by area."
        )
    if context.get("warnings"):
        lines.append(
            "- [ ] **Certification warnings** (from `certify-data`):\n  "
            + "\n  ".join(f"- {w}" for w in context["warnings"])
        )
    if context.get("tests_failed"):
        lines.append(
            "- [ ] **Failing tests** to update for the new pair: "
            + ", ".join(context["tests_failed"][:20])
        )
    lines.append("- [ ] One writer: the UK hub and María know this PR exists.")
    if context.get("footer"):
        lines += ["", context["footer"]]
    return "\n".join(lines) + "\n"


#: A commit trailer's co-author: one line, ``Name <email>``.
CO_AUTHOR_PATTERN = re.compile(r"^[^<>\n]+ <[^<>\s@]+@[^<>\s]+>$")


def commit_message(national_build: str, *, co_authors: Sequence[str] = ()) -> str:
    """The certification commit's message, crediting only the named co-authors."""

    message = f"Certify Microcosm UK build {national_build}"
    if co_authors:
        trailers = "\n".join(f"Co-Authored-By: {author}" for author in co_authors)
        message += f"\n\n{trailers}"
    return message


def readable(text: str | None) -> bool:
    """At least one printable, non-space character, as policyengine.py requires
    of a mismatch's reason, auditor and limitations (a zero-width space alone
    is not readable)."""

    return any(ch.isprintable() and not ch.isspace() for ch in text or "")


def parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--national-build",
        required=True,
        metavar="BUILD",
        help=f"the national build's name in releases.json ({NATIONAL_RELEASE})",
    )
    parser.add_argument(
        "--national-data-commit",
        metavar="COMMIT",
        help="pin the national build at this commit instead of the one that "
        "published its folder",
    )
    route = parser.add_mutually_exclusive_group(required=True)
    route.add_argument(
        "--dense-build",
        metavar="BUILD",
        help=f"route regions to this local-area build of {DENSE_RELEASE}",
    )
    route.add_argument("--keep-regional-release", action="store_true")
    route.add_argument("--area-coded-national-uri")
    route.add_argument("--drop-regional-release", action="store_true")
    parser.add_argument(
        "--accept-regional-model-mismatch",
        metavar="REASON",
        help=(
            "route regions to a regional build made with another policyengine-uk, "
            "recording who accepted it and why; certifies only with "
            "policyengine.py's audited input-surface evidence (#699)"
        ),
    )
    parser.add_argument(
        "--audited-by",
        metavar="WHO",
        help=(
            "with an accepted mismatch: who ran the audit, as it should appear "
            "in policyengine.py's public bundle"
        ),
    )
    parser.add_argument(
        "--regional-limitation",
        action="append",
        default=[],
        metavar="TEXT",
        help="with an accepted mismatch: a limitation to record (repeatable)",
    )
    parser.add_argument(
        "--dense-data-commit",
        metavar="COMMIT",
        help="with --dense-build: pin the local-area build at this commit instead "
        "of the one that published its folder",
    )
    parser.add_argument("--model-version", help="default: the build's own model")
    parser.add_argument("--core-version", help="default: the build's own core")
    parser.add_argument(
        "--pepy-repo",
        type=Path,
        required=True,
        help="a clone of PolicyEngine/policyengine.py with an origin remote",
    )
    parser.add_argument("--worktree", type=Path, help="default: beside the repo")
    parser.add_argument(
        "--resume-worktree",
        action="store_true",
        help="replay stages in an unfinished registered worktree at the exact base",
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="base branch/ref for both the old bundle and the new worktree",
    )
    parser.add_argument("--skip-writer-check", metavar="WHY")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--test-workers", type=int, default=4)
    parser.add_argument(
        "--max-load1",
        type=float,
        help="refuse each heavy step above this 1-minute load (default: no limit)",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_BYTES / 1e9,
        help="refuse each heavy step below this free disk (default %(default)s GB)",
    )
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--pr-base",
        default="main",
        help="pull-request target branch (default: main), independent of --base-ref",
    )
    parser.add_argument("--open-pr", action="store_true", help="draft; needs --push")
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    parser.add_argument(
        "--co-author",
        action="append",
        default=[],
        metavar="NAME <EMAIL>",
        help="add a Co-Authored-By trailer to the certification commit (repeatable)",
    )
    parser.add_argument(
        "--pr-body-footer",
        metavar="TEXT",
        help="a closing line for the pull request body, such as an agent's credit",
    )
    args = parser.parse_args(argv)
    if args.open_pr and not args.push:
        parser.error("--open-pr needs --push.")
    for author in args.co_author:
        if not CO_AUTHOR_PATTERN.match(author):
            parser.error(f"--co-author must be one line, 'Name <email>': {author!r}")
    if args.accept_regional_model_mismatch is not None:
        if not readable(args.accept_regional_model_mismatch):
            parser.error("--accept-regional-model-mismatch needs a reason.")
        if not readable(args.audited_by):
            parser.error("--accept-regional-model-mismatch needs --audited-by.")
        if not args.regional_limitation or not all(
            readable(text) for text in args.regional_limitation
        ):
            parser.error(
                "--accept-regional-model-mismatch needs at least one "
                "--regional-limitation, each with readable text."
            )
    elif args.audited_by or args.regional_limitation:
        parser.error(
            "--audited-by and --regional-limitation go with "
            "--accept-regional-model-mismatch."
        )
    if not 1 <= args.test_workers <= 4:
        parser.error("--test-workers must be between 1 and 4.")
    for flag, name in (
        ("--national-build", args.national_build),
        ("--dense-build", args.dense_build),
    ):
        if name is not None and not BUILD_NAME_PATTERN.match(name):
            parser.error(f"{flag} {name!r} is not a build name.")
    for flag, commit in (
        ("--national-data-commit", args.national_data_commit),
        ("--dense-data-commit", args.dense_data_commit),
    ):
        if commit is not None and not COMMIT_PATTERN.match(commit):
            parser.error(f"{flag} must be a full 40-character commit, not {commit!r}.")
    if args.dense_data_commit is not None and args.dense_build is None:
        parser.error("--dense-data-commit goes with --dense-build.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    try:
        return certify(args)
    except CertifyError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


def certify(args: argparse.Namespace) -> int:
    microcosm_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()
    repo = args.pepy_repo
    limits = {
        "max_load1": args.max_load1,
        "min_free_bytes": int(args.min_free_gb * 1e9),
    }
    suffix = args.national_build.rsplit("-", 1)[-1]
    branch = f"certify-uk-{suffix}"
    worktree = args.worktree or repo.parent / "_worktrees" / f"pepy-{branch}"
    log = worktree.parent / f"{worktree.name}.certify.log"
    timings = log.with_suffix(".timings.jsonl")
    if not args.skip_tests and not args.dry_run:
        with timed_step("test-temp-preflight", timings) as timing:
            timing["temporary_directory"] = str(check_test_temp_directory())
    base_ref = args.base_ref
    if base_ref.startswith("origin/"):
        remote_branch = base_ref.removeprefix("origin/")
        with timed_step("git-fetch", timings):
            subprocess.run(
                [
                    "git",
                    "fetch",
                    "--quiet",
                    "origin",
                    f"+refs/heads/{remote_branch}:refs/remotes/origin/{remote_branch}",
                ],
                cwd=repo,
                check=True,
                timeout=300,
            )
    with timed_step("resolve-base", timings):
        base_commit = subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{base_ref}^{{commit}}"],
            cwd=repo,
            text=True,
        ).strip()
    with timed_step("base-build-folder-support", timings):
        problem = build_folder_support(repo, base_commit)
    if problem is not None:
        raise CertifyError(
            f"policyengine.py at {base_ref} ({base_commit[:9]}) cannot certify a "
            f"build in its own folder: {problem} (policyengine.py#683). Use a "
            "--base-ref that has it."
        )

    replay_paths = []
    if args.resume_worktree:
        with timed_step("resume-worktree-validation", timings):
            replay_paths = validate_resume_worktree(
                repo, worktree, branch, base_commit, args.national_build
            )
    elif worktree.exists():
        raise CertifyError(
            f"{worktree} exists; use --resume-worktree for an unfinished handoff "
            "or choose a fresh --worktree."
        )

    if args.skip_writer_check:
        print(f"writer check skipped: {args.skip_writer_check}")
    else:
        with timed_step("writer-check", timings):
            others = writer_check(args.national_build, repo)
        if others:
            raise CertifyError(
                "this build is already being certified: "
                + "; ".join(others)
                + ". Coordinate with its owner instead of opening a second PR."
            )

    with timed_step("national-manifest", timings):
        national = read_published_build(
            NATIONAL_RELEASE, args.national_build, commit=args.national_data_commit
        )
    model = args.model_version or national.model
    core = args.core_version or national.core
    if not covers(national, model):
        raise CertifyError(
            f"the national build was built with policyengine-uk {national.model} "
            f"and claims {list(national.claims)}; it cannot be certified for "
            f"{model}."
        )
    with timed_step("old-bundle", timings):
        old_bundle = json.loads(
            subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{base_commit}:{BUNDLE_MANIFEST}",
                ],
                cwd=repo,
                text=True,
            )
        )
    record = old_bundle["data_releases"]["uk"]
    dense, mismatch, regional_artifacts = None, False, {}
    kept = kept_regional_pin(record) if args.keep_regional_release else None
    if args.dense_build or kept is not None:
        with timed_step("regional-manifest", timings):
            if args.dense_build:
                dense = read_published_build(
                    DENSE_RELEASE, args.dense_build, commit=args.dense_data_commit
                )
            else:
                dense = read_published_build(
                    kept.release, kept.build, commit=kept.commit
                )
    flags = regional_flags(args, record, dense)
    regional_summary = shlex.join(flags)
    # Before the regional microdata is downloaded: a refused resume must not
    # download it first.
    check_resumed_pins(
        worktree,
        replay_paths,
        national,
        flags[flags.index("--regional-manifest-uri") + 1]
        if "--regional-manifest-uri" in flags
        else None,
    )
    if dense is not None:
        if not covers(dense, model):
            if not args.accept_regional_model_mismatch:
                raise CertifyError(
                    f"the regional build {dense.build} was built with "
                    f"policyengine-uk {dense.model} and claims {list(dense.claims)}, "
                    f"so certification for {model} will refuse it. Routes: a "
                    f"local-area build on {model} (--dense-build), "
                    "--drop-regional-release, an enhanced-FRS release that claims "
                    "the version (--area-coded-national-uri), or an accepted, "
                    "recorded mismatch (--accept-regional-model-mismatch) with "
                    "policyengine.py's audited evidence."
                )
            problem = regional_audit_support(repo, base_commit)
            if problem is not None:
                raise CertifyError(
                    f"policyengine.py at {base_ref} ({base_commit[:9]}) cannot "
                    f"certify an accepted mismatch: {problem} (policyengine.py#699). "
                    "Use a --base-ref that has it, or another regional route."
                )
            mismatch = True
            regional_summary += f" (regional built on {dense.model}; mismatch accepted)"
            if not args.dry_run:
                regional_artifacts = fetch_regional_microdata(dense, timings=timings)
    old_regional = folder_reference(record.get("regional_source_manifest_uri"))
    old_national_commit = (record.get("data_package") or {}).get(
        "release_manifest_revision"
    )
    # The constants are rewritten as quoted literals: policyengine.py's rewrite
    # moves the old national commit, then the regional rewrite moves the old
    # local-area commit. When the old local-area commit is also the old or the
    # new national commit, and the new builds' commits differ, neither rewrite
    # can tell the two builds' constants apart.
    shared_old_commit = (
        old_regional.commit
        if old_regional is not None
        and dense is not None
        and old_regional.commit in {old_national_commit, national.commit}
        and dense.commit != national.commit
        else None
    )

    plan = {
        "base_ref": base_ref,
        "base_commit": base_commit,
        "national_build": national.build,
        "national_manifest_uri": national.manifest_uri,
        "model": model,
        "core": core,
        "regional": regional_summary,
        "branch": branch,
        "worktree": str(worktree),
        "resume_worktree": args.resume_worktree,
        "replayed_generated_paths": replay_paths,
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0
    if not args.resume_worktree:
        with timed_step("git-worktree", timings):
            subprocess.run(
                [
                    "git",
                    "worktree",
                    "add",
                    "--quiet",
                    "-b",
                    branch,
                    str(worktree),
                    base_commit,
                ],
                cwd=repo,
                check=True,
            )
    run = Shell(worktree, log, timings=timings, **limits)
    python = str(worktree / ".venv/bin/python")
    sync = ["uv", "sync", "--frozen", "--extra", "uk", "--extra", "dev",
            "--python", PEPY_PYTHON]  # fmt: skip
    run(sync, step="initial-sync", credentials=False)
    evidence = None
    if mismatch:
        # The audit reads the target model in process, so the worktree's
        # environment gets the target pair now; the certified sync below
        # restores it to the new lock exactly.
        run(
            ["uv", "pip", "install", "--no-config", "--default-index",
             "https://pypi.org/simple", "--python", python,
             f"policyengine-uk=={model}", f"policyengine-core=={core}"],
            step="audit-model-install",
            credentials=False,
        )  # fmt: skip
        evidence_path = worktree.parent / f"{worktree.name}.regional-input-surface.json"
        # An earlier attempt's evidence must not stand in for this audit.
        evidence_path.unlink(missing_ok=True)
        audit = regional_audit_argv(
            python,
            manifest_uri=flags[flags.index("--regional-manifest-uri") + 1],
            artifacts=regional_artifacts,
            model=model,
            audited_by=args.audited_by,
            limitations=args.regional_limitation,
            output=evidence_path,
        )
        missing = undeclared_flags(
            run([python, AUDIT_SCRIPT, "--help"], step="audit-help").stdout, audit
        )
        if missing:
            raise CertifyError(
                f"{AUDIT_SCRIPT} at {base_commit[:9]} does not declare "
                f"{', '.join(missing)}; update this tool to its interface."
            )
        run(audit, step="regional-input-audit")
        evidence = read_regional_evidence(
            evidence_path, model=model, build_model=dense.model
        )
        flags += [
            "--accept-regional-model-mismatch", args.accept_regional_model_mismatch,
            "--regional-input-surface-evidence", str(evidence_path),
        ]  # fmt: skip
    certify_result = run(
        [
            python, "scripts/bundle.py", "certify-data", "--country", "uk",
            "--data-producer", "populace",
            "--manifest-uri", national.manifest_uri,
            *flags,
            "--model-version", model, "--core-version", core,
        ], step="certify-data",
    )  # fmt: skip
    warnings = [
        line.strip()
        for line in certify_result.stdout.splitlines()
        if "warning" in line.lower()
    ]
    run(
        ["uv", "lock", "--no-config", "--no-sources", "--default-index",
         "https://pypi.org/simple", "--python", python], step="lock",
        credentials=False,
    )  # fmt: skip
    run(sync, step="certified-sync", credentials=False)
    run([python, "scripts/release_lock.py"], step="release-lock")
    run([python, "scripts/bundle.py", "generate", "--include-tros"], step="TROs")
    new_bundle = json.loads((worktree / BUNDLE_MANIFEST).read_text())
    new_record = new_bundle["data_releases"]["uk"]
    rewrite = run(
        [
            python, "-c",
            "import json, sys; from pathlib import Path; "
            "from policyengine.provenance.uk_qualification import "
            "certified_constant_replacements as r, rewrite_certified_constants as w; "
            "old, new = json.loads(sys.argv[1]), json.loads(sys.argv[2]); "
            "print(json.dumps(w(Path('.'), r(old, new))))",
            json.dumps(record), json.dumps(new_record),
        ], step="constant-rewrite",
        env={"POLICYENGINE_SKIP_COUNTRY_IMPORTS": "1"},
    )  # fmt: skip
    constants = json.loads(rewrite.stdout.strip().splitlines()[-1] or "[]")
    with timed_step("regional-constant-rewrite", timings):
        new_regional = None if dense is None else dense.pin
        if new_regional is not None and shared_old_commit is not None:
            # Its commit constants are left for a person (see the body).
            new_regional = new_regional._replace(commit=shared_old_commit)
        constants += rewrite_regional_constants(worktree, old_regional, new_regional)
    run(
        pytest_command(python, ["tests/test_household_calculator_snapshot.py", "-q"]),
        env={"PE_UPDATE_SNAPSHOTS": "1"},
        step="snapshots",
    )
    run(
        [python, "scripts/bundle.py", "check", "--published-spm", "--include-tros",
         "--strict-tros"], step="strict-check",
    )  # fmt: skip
    uc_path = worktree.parent / f"{worktree.name}.uc_floor.json"
    # An earlier attempt's receipt must not mask an operational failure.
    uc_path.unlink(missing_ok=True)
    uc = run(
        [python, "scripts/check_uk_uc_floor.py", "--dataset",
         new_record["default_dataset"], "--year", "2025", "--output", str(uc_path)],
        check=False,
        step="UC-floor",
        env={"POLICYENGINE_SKIP_COUNTRY_IMPORTS": "1"},
    )  # fmt: skip
    if not uc_path.exists():
        raise CertifyError(
            f"UC floor failed operationally (exit {uc.returncode}); no measured "
            f"receipt was produced. See {log}."
        )
    try:
        receipt = json.loads(uc_path.read_text())
    except json.JSONDecodeError as error:
        raise CertifyError(
            "UC floor failed operationally: invalid measured receipt."
        ) from error
    uc_passed, uc_summary = uc_floor_summary(receipt)
    if uc.returncode != (0 if uc_passed else 1):
        raise CertifyError(
            f"UC floor failed operationally (exit {uc.returncode}); "
            "the measured receipt disagrees with the process exit status."
        )
    tests_summary, tests_failed, tests_returncode = "skipped", [], 0
    if not args.skip_tests:
        xdist = run(
            [python, "-c", "import xdist"],
            check=False,
            step="check-pytest-xdist",
            credentials=False,
        )
        if xdist.returncode:
            run(
                ["uv", "pip", "install", "--python", python, "pytest-xdist"],
                step="install-pytest-xdist",
                credentials=False,
            )
        tests = run(
            pytest_command(python, ["-n", str(args.test_workers), "-vv",
             "--durations=20", "-o", "faulthandler_timeout=300",
             "-p", "no:cacheprovider"]),
            check=False,
            step="tests",
        )  # fmt: skip
        tests_returncode = tests.returncode
        tests_failed = re.findall(r"^FAILED (\S+)", tests.stdout, flags=re.MULTILINE)
        tests_summary = (tests.stdout.strip().splitlines() or ["?"])[-1]
        if tests_returncode and not tests_failed:
            tests_failed = [f"pytest exited {tests_returncode}: {tests_summary}"]
    context = {
        "national_build": national.build,
        "national_commit": national.commit,
        "regional_commit": None if dense is None else dense.commit,
        "shared_old_commit": shared_old_commit,
        "model": model,
        "core": core,
        "regional_summary": f"`{regional_summary}`",
        "basis": (new_record.get("certification") or {}).get("compatibility_basis"),
        "microcosm_commit": microcosm_commit,
        "commands": run.commands,
        "uc_floor": f"`{uc_summary}`",
        "uc_floor_passed": uc_passed,
        "tests": tests_summary,
        "tests_failed": tests_failed,
        "constants": ", ".join(constants),
        "dropped_overlays": overlay_diff(old_bundle, new_bundle),
        "warnings": warnings,
        "footer": args.pr_body_footer,
        "regional_evidence": evidence,
    }
    body = worktree.parent / f"{worktree.name}.pr-body.md"
    body.write_text(pr_body(context))
    with timed_step("generated-file-validation", timings):
        staged_paths = certification_changes(worktree, national.build)
    if staged_paths:
        run(["git", "add", "--", *staged_paths], step="stage-certification")
    run(
        ["git", "commit", "-q", "-m",
         commit_message(national.build, co_authors=args.co_author)]
    )  # fmt: skip
    print(f"\ncommitted on {branch} in {worktree}; PR body draft: {body}")
    if args.push:
        run(["git", "push", "-u", "origin", branch])
    if args.open_pr:
        run(
            ["gh", "pr", "create", "--draft", "--repo", PEPY_REPO, "--base", args.pr_base,
             "--head", branch, "--title",
             f"Certify Microcosm UK build {national.build}", "--body-file", str(body)]
        )  # fmt: skip
    return 0 if not tests_returncode else 1


if __name__ == "__main__":
    raise SystemExit(main())
