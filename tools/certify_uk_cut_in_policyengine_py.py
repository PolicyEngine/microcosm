"""Certify a published Microcosm UK cut into policyengine.py, on a fresh branch.

policyengine.py serves the UK cut its bundle manifest certifies; moving that
pin is a policyengine.py pull request and release. This script does the
mechanical part of that pull request the way policyengine.py#685 did it by
hand (``docs/engineering/skills/data-certification.md`` there), in a new
worktree branched from policyengine.py's ``origin/main``:

1. refuse if someone is already certifying this cut (one writer per cut);
2. read the national (and regional) release manifests at their tags and
   check, before anything is written, that the regional route can certify;
3. ``scripts/bundle.py certify-data`` with the national tag and the chosen
   regional route, at the cut's own policyengine-uk and core;
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

Regional routes (pick one):

``--dense-tag TAG``           a Microcosm local-area cut (ship together)
``--keep-regional-release``   the bundle's current regional release
``--area-coded-national-uri`` an enhanced-FRS release that claims the version
``--drop-regional-release``   no area routing (sim-api's UK default then
                              lacks constituency and local-authority runs)

A regional release built on another policyengine-uk is refused by
certification unless policyengine.py accepts a recorded mismatch:
``--accept-regional-model-mismatch REASON`` with ``--audited-by WHO`` and at
least one ``--regional-limitation TEXT``. The regional microdata is downloaded
at its manifest pin, policyengine.py's own ``scripts/audit_regional_input_surface.py``
(policyengine.py#699) audits its stored columns against the build and target
models, and ``certify-data`` takes that evidence; any dropped column refuses
(see ``docs/uk-cut-anyone.md``).

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
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from microcosm.data.gate_signing_keys import UK_GATE_SIGNING_KEY_ENV

PEPY_REPO = "PolicyEngine/policyengine.py"
CUT_REPO = "policyengine/populace-uk-private"
NATIONAL_RELEASE = "microcosm-uk-2024-25-national"
DENSE_RELEASE = "microcosm-uk-2024-25-dense"
PEPY_PYTHON = "3.13"  # certify-uk-release.yaml's interpreter
TAG_PATTERN = re.compile(
    r"^microcosm-uk-2024-25-(national|dense)-\d{8}T\d{6}Z-[0-9a-f]{8}$"
)
CONSTANT_PATTERN = r'(?P<pre>["@/-]){old}(?P<post>["/])'
CONSTANT_FILES = (
    "tests/test_models.py",
    "tests/test_release_manifests.py",
    "tests/test_uk_regions.py",
)


class CertifyError(RuntimeError):
    """A refusal the operator must act on."""


def manifest_uri(tag: str, release: str) -> str:
    return f"hf://dataset/{CUT_REPO}@{tag}/releases/{release}/release_manifest.json"


def check_tag(tag: str, kind: str) -> str:
    match = TAG_PATTERN.match(tag)
    if not match or match.group(1) != kind:
        raise CertifyError(
            f"{tag!r} is not a {kind} cut tag; certify cut tags only, never @main."
        )
    return tag


@dataclass(frozen=True)
class CutVersions:
    tag: str
    model: str
    core: str
    claims: tuple[str, ...]


def read_cut(tag: str, release: str) -> CutVersions:
    # Set before importing the hub: its HTTP timeout constants are import-time.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        CUT_REPO,
        f"releases/{release}/release_manifest.json",
        repo_type="dataset",
        revision=tag,
        token=os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN"),
        etag_timeout=60,
    )
    manifest = json.loads(Path(path).read_text())
    build = manifest.get("build") or {}
    if build.get("cut_tag") != tag:
        raise CertifyError(f"{tag}: the manifest names cut {build.get('cut_tag')!r}.")
    return CutVersions(
        tag=tag,
        model=build["built_with_model_package"]["version"],
        core=build["built_with_core_package"]["version"],
        claims=tuple(
            claim["specifier"]
            for claim in manifest.get("compatible_model_packages", ())
            if claim.get("name") == "policyengine-uk"
        ),
    )


def covers(cut: CutVersions, version: str) -> bool:
    """policyengine.py's rule: built with the version, or an exact claim of it."""

    return cut.model == version or f"=={version}" in cut.claims


def regional_flags(args: argparse.Namespace, bundle_record: dict) -> list[str]:
    if args.dense_tag:
        return [
            "--regional-manifest-uri",
            manifest_uri(check_tag(args.dense_tag, "dense"), DENSE_RELEASE),
        ]
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


def regional_tag(flags: list[str]) -> str | None:
    if "--regional-manifest-uri" not in flags:
        return None
    uri = flags[flags.index("--regional-manifest-uri") + 1]
    match = re.search(r"@([^/]+)/", uri)
    return (
        match.group(1) if match and match.group(1).startswith(DENSE_RELEASE) else None
    )


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


def certification_path(relative: str, national_tag: str) -> bool:
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
        f"changelog.d/certify-uk-{national_tag}.changed.md",
        "src/policyengine/data/bundle/manifest.json",
        "src/policyengine/data/bundle/uk.trace.tro.jsonld",
        "src/policyengine/data/bundle/us.trace.tro.jsonld",
        *(
            f"tests/fixtures/household_calculator_snapshots/{name}.json"
            for name in snapshots
        ),
    }


def certification_changes(worktree: Path, national_tag: str) -> list[str]:
    """Preserve operator files and refuse unrelated tracked/index changes."""
    tracked = subprocess.check_output(
        ["git", "diff", "--name-only", "-z", "HEAD"], cwd=worktree, text=True
    ).split("\0")
    unexpected = [
        path for path in tracked if path and not certification_path(path, national_tag)
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
    changelog = f"changelog.d/certify-uk-{national_tag}.changed.md"
    return sorted(
        {path for path in tracked if path}
        | {path for path in untracked if path == changelog}
    )


def validate_resume_worktree(
    repo: Path, worktree: Path, branch: str, base_commit: str, national_tag: str
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
    paths = certification_changes(worktree, national_tag)
    manifest_path = "src/policyengine/data/bundle/manifest.json"
    if manifest_path in paths:
        record = json.loads((worktree / manifest_path).read_text())["data_releases"][
            "uk"
        ]
        revision = (record.get("data_package") or {}).get("release_manifest_revision")
        if revision != national_tag:
            raise CertifyError("resume worktree's modified manifest names another cut.")
    return paths


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


def policyengine_py_environment() -> dict[str, str]:
    """This process's environment, less what policyengine.py must not see."""

    return {k: v for k, v in os.environ.items() if k not in WITHHELD_ENVIRONMENT}


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
                    env={**policyengine_py_environment(), **(env or {})},
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


def writer_check(national_tag: str, pepy_repo: Path) -> list[str]:
    """Open pull requests or branches already certifying this cut.

    A writer is an open pull request that names the tag, or any open pull
    request or remote branch whose name contains ``certify-uk-<id>``: this
    handoff's own branch, a suffixed one such as
    ``certify-uk-a133f37a-national``, or one under a namespace. A listing
    that cannot be read refuses rather than passing.
    """

    marker = f"certify-uk-{national_tag.rsplit('-', 1)[-1]}"
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
        if national_tag in text or marker in (pr.get("headRefName") or ""):
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
    root: Path, old: str | None, new: str | None
) -> list[str]:
    """Replace the old local-area cut tag in the certified-constant files."""

    if not old or not new or old == new:
        return []
    pattern = re.compile(CONSTANT_PATTERN.format(old=re.escape(old)))
    changed = []
    for relative in CONSTANT_FILES:
        path = root / relative
        text = path.read_text()
        updated = pattern.sub(lambda m: m.group("pre") + new + m.group("post"), text)
        if updated != text:
            path.write_text(updated)
            changed.append(relative)
    return changed


#: policyengine.py's audit of a regional file's stored columns (policyengine.py#699).
#: An accepted model mismatch certifies only with the evidence it writes.
AUDIT_SCRIPT = "scripts/audit_regional_input_surface.py"
#: The evidence version that measures drops against the regional build model.
REGIONAL_EVIDENCE_SCHEMA = 2


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


def fetch_regional_microdata(
    tag: str, release: str, *, timings: Path
) -> dict[str, Path]:
    """Every microdata H5 of a regional cut, by manifest name, checked by sha256.

    These are the files policyengine.py's audit reads; it re-checks each
    digest against the manifest itself.
    """

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")
    with timed_step("regional-microdata", timings) as record:
        manifest = json.loads(
            Path(
                hf_hub_download(
                    CUT_REPO,
                    f"releases/{release}/release_manifest.json",
                    repo_type="dataset",
                    revision=tag,
                    token=token,
                )
            ).read_text()
        )
        microdata = {
            name: artifact
            for name, artifact in (manifest.get("artifacts") or {}).items()
            if artifact.get("kind") == "microdata"
            and str(artifact.get("path", "")).lower().endswith(".h5")
        }
        if not microdata:
            raise CertifyError(f"{tag}: the manifest lists no microdata H5.")
        paths: dict[str, Path] = {}
        for name, artifact in sorted(microdata.items()):
            if not artifact.get("sha256") or not artifact.get("revision"):
                raise CertifyError(
                    f"{tag}: {name} is not pinned by revision and sha256."
                )
            path = Path(
                hf_hub_download(
                    artifact.get("repo_id") or CUT_REPO,
                    artifact["path"],
                    repo_type="dataset",
                    revision=artifact["revision"],
                    token=token,
                )
            )
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(chunk)
            if digest.hexdigest() != artifact["sha256"]:
                raise CertifyError(
                    f"{artifact['path']} at {artifact['revision']} hashes to "
                    f"{digest.hexdigest()}, not the manifest's {artifact['sha256']}."
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
        f"Certifies Microcosm UK national cut `{context['national_tag']}` "
        f"(policyengine-uk {context['model']}, core {context['core']}).",
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
            "- [ ] **UC floor acceptance.** This cut has a measured floor failure "
            "and needs explicit acceptance in its pull request "
            "(data-certification.md, recorded exceptions). Accepted by: ___"
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


def commit_message(national_tag: str, *, co_authors: Sequence[str] = ()) -> str:
    """The certification commit's message, crediting only the named co-authors."""

    message = f"Certify Microcosm UK cut {national_tag}"
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
    parser.add_argument("--national-tag", required=True)
    route = parser.add_mutually_exclusive_group(required=True)
    route.add_argument("--dense-tag")
    route.add_argument("--keep-regional-release", action="store_true")
    route.add_argument("--area-coded-national-uri")
    route.add_argument("--drop-regional-release", action="store_true")
    parser.add_argument(
        "--accept-regional-model-mismatch",
        metavar="REASON",
        help=(
            "route regions to a regional cut built on another policyengine-uk, "
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
    parser.add_argument("--model-version", help="default: the cut's build-time model")
    parser.add_argument("--core-version", help="default: the cut's build-time core")
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
    check_tag(args.national_tag, "national")
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
    suffix = args.national_tag.rsplit("-", 1)[-1]
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

    replay_paths = []
    if args.resume_worktree:
        with timed_step("resume-worktree-validation", timings):
            replay_paths = validate_resume_worktree(
                repo, worktree, branch, base_commit, args.national_tag
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
            others = writer_check(args.national_tag, repo)
        if others:
            raise CertifyError(
                "this cut is already being certified: "
                + "; ".join(others)
                + ". Coordinate with its owner instead of opening a second PR."
            )

    with timed_step("national-manifest", timings):
        national = read_cut(args.national_tag, NATIONAL_RELEASE)
    model = args.model_version or national.model
    core = args.core_version or national.core
    if not covers(national, model):
        raise CertifyError(
            f"the national cut was built with policyengine-uk {national.model} and "
            f"claims {list(national.claims)}; it cannot be certified for {model}."
        )
    with timed_step("old-bundle", timings):
        old_bundle = json.loads(
            subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{base_commit}:src/policyengine/data/bundle/manifest.json",
                ],
                cwd=repo,
                text=True,
            )
        )
    record = old_bundle["data_releases"]["uk"]
    flags = regional_flags(args, record)
    new_regional = regional_tag(flags)
    regional_summary = shlex.join(flags)
    dense, mismatch, regional_artifacts = None, False, {}
    if new_regional is not None:
        with timed_step("regional-manifest", timings):
            dense = read_cut(new_regional, DENSE_RELEASE)
        if not covers(dense, model):
            if not args.accept_regional_model_mismatch:
                raise CertifyError(
                    f"the regional cut {new_regional} was built with policyengine-uk "
                    f"{dense.model} and claims {list(dense.claims)}, so certification "
                    f"for {model} will refuse it. Routes: a dense cut on {model} "
                    "(--dense-tag), --drop-regional-release, an enhanced-FRS release "
                    "that claims the version (--area-coded-national-uri), or an "
                    "accepted, recorded mismatch (--accept-regional-model-mismatch) "
                    "with policyengine.py's audited evidence."
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
                regional_artifacts = fetch_regional_microdata(
                    new_regional, DENSE_RELEASE, timings=timings
                )
    old_regional = regional_tag(
        ["--regional-manifest-uri", record.get("regional_source_manifest_uri") or ""]
    )

    plan = {
        "base_ref": base_ref,
        "base_commit": base_commit,
        "national_tag": args.national_tag,
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
    run(sync, step="initial-sync")
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
            "--manifest-uri", manifest_uri(args.national_tag, NATIONAL_RELEASE),
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
    )  # fmt: skip
    run(sync, step="certified-sync")
    run([python, "scripts/release_lock.py"], step="release-lock")
    run([python, "scripts/bundle.py", "generate", "--include-tros"], step="TROs")
    new_bundle = json.loads(
        (worktree / "src/policyengine/data/bundle/manifest.json").read_text()
    )
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
            [python, "-c", "import xdist"], check=False, step="check-pytest-xdist"
        )
        if xdist.returncode:
            run(
                ["uv", "pip", "install", "--python", python, "pytest-xdist"],
                step="install-pytest-xdist",
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
        "national_tag": args.national_tag,
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
        staged_paths = certification_changes(worktree, args.national_tag)
    if staged_paths:
        run(["git", "add", "--", *staged_paths], step="stage-certification")
    run(
        ["git", "commit", "-q", "-m",
         commit_message(args.national_tag, co_authors=args.co_author)]
    )  # fmt: skip
    print(f"\ncommitted on {branch} in {worktree}; PR body draft: {body}")
    if args.push:
        run(["git", "push", "-u", "origin", branch])
    if args.open_pr:
        run(
            ["gh", "pr", "create", "--draft", "--repo", PEPY_REPO, "--base", args.pr_base,
             "--head", branch, "--title",
             f"Certify Microcosm UK cut {args.national_tag}", "--body-file", str(body)]
        )  # fmt: skip
    return 0 if not tests_returncode else 1


if __name__ == "__main__":
    raise SystemExit(main())
