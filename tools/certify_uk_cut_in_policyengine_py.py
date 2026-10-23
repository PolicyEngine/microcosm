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
certification unless policyengine.py accepts a recorded mismatch; pass
``--accept-regional-model-mismatch REASON`` only where that option exists
(see ``docs/uk-cut-anyone.md``).

Needs ``HF_TOKEN`` (read ``policyengine/populace-uk-private``), PyPI, and
``gh`` for the writer check and the pull request.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


def resource_guard(root: Path) -> dict[str, float | int]:
    """Refuse work before spawning a process on an overloaded or full host."""

    load1 = os.getloadavg()[0]
    while not root.exists():
        root = root.parent
    free = shutil.disk_usage(root).free
    if load1 > 110 or free < 55_000_000_000:
        raise CertifyError(
            f"resource guard: load1={load1:.2f} (maximum 110), "
            f"free={free / 1e9:.2f} GB (minimum 55 decimal GB)."
        )
    return {"load1": load1, "free_bytes": free}


def record_timing(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


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


class Shell:
    """Runs commands in the policyengine.py worktree and records each one."""

    def __init__(
        self,
        cwd: Path,
        log: Path,
        *,
        dry_run: bool = False,
        timings: Path | None = None,
    ) -> None:
        self.cwd = cwd
        self.log = log
        self.dry_run = dry_run
        self.timings = timings or log.with_suffix(".timings.jsonl")
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
            timing.update(resource_guard(self.cwd))
            with self.log.open("a") as handle:
                handle.write(f"\n# {datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} $ {line}\n")
                handle.flush()
                result = run_logged(
                    argv,
                    cwd=self.cwd,
                    env={**os.environ, **(env or {})},
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
    """Open pull requests or branches already certifying this cut."""

    found = []
    result = subprocess.run(
        [
            "gh", "pr", "list", "--repo", PEPY_REPO, "--state", "open",
            "--search", national_tag, "--json", "number,title,headRefName",
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
        found.append(f"#{pr['number']} {pr['title']} ({pr['headRefName']})")
    branch = f"certify-uk-{national_tag.rsplit('-', 1)[-1]}"
    remote = subprocess.run(
        ["git", "ls-remote", "--heads", "origin", branch],
        cwd=pepy_repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if remote.stdout.strip():
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


_SURFACE_SCRIPT = (
    "import json; from policyengine_uk import variables; "
    "print(json.dumps({'all': sorted(variables), 'inputs': sorted("
    "n for n, v in variables.items() if v.is_input_variable())}))"
)


def model_surface(
    version: str,
    core_version: str | None = None,
    *,
    timings: Path | None = None,
    resource_root: Path | None = None,
) -> dict[str, list[str]]:
    """Read the country's existing variable registry at the requested model/core."""

    packages = ["--with", f"policyengine-uk=={version}"]
    if core_version:
        packages += ["--with", f"policyengine-core=={core_version}"]
    started = time.monotonic()
    timing = {
        "step": f"model_surface:{version}",
        "model_version": version,
        "core_version": core_version,
        "started_at": datetime.now(UTC).isoformat(),
        "returncode": None,
    }
    print(f"[start] model_surface:{version}", flush=True)
    try:
        timing.update(resource_guard(resource_root or Path.cwd()))
        result = subprocess.run(
        [
            "uv", "run", "--no-project", "--python", PEPY_PYTHON,
            *packages, "python", "-c", _SURFACE_SCRIPT,
        ],
        capture_output=True,
        text=True,
        check=True,
        env={k: v for k, v in os.environ.items() if k not in {"UV_FROZEN", "VIRTUAL_ENV"}},
        )  # fmt: skip
        timing["returncode"] = result.returncode
    except Exception as error:
        timing["returncode"] = getattr(error, "returncode", None)
        timing["error"] = str(error)
        raise
    finally:
        timing["elapsed_seconds"] = time.monotonic() - started
        if timings:
            record_timing(timings, timing)
        print(
            f"[done] model_surface:{version}: {timing['elapsed_seconds']:.3f}s "
            f"(exit {timing.get('returncode')})",
            flush=True,
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def input_surface_change(
    built: dict[str, list[str]], certified: dict[str, list[str]]
) -> dict[str, list[str]]:
    """What area runs lose when a file built on one model runs under another.

    ``dropped``: the file's input columns the certified model no longer
    defines; policyengine-core ignores them on load. ``defaulted``: inputs
    of the certified model the file cannot carry (new variables, and
    variables that were formulas when it was built), which take defaults.
    A built input that became a formula is in neither: its stored values
    still load.
    """

    built_inputs = set(built["inputs"])
    return {
        "dropped": sorted(built_inputs - set(certified["all"])),
        "defaulted": sorted(set(certified["inputs"]) - built_inputs),
    }


def overlay_diff(old_bundle: dict, new_bundle: dict) -> list[str]:
    old = set((old_bundle.get("dataset_overlays") or {}).get("uk", {}) or {})
    new = set((new_bundle.get("dataset_overlays") or {}).get("uk", {}) or {})
    return sorted(old - new)


def uc_floor_summary(receipt: dict[str, Any]) -> str:
    """Summarize a complete measured floor without truncating its JSON receipt."""
    measures = receipt.get("measures") or []
    if (
        receipt.get("check") != "uk_uc_floor"
        or not isinstance(receipt.get("passed"), bool)
        or not {"benefit_units", "spend"}.issubset(
            {measure.get("name") for measure in measures}
        )
        or any(
            not isinstance(measure.get("passed"), bool)
            or not isinstance(measure.get("ratio"), (int, float))
            or not math.isfinite(measure["ratio"])
            for measure in measures
        )
        or receipt["passed"] != all(measure["passed"] for measure in measures)
    ):
        raise CertifyError(
            "UC floor failed operationally: incomplete measured receipt."
        )
    status = "passed" if receipt["passed"] else "FAILED"
    ratios = ", ".join(
        f"{measure['name']}={measure['ratio']:.6f}" for measure in measures
    )
    return f"{status}; measured/benchmark ratios: {ratios}"


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
    if (context.get("uc_floor_receipt") or {}).get("passed") is False:
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
    surface = context.get("input_surface")
    if surface is not None:
        lines.append(
            "- [ ] **Area runs on the older local-area file** (accepted mismatch, "
            f"built on {context.get('regional_model')}): columns ignored on load: "
            f"{', '.join(surface['dropped']) or 'none'}; inputs that take defaults: "
            f"{', '.join(surface['defaulted']) or 'none'}. Confirm none matters "
            "for a reform the Budget dashboard shows by area."
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
    lines += [
        "- [ ] One writer: the UK hub and María know this PR exists.",
        "",
        "🤖 Generated with [Claude Code](https://claude.com/claude-code)",
    ]
    return "\n".join(lines) + "\n"


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
        help="forwarded to certify-data where policyengine.py supports it",
    )
    parser.add_argument(
        "--allow-dropped-columns",
        metavar="REASON",
        help=(
            "with an accepted mismatch, proceed although the newer model no "
            "longer defines some of the regional file's input columns"
        ),
    )
    parser.add_argument("--model-version", help="default: the cut's build-time model")
    parser.add_argument("--core-version", help="default: the cut's build-time core")
    parser.add_argument(
        "--pepy-repo",
        type=Path,
        default=Path("~/PolicyEngine/policyengine.py").expanduser(),
    )
    parser.add_argument("--worktree", type=Path, help="default: beside the repo")
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="base branch/ref for both the old bundle and the new worktree",
    )
    parser.add_argument("--skip-writer-check", metavar="WHY")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--test-workers", type=int, default=4)
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--pr-base",
        default="main",
        help="pull-request target branch (default: main), independent of --base-ref",
    )
    parser.add_argument("--open-pr", action="store_true", help="draft; needs --push")
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    args = parser.parse_args(argv)
    if args.open_pr and not args.push:
        parser.error("--open-pr needs --push.")
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
    suffix = args.national_tag.rsplit("-", 1)[-1]
    branch = f"certify-uk-{suffix}"
    worktree = args.worktree or repo.parent / "_worktrees" / f"pepy-{branch}"
    log = worktree.parent / f"{worktree.name}.certify.log"
    timings = log.with_suffix(".timings.jsonl")
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
    dense, surface = None, None
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
                    "where policyengine.py supports one."
                )
            surface = input_surface_change(
                model_surface(
                    dense.model, dense.core, timings=timings, resource_root=repo
                ),
                model_surface(model, core, timings=timings, resource_root=repo),
            )
            if surface["dropped"] and not args.allow_dropped_columns:
                raise CertifyError(
                    f"policyengine-uk {model} no longer defines these input columns "
                    f"of the {dense.model} local-area file, so area runs would lose "
                    f"them: {', '.join(surface['dropped'])}. Build a dense cut on "
                    f"{model}, or pass --allow-dropped-columns with a reason."
                )
            print(json.dumps({"input_surface_change": surface}, indent=2))
            flags += [
                "--accept-regional-model-mismatch",
                args.accept_regional_model_mismatch,
            ]
            regional_summary += f" (regional built on {dense.model}; mismatch accepted)"
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
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0
    if worktree.exists():
        raise CertifyError(f"{worktree} exists; remove it or pass --worktree.")
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
    run = Shell(worktree, log, timings=timings)
    python = str(worktree / ".venv/bin/python")
    sync = ["uv", "sync", "--frozen", "--extra", "uk", "--extra", "dev",
            "--python", PEPY_PYTHON]  # fmt: skip
    run(sync, step="initial-sync")
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
        [python, "-m", "pytest", "tests/test_household_calculator_snapshot.py", "-q"],
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
        uc_summary = uc_floor_summary(receipt)
    except (json.JSONDecodeError, AttributeError, TypeError) as error:
        raise CertifyError(
            "UC floor failed operationally: invalid measured receipt."
        ) from error
    if uc.returncode != (0 if receipt["passed"] else 1):
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
            [python, "-m", "pytest", "-n", str(args.test_workers), "-vv",
             "--durations=20", "-p", "no:cacheprovider"],
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
        "uc_floor_receipt": receipt,
        "tests": tests_summary,
        "tests_failed": tests_failed,
        "constants": ", ".join(constants),
        "dropped_overlays": overlay_diff(old_bundle, new_bundle),
        "warnings": warnings,
        "input_surface": surface,
        "regional_model": dense.model if dense is not None else None,
    }
    body = worktree.parent / f"{worktree.name}.pr-body.md"
    body.write_text(pr_body(context))
    run(["git", "add", "-A"])
    run(
        ["git", "commit", "-q", "-m",
         f"Certify Microcosm UK cut {args.national_tag}\n\n"
         "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"]
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
