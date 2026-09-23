#!/usr/bin/env python3
"""Plans, argv and sha256 receipts for running a US build stage on Modal.

The Modal app is ``tools/modal_us_stage.py``; this module is its pure half.
It uses only the standard library, so the Modal client's own interpreter, the
container and the unit tests all import it, and nothing here imports
``modal``, touches the network or runs a build.

A *plan* is a small JSON file naming one stage of one registered tool, the
pushed commit whose tree runs it, the run it belongs to, and every input by
URI and sha256. The runner refuses a plan that pins anything loosely: the
commit is a full 40-hex sha, every input carries its digest, options and
environment overrides come from allowlists, and the flags the runner owns
(input paths, checkpoint and output locations) cannot be passed through.

Command line (no Modal needed)::

    python3 tools/modal_us_stage_plan.py validate PLAN.json
    python3 tools/modal_us_stage_plan.py digest FILE [FILE ...]
    python3 tools/modal_us_stage_plan.py verify-receipt RECEIPT.json \\
        --state-root DIR

See ``docs/us-modal-stage-runbook.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

PLAN_SCHEMA = "microcosm-modal-us-stage-plan/1"
RECEIPT_SCHEMA = "microcosm-modal-us-stage-receipt/1"
DEFAULT_REPO_URL = "https://github.com/PolicyEngine/microcosm.git"

# Named Modal volumes. Inputs live content-addressed under ``cas/sha256/``;
# each run keeps its tool state and receipts under ``runs/<run_id>/``.
INPUTS_VOLUME = "microcosm-us-stage-inputs"
RUNS_VOLUME = "microcosm-us-stage-runs"
INPUTS_MOUNT = "/vol/inputs"
RUNS_MOUNT = "/vol/runs"

# Container layout. The build tree is cloned at the plan's commit into
# IMAGE_REPO_ROOT and synced into IMAGE_VENV from its own uv.lock; the stage
# works on ephemeral local disk under WORK_ROOT and mirrors its state to the
# runs volume when it finishes.
IMAGE_REPO_ROOT = "/root/microcosm"
IMAGE_VENV = "/opt/venv"
# Local US builds run on 3.14 (runtime.python 3.14.4 in the #974 build
# manifest); the image matches so a Modal replay differs only by platform.
IMAGE_PYTHON_VERSION = "3.14"
IMAGE_UV_VERSION = "0.11.7"
RUNNER_HF_HUB_VERSION = "1.18.0"
WORK_ROOT = "/work"

# Modal list prices for standard (non-sandbox) compute, read from
# https://modal.com/pricing on 2026-09-22. Modal bills the higher of the
# request and actual use, so an estimate from the request is a floor only
# when the stage stays inside it.
CPU_USD_PER_CORE_SECOND = 0.0000131
MEMORY_USD_PER_GIB_SECOND = 0.00000222
# Modal applies this to the CPU and memory list price of a function set
# nonpreemptible=True (modal.com/docs/guide/preemption, read 2026-09-23).
NONPREEMPTIBLE_PRICE_MULTIPLIER = 3.0

# Every pattern is applied with ``fullmatch``: ``re.match`` with ``$`` would
# also accept the value followed by a newline.
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]{2,79}")
_BRANCH = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,99}")
_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")
_HF_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_HF_REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")
_ENV_KEY = re.compile(
    r"(?:MICROCOSM|POPULACE)_[A-Z0-9_]+|(?:OMP|MKL|OPENBLAS|NUMEXPR)_NUM_THREADS"
)
# Names that look like credentials. A plan may not set one, even under an
# allowlisted prefix (its value would be copied into every receipt), and the
# runner removes them from the tool's environment.
_CREDENTIAL_ENV = re.compile(
    r"KEY|TOKEN|SECRET|PASSW|SIGNING|CREDENTIAL", re.IGNORECASE
)
_GITHUB_URL = re.compile(r"https://github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.git")


class PlanError(ValueError):
    """A plan the runner refuses to execute."""


def is_credential_env_key(key: str) -> bool:
    """Whether an environment variable's name looks like a credential."""

    return _CREDENTIAL_ENV.search(key) is not None


# --------------------------------------------------------------------------- #
# Resources                                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Resources:
    """One Modal resource class: a fixed request per decorated function."""

    name: str
    cpu: float
    memory_mib: int
    timeout_s: int

    @property
    def memory_gib(self) -> float:
        return self.memory_mib / 1024

    def estimated_usd(self, wall_seconds: float, multiplier: float = 1.0) -> float:
        """List-price cost of holding this request for ``wall_seconds``."""

        per_second = (
            self.cpu * CPU_USD_PER_CORE_SECOND
            + self.memory_gib * MEMORY_USD_PER_GIB_SECOND
        )
        return round(per_second * wall_seconds * multiplier, 2)


# Sized from measured local peaks (see MEASURED below): the heavy class holds
# the ACS materialize stage on the state SOI surface (~94 GB reported on
# 2026-09-22, 74.8 GB measured on the totals surface) with headroom; the
# engine pass is single-threaded (4,990 CPU-s over 5,067 wall-s), so four
# cores are for hashing and BLAS, not the microsimulation.
CHECK = Resources("check", cpu=2.0, memory_mib=8 * 1024, timeout_s=30 * 60)
HEAVY = Resources("heavy", cpu=4.0, memory_mib=128 * 1024, timeout_s=8 * 3600)
LIGHT = Resources("light", cpu=2.0, memory_mib=48 * 1024, timeout_s=4 * 3600)
RESOURCE_CLASSES = {item.name: item for item in (CHECK, HEAVY, LIGHT)}


@dataclass(frozen=True)
class Measured:
    """A local measurement a resource class was sized from."""

    peak_rss_bytes: int
    wall_seconds: float
    cpu_seconds: float
    source: str


_ACS_974 = (
    "experiments/us-acs-local-hours-rebuild-20260922/"
    "run-resources-and-staging-excerpt.json (#974, totals SOI surface)"
)
MEASURED = {
    ("us-acs-local-release", "materialize"): Measured(
        74_760_110_080, 5067.1, 4990.4, _ACS_974
    ),
    ("us-acs-local-release", "calibrate"): Measured(
        67_617_161_216, 415.1, 471.9, _ACS_974
    ),
    ("us-acs-local-release", "qa"): Measured(21_867_954_176, 172.2, 168.1, _ACS_974),
    ("us-acs-local-release", "finalize"): Measured(
        23_333_388_288, 78.9, 74.6, _ACS_974
    ),
    ("us-acs-local-release", "package"): Measured(21_777_367_040, 78.9, 73.7, _ACS_974),
}


# --------------------------------------------------------------------------- #
# Tool registry                                                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OptionFlag:
    flag: str
    kind: type


@dataclass(frozen=True)
class StageSpec:
    name: str
    resources: Resources
    required_inputs: tuple[str, ...]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    # Path of the tool in the pinned tree; None for an inline ``-c`` tool.
    script: str | None
    inputs: tuple[str, ...]
    stages: Mapping[str, StageSpec]
    options: Mapping[str, OptionFlag]
    owned_flags: frozenset[str]
    argv_builder: Callable[[Plan, Mapping[str, str], str], list[str]]
    # Name of the tool's own argparse entry point, called by the check to
    # validate the built argv against the pinned commit without running.
    parse_function: str | None = None


ACS_LOCAL_RELEASE_ARTIFACT = "populace_us_2024_acs_local.h5"


def _acs_local_release_argv(
    plan: Plan, input_paths: Mapping[str, str], state_dir: str
) -> list[str]:
    state = PurePosixPath(state_dir)
    argv = [
        "--stage",
        plan.stage,
        "--staging-h5",
        input_paths["staging_h5"],
        "--staging-summary",
        input_paths["staging_summary"],
        "--ladder",
        input_paths["ladder"],
    ]
    if "feed" in plan.inputs:
        argv += [
            "--feed",
            input_paths["feed"],
            "--feed-sha256",
            plan.inputs["feed"].sha256,
        ]
    argv += [
        "--checkpoint-dir",
        str(state / "checkpoints"),
        "--out-h5",
        str(state / ACS_LOCAL_RELEASE_ARTIFACT),
    ]
    if plan.stage in {"package", "all"}:
        argv += ["--out", str(state / "out")]
    return argv


_ACS_BASE_INPUTS = ("staging_h5", "staging_summary", "ladder")
US_ACS_LOCAL_RELEASE = ToolSpec(
    name="us-acs-local-release",
    script="tools/build_us_acs_local_release.py",
    inputs=("staging_h5", "staging_summary", "feed", "ladder"),
    stages={
        "materialize": StageSpec("materialize", HEAVY, (*_ACS_BASE_INPUTS, "feed")),
        "calibrate": StageSpec("calibrate", HEAVY, _ACS_BASE_INPUTS),
        "qa": StageSpec("qa", LIGHT, _ACS_BASE_INPUTS),
        "finalize": StageSpec("finalize", LIGHT, _ACS_BASE_INPUTS),
        "package": StageSpec("package", LIGHT, _ACS_BASE_INPUTS),
        "all": StageSpec("all", HEAVY, (*_ACS_BASE_INPUTS, "feed")),
    },
    options={
        "families": OptionFlag("--families", str),
        "geographies": OptionFlag("--geographies", str),
        "soi_mode": OptionFlag("--soi-mode", str),
        "epochs": OptionFlag("--epochs", int),
        "epoch_batch": OptionFlag("--epoch-batch", int),
        "max_weight_ratio": OptionFlag("--max-weight-ratio", float),
        "target_loss_cap": OptionFlag("--target-loss-cap", float),
        "l2_lambda": OptionFlag("--l2-lambda", float),
        "seed": OptionFlag("--seed", int),
        "resume": OptionFlag("--resume", bool),
        "batch": OptionFlag("--batch", int),
        "hh_chunk": OptionFlag("--hh-chunk", int),
        "allow_partial_geography": OptionFlag("--allow-partial-geography", bool),
    },
    owned_flags=frozenset(
        {
            "--stage",
            "--staging-h5",
            "--staging-summary",
            "--feed",
            "--feed-sha256",
            "--ladder",
            "--checkpoint-dir",
            "--out-h5",
            "--out-summary",
            "--gate-report",
            "--out",
            # A Modal run always builds a clean clone of a pushed commit.
            "--allow-dirty",
        }
    ),
    argv_builder=_acs_local_release_argv,
    parse_function="_parse_args",
)

# A near-free end-to-end proof of the run path (inputs staged and verified,
# the synced environment imported, a state file written, mirrored to the
# runs volume and listed in a receipt) on any pushed commit, because the
# code is inline rather than a file in the pinned tree.
_RUNNER_SMOKE_CODE = """\
import importlib.metadata as metadata
import json
import pathlib
import sys

import microcosm.build  # noqa: F401 - proves the synced environment imports

state = pathlib.Path(sys.argv[1])
rows = [
    {"input": pathlib.Path(p).parent.name, "bytes": pathlib.Path(p).stat().st_size}
    for p in sys.argv[2:]
]
payload = {"inputs": rows, "policyengine_us": metadata.version("policyengine-us")}
(state / "smoke").mkdir(parents=True, exist_ok=True)
(state / "smoke" / "inputs.json").write_text(json.dumps(payload, indent=2) + "\\n")
print("runner smoke ok:", json.dumps(payload))
"""


def _runner_smoke_argv(
    plan: Plan, input_paths: Mapping[str, str], state_dir: str
) -> list[str]:
    return [
        "-c",
        _RUNNER_SMOKE_CODE,
        state_dir,
        *(input_paths[name] for name in sorted(plan.inputs)),
    ]


RUNNER_SMOKE = ToolSpec(
    name="runner-smoke",
    script=None,
    inputs=("feed", "ladder", "staging_summary", "hub_file"),
    stages={"smoke": StageSpec("smoke", CHECK, ())},
    options={},
    owned_flags=frozenset(),
    argv_builder=_runner_smoke_argv,
)

TOOLS: dict[str, ToolSpec] = {
    tool.name: tool for tool in (US_ACS_LOCAL_RELEASE, RUNNER_SMOKE)
}


# --------------------------------------------------------------------------- #
# Inputs                                                                       #
# --------------------------------------------------------------------------- #


def cas_volume_path(sha256: str, filename: str) -> str:
    """Content-addressed path of an uploaded input inside INPUTS_VOLUME."""

    _require_sha256(sha256, "cas digest")
    _require_filename(filename)
    return f"cas/sha256/{sha256}/{filename}"


@dataclass(frozen=True)
class InputRef:
    name: str
    uri: str
    sha256: str
    kind: str
    filename: str
    volume_path: str | None = None
    repo_type: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    path_in_repo: str | None = None

    def to_json(self) -> dict[str, str]:
        return {"uri": self.uri, "sha256": self.sha256}


def _require_sha256(value: object, what: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise PlanError(f"{what}: expected 64 lowercase hex characters, got {value!r}")
    return value


def _require_filename(name: str) -> str:
    if not _FILENAME.fullmatch(name):
        raise PlanError(f"unsafe file name {name!r}")
    return name


def _safe_relative_path(text: str, what: str) -> PurePosixPath:
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in text.split("/"))
    ):
        raise PlanError(f"{what}: {text!r} must be a clean relative path")
    for part in path.parts:
        _require_filename(part)
    return path


def parse_input(name: str, spec: object) -> InputRef:
    """Parse one plan input: ``{"uri": ..., "sha256": ...}``.

    ``volume://<path>`` names a file in INPUTS_VOLUME; a path under
    ``cas/sha256/<digest>/`` must carry the input's own digest.
    ``hf://<datasets|models>/<org>/<name>@<revision>/<path>`` names a file
    in a Hugging Face repo at an explicit revision (a public repo needs no
    token). The sha256 is verified after staging either way.
    """

    if not isinstance(spec, Mapping) or set(spec) != {"uri", "sha256"}:
        raise PlanError(f"input {name!r}: expected exactly {{'uri', 'sha256'}}")
    uri = spec["uri"]
    sha256 = _require_sha256(spec["sha256"], f"input {name!r} sha256")
    if not isinstance(uri, str):
        raise PlanError(f"input {name!r}: uri must be a string")
    if uri.startswith("volume://"):
        rel = _safe_relative_path(uri.removeprefix("volume://"), f"input {name!r}")
        parts = rel.parts
        if parts[:2] == ("cas", "sha256"):
            if len(parts) != 4 or parts[2] != sha256:
                raise PlanError(
                    f"input {name!r}: content-addressed path {uri!r} does not "
                    f"name its own digest {sha256}"
                )
        return InputRef(
            name=name,
            uri=uri,
            sha256=sha256,
            kind="volume",
            filename=rel.name,
            volume_path=str(rel),
        )
    if uri.startswith("hf://"):
        body = uri.removeprefix("hf://")
        repo_type_part, _, rest = body.partition("/")
        repo_type = {"datasets": "dataset", "models": "model"}.get(repo_type_part)
        if repo_type is None:
            raise PlanError(
                f"input {name!r}: hf URI must start hf://datasets/ or hf://models/"
            )
        org, _, rest = rest.partition("/")
        name_and_rev, _, path_in_repo = rest.partition("/")
        repo_name, at, revision = name_and_rev.partition("@")
        repo_id = f"{org}/{repo_name}"
        if not at or not _HF_REVISION.fullmatch(revision):
            raise PlanError(
                f"input {name!r}: hf URI needs an explicit @revision "
                "(tag, branch without '/', or commit)"
            )
        if not _HF_REPO_ID.fullmatch(repo_id):
            raise PlanError(f"input {name!r}: bad Hugging Face repo id {repo_id!r}")
        rel = _safe_relative_path(path_in_repo, f"input {name!r} path")
        return InputRef(
            name=name,
            uri=uri,
            sha256=sha256,
            kind="hf",
            filename=rel.name,
            repo_type=repo_type,
            repo_id=repo_id,
            revision=revision,
            path_in_repo=str(rel),
        )
    raise PlanError(f"input {name!r}: unsupported uri {uri!r} (volume:// or hf://)")


def input_local_path(work_root: str, ref: InputRef) -> str:
    """Where the container stages an input; stable across a run's stages."""

    return str(PurePosixPath(work_root) / "inputs" / ref.name / ref.filename)


# --------------------------------------------------------------------------- #
# Plans                                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Plan:
    tool: ToolSpec
    stage: str
    run_id: str
    repo_url: str
    commit: str
    branch: str
    inputs: Mapping[str, InputRef]
    options: Mapping[str, object] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    # Runner-side budget: the tool is stopped after this many seconds, so a
    # stage's cost is bounded below the resource class's hard timeout.
    max_wall_seconds: int | None = None
    # Run on Modal's non-preemptible placement, at NONPREEMPTIBLE_PRICE_MULTIPLIER
    # times the list price. A preempted stage restarts from scratch, and a
    # multi-hour materialize was preempted twice in three hours (runbook).
    nonpreemptible: bool = False

    @property
    def stage_spec(self) -> StageSpec:
        return self.tool.stages[self.stage]

    @property
    def resources(self) -> Resources:
        return self.stage_spec.resources

    @property
    def price_multiplier(self) -> float:
        return NONPREEMPTIBLE_PRICE_MULTIPLIER if self.nonpreemptible else 1.0


_PLAN_KEYS = {
    "schema",
    "tool",
    "stage",
    "run_id",
    "source",
    "inputs",
    "options",
    "env",
    "max_wall_seconds",
    "nonpreemptible",
}
# Time the container keeps for staging inputs, hashing and mirroring state.
_RUNNER_OVERHEAD_SECONDS = 15 * 60


def parse_plan(data: object) -> Plan:
    """Validate a plan mapping; refuse anything loosely pinned."""

    if not isinstance(data, Mapping):
        raise PlanError("plan must be a JSON object")
    unknown = set(data) - _PLAN_KEYS
    if unknown:
        raise PlanError(f"unknown plan keys: {sorted(unknown)}")
    if data.get("schema") != PLAN_SCHEMA:
        raise PlanError(f"plan schema must be {PLAN_SCHEMA!r}")
    tool_name = data.get("tool")
    tool = TOOLS.get(tool_name) if isinstance(tool_name, str) else None
    if tool is None:
        raise PlanError(f"unknown tool {tool_name!r}; known: {sorted(TOOLS)}")
    stage = data.get("stage")
    if not isinstance(stage, str) or stage not in tool.stages:
        raise PlanError(
            f"tool {tool.name!r} has no stage {stage!r}; known: {sorted(tool.stages)}"
        )
    run_id = data.get("run_id")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise PlanError(
            f"run_id {run_id!r}: 3-80 chars of lowercase letters, digits, . _ -"
        )

    source = data.get("source")
    if not isinstance(source, Mapping) or not {"commit", "branch"} <= set(source):
        raise PlanError("source needs 'commit' and 'branch' (and optional 'repo_url')")
    if set(source) - {"commit", "branch", "repo_url"}:
        raise PlanError(
            f"unknown source keys: {sorted(set(source) - {'commit', 'branch', 'repo_url'})}"
        )
    commit = source["commit"]
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise PlanError("source.commit must be a full 40-hex lowercase sha")
    branch = source["branch"]
    if not isinstance(branch, str) or not _BRANCH.fullmatch(branch) or ".." in branch:
        raise PlanError(f"source.branch {branch!r} is not a safe branch name")
    repo_url = source.get("repo_url", DEFAULT_REPO_URL)
    if not isinstance(repo_url, str) or not _GITHUB_URL.fullmatch(repo_url):
        raise PlanError("source.repo_url must be https://github.com/<org>/<repo>.git")

    raw_inputs = data.get("inputs")
    if not isinstance(raw_inputs, Mapping):
        raise PlanError("inputs must be an object of name -> {uri, sha256}")
    unknown_inputs = set(raw_inputs) - set(tool.inputs)
    if unknown_inputs:
        raise PlanError(
            f"tool {tool.name!r} takes no inputs {sorted(unknown_inputs)}; "
            f"known: {list(tool.inputs)}"
        )
    missing = [
        name for name in tool.stages[stage].required_inputs if name not in raw_inputs
    ]
    if missing:
        raise PlanError(f"stage {stage!r} requires inputs {missing}")
    inputs = {name: parse_input(name, raw_inputs[name]) for name in sorted(raw_inputs)}
    local_paths = [input_local_path(WORK_ROOT, ref) for ref in inputs.values()]
    if len(set(local_paths)) != len(local_paths):
        raise PlanError("two inputs would stage to the same path")

    raw_options = data.get("options", {})
    if not isinstance(raw_options, Mapping):
        raise PlanError("options must be an object")
    options: dict[str, object] = {}
    for key in sorted(raw_options):
        option = tool.options.get(key)
        if option is None:
            raise PlanError(
                f"option {key!r} is not allowlisted for {tool.name!r}; "
                f"known: {sorted(tool.options)}"
            )
        value = raw_options[key]
        if option.kind is bool:
            ok = isinstance(value, bool)
        elif option.kind is int:
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif option.kind is float:
            ok = isinstance(value, int | float) and not isinstance(value, bool)
        else:
            ok = isinstance(value, str) and value != "" and not value.startswith("-")
        if not ok:
            raise PlanError(
                f"option {key!r} must be {option.kind.__name__}, got {value!r}"
            )
        options[key] = value

    raw_env = data.get("env", {})
    if not isinstance(raw_env, Mapping):
        raise PlanError("env must be an object")
    env: dict[str, str] = {}
    for key in sorted(raw_env):
        if not isinstance(key, str) or not _ENV_KEY.fullmatch(key):
            raise PlanError(
                f"env {key!r} is not allowlisted (MICROCOSM_*, POPULACE_*, *_NUM_THREADS)"
            )
        if is_credential_env_key(key):
            raise PlanError(
                f"env {key!r} names a credential; a plan's values are copied into "
                "every receipt, and the runner never passes credentials to the tool"
            )
        if not isinstance(raw_env[key], str):
            raise PlanError(f"env {key!r} must be a string")
        env[key] = raw_env[key]

    max_wall = data.get("max_wall_seconds")
    ceiling = tool.stages[stage].resources.timeout_s - _RUNNER_OVERHEAD_SECONDS
    if max_wall is not None and (
        not isinstance(max_wall, int)
        or isinstance(max_wall, bool)
        or not 60 <= max_wall <= ceiling
    ):
        raise PlanError(
            f"max_wall_seconds must be an integer from 60 to {ceiling} for "
            f"stage {stage!r}"
        )

    nonpreemptible = data.get("nonpreemptible", False)
    if not isinstance(nonpreemptible, bool):
        raise PlanError("nonpreemptible must be true or false")
    if nonpreemptible and tool.stages[stage].resources.name == "check":
        raise PlanError("the check class always runs preemptible")

    return Plan(
        tool=tool,
        stage=stage,
        run_id=run_id,
        repo_url=repo_url,
        commit=commit,
        branch=branch,
        inputs=inputs,
        options=options,
        env=env,
        max_wall_seconds=max_wall,
        nonpreemptible=nonpreemptible,
    )


def load_plan(path: Path) -> tuple[dict, Plan]:
    data = json.loads(Path(path).read_text())
    return data, parse_plan(data)


def canonical_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def plan_digest(data: object) -> str:
    return hashlib.sha256(canonical_json(data).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Image and argv                                                               #
# --------------------------------------------------------------------------- #


def image_build_commands(plan: Plan, repo_root: str = IMAGE_REPO_ROOT) -> list[str]:
    """Shell steps that pin the image's tree and environment to the plan.

    The tree is a shallow clone of the pushed commit (so ``git rev-parse``
    and ``git status`` inside the tools report the real code vintage), and
    the environment is synced from that tree's own uv.lock with --frozen.
    """

    root = shlex.quote(repo_root)
    commit = shlex.quote(plan.commit)
    return [
        f"git init -q {root}",
        f"git -C {root} remote add origin {shlex.quote(plan.repo_url)}",
        f"git -C {root} fetch -q --depth 1 origin {commit}",
        f"git -C {root} checkout -q -B {shlex.quote(plan.branch)} FETCH_HEAD",
        f'test "$(git -C {root} rev-parse HEAD)" = {commit}',
        f"cd {root} && uv sync --all-packages --extra us --frozen",
        f'test -z "$(git -C {root} status --porcelain)"',
    ]


_BRANCH_CHECK_REF = "refs/branch-check/tip"


def branch_check_argvs(
    repo_url: str, branch: str, commit: str, scratch: str
) -> list[list[str]]:
    """Git steps that prove ``commit`` is reachable from ``branch`` on the remote.

    The image checks out the plan's branch name with ``checkout -B``, which
    names any commit; this is what makes the recorded branch true. A
    commits-only fetch (``--filter=tree:0``) of the branch into a scratch
    bare repository, then ``merge-base --is-ancestor``. The pinned clone is
    not touched. Run in order; see :func:`branch_verdict`.
    """

    return [
        ["git", "init", "-q", "--bare", scratch],
        [
            "git",
            "-C",
            scratch,
            "fetch",
            "-q",
            "--no-tags",
            "--filter=tree:0",
            repo_url,
            f"+refs/heads/{branch}:{_BRANCH_CHECK_REF}",
        ],
        ["git", "-C", scratch, "rev-parse", _BRANCH_CHECK_REF],
        [
            "git",
            "-C",
            scratch,
            "merge-base",
            "--is-ancestor",
            commit,
            _BRANCH_CHECK_REF,
        ],
    ]


def branch_verdict(
    branch: str,
    commit: str,
    *,
    fetch_returncode: int,
    fetch_stderr: str = "",
    tip: str | None = None,
    ancestor_returncode: int | None = None,
) -> dict[str, object]:
    """Interpret :func:`branch_check_argvs`; only a proven ancestry verifies."""

    if fetch_returncode != 0:
        detail = fetch_stderr.strip().splitlines()[-1:] or ["no output"]
        return {
            "branch_verified": False,
            "branch_tip": None,
            "branch_check": f"branch {branch!r} could not be fetched: {detail[0]}",
        }
    if ancestor_returncode == 0:
        how = "is the branch tip" if tip == commit else "is an ancestor of the tip"
        return {
            "branch_verified": True,
            "branch_tip": tip,
            "branch_check": f"commit {how}",
        }
    if ancestor_returncode == 1:
        why = f"commit is not reachable from branch {branch!r} (tip {tip})"
    else:
        why = f"ancestry check failed (exit {ancestor_returncode})"
    return {"branch_verified": False, "branch_tip": tip, "branch_check": why}


def option_argv(plan: Plan) -> list[str]:
    argv: list[str] = []
    for key, value in sorted(plan.options.items()):
        option = plan.tool.options[key]
        if option.flag in plan.tool.owned_flags:  # registry invariant
            raise PlanError(f"option {key!r} maps to runner-owned flag {option.flag}")
        if option.kind is bool:
            if value:
                argv.append(option.flag)
        else:
            argv += [option.flag, str(value)]
    return argv


def build_stage_argv(
    plan: Plan,
    *,
    python: str,
    input_paths: Mapping[str, str],
    state_dir: str,
) -> list[str]:
    """The exact argv the container runs (cwd = the cloned tree)."""

    missing = sorted(set(plan.inputs) - set(input_paths))
    if missing:
        raise PlanError(f"no staged path for inputs {missing}")
    tool_argv = plan.tool.argv_builder(plan, input_paths, state_dir)
    tool_argv += option_argv(plan)
    script = [] if plan.tool.script is None else [plan.tool.script]
    return [python, "-B", *script, *tool_argv]


def planned_argv(plan: Plan, work_root: str = WORK_ROOT) -> list[str]:
    paths = {
        name: input_local_path(work_root, ref) for name, ref in plan.inputs.items()
    }
    return build_stage_argv(
        plan,
        python=f"{IMAGE_VENV}/bin/python",
        input_paths=paths,
        state_dir=str(PurePosixPath(work_root) / "state"),
    )


def tool_environment(
    base: Mapping[str, str], plan_env: Mapping[str, str]
) -> tuple[dict[str, str], list[str]]:
    """The stage tool's environment, and the names removed from ``base``.

    The container's environment without any credential-looking variable
    (``HF_TOKEN`` from an attached Hub secret, Modal's own tokens), with
    ``HF_HUB_OFFLINE=1`` so the tool cannot reach the Hub, and with the
    plan's allowlisted overrides last. Only names are returned, for the
    receipt; never values.
    """

    for key in plan_env:
        if not _ENV_KEY.fullmatch(key) or is_credential_env_key(key):
            raise PlanError(f"env {key!r} may not be passed to the tool")
    removed = sorted(key for key in base if is_credential_env_key(key))
    dropped = set(removed)
    env = {key: value for key, value in base.items() if key not in dropped}
    env["HF_HUB_OFFLINE"] = "1"
    env.update(plan_env)
    return env, removed


# --------------------------------------------------------------------------- #
# Hashing, mirroring and receipts                                              #
# --------------------------------------------------------------------------- #

_CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path | str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def copy_with_sha256(src: Path | str, dst: Path | str) -> tuple[str, int]:
    """Copy ``src`` to ``dst`` hashing in the same pass; atomic rename."""

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    digest = hashlib.sha256()
    size = 0
    with open(src, "rb") as reader, open(tmp, "wb") as writer:
        while chunk := reader.read(_CHUNK):
            digest.update(chunk)
            writer.write(chunk)
            size += len(chunk)
    os.replace(tmp, dst)
    return digest.hexdigest(), size


def verify_digest(ref: InputRef, actual_sha256: str) -> None:
    if actual_sha256 != ref.sha256:
        raise PlanError(
            f"input {ref.name!r} ({ref.uri}) is sha256 {actual_sha256}, "
            f"plan pins {ref.sha256}"
        )


def tree_listing(root: Path | str) -> dict[str, tuple[int, int]]:
    """``{relative posix path: (bytes, mtime_ns)}`` for every regular file."""

    root = Path(root)
    listing: dict[str, tuple[int, int]] = {}
    if not root.exists():
        return listing
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            stat = path.stat()
            listing[path.relative_to(root).as_posix()] = (
                stat.st_size,
                stat.st_mtime_ns,
            )
    return listing


def mirror_actions(
    source: Mapping[str, tuple[int, int]], destination: Mapping[str, tuple[int, int]]
) -> tuple[list[str], list[str]]:
    """Files to copy and to delete so ``destination`` mirrors ``source``.

    A file is copied when it is new or its size or mtime differs (copies
    preserve mtime, so an untouched checkpoint is not re-sent); a file the
    stage deleted locally is deleted from the destination.
    """

    copy = sorted(
        path for path, meta in source.items() if destination.get(path) != meta
    )
    delete = sorted(path for path in destination if path not in source)
    return copy, delete


def mirror_tree(source: Path | str, destination: Path | str) -> dict[str, int]:
    source, destination = Path(source), Path(destination)
    copy, delete = mirror_actions(tree_listing(source), tree_listing(destination))
    for rel in copy:
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, target)
    for rel in delete:
        (destination / rel).unlink()
    return {"copied": len(copy), "deleted": len(delete)}


def hash_tree(root: Path | str) -> list[dict[str, object]]:
    root = Path(root)
    outputs = []
    for rel in tree_listing(root):
        sha, size = sha256_file(root / rel)
        outputs.append({"path": rel, "bytes": size, "sha256": sha})
    return outputs


def build_receipt(
    plan: Plan,
    plan_data: Mapping,
    *,
    argv: Sequence[str],
    returncode: int,
    started_at: str,
    finished_at: str,
    wall_seconds: float,
    peak_rss_bytes: int | None,
    inputs_verified: Sequence[Mapping[str, object]],
    outputs: Sequence[Mapping[str, object]],
    git: Mapping[str, object],
    runner: Mapping[str, object],
    prior_receipts: Sequence[Mapping[str, object]] = (),
    stopped_at_budget: bool = False,
    container_wall_seconds: float | None = None,
    prior_attempts: Sequence[Mapping[str, object]] = (),
    budget_seconds: int | None = None,
) -> dict[str, object]:
    resources = plan.resources
    # The tool's wall leaves out staging the inputs, hashing the state tree
    # and mirroring it to the volume; the container's wall is what is billed,
    # and so is every earlier attempt that preemption cut short.
    container: dict[str, object] = {}
    if container_wall_seconds is not None:
        prior_seconds = sum(
            float(item.get("elapsed_seconds") or 0.0) for item in prior_attempts
        )
        container = {
            "container_wall_seconds": round(container_wall_seconds, 1),
            "estimated_usd_container_at_list_price": resources.estimated_usd(
                container_wall_seconds, plan.price_multiplier
            ),
            "estimated_usd_all_attempts_at_list_price": resources.estimated_usd(
                container_wall_seconds + prior_seconds, plan.price_multiplier
            ),
        }
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "COMPLETED"
        if returncode == 0 and not stopped_at_budget
        else "FAILED",
        "tool": plan.tool.name,
        "stage": plan.stage,
        "run_id": plan.run_id,
        "plan_sha256": plan_digest(plan_data),
        "plan": plan_data,
        "source": {
            "repo_url": plan.repo_url,
            "commit": plan.commit,
            "branch": plan.branch,
            **dict(git),
        },
        "runner": dict(runner),
        "resources": {
            "class": resources.name,
            "cpu": resources.cpu,
            "memory_mib": resources.memory_mib,
            "timeout_s": resources.timeout_s,
            "nonpreemptible": plan.nonpreemptible,
        },
        "argv": list(argv),
        "returncode": returncode,
        "max_wall_seconds": plan.max_wall_seconds,
        "budget_seconds_this_attempt": budget_seconds,
        "stopped_at_budget": stopped_at_budget,
        "prior_unfinished_attempts": [dict(item) for item in prior_attempts],
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": round(wall_seconds, 1),
        "peak_rss_bytes": peak_rss_bytes,
        "estimated_usd_at_list_price": resources.estimated_usd(
            wall_seconds, plan.price_multiplier
        ),
        **container,
        "inputs": [dict(item) for item in inputs_verified],
        "prior_receipts": [dict(item) for item in prior_receipts],
        "outputs": [dict(item) for item in outputs],
    }


def verify_receipt(
    receipt: Mapping, state_root: Path | str, *, strict: bool = False
) -> list[str]:
    """Re-hash a fetched state tree against a receipt; return the problems."""

    if receipt.get("schema") != RECEIPT_SCHEMA:
        return [f"not a {RECEIPT_SCHEMA} receipt"]
    root = Path(state_root)
    problems: list[str] = []
    declared = set()
    for item in receipt.get("outputs", []):
        rel = str(item["path"])
        declared.add(rel)
        path = root / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
            continue
        sha, size = sha256_file(path)
        if size != item["bytes"]:
            problems.append(
                f"size mismatch: {rel} is {size} bytes, receipt {item['bytes']}"
            )
        elif sha != item["sha256"]:
            problems.append(f"sha256 mismatch: {rel}")
    if strict:
        extra = sorted(set(tree_listing(root)) - declared)
        problems += [f"not in receipt: {rel}" for rel in extra]
    return problems


def prior_state_problems(plan: Plan, run_identity: Mapping | None) -> list[str]:
    """Refusals for a stage whose predecessor state is missing or foreign.

    Every ACS stage after materialize re-verifies the staging digest itself
    (``_verify_run_identity``); this catches the same mismatch in the cheap
    check, before a paid container starts.
    """

    if plan.tool is not US_ACS_LOCAL_RELEASE or plan.stage in {"materialize", "all"}:
        return []
    if not run_identity:
        return [
            f"run {plan.run_id!r} has no checkpoints/run_identity.json on "
            f"{RUNS_VOLUME}; run materialize for this run first"
        ]
    problems = []
    if run_identity.get("staging_sha256") != plan.inputs["staging_h5"].sha256:
        problems.append(
            "staging_h5 sha256 differs from the run's materialized identity "
            f"({run_identity.get('staging_sha256')})"
        )
    if run_identity.get("ladder_sha256") not in (None, plan.inputs["ladder"].sha256):
        problems.append(
            "ladder sha256 differs from the run's materialized identity "
            f"({run_identity.get('ladder_sha256')})"
        )
    return problems


# --------------------------------------------------------------------------- #
# Attempts: a budget that holds across preemption restarts                      #
# --------------------------------------------------------------------------- #

ATTEMPT_SCHEMA = "microcosm-modal-us-stage-attempt/1"
# How often a running attempt records that it is still alive; a preempted
# attempt is charged up to its last heartbeat, so it can be undercounted by
# at most this much.
ATTEMPT_HEARTBEAT_SECONDS = 120
# An attempt left with less tool time than this refuses to start.
MIN_ATTEMPT_SECONDS = 60


def attempt_record(
    plan: Plan,
    plan_sha256: str,
    *,
    attempt_id: str,
    started_epoch: float,
    last_seen_epoch: float,
    finished: bool = False,
    receipt: str | None = None,
) -> dict[str, object]:
    return {
        "schema": ATTEMPT_SCHEMA,
        "attempt_id": attempt_id,
        "run_id": plan.run_id,
        "stage": plan.stage,
        "plan_sha256": plan_sha256,
        "started_epoch": round(started_epoch, 1),
        "last_seen_epoch": round(last_seen_epoch, 1),
        "elapsed_seconds": round(max(0.0, last_seen_epoch - started_epoch), 1),
        "finished": finished,
        "receipt": receipt,
    }


def unfinished_attempts(
    records: Iterable[Mapping], plan: Plan, plan_sha256: str
) -> list[dict[str, object]]:
    """Earlier attempts of this exact plan and stage that never wrote a receipt.

    Modal restarts a preempted function on the same input, from scratch and
    regardless of ``retries``; each such attempt is billed.
    """

    return [
        dict(record)
        for record in records
        if record.get("schema") == ATTEMPT_SCHEMA
        and record.get("run_id") == plan.run_id
        and record.get("stage") == plan.stage
        and record.get("plan_sha256") == plan_sha256
        and not record.get("finished")
    ]


def remaining_wall_seconds(
    plan: Plan, prior_unfinished: Sequence[Mapping]
) -> int | None:
    """The tool's wall budget for this attempt: the plan's ``max_wall_seconds``
    less the container time of every earlier unfinished attempt."""

    if plan.max_wall_seconds is None:
        return None
    spent = sum(float(item.get("elapsed_seconds") or 0.0) for item in prior_unfinished)
    return int(plan.max_wall_seconds - spent)


def summarize(plan: Plan) -> dict[str, object]:
    resources = plan.resources
    measured = MEASURED.get((plan.tool.name, plan.stage))
    summary: dict[str, object] = {
        "tool": plan.tool.name,
        "stage": plan.stage,
        "run_id": plan.run_id,
        "commit": plan.commit,
        "branch": plan.branch,
        "resources": {
            "class": resources.name,
            "cpu": resources.cpu,
            "memory_gib": resources.memory_gib,
            "timeout_h": resources.timeout_s / 3600,
            "nonpreemptible": plan.nonpreemptible,
        },
        "inputs": {name: ref.to_json() for name, ref in plan.inputs.items()},
        "argv": planned_argv(plan),
        "image_build_commands": image_build_commands(plan),
    }
    if measured is not None:
        summary["measured_locally"] = {
            "peak_rss_gb": round(measured.peak_rss_bytes / 1e9, 1),
            "wall_seconds": measured.wall_seconds,
            "source": measured.source,
        }
        summary["estimated_usd_at_measured_wall"] = resources.estimated_usd(
            measured.wall_seconds, plan.price_multiplier
        )
    if plan.max_wall_seconds is not None:
        summary["estimated_usd_at_max_wall"] = resources.estimated_usd(
            plan.max_wall_seconds + _RUNNER_OVERHEAD_SECONDS, plan.price_multiplier
        )
    return summary


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _digest_lines(paths: Iterable[Path]) -> list[str]:
    lines = []
    for path in paths:
        sha, size = sha256_file(path)
        remote = cas_volume_path(sha, path.name)
        lines.append(
            json.dumps(
                {
                    "file": str(path),
                    "bytes": size,
                    "input": {"uri": f"volume://{remote}", "sha256": sha},
                    "upload": (
                        f"modal volume put {INPUTS_VOLUME} "
                        f"{shlex.quote(str(path))} {remote}"
                    ),
                }
            )
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate a plan; print argv and sizing")
    validate.add_argument("plan", type=Path)
    digest = sub.add_parser("digest", help="sha256 + CAS upload command per file")
    digest.add_argument("paths", nargs="+", type=Path)
    verify = sub.add_parser("verify-receipt", help="re-hash a fetched state tree")
    verify.add_argument("receipt", type=Path)
    verify.add_argument("--state-root", type=Path, required=True)
    verify.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "validate":
        try:
            _, plan = load_plan(args.plan)
        except PlanError as error:
            print(f"REFUSED: {error}", file=sys.stderr)
            return 2
        print(json.dumps(summarize(plan), indent=2))
        return 0
    if args.command == "digest":
        for line in _digest_lines(args.paths):
            print(line)
        return 0
    receipt = json.loads(args.receipt.read_text())
    problems = verify_receipt(receipt, args.state_root, strict=args.strict)
    for problem in problems:
        print(problem, file=sys.stderr)
    print(
        json.dumps(
            {
                "verified": not problems,
                "outputs": len(receipt.get("outputs", [])),
                "problems": len(problems),
            }
        )
    )
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
