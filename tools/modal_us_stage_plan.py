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
    python3 tools/modal_us_stage_plan.py upload-commands PLAN.json \\
        ROLE=PATH [ROLE=PATH ...] [--shell]
    python3 tools/modal_us_stage_plan.py verify-receipt RECEIPT.json \\
        --state-root DIR [--strict] [--prefix SUBDIR] [--allow-failed]
    python3 tools/modal_us_stage_plan.py compare-lineage RECEIPT.json \\
        --reference LOCAL_REFERENCE.json --run-context stage_run_context.json

``upload-commands`` prints ``modal volume`` commands; it runs none of them.

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
import signal
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import zip_longest
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
# manifest); the image matches the minor version. It is the standard (GIL)
# build on Linux x86_64, so a replay of a local run on free-threaded 3.14 on
# macOS arm64 (Route A's base) differs by interpreter build and platform
# too. compare-lineage checks the outputs the local run left (its first two
# outer stages), not the stages after them.
IMAGE_PYTHON_VERSION = "3.14"
IMAGE_UV_VERSION = "0.11.7"
RUNNER_HF_HUB_VERSION = "1.18.0"
WORK_ROOT = "/work"

# Modal list prices for standard (non-sandbox) compute, read from
# https://modal.com/pricing on 2026-09-22. Modal bills the higher of the
# request and actual use, so an estimate from the request is exact while the
# stage stays inside its request and a floor when it uses more.
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
# One component of a home-cache path: a file name that may start with a dot.
_HOME_PART = re.compile(r"\.?[A-Za-z0-9][A-Za-z0-9._-]{0,199}")
_HF_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_HF_REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")
#: The thread-count variables a plan may override. BLIS is here because Modal
#: sets BLIS_NUM_THREADS in the container alongside the other three.
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
#: PYTHONUNBUFFERED is allowed so a plan can match a local run's environment
#: exactly (Route A's base ran with only PYTHONUNBUFFERED=1). It changes when
#: log lines reach the runner, not what the tool computes.
_ENV_KEY = re.compile(
    r"(?:MICROCOSM|POPULACE)_[A-Z0-9_]+"
    r"|(?:OMP|MKL|OPENBLAS|NUMEXPR|BLIS)_NUM_THREADS"
    r"|PYTHONUNBUFFERED"
)
# Names that look like credentials. A plan may not set one, even under an
# allowlisted prefix (its value would be copied into every receipt), and the
# runner removes them from the tool's environment.
_CREDENTIAL_ENV = re.compile(
    r"KEY|TOKEN|SECRET|PASSW|SIGNING|CREDENTIAL", re.IGNORECASE
)
_GITHUB_URL = re.compile(r"https://github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.git")
#: The hash seed is a determinism input of the base's pinned tool: at
#: 4b57d15a2 its ``--stage all`` parent starts every outer stage in a child
#: interpreter with ``PYTHONHASHSEED`` defaulted to "0"
#: (``_staged_subprocess_environment``: ``setdefault``, so a value already in
#: its environment wins), records the value in the run config it locks
#: (``_stage_run_config``, ``thread_environment``; "0" when unset), and
#: refuses a named stage without one. Route A's local base recorded "0"
#: (its stage_run_context.json), and its command set only PYTHONUNBUFFERED
#: (base-config.json). The runner never passes the container's value on (``tool_environment``),
#: a plan cannot set it (``_ENV_KEY``), and the runner refuses a tool
#: environment that would carry any other value (``hash_seed_problem``).
HASH_SEED_ENV = "PYTHONHASHSEED"
HASH_SEED = "0"


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
    """One Modal resource class: a fixed request per decorated function.

    ``cpu_limit`` is an explicit CPU limit (``cpu=(request, limit)`` on the
    function). Without one, Modal's default soft CPU limit is 16 physical
    cores above the request, the host throttles CPU use above the limit, and
    CPU and memory are billed at the higher of the request and actual use
    (modal.com/docs/guide/resources, read 2026-09-29). So a class without a
    limit can bill CPU above its request, and :meth:`estimated_usd` is then a
    floor. For a class whose limit equals its request, Modal throttles CPU
    use above the limit, which is the request, and the estimate counts CPU
    at the request on that basis. Memory has no limit in any class: use
    above the request is billed at use.
    """

    name: str
    cpu: float
    memory_mib: int
    timeout_s: int
    cpu_limit: float | None = None

    def __post_init__(self) -> None:
        if self.cpu_limit is not None and self.cpu_limit < self.cpu:
            raise ValueError(
                f"class {self.name!r}: CPU limit {self.cpu_limit} is below "
                f"the request {self.cpu}"
            )

    @property
    def modal_cpu(self) -> float | tuple[float, float]:
        """The ``cpu`` argument of the class's Modal functions."""

        return self.cpu if self.cpu_limit is None else (self.cpu, self.cpu_limit)

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
# The PUF-support base (tools/build_us_puf_support_base.py, Route A). Its
# local peak was 72.47 GB (67.5 GiB; MEASURED below, on a build seven
# builder commits older than 4b57d15a2) and the runbook saw Modal hold 15 to
# 24 GB more RSS than local at the same point of another stage, so the
# worst case seen is about 90 GiB. 112 GiB leaves about 22 GiB
# over that and is at least 1.5 times the local peak. It is not HEAVY
# because Max capped the run at about $15 (2026-09-29, BASE_COST_CAP_USD):
# non-preemptible, a 4-hour budget plus 30 minutes of runner time
# (BASE_RUNNER_OVERHEAD_SECONDS) lists at $16.36 with HEAVY's 128 GiB and
# $14.63 with 112 GiB. Four cores: the base used 6,238 CPU-s in 2,788 s of
# wall locally, 2.2 cores on average.
#
# The CPU limit equals the request, and Modal throttles CPU use above the
# limit. Without it Modal's default soft limit would be 20 cores, and Modal
# bills the higher of request and use (modal.com/docs/guide/resources, read
# 2026-09-29). The pinned tool sizes
# its primary-QRF predict pool from os.cpu_count() and fits with n_jobs=-1
# when POPULACE_FIT_PREDICT_WORKERS and POPULACE_FIT_N_JOBS are unset, as
# they were in the local run (4b57d15a2 microcosm-fit qrf.py _fit_n_jobs and
# _predict_workers). Without the limit the container could therefore be
# billed above the 4-core request: at 20 cores for the whole timeout one
# attempt would list at $25.28. The limit may lengthen phases that used more
# than 4 cores locally. The plan does not pin those two variables either.
# Pinning would not cost parity with the local run's locked config, which
# the Modal run never has (the tool locks resolved paths and the thread
# variables it sees, and Modal sets some of those itself); it would change
# only the config's thread_environment, which compare-lineage reports and,
# but for the hash seed, does not compare (RUN_CONFIG_REPORTED_ONLY,
# THREAD_ENVIRONMENT_COMPARED). What is unknown is whether a
# worker count changes the numbers: the pinned qrf.py says it does not, but
# no test at 4b57d15a2 compares two counts and no run has. So pinning is
# left to the builder or Max (runbook, "Pinning the pools").
#
# The timeout sets the ceiling at the request, and it is chosen so that
# ceiling stays inside the cap: Modal's timeout bounds a function's
# execution time (modal.com/docs/guide/timeouts, read 2026-09-29), and
# 16,500 s of this class's request lists at $14.90 non-preemptible. That is
# the 4-hour tool budget, the 30-minute runner reserve and 5 minutes for
# staging. The one way past it is memory: there is no memory limit, so use
# above 112 GiB is billed at use (about $0.024 per GiB-hour non-preemptible).
# A limit at the request would turn that into an OOM kill of the whole
# container, which would lose the attempt with no receipt and no mirrored
# state. The runner also keeps the tool inside the timeout: it stops the
# tool no later than the timeout less the reserve (container_tool_seconds),
# so time spent before the tool (pulling state, staging inputs) comes out of
# the tool's time, never out of the reserve for hashing, mirroring and the
# receipt. No ephemeral_disk request: Modal gives each container a disk
# quota of 512 GiB by default (modal.com/docs/guide/resources, read
# 2026-09-29) and the base writes about 50 GB; StageSpec.min_free_disk_gib
# checks the disk before anything is staged.
BASE = Resources(
    "base", cpu=4.0, memory_mib=112 * 1024, timeout_s=16_500, cpu_limit=4.0
)
# Route A's head-to-head: each spawned worker retains a full repaired frame.
# The independent review's 9-10 GiB/worker is unmeasured: 20 * 10 GiB plus
# 24 GiB for the parent and headroom, not a measured aggregate peak. At
# 1,030 slices * 4.7 minutes / 20 workers the tool takes about 14,523 s.
# 17,100 s leaves staging, the normal 900 s runner reserve and compute margin.
# CPU is capped at the request; memory remains request-only like the other
# classes. The non-preemptible ceiling at the request is $38.95 at timeout.
HEAD_TO_HEAD = Resources(
    "head-to-head", cpu=20.0, memory_mib=224 * 1024, timeout_s=17_100, cpu_limit=20.0
)
HEAD_TO_HEAD_COST_CAP_USD = 40.0
RESOURCE_CLASSES = {
    item.name: item for item in (CHECK, HEAVY, LIGHT, BASE, HEAD_TO_HEAD)
}
#: Max's cap for the base run (2026-09-29: non-preemptible, 4 hours, "about
#: $15"). The base class's ceiling at its request (its timeout,
#: non-preemptible, at list price) must not exceed it; a test holds this.
#: The runbook reads "4 hours" as the tool's wall, not the container's.
BASE_COST_CAP_USD = 15.0

# Time the container keeps for staging inputs, hashing and mirroring state,
# beyond the plan's max_wall_seconds. It bounds max_wall_seconds below the
# class's timeout, is added to the cost estimate at the budget, and is the
# reserve the runner keeps after the tool before the timeout
# (container_tool_seconds).
RUNNER_OVERHEAD_SECONDS = 15 * 60
# The base mirrors its checkpoints (about 44 GB) and its output (about 2.4 GB)
# after the tool exits: one pass that hashes each file while copying it to
# the runs volume (mirror_tree_hashed). The runbook measured staging inputs
# from the volume at no less than 58 MB/s (11.4 GB in under 3.5 minutes,
# container start included); no container-to-volume write rate has been
# measured, so the check measures one (write_probe_verdict) and refuses when
# the base's state could not be written inside this reserve. Thirty minutes
# is an allowance, not a measurement; the receipt records the container's
# wall and its list price.
BASE_RUNNER_OVERHEAD_SECONDS = 30 * 60
# The state the base leaves to hash and mirror, for the check's write probe:
# about 44 GB of frame checkpoints and the 2.35 GB H5 locally (route_a.sh
# section 4), plus the final checkpoint's hard-linked alias
# (_link_all_stage_checkpoint), which the mirror copies as a second file. An
# allowance from the local figures, not a measurement.
BASE_MIRRORED_STATE_GIB = 50
# How long the runner waits after SIGTERM before SIGKILL (stop_process_group).
STOP_GRACE_SECONDS = 60
# Of the runner's reserve, the part kept after the mirror for the runner's
# identity, the receipt write and the attempt's final record and commit. The
# write probe does not count it as mirror time (write_probe_verdict). An
# allowance, not a measurement.
RECEIPT_RESERVE_SECONDS = 120
# The check's container-to-volume write probe (write_probe_verdict).
WRITE_PROBE_BYTES = 1024**3


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
    # /usr/bin/time -l of the 2026-09-16 A1d base (policyengine-us 1.819.0):
    # 2,787.90 s real, 4,876.65 s user + 1,361.52 s sys, maximum resident set
    # size 72,467,169,280 bytes. Route A's driver (route_a.sh section 4)
    # quotes it. It is not a measurement of the build this registration
    # runs: A1d ran commit 51c31438 (_buildq-runtime/RECEIPT-phase2.md), and
    # seven later commits that 4b57d15a2 includes changed the builder itself
    # (git log 51c3143829..4b57d15a2 -- tools/build_us_puf_support_base.py),
    # among them the SPM independence role stage (9d26595bc) and the
    # restored Census person columns (39b8e7b63), besides the move to
    # policyengine-us 2.2.1 (6aad4e1bd). Route A's local run of 4b57d15a2
    # stopped after two outer stages, so the first Modal receipt's
    # peak_rss_bytes is the first peak measured for this build.
    ("us-puf-support-base", "all"): Measured(
        72_467_169_280,
        2787.9,
        6238.17,
        "_buildq-runtime/logs/A1d.log (A1d base, 2026-09-16, /usr/bin/time -l)",
    ),
}


# --------------------------------------------------------------------------- #
# Tool registry                                                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OptionFlag:
    flag: str
    kind: type


_LONG_FLAG = re.compile(r"--[A-Za-z0-9][A-Za-z0-9-]*")


def option_flag_problem(key: str, flag: str, owned: frozenset[str]) -> str | None:
    """Why an allowlisted option's flag could reach a runner-owned flag.

    Exact equality is not enough: the tools parse with argparse, whose
    ``allow_abbrev`` (on by default, and on in the base's pinned
    ``_parse_args``) reads a unique prefix of a long flag as that flag, and
    reads ``--flag=value`` as the flag and a value. So the flag must be a
    plain long flag, contain no ``=``, and be neither an owned flag nor a
    prefix of one. A registry error, refused by parse_plan and option_argv.
    """

    if flag in owned:
        return f"option {key!r} maps to runner-owned flag {flag}"
    if "=" in flag:
        return (
            f"option {key!r} flag {flag!r} contains '=': argparse reads "
            "--flag=value as a flag and its value, so a registration names the "
            "flag alone"
        )
    if not _LONG_FLAG.fullmatch(flag):
        return (
            f"option {key!r} flag {flag!r} is not a plain long flag "
            "(--name); a short flag could alias a runner-owned one"
        )
    abbreviated = sorted(item for item in owned if item.startswith(flag))
    if abbreviated:
        return (
            f"option {key!r} flag {flag} abbreviates runner-owned flag(s) "
            f"{', '.join(abbreviated)}: argparse's allow_abbrev reads a unique "
            "prefix as the flag it begins"
        )
    return None


@dataclass(frozen=True)
class StageSpec:
    name: str
    resources: Resources
    required_inputs: tuple[str, ...]
    # Container time beyond max_wall_seconds for staging, hashing and
    # mirroring; see RUNNER_OVERHEAD_SECONDS.
    runner_overhead_seconds: int = RUNNER_OVERHEAD_SECONDS
    # Free space the stage needs under WORK_ROOT before anything is staged;
    # a container with less refuses (cheaply) instead of failing hours in.
    min_free_disk_gib: int | None = None
    # The state the stage may leave to hash and mirror after the tool. When
    # set, the check probes the runs volume's write rate and refuses if that
    # much state could not be written inside the runner's reserve.
    mirrored_state_gib: float | None = None
    # State paths (a file, or a directory and everything under it) that the
    # push copies to the runs volume before any other file, in this order:
    # the stage's output and the evidence for it, so that a push cut short
    # by the class timeout has written those first.
    mirror_first: tuple[str, ...] = ()
    # True when every state file outside mirror_first exists only so that a
    # stopped stage can resume (the base's frame checkpoints). After a clean
    # exit (return code 0, not stopped at the budget) the push then copies
    # only mirror_first, and hashes the rest into the receipt without copying
    # it (the receipt's outputs_not_mirrored); see push_copy_only.
    rest_is_resume_state: bool = False

    def __post_init__(self) -> None:
        for path in self.mirror_first:
            if any(part in {"", ".", ".."} for part in path.split("/")):
                raise ValueError(
                    f"stage {self.name!r}: mirror_first path {path!r} must be a "
                    "clean relative path"
                )
        if self.rest_is_resume_state and not self.mirror_first:
            raise ValueError(
                f"stage {self.name!r}: rest_is_resume_state needs mirror_first"
            )


@dataclass(frozen=True)
class TreeFile:
    """A file of the pinned tree the tool reads by path, pinned by sha256.

    The commit already pins its bytes; the digest makes the registration
    refuse a plan whose commit carries a different version of the file, and
    the receipt records what was read.
    """

    path: str
    sha256: str


@dataclass(frozen=True)
class HomeSeed:
    """A staged input the runner also copies into the tool's home cache.

    For a tool that fetches a pinned file into ``~/...`` when it is not
    already there. Seeding the cache from a content-addressed input keeps the
    stage off the network; the tool still verifies the file itself.
    ``sha256`` is the digest the tool pins, and a plan whose input carries a
    different digest is refused.
    """

    input: str
    path: str
    sha256: str


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
    tree_files: Mapping[str, TreeFile] = field(default_factory=dict)
    home_seeds: tuple[HomeSeed, ...] = ()
    # The PYTHONHASHSEED the tool gives its stage interpreters when its own
    # environment has none (read from the pinned tool), for the receipt's
    # hash-seed record; None when it sets none.
    stage_hash_seed_default: str | None = None


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
        "cd_holdout_fraction": OptionFlag("--cd-holdout-fraction", float),
        "sample_fraction": OptionFlag("--sample-fraction", float),
        "sample_seed": OptionFlag("--sample-seed", int),
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

# --- The PUF-support base (Route A) ----------------------------------------
#
# tools/build_us_puf_support_base.py builds a US base H5 from raw sources:
# three pooled ASEC years, the processed PUF and the restricted TY2015 IRS
# PUF CSV, ACS 2022 rents, the ASEC Census person archives, the Ledger feed
# (SOI congressional-district return counts) and the block ladder. The
# registration reproduces Route A's command (route_a.sh section 4,
# base-config.json for 4b57d15a2) flag for flag: the scalar settings are
# fixed here, not plan options, because a different seed, year or forest
# size is a different base than the one Route A certifies, and a change to
# them should be a reviewed registry change rather than a plan edit.
#
# State layout, all under the mirrored state directory:
#   base-checkpoints/  the frame checkpoint after every outer stage (about
#                      44 GB in the 2026-09-16 run), stage_run_context.json
#                      and stage_profile.json
#   base-out/          the base H5, its summary JSON and the capital-gains
#                      tail manifest; the release reads this directory
#
# After a stop or a failure the checkpoints are mirrored, deliberately.
# `--stage all` runs each outer stage in a fresh interpreter and resumes
# from the completed prefix (_run_staged_all), with its whole run config
# (input paths and digests, settings, code identity, thread variables)
# locked in stage_run_context.json; a resume whose config differs is
# refused. Every path the tool records is the same in every attempt of a
# run (/work/inputs/..., /work/state/...), and so are the image, the plan's
# environment and the class, so a later attempt of the same run_id should
# resume where the last one stopped. The placement is non-preemptible, so
# what this protects against is mostly the budget: at
# the per-chunk slowdowns the runbook measured for materialize's engine pass
# on Modal (3.3 to 7.6 times the build machine), the 2,788-second local base
# would take 2.6 to 5.9 hours, and a stop at the 4-hour budget without the
# checkpoints would throw the whole run away. (The runner stops the tool's
# whole process group at the budget, so a stage child cannot outlive its
# parent and hold the stop past the budget; see stop_process_group.) The cost
# is one copy of about 44 GB, hashed as it is copied, when the tool stops
# (BASE_RUNNER_OVERHEAD_SECONDS); a resuming attempt pulls it back, hashed
# and verified in the same pass, and that pull comes out of its tool time
# (container_tool_seconds); and the volume storage until the checkpoints are
# deleted after the release.
#
# Every push copies PUF_SUPPORT_BASE_MIRROR_FIRST before anything else: the
# output, the logs, the run context and profile, and the two checkpoints
# compare-lineage checks against the local run. After a clean exit nothing
# else is copied: a completed `--stage all` has nothing to resume, so the
# other checkpoints (about 42 GB) are hashed into the receipt and left in the
# container (rest_is_resume_state). That push is about 4 GB, not about 50.
PUF_SUPPORT_BASE_CHECKPOINTS = "base-checkpoints"
PUF_SUPPORT_BASE_OUT = "base-out"
PUF_SUPPORT_BASE_MIRROR_FIRST = (
    PUF_SUPPORT_BASE_OUT,
    "logs",
    f"{PUF_SUPPORT_BASE_CHECKPOINTS}/stage_run_context.json",
    f"{PUF_SUPPORT_BASE_CHECKPOINTS}/stage_profile.json",
    f"{PUF_SUPPORT_BASE_CHECKPOINTS}/000_source_construction.frame.h5",
    f"{PUF_SUPPORT_BASE_CHECKPOINTS}/001_pre_clone_enrichment.frame.h5",
)
PUF_SUPPORT_BASE_ARTIFACT = "base_populace_us_2024_puf_support.h5"
PUF_SUPPORT_BASE_TARGET_YEAR = 2024
# The pooled ASEC order is Route A's: the tool hands the --asec-h5 values to
# the pooling in the order given and records them in that order in the run
# context it locks, so the order is part of the build.
PUF_SUPPORT_BASE_ASEC_YEARS = (2024, 2023, 2022)
# Income years of the ASEC Census person archives (income year YYYY is the
# survey-year YYYY+1 archive, asecpubYY+1csv.zip).
PUF_SUPPORT_BASE_EDUCATION_YEARS = (2022, 2023, 2024)
PUF_SUPPORT_BASE_SEED = 0
PUF_SUPPORT_BASE_N_ESTIMATORS = 32
PUF_SUPPORT_BASE_CD_SEED = 0
# In-tree at 4b57d15a2; main has since changed it (a347303f... on 2026-09-29).
CD_VINTAGE_CROSSWALK = TreeFile(
    path=(
        "packages/microcosm-build/src/microcosm/build/us_runtime/data/"
        "congressional_district_vintage_crosswalk.csv"
    ),
    sha256="c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec",
)
# The 2023 ASEC archive is also the base's weeks-unemployed source. Route A
# does not pass --asec-2023-weeks-unemployed-source, so the tool looks in
# ~/.cache/microcosm/cps/asec_2023/ and downloads it from www2.census.gov
# when it is missing (fetch_asec_2023_weeks_unemployed_source). The build
# machine had it cached; the runner seeds the same cache from the staged
# asec_education_2022_zip input, which is the same file (the tool pins
# both to d2e00025...), so the stage makes no Census request.
ASEC_2023_ARCHIVE_SEED = HomeSeed(
    input="asec_education_2022_zip",
    path=".cache/microcosm/cps/asec_2023/asecpub23csv.zip",
    sha256="d2e000250782adfbdd7f29c82b66d866591a30f0d330496698ec19f9c784ce11",
)
PUF_SUPPORT_BASE_INPUTS = (
    *(f"asec_{year}_h5" for year in PUF_SUPPORT_BASE_ASEC_YEARS),
    "puf_2024_h5",
    "puf_2015_csv",
    "acs_2022_h5",
    *(f"asec_education_{year}_zip" for year in PUF_SUPPORT_BASE_EDUCATION_YEARS),
    "base_ledger_facts",
    "block_ladder_npz",
)


def _puf_support_base_argv(
    plan: Plan, input_paths: Mapping[str, str], state_dir: str
) -> list[str]:
    state = PurePosixPath(state_dir)
    argv = [
        "--stage",
        plan.stage,
        "--checkpoint-dir",
        str(state / PUF_SUPPORT_BASE_CHECKPOINTS),
    ]
    for year in PUF_SUPPORT_BASE_ASEC_YEARS:
        argv += ["--asec-h5", f"{year}={input_paths[f'asec_{year}_h5']}"]
    for year in PUF_SUPPORT_BASE_ASEC_YEARS:
        argv += ["--asec-h5-sha256", f"{year}={plan.inputs[f'asec_{year}_h5'].sha256}"]
    argv += [
        "--puf-h5",
        input_paths["puf_2024_h5"],
        "--puf-source-year-csv",
        input_paths["puf_2015_csv"],
        "--acs-h5",
        input_paths["acs_2022_h5"],
    ]
    for year in PUF_SUPPORT_BASE_EDUCATION_YEARS:
        argv += [
            "--asec-education-source",
            f"{year}={input_paths[f'asec_education_{year}_zip']}",
        ]
    argv += [
        "--target-year",
        str(PUF_SUPPORT_BASE_TARGET_YEAR),
        "--seed",
        str(PUF_SUPPORT_BASE_SEED),
        "--n-estimators",
        str(PUF_SUPPORT_BASE_N_ESTIMATORS),
        "--ledger-facts",
        input_paths["base_ledger_facts"],
        "--assign-congressional-districts",
        "--congressional-district-vintage-crosswalk",
        str(PurePosixPath(IMAGE_REPO_ROOT) / CD_VINTAGE_CROSSWALK.path),
        "--congressional-district-seed",
        str(PUF_SUPPORT_BASE_CD_SEED),
        "--block-ladder-artifact",
        input_paths["block_ladder_npz"],
        "--out",
        str(state / PUF_SUPPORT_BASE_OUT),
    ]
    return argv


#: Every flag _puf_support_base_argv emits.
PUF_SUPPORT_BASE_BUILDER_FLAGS = frozenset(
    {
        "--stage",
        "--checkpoint-dir",
        "--asec-h5",
        "--asec-h5-sha256",
        "--puf-h5",
        "--puf-source-year-csv",
        "--acs-h5",
        "--asec-education-source",
        "--target-year",
        "--seed",
        "--n-estimators",
        "--ledger-facts",
        "--assign-congressional-districts",
        "--congressional-district-vintage-crosswalk",
        "--congressional-district-seed",
        "--block-ladder-artifact",
        "--out",
    }
)
#: The tool's other flags (4b57d15a2 _parse_args), which no plan may pass:
#: an alternative source (--base-h5, --support-spine-spec, the weeks-
#: unemployed path, which the home seed covers), a smoke limit, the
#: equivalence-test harness, and the geography-ladder escape hatches.
PUF_SUPPORT_BASE_WITHHELD_FLAGS = frozenset(
    {
        "--base-h5",
        "--support-spine-spec",
        "--asec-2023-weeks-unemployed-source",
        "--asec-max-households",
        "--equivalence-boundary-dir",
        "--equivalence-deterministic-h5-metadata",
        "--without-block-ladder",
        "--geography-ladder-seed",
        "--allow-geography-ladder-gate-failures",
    }
)

US_PUF_SUPPORT_BASE = ToolSpec(
    name="us-puf-support-base",
    script="tools/build_us_puf_support_base.py",
    inputs=PUF_SUPPORT_BASE_INPUTS,
    stages={
        "all": StageSpec(
            "all",
            BASE,
            PUF_SUPPORT_BASE_INPUTS,
            runner_overhead_seconds=BASE_RUNNER_OVERHEAD_SECONDS,
            # The build machine's admission floor for the same command
            # (base-config.json limits.disk_admission_bytes, 69 GiB): 44 GB
            # of checkpoints, the 2.35 GB H5, about 2.4 GB of inputs and
            # the 2023 archive with its extracted member, plus headroom.
            min_free_disk_gib=70,
            mirrored_state_gib=BASE_MIRRORED_STATE_GIB,
            mirror_first=PUF_SUPPORT_BASE_MIRROR_FIRST,
            rest_is_resume_state=True,
        )
    },
    options={},
    owned_flags=PUF_SUPPORT_BASE_BUILDER_FLAGS | PUF_SUPPORT_BASE_WITHHELD_FLAGS,
    argv_builder=_puf_support_base_argv,
    parse_function="_parse_args",
    tree_files={"cd_vintage_crosswalk": CD_VINTAGE_CROSSWALK},
    home_seeds=(ASEC_2023_ARCHIVE_SEED,),
    # 4b57d15a2 _staged_subprocess_environment (see HASH_SEED_ENV).
    stage_hash_seed_default=HASH_SEED,
)


def _release_head_to_head_argv(
    plan: Plan, input_paths: Mapping[str, str], state_dir: str
) -> list[str]:
    return [
        "--incumbent",
        input_paths["incumbent_h5"],
        "--candidate",
        input_paths["candidate_h5"],
        "--ledger-facts",
        input_paths["ledger_facts"],
        "--out-prefix",
        str(PurePosixPath(state_dir) / "h2h"),
    ]


US_RELEASE_HEAD_TO_HEAD = ToolSpec(
    name="us-release-head-to-head",
    script="tools/score_us_release_head_to_head.py",
    inputs=("incumbent_h5", "candidate_h5", "ledger_facts"),
    stages={
        "score": StageSpec(
            "score",
            HEAD_TO_HEAD,
            ("incumbent_h5", "candidate_h5", "ledger_facts"),
            mirror_first=("h2h.json", "h2h.md"),
        )
    },
    options={
        "workers": OptionFlag("--workers", int),
        "maximum_microsim_batch_size": OptionFlag("--maximum-microsim-batch-size", int),
        # Off unless a plan sets it; the scorer refuses it on interpreters
        # where replacing a recycled worker can deadlock.
        "worker_max_slices": OptionFlag("--worker-max-slices", int),
    },
    owned_flags=frozenset(
        {
            "--incumbent",
            "--candidate",
            "--ledger-facts",
            "--out-prefix",
            "--candidate-manifest-sha256",
            "--candidate-worker-identity-attestation",
            "--congressional-district-vintage-crosswalk",
            # Omit both BooleanOptionalActions to preserve the local run's
            # defaults (age_targets=False, allow_unaged_dollar_targets=True).
            # Reserve both spellings so plans cannot change that yardstick.
            "--age-targets",
            "--no-age-targets",
            "--allow-unaged-dollar-targets",
            "--no-allow-unaged-dollar-targets",
            # Only the canonical batch-size spelling is a plan option.
            "--maximum-microsimulation-batch-size",
        }
    ),
    argv_builder=_release_head_to_head_argv,
    parse_function="_parse_args",
)

TOOLS: dict[str, ToolSpec] = {
    tool.name: tool
    for tool in (
        US_ACS_LOCAL_RELEASE,
        RUNNER_SMOKE,
        US_PUF_SUPPORT_BASE,
        US_RELEASE_HEAD_TO_HEAD,
    )
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
    token). The revision may be a tag, a commit or a branch name; the bytes
    are pinned by the sha256 either way, which is verified after staging, so
    a branch that has moved fails verification instead of running.
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
    for seed in tool.home_seeds:
        ref = inputs.get(seed.input)
        if ref is not None and ref.sha256 != seed.sha256:
            raise PlanError(
                f"input {seed.input!r} is sha256 {ref.sha256}, but the runner seeds "
                f"~/{seed.path} with it and the tool pins that file to {seed.sha256}"
            )

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
        # A registry error, refused early.
        flag_problem = option_flag_problem(key, option.flag, tool.owned_flags)
        if flag_problem:
            raise PlanError(flag_problem)
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
                f"env {key!r} is not allowlisted (MICROCOSM_*, POPULACE_*, or one "
                f"of {', '.join(THREAD_ENV_KEYS)})"
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
    stage_spec = tool.stages[stage]
    ceiling = stage_spec.resources.timeout_s - stage_spec.runner_overhead_seconds
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
        # Registry invariant, held here too for a Plan built without parse_plan.
        flag_problem = option_flag_problem(key, option.flag, plan.tool.owned_flags)
        if flag_problem:
            raise PlanError(flag_problem)
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


def planned_argv(plan: Plan, work_root: str | None = None) -> list[str]:
    # Read WORK_ROOT when called, so the argv and the runner's own paths
    # (which read it the same way) always agree.
    work_root = WORK_ROOT if work_root is None else work_root
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
    (``HF_TOKEN`` from an attached Hub secret, Modal's own tokens) and
    without its ``PYTHONHASHSEED`` (HASH_SEED_ENV: with none, the base's
    pinned tool gives its stages "0", the value Route A's local base
    recorded), with ``HF_HUB_OFFLINE=1`` so the tool cannot reach the Hub, and
    with the plan's allowlisted overrides last. Only names are returned, for
    the receipt; never values.
    """

    for key in plan_env:
        if not _ENV_KEY.fullmatch(key) or is_credential_env_key(key):
            raise PlanError(f"env {key!r} may not be passed to the tool")
    removed = sorted(
        key for key in base if is_credential_env_key(key) or key == HASH_SEED_ENV
    )
    dropped = set(removed)
    env = {key: value for key, value in base.items() if key not in dropped}
    env["HF_HUB_OFFLINE"] = "1"
    env.update(plan_env)
    return env, removed


def hash_seed_problem(env: Mapping[str, str]) -> str | None:
    """A refusal when the tool's environment would force another hash seed.

    The tool must run with ``PYTHONHASHSEED`` unset (the base's pinned tool
    then gives its stage interpreters HASH_SEED) or set to HASH_SEED. Any
    other value would reach every stage (the pinned tool's default does not
    override it) and differ from Route A's local run.
    """

    value = env.get(HASH_SEED_ENV)
    if value is None or value == HASH_SEED:
        return None
    return (
        f"the tool's environment sets {HASH_SEED_ENV}={value!r}; the tool runs "
        f"with it unset or {HASH_SEED!r} (Route A's local base recorded "
        f"{HASH_SEED!r})"
    )


def hash_seed_record(
    tool: ToolSpec, base: Mapping[str, str], env: Mapping[str, str]
) -> dict[str, str | None]:
    """The receipt's ``PYTHONHASHSEED`` record.

    The container's value (which the tool never gets), what the tool's
    environment carries (None: unset), and the value the tool's stage
    interpreters run with: the passed value, else the tool's own default
    (``ToolSpec.stage_hash_seed_default``; None for a tool without one,
    whose interpreters then pick a random seed each).
    """

    passed = env.get(HASH_SEED_ENV)
    stages = tool.stage_hash_seed_default if passed is None else passed
    return {
        "container": base.get(HASH_SEED_ENV),
        "passed_to_tool": passed,
        "stages_run_with": stages,
    }


def stop_process_group(
    proc: subprocess.Popen, *, grace_seconds: float = STOP_GRACE_SECONDS
) -> dict[str, bool]:
    """Stop a tool started with ``process_group=0`` and everything it spawned.

    ``proc.terminate()`` signals only the tool's own process. A tool that
    runs its stages as child interpreters (the base's ``--stage all``) would
    lose its parent while the running stage child carried on with the log
    pipe open, so the runner, reading that pipe to its end, would wait out
    the child's whole stage past the budget. SIGTERM goes to the whole
    group; whatever is left after ``grace_seconds`` gets SIGKILL. Returns
    which signals were sent.
    """

    sent = {"sigterm": False, "sigkill": False}
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        sent["sigterm"] = True
    except ProcessLookupError:
        return sent
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # The group outlives its leader while any member runs.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
        sent["sigkill"] = True
    except ProcessLookupError:
        pass
    return sent


def tree_file_rows(tool: ToolSpec, repo_root: Path | str) -> list[dict[str, object]]:
    """Hash each of the tool's pinned tree files in the clone at ``repo_root``."""

    rows: list[dict[str, object]] = []
    for name, tree_file in sorted(tool.tree_files.items()):
        path = Path(repo_root) / tree_file.path
        row: dict[str, object] = {
            "name": name,
            "path": tree_file.path,
            "expected": tree_file.sha256,
        }
        if not path.is_file():
            row["problem"] = "not in the pinned tree"
        else:
            row["sha256"], row["bytes"] = sha256_file(path)
            if row["sha256"] != tree_file.sha256:
                row["problem"] = (
                    "sha256 differs from the registration; the plan's commit "
                    "carries another version of this file"
                )
        rows.append(row)
    return rows


def home_seed_targets(plan: Plan, home: Path | str) -> list[tuple[HomeSeed, Path]]:
    """Where each of the plan's home seeds goes under ``home``."""

    targets = []
    for seed in plan.tool.home_seeds:
        if seed.input not in plan.inputs:
            continue
        # Like _safe_relative_path, but a cache directory may be hidden.
        parts = seed.path.split("/")
        if (
            PurePosixPath(seed.path).is_absolute()
            or not parts
            or any(part in {"", ".", ".."} for part in parts)
            or not all(_HOME_PART.fullmatch(part) for part in parts)
        ):
            raise PlanError(
                f"home seed for {seed.input!r}: {seed.path!r} must be a clean "
                "relative path"
            )
        targets.append((seed, Path(home).joinpath(*parts)))
    return targets


def seed_home_cache(
    plan: Plan, input_paths: Mapping[str, str], home: Path | str
) -> list[dict[str, object]]:
    """Copy each seeded input into the tool's home cache; verify the copy."""

    seeded = []
    for seed, target in home_seed_targets(plan, home):
        sha, size = copy_with_sha256(input_paths[seed.input], target)
        try:
            verify_digest(plan.inputs[seed.input], sha)
        except PlanError:
            target.unlink(missing_ok=True)
            raise
        seeded.append(
            {"input": seed.input, "path": str(target), "sha256": sha, "bytes": size}
        )
    return seeded


def work_disk_problem(stage: StageSpec, free_bytes: int) -> str | None:
    """A refusal when the container's work disk is smaller than the stage needs."""

    if stage.min_free_disk_gib is None:
        return None
    needed = stage.min_free_disk_gib * 1024**3
    if free_bytes >= needed:
        return None
    return (
        f"{WORK_ROOT} has {free_bytes / 1024**3:.1f} GiB free; stage "
        f"{stage.name!r} needs {stage.min_free_disk_gib} GiB"
    )


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


# A file mid-copy in mirror_tree; never part of a state tree.
MIRROR_PARTIAL_SUFFIX = ".mirror-partial"


def tree_listing(root: Path | str) -> dict[str, tuple[int, int]]:
    """``{relative posix path: (bytes, mtime_ns)}`` for every regular file.

    A ``MIRROR_PARTIAL_SUFFIX`` file (a copy a preemption cut short) is left
    out; mirror_tree removes it on the next push.
    """

    root = Path(root)
    listing: dict[str, tuple[int, int]] = {}
    if not root.exists():
        return listing
    for path in sorted(root.rglob("*")):
        if path.name.endswith(MIRROR_PARTIAL_SUFFIX):
            continue
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


def copy_hashed(src: Path | str, dst: Path | str) -> tuple[str, int]:
    """Copy ``src`` to ``dst`` like ``shutil.copy2``, hashing in the same read.

    The copy keeps ``src``'s mtime (``shutil.copystat``), which is what lets
    mirror_actions skip a file that has not changed since it was mirrored.
    """

    digest = hashlib.sha256()
    size = 0
    with open(src, "rb") as reader, open(dst, "wb") as writer:
        while chunk := reader.read(_CHUNK):
            digest.update(chunk)
            writer.write(chunk)
            size += len(chunk)
    shutil.copystat(src, dst)
    return digest.hexdigest(), size


#: ``{relative path: (bytes, mtime_ns, sha256)}`` for files already hashed.
KnownHashes = Mapping[str, tuple[int, int, str]]


def under_state_path(rel: str, path: str) -> bool:
    """Whether state file ``rel`` is ``path`` or lies in directory ``path``."""

    path = path.rstrip("/")
    return rel == path or rel.startswith(path + "/")


def push_order(paths: Iterable[str], first: Sequence[str] = ()) -> list[str]:
    """``paths`` in the order the push copies them.

    The files under ``first[0]``, then those under ``first[1]`` and so on,
    each group sorted, then every other file, sorted. Each file appears once.
    """

    remaining = sorted(paths)
    ordered: list[str] = []
    for prefix in first:
        group = [rel for rel in remaining if under_state_path(rel, prefix)]
        ordered += group
        taken = set(group)
        remaining = [rel for rel in remaining if rel not in taken]
    return ordered + remaining


def _mirror(
    source: Path,
    destination: Path,
    *,
    hash_every_file: bool,
    known: KnownHashes | None = None,
    first: Sequence[str] = (),
    copy_only: Sequence[str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, int], list[str]]:
    partials = (
        sorted(destination.rglob(f"*{MIRROR_PARTIAL_SUFFIX}"))
        if destination.exists()
        else []
    )
    for stale in partials:
        stale.unlink()
    listing = tree_listing(source)
    copy, delete = mirror_actions(listing, tree_listing(destination))
    to_copy = set(copy)
    known = known or {}
    hashed: dict[str, dict[str, object]] = {}
    not_mirrored: list[str] = []
    counts = {
        "copied": 0,
        "hashed_in_place": 0,
        "hashes_reused": 0,
        "hashed_not_copied": 0,
        "stale_removed": 0,
    }
    for rel in push_order(listing, first):
        meta = listing[rel]
        entry = known.get(rel)
        if rel in to_copy and (
            copy_only is None or any(under_state_path(rel, p) for p in copy_only)
        ):
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}{MIRROR_PARTIAL_SUFFIX}")
            sha, size = copy_hashed(source / rel, tmp)
            os.replace(tmp, target)
            counts["copied"] += 1
        elif rel in to_copy:
            # Not copied (copy_only): hashed for the receipt. An older copy
            # on the destination would not match the receipt, so it goes.
            stale = destination / rel
            if stale.is_file() or stale.is_symlink():
                stale.unlink()
                counts["stale_removed"] += 1
            if entry is not None and (entry[0], entry[1]) == meta:
                sha, size = entry[2], meta[0]
            else:
                sha, size = sha256_file(source / rel)
            counts["hashed_not_copied"] += 1
            not_mirrored.append(rel)
        elif not hash_every_file:
            continue
        elif entry is not None and (entry[0], entry[1]) == meta:
            # Hashed earlier (pulled and verified) and not touched since.
            sha, size = entry[2], meta[0]
            counts["hashes_reused"] += 1
        else:
            sha, size = sha256_file(source / rel)
            counts["hashed_in_place"] += 1
        hashed[rel] = {"path": rel, "bytes": size, "sha256": sha}
    for rel in delete:
        (destination / rel).unlink()
    counts.update(deleted=len(delete), partials_removed=len(partials))
    # The listing's order, whatever order the files were copied in.
    outputs = [hashed[rel] for rel in listing if rel in hashed]
    return outputs, counts, sorted(not_mirrored)


def mirror_tree(source: Path | str, destination: Path | str) -> dict[str, int]:
    """Make ``destination`` mirror ``source``, one file at a time, atomically.

    Each file is copied to a temporary name in its own directory and renamed
    over the target, so no file on the destination is ever half-written. A
    preemption between files still leaves a mix of new and old files (and
    files not yet deleted); the next attempt's pulled-state check against
    the latest receipt refuses that mix.
    """

    _, counts, _ = _mirror(Path(source), Path(destination), hash_every_file=False)
    return {
        "copied": counts["copied"],
        "deleted": counts["deleted"],
        "partials_removed": counts["partials_removed"],
    }


def mirror_tree_hashed(
    source: Path | str,
    destination: Path | str,
    *,
    known: KnownHashes | None = None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """``mirror_tree`` and ``hash_tree(source)`` in one read of each file.

    Returns the source tree hashed exactly as :func:`hash_tree` lists it,
    and the mirror's counts. A copied file is hashed as it is copied, with
    the same atomic rename as ``mirror_tree``. A file the destination already
    holds (same size and mtime) is not copied: its sha256 comes from
    ``known`` when ``known`` has it at the same size and mtime (a file
    pulled and verified earlier and left untouched since), and is otherwise
    read once to hash it.
    """

    outputs, counts, _ = _mirror(
        Path(source), Path(destination), hash_every_file=True, known=known
    )
    for key in ("hashed_not_copied", "stale_removed"):
        del counts[key]  # always 0 without copy_only
    return outputs, counts


def push_copy_only(
    stage: StageSpec, *, returncode: int, stopped_at_budget: bool
) -> tuple[str, ...] | None:
    """The only state paths a push copies, or None to copy the whole state.

    A stage that declares ``rest_is_resume_state`` copies only its
    ``mirror_first`` paths after a clean exit (return code 0, not stopped at
    the budget): the rest of its state exists to resume a stopped stage, and
    a finished stage has nothing to resume. After a stop or a failure the
    whole state is copied, so the next attempt can resume.
    """

    if stage.rest_is_resume_state and returncode == 0 and not stopped_at_budget:
        return stage.mirror_first
    return None


def push_state(
    source: Path | str,
    destination: Path | str,
    *,
    known: KnownHashes | None = None,
    first: Sequence[str] = (),
    copy_only: Sequence[str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, int], list[str]]:
    """The runner's push: :func:`mirror_tree_hashed` in a set order.

    Files under ``first`` are copied before any other (:func:`push_order`),
    so a push that the class timeout cuts short has written them. With
    ``copy_only`` (:func:`push_copy_only`), a changed file outside those
    paths is hashed but not copied, and an older copy of it on the
    destination is removed, since it would not match the receipt. Returns
    the source tree hashed exactly as :func:`hash_tree` lists it (whatever
    was copied), the counts, and the sorted paths hashed but not copied (the
    receipt's ``outputs_not_mirrored``).
    """

    return _mirror(
        Path(source),
        Path(destination),
        hash_every_file=True,
        known=known,
        first=first,
        copy_only=copy_only,
    )


def known_hashes(
    root: Path | str, outputs: Iterable[Mapping[str, object]]
) -> dict[str, tuple[int, int, str]]:
    """``outputs`` (a hashed listing of ``root``) keyed with each file's
    current size and mtime, for :func:`mirror_tree_hashed`'s ``known``."""

    listing = tree_listing(root)
    known: dict[str, tuple[int, int, str]] = {}
    for item in outputs:
        rel = str(item["path"])
        meta = listing.get(rel)
        if meta is not None and meta[0] == item["bytes"]:
            known[rel] = (meta[0], meta[1], str(item["sha256"]))
    return known


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
    attempt_id: str | None = None,
    prior_state_verified_against: str | None = None,
    outputs_not_mirrored: Sequence[str] = (),
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
            "cpu_limit": resources.cpu_limit,
            "memory_mib": resources.memory_mib,
            "timeout_s": resources.timeout_s,
            "nonpreemptible": plan.nonpreemptible,
        },
        "argv": list(argv),
        "returncode": returncode,
        "attempt_id": attempt_id,
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
        "prior_state_verified_against": prior_state_verified_against,
        "outputs": [dict(item) for item in outputs],
        # Outputs hashed in the container but not copied to the runs volume
        # (push_copy_only): the receipt lists their digests, the volume does
        # not hold them.
        "outputs_not_mirrored": sorted(outputs_not_mirrored),
    }


def _under_prefix(rel: str, prefix: str | None) -> bool:
    return prefix is None or rel.startswith(prefix.rstrip("/") + "/")


def _missing_problem(receipt: Mapping, rel: str) -> str:
    """The problem for an output the receipt lists that is not there."""

    if rel in set(receipt.get("outputs_not_mirrored") or ()):
        return (
            f"missing: {rel} (outputs_not_mirrored: hashed in the container, "
            "never copied to the runs volume)"
        )
    return f"missing: {rel}"


def receipt_status_problems(receipt: Mapping) -> list[str]:
    """Why a receipt does not describe a finished stage, if it does not.

    A FAILED receipt lists exactly what its state held when the stage
    stopped, which can include a partly written output: at 4b57d15a2 the
    base writes its H5 at its final path and then reopens it to add
    attributes (``_export_staged_result``), so a stop at the budget during
    ``final_export`` leaves a partial H5 that the receipt lists faithfully.
    Its bytes verify, but it is not a stage's output.
    """

    problems = []
    if receipt.get("status") != "COMPLETED":
        problems.append(
            f"the receipt's status is {receipt.get('status')!r}, not 'COMPLETED' "
            f"(returncode {receipt.get('returncode')})"
        )
    if receipt.get("stopped_at_budget"):
        problems.append("the stage was stopped at its budget (stopped_at_budget)")
    return problems


def verify_receipt(
    receipt: Mapping,
    state_root: Path | str,
    *,
    strict: bool = False,
    prefix: str | None = None,
    require_completed: bool = False,
) -> list[str]:
    """Re-hash a fetched state tree against a receipt; return the problems.

    ``prefix`` limits the check to the receipt's outputs under that
    directory of the state (``base-out`` for the base's release inputs, so
    its 44 GB of checkpoints need not be fetched); ``strict`` then also
    limits its extra-file check to that directory. ``state_root`` is still
    the state directory, not the prefix. ``require_completed`` also refuses
    a receipt of a stage that did not finish (:func:`receipt_status_problems`);
    the runner's own pulled-state check leaves it off, because a stage
    resumes from the state of a FAILED receipt.
    """

    if receipt.get("schema") != RECEIPT_SCHEMA:
        return [f"not a {RECEIPT_SCHEMA} receipt"]
    if prefix is not None:
        _safe_relative_path(prefix.rstrip("/"), "prefix")
    root = Path(state_root)
    problems: list[str] = receipt_status_problems(receipt) if require_completed else []
    declared = set()
    selected = [
        item
        for item in receipt.get("outputs", [])
        if _under_prefix(str(item["path"]), prefix)
    ]
    if prefix is not None and not selected:
        problems.append(f"the receipt lists no outputs under {prefix}/")
    for item in selected:
        rel = str(item["path"])
        declared.add(rel)
        path = root / rel
        if not path.is_file():
            problems.append(_missing_problem(receipt, rel))
            continue
        sha, size = sha256_file(path)
        if size != item["bytes"]:
            problems.append(
                f"size mismatch: {rel} is {size} bytes, receipt {item['bytes']}"
            )
        elif sha != item["sha256"]:
            problems.append(f"sha256 mismatch: {rel}")
    if strict:
        present = {rel for rel in tree_listing(root) if _under_prefix(rel, prefix)}
        problems += [f"not in receipt: {rel}" for rel in sorted(present - declared)]
    return problems


def latest_receipt(
    receipts: Iterable[tuple[str, Mapping]], run_id: str
) -> tuple[str, Mapping] | None:
    """The run's most recent receipt, by ``finished_at`` (then file name).

    Every receipt is written after its state was mirrored, and one stage of
    a run runs at a time, so this is the receipt that describes the state
    on the runs volume. Receipt names begin with the stage, so sorting them
    by name would not do.
    """

    mine = [
        (str(receipt.get("finished_at") or ""), name, receipt)
        for name, receipt in receipts
        if receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("run_id") == run_id
    ]
    if not mine:
        return None
    _, name, receipt = max(mine, key=lambda item: (item[0], item[1]))
    return name, receipt


def pulled_state_problems(
    state_root: Path | str, latest: tuple[str, Mapping] | None
) -> list[str]:
    """Refusals for a run's state that its latest receipt does not describe.

    A preemption or error after the state was mirrored and before the
    receipt was written, or during a mirror, leaves state no receipt lists.
    With no receipt the state must be empty.
    """

    if latest is None:
        files = tree_listing(state_root)
        if not files:
            return []
        return [
            f"the run's state has {len(files)} file(s) and the run has no receipt "
            f"(e.g. {sorted(files)[:3]})"
        ]
    name, receipt = latest
    return [
        f"state differs from receipt {name}: {problem}"
        for problem in verify_receipt(receipt, state_root, strict=True)
    ]


def receipt_outputs_problems(
    receipt: Mapping, outputs: Sequence[Mapping[str, object]]
) -> list[str]:
    """``verify_receipt(strict=True)`` against a tree already hashed.

    ``outputs`` is a whole tree as :func:`hash_tree` or
    :func:`mirror_tree_hashed` lists it, so nothing is read again. The
    problems and their order are verify_receipt's.
    """

    if receipt.get("schema") != RECEIPT_SCHEMA:
        return [f"not a {RECEIPT_SCHEMA} receipt"]
    observed = {str(item["path"]): item for item in outputs}
    problems: list[str] = []
    declared = set()
    for item in receipt.get("outputs", []):
        rel = str(item["path"])
        declared.add(rel)
        seen = observed.get(rel)
        if seen is None:
            problems.append(_missing_problem(receipt, rel))
        elif seen["bytes"] != item["bytes"]:
            problems.append(
                f"size mismatch: {rel} is {seen['bytes']} bytes, receipt {item['bytes']}"
            )
        elif seen["sha256"] != item["sha256"]:
            problems.append(f"sha256 mismatch: {rel}")
    problems += [f"not in receipt: {rel}" for rel in sorted(set(observed) - declared)]
    return problems


def pulled_outputs_problems(
    outputs: Sequence[Mapping[str, object]], latest: tuple[str, Mapping] | None
) -> list[str]:
    """:func:`pulled_state_problems` for a state tree hashed as it was pulled.

    The runner pulls the run's state with :func:`mirror_tree_hashed`, so the
    pulled bytes are verified against the latest receipt without a second
    read of the (for the base, about 46 GB) tree.
    """

    if latest is None:
        if not outputs:
            return []
        paths = sorted(str(item["path"]) for item in outputs)
        return [
            f"the run's state has {len(paths)} file(s) and the run has no receipt "
            f"(e.g. {paths[:3]})"
        ]
    name, receipt = latest
    return [
        f"state differs from receipt {name}: {problem}"
        for problem in receipt_outputs_problems(receipt, outputs)
    ]


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
# How often a running attempt rewrites its record. The record runs from the
# start of ``_run_stage`` until the attempt's final record, so the lock wait,
# staging, the tool, hashing and mirroring are all inside it. A preempted
# attempt is charged from its start to its last record, which undercounts
# what Modal bills for it by: the container's cold start and image load
# before ``_run_stage`` begins; up to one interval after the last record
# (longer if a heartbeat write failed; failures are logged, not retried);
# and the preemption grace period.
ATTEMPT_HEARTBEAT_SECONDS = 120
# An unfinished record newer than this may belong to a running attempt.
ATTEMPT_LIVE_WINDOW_SECONDS = 2 * ATTEMPT_HEARTBEAT_SECONDS + 60
# How long a new attempt waits to see whether such a record moves: two
# heartbeats and slack, so a single failed write does not read as a death.
ATTEMPT_RECHECK_SECONDS = 2 * ATTEMPT_HEARTBEAT_SECONDS + 30
# An attempt left with less tool time than this refuses to start.
MIN_ATTEMPT_SECONDS = 60

# How an attempt ended. A running (or preempted) attempt has no outcome.
OUTCOME_RECEIPT = "receipt"  # the tool ran and a receipt was written
OUTCOME_REFUSED = "refused"  # the lock or the budget stopped it before staging
OUTCOME_ERROR = "error"  # an exception after its first record


def attempt_record(
    plan: Plan,
    plan_sha256: str,
    *,
    attempt_id: str,
    started_epoch: float,
    last_seen_epoch: float,
    finished: bool = False,
    receipt: str | None = None,
    outcome: str | None = None,
    note: str | None = None,
    modal: Mapping[str, object] | None = None,
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
        "outcome": outcome,
        "note": note,
        "modal": dict(modal or {}),
    }


def unfinished_attempts(
    records: Iterable[Mapping],
    plan: Plan,
    plan_sha256: str,
    *,
    exclude: str | None = None,
) -> list[dict[str, object]]:
    """Attempts of this exact plan and stage that are charged to its budget.

    Modal restarts a preempted function on the same input, from scratch and
    regardless of ``retries``; each such attempt is billed. Every attempt
    that is not ``finished`` is charged: one preemption cut short, and one
    that raised after its first record (a digest mismatch, a failed pull or
    mirror, a runner bug). A finished attempt is not: one that wrote a
    receipt, COMPLETED or FAILED (including a stop at the budget, after
    which the same plan starts again with its whole budget), and one the
    lock or the budget refused. ``exclude`` is the caller's own attempt.
    """

    return [
        dict(record)
        for record in records
        if record.get("schema") == ATTEMPT_SCHEMA
        and record.get("run_id") == plan.run_id
        and record.get("stage") == plan.stage
        and record.get("plan_sha256") == plan_sha256
        and record.get("attempt_id") != exclude
        and not record.get("finished")
    ]


def recent_unfinished_attempts(
    records: Iterable[Mapping],
    run_id: str,
    *,
    now: float,
    exclude: str | None = None,
) -> list[dict[str, object]]:
    """Attempts of the run, any stage or plan, that may still be running.

    Unfinished, without an outcome, and with a record newer than
    ATTEMPT_LIVE_WINDOW_SECONDS. A preempted attempt looks like this until
    its record ages out, so this alone cannot tell the two apart.
    """

    return [
        dict(record)
        for record in records
        if record.get("schema") == ATTEMPT_SCHEMA
        and record.get("run_id") == run_id
        and record.get("attempt_id") != exclude
        and not record.get("finished")
        and record.get("outcome") is None
        and now - float(record.get("last_seen_epoch") or 0.0)
        < ATTEMPT_LIVE_WINDOW_SECONDS
    ]


def live_attempts(
    read_records: Callable[[], Iterable[Mapping]],
    run_id: str,
    *,
    own_attempt_id: str,
    now: Callable[[], float],
    sleep: Callable[[float], None],
    log: Callable[[str], None] = lambda _message: None,
) -> list[dict[str, object]]:
    """The run's lock: earlier attempts that are still running.

    Two attempts of one run would race on its state directory. An earlier
    attempt (smaller ``attempt_id``, a start timestamp) whose record is
    recent is either running or was preempted moments ago, as when Modal
    restarts this very input. Wait ATTEMPT_RECHECK_SECONDS and read again: a
    running attempt has rewritten its record, a dead one has not. Returns the
    running ones; the caller refuses to start when there are any. A later
    attempt is left alone: it sees this one and refuses itself. Modal
    volumes have no atomic lock, so two attempts that start within one
    commit of each other can both pass; the runbook asks for one stage of a
    run at a time.
    """

    before = [
        record
        for record in recent_unfinished_attempts(
            read_records(), run_id, now=now(), exclude=own_attempt_id
        )
        if str(record.get("attempt_id") or "") < own_attempt_id
    ]
    if not before:
        return []
    names = ", ".join(str(record.get("attempt_id")) for record in before)
    log(
        f"LOCK: earlier attempt(s) {names} of run {run_id!r} wrote a record "
        f"recently; waiting {ATTEMPT_RECHECK_SECONDS}s to see whether they run"
    )
    sleep(ATTEMPT_RECHECK_SECONDS)
    after = {
        record.get("attempt_id"): record
        for record in read_records()
        if record.get("schema") == ATTEMPT_SCHEMA
    }
    running = []
    for record in before:
        latest = after.get(record.get("attempt_id"))
        if (
            latest is not None
            and not latest.get("finished")
            and latest.get("outcome") is None
            and float(latest.get("last_seen_epoch") or 0.0)
            > float(record.get("last_seen_epoch") or 0.0)
        ):
            running.append(dict(latest))
    return running


def remaining_wall_seconds(
    plan: Plan, prior_unfinished: Sequence[Mapping]
) -> int | None:
    """The tool's wall budget for this attempt: the plan's ``max_wall_seconds``
    less the container time of every earlier unfinished attempt."""

    if plan.max_wall_seconds is None:
        return None
    spent = sum(float(item.get("elapsed_seconds") or 0.0) for item in prior_unfinished)
    return int(plan.max_wall_seconds - spent)


def container_tool_seconds(plan: Plan, elapsed_seconds: float) -> int:
    """How long the tool may run so the runner keeps its reserve.

    The class timeout bounds the function's execution time. The runner stops
    the tool no later than ``timeout_s - runner_overhead_seconds`` after
    ``_run_stage`` began, so what came before the tool (the lock wait,
    pulling and verifying the run's state, staging the inputs), which is
    ``elapsed_seconds``, comes out of the tool's time rather than out of the
    reserve that stopping the tool, hashing, mirroring and the receipt need.
    """

    return int(
        plan.resources.timeout_s
        - plan.stage_spec.runner_overhead_seconds
        - elapsed_seconds
    )


def tool_budget(
    plan: Plan, plan_remaining_seconds: int | None, elapsed_seconds: float
) -> dict[str, object]:
    """The tool's wall budget for this attempt, and which limit set it.

    The smaller of what is left of the plan's ``max_wall_seconds``
    (:func:`remaining_wall_seconds`) and :func:`container_tool_seconds`. A
    plan without ``max_wall_seconds`` still gets the container's limit, so
    no stage runs into the class timeout, which would leave no receipt.
    """

    container = container_tool_seconds(plan, elapsed_seconds)
    if plan_remaining_seconds is not None and plan_remaining_seconds <= container:
        seconds, limited_by = plan_remaining_seconds, "max_wall_seconds"
    else:
        seconds, limited_by = container, "container_timeout"
    return {
        "seconds": seconds,
        "limited_by": limited_by,
        "plan_remaining_seconds": plan_remaining_seconds,
        "container_seconds": container,
        "elapsed_before_tool_seconds": round(elapsed_seconds, 1),
    }


def write_probe_verdict(
    stage: StageSpec, probe_bytes: int, probe_seconds: float
) -> dict[str, object]:
    """Whether the runs volume writes fast enough for the stage's state.

    The check copies ``probe_bytes`` to the runs volume and commits it,
    timed as ``probe_seconds``. Extrapolated to the stage's
    ``mirrored_state_gib``, that is the time mirroring its whole state would
    take after the tool (the push after a stop, which copies everything). The
    runner keeps ``runner_overhead_seconds`` after the tool's deadline, of
    which the stop may take STOP_GRACE_SECONDS and the runner's identity, the
    receipt and the attempt's final commit RECEIPT_RESERVE_SECONDS; if the
    extrapolated mirror does not fit in the rest, a stop at the budget would
    run into the class timeout mid-mirror and leave no receipt, so the check
    reports a problem. One small probe from the check container is a hint,
    not a guarantee.
    """

    if stage.mirrored_state_gib is None:
        return {"skipped": f"stage {stage.name!r} declares no mirrored state size"}
    rate = probe_bytes / max(probe_seconds, 1e-6)
    needed = stage.mirrored_state_gib * 1024**3 / rate
    reserve = (
        stage.runner_overhead_seconds - STOP_GRACE_SECONDS - RECEIPT_RESERVE_SECONDS
    )
    verdict: dict[str, object] = {
        "bytes": probe_bytes,
        "seconds": round(probe_seconds, 2),
        "mb_per_s": round(rate / 1e6, 1),
        "mirrored_state_gib": stage.mirrored_state_gib,
        "implied_mirror_seconds": round(needed),
        "mirror_reserve_seconds": reserve,
        "receipt_reserve_seconds": RECEIPT_RESERVE_SECONDS,
    }
    if needed > reserve:
        verdict["problem"] = (
            f"the runs volume took {probe_seconds:.1f}s for {probe_bytes} bytes "
            f"({rate / 1e6:.1f} MB/s); {stage.mirrored_state_gib} GiB of state "
            f"would take about {needed:.0f}s to mirror, more than the "
            f"{reserve}s the runner keeps for the mirror after the tool"
        )
    return verdict


def estimated_usd_at_max_wall(plan: Plan) -> float | None:
    """List-price cost of a stage that runs its whole budget, runner time included."""

    if plan.max_wall_seconds is None:
        return None
    return plan.resources.estimated_usd(
        plan.max_wall_seconds + plan.stage_spec.runner_overhead_seconds,
        plan.price_multiplier,
    )


def estimated_usd_at_timeout(plan: Plan) -> float:
    """List-price cost of the class's request held for its whole timeout.

    Modal's timeout bounds a function's execution time
    (modal.com/docs/guide/timeouts). This is the ceiling of one attempt at
    the request: for CPU it rests on a CPU limit equal to the request
    (``Resources.cpu_limit``), above which Modal throttles CPU use, and it
    never holds for memory used above the request, which is billed at use
    (modal.com/docs/guide/resources). Scheduling is outside the timeout,
    container startup is timed separately (``startup_timeout``), and a
    function may run a handful of seconds past its timeout
    (modal.com/docs/guide/timeouts); Modal's billing report is the billed
    figure.
    """

    return plan.resources.estimated_usd(plan.resources.timeout_s, plan.price_multiplier)


def summarize(plan: Plan) -> dict[str, object]:
    resources = plan.resources
    stage_spec = plan.stage_spec
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
            "cpu_limit": resources.cpu_limit,
            "memory_gib": resources.memory_gib,
            "timeout_h": resources.timeout_s / 3600,
            "timeout_s": resources.timeout_s,
            "nonpreemptible": plan.nonpreemptible,
            "runner_overhead_seconds": stage_spec.runner_overhead_seconds,
            "min_free_disk_gib": stage_spec.min_free_disk_gib,
            "mirrored_state_gib": stage_spec.mirrored_state_gib,
            "mirror_first": list(stage_spec.mirror_first),
            "rest_is_resume_state": stage_spec.rest_is_resume_state,
        },
        "max_wall_seconds": plan.max_wall_seconds,
        "env": dict(plan.env),
        "inputs": {name: ref.to_json() for name, ref in plan.inputs.items()},
        "argv": planned_argv(plan),
        "image_build_commands": image_build_commands(plan),
    }
    if plan.tool.tree_files:
        summary["tree_files"] = {
            name: {"path": item.path, "sha256": item.sha256}
            for name, item in sorted(plan.tool.tree_files.items())
        }
    if plan.tool.home_seeds:
        summary["home_seeds"] = [
            {"input": seed.input, "path": f"~/{seed.path}", "sha256": seed.sha256}
            for seed in plan.tool.home_seeds
            if seed.input in plan.inputs
        ]
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
        summary["estimated_usd_at_max_wall"] = estimated_usd_at_max_wall(plan)
    # The ceiling of one attempt at the request: the class timeout at list
    # price (see estimated_usd_at_timeout for what it does not bound).
    summary["estimated_usd_at_timeout"] = estimated_usd_at_timeout(plan)
    return summary


# --------------------------------------------------------------------------- #
# Uploading inputs and comparing a Modal run with a local one                  #
# --------------------------------------------------------------------------- #


def upload_rows(plan: Plan, files: Mapping[str, Path | str]) -> list[dict[str, object]]:
    """Hash local files against a plan's volume inputs; the upload for each.

    Every role must be a content-addressed volume input of the plan
    (``volume://cas/sha256/<digest>/<name>``) and its file must hash to the
    plan's digest. Any problem refuses the whole set (PlanError), so no
    upload command is produced for a file the plan does not pin.
    """

    rows: list[dict[str, object]] = []
    problems: list[str] = []
    for role, raw_path in files.items():
        ref = plan.inputs.get(role)
        if ref is None:
            problems.append(f"{role}: not an input of the plan ({sorted(plan.inputs)})")
            continue
        if ref.kind != "volume" or not str(ref.volume_path).startswith("cas/sha256/"):
            problems.append(
                f"{role}: {ref.uri} is not a content-addressed volume input"
            )
            continue
        path = Path(raw_path)
        if not path.is_file():
            problems.append(f"{role}: {path} is not a file")
            continue
        sha, size = sha256_file(path)
        if sha != ref.sha256:
            problems.append(
                f"{role}: {path} is sha256 {sha}; the plan pins {ref.sha256}"
            )
            continue
        remote = str(ref.volume_path)
        rows.append(
            {
                "role": role,
                "file": str(path),
                "bytes": size,
                "sha256": sha,
                "volume_path": remote,
                "exists": (
                    f"modal volume ls {INPUTS_VOLUME} {PurePosixPath(remote).parent}"
                ),
                "upload": (
                    f"modal volume put {INPUTS_VOLUME} {shlex.quote(str(path))} {remote}"
                ),
            }
        )
    if problems:
        raise PlanError("; ".join(problems))
    return rows


def upload_script(rows: Sequence[Mapping[str, object]]) -> str:
    """A bash script that uploads each verified file unless it is already there."""

    lines = [
        "#!/bin/bash",
        "# Written by tools/modal_us_stage_plan.py upload-commands. Each file below",
        "# hashed to its plan digest when this script was written.",
        "set -euo pipefail",
    ]
    for row in rows:
        name = PurePosixPath(str(row["volume_path"])).name
        where = f"{row['role']} {row['volume_path']}"
        lines += [
            f"# {row['role']}: {row['bytes']} bytes, sha256 {row['sha256']}",
            f"if {row['exists']} 2>/dev/null | grep -qF -- {shlex.quote(name)}; then",
            f"  echo {shlex.quote('present ' + where)}",
            "else",
            f"  {row['upload']}",
            f"  echo {shlex.quote('uploaded ' + where)}",
            "fi",
        ]
    return "\n".join(lines) + "\n"


LOCAL_REFERENCE_SCHEMA = "microcosm-modal-us-stage-local-reference/1"

#: run_config keys compare-lineage reports and does not require to match,
#: but for THREAD_ENVIRONMENT_COMPARED. ``thread_environment`` is what the
#: tool read from its own environment (the BLAS/OpenMP thread counts,
#: POPULACE_FIT_N_JOBS, POPULACE_FIT_PREDICT_WORKERS and PYTHONHASHSEED;
#: ``_stage_run_config`` at 4b57d15a2). Modal set four of those thread counts
#: in the check container where the local run set none (runbook, acceptance
#: attempt), so it differs from the local run's without any change to the
#: build's inputs or settings.
RUN_CONFIG_REPORTED_ONLY = frozenset({"thread_environment"})
#: ``thread_environment`` variables compare-lineage requires to match: the
#: hash seed is a determinism input of the pinned tool (HASH_SEED_ENV), and
#: the runner holds it to the local run's, so a difference means the run is
#: not the local run's build.
THREAD_ENVIRONMENT_COMPARED = frozenset({HASH_SEED_ENV})
#: builder_code_identity keys compare-lineage reports and does not require
#: to match: the interpreter build string, which differs by design (the image
#: runs the standard CPython build on Linux, the local run free-threaded
#: CPython on macOS).
CODE_IDENTITY_REPORTED_ONLY = frozenset({"python"})
_YEAR_PATH = re.compile(r"(\d{4})=(/.*)", re.DOTALL)


def mask_config_paths(value: object) -> object:
    """A run config with each absolute path cut to its file name.

    The base's tool records resolved paths in the run config it locks
    (``_stage_run_config`` at 4b57d15a2), so one build differs by directory
    between machines: ``/Users/...`` on the Mac, ``/work/inputs/<role>/`` and
    ``/work/state/`` on Modal. A ``YEAR=PATH`` entry keeps its year. Mapping
    keys, numbers, booleans, digests and every other string are kept, so a
    relative path (which the tool never records) would still differ.
    """

    if isinstance(value, Mapping):
        return {str(key): mask_config_paths(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [mask_config_paths(item) for item in value]
    if isinstance(value, str):
        match = _YEAR_PATH.fullmatch(value)
        if match:
            return f"{match[1]}={PurePosixPath(match[2]).name}"
        if value.startswith("/"):
            return PurePosixPath(value).name
    return value


def tokenize_stage_argv(argv: Sequence[str], work_root: str | None = None) -> list[str]:
    """A container argv in the tokens of a local command's fixture.

    ``<python>`` for the image's venv interpreter, ``<input:ROLE>/<file>``
    for a staged input, ``<state>`` for the state directory and ``<repo>``
    for the pinned clone; a ``YEAR=`` prefix is kept. Anything else is kept
    verbatim, so a path the runner did not build still shows as a
    difference. The tokens are those of
    ``packages/microcosm-build/tests/fixtures/modal_us_stage/
    route_a_base_command_4b57d15a287c.json``.
    """

    work_root = WORK_ROOT if work_root is None else work_root
    inputs_root = PurePosixPath(work_root) / "inputs"
    state = str(PurePosixPath(work_root) / "state")
    python = f"{IMAGE_VENV}/bin/python"

    def token(value: str) -> str:
        year, eq, rest = value.partition("=")
        if eq and year.isdigit() and rest.startswith("/"):
            return f"{year}={token(rest)}"
        if not value.startswith("/"):
            return value
        path = PurePosixPath(value)
        if path.is_relative_to(inputs_root):
            parts = path.relative_to(inputs_root).parts
            if len(parts) == 2:
                return f"<input:{parts[0]}>/{parts[1]}"
            return value
        for root, name in ((state, "<state>"), (IMAGE_REPO_ROOT, "<repo>")):
            if path.is_relative_to(root):
                return name + value[len(root) :]
        return value

    return [
        "<python>" if index == 0 and value == python else token(value)
        for index, value in enumerate(argv)
    ]


def _argv_problem(tokens: Sequence[str], expected: Sequence[str]) -> str | None:
    if list(tokens) == list(expected):
        return None
    diffs = [
        f"[{index}] Modal {modal!r}, local {local!r}"
        for index, (modal, local) in enumerate(zip_longest(tokens, expected))
        if modal != local
    ]
    more = f"; and {len(diffs) - 5} more" if len(diffs) > 5 else ""
    return (
        f"argv: the receipt's command (tokenized, {len(tokens)} elements) is not "
        f"the local run's ({len(expected)}): " + "; ".join(diffs[:5]) + more
    )


def _thread_environment(config: Mapping) -> Mapping:
    threads = config.get("thread_environment")
    return threads if isinstance(threads, Mapping) else {}


def _run_config_problems(modal: Mapping, local: Mapping) -> list[str]:
    """Every run_config key but the reported-only ones, paths masked, and
    the THREAD_ENVIRONMENT_COMPARED variables of ``thread_environment``."""

    modal = mask_config_paths(modal) if isinstance(modal, Mapping) else {}
    local = mask_config_paths(local) if isinstance(local, Mapping) else {}
    problems: list[str] = []
    skip = RUN_CONFIG_REPORTED_ONLY | {"builder_code_identity"}
    for key in sorted((set(modal) | set(local)) - skip):
        if key not in modal:
            problems.append(f"run_config.{key}: not in the Modal run's config")
        elif key not in local:
            problems.append(
                f"run_config.{key}: not in the local run's config "
                f"(Modal {modal[key]!r})"
            )
        elif modal[key] != local[key]:
            problems.append(
                f"run_config.{key}: Modal {modal[key]!r}, local {local[key]!r}"
            )
    modal_threads = _thread_environment(modal)
    local_threads = _thread_environment(local)
    for key in sorted(THREAD_ENVIRONMENT_COMPARED):
        if modal_threads.get(key) != local_threads.get(key):
            problems.append(
                f"run_config.thread_environment.{key}: Modal "
                f"{modal_threads.get(key)!r}, local {local_threads.get(key)!r} "
                "(a determinism input of the tool, compared, never only reported)"
            )
    identity = modal.get("builder_code_identity") or {}
    expected = local.get("builder_code_identity") or {}
    first = ("source_sha256", "dependency_versions")
    rest = sorted((set(identity) | set(expected)) - set(first))
    for key in (*first, *rest):
        if key in CODE_IDENTITY_REPORTED_ONLY:
            continue
        if identity.get(key) != expected.get(key):
            problems.append(
                f"builder_code_identity.{key}: Modal {identity.get(key)!r}, "
                f"local {expected.get(key)!r}"
            )
    return problems


def lineage_problems(
    receipt: Mapping,
    reference: Mapping,
    run_context: Mapping | None,
    run_context_sha256: str | None,
) -> list[str]:
    """Where a Modal run departs from a local run of the same commit and inputs.

    ``reference`` (``microcosm-modal-us-stage-local-reference/1``) records
    the local run: its command (``argv``, paths tokenized as in the fixture,
    and ``env``), its inputs' sha256, the sha256 of deterministic outputs
    (the base's frame checkpoints, written without HDF5 timestamps), and the
    run config its tool locked (``run_config``, paths cut to file names;
    ``pipeline`` and ``pipeline_sha256``). The receipt must say COMPLETED
    (:func:`receipt_status_problems`), carry the same command and input
    digests and list the same bytes for each output. ``run_context`` (the
    Modal run's own file, whose sha256 must be the one the receipt lists)
    must lock the same pipeline and the same run config: every key but
    ``thread_environment``, with paths compared by file name, including
    the builder code identity but for its interpreter string, and, of
    ``thread_environment``, the hash seed (THREAD_ENVIRONMENT_COMPARED). The
    plan's environment must match too, except for the variables the tool
    records in ``thread_environment``. The interpreter, platform and the
    other ``thread_environment`` variables differ by design and are only
    reported (:func:`lineage_information`).

    No problems means the compared outputs match, not that the run
    reproduced the local one: the reference covers only the stages whose
    outputs it lists (``stages_compared``; see :func:`lineage_scope`).
    """

    if reference.get("schema") != LOCAL_REFERENCE_SCHEMA:
        return [f"not a {LOCAL_REFERENCE_SCHEMA} reference"]
    if receipt.get("schema") != RECEIPT_SCHEMA:
        return [f"not a {RECEIPT_SCHEMA} receipt"]
    problems: list[str] = receipt_status_problems(receipt)
    for key in ("tool", "stage"):
        if receipt.get(key) != reference.get(key):
            problems.append(
                f"{key}: the receipt's {receipt.get(key)!r}, the reference's "
                f"{reference.get(key)!r}"
            )
    commit = (receipt.get("source") or {}).get("commit")
    if commit != reference.get("commit"):
        problems.append(
            f"commit: the receipt's {commit}, the local run's {reference.get('commit')}"
        )
    argv_problem = _argv_problem(
        tokenize_stage_argv([str(item) for item in receipt.get("argv") or []]),
        [str(item) for item in reference.get("argv") or []],
    )
    if argv_problem:
        problems.append(argv_problem)
    local_config = reference.get("run_config") or {}
    recorded_env = set(local_config.get("thread_environment") or {})
    plan_env = {
        key: value
        for key, value in sorted(((receipt.get("plan") or {}).get("env") or {}).items())
        if key not in recorded_env
    }
    local_env = {
        key: value
        for key, value in sorted((reference.get("env") or {}).items())
        if key not in recorded_env
    }
    if plan_env != local_env:
        problems.append(
            f"env: the Modal plan sets {plan_env}, the local run set {local_env} "
            "(the thread and worker variables the run context records aside)"
        )
    staged = {
        str(item.get("name")): item.get("sha256")
        for item in receipt.get("inputs") or []
    }
    local_inputs = dict(reference.get("inputs") or {})
    for role in sorted(set(staged) | set(local_inputs)):
        if role not in staged:
            problems.append(f"input {role}: not among the receipt's verified inputs")
        elif role not in local_inputs:
            problems.append(f"input {role}: the local run had no such input")
        elif staged[role] != local_inputs[role]:
            problems.append(
                f"input {role}: Modal staged sha256 {staged[role]}, the local run "
                f"used {local_inputs[role]}"
            )
    listed = {str(item["path"]): item for item in receipt.get("outputs", [])}
    for rel, local_sha in sorted(dict(reference.get("outputs", {})).items()):
        item = listed.get(rel)
        if item is None:
            problems.append(f"{rel}: not in the receipt")
        elif item["sha256"] != local_sha:
            problems.append(
                f"{rel}: Modal wrote sha256 {item['sha256']}, the local run {local_sha}"
            )
    context_path = str(reference.get("run_context_path"))
    item = listed.get(context_path)
    if item is None:
        problems.append(f"{context_path}: not in the receipt")
    elif run_context_sha256 != item["sha256"]:
        problems.append(
            f"{context_path}: the file given is sha256 {run_context_sha256}, "
            f"the receipt lists {item['sha256']}"
        )
    pipeline_sha256 = (run_context or {}).get("pipeline_sha256")
    if pipeline_sha256 != reference.get("pipeline_sha256"):
        problems.append(
            f"pipeline_sha256: Modal {pipeline_sha256!r}, local "
            f"{reference.get('pipeline_sha256')!r} (a different outer-stage list)"
        )
    problems += _run_config_problems(
        (run_context or {}).get("run_config") or {}, local_config
    )
    return problems


def lineage_information(reference: Mapping, run_context: Mapping | None) -> dict:
    """What compare-lineage reports without refusing: the interpreters, and
    each ``thread_environment`` variable whose value differs, but for the
    ones it compares (THREAD_ENVIRONMENT_COMPARED), which are problems."""

    modal_config = (run_context or {}).get("run_config") or {}
    local_config = reference.get("run_config") or {}
    modal_threads = _thread_environment(modal_config)
    local_threads = _thread_environment(local_config)
    return {
        "python": {
            "modal": (modal_config.get("builder_code_identity") or {}).get("python"),
            "local": (local_config.get("builder_code_identity") or {}).get("python"),
        },
        "thread_environment_differences": {
            key: {"modal": modal_threads.get(key), "local": local_threads.get(key)}
            for key in sorted(
                (set(modal_threads) | set(local_threads)) - THREAD_ENVIRONMENT_COMPARED
            )
            if modal_threads.get(key) != local_threads.get(key)
        },
    }


def lineage_scope(reference: Mapping) -> dict[str, object]:
    """Which of the local run's outer stages the reference lets a run compare.

    The reference lists the checkpoints of ``stages_compared`` only (Route
    A's local base stopped after two). Every later stage of ``pipeline`` has
    no local output to compare with, so a match says nothing about them.
    """

    stages = [
        str(stage["name"])
        for stage in ((reference.get("pipeline") or {}).get("stages") or [])
    ]
    compared = [str(name) for name in reference.get("stages_compared") or []]
    return {
        "stages_compared": len(compared),
        "stages_total": len(stages),
        "stages_compared_names": compared,
        "stages_not_compared": [name for name in stages if name not in compared],
    }


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


def _role_paths(pairs: Sequence[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for pair in pairs:
        role, eq, path = pair.partition("=")
        if not eq or not role or not path:
            raise PlanError(f"{pair!r}: expected ROLE=PATH")
        if role in files:
            raise PlanError(f"role {role!r} given twice")
        files[role] = Path(path)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate a plan; print argv and sizing")
    validate.add_argument("plan", type=Path)
    digest = sub.add_parser("digest", help="sha256 + CAS upload command per file")
    digest.add_argument("paths", nargs="+", type=Path)
    upload = sub.add_parser(
        "upload-commands",
        help="hash local files against a plan's inputs; print their upload commands",
    )
    upload.add_argument("plan", type=Path)
    upload.add_argument("files", nargs="+", metavar="ROLE=PATH")
    upload.add_argument(
        "--shell",
        action="store_true",
        help="print a bash script that uploads each file unless it is on the volume",
    )
    verify = sub.add_parser("verify-receipt", help="re-hash a fetched state tree")
    verify.add_argument("receipt", type=Path)
    verify.add_argument("--state-root", type=Path, required=True)
    verify.add_argument("--strict", action="store_true")
    verify.add_argument(
        "--prefix",
        help="verify only the outputs under this directory of the state "
        "(e.g. base-out), for a partial fetch",
    )
    verify.add_argument(
        "--allow-failed",
        action="store_true",
        help="verify the bytes of a FAILED or stopped_at_budget receipt too; "
        "without it such a receipt does not verify",
    )
    lineage = sub.add_parser(
        "compare-lineage",
        help="compare a Modal run's receipt and run context with a local run's",
    )
    lineage.add_argument("receipt", type=Path)
    lineage.add_argument("--reference", type=Path, required=True)
    lineage.add_argument("--run-context", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "validate":
        try:
            _, plan = load_plan(args.plan)
            summary = summarize(plan)
        except PlanError as error:
            print(f"REFUSED: {error}", file=sys.stderr)
            return 2
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "digest":
        for line in _digest_lines(args.paths):
            print(line)
        return 0
    if args.command == "upload-commands":
        try:
            _, plan = load_plan(args.plan)
            rows = upload_rows(plan, _role_paths(args.files))
        except PlanError as error:
            print(f"REFUSED: {error}", file=sys.stderr)
            return 2
        if args.shell:
            print(upload_script(rows), end="")
        else:
            for row in rows:
                print(json.dumps(row))
        return 0
    receipt = json.loads(args.receipt.read_text())
    if args.command == "compare-lineage":
        reference = json.loads(args.reference.read_text())
        context_bytes = args.run_context.read_bytes()
        run_context = json.loads(context_bytes)
        problems = lineage_problems(
            receipt,
            reference,
            run_context,
            hashlib.sha256(context_bytes).hexdigest(),
        )
        for problem in problems:
            print(problem, file=sys.stderr)
        modal_threads = _thread_environment((run_context or {}).get("run_config") or {})
        local_threads = _thread_environment(reference.get("run_config") or {})
        print(
            json.dumps(
                {
                    # Only the reference's outputs are compared; the stages
                    # after them have no local output (lineage_scope).
                    "compared_outputs_match": not problems,
                    "problems": len(problems),
                    "outputs_compared": sorted(reference.get("outputs", {})),
                    **lineage_scope(reference),
                    # Compared: a difference is one of the problems.
                    "thread_environment_compared": {
                        key: {
                            "modal": modal_threads.get(key),
                            "local": local_threads.get(key),
                        }
                        for key in sorted(THREAD_ENVIRONMENT_COMPARED)
                    },
                    # Reported, never refused (lineage_information).
                    **lineage_information(reference, run_context),
                }
            )
        )
        return 0 if not problems else 1
    try:
        problems = verify_receipt(
            receipt,
            args.state_root,
            strict=args.strict,
            prefix=args.prefix,
            require_completed=not args.allow_failed,
        )
    except PlanError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    for problem in problems:
        print(problem, file=sys.stderr)
    outputs = [
        item
        for item in receipt.get("outputs", [])
        if _under_prefix(str(item.get("path", "")), args.prefix)
    ]
    print(
        json.dumps(
            {
                "verified": not problems,
                "status": receipt.get("status"),
                "outputs": len(outputs),
                "problems": len(problems),
            }
        )
    )
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
