#!/usr/bin/env python3
"""Accept a separate census reform-coverage smoke before a tag-only US publish.

A US release built with ``--skip-reform-coverage-smoke`` carries no smoke
verdict: both manifests record ``gate_evidence.reform_coverage_smoke`` as
``"skipped"``. When a ruling lets such a build publish as a tag-only release
on the strength of a separate full-population smoke on its own H5
(``tools/probe_us_post_export.py --census --stages reform_coverage_smoke``),
this tool decides whether that smoke stands in for the build's own.

``expect``
    Derives, from the build commit's source, the probes the smoke must score,
    the baseline plan and the scoring passes. It reads blobs at that commit
    (``git cat-file``) and refuses unless every definition it mirrors matches,
    byte for byte, the source it was reviewed against
    (:data:`MIRRORED_DEFINITIONS`, :data:`PINNED_FILES`).

``launch-record``
    Records each boundary of the build and the smoke. A start record snapshots
    the repository, interpreter, installed distributions, import origins,
    environment, dataset and output directory, then replaces itself with the
    stage's command (``os.execv``), so the record describes the process that
    runs: same environment, interpreter and working directory. The smoke's
    start record also compares its launch with the build's and refuses before
    the smoke runs. An end record repeats the snapshot after the stage exits,
    measuring the runtime with the start's environment, so the check sees any
    change to the installation. The probe's report names only four package
    versions; these records let the check compare the two processes' runtimes.

``check``
    Reads the release directory, the probe's output and the four launch
    records, and collects every failure. Only when there is none does it write
    the attachments into the release directory (the report, the gate file
    under another name, the pass log, a note and the acceptance receipt, which
    goes last) and print the publisher's preflight and publish commands.

The tool never publishes: publication stays a deliberate human step. It uses
only the standard library, so it runs beside the engine environment it
inspects without importing it.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TOOL = "tools/check_us_census_smoke.py"
EXPECTATION_SCHEMA_VERSION = 1
LAUNCH_SCHEMA_VERSION = 2
RECEIPT_SCHEMA_VERSION = 2
SCOPE = "separate-census-smoke-before-tag-publish"

MANIFEST_PATH = (
    "packages/microcosm-build/src/microcosm/build/us/"
    "release_input_coverage_manifest.json"
)
LOADER_PATH = (
    "packages/microcosm-build/src/microcosm/build/us_runtime/release_input_coverage.py"
)
GATE_PATH = (
    "packages/microcosm-build/src/microcosm/build/us_runtime/reform_coverage_smoke.py"
)
BUILDER_PATH = "tools/build_us_fiscal_refresh_release.py"
PROBE_PATH = "tools/probe_us_post_export.py"
SAMPLER_PATH = "tools/sample_us_export_households.py"
SOURCE_PATHS = (
    MANIFEST_PATH,
    LOADER_PATH,
    GATE_PATH,
    BUILDER_PATH,
    PROBE_PATH,
    SAMPLER_PATH,
)

#: The commit whose source this tool mirrors. ``expect`` refuses a build commit
#: whose source differs from it in any pinned definition: the mirror would then
#: describe code that did not run. Re-review the mirror before moving a pin.
REVIEWED_COMMIT = "dc00d158fe429b188c5066db99e7ede94f79cfdf"

#: sha256 of each mirrored top-level definition's exact source lines
#: (decorators included) at :data:`REVIEWED_COMMIT`; see
#: :func:`top_level_definitions`.
MIRRORED_DEFINITIONS: dict[str, dict[str, str]] = {
    LOADER_PATH: {
        "_US_PACKAGE": (
            "0b1e1c6e506219636c67aacf66ad0635156065744679d941e3b1dfdcaf30a397"
        ),
        "US_RELEASE_INPUT_COVERAGE_RESOURCE": (
            "ac45cc0e29e8fcbe23041933a3498e50b994caf6f96117fcc2ec8fc20466a333"
        ),
        "_ECPS_PARITY_REFERENCE_RESOURCE": (
            "109e60a7db372e82900064f125296e08d6fdaa75eef733546cc870c71f71a7dc"
        ),
        "REQUIRED_STATUS": (
            "75688fb6bd2062e362ef894ed34f5ed8e718d0569d2d1c9392b52d614dbc6af6"
        ),
        "REVIEWED_EXCLUSION_STATUS": (
            "f408425f1947ae0cf029fda59b4b582950f33cec141be454fe8c402a528ab7d1"
        ),
        "_VALID_STATUSES": (
            "a97ff15f73f319464e053d54ec3efaf03b17776f578d6b5bce8727f5a8d28544"
        ),
        "ReleaseInputColumn": (
            "837a0d54874c04709f1a4566ee9430bfe5e1fe389be1a986f5f4ad0aa7f89769"
        ),
        "ReformCoverageProbe": (
            "ee99720a8ec760da3363aae24250345a7cf460b8b9c56b4a32fea43e5b11235b"
        ),
        "ReleaseInputCoverageManifest": (
            "30d5e29fa592d33429bbec656796f4ac678d57c3d4b81ff8401546e153d36810"
        ),
        "_resource_text": (
            "7de7c8011581d8402893cee48db7ee0b5d4e12de72d81298df7af81be1821f06"
        ),
        "_resource_payload": (
            "9d60ee2735e4ec48c552837c0b73e52c105d7c8265aeff78b808cb0093f08312"
        ),
        "load_release_input_coverage_manifest": (
            "90b5a75f9ead40359e12dab73204d6caae36bb050618e560459cee1371b34fc2"
        ),
        "us_release_reform_coverage_probes": (
            "fcb1c90bfc82e56eaf12943877aa617a3c86b67c6f90fb82b596603d82c25aff"
        ),
    },
    GATE_PATH: {
        "_weighted_total": (
            "e80f31a74fde0595b1c3ae2b2ddadca7a70be936bf765b101502f2512aa35c55"
        ),
        "_build_reform": (
            "77dd882b9997e4f458e62533e0c46d1a552dba7af0c7c01db3c759da8076ab38"
        ),
        "us_reform_coverage_smoke_gate": (
            "f806b36adde68bd67cf3a0d395ff54dd89945030969709c0067babf76fca7fb7"
        ),
    },
    BUILDER_PATH: {
        "_THREAD_POOL_DEFAULT": (
            "57ece5d009cb7d187702cf44d1be04d00d3c65251073908cdcec5b434410891e"
        ),
        "for _thread_pool_variable": (
            "ee4e914c20b64b0a03ee6a86f54d35faea45832667a5385cc92c407369f6673e"
        ),
        "PERIOD": ("1870d4a5e4e7eed7b03f42ae518295428b3b294a5145c9765765e6982e665e8c"),
        "_runtime_versions": (
            "22e57347eb65f87bd6db3d486a3ba9d07506fe1655b0148a462b8b2f640d3de5"
        ),
        "_package_or_workspace_version": (
            "a5a1bd4d2da78f15f855a3827f0d3a23a8dacd8b5ad10d3236dac827b326623e"
        ),
        "_local_workspace_package_version": (
            "a40cd4c41c0fa58252252fb402ece2568fb8575118f8ebf798988cf00b2f2253"
        ),
        "_household_position_batches": (
            "5d302c813676d7101c3a44a39c18900d5bc9d3a5f26245bca47a178108e3c2c2"
        ),
        "POST_EXPORT_SCORING_METHOD": (
            "a2f5d0da1b56f3db5877b0806c8b6092d014ff083aa5311b4b7145ed055b2411"
        ),
        "PostExportKey": (
            "0c58e786985d4342dbda5346a4bdaef8ef8a9d978f106cce3482dc3894c72f8c"
        ),
        "_post_export_key": (
            "6b7155d290b8bd8d53b19206d6e05be3ff93728ab2cd92f4fa5c567f03278b85"
        ),
        "_post_export_key_record": (
            "afb1d9cc65065147f9404b4aac29cca84f0f6969795aae5f4029ea4180016dfb"
        ),
        "_ascending_period_plan": (
            "231d4e03be8825968ecd929e7515a30d54896f895dc16381b10ea0ee2ad82c28"
        ),
        "_post_export_baseline_plan_record": (
            "4eda1493fa21a45ad5830a75f4c655d320996974b053411fdefea943f1e81717"
        ),
        "_RecordingPostExportSimulation": (
            "e522c6603773e58579fa1ea6f739c57edad34ff073ac531d76603031113e828f"
        ),
        "_record_post_export_baseline_plan": (
            "ba7fccd62653b2ddd338f8e02c0dbcd4826e97fd69a8381efe2831e733fbd217"
        ),
        "_reform_coverage_smoke_consumer": (
            "76bc27033907d81b9e5e90fe083d3840dedb43b537a89ccca6a525dfd2b8568f"
        ),
        "_HouseholdBatchedPostExportScorer": (
            "43f5ee484c22975a49e2a519698d864eeb1890c5b3b3d3acb97c20d79f107393"
        ),
        "_post_export_scoring_manifest_block": (
            "e5320fa2c018442e6dd3612abd040f74ee7d5cbeafb04f7a903fc25b2cd636b9"
        ),
        "_close_post_export_scorer": (
            "879937ec7663cd461bc552e726bcaacf76a86354185b8c51662651aeb5aa2368"
        ),
        "_PostExportConsumer": (
            "fa72663fd3aa4fcee266abd86750efd021e8157728fb850cd5fc5e4293dfa185"
        ),
        "_PlannedPostExportBaseline": (
            "92e2ffe5d73860baa16d0a45822b07c18e7a2979e48abbc8222fd3f9c0b7b987"
        ),
        "_BatchedPostExportReform": (
            "e57fa76f335c35fefd8aab546b63f60b13b7c31de699b9935d25bcc2f3f64c38"
        ),
        "US_RELEASE_GATE_EVIDENCE_FILES": (
            "180a1b7863f3c73e81e0a7ef1afcc1d97b91cfed0540699f28dea4dd22ff196c"
        ),
        "_gate_evidence_status": (
            "9618d0070cbb750c0a382e48855da42dd3d9682fe3f439682ec27aed8bec23b8"
        ),
    },
}

#: Files pinned whole at :data:`REVIEWED_COMMIT`: the probe's report layout and
#: authority rules are what this check reads.
PINNED_FILES: dict[str, str] = {
    PROBE_PATH: "61e96eec077564a902745fec9871c94cf39fc04ec4236179d6d3c1c9a90be848",
}

# The release tool's values this check reproduces; each is pinned above.
PERIOD = 2024
SMOKE_CONSUMER = "reform_coverage_smoke"
SCORING_METHOD = "household_batched_written_h5"
RUNTIME_PACKAGES = (
    "microcosm-build",
    "microcosm-calibrate",
    "microcosm-data",
    "microcosm-frame",
    "policyengine-core",
    "policyengine-us",
    "numpy",
    "pandas",
    "torch",
)
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
)
THREAD_POOL_CEILING = 16
#: Resources the release tool's loaders read from the working directory before
#: the package (``release_input_coverage._resource_text``).
RESOURCE_NAMES = (
    "release_input_coverage_manifest.json",
    "ecps_parity_reference.json",
)

# The probe's report layout (pinned whole above).
PROBE_SCHEMA_VERSION = 2
PROBE_CLASSIFICATION = "probe diagnostics (not release evidence)"
CENSUS_REASON = "scored on the full export (no sample)"
PROBE_MINIMUM_BATCHES = 3
PROBE_PACKAGES = ("policyengine-us", "policyengine-core", "numpy", "pandas")
PROBE_STAGES = (
    "identify",
    "load",
    "design",
    "open_scorer",
    "reform_coverage_smoke",
    "post_export_scoring_plans",
)
PROBE_REFERENCE_KEYS = (
    "smoke",
    "validation",
    "build_manifest",
    "release_manifest",
    "calibration_diagnostics",
    "qrf_tail",
)
PROBE_REPORT = "probe_report.json"
PROBE_GATE = "reform_coverage_smoke.json"
PROBE_PASSES = "passes.jsonl"
GATE_RESULT_FIELDS = frozenset(
    {
        "name",
        "period",
        "budget_measure",
        "baseline_total",
        "reform_total",
        "effect",
        "expected_sign",
        "min_abs_effect",
        "binding_inputs",
        "issue",
        "passed",
    }
)
PASS_FIELDS = frozenset(
    {
        "stage",
        "label",
        "keys",
        "reform",
        "batches",
        "wall_seconds",
        "cpu_seconds",
        "rss_before_gib",
        "peak_rss_gib",
    }
)

#: What a launch record captures: the thread pools, then the variables that
#: change which code, optimization level, hash seed or CPU count a process
#: runs with.
ENVIRONMENT_VARIABLES = (
    *THREAD_VARIABLES,
    "PYTHONPATH",
    "PYTHONHASHSEED",
    "PYTHONSAFEPATH",
    "PYTHONNOUSERSITE",
    "PYTHONHOME",
    "PYTHONOPTIMIZE",
    "PYTHONUSERBASE",
    "PYTHONPLATLIBDIR",
    "PYTHON_CPU_COUNT",
)
IMPORT_MODULES = (
    "microcosm.build",
    "microcosm.calibrate",
    "microcosm.data",
    "microcosm.frame",
    "policyengine_us",
    "policyengine_core",
    "numpy",
    "pandas",
    "torch",
)
#: Where each microcosm module must import from: the worktree's own source,
#: not an install that git ignores (such as one under ``.venv``).
MICROCOSM_SOURCES = {
    "microcosm.build": "packages/microcosm-build/src",
    "microcosm.calibrate": "packages/microcosm-calibrate/src",
    "microcosm.data": "packages/microcosm-data/src",
    "microcosm.frame": "packages/microcosm-frame/src",
}
LAUNCH_FILES = {
    ("build", "start"): "build-start.json",
    ("build", "end"): "build-end.json",
    ("smoke", "start"): "smoke-start.json",
    ("smoke", "end"): "smoke-end.json",
}
#: The script each stage's command runs, and the interpreter options it may
#: pass before it (Route A runs the release tool with ``-B``).
STAGE_SCRIPTS = {"build": BUILDER_PATH, "smoke": PROBE_PATH}
INTERPRETER_OPTIONS = frozenset({"-B", "-u"})
#: The probe arguments a census smoke passes, and no others: a sample, a
#: reference or another stage would change what it scores.
SMOKE_VALUE_FLAGS = ("--stages", "--batch-size", "--export", "--out")
SMOKE_SWITCHES = ("--census",)
#: ``GIT_*`` variables that point git at another repository, index, object
#: store or configuration. The build and the probe run git to record their own
#: source, so a start refuses an environment that sets one.
GIT_REDIRECTING_VARIABLES = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_NAMESPACE",
        "GIT_CEILING_DIRECTORIES",
        "GIT_CONFIG",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
        "GIT_REPLACE_REF_BASE",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_SHALLOW_FILE",
    }
)
EMPTY_STATUS_SHA256 = hashlib.sha256(b"").hexdigest()

#: Files the check writes into the release directory, in writing order; the
#: receipt goes last and records the others' digests.
ATTACHMENTS = {
    "report": "post_export_smoke_report.json",
    "gate": "post_export_smoke_gate.json",
    "passes": "post_export_smoke_passes.jsonl",
    "note": "post_export_smoke_note.md",
    "receipt": "post_export_smoke_acceptance.json",
}
#: Canonical evidence names the build must not have written: their presence
#: would contradict the skipped gate, or bind validation nobody certified.
FORBIDDEN_RELEASE_FILES = ("reform_coverage_smoke.json", "reform_validation.json")

LIMITS = (
    "Every input is a file the operator's run writes, and none is signed, so "
    "a set someone fabricates to agree with itself passes. The check catches "
    "a run that departs from the build, not a forgery.",
    "Launch records capture each stage's start and end, so they miss a change "
    "that someone makes and undoes between two records; the operator attests "
    "that the worktree served only this run.",
    "Git reads the worktree with the repository's own ignore rules, so the "
    "records never see a git-ignored file, such as a bytecode cache; Python "
    "trusts a cached .pyc whose recorded source size and time match.",
    "A start record describes the process it starts, because it records the "
    "environment and then becomes the stage's command. The check compares the "
    "variables in ENVIRONMENT_VARIABLES and no others, such as a numeric "
    "library's own settings.",
    "The check compares installed distributions by name and version, and the "
    "listed packages' RECORD files by digest; it never reads the installed "
    "files' bytes.",
    "The probe exits 0 whenever it finishes, so the smoke's exit code catches "
    "only a crash; the report's verdicts carry the result.",
    "Reform validation never ran, and nothing in this receipt certifies it.",
    "The check records the report's per-household analysis as a diagnostic and "
    "accepts on the gate's own verdict.",
    "The publisher uploads these files but checks nothing in them: only the "
    "printed command's --no-latest --tag-only keeps this publish off latest. "
    "The check does not run the publisher; the operator runs its preflight "
    "next, before any publish.",
)


class RefusalError(Exception):
    """The inputs cannot be accepted; the message says why."""


# ---------------------------------------------------------------------------
# Strict JSON and comparison
# ---------------------------------------------------------------------------


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _no_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name}")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite number {text}")
    return value


def strict_loads(text: str | bytes) -> Any:
    """``json.loads`` refusing duplicate keys and non-finite numbers.

    The release tool and the probe write with ``json.dumps``, which allows
    ``NaN`` and ``Infinity``; a passing census smoke writes neither, so either
    one here means a value this check cannot compare.
    """
    return json.loads(
        text,
        object_pairs_hook=_no_duplicate_keys,
        parse_constant=_no_constant,
        parse_float=_finite_float,
    )


def strict_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def same(left: Any, right: Any) -> bool:
    """Equal values of equal JSON types (``1`` is not ``1.0`` or ``True``).

    A gap (:data:`MISSING`) equals nothing, not even another gap: two records
    that both lack a field do not agree on it.
    """
    if left is MISSING or right is MISSING:
        return False
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            same(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            same(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


MISSING = _Missing()


def dig(value: Any, *path: str | int) -> Any:
    """``value[path[0]][path[1]]...``, or :data:`MISSING` at the first gap."""
    for key in path:
        if isinstance(key, int):
            if not isinstance(value, list) or not -len(value) <= key < len(value):
                return MISSING
        elif not isinstance(value, dict) or key not in value:
            return MISSING
        value = value[key]
    return value


def _show(value: Any) -> str:
    if value is MISSING:
        return "nothing"
    try:
        text = json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= 160 else text[:157] + "..."


class Findings:
    """Every failure the check finds; it reports all of them, not the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def require(self, condition: object, message: str) -> bool:
        if not condition:
            self.failures.append(message)
        return bool(condition)

    def equal(self, actual: Any, expected: Any, what: str) -> bool:
        if same(actual, expected):
            return True
        self.failures.append(f"{what} is {_show(actual)}; required {_show(expected)}")
        return False


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


# ---------------------------------------------------------------------------
# Source pins
# ---------------------------------------------------------------------------


def _definition_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return node.name
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
        return f"for {node.target.id}"
    return None


def top_level_definitions(source: str) -> dict[str, list[str]]:
    """Each top-level definition's exact source lines, by name.

    Functions and classes include their decorators; ``for name in ...`` loops
    are keyed ``"for name"``. A name bound more than once keeps every segment,
    so a pin can refuse the ambiguity instead of choosing one.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    found: dict[str, list[str]] = {}
    for node in tree.body:
        name = _definition_name(node)
        if name is None:
            continue
        decorators = getattr(node, "decorator_list", ())
        start = min([node.lineno, *(item.lineno for item in decorators)])
        segment = "".join(lines[start - 1 : node.end_lineno])
        found.setdefault(name, []).append(segment)
    return found


def definition_digests(source: str, names: Iterable[str]) -> dict[str, str | None]:
    """sha256 of each named definition (``None``: absent or bound twice)."""
    found = top_level_definitions(source)
    return {
        name: sha256_bytes(found[name][0].encode("utf-8"))
        if len(found.get(name, ())) == 1
        else None
        for name in names
    }


def pin_problems(blobs: Mapping[str, bytes]) -> list[str]:
    """Where the given source departs from :data:`REVIEWED_COMMIT`'s."""
    problems: list[str] = []
    for path, expected in PINNED_FILES.items():
        actual = sha256_bytes(blobs[path])
        if actual != expected:
            problems.append(
                f"{path} has sha256 {actual}; this check reads the layout of "
                f"{expected} ({REVIEWED_COMMIT[:12]})"
            )
    for path, pins in MIRRORED_DEFINITIONS.items():
        try:
            source = blobs[path].decode("utf-8")
            digests = definition_digests(source, pins)
        except (SyntaxError, UnicodeDecodeError, ValueError) as error:
            problems.append(f"{path} does not parse ({type(error).__name__})")
            continue
        for name, expected in pins.items():
            actual = digests[name]
            if actual is None:
                problems.append(
                    f"{path}: {name} is absent or bound more than once at top level"
                )
            elif actual != expected:
                problems.append(
                    f"{path}: {name} differs from the source this check mirrors "
                    f"({REVIEWED_COMMIT[:12]})"
                )
    return problems


# ---------------------------------------------------------------------------
# The expectation: the release tool's probe parse, gate and plan, mirrored
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    """``release_input_coverage.ReformCoverageProbe``, mirrored field for field."""

    id: str
    name: str
    parameter_changes: dict[str, Any]
    budget_measure: str
    binding_inputs: tuple[str, ...]
    min_abs_effect: float
    reason: str
    issue: str
    effect_direction: str = "reform_minus_baseline"
    period: int | None = None
    expected_sign: str = "positive"
    neutralized_variable: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ReformCoverageProbe.id is required.")
        has_params = bool(self.parameter_changes)
        has_neutralize = bool(self.neutralized_variable)
        if has_params == has_neutralize:
            raise ValueError(
                f"{self.id}: provide exactly one of parameter_changes or "
                "neutralized_variable."
            )
        if (
            self.neutralized_variable
            and self.neutralized_variable not in self.binding_inputs
        ):
            raise ValueError(
                f"{self.id}: neutralized_variable must be one of binding_inputs."
            )
        if not self.budget_measure:
            raise ValueError(f"{self.id}: budget_measure is required.")
        if self.effect_direction not in {
            "reform_minus_baseline",
            "baseline_minus_reform",
        }:
            raise ValueError(
                f"{self.id}: effect_direction must be 'reform_minus_baseline' or "
                "'baseline_minus_reform'."
            )
        if not (self.min_abs_effect > 0):
            raise ValueError(f"{self.id}: min_abs_effect must be positive.")
        if self.period is not None and self.period < 1990:
            raise ValueError(f"{self.id}: period must be a plausible year.")
        if self.expected_sign not in {"positive", "negative", "either"}:
            raise ValueError(
                f"{self.id}: expected_sign must be 'positive', 'negative', or 'either'."
            )


_REQUIRED_STATUS = "required"
_REVIEWED_EXCLUSION_STATUS = "reviewed_exclusion"
_VALID_STATUSES = frozenset({_REQUIRED_STATUS, _REVIEWED_EXCLUSION_STATUS})


def _validate_column(name: str, status: str, reason: str, issue: str) -> None:
    """``release_input_coverage.ReleaseInputColumn.__post_init__``."""
    if not name:
        raise ValueError("ReleaseInputColumn.name is required.")
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"{name}: status must be one of {sorted(_VALID_STATUSES)}, got {status!r}."
        )
    if status == _REVIEWED_EXCLUSION_STATUS:
        if not reason:
            raise ValueError(f"{name}: a reviewed exclusion needs a reason.")
        if not issue:
            raise ValueError(f"{name}: a reviewed exclusion needs a tracking issue.")


def parse_probe_manifest(payload: Any, resource: str = MANIFEST_PATH) -> list[Probe]:
    """``load_release_input_coverage_manifest(...).probes``, mirrored.

    Every expression that builds a field is the loader's, so a value the
    loader coerces (``str(...)``, ``float(...)``, ``int(...)``) comes out the
    same here; every refusal the loader raises is raised here too.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"{resource}: expected a JSON object.")
    raw_reference = payload.get("reference")
    if not isinstance(raw_reference, Mapping):
        raise ValueError(f"{resource}: 'reference' must be a JSON object.")
    raw_columns = payload.get("columns")
    if not isinstance(raw_columns, Mapping) or not raw_columns:
        raise ValueError(f"{resource}: 'columns' must be a non-empty JSON object.")
    for name, entry in sorted(raw_columns.items()):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{resource}: column {name!r} must be a JSON object.")
        _validate_column(
            str(name),
            str(entry.get("status", "")),
            str(entry.get("reason", "")),
            str(entry.get("issue", "")),
        )
        str(entry.get("note", ""))
    probes: list[Probe] = []
    for raw_probe in payload.get("reform_coverage_probes", []) or []:
        if not isinstance(raw_probe, Mapping):
            raise ValueError(f"{resource}: each reform probe must be a JSON object.")
        parameter_changes = raw_probe.get("parameter_changes")
        if not isinstance(parameter_changes, Mapping):
            raise ValueError(
                f"{resource}: probe {raw_probe.get('id')!r} parameter_changes "
                "must be a JSON object."
            )
        probes.append(
            Probe(
                id=str(raw_probe.get("id", "")),
                name=str(raw_probe.get("name", "")),
                parameter_changes=dict(parameter_changes),
                budget_measure=str(raw_probe.get("budget_measure", "")),
                binding_inputs=tuple(
                    str(leaf) for leaf in raw_probe.get("binding_inputs", ())
                ),
                min_abs_effect=float(raw_probe.get("min_abs_effect", 0.0)),
                reason=str(raw_probe.get("reason", "")),
                issue=str(raw_probe.get("issue", "")),
                effect_direction=str(
                    raw_probe.get("effect_direction", "reform_minus_baseline")
                ),
                period=(
                    None
                    if raw_probe.get("period") is None
                    else int(raw_probe["period"])
                ),
                expected_sign=str(raw_probe.get("expected_sign", "positive")),
                neutralized_variable=(
                    None
                    if raw_probe.get("neutralized_variable") is None
                    else str(raw_probe["neutralized_variable"])
                ),
            )
        )
    schema_version = payload.get("schema_version", 1)
    if not isinstance(schema_version, int):
        raise ValueError(f"{resource}: 'schema_version' must be an integer.")
    seen: set[str] = set()
    for probe in probes:
        if probe.id in seen:
            raise ValueError(f"Duplicate reform coverage probe id {probe.id!r}.")
        seen.add(probe.id)
    return probes


def gate_effect(
    baseline_total: float, reform_total: float, effect_direction: str
) -> float:
    """``us_reform_coverage_smoke_gate``'s effect."""
    if effect_direction == "baseline_minus_reform":
        return baseline_total - reform_total
    return reform_total - baseline_total


def gate_signed_magnitude(effect: float, expected_sign: str) -> float:
    """``us_reform_coverage_smoke_gate``'s direction-normalized magnitude."""
    if expected_sign == "either":
        return abs(effect)
    return effect if expected_sign == "positive" else -effect


def ascending_period_plan(
    keys: Iterable[tuple[str, int, str | None]],
) -> list[tuple[str, int, str | None]]:
    """The release tool's ``_ascending_period_plan``: unique keys in first-seen
    order, stably sorted by period."""
    return sorted(dict.fromkeys(keys), key=lambda key: key[1])


def batching(n_households: int, batch_size: Any) -> tuple[int, int]:
    """``(n_batches, max_batch_households)`` of the release tool's
    ``_household_position_batches``."""
    if n_households <= 0:
        return 0, 0
    if not isinstance(batch_size, int) or batch_size <= 0 or batch_size >= n_households:
        return 1, n_households
    return -(-n_households // batch_size), batch_size


def git_blob(repo: Path, commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "blob", f"{commit}:{path}"],
        capture_output=True,
        env=_git_env(),
        check=False,
    )
    if completed.returncode != 0:
        raise RefusalError(f"{path} is not readable at {commit} in {repo}")
    return completed.stdout


def _git_env() -> dict[str, str]:
    """The environment for this tool's own git calls.

    Git reads the repository it is pointed at, with nothing from the caller:
    the tool drops every inherited ``GIT_*`` variable (``GIT_DIR``,
    ``GIT_INDEX_FILE`` or ``GIT_CONFIG_*`` would describe another repository
    or state) and the system and global configuration. The calls take no
    optional locks and never prompt, so a snapshot neither refreshes the index
    nor starts maintenance in a worktree a run is using.
    """
    env = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    env.update(
        {
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        }
    )
    return env


#: For git status in a snapshot: only the repository's own ignore rules (not
#: the user's excludes file), and no monitor or cache that could answer stale.
GIT_STATUS_CONFIG = (
    "-c",
    f"core.excludesFile={os.devnull}",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.untrackedCache=false",
)


def resolve_commit(repo: Path, commit: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", f"{commit}^{{commit}}"],
        capture_output=True,
        text=True,
        env=_git_env(),
        check=False,
    )
    resolved = completed.stdout.strip()
    if completed.returncode != 0 or resolved != commit:
        raise RefusalError(f"{commit!r} is not a full commit id in {repo}")
    return resolved


def build_expectation(repo: Path, commit: str) -> dict[str, Any]:
    """What a faithful census smoke of a build at ``commit`` records.

    Raises:
        RefusalError: The commit is unreadable, its source departs from the
            reviewed source, or its probe manifest does not load.
    """
    commit = resolve_commit(repo, commit)
    blobs = {path: git_blob(repo, commit, path) for path in SOURCE_PATHS}
    problems = pin_problems(blobs)
    if problems:
        raise RefusalError("; ".join(problems))
    try:
        payload = strict_loads(blobs[MANIFEST_PATH])
        probes = parse_probe_manifest(payload)
    except (ValueError, TypeError) as error:
        raise RefusalError(
            f"{MANIFEST_PATH} at {commit} does not load: {error}"
        ) from None
    if not probes:
        raise RefusalError(f"{MANIFEST_PATH} at {commit} declares no probes")
    rows = []
    keys = []
    for probe in probes:
        probe_period = int(probe.period if probe.period is not None else PERIOD)
        keys.append((probe.budget_measure, probe_period, None))
        rows.append(
            {
                "id": probe.id,
                "name": probe.name,
                "period": probe_period,
                "budget_measure": probe.budget_measure,
                "binding_inputs": list(probe.binding_inputs),
                "min_abs_effect": probe.min_abs_effect,
                "expected_sign": probe.expected_sign,
                "effect_direction": probe.effect_direction,
                "issue": probe.issue,
            }
        )
    plan = ascending_period_plan(keys)
    passes = [
        {"label": f"{SMOKE_CONSUMER} baseline plan", "keys": len(plan), "reform": False}
    ] + [
        {
            "label": f"{SMOKE_CONSUMER} reform {row['budget_measure']}@{row['period']}",
            "keys": 1,
            "reform": True,
        }
        for row in rows
    ]
    return {
        "schema_version": EXPECTATION_SCHEMA_VERSION,
        "tool": TOOL,
        "build_commit": commit,
        "reviewed_commit": REVIEWED_COMMIT,
        "sources": {path: sha256_bytes(blobs[path]) for path in SOURCE_PATHS},
        "probe_manifest": {
            "path": MANIFEST_PATH,
            "sha256": sha256_bytes(blobs[MANIFEST_PATH]),
            "schema_version": payload.get("schema_version", 1),
            "probes": len(rows),
        },
        "default_period": PERIOD,
        "probes": rows,
        "baseline_plan": {
            "keys": [
                {"variable": variable, "period": period, "map_to": map_to}
                for variable, period, map_to in plan
            ],
            "period_order": sorted({period for _, period, _ in plan}),
        },
        "baseline_passes": 1,
        "reform_passes": len(rows),
        "reform_systems": len(rows),
        "passes": passes,
    }


# ---------------------------------------------------------------------------
# Launch records
# ---------------------------------------------------------------------------

#: Run in the stage's interpreter, from the stage's working directory, with the
#: tools directory as ``sys.path[0]`` (as ``python tools/<tool>.py`` sets it).
#: ``installed_distributions`` is the probe's ``_installed_distributions``
#: verbatim; ``version`` is the release tool's ``_package_or_workspace_version``.
RUNTIME_SCRIPT = r"""
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path

sys.path[0] = sys.argv[1]
PACKAGES = json.loads(sys.argv[2])
MODULES = json.loads(sys.argv[3])


def _local_workspace_package_version(package):
    pyproject = Path("packages") / package / "pyproject.toml"
    if not pyproject.is_file():
        return "not-installed"
    with pyproject.open("rb") as file:
        project = tomllib.load(file).get("project")
    if not isinstance(project, Mapping):
        return "not-installed"
    version = project.get("version")
    return version if isinstance(version, str) and version else "not-installed"


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return _local_workspace_package_version(package)


def installed_distributions():
    import importlib.metadata as metadata

    try:
        distributions = list(metadata.distributions())
    except Exception as error:  # the inventory must never stop a run
        try:
            why = f"{type(error).__name__}: {error}"
        except Exception:
            why = type(error).__name__
        return {"count": None, "sha256": None, "unreadable": None, "error": why}
    lines: set[str] = set()
    unreadable = 0
    for dist in distributions:
        try:
            lines.add(f"{str(dist.metadata['Name'] or '').lower()}=={dist.version}")
        except Exception as error:  # one bad record costs only itself
            unreadable += 1
            try:
                where = str(getattr(dist, "_path", None))
            except Exception:
                where = "unknown location"
            lines.add(f"!unreadable {where}: {type(error).__name__}")
    ordered = sorted(lines)
    return {
        "count": len(ordered),
        "sha256": hashlib.sha256(
            "\n".join(ordered).encode("utf-8", "surrogatepass")
        ).hexdigest(),
        "unreadable": unreadable,
    }


def origin(name):
    try:
        spec = importlib.util.find_spec(name)
    except Exception as error:
        return {"error": type(error).__name__}
    if spec is None:
        return None
    locations = spec.submodule_search_locations
    return {
        "origin": spec.origin,
        "locations": None if locations is None else sorted(str(p) for p in locations),
    }


def distribution(package):
    try:
        dist = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        return None
    record = dist.read_text("RECORD")
    direct_url = dist.read_text("direct_url.json")
    return {
        "version": dist.version,
        "record_sha256": None
        if record is None
        else hashlib.sha256(record.encode("utf-8", "surrogatepass")).hexdigest(),
        "direct_url": None if direct_url is None else json.loads(direct_url),
        "location": str(dist.locate_file("")),
    }


print(
    json.dumps(
        {
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "version": sys.version.split()[0],
            "runtime_versions": {
                "python": platform.python_version(),
                **{package: version(package) for package in PACKAGES},
            },
            "installed_distributions": installed_distributions(),
            "import_origins": {name: origin(name) for name in MODULES},
            "distributions": {package: distribution(package) for package in PACKAGES},
            "sys_path": list(sys.path),
        },
        sort_keys=True,
    )
)
"""


def _git_output(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *GIT_STATUS_CONFIG, "-C", str(repo), *args],
        capture_output=True,
        env=_git_env(),
        check=False,
    )
    if completed.returncode != 0:
        raise RefusalError(f"git {' '.join(args)} failed in {repo}")
    return completed.stdout


def _git_status(repo: Path) -> bytes:
    """The worktree's status as a snapshot sees it: every untracked file, with
    only the repository's own ignore rules."""
    return _git_output(
        repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )


def _repo_state(repo: Path) -> dict[str, Any]:
    status = _git_status(repo)
    return {
        "toplevel": _git_output(repo, "rev-parse", "--show-toplevel").decode().strip(),
        "head": _git_output(repo, "rev-parse", "HEAD").decode().strip(),
        "clean": status == b"",
        "status_sha256": sha256_bytes(status),
        "status_entries": status.count(b"\0"),
    }


def _runtime(
    python: str,
    tools: Path,
    environment: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """The interpreter's runtime, reported by a fresh process.

    ``environment`` sets the listed variables for that process (``None``
    removes one), so an end record measures with its start's settings.
    """
    env = dict(os.environ)
    for name, value in (environment or {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    completed = subprocess.run(
        [
            python,
            "-c",
            RUNTIME_SCRIPT,
            str(tools),
            json.dumps(list(RUNTIME_PACKAGES)),
            json.dumps(list(IMPORT_MODULES)),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RefusalError(
            f"{python} could not report its runtime: {completed.stderr.strip()[-400:]}"
        )
    return strict_loads(completed.stdout)


def _path_state(path: Path) -> str:
    if not path.exists():
        return "absent"
    if path.is_dir() and not any(path.iterdir()):
        return "empty"
    return "nonempty"


def snapshot(
    *,
    stage: str,
    phase: str,
    run_id: str,
    repo: Path,
    python: str,
    dataset: Path | None,
    out_dir: Path | None,
    exit_code: int | None,
    command: Sequence[str] | None = None,
    runtime_environment: Mapping[str, str | None] | None = None,
    now: Callable[[], datetime] = _utc_now,
) -> dict[str, Any]:
    """One launch record (see :func:`launch_record_main`).

    ``environment`` is this process's; ``runtime_environment`` is the one the
    runtime was measured with: the same for a start record, the start's for an
    end record.
    """
    environment = {name: os.environ.get(name) for name in ENVIRONMENT_VARIABLES}
    measured = (
        dict(environment)
        if runtime_environment is None
        else {name: runtime_environment.get(name) for name in ENVIRONMENT_VARIABLES}
    )
    cwd = Path.cwd().resolve()
    state = _repo_state(repo)
    toplevel = Path(state["toplevel"])
    dataset_state = None
    if dataset is not None:
        resolved = dataset.resolve()
        exists = resolved.is_file()
        dataset_state = {
            "path": str(resolved),
            "exists": exists,
            "bytes": resolved.stat().st_size if exists else None,
            "sha256": sha256_file(resolved) if exists else None,
        }
    return {
        "schema_version": LAUNCH_SCHEMA_VERSION,
        "tool": TOOL,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "stage": stage,
        "phase": phase,
        "run_id": run_id,
        "recorded_at": now().isoformat(),
        "repo": state,
        "files": {
            name: (
                sha256_file(toplevel / name) if (toplevel / name).is_file() else None
            )
            for name in ("uv.lock", "pyproject.toml")
        },
        "cwd": str(cwd),
        "cwd_shadows": sorted(name for name in RESOURCE_NAMES if (cwd / name).exists()),
        "command": None if command is None else list(command),
        "python": {"requested": python, "resolved": str(Path(python).resolve())},
        "runtime": _runtime(python, toplevel / "tools", measured),
        "environment": environment,
        "runtime_environment": measured,
        "git_environment": sorted(
            name for name in os.environ if name.startswith("GIT_")
        ),
        "cpu_count": os.cpu_count(),
        "dataset": dataset_state,
        "out_dir": None
        if out_dir is None
        else {"path": str(out_dir.resolve()), "state": _path_state(out_dir)},
        "exit_code": exit_code,
    }


def _write_new(path: Path, data: bytes) -> None:
    """Write ``data`` to a file that must not exist yet."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _exec(path: str, argv: Sequence[str]) -> None:
    """Become the stage's command (the tests replace this)."""
    os.execv(path, list(argv))


def _script_index(command: Sequence[str]) -> int:
    """The position of the script, after the interpreter and its options."""
    position = 1
    while position < len(command) and command[position] in INTERPRETER_OPTIONS:
        position += 1
    return position


def _smoke_arguments(
    arguments: Sequence[str],
) -> tuple[dict[str, list[str]], list[str]]:
    """The probe arguments by flag (``--flag value`` or ``--flag=value``), and
    every token that is not one of a census smoke's."""
    values: dict[str, list[str]] = {}
    unknown: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        flag, equals, inline = token.partition("=")
        if flag in SMOKE_SWITCHES and not equals:
            values.setdefault(flag, []).append("")
        elif flag in SMOKE_VALUE_FLAGS and equals:
            values.setdefault(flag, []).append(inline)
        elif flag in SMOKE_VALUE_FLAGS and index + 1 < len(arguments):
            index += 1
            values.setdefault(flag, []).append(arguments[index])
        else:
            unknown.append(token)
        index += 1
    return values, unknown


def _command_problems(
    stage: str,
    command: Sequence[str],
    *,
    python: str,
    cwd: Path,
    dataset: Path | None,
    out_dir: Path,
) -> list[str]:
    """Why ``command`` is not the stage's documented launch, if it isn't.

    The command runs the ``--python`` interpreter, spelled the same, with at
    most :data:`INTERPRETER_OPTIONS` before the stage's script. The smoke
    passes exactly the census smoke's arguments: ``--census``, ``--stages``
    naming only the smoke, a positive ``--batch-size``, and ``--export`` and
    ``--out`` naming the recorded dataset and output directory. Relative paths
    resolve against ``cwd``, the directory the command runs in.
    """
    if not command:
        return [f"a start record launches the {stage}: give its command after --"]
    problems = []
    if command[0] != python:
        problems.append(
            f"the command runs {command[0]!r}, not the --python interpreter {python!r}"
        )
    position = _script_index(command)
    script = STAGE_SCRIPTS[stage]
    if (
        position >= len(command)
        or (cwd / command[position]).resolve() != (cwd / script).resolve()
    ):
        problems.append(
            f"the command must run {script}, with no interpreter options but "
            f"{sorted(INTERPRETER_OPTIONS)} before it; it runs "
            f"{list(command[1 : position + 1])}"
        )
        return problems
    if stage != "smoke":
        return problems
    values, unknown = _smoke_arguments(command[position + 1 :])
    if unknown:
        problems.append(
            f"the smoke passes {unknown}; a census smoke passes only "
            f"{[*SMOKE_SWITCHES, *SMOKE_VALUE_FLAGS]}"
        )
    missing = [
        flag
        for flag in (*SMOKE_SWITCHES, *SMOKE_VALUE_FLAGS)
        if len(values.get(flag, ())) != 1
    ]
    if missing:
        problems.append(f"the smoke must pass each of {missing} once")
        return problems
    (stages,) = values["--stages"]
    if stages != SMOKE_CONSUMER:
        problems.append(
            f"the smoke runs --stages {stages!r}; required {SMOKE_CONSUMER!r}"
        )
    (batch,) = values["--batch-size"]
    if not (batch.isascii() and batch.isdigit() and int(batch) > 0):
        problems.append(f"the smoke's --batch-size {batch!r} is not a positive integer")
    (export,) = values["--export"]
    if dataset is None:
        problems.append("the smoke's start record names no dataset for --export")
    elif (cwd / export).resolve() != dataset.resolve():
        problems.append(
            f"the smoke scores --export {export}, not the recorded dataset "
            f"{dataset.resolve()}"
        )
    (out,) = values["--out"]
    if (cwd / out).resolve() != out_dir.resolve():
        problems.append(
            f"the smoke writes --out {out}, not the recorded output directory "
            f"{out_dir.resolve()}"
        )
    return problems


def _build_batch_size(build_start: Any) -> int | None:
    """The build manifest's batch size, when the build's output directory holds
    exactly one release (``None`` otherwise: the check compares later)."""
    out = dig(build_start, "out_dir", "path")
    if not isinstance(out, str):
        return None
    manifests = sorted(Path(out).glob("releases/*/build_manifest.json"))
    if len(manifests) != 1:
        return None
    try:
        manifest = strict_loads(manifests[0].read_bytes())
    except (OSError, ValueError):
        return None
    value = dig(manifest, "post_export_scoring", "maximum_batch_size")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _check_smoke_launch(
    findings: Findings, smoke_start: Any, build_start: Any, build_end: Any
) -> None:
    """The smoke's launch against the build's: the run, commit, lockfile, tool,
    interpreter, runtime, environment and dataset. ``launch-record`` runs this
    before the smoke starts, so a wrong launch costs no smoke; ``check`` runs
    it again."""
    for field in ("run_id", "tool_sha256", "files"):
        findings.equal(
            dig(smoke_start, field), dig(build_start, field), f"smoke-start {field}"
        )
    findings.equal(
        dig(smoke_start, "repo", "head"),
        dig(build_start, "repo", "head"),
        "smoke-start HEAD",
    )
    findings.equal(
        dig(smoke_start, "python", "resolved"),
        dig(build_start, "python", "resolved"),
        "smoke-start interpreter",
    )
    findings.equal(
        dig(smoke_start, "runtime"), dig(build_start, "runtime"), "smoke-start runtime"
    )
    findings.equal(dig(build_end, "exit_code"), 0, "the build's exit code")
    build_dataset = dig(build_end, "dataset")
    findings.require(
        isinstance(dig(build_dataset, "sha256"), str),
        "build-end recorded no dataset digest",
    )
    findings.equal(
        dig(smoke_start, "dataset", "path"),
        dig(build_dataset, "path"),
        "smoke-start dataset path (build-end's)",
    )
    findings.equal(
        dig(smoke_start, "dataset", "sha256"),
        dig(build_dataset, "sha256"),
        "smoke-start dataset (build-end's)",
    )
    smoke_environment = dig(smoke_start, "environment")
    build_environment = dig(build_start, "environment")
    if not findings.require(
        isinstance(smoke_environment, dict) and isinstance(build_environment, dict),
        "a start record lists no environment",
    ):
        return
    # The smoke sets the thread pools to the build's effective values
    # explicitly: the probe imports numpy before it loads the release tool,
    # whose defaults would come too late.
    findings.equal(
        {name: smoke_environment.get(name) for name in THREAD_VARIABLES},
        _effective_threads(build_start),
        "the smoke's thread-pool settings (the build's effective values)",
    )
    for name in ENVIRONMENT_VARIABLES[len(THREAD_VARIABLES) :]:
        findings.equal(
            smoke_environment.get(name),
            build_environment.get(name),
            f"the smoke's {name}",
        )


def launch_record_main(args: argparse.Namespace) -> int:
    """Record one stage boundary; refuse a start that would waste a run.

    Every record names the interpreter by its path, and each boundary is
    recorded once. Every record but the build's start needs ``--dataset``, the
    national H5 (the build's start may name it before it exists). A start
    record also needs ``--out-dir`` and the stage's command after ``--``. It
    refuses, writing nothing, when:

    - the command is not the stage's documented launch
      (:func:`_command_problems`);
    - the tree has uncommitted changes, or the environment sets a variable in
      :data:`GIT_REDIRECTING_VARIABLES`;
    - the working directory differs from the repository root or holds a
      resource the release tool's loaders read before the package's own;
    - the output directory already holds files;
    - for the smoke: no build records exist, the smoke's launch differs from
      the build's (:func:`_check_smoke_launch`), its batch size differs from
      the build manifest's, or a ``sample_receipt.json`` beside the H5 would
      make the probe score a sample.

    Then it writes the record and becomes the command (:func:`_exec`). An end
    record needs ``--exit-code`` and takes no command or ``--out-dir``. It
    refuses when no start record exists, the start names another run, or the
    interpreter cannot report its runtime; otherwise it writes the record even
    if the tree changed, because it is evidence and ``check`` judges it.
    """
    launch_dir = Path(args.launch_dir)
    launch_dir.mkdir(parents=True, exist_ok=True)
    target = launch_dir / LAUNCH_FILES[(args.stage, args.phase)]
    if target.exists():
        raise RefusalError(f"{target} already exists; each boundary is recorded once")
    if os.sep not in args.python:
        raise RefusalError(
            f"--python {args.python!r} must name the interpreter by its path"
        )
    if args.dataset is None and (args.stage, args.phase) != ("build", "start"):
        raise RefusalError("this launch record needs --dataset, the national H5")
    repo = Path(args.repo).resolve()
    dataset = None if args.dataset is None else Path(args.dataset)
    command = list(args.stage_command)
    builds: tuple[Any, Any] | None = None
    runtime_environment = None
    if args.phase == "start":
        if args.exit_code is not None:
            raise RefusalError("--exit-code belongs to an end record")
        if args.out_dir is None:
            raise RefusalError("a start record needs --out-dir")
        problems = _command_problems(
            args.stage,
            command,
            python=args.python,
            cwd=Path.cwd(),
            dataset=dataset,
            out_dir=Path(args.out_dir),
        )
        if not (Path(args.python).is_file() and os.access(args.python, os.X_OK)):
            problems.append(f"{args.python} is not an executable file")
        if problems:
            raise RefusalError("; ".join(problems))
        if args.stage == "smoke":
            paths = [
                launch_dir / LAUNCH_FILES[("build", phase)]
                for phase in ("start", "end")
            ]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise RefusalError(
                    f"the smoke starts after the build ends; no {missing}"
                )
            builds = (
                strict_loads(paths[0].read_bytes()),
                strict_loads(paths[1].read_bytes()),
            )
    else:
        if command:
            raise RefusalError("an end record launches nothing; give no command")
        if args.out_dir is not None:
            raise RefusalError("--out-dir belongs to a start record")
        if args.exit_code is None:
            raise RefusalError("an end record needs --exit-code")
        start = launch_dir / LAUNCH_FILES[(args.stage, "start")]
        if not start.is_file():
            raise RefusalError(f"no start record {start}")
        start_record = strict_loads(start.read_bytes())
        if dig(start_record, "run_id") != args.run_id:
            raise RefusalError("the start record names another run")
        runtime_environment = dig(start_record, "environment")
        if not isinstance(runtime_environment, dict):
            raise RefusalError("the start record lists no environment")
    record = snapshot(
        stage=args.stage,
        phase=args.phase,
        run_id=args.run_id,
        repo=repo,
        python=args.python,
        dataset=dataset,
        out_dir=None if args.out_dir is None else Path(args.out_dir),
        exit_code=args.exit_code,
        command=command if args.phase == "start" else None,
        runtime_environment=runtime_environment,
    )
    if args.phase == "start":
        problems = []
        if not record["repo"]["clean"]:
            entries = [
                entry.decode("utf-8", "replace")
                for entry in _git_status(repo).split(b"\0")
                if entry
            ]
            problems.append(
                f"{record['repo']['toplevel']} has uncommitted changes, counting "
                "files that only a global excludes file ignores: "
                f"{entries[:5]}{' ...' if len(entries) > 5 else ''}"
            )
        redirecting = sorted(set(record["git_environment"]) & GIT_REDIRECTING_VARIABLES)
        if redirecting:
            problems.append(
                f"the environment sets {redirecting}, which point git at another "
                "repository or configuration"
            )
        if record["cwd"] != str(Path(record["repo"]["toplevel"]).resolve()):
            problems.append(
                f"the working directory {record['cwd']} is not the repository root"
            )
        if record["cwd_shadows"]:
            problems.append(
                f"the working directory holds {record['cwd_shadows']}, which the "
                "release tool's loaders read before the package's own"
            )
        if record["out_dir"]["state"] == "nonempty":
            problems.append(f"{record['out_dir']['path']} already holds files")
        if builds is not None:
            findings = Findings()
            _check_smoke_launch(findings, record, *builds)
            problems += findings.failures
            values, _ = _smoke_arguments(command[_script_index(command) + 1 :])
            batch_size = _build_batch_size(builds[0])
            if batch_size is not None and values["--batch-size"] != [str(batch_size)]:
                problems.append(
                    f"the smoke's --batch-size {values['--batch-size'][0]} is not "
                    f"the build manifest's maximum_batch_size {batch_size}"
                )
            assert dataset is not None
            receipt_path = dataset.resolve().parent / "sample_receipt.json"
            if receipt_path.exists():
                problems.append(
                    f"{receipt_path} exists, and the probe would score the sample "
                    "it describes"
                )
        if problems:
            raise RefusalError("; ".join(problems))
    _write_new(target, strict_dumps(record).encode("utf-8"))
    print(f"recorded {target}", flush=True)
    if args.phase == "start":
        sys.stderr.flush()
        _exec(command[0], command)
    return 0


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def _load_json(path: Path, findings: Findings) -> tuple[Any, bytes | None]:
    try:
        data = path.read_bytes()
    except OSError as error:
        findings.fail(f"{path} is unreadable ({type(error).__name__})")
        return MISSING, None
    try:
        return strict_loads(data), data
    except (ValueError, UnicodeDecodeError) as error:
        findings.fail(f"{path} is not strict JSON: {error}")
        return MISSING, None


def _load_jsonl(path: Path, findings: Findings) -> tuple[list[Any], bytes | None]:
    try:
        data = path.read_bytes()
    except OSError as error:
        findings.fail(f"{path} is unreadable ({type(error).__name__})")
        return [], None
    rows = []
    text = data.decode("utf-8", "replace")
    if text and not text.endswith("\n"):
        findings.fail(f"{path} ends mid-line")
    for number, line in enumerate(text.splitlines(), start=1):
        try:
            rows.append(strict_loads(line))
        except ValueError as error:
            findings.fail(f"{path} line {number} is not strict JSON: {error}")
    return rows, data


@dataclass
class Inputs:
    release_dir: Path
    artifact_root: Path
    probe_dir: Path
    launch_dir: Path
    repo: Path
    authorization: str
    dedicated_worktree: bool
    #: How many probes the ruling requires, every one present and passing.
    probe_count: int


@dataclass
class Accepted:
    """What a passing check read, for the attachments and the receipt."""

    release_id: str
    commit: str
    run_id: str
    dataset_key: str
    dataset_path: Path
    dataset_sha256: str
    dataset_bytes: int
    repo_id: str
    expectation: dict[str, Any]
    files: dict[str, bytes]
    launches: dict[str, dict[str, Any]]
    smoke_summary: dict[str, Any]


def check(inputs: Inputs, *, now: Callable[[], datetime] = _utc_now) -> Accepted:
    """Every acceptance condition, collected.

    Raises:
        RefusalError: With every failure found, one per line.
    """
    findings = Findings()
    release_dir = inputs.release_dir.resolve()
    probe_dir = inputs.probe_dir.resolve()
    repo = inputs.repo.resolve()
    if not inputs.authorization.strip():
        findings.fail("--authorization must name the ruling that allows this publish")
    if not inputs.dedicated_worktree:
        findings.fail(
            "--attest-dedicated-worktree is required: launch records cannot see a "
            "change made and undone between two snapshots"
        )
    if (
        not isinstance(inputs.probe_count, int)
        or isinstance(inputs.probe_count, bool)
        or inputs.probe_count < 1
    ):
        findings.fail("--probe-count must be the ruling's positive probe count")

    build, build_bytes = _load_json(release_dir / "build_manifest.json", findings)
    release, release_bytes = _load_json(release_dir / "release_manifest.json", findings)
    report, report_bytes = _load_json(probe_dir / PROBE_REPORT, findings)
    gate_file, gate_bytes = _load_json(probe_dir / PROBE_GATE, findings)
    passes, passes_bytes = _load_jsonl(probe_dir / PROBE_PASSES, findings)
    launches: dict[str, Any] = {}
    for (stage, phase), filename in LAUNCH_FILES.items():
        launches[f"{stage}-{phase}"], _ = _load_json(
            inputs.launch_dir / filename, findings
        )
    if findings.failures:
        raise RefusalError("\n".join(findings.failures))

    # -- the release ---------------------------------------------------------
    release_id = dig(build, "build_id")
    if not findings.require(
        isinstance(release_id, str) and release_id,
        "build_manifest.json names no build_id",
    ):
        raise RefusalError("\n".join(findings.failures))
    findings.equal(release_dir.name, release_id, "the release directory's name")
    findings.equal(dig(release, "build", "build_id"), release_id, "release build_id")
    findings.equal(
        dig(release, "build", "built_at"),
        dig(build, "created_at"),
        "release build.built_at (the build manifest's created_at)",
    )
    findings.equal(
        dig(release, "build", "calibration_runtime"),
        dig(build, "calibration_runtime"),
        "release build.calibration_runtime",
    )
    commit = dig(build, "code", "git_commit")
    findings.equal(dig(build, "code", "git_dirty"), False, "build code.git_dirty")
    findings.equal(
        dig(build, "code", "repository"),
        "PolicyEngine/microcosm",
        "build code.repository",
    )
    key = dig(release, "default_datasets", "national")
    artifact = dig(release, "artifacts", key) if isinstance(key, str) else MISSING
    findings.equal(dig(artifact, "kind"), "microdata", "the national artifact's kind")
    findings.equal(
        dig(artifact, "path"),
        dig(build, "dataset", "filename"),
        "the national artifact's path",
    )
    findings.equal(
        dig(artifact, "revision"), release_id, "the national artifact's revision"
    )
    repo_id = dig(artifact, "repo_id")
    findings.require(
        isinstance(repo_id, str) and repo_id, "the national artifact names no repo_id"
    )
    dataset_path = MISSING
    dataset_sha = None
    dataset_bytes = 0
    relative = dig(artifact, "path")
    if isinstance(relative, str) and relative and not Path(relative).is_absolute():
        candidate = (inputs.artifact_root / relative).resolve()
        root = inputs.artifact_root.resolve()
        if findings.require(
            candidate.is_relative_to(root) and candidate.is_file(),
            f"{relative} is not a file under {root}",
        ):
            dataset_path = candidate
            dataset_bytes = candidate.stat().st_size
            dataset_sha = sha256_file(candidate)
    else:
        findings.fail("the national artifact's path is not a relative path")
    findings.equal(
        dig(artifact, "sha256"), dataset_sha, "the national artifact's sha256"
    )
    findings.equal(dig(build, "dataset", "sha256"), dataset_sha, "build dataset.sha256")

    evidence = dig(build, "gate_evidence")
    findings.equal(
        dig(evidence, "reform_coverage_smoke"),
        "skipped",
        "build gate_evidence.reform_coverage_smoke",
    )
    findings.equal(
        dig(release, "build", "gate_evidence"), evidence, "release build.gate_evidence"
    )
    scoring = dig(build, "post_export_scoring")
    findings.require(
        isinstance(scoring, dict), "the build recorded no post_export_scoring"
    )
    findings.equal(
        dig(release, "build", "post_export_scoring"),
        scoring,
        "release build.post_export_scoring",
    )
    findings.equal(dig(scoring, "method"), SCORING_METHOD, "build scoring method")
    findings.equal(
        dig(scoring, "dataset_sha256"), dataset_sha, "build scoring dataset_sha256"
    )
    consumers = dig(scoring, "consumers")
    if findings.require(
        isinstance(consumers, dict), "build scoring lists no consumers"
    ):
        for name in (SMOKE_CONSUMER, "reform_validation"):
            findings.require(
                name not in consumers,
                f"the build scored {name}, which a skipped-smoke build does not",
            )
    for filename in FORBIDDEN_RELEASE_FILES:
        findings.require(
            not (release_dir / filename).exists(),
            f"the release directory holds {filename}",
        )
    for filename in ATTACHMENTS.values():
        findings.require(
            not (release_dir / filename).exists(),
            f"the release directory already holds {filename}",
        )

    n_households = dig(scoring, "n_households")
    batch_size = dig(scoring, "maximum_batch_size")
    if findings.require(
        isinstance(n_households, int)
        and not isinstance(n_households, bool)
        and n_households > 0,
        "build scoring n_households is not a positive integer",
    ):
        n_batches, max_batch = batching(n_households, batch_size)
        findings.equal(dig(scoring, "n_batches"), n_batches, "build scoring n_batches")
        findings.equal(
            dig(scoring, "max_batch_households"),
            max_batch,
            "build scoring max_batch_households",
        )
        findings.require(
            n_batches >= PROBE_MINIMUM_BATCHES,
            f"the build scored {n_batches} batch(es); the probe needs "
            f"{PROBE_MINIMUM_BATCHES} or more at the same batch size",
        )

    # -- the expectation from the build commit --------------------------------
    expectation: dict[str, Any] | None = None
    if isinstance(commit, str):
        try:
            expectation = build_expectation(repo, commit)
        except RefusalError as error:
            findings.fail(f"no expectation for {commit}: {error}")
    else:
        findings.fail("build code.git_commit is missing")
    if expectation is None or dataset_sha is None or not isinstance(dataset_path, Path):
        raise RefusalError("\n".join(findings.failures))
    probes = expectation["probes"]
    n_probes = len(probes)
    plan = expectation["baseline_plan"]
    # The ruling names a probe count; a build commit whose manifest lists more
    # or fewer probes is not the build the ruling allowed.
    findings.equal(
        n_probes, inputs.probe_count, "the build commit's probe count (--probe-count)"
    )

    # -- the report ------------------------------------------------------------
    _check_report(
        findings,
        report,
        build=build,
        scoring=scoring,
        expectation=expectation,
        repo=repo,
        dataset_sha=dataset_sha,
    )

    # -- the gate file ---------------------------------------------------------
    findings.equal(dig(gate_file, "schema_version"), 1, "gate schema_version")
    findings.equal(dig(gate_file, "enforced"), True, "gate enforced")
    gate = dig(gate_file, "reform_coverage_smoke")
    findings.equal(dig(gate, "passed"), True, "gate passed")
    findings.equal(dig(gate, "failures"), [], "gate failures")
    findings.equal(
        dig(gate, "details", "default_period"), PERIOD, "gate default_period"
    )
    findings.equal(dig(gate, "details", "probes"), n_probes, "gate probe count")
    results = dig(gate, "details", "results")
    if findings.require(isinstance(results, dict), "the gate recorded no results"):
        findings.equal(
            sorted(results), sorted(row["id"] for row in probes), "gate result ids"
        )
    totals: dict[str, tuple[float, float, float]] = {}
    for row in probes:
        result = dig(results, row["id"])
        where = f"gate result {row['id']}"
        if not findings.require(isinstance(result, dict), f"{where} is missing"):
            continue
        findings.equal(sorted(result), sorted(GATE_RESULT_FIELDS), f"{where} fields")
        for field in (
            "name",
            "period",
            "budget_measure",
            "expected_sign",
            "min_abs_effect",
            "binding_inputs",
            "issue",
        ):
            findings.equal(dig(result, field), row[field], f"{where} {field}")
        baseline_total = dig(result, "baseline_total")
        reform_total = dig(result, "reform_total")
        if not findings.require(
            isinstance(baseline_total, float) and isinstance(reform_total, float),
            f"{where} totals are not finite floats",
        ):
            continue
        effect = gate_effect(baseline_total, reform_total, row["effect_direction"])
        magnitude = gate_signed_magnitude(effect, row["expected_sign"])
        findings.equal(dig(result, "effect"), effect, f"{where} effect")
        findings.equal(dig(result, "passed"), True, f"{where} passed")
        findings.require(
            magnitude >= row["min_abs_effect"],
            f"{where}: the recorded totals give {magnitude!r}, under the floor "
            f"{row['min_abs_effect']!r}",
        )
        totals[row["id"]] = (baseline_total, reform_total, effect)
    consumer_record = {
        "method": SCORING_METHOD,
        "dataset_sha256": dataset_sha,
        "maximum_batch_size": batch_size,
        "n_households": n_households,
        "n_batches": dig(scoring, "n_batches"),
        "max_batch_households": dig(scoring, "max_batch_households"),
        "spm": dig(scoring, "spm"),
        "baseline_plan": plan,
        "baseline_passes": 1,
        "reform_passes": n_probes,
        "reform_systems": n_probes,
    }
    findings.equal(
        dig(gate_file, "post_export_scoring"),
        consumer_record,
        "the gate file's post_export_scoring",
    )
    _check_smoke_rows(findings, report, probes, totals)

    # -- the pass log ----------------------------------------------------------
    expected_passes = expectation["passes"]
    findings.equal(len(passes), len(expected_passes), "scoring passes in the log")
    for index, (row, expected) in enumerate(
        zip(passes, expected_passes, strict=False), start=1
    ):
        where = f"pass {index}"
        if not findings.require(isinstance(row, dict), f"{where} is not an object"):
            continue
        findings.equal(sorted(row), sorted(PASS_FIELDS), f"{where} fields")
        findings.equal(dig(row, "stage"), SMOKE_CONSUMER, f"{where} stage")
        findings.equal(dig(row, "label"), expected["label"], f"{where} label")
        findings.equal(dig(row, "keys"), expected["keys"], f"{where} keys")
        findings.equal(dig(row, "reform"), expected["reform"], f"{where} reform")
        findings.equal(
            dig(row, "batches"), dig(scoring, "n_batches"), f"{where} batches"
        )

    # -- the launch records ----------------------------------------------------
    run_id = _check_launches(
        findings,
        launches,
        build=build,
        report=report,
        repo=repo,
        commit=commit,
        dataset_path=dataset_path,
        dataset_sha=dataset_sha,
        release_dir=release_dir,
        artifact_root=inputs.artifact_root.resolve(),
        probe_dir=probe_dir,
        now=now,
    )

    if findings.failures:
        raise RefusalError("\n".join(findings.failures))
    assert report_bytes is not None and gate_bytes is not None
    assert passes_bytes is not None and build_bytes is not None
    assert release_bytes is not None and isinstance(dataset_path, Path)
    smoke = dig(report, "stages", "reform_coverage_smoke")
    decompositions = {
        str(row.get("probe")): row.get("decomposition")
        for row in dig(smoke, "probes")
        if isinstance(row, dict)
    }
    return Accepted(
        release_id=release_id,
        commit=commit,
        run_id=run_id,
        dataset_key=key,
        dataset_path=dataset_path,
        dataset_sha256=dataset_sha,
        dataset_bytes=dataset_bytes,
        repo_id=repo_id,
        expectation=expectation,
        files={
            "build_manifest.json": build_bytes,
            "release_manifest.json": release_bytes,
            "report": report_bytes,
            "gate": gate_bytes,
            "passes": passes_bytes,
        },
        launches={name: launches[name] for name in launches},
        smoke_summary={
            "census": True,
            "probes": n_probes,
            "passed": n_probes,
            "n_households": n_households,
            "n_batches": dig(scoring, "n_batches"),
            "maximum_batch_size": batch_size,
            "baseline_keys": len(plan["keys"]),
            "reform_passes": n_probes,
            "reform_systems": n_probes,
            "created_at": dig(report, "created_at"),
            # The report's per-household analysis: a diagnostic the check
            # copies without judging it.
            "diagnostic_analysis": {
                "decomposed": sum(
                    value == "decomposed" for value in decompositions.values()
                ),
                "other": {
                    probe: value
                    for probe, value in sorted(decompositions.items())
                    if value != "decomposed"
                },
            },
        },
    )


def _check_report(
    findings: Findings,
    report: Any,
    *,
    build: Any,
    scoring: Any,
    expectation: Mapping[str, Any],
    repo: Path,
    dataset_sha: str,
) -> None:
    probes = expectation["probes"]
    n_probes = len(probes)
    sources = expectation["sources"]
    commit = expectation["build_commit"]
    runtime = dig(build, "runtime")
    findings.equal(
        dig(report, "schema_version"), PROBE_SCHEMA_VERSION, "report schema_version"
    )
    findings.equal(dig(report, "tool"), PROBE_PATH, "report tool")
    findings.equal(
        dig(report, "classification"), PROBE_CLASSIFICATION, "report classification"
    )
    findings.equal(
        dig(report, "stages_requested"), [SMOKE_CONSUMER], "report stages_requested"
    )
    findings.equal(dig(report, "sample"), None, "report sample (a census has none)")
    findings.equal(dig(report, "source_export"), None, "report source_export")
    findings.equal(
        dig(report, "references"),
        {
            "paths": dict.fromkeys(PROBE_REFERENCE_KEYS),
            "smoke": False,
            "validation": False,
            "build_manifest": False,
            "calibration_diagnostics": False,
            "qrf_tail": False,
        },
        "report references",
    )
    findings.require(
        not isinstance(report, dict) or "reference_plan_comparison" not in report,
        "the report compared plans against a reference release",
    )
    stages = dig(report, "stages")
    if findings.require(isinstance(stages, dict), "the report has no stages"):
        findings.equal(sorted(stages), sorted(PROBE_STAGES), "report stages")
    findings.equal(
        dig(report, "summary"),
        {
            "probe_failures": [],
            "authoritative_failures": [],
            "informational_failures": [],
            "stage_status": dict.fromkeys(PROBE_STAGES, "completed"),
        },
        "report summary",
    )
    for name in PROBE_STAGES:
        findings.equal(
            dig(stages, name, "status"), "completed", f"report stage {name} status"
        )
    findings.equal(dig(stages, "identify", "sha256"), dataset_sha, "identify sha256")
    findings.equal(dig(stages, "load", "sha256"), dataset_sha, "load sha256")
    n_households = dig(scoring, "n_households")
    batch_size = dig(scoring, "maximum_batch_size")
    findings.equal(
        dig(stages, "load", "rows", "household"), n_households, "load household rows"
    )
    design = dig(stages, "design")
    findings.equal(dig(design, "census"), True, "design census")
    findings.equal(dig(design, "problems"), [], "design problems")
    findings.equal(
        dig(design, "verified_against_receipt"), True, "design verified_against_receipt"
    )
    findings.equal(dig(design, "strata"), 0, "design strata")
    findings.equal(dig(design, "batch_size"), batch_size, "design batch_size")
    findings.equal(
        dig(design, "n_batches"), dig(scoring, "n_batches"), "design n_batches"
    )
    opened = dig(stages, "open_scorer")
    for field in ("n_households", "n_batches", "max_batch_households"):
        findings.equal(dig(opened, field), dig(scoring, field), f"open_scorer {field}")
    findings.equal(
        dig(opened, "maximum_batch_size"), batch_size, "open_scorer maximum_batch_size"
    )
    smoke = dig(stages, SMOKE_CONSUMER)
    findings.equal(dig(smoke, "passed"), True, "smoke stage passed")
    findings.equal(dig(smoke, "failures"), [], "smoke stage failures")
    findings.equal(
        dig(smoke, "analysis_setup_error"), None, "smoke stage analysis_setup_error"
    )
    findings.equal(dig(smoke, "reference_error"), None, "smoke stage reference_error")
    findings.equal(
        dig(smoke, "baseline_keys"),
        len(expectation["baseline_plan"]["keys"]),
        "smoke stage baseline_keys",
    )
    findings.equal(dig(smoke, "reform_passes"), n_probes, "smoke stage reform_passes")
    findings.equal(dig(smoke, "reform_systems"), n_probes, "smoke stage reform_systems")
    findings.equal(
        dig(stages, "post_export_scoring_plans", "compared"),
        False,
        "post_export_scoring_plans compared",
    )
    findings.equal(
        dig(report, "verdicts"),
        [
            {
                "stage": SMOKE_CONSUMER,
                "check": f"probe {row['id']}",
                "verdict": "pass",
                "authority": "authoritative",
                "reason": CENSUS_REASON,
                "release_consequence": "raises",
            }
            for row in probes
        ],
        "report verdicts",
    )
    findings.equal(
        dig(report, "post_export_scoring"),
        {
            "method": SCORING_METHOD,
            "dataset_sha256": dataset_sha,
            "maximum_batch_size": batch_size,
            "n_households": n_households,
            "n_batches": dig(scoring, "n_batches"),
            "max_batch_households": dig(scoring, "max_batch_households"),
            "spm": dig(scoring, "spm"),
            "consumers": {
                SMOKE_CONSUMER: {
                    "baseline_plan": expectation["baseline_plan"],
                    "baseline_passes": 1,
                    "reform_passes": n_probes,
                    "reform_systems": n_probes,
                }
            },
        },
        "report post_export_scoring",
    )
    findings.equal(
        dig(report, "scored_dataset_sha256"), dataset_sha, "report scored digest"
    )
    source = dig(report, "tool_source")
    for field, expected in (
        ("commit", commit),
        ("dirty", False),
        ("changes_sha256", None),
        ("commit_at_write", commit),
        ("dirty_at_write", False),
        ("changes_sha256_at_write", None),
        ("moved_since_load", False),
        ("sha256", sources[PROBE_PATH]),
    ):
        findings.equal(dig(source, field), expected, f"report tool_source.{field}")
    findings.equal(
        dig(source, "tools"),
        {
            path: {
                "sha256": sources[path],
                "file": str(repo / path),
                "hashed": "at load",
            }
            for path in (BUILDER_PATH, SAMPLER_PATH)
        },
        "report tool_source.tools",
    )
    findings.equal(
        dig(source, "installed_distributions", "unreadable"),
        0,
        "report installed_distributions.unreadable",
    )
    if isinstance(runtime, dict):
        findings.equal(
            dig(report, "packages"),
            {name: runtime.get(name, MISSING) for name in PROBE_PACKAGES},
            "report packages (the build runtime's)",
        )
    else:
        findings.fail("the build manifest records no runtime")
    findings.equal(
        dig(report, "host", "python"), dig(runtime, "python"), "report host.python"
    )


def _check_smoke_rows(
    findings: Findings,
    report: Any,
    probes: Sequence[Mapping[str, Any]],
    totals: Mapping[str, tuple[float, float, float]],
) -> None:
    rows = dig(report, "stages", SMOKE_CONSUMER, "probes")
    if not findings.require(isinstance(rows, list), "the smoke stage lists no probes"):
        return
    findings.equal(len(rows), len(probes), "smoke stage probe rows")
    for row, probe in zip(rows, probes, strict=False):
        where = f"smoke row {probe['id']}"
        findings.equal(dig(row, "probe"), probe["id"], f"{where} probe")
        for field in ("period", "budget_measure", "min_abs_effect", "expected_sign"):
            findings.equal(dig(row, field), probe[field], f"{where} {field}")
        findings.equal(dig(row, "passed"), True, f"{where} passed")
        findings.equal(dig(row, "authority"), "authoritative", f"{where} authority")
        findings.equal(
            dig(row, "authority_reason"), CENSUS_REASON, f"{where} authority_reason"
        )
        if probe["id"] not in totals:
            continue
        baseline_total, reform_total, effect = totals[probe["id"]]
        magnitude = gate_signed_magnitude(effect, probe["expected_sign"])
        for field, expected in (
            ("baseline_total", baseline_total),
            ("reform_total", reform_total),
            ("effect", effect),
            ("signed_magnitude", magnitude),
            ("margin_to_floor", magnitude - probe["min_abs_effect"]),
        ):
            findings.equal(dig(row, field), expected, f"{where} {field}")


def _effective_threads(record: Any) -> dict[str, Any] | _Missing:
    """The thread-pool sizes the release tool runs with.

    The tool sets each of :data:`THREAD_VARIABLES` with ``setdefault`` to
    ``str(min(os.cpu_count() or 1, 16))``, so a variable the environment sets
    (even to an empty string) keeps its value. :data:`MISSING` when the record
    lacks its environment or a CPU count ``os.cpu_count()`` could return.
    """
    environment = dig(record, "environment")
    cpus = dig(record, "cpu_count")
    if not isinstance(environment, dict) or not (
        cpus is None
        or (isinstance(cpus, int) and not isinstance(cpus, bool) and cpus >= 0)
    ):
        return MISSING
    default = str(min(cpus or 1, THREAD_POOL_CEILING))
    return {
        name: default if environment.get(name) is None else environment[name]
        for name in THREAD_VARIABLES
    }


def _check_launches(
    findings: Findings,
    launches: Mapping[str, Any],
    *,
    build: Any,
    report: Any,
    repo: Path,
    commit: Any,
    dataset_path: Path,
    dataset_sha: str,
    release_dir: Path,
    artifact_root: Path,
    probe_dir: Path,
    now: Callable[[], datetime],
) -> str:
    names = ("build-start", "build-end", "smoke-start", "smoke-end")
    records = [launches[name] for name in names]
    for name, record in zip(names, records, strict=True):
        stage, phase = name.split("-")
        findings.equal(
            dig(record, "schema_version"),
            LAUNCH_SCHEMA_VERSION,
            f"{name} schema_version",
        )
        findings.equal(dig(record, "tool"), TOOL, f"{name} tool")
        findings.equal(dig(record, "stage"), stage, f"{name} stage")
        findings.equal(dig(record, "phase"), phase, f"{name} phase")
        findings.equal(dig(record, "repo", "toplevel"), str(repo), f"{name} repository")
        findings.equal(dig(record, "repo", "head"), commit, f"{name} HEAD")
        findings.equal(dig(record, "repo", "clean"), True, f"{name} clean tree")
        findings.equal(
            dig(record, "repo", "status_sha256"),
            EMPTY_STATUS_SHA256,
            f"{name} git status digest (an empty status)",
        )
        findings.equal(
            dig(record, "repo", "status_entries"), 0, f"{name} git status entries"
        )
        findings.equal(dig(record, "cwd"), str(repo), f"{name} working directory")
        findings.equal(dig(record, "cwd_shadows"), [], f"{name} shadowing resources")
    build_start, build_end, smoke_start, smoke_end = records
    run_id = dig(build_start, "run_id")
    findings.require(isinstance(run_id, str) and run_id, "build-start names no run_id")
    scoring = dig(build, "post_export_scoring")

    # A start record describes the process it became: its command, measured
    # with its own environment, which sets no variable that points git away.
    for name, record in (("build-start", build_start), ("smoke-start", smoke_start)):
        stage = name.split("-")[0]
        findings.equal(dig(record, "exit_code"), None, f"{name} exit_code")
        findings.equal(
            dig(record, "runtime_environment"),
            dig(record, "environment"),
            f"{name} runtime_environment (its own environment)",
        )
        git_names = dig(record, "git_environment")
        if findings.require(
            isinstance(git_names, list)
            and all(isinstance(item, str) for item in git_names),
            f"{name} lists no git environment",
        ):
            findings.equal(
                sorted(set(git_names) & GIT_REDIRECTING_VARIABLES),
                [],
                f"{name} git-redirecting variables",
            )
        command = dig(record, "command")
        python = dig(record, "python", "requested")
        out = dig(record, "out_dir", "path")
        dataset = dig(record, "dataset", "path")
        if not findings.require(
            isinstance(command, list)
            and all(isinstance(item, str) for item in command)
            and isinstance(python, str)
            and isinstance(out, str),
            f"{name} records no command, interpreter or output directory",
        ):
            continue
        for problem in _command_problems(
            stage,
            command,
            python=python,
            cwd=repo,
            dataset=Path(dataset) if isinstance(dataset, str) else None,
            out_dir=Path(out),
        ):
            findings.fail(f"{name}: {problem}")
        if stage == "smoke":
            values, _ = _smoke_arguments(command[_script_index(command) + 1 :])
            batch_size = dig(scoring, "maximum_batch_size")
            findings.equal(
                values.get("--batch-size"),
                [str(batch_size)],
                "smoke-start --batch-size (the build's maximum_batch_size)",
            )

    # An end record repeats its start's snapshot after the stage exits,
    # measuring the runtime with the start's environment.
    for start_name, start, end_name, end in (
        ("build-start", build_start, "build-end", build_end),
        ("smoke-start", smoke_start, "smoke-end", smoke_end),
    ):
        for field in ("run_id", "tool_sha256", "files", "runtime"):
            findings.equal(
                dig(end, field),
                dig(start, field),
                f"{end_name} {field} ({start_name}'s)",
            )
        findings.equal(
            dig(end, "python", "resolved"),
            dig(start, "python", "resolved"),
            f"{end_name} interpreter ({start_name}'s)",
        )
        findings.equal(
            dig(end, "runtime_environment"),
            dig(start, "environment"),
            f"{end_name} runtime_environment ({start_name}'s environment)",
        )
        findings.equal(dig(end, "command"), None, f"{end_name} command")
        findings.equal(dig(end, "out_dir"), None, f"{end_name} out_dir")
    findings.equal(dig(smoke_end, "exit_code"), 0, "the smoke's exit code")
    # The smoke against the build: run, commit, lockfiles, tool, interpreter,
    # runtime, environment, the build's exit code and the scored dataset.
    _check_smoke_launch(findings, smoke_start, build_start, build_end)

    findings.require(
        isinstance(dig(build_start, "files", "uv.lock"), str),
        "build-start found no uv.lock",
    )
    runtime = dig(build_start, "runtime")
    findings.equal(
        dig(runtime, "runtime_versions"),
        dig(build, "runtime"),
        "the launch runtime (the build manifest's runtime)",
    )
    findings.equal(
        dig(runtime, "installed_distributions"),
        dig(report, "tool_source", "installed_distributions"),
        "the launch inventory (the report's)",
    )
    findings.equal(
        dig(runtime, "version"), dig(report, "host", "python"), "launch python"
    )
    # The build records the OMP setting its calibration ran with: it must be
    # the value build-start's environment gives the release tool.
    findings.equal(
        dig(build, "calibration_runtime", "omp_num_threads"),
        dig(_effective_threads(build_start), "OMP_NUM_THREADS"),
        "the build's calibration_runtime.omp_num_threads (build-start's effective "
        "OMP_NUM_THREADS)",
    )
    origins = dig(runtime, "import_origins")
    for module, source in MICROCOSM_SOURCES.items():
        where = dig(origins, module)
        root = (repo / source).resolve()
        locations = dig(where, "locations")
        paths = [
            dig(where, "origin"),
            *(locations if isinstance(locations, list) else []),
        ]
        paths = [path for path in paths if isinstance(path, str)]
        findings.require(
            bool(paths)
            and all(Path(path).resolve().is_relative_to(root) for path in paths),
            f"{module} imports from outside {root}: {_show(where)}",
        )

    # Outputs.
    build_out = dig(build_start, "out_dir", "path")
    findings.require(
        dig(build_start, "out_dir", "state") in ("absent", "empty"),
        "the build did not start in a new, empty output directory",
    )
    if isinstance(build_out, str):
        for path, what in (
            (release_dir, "release directory"),
            (artifact_root, "artifact root"),
        ):
            findings.require(
                path.is_relative_to(Path(build_out)),
                f"the {what} {path} is not inside the build's output {build_out}",
            )
    findings.equal(dig(smoke_start, "out_dir", "path"), str(probe_dir), "smoke output")
    findings.require(
        dig(smoke_start, "out_dir", "state") in ("absent", "empty"),
        "the smoke did not start in a new, empty output directory",
    )
    # The scored bytes: the build ended with the release's national H5; the
    # smoke read the same path from start to end and the report names it.
    build_dataset = dig(build_start, "dataset")
    if build_dataset is not None:
        findings.equal(
            dig(build_dataset, "path"),
            dig(build_end, "dataset", "path"),
            "build-start dataset path (build-end's)",
        )
        findings.equal(
            dig(build_dataset, "exists"),
            False,
            "build-start dataset exists (it must not before the build)",
        )
    findings.equal(
        dig(build_end, "dataset", "path"),
        str(dataset_path),
        "build-end dataset path (the release's national artifact)",
    )
    findings.equal(
        dig(build_end, "dataset", "sha256"), dataset_sha, "build-end dataset"
    )
    smoke_path = dig(smoke_start, "dataset", "path")
    for name, record in (("smoke-start", smoke_start), ("smoke-end", smoke_end)):
        findings.equal(dig(record, "dataset", "sha256"), dataset_sha, f"{name} dataset")
        findings.equal(
            dig(record, "dataset", "path"), smoke_path, f"{name} dataset path"
        )
    findings.equal(dig(report, "export"), smoke_path, "the report's export path")
    # Order: build start < manifests < build end <= smoke start <= report <= smoke end.
    moments = {
        "build-start": _parse_time(dig(build_start, "recorded_at")),
        "manifests": _parse_time(dig(build, "created_at")),
        "build-end": _parse_time(dig(build_end, "recorded_at")),
        "smoke-start": _parse_time(dig(smoke_start, "recorded_at")),
        "report": _parse_time(dig(report, "created_at")),
        "smoke-end": _parse_time(dig(smoke_end, "recorded_at")),
    }
    missing = [name for name, moment in moments.items() if moment is None]
    if missing:
        findings.fail(f"no timezone-aware time for {missing}")
        return run_id if isinstance(run_id, str) else ""
    # The report's time has whole seconds; compare the smoke start at that grain.
    smoke_start_second = moments["smoke-start"].replace(microsecond=0)
    for earlier, later, ok in (
        ("build-start", "manifests", moments["build-start"] < moments["manifests"]),
        ("manifests", "build-end", moments["manifests"] < moments["build-end"]),
        ("build-end", "smoke-start", moments["build-end"] <= moments["smoke-start"]),
        ("smoke-start", "report", smoke_start_second <= moments["report"]),
        ("report", "smoke-end", moments["report"] <= moments["smoke-end"]),
        ("smoke-end", "now", moments["smoke-end"] <= now()),
    ):
        findings.require(ok, f"{earlier} does not precede {later}")
    return run_id if isinstance(run_id, str) else ""


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------


def note_text(accepted: Accepted) -> str:
    probes = accepted.smoke_summary["probes"]
    return (
        f"# Census reform-coverage smoke for {accepted.release_id}\n\n"
        "The build skipped its own reform-coverage smoke and its reform "
        "validation. Both manifests record `gate_evidence.reform_coverage_smoke` "
        'as `"skipped"`.\n\n'
        "Before publication, a separate full-population smoke "
        "(`tools/probe_us_post_export.py --census --stages reform_coverage_smoke`) "
        f"scored the H5 both manifests pin (sha256 `{accepted.dataset_sha256}`). "
        "It ran from the build's worktree at the build's commit "
        f"(`{accepted.commit}`), clean at each stage's start and end, with the "
        "build's interpreter, installed distributions (by name and version) and "
        f"thread-pool settings. All {probes} probes passed. The smoke wrote the "
        "report, gate file and "
        f"pass log that this tag carries as `{ATTACHMENTS['report']}`, "
        f"`{ATTACHMENTS['gate']}` and `{ATTACHMENTS['passes']}`. "
        f"`{ATTACHMENTS['receipt']}` records the checks that tie them to this "
        "build. The gate file's `enforced: true` describes the separate smoke, "
        "not the build.\n\n"
        "Reform validation never ran, and none of these files certifies it. The "
        "check accepted the smoke only for a tag-only publish (`--no-latest "
        "--tag-only`), not for promotion to the default release. The publisher "
        "uploads this note and the receipt but checks nothing in them.\n"
    )


def receipt(
    accepted: Accepted,
    inputs: Inputs,
    written: Mapping[str, bytes],
    *,
    checked_at: datetime,
) -> dict[str, Any]:
    def entry(data: bytes) -> dict[str, Any]:
        return {"sha256": sha256_bytes(data), "bytes": len(data)}

    runtime = accepted.launches["build-start"]["runtime"]
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "tool": TOOL,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "scope": SCOPE,
        "accepted": True,
        "checked_at": checked_at.isoformat(),
        "release_id": accepted.release_id,
        "tag": accepted.release_id,
        "build_commit": accepted.commit,
        "run_id": accepted.run_id,
        "authorization": inputs.authorization,
        "required_probes": inputs.probe_count,
        "operator_attestations": {"dedicated_worktree": inputs.dedicated_worktree},
        "dataset": {
            "key": accepted.dataset_key,
            "path": str(accepted.dataset_path),
            "sha256": accepted.dataset_sha256,
            "bytes": accepted.dataset_bytes,
        },
        "files": {
            "build_manifest.json": entry(accepted.files["build_manifest.json"]),
            "release_manifest.json": entry(accepted.files["release_manifest.json"]),
            **{name: entry(data) for name, data in written.items()},
        },
        "sources": {
            "probe_report": str(inputs.probe_dir.resolve() / PROBE_REPORT),
            "probe_gate": str(inputs.probe_dir.resolve() / PROBE_GATE),
            "probe_passes": str(inputs.probe_dir.resolve() / PROBE_PASSES),
        },
        "runtime": runtime,
        "launches": accepted.launches,
        "expectation": accepted.expectation,
        "smoke": accepted.smoke_summary,
        "status": {
            "build_smoke": "skipped",
            "external_smoke": "passed-before-publish",
            "reform_validation": "not-run-not-certified",
        },
        "publication": {
            # What the check printed; the publisher uploads these files but
            # checks nothing in them, so only the flags the operator runs keep
            # the publish off latest.
            "printed_flags": ["--no-latest", "--tag-only"],
            "publisher_checks_these_files": False,
            "publisher_preflight": "the operator runs it next; this check does not",
            "extra_files": list(ATTACHMENTS.values()),
        },
        "limits": list(LIMITS),
    }


def write_attachments(
    accepted: Accepted,
    inputs: Inputs,
    *,
    now: Callable[[], datetime] = _utc_now,
) -> dict[str, Path]:
    """Write the attachments, receipt last; remove them all if any write fails."""
    release_dir = inputs.release_dir.resolve()
    contents = {
        ATTACHMENTS["report"]: accepted.files["report"],
        ATTACHMENTS["gate"]: accepted.files["gate"],
        ATTACHMENTS["passes"]: accepted.files["passes"],
        ATTACHMENTS["note"]: note_text(accepted).encode("utf-8"),
    }
    written: list[Path] = []
    try:
        for filename, data in contents.items():
            path = release_dir / filename
            _write_new(path, data)
            written.append(path)
        payload = receipt(accepted, inputs, contents, checked_at=now())
        path = release_dir / ATTACHMENTS["receipt"]
        _write_new(path, strict_dumps(payload).encode("utf-8"))
        written.append(path)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return {path.name: path for path in written}


def publish_commands(accepted: Accepted, inputs: Inputs) -> tuple[str, str]:
    extras = " ".join(f"--extra-file {name}" for name in ATTACHMENTS.values())
    base = (
        f"tools/publish_release.sh {inputs.release_dir.resolve()} "
        f"--artifact-root {inputs.artifact_root.resolve()} --no-latest --tag-only "
        f"{extras}"
    )
    return f"{base} --preflight-only", f"{base} --repo-id {accepted.repo_id}"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _default_repo() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    expect = commands.add_parser("expect", help="print the expectation for a commit")
    expect.add_argument("--commit", required=True, help="the build's full commit id")
    expect.add_argument("--repo", type=Path, default=_default_repo())
    expect.add_argument("--out", type=Path, default=None, help="write it here")

    record = commands.add_parser("launch-record", help="record a stage boundary")
    record.add_argument("--stage", choices=("build", "smoke"), required=True)
    record.add_argument("--phase", choices=("start", "end"), required=True)
    record.add_argument("--run-id", required=True)
    record.add_argument("--launch-dir", type=Path, required=True)
    record.add_argument("--python", required=True, help="the stage's interpreter")
    record.add_argument("--repo", type=Path, default=_default_repo())
    record.add_argument("--dataset", type=Path, default=None, help="the H5")
    record.add_argument("--out-dir", type=Path, default=None)
    record.add_argument("--exit-code", type=int, default=None)
    record.epilog = (
        "A start record takes the stage's command after --, and becomes it once "
        "the record is written."
    )

    run = commands.add_parser("check", help="accept the smoke, or say why not")
    run.add_argument("--release-dir", type=Path, required=True)
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument("--probe-dir", type=Path, required=True)
    run.add_argument("--launch-dir", type=Path, required=True)
    run.add_argument("--repo", type=Path, default=_default_repo())
    run.add_argument(
        "--authorization",
        required=True,
        help="the ruling that allows this publish, recorded in the receipt",
    )
    run.add_argument(
        "--probe-count",
        type=int,
        required=True,
        help="how many probes the ruling requires, every one present and passing",
    )
    run.add_argument(
        "--attest-dedicated-worktree",
        action="store_true",
        help="attest that nothing but this run used the worktree",
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    stage_command: list[str] = []
    if "--" in arguments:
        split = arguments.index("--")
        arguments, stage_command = arguments[:split], arguments[split + 1 :]
    args = parser.parse_args(arguments)
    if stage_command and args.command != "launch-record":
        parser.error("only launch-record takes a command after --")
    args.stage_command = stage_command
    try:
        if args.command == "expect":
            expectation = build_expectation(args.repo.resolve(), args.commit)
            text = strict_dumps(expectation)
            if args.out is None:
                sys.stdout.write(text)
            else:
                if args.out.exists():
                    raise RefusalError(f"{args.out} already exists")
                args.out.parent.mkdir(parents=True, exist_ok=True)
                _write_new(args.out, text.encode("utf-8"))
                print(f"wrote {args.out}")
            return 0
        if args.command == "launch-record":
            return launch_record_main(args)
        inputs = Inputs(
            release_dir=args.release_dir,
            artifact_root=args.artifact_root,
            probe_dir=args.probe_dir,
            launch_dir=args.launch_dir,
            repo=args.repo,
            authorization=args.authorization,
            probe_count=args.probe_count,
            dedicated_worktree=args.attest_dedicated_worktree,
        )
        accepted = check(inputs)
        written = write_attachments(accepted, inputs)
    except RefusalError as error:
        print(f"refused:\n{error}", file=sys.stderr)
        return 1
    preflight, publish = publish_commands(accepted, inputs)
    print(f"accepted {accepted.release_id}; wrote {', '.join(written)}")
    print("Next, the publisher's preflight:")
    print(f"  {preflight}")
    print("Publication is a separate, deliberate step after the preflight passes:")
    print(f"  {publish}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
