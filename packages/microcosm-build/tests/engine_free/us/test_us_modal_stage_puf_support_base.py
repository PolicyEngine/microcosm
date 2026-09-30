"""Route A's base stage on the US Modal stage runner.

``us-puf-support-base`` registers ``tools/build_us_puf_support_base.py`` so
that one plan (``docs/us-modal-stage-route-a-base-plan.json``) runs Route A's
base on Modal. The argv it builds must equal the command Route A's local
driver built for commit 4b57d15a287c (``base-config.json``) flag for flag and
value for value, differing only in file paths. These tests hold that with a
differential test against a tokenized copy of that command, a second one
through the pinned tool's own parser and per-stage child commands (loaded
from 4b57d15a2's source; skipped, visibly, in a clone without that commit),
Hypothesis properties of the argv builder, the plan refusals (an owned
flag by name, by an abbreviation argparse would expand, or with '='), the
budget stop of a multi-process tool, the home-cache seed, the pinned tree
file, the disk guard, the cost ceiling, the CPU limit and the headroom a
check shares, the refusal of unfinished receipts, the comparison with the
local run (its command, inputs and locked run config, derived from the
Modal argv through the pinned tool's own _stage_run_config, its hash seed,
and its checkpoints, whose scope is two of 24 outer stages), the push
order, the upload commands, the write probe, and the runbook's hand-off
block, run as written. Nothing here needs Modal, a country engine or the
network.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from contextlib import suppress
from pathlib import Path, PurePosixPath

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from test_support.microcosm_build.us_modal_stage_plan_tool import (
    ROOT,
    ROUTE_A_BASE_COMMAND,
    base_plan_data,
    load_app,
    plan_lib,
)

BASE = plan_lib.US_PUF_SUPPORT_BASE
BASE_TOOL = ROOT / BASE.script
SEED = plan_lib.ASEC_2023_ARCHIVE_SEED
MAPPING_FLAGS = ("--asec-h5", "--asec-h5-sha256", "--asec-education-source")

# route_a.sh input_rows(): role, file name and sha256 of each base input, as
# the driver resolved and hashed them on the build machine on 2026-09-23.
ROUTE_A_BASE_INPUTS = {
    "asec_2024_h5": (
        "census_cps_2024.h5",
        "ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d",
    ),
    "asec_2023_h5": (
        "census_cps_2023.h5",
        "cb57817327799f42b741caed5f9be94d04021c2e6809c1ad7bd0686da5428d88",
    ),
    "asec_2022_h5": (
        "census_cps_2022.h5",
        "7ccca976284bb47815d84460cc4f75a0a65d26d7754ab0a0f417de351b3d474e",
    ),
    "puf_2024_h5": (
        "puf_2024.h5",
        "7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df",
    ),
    "puf_2015_csv": (
        "puf_2015.csv",
        "0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df",
    ),
    "acs_2022_h5": (
        "acs_2022.h5",
        "0b319b496f19a6913066f9c5ea572edfda3d78a187be6f375846617d0b441bd4",
    ),
    "asec_education_2022_zip": (
        "asecpub23csv.zip",
        "d2e000250782adfbdd7f29c82b66d866591a30f0d330496698ec19f9c784ce11",
    ),
    "asec_education_2023_zip": (
        "asecpub24csv.zip",
        "cdb39cdac34bef99dd0940ab28e306f692404c2eea44d85dfd634214872a0a09",
    ),
    "asec_education_2024_zip": (
        "asecpub25csv.zip",
        "318845a2b5e0034eb2973898de1738f4df0025727de38499e7669cb9c0deef0b",
    ),
    "base_ledger_facts": (
        "consumer_facts_us_c5e5bf8.jsonl",
        "b85437390021777e746f507c5890305496baf5fc7f2c78ba08ddb090f4839801",
    ),
    "block_ladder_npz": (
        "us_block_ladder_2020.npz",
        "7ba39b959068181b56a7ce1dc8f2477b31af33f28d0a1803ff6cba84cdfab06b",
    ),
}


@pytest.fixture
def app(monkeypatch):
    return load_app(monkeypatch)


def _committed_plan() -> plan_lib.Plan:
    return plan_lib.parse_plan(base_plan_data())


def _fixture() -> dict:
    return json.loads(ROUTE_A_BASE_COMMAND.read_text())


def _flags(argv: list[str]) -> list[str]:
    return [item for item in argv if item.startswith("--")]


def _values(argv: list[str]) -> list[str]:
    """Every argument value after the script, with YEAR=VALUE split to VALUE."""

    values = []
    for index, item in enumerate(argv[3:], start=3):
        if item.startswith("--"):
            continue
        if argv[index - 1] in MAPPING_FLAGS:
            item = item.partition("=")[2]
        values.append(item)
    return values


def _mapping(argv: list[str], flag: str) -> list[tuple[int, str]]:
    return [
        (int(argv[i + 1].partition("=")[0]), argv[i + 1].partition("=")[2])
        for i, item in enumerate(argv)
        if item == flag
    ]


def _option(argv: list[str], flag: str) -> str:
    (index,) = [i for i, item in enumerate(argv) if item == flag]
    return argv[index + 1]


# --------------------------------------------------------------------------- #
# (a) Differential: the committed plan against Route A's local command        #
# --------------------------------------------------------------------------- #


def _pinned_tool_source() -> str | None:
    """The base tool at the committed plan's commit, or None outside its history.

    CI checks out one commit (actions/checkout's default depth 1), so there
    the pinned commit is absent and every test that needs it skips, visibly.
    """

    shown = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            f"{_committed_plan().commit}:{BASE.script}",
        ],
        capture_output=True,
        text=True,
    )
    return shown.stdout if shown.returncode == 0 else None


def _tool_flags(source: str) -> set[str]:
    return set(re.findall(r'add_argument\(\s*"(--[A-Za-z0-9-]+)"', source))


_NO_PINNED_COMMIT = (
    "the committed plan's commit is not in this clone's history (CI's depth-1 "
    "checkout); the pinned-tool checks run in a clone that has it"
)


@pytest.fixture(scope="module")
def pinned_source() -> str:
    source = _pinned_tool_source()
    if source is None:
        pytest.skip(_NO_PINNED_COMMIT)
    return source


@pytest.fixture(scope="module")
def pinned_tool(pinned_source: str, tmp_path_factory):
    """The base tool as the plan's commit has it, loaded from that source.

    Not this tree's copy: the plan runs 4b57d15a2's tool whatever main does
    to tools/build_us_puf_support_base.py later. Importing it is engine-free.
    """

    path = tmp_path_factory.mktemp("pinned") / "build_us_puf_support_base.py"
    path.write_text(pinned_source)
    spec = importlib.util.spec_from_file_location("route_a_base_tool_pinned", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _local_argv() -> list[str]:
    """The fixture's argv with its tokens replaced by stand-in local paths."""

    def local(value: str) -> str:
        value = re.sub(r"<input:([a-z0-9_]+)>", r"/local/inputs/\1", value)
        return value.replace("<state>", "/local/run").replace("<repo>", "/local/wt")

    return ["/local/python", *(local(value) for value in _fixture()["argv"][1:])]


def _paths_masked(value: object) -> object:
    """A parsed value with every path replaced by a marker, YEAR= kept."""

    if isinstance(value, Path):
        return "<path>"
    if isinstance(value, list):
        return [_paths_masked(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return "<path>"
    if isinstance(value, str) and re.fullmatch(r"\d{4}=/.*", value):
        return value.split("=", 1)[0] + "=<path>"
    return value


def test_committed_plan_argv_equals_route_a_base_config() -> None:
    fixture = _fixture()
    plan = _committed_plan()
    argv = plan_lib.planned_argv(plan)
    # The tokenizer compare-lineage applies to a receipt's argv.
    tokens = plan_lib.tokenize_stage_argv(argv)
    assert tokens == fixture["argv"]
    # Every path became a token: none of the container's paths is left.
    assert not [value for value in tokens if "/work/" in value or "/opt/" in value]
    assert {
        match
        for value in tokens
        for match in re.findall(r"<input:([a-z0-9_]+)>", value)
    } == set(plan.inputs)
    # The one environment variable the local run set, and nothing else.
    assert dict(plan.env) == fixture["env"] == {"PYTHONUNBUFFERED": "1"}
    # The build commit the local driver recorded is the plan's commit.
    assert plan.commit == fixture["build_commit"]
    assert plan.branch == "main"


def test_route_a_command_uses_exactly_the_builder_flags() -> None:
    flags = _flags(_fixture()["argv"])
    assert set(flags) == plan_lib.PUF_SUPPORT_BASE_BUILDER_FLAGS
    assert set(flags) <= BASE.owned_flags
    assert not set(flags) & plan_lib.PUF_SUPPORT_BASE_WITHHELD_FLAGS


def test_the_tool_s_own_parser_reads_the_same_build_from_both_commands(
    pinned_tool,
) -> None:
    # Differential through the pinned tool itself (4b57d15a2's source, not
    # this tree's): its _parse_args reads the Modal argv and Route A's local
    # argv into the same settings (paths aside), and the per-stage child
    # commands that --stage all spawns (_stage_cli_args) match stage for
    # stage.
    tool = pinned_tool
    modal_argv = plan_lib.planned_argv(_committed_plan())
    local_argv = _local_argv()
    assert modal_argv[1:3] == local_argv[1:3] == ["-B", BASE.script]
    modal_args = tool._parse_args(modal_argv[3:])
    local_args = tool._parse_args(local_argv[3:])
    modal_settings = {k: _paths_masked(v) for k, v in vars(modal_args).items()}
    local_settings = {k: _paths_masked(v) for k, v in vars(local_args).items()}
    assert modal_settings == local_settings
    # Route A's scalar settings, and every escape hatch left at its default.
    assert {
        key: modal_settings[key]
        for key in (
            "stage",
            "target_year",
            "seed",
            "n_estimators",
            "assign_congressional_districts",
            "congressional_district_seed",
            "geography_ladder_seed",
            "base_h5",
            "support_spine_spec",
            "asec_max_households",
            "asec_2023_weeks_unemployed_source",
            "without_block_ladder",
            "allow_geography_ladder_gate_failures",
            "equivalence_boundary_dir",
            "equivalence_deterministic_h5_metadata",
        )
    } == {
        "stage": "all",
        "target_year": 2024,
        "seed": 0,
        "n_estimators": 32,
        "assign_congressional_districts": True,
        "congressional_district_seed": 0,
        "geography_ladder_seed": 0,
        "base_h5": None,
        "support_spine_spec": None,
        "asec_max_households": None,
        "asec_2023_weeks_unemployed_source": None,
        "without_block_ladder": False,
        "allow_geography_ladder_gate_failures": False,
        "equivalence_boundary_dir": None,
        "equivalence_deterministic_h5_metadata": False,
    }
    for stage in tool.PIPELINE_STEPS:
        modal_child = tool._stage_cli_args(modal_args, stage)
        local_child = tool._stage_cli_args(local_args, stage)
        assert _paths_masked(modal_child) == _paths_masked(local_child), stage


def test_committed_plan_pins_route_a_s_eleven_inputs() -> None:
    plan = _committed_plan()
    assert set(plan.inputs) == set(ROUTE_A_BASE_INPUTS) == set(BASE.inputs)
    assert len(plan.inputs) == 11
    for role, (name, sha256) in ROUTE_A_BASE_INPUTS.items():
        ref = plan.inputs[role]
        assert (ref.kind, ref.filename, ref.sha256) == ("volume", name, sha256)
        assert ref.volume_path == plan_lib.cas_volume_path(sha256, name)
    assert (plan.run_id, plan.stage) == ("route-a-base-4b57d15a287c", "all")
    assert (plan.nonpreemptible, plan.max_wall_seconds) == (True, 14_400)
    assert plan.resources is plan_lib.BASE


def test_the_crosswalk_is_the_pinned_tree_s_file_not_an_input() -> None:
    plan = _committed_plan()
    argv = plan_lib.planned_argv(plan)
    crosswalk = _option(argv, "--congressional-district-vintage-crosswalk")
    assert crosswalk == (
        f"{plan_lib.IMAGE_REPO_ROOT}/{plan_lib.CD_VINTAGE_CROSSWALK.path}"
    )
    assert "crosswalk" not in " ".join(plan.inputs)
    # The registered path is a file of this tree too (its bytes moved on
    # main after 4b57d15a2, which the digest pins; see the next test).
    assert (ROOT / plan_lib.CD_VINTAGE_CROSSWALK.path).is_file()


def test_the_registered_crosswalk_digest_is_4b57d15a2_s(pinned_source: str) -> None:
    # pinned_source skips this test in a clone without the plan's commit.
    commit = _committed_plan().commit
    shown = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            f"{commit}:{plan_lib.CD_VINTAGE_CROSSWALK.path}",
        ],
        capture_output=True,
    )
    assert shown.returncode == 0, shown.stderr
    assert hashlib.sha256(shown.stdout).hexdigest() == (
        plan_lib.CD_VINTAGE_CROSSWALK.sha256
    )


# --------------------------------------------------------------------------- #
# (b) Properties of the argv builder                                           #
# --------------------------------------------------------------------------- #

_FILE_NAME = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9._-]{0,24}", fullmatch=True)
_DIGEST = st.binary(min_size=1, max_size=32).map(
    lambda raw: hashlib.sha256(raw).hexdigest()
)
_SEGMENT = st.from_regex(r"[a-z0-9][a-z0-9_-]{0,8}", fullmatch=True)
_ROOT_DIR = st.lists(_SEGMENT, min_size=1, max_size=4).map(
    lambda parts: "/" + "/".join(parts)
)


@st.composite
def _base_plans(draw) -> plan_lib.Plan:
    """Valid base plans with arbitrary digests and file names."""

    roles = plan_lib.PUF_SUPPORT_BASE_INPUTS
    digests = draw(st.lists(_DIGEST, min_size=len(roles), max_size=len(roles)))
    names = draw(st.lists(_FILE_NAME, min_size=len(roles), max_size=len(roles)))
    inputs = {}
    for role, sha256, name in zip(roles, digests, names, strict=True):
        if role == SEED.input:  # the tool pins the seeded archive
            sha256 = SEED.sha256
        inputs[role] = {
            "uri": f"volume://{plan_lib.cas_volume_path(sha256, name)}",
            "sha256": sha256,
        }
    return plan_lib.parse_plan(base_plan_data(inputs=inputs))


_PROPERTY_SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@_PROPERTY_SETTINGS
@given(plan=_base_plans(), work_root=_ROOT_DIR, state_root=_ROOT_DIR)
def test_builder_properties(
    plan: plan_lib.Plan, work_root: str, state_root: str
) -> None:
    paths = {
        name: plan_lib.input_local_path(work_root, ref)
        for name, ref in plan.inputs.items()
    }
    state = state_root + "/state"
    argv = plan_lib.build_stage_argv(
        plan, python="/venv/bin/python", input_paths=paths, state_dir=state
    )
    assert argv[:3] == ["/venv/bin/python", "-B", "tools/build_us_puf_support_base.py"]

    # Every flag it emits is runner-owned, and the flag sequence is Route A's
    # whatever the inputs are.
    flags = _flags(argv)
    assert set(flags) <= BASE.owned_flags
    assert flags == _flags(_fixture()["argv"])

    # Every input path appears exactly once, as a value or a YEAR=PATH value.
    values = collections.Counter(_values(argv))
    for name, path in paths.items():
        assert values[path] == 1, name
    # No value could be read as a flag.
    assert not any(value.startswith("-") for value in values)

    # The sha256 mappings are the plan's digests, year for year.
    assert _mapping(argv, "--asec-h5-sha256") == [
        (year, plan.inputs[f"asec_{year}_h5"].sha256)
        for year in plan_lib.PUF_SUPPORT_BASE_ASEC_YEARS
    ]
    # Each YEAR=PATH maps the year to that year's input, in Route A's order.
    assert _mapping(argv, "--asec-h5") == [
        (year, paths[f"asec_{year}_h5"])
        for year in plan_lib.PUF_SUPPORT_BASE_ASEC_YEARS
    ]
    assert _mapping(argv, "--asec-education-source") == [
        (year, paths[f"asec_education_{year}_zip"])
        for year in plan_lib.PUF_SUPPORT_BASE_EDUCATION_YEARS
    ]

    # Checkpoints and outputs stay under the mirrored state root, apart.
    checkpoints = PurePosixPath(_option(argv, "--checkpoint-dir"))
    out = PurePosixPath(_option(argv, "--out"))
    for path in (checkpoints, out):
        assert path.is_relative_to(state) and path != PurePosixPath(state)
    assert not checkpoints.is_relative_to(out)
    assert not out.is_relative_to(checkpoints)
    # The crosswalk is read from the pinned clone.
    assert PurePosixPath(
        _option(argv, "--congressional-district-vintage-crosswalk")
    ).is_relative_to(plan_lib.IMAGE_REPO_ROOT)

    # Deterministic.
    assert argv == plan_lib.build_stage_argv(
        plan, python="/venv/bin/python", input_paths=paths, state_dir=state
    )


@_PROPERTY_SETTINGS
@given(plan=_base_plans())
def test_planned_argv_never_depends_on_the_file_names_for_its_shape(
    plan: plan_lib.Plan,
) -> None:
    # Only paths differ between any valid plan and the committed one.
    committed = plan_lib.planned_argv(_committed_plan())
    argv = plan_lib.planned_argv(plan)
    assert len(argv) == len(committed)
    shape = [(a.startswith("--"), a if a.startswith("--") else None) for a in argv]
    assert shape == [
        (a.startswith("--"), a if a.startswith("--") else None) for a in committed
    ]


@_PROPERTY_SETTINGS
@given(work_root=_ROOT_DIR)
def test_the_tokenized_argv_is_route_a_s_wherever_the_work_root_is(
    work_root: str,
) -> None:
    # Differential, for any work root: the committed plan's argv, tokenized
    # as compare-lineage tokenizes a receipt's, is Route A's local command.
    argv = plan_lib.planned_argv(_committed_plan(), work_root=work_root)
    assert plan_lib.tokenize_stage_argv(argv, work_root) == _fixture()["argv"]


# Run-config values as the tool records them: plain scalars, digests, and
# paths drawn as (directory, file name) or (year, directory, file name).
_CONFIG_LEAF = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**6), max_value=10**6),
    _DIGEST,
    _FILE_NAME,
    st.tuples(_ROOT_DIR, _FILE_NAME),
    st.tuples(st.integers(min_value=1990, max_value=2100), _ROOT_DIR, _FILE_NAME),
)
_CONFIG_TREE = st.recursive(
    _CONFIG_LEAF,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(_SEGMENT, children, max_size=4),
    ),
    max_leaves=20,
)


def _render(tree: object, *, root: str | None = None, names_only: bool = False):
    """The tree with its path leaves written out (under ``root`` if given)."""

    if isinstance(tree, dict):
        return {
            k: _render(v, root=root, names_only=names_only) for k, v in tree.items()
        }
    if isinstance(tree, list):
        return [_render(v, root=root, names_only=names_only) for v in tree]
    if isinstance(tree, tuple):
        *year, directory, name = tree
        path = name if names_only else f"{root or directory}/{name}"
        return f"{year[0]}={path}" if year else path
    return tree


@_PROPERTY_SETTINGS
@given(tree=_CONFIG_TREE, other_root=_ROOT_DIR)
def test_masking_a_run_config_keeps_everything_but_the_directories(
    tree: object, other_root: str
) -> None:
    config = _render(tree)
    masked = plan_lib.mask_config_paths(config)
    # Each path becomes its file name, a YEAR= prefix kept; nothing else moves.
    assert masked == _render(tree, names_only=True)
    # So two machines' configs of one build (same files, other directories)
    # mask equal, and masking twice changes nothing.
    assert plan_lib.mask_config_paths(_render(tree, root=other_root)) == masked
    assert plan_lib.mask_config_paths(masked) == masked
    # No absolute path is left anywhere.
    assert "/" not in json.dumps(masked)


@_PROPERTY_SETTINGS
@given(
    role=st.sampled_from(plan_lib.PUF_SUPPORT_BASE_INPUTS),
    other=_DIGEST,
)
def test_any_cas_path_that_does_not_name_its_digest_is_refused(
    role: str, other: str
) -> None:
    data = base_plan_data()
    ref = data["inputs"][role]
    if other == ref["sha256"]:
        return
    name = ref["uri"].rsplit("/", 1)[1]
    ref["uri"] = f"volume://{plan_lib.cas_volume_path(other, name)}"
    with pytest.raises(plan_lib.PlanError, match="own digest"):
        plan_lib.parse_plan(data)


@settings(max_examples=200, deadline=None)
@given(
    first=st.floats(min_value=0, max_value=1e6, allow_nan=False),
    extra=st.floats(min_value=0, max_value=1e6, allow_nan=False),
)
def test_cost_estimate_is_nonnegative_monotone_and_tripled_off_preemption(
    first: float, extra: float
) -> None:
    resources = plan_lib.BASE
    low = resources.estimated_usd(first)
    assert low >= 0
    assert resources.estimated_usd(first + extra) >= low
    tripled = resources.estimated_usd(first, plan_lib.NONPREEMPTIBLE_PRICE_MULTIPLIER)
    # Each figure is rounded to the cent once.
    assert abs(tripled - 3 * low) <= 0.02


# --------------------------------------------------------------------------- #
# (c) Plan refusals                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("role", plan_lib.PUF_SUPPORT_BASE_INPUTS)
def test_a_missing_input_is_refused(role: str) -> None:
    data = base_plan_data()
    del data["inputs"][role]
    with pytest.raises(plan_lib.PlanError, match="requires inputs"):
        plan_lib.parse_plan(data)


def _replace_sha(data: dict, role: str, sha256: str) -> None:
    ref = data["inputs"][role]
    name = ref["uri"].rsplit("/", 1)[1]
    ref.update(uri=f"volume://{plan_lib.cas_volume_path(sha256, name)}", sha256=sha256)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(stage="materialize"), "has no stage"),
        (lambda d: d.update(stage="final_export"), "has no stage"),
        (lambda d: d.update(tool="us-puf-support"), "unknown tool"),
        # Digest formats.
        (
            lambda d: d["inputs"]["puf_2015_csv"].update(
                sha256=d["inputs"]["puf_2015_csv"]["sha256"].upper()
            ),
            "64 lowercase hex",
        ),
        (
            lambda d: d["inputs"]["puf_2024_h5"].update(sha256="7669f5b5"),
            "64 lowercase",
        ),
        (lambda d: d["inputs"]["acs_2022_h5"].pop("sha256"), "exactly"),
        (
            lambda d: d["inputs"]["asec_2024_h5"].update(sha256="0" * 64),
            "own digest",
        ),
        # The seeded archive must be the file the tool pins.
        (
            lambda d: _replace_sha(d, "asec_education_2022_zip", "e" * 64),
            "tool pins",
        ),
        # Options: the registration allows none; its flags are owned.
        (lambda d: d.update(options={"seed": 1}), "not allowlisted"),
        (lambda d: d.update(options={"n_estimators": 64}), "not allowlisted"),
        (lambda d: d.update(options={"checkpoint_dir": "/x"}), "not allowlisted"),
        (
            lambda d: d.update(options={"allow_geography_ladder_gate_failures": True}),
            "not allowlisted",
        ),
        # Inputs the registration does not take.
        (
            lambda d: d["inputs"].update(
                base_h5={
                    "uri": f"volume://cas/sha256/{'d' * 64}/b.h5",
                    "sha256": "d" * 64,
                }
            ),
            "takes no inputs",
        ),
        # Environment.
        (lambda d: d.update(env={"PYTHONHASHSEED": "1"}), "not allowlisted"),
        (lambda d: d.update(env={"PYTHONUNBUFFERED": 1}), "must be a string"),
        (lambda d: d.update(env={"HF_TOKEN": "x"}), "not allowlisted"),
        # The budget must leave the runner its 30 minutes inside the timeout.
        (
            lambda d: d.update(
                max_wall_seconds=plan_lib.BASE.timeout_s
                - plan_lib.BASE_RUNNER_OVERHEAD_SECONDS
                + 1
            ),
            "max_wall_seconds",
        ),
        (lambda d: d.update(nonpreemptible="true"), "true or false"),
    ],
)
def test_loose_base_plans_are_refused(mutate, message: str) -> None:
    data = base_plan_data()
    mutate(data)
    with pytest.raises(plan_lib.PlanError, match=message):
        plan_lib.parse_plan(data)


def test_the_budget_ceiling_is_the_timeout_less_the_base_s_overhead() -> None:
    ceiling = plan_lib.BASE.timeout_s - plan_lib.BASE_RUNNER_OVERHEAD_SECONDS
    assert ceiling == 14_700
    assert plan_lib.parse_plan(base_plan_data(max_wall_seconds=ceiling))
    # Max's 4 hours fit, with 5 minutes to spare for staging.
    assert ceiling - _committed_plan().max_wall_seconds == 300


@pytest.mark.parametrize(
    "flag",
    sorted(
        plan_lib.PUF_SUPPORT_BASE_WITHHELD_FLAGS
        | plan_lib.PUF_SUPPORT_BASE_BUILDER_FLAGS
    ),
)
def test_an_option_that_maps_to_an_owned_flag_is_refused(monkeypatch, flag) -> None:
    # A registry mistake: an allowlisted option for a flag the builder sets
    # would pass it twice, and one for a withheld flag would open an escape
    # hatch (--base-h5, --allow-geography-ladder-gate-failures, ...). The
    # plan is refused, and so is the argv, for every one of the 26 flags.
    rogue = dataclasses.replace(BASE, options={"x": plan_lib.OptionFlag(flag, str)})
    monkeypatch.setitem(plan_lib.TOOLS, BASE.name, rogue)
    with pytest.raises(plan_lib.PlanError, match=f"runner-owned flag {flag}"):
        plan_lib.parse_plan(base_plan_data(options={"x": "v"}))
    plan = dataclasses.replace(_committed_plan(), tool=rogue, options={"x": "v"})
    with pytest.raises(plan_lib.PlanError, match=f"runner-owned flag {flag}"):
        plan_lib.planned_argv(plan)


_OWNED = (
    plan_lib.PUF_SUPPORT_BASE_WITHHELD_FLAGS | plan_lib.PUF_SUPPORT_BASE_BUILDER_FLAGS
)


def _abbreviations(flag: str) -> list[str]:
    """Every proper prefix of ``flag`` after its ``--``."""

    return [flag[:end] for end in range(3, len(flag))]


@pytest.mark.parametrize("flag", sorted(plan_lib.PUF_SUPPORT_BASE_WITHHELD_FLAGS))
def test_an_abbreviation_of_a_withheld_flag_is_refused(monkeypatch, flag) -> None:
    # The tool's parser (argparse, allow_abbrev on) reads a unique prefix
    # as the flag it begins, so '--allow-geography' would be
    # '--allow-geography-ladder-gate-failures'. A registration naming any
    # prefix of a withheld flag, or any flag with '=' in it, is refused by
    # the plan and by the argv.
    rogues = [*_abbreviations(flag), f"{flag}=1", f"{flag[:6]}=x"]
    for rogue_flag in rogues:
        for kind, value in ((bool, True), (str, "v")):
            rogue = dataclasses.replace(
                BASE, options={"x": plan_lib.OptionFlag(rogue_flag, kind)}
            )
            monkeypatch.setitem(plan_lib.TOOLS, BASE.name, rogue)
            expected = "contains '='" if "=" in rogue_flag else "runner-owned flag"
            with pytest.raises(plan_lib.PlanError, match=expected):
                plan_lib.parse_plan(base_plan_data(options={"x": value}))
            plan = dataclasses.replace(
                _committed_plan(), tool=rogue, options={"x": value}
            )
            with pytest.raises(plan_lib.PlanError, match=expected):
                plan_lib.planned_argv(plan)
    # The example the review gave, by name.
    if flag == "--allow-geography-ladder-gate-failures":
        assert "--allow-geography" in rogues


def _owned_flag_parser():
    """An argparse parser with the base's 26 owned flags, as the tool's own
    (4b57d15a2 ``_parse_args``: ``argparse.ArgumentParser()``, so
    ``allow_abbrev`` is on). Each flag takes an optional value, so a bool
    flag and a valued one both parse."""

    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    for owned in sorted(_OWNED):
        parser.add_argument(owned, nargs="?", const=True, default=None)
    return parser


def test_argparse_expands_each_refused_abbreviation_to_an_owned_flag() -> None:
    # Why the guard refuses prefixes: argparse itself reads each
    # unambiguous one as the owned flag it begins.
    parser = _owned_flag_parser()
    expanded = 0
    for flag in sorted(plan_lib.PUF_SUPPORT_BASE_WITHHELD_FLAGS):
        for prefix in _abbreviations(flag):
            matches = [item for item in _OWNED if item.startswith(prefix)]
            if len(matches) != 1:
                continue  # ambiguous: argparse errors; refused all the same
            parsed, extra = parser.parse_known_args([prefix, "v"])
            assert extra == []
            dest = matches[0].lstrip("-").replace("-", "_")
            assert getattr(parsed, dest) == "v", prefix
            expanded += 1
    assert expanded > 100


_FLAG_TEXT = st.text(alphabet="abcdefghilnorstuwx-=0123", max_size=4)


@_PROPERTY_SETTINGS
@given(
    owned=st.sampled_from(sorted(_OWNED)),
    cut=st.integers(min_value=0, max_value=45),
    suffix=_FLAG_TEXT,
    lead=st.sampled_from(["--", "-", ""]),
)
def test_a_flag_the_guard_accepts_never_reaches_an_owned_flag(
    owned: str, cut: int, suffix: str, lead: str
) -> None:
    # Soundness against argparse itself: any flag option_flag_problem
    # accepts, given to a parser with the owned flags, sets none of them.
    flag = lead + owned[2 : 2 + cut] + suffix
    if plan_lib.option_flag_problem("x", flag, BASE.owned_flags) is not None:
        return
    parser = _owned_flag_parser()
    try:
        parsed, _ = parser.parse_known_args([flag, "v"])
    except (SystemExit, argparse.ArgumentError):  # a parse error sets nothing
        return
    assert all(value is None for value in vars(parsed).values()), flag


def test_validate_cli_refuses_a_rogue_registration(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    rogue = dataclasses.replace(
        BASE, options={"out": plan_lib.OptionFlag("--out", str)}
    )
    monkeypatch.setitem(plan_lib.TOOLS, BASE.name, rogue)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(base_plan_data(options={"out": "x"})))
    assert plan_lib.main(["validate", str(plan_path)]) == 2
    assert "runner-owned flag --out" in capsys.readouterr().err


def test_the_owned_flags_are_exactly_the_tool_s_flags() -> None:
    # The registration's owned flags (what parse_plan and option_argv
    # enforce) are the builder's and the withheld ones, which never overlap,
    # and together they are every flag of the tool's _parse_args. A flag the
    # tool gains later is caught here (in this tree) before a plan could rely
    # on it being unowned.
    assert not (
        plan_lib.PUF_SUPPORT_BASE_WITHHELD_FLAGS
        & plan_lib.PUF_SUPPORT_BASE_BUILDER_FLAGS
    )
    assert len(_OWNED) == 26
    assert BASE.owned_flags == _OWNED
    assert plan_lib.TOOLS[BASE.name].owned_flags == _OWNED
    assert _tool_flags(BASE_TOOL.read_text()) == _OWNED


def test_the_owned_flags_are_the_pinned_tool_s_flags(pinned_source: str) -> None:
    assert _tool_flags(pinned_source) == _OWNED == BASE.owned_flags


def test_a_staged_input_whose_bytes_differ_is_refused() -> None:
    ref = _committed_plan().inputs["puf_2015_csv"]
    plan_lib.verify_digest(ref, ref.sha256)
    with pytest.raises(plan_lib.PlanError, match="plan pins"):
        plan_lib.verify_digest(ref, "f" * 64)


# --------------------------------------------------------------------------- #
# (d) Cost                                                                     #
# --------------------------------------------------------------------------- #


def test_the_plan_s_estimate_at_its_budget_is_within_max_s_cap() -> None:
    plan = _committed_plan()
    estimate = plan_lib.estimated_usd_at_max_wall(plan)
    assert estimate is not None and estimate <= 15.0
    # 4 cores + 112 GiB at list for 4 h of tool and 30 min of runner, x3.
    per_second = (
        4 * plan_lib.CPU_USD_PER_CORE_SECOND + 112 * plan_lib.MEMORY_USD_PER_GIB_SECOND
    )
    assert estimate == pytest.approx(per_second * 3 * (14_400 + 1_800), abs=0.01)
    assert estimate == pytest.approx(14.63, abs=0.01)
    summary = plan_lib.summarize(plan)
    assert summary["estimated_usd_at_max_wall"] == estimate
    # At the local wall of 2,788 s it would be about $2.52.
    assert summary["estimated_usd_at_measured_wall"] == pytest.approx(2.52, abs=0.01)
    # HEAVY at the same budget would not fit the cap.
    assert plan_lib.HEAVY.estimated_usd(14_400 + 1_800, 3.0) > 15.0


def test_the_class_timeout_keeps_the_ceiling_at_the_request_inside_max_s_cap() -> None:
    # The timeout bounds the function's execution time, so the class's
    # request held for its whole timeout, at the non-preemptible list price,
    # is the most one attempt can list at while it stays inside its request,
    # whatever the plan's budget, a hung stop or a slow mirror do. Modal bills
    # the higher of request and use, so this holds for CPU only because the
    # class's CPU limit equals its request (the host throttles above the
    # limit), and not for memory used above 112 GiB, which has no limit.
    ceiling = plan_lib.BASE.estimated_usd(
        plan_lib.BASE.timeout_s, plan_lib.NONPREEMPTIBLE_PRICE_MULTIPLIER
    )
    assert ceiling <= plan_lib.BASE_COST_CAP_USD == 15.0
    assert ceiling == pytest.approx(14.90, abs=0.01)
    assert plan_lib.BASE.cpu_limit == plan_lib.BASE.cpu == 4.0
    plan = _committed_plan()
    assert plan_lib.estimated_usd_at_timeout(plan) == ceiling
    assert plan_lib.summarize(plan)["estimated_usd_at_timeout"] == ceiling
    # The old 6-hour timeout would have listed at $19.51, over the cap.
    assert plan_lib.BASE.estimated_usd(6 * 3600, 3.0) > plan_lib.BASE_COST_CAP_USD
    # Without the limit, Modal's default soft limit (the request plus 16
    # cores) held for the whole timeout would list at $25.28.
    unlimited = dataclasses.replace(plan_lib.BASE, cpu=4.0 + 16, cpu_limit=None)
    assert unlimited.estimated_usd(plan_lib.BASE.timeout_s, 3.0) == pytest.approx(
        25.28, abs=0.01
    )
    # Memory is the one cost path above the ceiling: each GiB-hour over the
    # request lists at about $0.024 non-preemptible.
    assert plan_lib.MEMORY_USD_PER_GIB_SECOND * 3600 * 3.0 == pytest.approx(
        0.024, abs=0.0005
    )


def test_the_cap_s_headroom_is_shared_with_the_check() -> None:
    # The runbook's "How the cap holds": a check lists at up to about $0.08
    # at its request (2 cores, 8 GiB, its whole 30-minute timeout, always
    # preemptible), outside the attempt's $14.90, so the $0.10 of headroom
    # under the $15 cap is shared; one full check leaves about $0.02.
    check = plan_lib.CHECK
    assert (check.cpu, check.memory_mib, check.timeout_s) == (2.0, 8 * 1024, 1800)
    assert check.cpu_limit is None  # use above the request bills above this
    check_ceiling = check.estimated_usd(check.timeout_s)
    assert check_ceiling == pytest.approx(0.08, abs=0.005)
    base_ceiling = plan_lib.estimated_usd_at_timeout(_committed_plan())
    headroom = plan_lib.BASE_COST_CAP_USD - base_ceiling
    assert headroom == pytest.approx(0.10, abs=0.005)
    assert base_ceiling + check_ceiling <= plan_lib.BASE_COST_CAP_USD
    assert headroom - check_ceiling == pytest.approx(0.02, abs=0.005)
    # The check class never runs at the non-preemptible price.
    smoke = json.loads((ROOT / "docs" / "us-modal-stage-smoke-plan.json").read_text())
    with pytest.raises(plan_lib.PlanError, match="always runs preemptible"):
        plan_lib.parse_plan({**smoke, "nonpreemptible": True})


def test_a_cpu_limit_below_the_request_is_refused() -> None:
    with pytest.raises(ValueError, match="below the request"):
        plan_lib.Resources("x", cpu=4.0, memory_mib=1024, timeout_s=60, cpu_limit=2.0)
    assert plan_lib.HEAVY.modal_cpu == 4.0
    assert plan_lib.BASE.modal_cpu == (4.0, 4.0)


@_PROPERTY_SETTINGS
@given(
    max_wall=st.integers(
        min_value=60,
        max_value=plan_lib.BASE.timeout_s - plan_lib.BASE_RUNNER_OVERHEAD_SECONDS,
    ),
    elapsed=st.floats(min_value=0, max_value=20_000, allow_nan=False),
    spent=st.floats(min_value=0, max_value=20_000, allow_nan=False),
)
def test_the_tool_s_budget_always_leaves_the_runner_its_reserve(
    max_wall: int, elapsed: float, spent: float
) -> None:
    # Whatever time staging took (elapsed) and whatever earlier attempts
    # spent, the tool is stopped early enough that the runner's reserve
    # still fits inside the class timeout, and never later than the plan's
    # own budget allows.
    plan = plan_lib.parse_plan(base_plan_data(max_wall_seconds=max_wall))
    remaining = plan_lib.remaining_wall_seconds(
        plan, [{"elapsed_seconds": spent}] if spent else []
    )
    budget = plan_lib.tool_budget(plan, remaining, elapsed)
    seconds = int(budget["seconds"])
    assert seconds <= remaining
    assert elapsed + seconds + plan_lib.BASE_RUNNER_OVERHEAD_SECONDS <= (
        plan_lib.BASE.timeout_s + 1
    )
    assert seconds == min(remaining, int(budget["container_seconds"]))
    assert budget["limited_by"] == (
        "max_wall_seconds"
        if remaining <= int(budget["container_seconds"])
        else "container_timeout"
    )
    # A first attempt that staged in under 5 minutes gets Max's whole 4 hours.
    if max_wall == 14_400 and not spent and elapsed <= 300:
        assert seconds == 14_400


def test_validate_cli_prints_the_base_argv_class_and_estimate(capsys) -> None:
    assert (
        plan_lib.main(
            ["validate", str(ROOT / "docs" / "us-modal-stage-route-a-base-plan.json")]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["resources"] == {
        "class": "base",
        "cpu": 4.0,
        "cpu_limit": 4.0,
        "memory_gib": 112.0,
        "timeout_h": 16_500 / 3600,
        "timeout_s": 16_500,
        "nonpreemptible": True,
        "runner_overhead_seconds": 1800,
        "min_free_disk_gib": 70,
        "mirrored_state_gib": 50,
        "mirror_first": list(plan_lib.PUF_SUPPORT_BASE_MIRROR_FIRST),
        "rest_is_resume_state": True,
    }
    assert summary["estimated_usd_at_max_wall"] == pytest.approx(14.63, abs=0.01)
    assert summary["estimated_usd_at_timeout"] == pytest.approx(14.90, abs=0.01)
    assert summary["measured_locally"]["peak_rss_gb"] == 72.5
    assert summary["home_seeds"] == [
        {"input": SEED.input, "path": f"~/{SEED.path}", "sha256": SEED.sha256}
    ]
    assert summary["argv"][2] == "tools/build_us_puf_support_base.py"


# --------------------------------------------------------------------------- #
# Resources and the Modal functions                                           #
# --------------------------------------------------------------------------- #


def test_the_base_class_covers_the_measured_peak_and_the_modal_excess() -> None:
    measured = plan_lib.MEASURED[("us-puf-support-base", "all")]
    request = plan_lib.BASE.memory_mib * 1024 * 1024
    assert request >= 1.5 * measured.peak_rss_bytes
    # The runbook saw Modal hold up to 24 GB more than local at the same point.
    assert request >= measured.peak_rss_bytes + 24e9 + 20 * 1024**3
    # The ACS classes are unchanged.
    assert (plan_lib.HEAVY.cpu, plan_lib.HEAVY.memory_mib) == (4.0, 128 * 1024)
    assert plan_lib.US_ACS_LOCAL_RELEASE.stages["materialize"].min_free_disk_gib is None
    assert (
        plan_lib.US_ACS_LOCAL_RELEASE.stages["materialize"].runner_overhead_seconds
        == plan_lib.RUNNER_OVERHEAD_SECONDS
        == 15 * 60
    )


def test_the_base_runs_on_its_own_modal_functions(app) -> None:
    for nonpreemptible in (False, True):
        runner = app.RUNNERS[("base", nonpreemptible)]
        options = runner.modal_options
        # CPU is (request, limit): the limit keeps the bill at the request.
        assert (options["cpu"], options["memory"], options["timeout"]) == (
            (4.0, 4.0),
            112 * 1024,
            16_500,
        )
        assert options.get("nonpreemptible", False) is nonpreemptible
        assert options["retries"] == 0
        # The default per-container disk quota holds the stage.
        assert "ephemeral_disk" not in options
    plan = _committed_plan()
    assert app.RUNNERS[(plan.resources.name, plan.nonpreemptible)] is (
        app.run_stage_base_nonpreemptible
    )


# --------------------------------------------------------------------------- #
# The budget stop reaches every process the tool spawned                      #
# --------------------------------------------------------------------------- #

# A parent that runs a stage child the way ``--stage all`` does (a second
# interpreter in the same process group, sharing stdout), then waits.
_STAGED_TOOL = (
    "import subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "print('stage child', child.pid, flush=True)\n"
    "time.sleep(60)\n"
)


def _staged_tool() -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", _STAGED_TOOL],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        process_group=0,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().startswith("stage child")
    return proc


def _kill_group(proc: subprocess.Popen) -> None:
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    with suppress(subprocess.TimeoutExpired):
        proc.communicate(timeout=10)


def test_terminating_only_the_tool_leaves_its_stage_child_running() -> None:
    # Why the runner stops the process group: after proc.terminate() the
    # stage child still holds the log pipe, so the runner, reading the pipe
    # to its end, would wait out the child's whole stage past the budget.
    proc = _staged_tool()
    try:
        proc.terminate()
        with pytest.raises(subprocess.TimeoutExpired):
            proc.communicate(timeout=2)
    finally:
        _kill_group(proc)


def test_stop_process_group_ends_the_tool_and_its_children() -> None:
    proc = _staged_tool()
    try:
        started = time.monotonic()
        sent = plan_lib.stop_process_group(proc, grace_seconds=5)
        # EOF on the pipe: no process of the group is left holding it.
        proc.communicate(timeout=10)
        assert sent["sigterm"] is True
        assert proc.returncode == -signal.SIGTERM
        assert time.monotonic() - started < 10
    finally:
        _kill_group(proc)


def test_stop_process_group_after_the_tool_exited_is_a_no_op() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"], process_group=0)
    proc.wait(timeout=30)
    assert plan_lib.stop_process_group(proc, grace_seconds=1) == {
        "sigterm": False,
        "sigkill": False,
    }


def test_the_app_s_budget_watch_stops_the_whole_group(app) -> None:
    proc = _staged_tool()
    try:
        watch = app._BudgetWatch(proc, 1, grace_seconds=5)
        watch.start()
        proc.communicate(timeout=20)
        watch.join(timeout=20)
        assert watch.fired
    finally:
        _kill_group(proc)


# --------------------------------------------------------------------------- #
# The home-cache seed                                                          #
# --------------------------------------------------------------------------- #


def _plan_with_seed_input(sha256: str) -> plan_lib.Plan:
    plan = _committed_plan()
    inputs = dict(plan.inputs)
    inputs[SEED.input] = dataclasses.replace(inputs[SEED.input], sha256=sha256)
    return dataclasses.replace(plan, inputs=inputs)


def test_seed_home_cache_places_the_verified_archive(tmp_path: Path) -> None:
    payload = b"asecpub23csv"
    sha256 = hashlib.sha256(payload).hexdigest()
    staged = tmp_path / "work" / "inputs" / SEED.input / "asecpub23csv.zip"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(payload)
    home = tmp_path / "home"
    plan = _plan_with_seed_input(sha256)
    rows = plan_lib.seed_home_cache(plan, {SEED.input: str(staged)}, home)
    target = home / ".cache" / "microcosm" / "cps" / "asec_2023" / "asecpub23csv.zip"
    assert rows == [
        {"input": SEED.input, "path": str(target), "sha256": sha256, "bytes": 12}
    ]
    assert target.read_bytes() == payload
    with pytest.raises(plan_lib.PlanError, match="plan pins"):
        plan_lib.seed_home_cache(
            _plan_with_seed_input("0" * 64), {SEED.input: str(staged)}, home
        )
    # A plan without the seeded input seeds nothing.
    smoke = dataclasses.replace(plan, inputs={})
    assert plan_lib.seed_home_cache(smoke, {}, home) == []


def test_an_unsafe_seed_path_is_refused(tmp_path: Path) -> None:
    rogue = dataclasses.replace(
        BASE, home_seeds=(dataclasses.replace(SEED, path="../../etc/x"),)
    )
    plan = dataclasses.replace(_committed_plan(), tool=rogue)
    with pytest.raises(plan_lib.PlanError, match="clean relative"):
        plan_lib.home_seed_targets(plan, tmp_path)


def test_the_seed_stops_the_tool_s_census_download(tmp_path: Path, monkeypatch) -> None:
    # The base's weeks-unemployed fetcher (unchanged from 4b57d15a2 on main)
    # reads ~/<SEED.path> when it exists and downloads only when it does not.
    from microcosm.build.us_runtime import weeks_unemployed as weeks

    assert weeks.ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SHA256 == SEED.sha256
    member = weeks.ASEC_2023_WEEKS_UNEMPLOYED_MEMBER
    archive = tmp_path / "staged.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member, "PERIDNUM,LKWEEKS\n1,0\n")
    sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    plan_lib.seed_home_cache(
        _plan_with_seed_input(sha256), {SEED.input: str(archive)}, Path.home()
    )

    def no_network(*_args, **_kwargs):
        raise AssertionError("the seeded fetcher must not download")

    monkeypatch.setattr(weeks.urllib.request, "urlopen", no_network)
    path = weeks.fetch_asec_2023_weeks_unemployed_source(
        expected_zip_size_bytes=None,
        expected_zip_sha256=None,
        expected_member_size_bytes=None,
        expected_member_crc32=None,
        expected_member_sha256=None,
    )
    assert Path(path).parent == (home / SEED.path).parent
    assert Path(path).name == member


# --------------------------------------------------------------------------- #
# The pinned tree file and the disk guard                                     #
# --------------------------------------------------------------------------- #


def test_tree_file_rows_verify_the_clone(tmp_path: Path) -> None:
    rel = "packages/x/data/crosswalk.csv"
    good = b"source,target,weight\n"
    tool = dataclasses.replace(
        BASE,
        tree_files={
            "crosswalk": plan_lib.TreeFile(rel, hashlib.sha256(good).hexdigest())
        },
    )
    (rows,) = plan_lib.tree_file_rows(tool, tmp_path)
    assert rows["problem"] == "not in the pinned tree"
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_bytes(good)
    (rows,) = plan_lib.tree_file_rows(tool, tmp_path)
    assert "problem" not in rows and rows["bytes"] == len(good)
    (tmp_path / rel).write_bytes(good + b"x")
    (rows,) = plan_lib.tree_file_rows(tool, tmp_path)
    assert "another version" in rows["problem"]
    assert plan_lib.tree_file_rows(plan_lib.US_ACS_LOCAL_RELEASE, tmp_path) == []


def test_work_disk_problem_needs_the_declared_space() -> None:
    stage = BASE.stages["all"]
    gib = 1024**3
    assert plan_lib.work_disk_problem(stage, 70 * gib) is None
    assert "needs 70 GiB" in plan_lib.work_disk_problem(stage, 69 * gib)
    acs = plan_lib.US_ACS_LOCAL_RELEASE.stages["materialize"]
    assert plan_lib.work_disk_problem(acs, 0) is None


def _clone_ok(**overrides) -> dict:
    return {
        "head_matches_plan": True,
        "tree_clean": True,
        "branch_verified": True,
        "tool_present": True,
        "tree_files_verified": True,
        **overrides,
    }


def test_run_stage_refuses_a_clone_whose_crosswalk_differs(
    app, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(plan_lib, "RUNS_MOUNT", str(tmp_path))
    monkeypatch.setattr(
        app, "_git_state", lambda plan: _clone_ok(tree_files_verified=False)
    )
    with pytest.raises(plan_lib.PlanError, match="does not match the plan"):
        app._run_stage(base_plan_data())
    # Refused before an attempt was recorded: nothing charged, nothing staged.
    assert not (tmp_path / "runs").exists()


def test_run_stage_refuses_a_small_disk_before_staging(
    app, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(plan_lib, "RUNS_MOUNT", str(tmp_path / "runs-volume"))
    monkeypatch.setattr(plan_lib, "WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(app, "_git_state", lambda plan: _clone_ok())
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(
        app.shutil, "disk_usage", lambda _path: usage(100 * 1024**3, 0, 20 * 1024**3)
    )
    with pytest.raises(app._Refusal, match="20.0 GiB free"):
        app._run_stage(base_plan_data())
    run_dir = tmp_path / "runs-volume" / "runs" / "route-a-base-4b57d15a287c"
    (record_path,) = (run_dir / "attempts").glob("all-*.json")
    record = json.loads(record_path.read_text())
    # A refusal is finished and not charged to the budget.
    assert (record["outcome"], record["finished"]) == ("refused", True)
    assert not (tmp_path / "work" / "inputs").exists()
    assert not (run_dir / "state").exists()


# --------------------------------------------------------------------------- #
# Verifying a partial fetch of the state                                       #
# --------------------------------------------------------------------------- #


def _base_receipt(
    state: Path, *, returncode: int = 0, stopped_at_budget: bool = False
) -> dict:
    data = base_plan_data()
    plan = plan_lib.parse_plan(data)
    receipt = plan_lib.build_receipt(
        plan,
        data,
        argv=plan_lib.planned_argv(plan),
        returncode=returncode,
        stopped_at_budget=stopped_at_budget,
        started_at="2026-09-29T12:00:00Z",
        finished_at="2026-09-29T15:00:00Z",
        wall_seconds=10_800.0,
        peak_rss_bytes=None,
        inputs_verified=[],
        outputs=plan_lib.hash_tree(state),
        git={},
        runner={},
    )
    return json.loads(json.dumps(receipt))


def test_verify_receipt_can_check_only_the_release_inputs(tmp_path: Path) -> None:
    state = tmp_path / "state"
    for rel, payload in {
        "base-checkpoints/000_source_construction.frame.h5": b"frame",
        "base-checkpoints/stage_run_context.json": b"{}",
        f"base-out/{plan_lib.PUF_SUPPORT_BASE_ARTIFACT}": b"h5",
        "base-out/base_populace_us_2024_puf_support.summary.json": b"{}",
    }.items():
        (state / rel).parent.mkdir(parents=True, exist_ok=True)
        (state / rel).write_bytes(payload)
    receipt = _base_receipt(state)
    # A fetch of base-out only.
    fetched = tmp_path / "fetched"
    (fetched / "base-out").mkdir(parents=True)
    for path in (state / "base-out").iterdir():
        (fetched / "base-out" / path.name).write_bytes(path.read_bytes())
    assert (
        plan_lib.verify_receipt(receipt, fetched, strict=True, prefix="base-out") == []
    )
    missing = plan_lib.verify_receipt(receipt, fetched)
    assert missing == [
        "missing: base-checkpoints/000_source_construction.frame.h5",
        "missing: base-checkpoints/stage_run_context.json",
    ]
    (fetched / "base-out" / "stray.json").write_text("{}")
    assert plan_lib.verify_receipt(
        receipt, fetched, strict=True, prefix="base-out/"
    ) == ["not in receipt: base-out/stray.json"]
    (fetched / "base-out" / plan_lib.PUF_SUPPORT_BASE_ARTIFACT).write_bytes(b"h6")
    assert plan_lib.verify_receipt(receipt, fetched, prefix="base-out") == [
        f"sha256 mismatch: base-out/{plan_lib.PUF_SUPPORT_BASE_ARTIFACT}"
    ]
    assert plan_lib.verify_receipt(receipt, fetched, prefix="nothing-here") == [
        "the receipt lists no outputs under nothing-here/"
    ]
    with pytest.raises(plan_lib.PlanError, match="clean relative"):
        plan_lib.verify_receipt(receipt, fetched, prefix="../base-out")


def test_verify_receipt_cli_takes_a_prefix(tmp_path: Path, capsys) -> None:
    state = tmp_path / "state"
    (state / "base-out").mkdir(parents=True)
    (state / "base-out" / "x.h5").write_bytes(b"h5")
    (state / "base-checkpoints").mkdir()
    (state / "base-checkpoints" / "c.h5").write_bytes(b"c")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(_base_receipt(state)))
    (state / "base-checkpoints" / "c.h5").unlink()
    args = ["verify-receipt", str(receipt_path), "--state-root", str(state)]
    assert plan_lib.main([*args, "--prefix", "base-out", "--strict"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "verified": True,
        "status": "COMPLETED",
        "outputs": 1,
        "problems": 0,
    }
    assert plan_lib.main(args) == 1
    capsys.readouterr()
    assert plan_lib.main([*args, "--prefix", "/abs"]) == 2
    assert "REFUSED" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("returncode", "stopped", "problems"),
    [
        (
            1,
            False,
            ["the receipt's status is 'FAILED', not 'COMPLETED' (returncode 1)"],
        ),
        # A stop at the budget during final_export: the tool (4b57d15a2)
        # writes its H5 at the final path and reopens it, so base-out can
        # hold a truncated H5 that the FAILED receipt lists faithfully.
        (
            -15,
            True,
            [
                "the receipt's status is 'FAILED', not 'COMPLETED' (returncode -15)",
                "the stage was stopped at its budget (stopped_at_budget)",
            ],
        ),
    ],
)
def test_verify_receipt_cli_refuses_a_stage_that_did_not_finish(
    tmp_path: Path, capsys, returncode: int, stopped: bool, problems: list[str]
) -> None:
    state = tmp_path / "state"
    (state / "base-out").mkdir(parents=True)
    (state / "base-out" / plan_lib.PUF_SUPPORT_BASE_ARTIFACT).write_bytes(b"\x89HD")
    receipt = _base_receipt(state, returncode=returncode, stopped_at_budget=stopped)
    assert receipt["status"] == "FAILED"
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    args = [
        "verify-receipt",
        str(receipt_path),
        "--state-root",
        str(state),
        "--prefix",
        "base-out",
        "--strict",
    ]
    assert plan_lib.main(args) == 1
    captured = capsys.readouterr()
    assert captured.err.splitlines() == problems
    assert json.loads(captured.out) == {
        "verified": False,
        "status": "FAILED",
        "outputs": 1,
        "problems": len(problems),
    }
    # The bytes still verify when the operator asks for exactly that.
    assert plan_lib.main([*args, "--allow-failed"]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True
    # The runner's resume check still accepts a FAILED receipt's state.
    latest = ("all-x.json", receipt)
    assert plan_lib.pulled_state_problems(state, latest) == []


# --------------------------------------------------------------------------- #
# Comparing the Modal base with the local run of the same commit               #
# --------------------------------------------------------------------------- #

LOCAL_REFERENCE = ROOT / "docs" / "us-modal-stage-route-a-base-local-reference.json"
#: The thread counts Modal set in the check container (runbook, acceptance
#: attempt), at this class's request.
MODAL_THREAD_ENV = {
    key: "4"
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    )
}
MODAL_PYTHON = "3.14.2 (main) [GCC]"


def _reference() -> dict:
    return json.loads(LOCAL_REFERENCE.read_text())


def test_the_local_reference_matches_the_plan_and_the_fixture() -> None:
    reference = _reference()
    plan = _committed_plan()
    fixture = _fixture()
    assert reference["schema"] == plan_lib.LOCAL_REFERENCE_SCHEMA
    assert (reference["tool"], reference["stage"]) == (BASE.name, plan.stage)
    assert reference["commit"] == plan.commit == fixture["build_commit"]
    # The local run's command is the fixture's, and its inputs are Route A's
    # (and so the plan's), digest for digest.
    assert reference["argv"] == fixture["argv"]
    assert reference["env"] == fixture["env"]
    assert (
        reference["inputs"]
        == {role: sha256 for role, (_, sha256) in ROUTE_A_BASE_INPUTS.items()}
        == {role: ref.sha256 for role, ref in plan.inputs.items()}
    )
    # The outputs are frame checkpoints under the plan's checkpoint dir.
    checkpoints = plan_lib.PUF_SUPPORT_BASE_CHECKPOINTS
    assert sorted(reference["outputs"]) == [
        f"{checkpoints}/000_source_construction.frame.h5",
        f"{checkpoints}/001_pre_clone_enrichment.frame.h5",
    ]
    assert reference["run_context_path"] == f"{checkpoints}/stage_run_context.json"
    # The pipeline is the local run's 24 outer stages, and its digest is the
    # tool's (canonical JSON of the ordered names and descriptions,
    # outer_stage_runtime._mapping_sha256 at 4b57d15a2).
    pipeline = reference["pipeline"]
    canonical = json.dumps(
        pipeline,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert (
        hashlib.sha256(canonical.encode()).hexdigest() == reference["pipeline_sha256"]
    )
    names = [stage["name"] for stage in pipeline["stages"]]
    assert len(names) == 24
    assert names[0] == "source_construction" and names[-1] == "final_export"
    assert "primary_qrf_chain" in names
    # The compared outputs are exactly the checkpoints of the compared
    # stages, which are the pipeline's first two, named as the tool names
    # them (NNN_<stage>.frame.h5).
    compared = reference["stages_compared"]
    assert compared == names[:2]
    assert sorted(reference["outputs"]) == [
        f"{checkpoints}/{index:03d}_{name}.frame.h5"
        for index, name in enumerate(compared)
    ]
    # Every file compare-lineage reads is pushed first, and survives a clean
    # exit's push, which copies only those paths.
    stage = BASE.stages["all"]
    for rel in [*reference["outputs"], reference["run_context_path"]]:
        assert any(plan_lib.under_state_path(rel, p) for p in stage.mirror_first)


def test_the_local_run_config_is_the_registration_s_build() -> None:
    # The run config the local run's tool locked, as the reference keeps it
    # (paths cut to file names), is the build this registration runs: the
    # plan's files and digests in Route A's order, the registry's settings,
    # every escape hatch at its default. Without the pinned commit this is
    # what holds it; with it, the next test derives it from the Modal argv.
    reference = _reference()
    config = reference["run_config"]
    plan = _committed_plan()
    # The 27 keys of _stage_run_config at 4b57d15a2, already masked.
    assert len(config) == 27
    assert plan_lib.mask_config_paths(config) == config
    name = {role: ref.filename for role, ref in plan.inputs.items()}
    sha = {role: ref.sha256 for role, ref in plan.inputs.items()}
    asec_years = plan_lib.PUF_SUPPORT_BASE_ASEC_YEARS
    education_years = plan_lib.PUF_SUPPORT_BASE_EDUCATION_YEARS
    identity = config.pop("builder_code_identity")
    threads = config.pop("thread_environment")
    assert config == {
        "acs_h5": name["acs_2022_h5"],
        "allow_geography_ladder_gate_failures": False,
        "asec_2023_weeks_unemployed_source": None,
        "asec_education_source": {
            str(year): name[f"asec_education_{year}_zip"] for year in education_years
        },
        "asec_h5": [f"{year}={name[f'asec_{year}_h5']}" for year in asec_years],
        "asec_h5_sha256": {
            str(year): sha[f"asec_{year}_h5"] for year in sorted(asec_years)
        },
        "asec_max_households": None,
        "assign_congressional_districts": True,
        "base_h5": None,
        "block_ladder_artifact": name["block_ladder_npz"],
        "congressional_district_seed": plan_lib.PUF_SUPPORT_BASE_CD_SEED,
        "congressional_district_vintage_crosswalk": PurePosixPath(
            plan_lib.CD_VINTAGE_CROSSWALK.path
        ).name,
        "equivalence_deterministic_h5_metadata": False,
        "geography_ladder_seed": 0,
        "ledger_facts": name["base_ledger_facts"],
        "n_estimators": plan_lib.PUF_SUPPORT_BASE_N_ESTIMATORS,
        "out": plan_lib.PUF_SUPPORT_BASE_OUT,
        "puf_h5": name["puf_2024_h5"],
        "puf_h5_sha256": sha["puf_2024_h5"],
        "puf_source_year_csv": name["puf_2015_csv"],
        "puf_source_year_csv_sha256": sha["puf_2015_csv"],
        "seed": plan_lib.PUF_SUPPORT_BASE_SEED,
        "support_spine_spec": None,
        "target_year": plan_lib.PUF_SUPPORT_BASE_TARGET_YEAR,
        "without_block_ladder": False,
    }
    # The six distributions the tool fingerprints, pinned once each, and the
    # local interpreter the report prints.
    assert set(identity) == {"dependency_versions", "python", "source_sha256"}
    assert set(identity["dependency_versions"]) == {
        "h5py",
        "numpy",
        "pandas",
        "policyengine-us",
        "quantile-forest",
        "scikit-learn",
    }
    assert identity["python"] == reference["local_platform"]["python"]
    # The local run set no thread or worker variable; the tool records
    # PYTHONHASHSEED as 0 when it is unset.
    assert threads == {
        "BLIS_NUM_THREADS": None,
        "MKL_NUM_THREADS": None,
        "NUMEXPR_NUM_THREADS": None,
        "OMP_NUM_THREADS": None,
        "OPENBLAS_NUM_THREADS": None,
        "POPULACE_FIT_N_JOBS": None,
        "POPULACE_FIT_PREDICT_WORKERS": None,
        "PYTHONHASHSEED": "0",
        "VECLIB_MAXIMUM_THREADS": None,
    }


def test_the_pinned_tool_locks_the_reference_s_run_config_from_the_modal_argv(
    pinned_tool, tmp_path: Path, monkeypatch
) -> None:
    # Differential through the pinned tool (4b57d15a2's _parse_args and
    # _stage_run_config): the config it locks from the Modal argv, paths
    # masked, is the local run's. The staged files are empty stand-ins, so
    # the tool's file digest returns the plan's digest by file name, and its
    # code identity (a digest of the clone it runs from) is the local run's;
    # compare-lineage checks both in the Modal run's own context. With no
    # thread variable set, as in the local run, the config is equal
    # including thread_environment.
    reference = _reference()
    plan = _committed_plan()
    work = tmp_path / "work"
    for ref in plan.inputs.values():
        staged = Path(plan_lib.input_local_path(str(work), ref))
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(b"")
    digests = {ref.filename: ref.sha256 for ref in plan.inputs.values()}
    monkeypatch.setattr(pinned_tool, "_sha256", lambda path: digests[Path(path).name])
    identity = dict(reference["run_config"]["builder_code_identity"])
    monkeypatch.setattr(pinned_tool, "_builder_code_identity", lambda: dict(identity))
    for key in reference["run_config"]["thread_environment"]:
        monkeypatch.delenv(key, raising=False)
    argv = plan_lib.planned_argv(plan, work_root=str(work))
    args = pinned_tool._parse_args(argv[3:])
    config = pinned_tool._stage_run_config(args)
    assert plan_lib.mask_config_paths(config) == reference["run_config"]
    # And from that config compare-lineage finds nothing to refuse.
    assert _pinned_lineage(reference, config) == []

    # The hash seed, through the pinned tool. A probe seed in the tool's
    # environment would reach every stage (its default only fills an unset
    # value) and be locked in the run config, which compare-lineage refuses.
    monkeypatch.setenv("PYTHONHASHSEED", "12345")
    assert pinned_tool._staged_subprocess_environment()["PYTHONHASHSEED"] == "12345"
    forced = pinned_tool._stage_run_config(args)
    assert forced["thread_environment"]["PYTHONHASHSEED"] == "12345"
    assert _pinned_lineage(reference, forced) == [
        "run_config.thread_environment.PYTHONHASHSEED: Modal '12345', local '0' "
        "(a determinism input of the tool, compared, never only reported)"
    ]
    # The runner's tool environment, from a container with that seed:
    # without it the tool gives its stages "0" and locks the local config.
    env, removed = plan_lib.tool_environment(dict(os.environ), plan.env)
    assert "PYTHONHASHSEED" in removed and "PYTHONHASHSEED" not in env
    for key in set(os.environ) - set(env):
        monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert dict(os.environ) == env
    assert pinned_tool._staged_subprocess_environment()["PYTHONHASHSEED"] == "0"
    neutral = pinned_tool._stage_run_config(args)
    assert plan_lib.mask_config_paths(neutral) == reference["run_config"]
    assert _pinned_lineage(reference, neutral) == []


def _pinned_lineage(reference: dict, config: dict) -> list[str]:
    """lineage_problems of a COMPLETED receipt whose run context locks ``config``."""

    context = {"pipeline_sha256": reference["pipeline_sha256"], "run_config": config}
    receipt = _modal_receipt(context)
    context_sha = hashlib.sha256(_context_bytes(context)).hexdigest()
    return plan_lib.lineage_problems(receipt, reference, context, context_sha)


def _modal_run_config() -> dict:
    """The run config the pinned tool would lock on Modal.

    The local run's, with each file at the path the runner stages it (read
    from the committed plan's argv), Modal's interpreter and the thread
    counts Modal set in its containers.
    """

    full: dict[str, str] = {}
    for value in plan_lib.planned_argv(_committed_plan()):
        path = value.partition("=")[2] if re.fullmatch(r"\d{4}=/.*", value) else value
        if path.startswith("/"):
            full[PurePosixPath(path).name] = path

    def relocate(value: object) -> object:
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str):
            year, eq, rest = value.partition("=")
            if eq and year.isdigit() and rest in full:
                return f"{year}={full[rest]}"
            return full.get(value, value)
        return value

    config = relocate(_reference()["run_config"])
    assert isinstance(config, dict)
    assert (
        config["out"] == f"{plan_lib.WORK_ROOT}/state/{plan_lib.PUF_SUPPORT_BASE_OUT}"
    )
    config["builder_code_identity"]["python"] = MODAL_PYTHON
    config["thread_environment"].update(MODAL_THREAD_ENV)
    return config


def _context_bytes(context: dict) -> bytes:
    return json.dumps(context, indent=2).encode()


def _modal_receipt(
    context: dict, *, frames: dict[str, str] | None = None, **overrides
) -> dict:
    """A COMPLETED receipt of the committed plan listing ``context``."""

    reference = _reference()
    data = base_plan_data()
    plan = plan_lib.parse_plan(data)
    context_bytes = _context_bytes(context)
    outputs = [
        {"path": rel, "bytes": 1, "sha256": sha}
        for rel, sha in (frames or reference["outputs"]).items()
    ]
    outputs.append(
        {
            "path": reference["run_context_path"],
            "bytes": len(context_bytes),
            "sha256": hashlib.sha256(context_bytes).hexdigest(),
        }
    )
    receipt = plan_lib.build_receipt(
        plan,
        data,
        argv=plan_lib.planned_argv(plan),
        returncode=overrides.pop("returncode", 0),
        stopped_at_budget=overrides.pop("stopped_at_budget", False),
        started_at="2026-09-29T12:00:00Z",
        finished_at="2026-09-29T15:00:00Z",
        wall_seconds=10_800.0,
        peak_rss_bytes=None,
        inputs_verified=[
            {"name": role, "uri": ref.uri, "sha256": ref.sha256, "bytes": 1}
            for role, ref in plan.inputs.items()
        ],
        outputs=outputs,
        git={},
        runner={},
    )
    receipt.update(overrides)
    return json.loads(json.dumps(receipt))


def _modal_run(
    tmp_path: Path,
    *,
    frames: dict[str, str] | None = None,
    identity: dict | None = None,
    config: dict | None = None,
    **receipt_overrides,
) -> tuple[Path, Path]:
    """A receipt and run context for a Modal run of the committed plan."""

    reference = _reference()
    config = config if config is not None else _modal_run_config()
    config["builder_code_identity"].update(identity or {})
    context = {"pipeline_sha256": reference["pipeline_sha256"], "run_config": config}
    receipt = _modal_receipt(context, frames=frames, **receipt_overrides)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    context_path = tmp_path / "stage_run_context.json"
    context_path.write_bytes(_context_bytes(context))
    return receipt_path, context_path


def _lineage(receipt_path: Path, context_path: Path) -> list[str]:
    return [
        "compare-lineage",
        str(receipt_path),
        "--reference",
        str(LOCAL_REFERENCE),
        "--run-context",
        str(context_path),
    ]


def _lineage_of(tmp_path: Path, **kwargs) -> list[str]:
    """lineage_problems of a Modal run built by :func:`_modal_run`."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    receipt_path, context_path = _modal_run(tmp_path, **kwargs)
    context_bytes = context_path.read_bytes()
    return plan_lib.lineage_problems(
        json.loads(receipt_path.read_text()),
        _reference(),
        json.loads(context_bytes),
        hashlib.sha256(context_bytes).hexdigest(),
    )


def test_compare_lineage_passes_a_modal_run_that_reproduces_the_local_bytes(
    tmp_path: Path, capsys
) -> None:
    receipt_path, context_path = _modal_run(tmp_path)
    assert plan_lib.main(_lineage(receipt_path, context_path)) == 0
    out = json.loads(capsys.readouterr().out)
    # A match covers the compared outputs only: two of 24 outer stages.
    assert "reproduced" not in out
    assert (out["compared_outputs_match"], out["problems"]) == (True, 0)
    assert (out["stages_compared"], out["stages_total"]) == (2, 24)
    assert out["stages_compared_names"] == [
        "source_construction",
        "pre_clone_enrichment",
    ]
    assert len(out["stages_not_compared"]) == 22
    assert out["stages_not_compared"][0] == "clone_feature_extraction"
    assert "primary_qrf_chain" in out["stages_not_compared"]
    assert out["stages_not_compared"][-1] == "final_export"
    # Reported, not refused: the interpreters and the thread variables.
    assert out["python"]["local"].startswith("3.14.7 free-threading build")
    assert out["python"]["modal"] == MODAL_PYTHON
    assert out["thread_environment_differences"] == {
        key: {"modal": "4", "local": None} for key in sorted(MODAL_THREAD_ENV)
    }
    # Compared, and equal: the hash seed the tool's stages ran with.
    assert out["thread_environment_compared"] == {
        "PYTHONHASHSEED": {"modal": "0", "local": "0"}
    }


@pytest.mark.parametrize("seed", ["12345", "1", None])
def test_compare_lineage_refuses_another_hash_seed(
    tmp_path: Path, capsys, seed: str | None
) -> None:
    # The hash seed is a determinism input of the pinned tool, so unlike the
    # thread counts it is refused, not reported: a probe value, another
    # seed, or none recorded at all.
    config = _modal_run_config()
    if seed is None:
        del config["thread_environment"]["PYTHONHASHSEED"]
    else:
        config["thread_environment"]["PYTHONHASHSEED"] = seed
    receipt_path, context_path = _modal_run(tmp_path, config=config)
    assert plan_lib.main(_lineage(receipt_path, context_path)) == 1
    captured = capsys.readouterr()
    assert captured.err.splitlines() == [
        f"run_config.thread_environment.PYTHONHASHSEED: Modal {seed!r}, local '0' "
        "(a determinism input of the tool, compared, never only reported)"
    ]
    out = json.loads(captured.out)
    assert (out["compared_outputs_match"], out["problems"]) == (False, 1)
    assert out["thread_environment_compared"] == {
        "PYTHONHASHSEED": {"modal": seed, "local": "0"}
    }
    # Refused, so not also listed with the differences that are only reported.
    assert "PYTHONHASHSEED" not in out["thread_environment_differences"]
    assert plan_lib.THREAD_ENVIRONMENT_COMPARED == {"PYTHONHASHSEED"}


def test_compare_lineage_refuses_a_run_with_another_pipeline(
    tmp_path: Path, capsys
) -> None:
    # The stage counts are the local pipeline's, so a Modal run whose tool
    # locked a different outer-stage list is refused.
    receipt_path, context_path = _modal_run(tmp_path)
    context = json.loads(context_path.read_text())
    context["pipeline_sha256"] = "2" * 64
    context_bytes = json.dumps(context).encode()
    context_path.write_bytes(context_bytes)
    receipt = json.loads(receipt_path.read_text())
    for item in receipt["outputs"]:
        if item["path"] == _reference()["run_context_path"]:
            item["sha256"] = hashlib.sha256(context_bytes).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    assert plan_lib.main(_lineage(receipt_path, context_path)) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("pipeline_sha256: Modal '2222")
    assert json.loads(captured.out)["compared_outputs_match"] is False


def test_compare_lineage_reports_every_departure(tmp_path: Path, capsys) -> None:
    reference = _reference()
    frames = dict(reference["outputs"])
    first = "base-checkpoints/000_source_construction.frame.h5"
    frames[first] = "0" * 64
    identity = {
        "source_sha256": "1" * 64,
        "dependency_versions": {
            **reference["run_config"]["builder_code_identity"]["dependency_versions"],
            "numpy": "2.4.5",
        },
    }
    receipt_path, context_path = _modal_run(tmp_path, frames=frames, identity=identity)
    assert plan_lib.main(_lineage(receipt_path, context_path)) == 1
    err = capsys.readouterr().err.splitlines()
    assert err[0] == (
        f"{first}: Modal wrote sha256 {'0' * 64}, the local run {reference['outputs'][first]}"
    )
    assert err[1].startswith("builder_code_identity.source_sha256: Modal '1111")
    assert err[2].startswith("builder_code_identity.dependency_versions: Modal ")
    assert len(err) == 3
    # A run context that is not the file the receipt lists is refused.
    context_path.write_text("{}")
    assert plan_lib.main(_lineage(receipt_path, context_path)) == 1
    assert "the file given is sha256" in capsys.readouterr().err


def test_compare_lineage_refuses_a_run_with_other_settings(
    tmp_path: Path, capsys
) -> None:
    # The review's probe: none of these settings reaches stages 000 or 001,
    # so matching checkpoints cannot catch them. The run config, the
    # receipt's status and its argv do.
    config = _modal_run_config()
    changed = {
        "n_estimators": 64,
        "congressional_district_seed": 7,
        "geography_ladder_seed": 3,
        "assign_congressional_districts": False,
        "without_block_ladder": True,
        "allow_geography_ladder_gate_failures": True,
    }
    config.update(changed)
    config["thread_environment"]["POPULACE_FIT_N_JOBS"] = "1"
    receipt_path, context_path = _modal_run(
        tmp_path,
        config=config,
        returncode=1,
        status="FAILED",
        argv=["--n-estimators", "64"],
    )
    assert plan_lib.main(_lineage(receipt_path, context_path)) == 1
    captured = capsys.readouterr()
    err = captured.err.splitlines()
    assert err[0] == "the receipt's status is 'FAILED', not 'COMPLETED' (returncode 1)"
    assert err[1].startswith(
        "argv: the receipt's command (tokenized, 2 elements) is not the local "
        f"run's ({len(_fixture()['argv'])}): [0] Modal '--n-estimators', local "
        "'<python>'"
    )
    for key, value in changed.items():
        local = _reference()["run_config"][key]
        assert f"run_config.{key}: Modal {value!r}, local {local!r}" in err
    assert len(err) == 2 + len(changed)
    out = json.loads(captured.out)
    assert (out["compared_outputs_match"], out["problems"]) == (False, len(err))
    # A pinned worker count is reported, not refused.
    assert out["thread_environment_differences"]["POPULACE_FIT_N_JOBS"] == {
        "modal": "1",
        "local": None,
    }


_COMPARED_KEYS = sorted(
    set(json.loads(LOCAL_REFERENCE.read_text())["run_config"])
    - plan_lib.RUN_CONFIG_REPORTED_ONLY
    - {"builder_code_identity"}
)


@pytest.mark.parametrize("key", _COMPARED_KEYS)
def test_every_run_config_key_but_the_thread_variables_is_compared(
    tmp_path: Path, key: str
) -> None:
    assert len(_COMPARED_KEYS) == 25  # 27 keys less the two handled apart
    config = _modal_run_config()
    config[key] = "changed"
    problems = _lineage_of(tmp_path / "changed", config=config)
    assert len(problems) == 1 and problems[0].startswith(f"run_config.{key}: Modal ")
    config = _modal_run_config()
    del config[key]
    assert _lineage_of(tmp_path / "missing", config=config) == [
        f"run_config.{key}: not in the Modal run's config"
    ]


def test_paths_are_compared_by_file_name_and_the_interpreter_is_not(
    tmp_path: Path,
) -> None:
    # Another directory is the same file; another file name is not.
    config = _modal_run_config()
    config["acs_h5"] = "/somewhere/else/acs_2022.h5"
    config["asec_h5"][0] = "2024=/elsewhere/census_cps_2024.h5"
    config["builder_code_identity"]["python"] = "3.15.0"
    config["thread_environment"] = {"OMP_NUM_THREADS": "64", "PYTHONHASHSEED": "0"}
    assert _lineage_of(tmp_path / "a", config=config) == []
    config["acs_h5"] = "/work/inputs/acs_2022_h5/acs_2023.h5"
    config["asec_h5"] = list(reversed(config["asec_h5"]))
    problems = _lineage_of(tmp_path / "b", config=config)
    assert problems == [
        "run_config.acs_h5: Modal 'acs_2023.h5', local 'acs_2022.h5'",
        "run_config.asec_h5: Modal ['2022=census_cps_2022.h5', "
        "'2023=census_cps_2023.h5', '2024=census_cps_2024.h5'], local "
        "['2024=census_cps_2024.h5', '2023=census_cps_2023.h5', "
        "'2022=census_cps_2022.h5']",
    ]


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        (
            {"stopped_at_budget": True, "returncode": 0},
            "the stage was stopped at its budget (stopped_at_budget)",
        ),
        (
            {"returncode": -9},
            "the receipt's status is 'FAILED', not 'COMPLETED' (returncode -9)",
        ),
    ],
)
def test_compare_lineage_refuses_a_stage_that_did_not_finish(
    tmp_path: Path, overrides: dict, problem: str
) -> None:
    assert problem in _lineage_of(tmp_path, **overrides)


def test_compare_lineage_refuses_another_command_env_or_input(tmp_path: Path) -> None:
    plan = _committed_plan()
    argv = plan_lib.planned_argv(plan)
    # The ASEC years pooled in another order.
    swapped = list(argv)
    i2024 = argv.index(next(item for item in argv if item.startswith("2024=/")))
    i2023 = argv.index(next(item for item in argv if item.startswith("2023=/")))
    swapped[i2024], swapped[i2023] = argv[i2023], argv[i2024]
    problems = _lineage_of(tmp_path / "argv", argv=swapped)
    assert len(problems) == 1 and problems[0].startswith("argv: ")
    swapped_in = f"[{i2024}] Modal '2023=<input:asec_2023_h5>/census_cps_2023.h5'"
    assert swapped_in in problems[0]
    # A path the runner did not build is kept verbatim, so it differs.
    moved = [item.replace("/work/state/", "/scratch/") for item in argv]
    problems = _lineage_of(tmp_path / "moved", argv=moved)
    assert len(problems) == 1 and "'/scratch/base-checkpoints'" in problems[0]
    # A plan environment variable the local run did not set is refused,
    # unless the tool records it in thread_environment (reported instead).
    data = base_plan_data(env={"PYTHONUNBUFFERED": "1", "MICROCOSM_X": "1"})
    assert _lineage_of(tmp_path / "env", plan=data) == [
        "env: the Modal plan sets {'MICROCOSM_X': '1', 'PYTHONUNBUFFERED': '1'}, "
        "the local run set {'PYTHONUNBUFFERED': '1'} (the thread and worker "
        "variables the run context records aside)"
    ]
    data = base_plan_data(env={"PYTHONUNBUFFERED": "1", "POPULACE_FIT_N_JOBS": "4"})
    assert _lineage_of(tmp_path / "pinned", plan=data) == []
    # An input the local run did not use, or one the receipt did not verify.
    inputs = [{"name": role, "sha256": ref.sha256} for role, ref in plan.inputs.items()]
    inputs[0] = {"name": inputs[0]["name"], "sha256": "e" * 64}
    assert (inputs[0]["name"], inputs[-1]["name"]) == ("acs_2022_h5", "puf_2024_h5")
    assert _lineage_of(tmp_path / "inputs", inputs=inputs[:-1]) == [
        f"input acs_2022_h5: Modal staged sha256 {'e' * 64}, the local run used "
        f"{plan.inputs['acs_2022_h5'].sha256}",
        "input puf_2024_h5: not among the receipt's verified inputs",
    ]


def test_lineage_problems_refuses_the_wrong_run() -> None:
    reference = _reference()
    receipt = {
        "schema": plan_lib.RECEIPT_SCHEMA,
        "status": "COMPLETED",
        "tool": BASE.name,
        "stage": "all",
        "source": {"commit": "f" * 40},
        "outputs": [],
    }
    problems = plan_lib.lineage_problems(receipt, reference, None, None)
    assert (
        problems[0]
        == f"commit: the receipt's {'f' * 40}, the local run's {reference['commit']}"
    )
    assert "base-checkpoints/000_source_construction.frame.h5: not in the receipt" in (
        problems
    )
    assert plan_lib.lineage_problems({}, reference, None, None) == [
        f"not a {plan_lib.RECEIPT_SCHEMA} receipt"
    ]
    assert plan_lib.lineage_problems(receipt, {}, None, None) == [
        f"not a {plan_lib.LOCAL_REFERENCE_SCHEMA} reference"
    ]


# --------------------------------------------------------------------------- #
# Uploading the inputs                                                         #
# --------------------------------------------------------------------------- #


def _plan_with_local_files(tmp_path: Path, roles: tuple[str, ...]) -> tuple[Path, dict]:
    data = base_plan_data()
    files = {}
    for role in roles:
        path = tmp_path / "local" / f"{role} file.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(role.encode())
        _replace_sha(data, role, hashlib.sha256(path.read_bytes()).hexdigest())
        files[role] = path
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(data))
    return plan_path, files


def test_upload_commands_verify_each_file_against_the_plan(
    tmp_path: Path, capsys
) -> None:
    plan_path, files = _plan_with_local_files(tmp_path, ("puf_2015_csv", "acs_2022_h5"))
    pairs = [f"{role}={path}" for role, path in files.items()]
    assert plan_lib.main(["upload-commands", str(plan_path), *pairs]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    plan = plan_lib.parse_plan(json.loads(plan_path.read_text()))
    assert [row["role"] for row in rows] == list(files)
    for row in rows:
        ref = plan.inputs[row["role"]]
        assert row["sha256"] == ref.sha256
        assert row["volume_path"] == ref.volume_path
        assert row["upload"] == (
            f"modal volume put {plan_lib.INPUTS_VOLUME} "
            f"'{files[row['role']]}' {ref.volume_path}"
        )
    # The script form uploads each file unless the volume already lists it.
    assert plan_lib.main(["upload-commands", "--shell", str(plan_path), *pairs]) == 0
    script = capsys.readouterr().out
    assert script.startswith("#!/bin/bash\n")
    assert "set -euo pipefail" in script
    assert script.count("modal volume put ") == 2
    assert script.count("modal volume ls ") == 2


@pytest.mark.parametrize(
    ("pair", "message"),
    [
        ("puf_2015_csv={other}", "the plan pins"),
        ("crosswalk={good}", "not an input of the plan"),
        ("puf_2015_csv={missing}", "is not a file"),
        ("puf_2015_csv", "expected ROLE=PATH"),
    ],
)
def test_upload_commands_refuse_a_file_the_plan_does_not_pin(
    tmp_path: Path, capsys, pair: str, message: str
) -> None:
    plan_path, files = _plan_with_local_files(tmp_path, ("puf_2015_csv",))
    other = tmp_path / "other.csv"
    other.write_text("not the pinned bytes")
    pair = pair.format(
        good=files["puf_2015_csv"], other=other, missing=tmp_path / "absent"
    )
    good = f"puf_2015_csv={files['puf_2015_csv']}"
    extra = [] if pair.startswith("puf_2015_csv") else [good]
    assert plan_lib.main(["upload-commands", str(plan_path), *extra, pair]) == 2
    captured = capsys.readouterr()
    # Refused as a whole: not even the good file's command is printed.
    assert captured.out == ""
    assert "REFUSED" in captured.err and message in captured.err


# --------------------------------------------------------------------------- #
# The push: the output first, and only the evidence after a clean exit        #
# --------------------------------------------------------------------------- #

# The base's state as the tool leaves it (4b57d15a2): a frame checkpoint per
# outer stage, the primary-QRF target checkpoints, the final checkpoint's
# alias, the run context and profile, the output and the runner's log.
_BASE_STATE = (
    "base-checkpoints/000_source_construction.frame.h5",
    "base-checkpoints/001_pre_clone_enrichment.frame.h5",
    "base-checkpoints/002_clone_feature_extraction.frame.h5",
    "base-checkpoints/asec_raw_stage.checkpoint.h5",
    "base-checkpoints/primary_qrf/targets/001__x.h5",
    "base-checkpoints/stage_all.frame.h5",
    "base-checkpoints/stage_profile.json",
    "base-checkpoints/stage_run_context.json",
    "base-out/base_populace_us_2024_puf_support.h5",
    "base-out/base_populace_us_2024_puf_support.summary.json",
    "logs/all-2026-10-01T000000Z.log",
)


def _base_push(tmp_path: Path, *, returncode: int, stopped: bool):
    state, volume = tmp_path / "state", tmp_path / "volume"
    for rel in _BASE_STATE:
        (state / rel).parent.mkdir(parents=True, exist_ok=True)
        (state / rel).write_bytes(rel.encode())
    volume.mkdir()
    stage = BASE.stages["all"]
    copied: list[str] = []
    real = plan_lib.copy_hashed

    def spy(source, target):
        copied.append(Path(source).relative_to(state).as_posix())
        return real(source, target)

    plan_lib.copy_hashed = spy
    try:
        outputs, counts, not_mirrored = plan_lib.push_state(
            state,
            volume,
            first=stage.mirror_first,
            copy_only=plan_lib.push_copy_only(
                stage, returncode=returncode, stopped_at_budget=stopped
            ),
        )
    finally:
        plan_lib.copy_hashed = real
    assert outputs == plan_lib.hash_tree(state)
    return copied, not_mirrored, volume


_BASE_EVIDENCE_ORDER = [
    "base-out/base_populace_us_2024_puf_support.h5",
    "base-out/base_populace_us_2024_puf_support.summary.json",
    "logs/all-2026-10-01T000000Z.log",
    "base-checkpoints/stage_run_context.json",
    "base-checkpoints/stage_profile.json",
    "base-checkpoints/000_source_construction.frame.h5",
    "base-checkpoints/001_pre_clone_enrichment.frame.h5",
]


def test_a_stopped_base_pushes_its_output_before_its_checkpoints(
    tmp_path: Path,
) -> None:
    # A push the class timeout cuts short has written the output and the
    # evidence first; the checkpoints the resume needs follow.
    copied, not_mirrored, _ = _base_push(tmp_path, returncode=-15, stopped=True)
    assert copied == [
        *_BASE_EVIDENCE_ORDER,
        "base-checkpoints/002_clone_feature_extraction.frame.h5",
        "base-checkpoints/asec_raw_stage.checkpoint.h5",
        "base-checkpoints/primary_qrf/targets/001__x.h5",
        "base-checkpoints/stage_all.frame.h5",
    ]
    assert not_mirrored == []


def test_a_finished_base_pushes_only_its_output_and_evidence(tmp_path: Path) -> None:
    # A completed --stage all has nothing to resume: the other checkpoints
    # (about 42 GB) are hashed into the receipt and not copied.
    copied, not_mirrored, volume = _base_push(tmp_path, returncode=0, stopped=False)
    assert copied == _BASE_EVIDENCE_ORDER
    assert not_mirrored == [
        "base-checkpoints/002_clone_feature_extraction.frame.h5",
        "base-checkpoints/asec_raw_stage.checkpoint.h5",
        "base-checkpoints/primary_qrf/targets/001__x.h5",
        "base-checkpoints/stage_all.frame.h5",
    ]
    assert sorted(plan_lib.tree_listing(volume)) == sorted(_BASE_EVIDENCE_ORDER)
    # A tool that exits 0 after a budget stop is still a stop: all is copied.
    _, not_mirrored, _ = _base_push(tmp_path / "stop0", returncode=0, stopped=True)
    assert not_mirrored == []


# --------------------------------------------------------------------------- #
# The runs volume's write rate against the runner's reserve                    #
# --------------------------------------------------------------------------- #


def test_write_probe_verdict_sizes_the_mirror_against_the_reserve() -> None:
    stage = BASE.stages["all"]
    gib = 1024**3
    # The reserve less the stop's grace and the time kept after the mirror
    # for the runner identity, the receipt and the final commit.
    reserve = (
        plan_lib.BASE_RUNNER_OVERHEAD_SECONDS
        - plan_lib.STOP_GRACE_SECONDS
        - plan_lib.RECEIPT_RESERVE_SECONDS
    )
    # The slowest rate that still mirrors 50 GiB inside that (about 33 MB/s).
    floor = 50 * gib / reserve
    assert floor / 1e6 == pytest.approx(33.1, abs=0.1)
    fast = plan_lib.write_probe_verdict(stage, gib, gib / (floor * 1.01))
    assert "problem" not in fast
    assert fast["mirror_reserve_seconds"] == reserve == 1620
    assert fast["receipt_reserve_seconds"] == 120
    slow = plan_lib.write_probe_verdict(stage, gib, gib / (floor * 0.99))
    assert "more than the 1620s the runner keeps for the mirror" in slow["problem"]
    # The runbook's one measured volume rate (reads, at least 58 MB/s) fits.
    assert "problem" not in plan_lib.write_probe_verdict(stage, 58_000_000, 1.0)
    acs = plan_lib.US_ACS_LOCAL_RELEASE.stages["materialize"]
    assert "skipped" in plan_lib.write_probe_verdict(acs, 0, 0.0)


def test_the_app_s_write_probe_times_a_committed_copy_and_cleans_up(
    app, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(plan_lib, "RUNS_MOUNT", str(tmp_path / "runs"))
    monkeypatch.setattr(plan_lib, "WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(plan_lib, "WRITE_PROBE_BYTES", 8 * 1024 * 1024)
    app.runs_volume.commit.reset_mock()
    verdict = app._write_probe(_committed_plan())
    assert verdict["bytes"] == 8 * 1024 * 1024
    assert verdict["seconds"] >= 0 and "mb_per_s" in verdict
    # The timed commit and the cleanup commit.
    assert app.runs_volume.commit.call_count == 2
    assert not list((tmp_path / "runs").rglob("*.bin*"))
    assert not list((tmp_path / "work").rglob("*.bin"))

    def broken(*_args, **_kwargs):
        raise OSError("volume full")

    monkeypatch.setattr(plan_lib, "copy_hashed", broken)
    failed = app._write_probe(_committed_plan())
    assert failed == {"problem": "write probe failed: OSError: volume full"}
    # A stage that declares no mirrored state is not probed.
    acs = dataclasses.replace(_committed_plan(), tool=plan_lib.US_ACS_LOCAL_RELEASE)
    acs = dataclasses.replace(acs, stage="materialize")
    assert "skipped" in app._write_probe(acs)


# --------------------------------------------------------------------------- #
# The runbook: the hand-off block, the registered tools, the pinned tests      #
# --------------------------------------------------------------------------- #

RUNBOOK = ROOT / "docs" / "us-modal-stage-runbook.md"
RUN_ID = "route-a-base-4b57d15a287c"


def _hand_off_block() -> str:
    """Step 7 of the runbook's base section, verbatim, as a bash script."""

    text = RUNBOOK.read_text()
    start = text.index("```bash\n# 7. $RUN is Route A's")
    end = text.index("\n```", start + 3)
    block = text[start + len("```bash\n") : end]
    assert block.count("RUN=<Route A run directory>\n") == 1
    return block.replace("RUN=<Route A run directory>\n", 'RUN="$HANDOFF_RUN"\n')


def _hand_off(tmp_path: Path, *, config: dict | None = None, run: Path | None = None):
    """Run step 7 against a Modal run of the committed plan and a Route A
    run directory holding the failed local run's checkpoints."""

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for name in ("tools", "docs"):
        (checkout / name).symlink_to(ROOT / name)
    modal_run = checkout / "modal-runs" / RUN_ID
    base_out = modal_run / "state" / plan_lib.PUF_SUPPORT_BASE_OUT
    base_out.mkdir(parents=True)
    (base_out / plan_lib.PUF_SUPPORT_BASE_ARTIFACT).write_bytes(b"\x89HDF base")
    (base_out / "base_populace_us_2024_puf_support.summary.json").write_text("{}")
    reference = _reference()
    context = {
        "pipeline_sha256": reference["pipeline_sha256"],
        "run_config": config if config is not None else _modal_run_config(),
    }
    (modal_run / "stage_run_context.json").write_bytes(_context_bytes(context))
    receipt = _modal_receipt(context)
    receipt["outputs"] += [
        {**item, "path": f"{plan_lib.PUF_SUPPORT_BASE_OUT}/{item['path']}"}
        for item in plan_lib.hash_tree(base_out)
    ]
    (modal_run / "receipts").mkdir()
    receipt_name = "all-2026-10-01T120000Z.json"
    (modal_run / "receipts" / receipt_name).write_text(json.dumps(receipt))
    if run is None:
        run = tmp_path / "route-a-run"
        (run / "base-checkpoints").mkdir(parents=True)
        (run / "base-checkpoints" / "000_source_construction.frame.h5").write_bytes(
            b"local"
        )
    # python3 is this interpreter (the plan module needs only the standard
    # library); shasum is the system's, or sha256sum where it is missing.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").symlink_to(sys.executable)
    if shutil.which("shasum") is None:
        shim = bin_dir / "shasum"
        shim.write_text('#!/bin/sh\nshift 2\nexec sha256sum "$@"\n')
        shim.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "R": RUN_ID,
        "RECEIPT": f"./modal-runs/{RUN_ID}/receipts/{receipt_name}",
        "HANDOFF_RUN": str(run),
    }
    proc = subprocess.run(
        ["bash", "-c", _hand_off_block()],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc, run, modal_run


def _result_accepted(result: Path) -> str | None:
    """route_a.sh's result_accepted for "0": the return code, or None."""

    text = result.read_text()
    if '"refusal": null' not in text:
        return None
    codes = re.findall(r'^ *"returncode": (-?[0-9]*),*$', text, re.MULTILINE)
    return codes[0] if codes == ["0"] else None


def test_the_hand_off_runs_compare_lineage_and_keeps_its_report(
    tmp_path: Path,
) -> None:
    proc, run, modal_run = _hand_off(tmp_path)
    assert proc.returncode == 0, proc.stderr
    sup = run / "base-sup"
    assert sorted(path.name for path in sup.iterdir()) == [
        "ACCEPTED",
        "RESULT.json",
        "compare-lineage.json",
        "modal-receipt.json",
        "modal-stage_run_context.json",
    ]
    report = json.loads((sup / "compare-lineage.json").read_text())
    assert (report["compared_outputs_match"], report["problems"]) == (True, 0)
    assert report["thread_environment_compared"] == {
        "PYTHONHASHSEED": {"modal": "0", "local": "0"}
    }
    assert (sup / "modal-stage_run_context.json").read_bytes() == (
        modal_run / "stage_run_context.json"
    ).read_bytes()
    result = json.loads((sup / "RESULT.json").read_text())
    assert result["compare_lineage"] == "compare-lineage.json"
    assert _result_accepted(sup / "RESULT.json") == "0"
    # The base is the Modal bytes, and the local checkpoints are kept aside.
    h5 = run / "base-out" / plan_lib.PUF_SUPPORT_BASE_ARTIFACT
    assert h5.read_bytes() == b"\x89HDF base"
    digest = hashlib.sha256(h5.read_bytes()).hexdigest()
    assert (run / "base.sha256").read_text() == f"{digest}  {h5}\n"
    assert not (run / "base-checkpoints").exists()
    assert len(list(run.glob("base-checkpoints.local-*"))) == 1


@pytest.mark.parametrize("seed", ["12345", None])
def test_the_hand_off_stops_before_writing_when_compare_lineage_fails(
    tmp_path: Path, seed: str | None
) -> None:
    # A Modal run that is not the local run's build (here: another hash
    # seed) never reaches Route A: compare-lineage ends the block before
    # anything under $RUN is written.
    config = _modal_run_config()
    if seed is None:
        del config["thread_environment"]["PYTHONHASHSEED"]
    else:
        config["thread_environment"]["PYTHONHASHSEED"] = seed
    proc, run, modal_run = _hand_off(tmp_path, config=config)
    assert proc.returncode == 1
    assert "run_config.thread_environment.PYTHONHASHSEED" in proc.stderr
    assert sorted(path.name for path in run.iterdir()) == ["base-checkpoints"]
    assert [p.name for p in (run / "base-checkpoints").iterdir()] == [
        "000_source_construction.frame.h5"
    ]
    report = json.loads((modal_run / "compare-lineage.json").read_text())
    assert report["compared_outputs_match"] is False


def test_the_hand_off_never_overwrites_a_base(tmp_path: Path) -> None:
    run = tmp_path / "route-a-run"
    (run / "base-out").mkdir(parents=True)
    proc, run, modal_run = _hand_off(tmp_path, run=run)
    assert proc.returncode == 1
    assert "never overwrite a base" in proc.stderr
    assert sorted(path.name for path in run.iterdir()) == ["base-out"]
    # Refused before compare-lineage ran.
    assert not (modal_run / "compare-lineage.json").exists()


def test_the_runbook_names_every_registered_tool() -> None:
    text = RUNBOOK.read_text()
    intro = text[: text.index("Two files do the work")]
    assert len(plan_lib.TOOLS) == 3
    assert "Three tools are registered" in intro
    for name, tool in plan_lib.TOOLS.items():
        assert f"`{name}`" in intro, name
        if tool.script is not None:
            assert f"`{tool.script}`" in intro, tool.script


#: The tests that read the plan's commit with ``git show`` and skip without
#: it (CI's depth-1 checkout); the runbook names each.
PINNED_COMMIT_TESTS = (
    "test_the_tool_s_own_parser_reads_the_same_build_from_both_commands",
    "test_the_owned_flags_are_the_pinned_tool_s_flags",
    "test_the_registered_crosswalk_digest_is_4b57d15a2_s",
    "test_the_pinned_tool_locks_the_reference_s_run_config_from_the_modal_argv",
)


def test_the_runbook_names_every_pinned_commit_test() -> None:
    module = sys.modules[__name__]
    tests = {
        name: function
        for name, function in vars(module).items()
        if name.startswith("test_") and callable(function)
    }
    pinned = {"pinned_source", "pinned_tool"}
    uses_pinned = {
        name
        for name, function in tests.items()
        if pinned & set(inspect.signature(function).parameters)
    }
    assert uses_pinned == set(PINNED_COMMIT_TESTS)
    text = RUNBOOK.read_text()
    for name in PINNED_COMMIT_TESTS:
        assert f"`{name}`" in text, name
    assert "four tests in\n  `test_us_modal_stage_puf_support_base.py`" in text
