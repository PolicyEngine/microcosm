"""Engine-free registration and d844 plan checks for the Modal scorer stage."""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_support.microcosm_build.us_modal_stage_plan_tool import (
    ROOT,
    load_app,
    plan_lib,
)

PLAN_PATH = ROOT / "docs" / "us-modal-stage-route-a-h2h-plan.json"
SCORER = ROOT / "tools" / "score_us_release_head_to_head.py"
INPUTS = {
    "incumbent_h5": (
        "populace_us_2024.h5",
        "6496cc4393d4d3c6574f76eca231de5898c803b9067645591fd5c4d3e65aee84",
    ),
    "candidate_h5": (
        "populace_us_2024.h5",
        "417d23aed4d044de2e657a7b5d856136975d25739dbdb4ddd72866f841de1559",
    ),
    "ledger_facts": (
        "consumer_facts_us_c5e5bf8.jsonl",
        "b85437390021777e746f507c5890305496baf5fc7f2c78ba08ddb090f4839801",
    ),
}
LOCAL_ARGV = [
    "--incumbent",
    "/local/incumbent/populace_us_2024.h5",
    "--candidate",
    "/local/candidate/populace_us_2024.h5",
    "--ledger-facts",
    "/local/consumer_facts_us_c5e5bf8.jsonl",
    "--out-prefix",
    "/local/h2h",
    "--maximum-microsim-batch-size",
    "2000",
]
PATH_FLAGS = {
    "--incumbent": "incumbent_h5",
    "--candidate": "candidate_h5",
    "--ledger-facts": "ledger_facts",
}


def _plan_data() -> dict:
    return json.loads(PLAN_PATH.read_text())


def _plan() -> plan_lib.Plan:
    return plan_lib.parse_plan(_plan_data())


@pytest.fixture(scope="module")
def scorer_parser():
    """Run only the scorer's argparse functions, without its engine imports.

    Its BooleanOptionalAction generates --no-* aliases, which a regex of
    add_argument calls would miss. Capturing the real parser also catches
    the maximum-microsimulation spelling alias.
    """

    parsers = []

    class RecordingParser(argparse.ArgumentParser):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            parsers.append(self)

    namespace = {
        "argparse": SimpleNamespace(
            ArgumentParser=RecordingParser,
            ArgumentTypeError=argparse.ArgumentTypeError,
            BooleanOptionalAction=argparse.BooleanOptionalAction,
            Namespace=argparse.Namespace,
        ),
        "Path": Path,
        "Sequence": Sequence,
        # The run passes its batch size explicitly. Loading the release
        # module just to provide this unused default would import the engine.
        "release": SimpleNamespace(DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE=2000),
    }
    functions = [
        node
        for node in ast.parse(SCORER.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_parse_args", "_positive_workers"}
    ]
    assert {node.name for node in functions} == {"_parse_args", "_positive_workers"}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(SCORER), "exec"),
        namespace,
    )
    parse = namespace["_parse_args"]
    parse(LOCAL_ARGV)
    return SimpleNamespace(parse=parse, parser=parsers[0])


def test_d844_plan_validates_with_digest_pinned_loader_filenames() -> None:
    plan = _plan()
    tool = plan_lib.US_RELEASE_HEAD_TO_HEAD
    assert plan.tool is tool
    assert (tool.name, tool.script, plan.stage) == (
        "us-release-head-to-head",
        "tools/score_us_release_head_to_head.py",
        "score",
    )
    assert set(tool.stages) == {"score"}
    assert set(tool.inputs) == set(tool.stages["score"].required_inputs) == set(INPUTS)
    assert plan.options == {"maximum_microsim_batch_size": 2000, "workers": 20}
    assert plan.nonpreemptible is True
    assert plan.max_wall_seconds == 15_900
    for role, (filename, digest) in INPUTS.items():
        ref = plan.inputs[role]
        assert (ref.kind, ref.filename, ref.sha256) == ("volume", filename, digest)
        assert ref.volume_path == f"cas/sha256/{digest}/{filename}"
        assert plan_lib.input_local_path("/work", ref) == (
            f"/work/inputs/{role}/{filename}"
        )
    # The two loader filenames stay identical but are staged independently.
    assert plan.inputs["incumbent_h5"].filename == "populace_us_2024.h5"
    assert plan.inputs["candidate_h5"].filename == "populace_us_2024.h5"
    assert plan.stage_spec.mirror_first == ("h2h.json", "h2h.md")


def test_modal_argv_equals_the_local_d844_command_apart_from_paths_and_workers(
    scorer_parser,
) -> None:
    plan = _plan()
    argv = plan_lib.planned_argv(plan)
    assert argv[:3] == [
        "/opt/venv/bin/python",
        "-B",
        "tools/score_us_release_head_to_head.py",
    ]
    expected = list(LOCAL_ARGV)
    for index, item in enumerate(expected[:-1]):
        if item in PATH_FLAGS:
            ref = plan.inputs[PATH_FLAGS[item]]
            expected[index + 1] = plan_lib.input_local_path("/work", ref)
        elif item == "--out-prefix":
            expected[index + 1] = "/work/state/h2h"
    assert argv[3:] == [*expected, "--workers", "20"]
    local_args = vars(scorer_parser.parse(LOCAL_ARGV))
    modal_args = vars(scorer_parser.parse(argv[3:]))
    for flag in (*PATH_FLAGS, "--out-prefix"):
        name = flag.removeprefix("--").replace("-", "_")
        assert isinstance(local_args.pop(name), Path)
        assert isinstance(modal_args.pop(name), Path)
    assert local_args.pop("workers") == 1
    assert modal_args.pop("workers") == 20
    assert local_args == modal_args
    assert modal_args["age_targets"] is False
    assert modal_args["allow_unaged_dollar_targets"] is True
    assert modal_args["maximum_microsim_batch_size"] == 2000
    # Each scorer worker gets one native numerical thread.
    assert all(plan.env[name] == "1" for name in plan_lib.THREAD_ENV_KEYS)


def test_owned_flags_cover_the_scorer_surface_except_the_two_plan_options(
    scorer_parser,
) -> None:
    tool = plan_lib.US_RELEASE_HEAD_TO_HEAD
    flags = {
        flag
        for flag in scorer_parser.parser._option_string_actions
        if flag.startswith("--")
    } - {"--help"}
    option_flags = {"--workers", "--maximum-microsim-batch-size"}
    assert {option.flag for option in tool.options.values()} == option_flags
    assert set(tool.options) == {"workers", "maximum_microsim_batch_size"}
    assert tool.owned_flags == flags - option_flags
    assert {
        "--age-targets",
        "--no-age-targets",
        "--allow-unaged-dollar-targets",
        "--no-allow-unaged-dollar-targets",
        "--maximum-microsimulation-batch-size",
    } <= tool.owned_flags


@pytest.mark.parametrize(
    "key",
    [
        "incumbent",
        "candidate",
        "ledger_facts",
        "out_prefix",
        "candidate_manifest_sha256",
        "candidate_worker_identity_attestation",
        "congressional_district_vintage_crosswalk",
        "age_targets",
        "no_age_targets",
        "allow_unaged_dollar_targets",
        "no_allow_unaged_dollar_targets",
        "maximum_microsimulation_batch_size",
    ],
)
def test_paths_and_boolean_overrides_are_not_plan_options(key: str) -> None:
    data = _plan_data()
    data["options"][key] = True
    with pytest.raises(plan_lib.PlanError, match="not allowlisted"):
        plan_lib.parse_plan(data)


@pytest.mark.parametrize("value", ["20", True])
def test_workers_must_be_an_integer_in_the_plan(value) -> None:
    data = _plan_data()
    data["options"]["workers"] = value
    with pytest.raises(plan_lib.PlanError, match="must be int"):
        plan_lib.parse_plan(data)


def test_head_to_head_timeout_covers_the_slice_estimate_and_cost_cap() -> None:
    plan = _plan()
    resources = plan_lib.HEAD_TO_HEAD
    assert plan.resources is resources
    assert (resources.cpu, resources.cpu_limit, resources.memory_gib) == (20, 20, 224)
    assert resources.timeout_s == 17_100
    slice_wall_seconds = 1030 * 4.7 * 60 / plan.options["workers"]
    # At least ten minutes remain for initialization before the tool budget;
    # the container additionally reserves runner time and timeout margin.
    assert plan.max_wall_seconds >= slice_wall_seconds + 10 * 60
    assert resources.timeout_s > (
        plan.max_wall_seconds + plan.stage_spec.runner_overhead_seconds
    )
    per_second = (
        20 * plan_lib.CPU_USD_PER_CORE_SECOND + 224 * plan_lib.MEMORY_USD_PER_GIB_SECOND
    )
    ceiling = round(per_second * 17_100 * 3, 2)
    assert ceiling == pytest.approx(38.95, abs=0.005)
    assert ceiling <= plan_lib.HEAD_TO_HEAD_COST_CAP_USD == 40
    assert resources.estimated_usd(resources.timeout_s, 3) == ceiling
    assert plan_lib.estimated_usd_at_timeout(plan) == ceiling
    assert plan_lib.summarize(plan)["estimated_usd_at_timeout"] == ceiling


def test_head_to_head_registers_its_modal_functions_without_retries(
    monkeypatch,
) -> None:
    app = load_app(monkeypatch)
    resources = plan_lib.HEAD_TO_HEAD
    for nonpreemptible in (False, True):
        options = app.RUNNERS[(resources.name, nonpreemptible)].modal_options
        assert (options["cpu"], options["memory"], options["timeout"]) == (
            (20, 20),
            224 * 1024,
            17_100,
        )
        assert options.get("nonpreemptible", False) is nonpreemptible
        assert options["retries"] == 0
    plan = _plan()
    runner = app.RUNNERS[(plan.resources.name, plan.nonpreemptible)]
    assert runner is app.run_stage_head_to_head_nonpreemptible
    calls = []
    monkeypatch.setattr(
        app, "_run_stage", lambda data: calls.append(data) or {"ok": True}
    )
    data = _plan_data()
    assert runner(data) == {"ok": True}
    assert calls == [data]
