"""Unit tests for the US Modal stage runner.

``tools/modal_us_stage_plan.py`` builds the plan, argv and sha256 receipts
that ``tools/modal_us_stage.py`` executes on Modal. The app's helpers are
imported against a stub ``modal`` module, so nothing here needs the Modal
client, a Modal connection or the network (git runs against a local repo).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from test_support.microcosm_build.us_modal_stage_plan_tool import (
    COMMIT,
    FEED_SHA,
    LADDER_SHA,
    ROOT,
    STAGING_SHA,
    plan_lib,
)
from test_support.microcosm_build.us_modal_stage_plan_tool import (
    plan_data as _plan_data,
)
from test_support.microcosm_build.us_modal_stage_plan_tool import (
    smoke_plan_data as _smoke_plan_data,
)


@pytest.fixture
def app(monkeypatch):
    """``tools/modal_us_stage.py`` imported against a stub ``modal`` module.

    The stub answers ``is_local()`` with False, so the module defines its
    image and functions without reading a plan or contacting Modal; volume
    calls (commit, reload) are recorded, not performed.
    """

    stub = MagicMock(name="modal")
    stub.is_local.return_value = False
    # @app.function(...) keeps the function and records its Modal options.
    stub.App.return_value.function.side_effect = lambda **options: (
        lambda function: setattr(function, "modal_options", options) or function
    )
    stub.App.return_value.local_entrypoint.side_effect = lambda **_: lambda f: f
    stub.current_input_id.return_value = "in-test"
    stub.current_function_call_id.return_value = "fc-test"
    monkeypatch.setitem(sys.modules, "modal", stub)
    monkeypatch.setitem(sys.modules, "modal_us_stage_plan", plan_lib)
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location(
        "modal_us_stage_under_test", ROOT / "tools" / "modal_us_stage.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Plan validation                                                              #
# --------------------------------------------------------------------------- #


def test_valid_plan_parses_with_stage_resources() -> None:
    plan = plan_lib.parse_plan(_plan_data())
    assert plan.stage == "materialize"
    assert plan.resources is plan_lib.HEAVY
    assert plan.repo_url == plan_lib.DEFAULT_REPO_URL
    assert set(plan.inputs) == {"staging_h5", "staging_summary", "feed", "ladder"}
    ladder = plan.inputs["ladder"]
    assert (ladder.kind, ladder.repo_type, ladder.repo_id) == (
        "hf",
        "dataset",
        "policyengine/populace-us",
    )
    assert ladder.revision == "populace-us-2024-spm-receipts-20260923"
    assert ladder.path_in_repo == "inputs/us_puma_ladder_2020.npz"
    assert plan.inputs["feed"].volume_path.endswith("/consumer_facts.jsonl")


@pytest.mark.parametrize(
    ("stage", "resources"),
    [
        ("materialize", "heavy"),
        ("calibrate", "heavy"),
        ("all", "heavy"),
        ("qa", "light"),
        ("finalize", "light"),
        ("package", "light"),
    ],
)
def test_every_stage_maps_to_a_sized_class(stage: str, resources: str) -> None:
    assert plan_lib.parse_plan(_plan_data(stage)).resources.name == resources


def test_heavy_class_covers_measured_peaks_with_headroom() -> None:
    for (tool, stage), measured in plan_lib.MEASURED.items():
        spec = plan_lib.TOOLS[tool].stages[stage]
        request_bytes = spec.resources.memory_mib * 1024 * 1024
        assert request_bytes >= 1.5 * measured.peak_rss_bytes, (stage, measured)
    # The state SOI surface reported ~94 GB for materialize (2026-09-22).
    assert plan_lib.HEAVY.memory_mib * 1024 * 1024 > 94e9 * 1.3


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(schema="other/1"), "schema"),
        (lambda d: d.update(tool="us-nope"), "unknown tool"),
        (lambda d: d.update(stage="publish"), "no stage"),
        (lambda d: d.update(run_id="Bad Run"), "run_id"),
        (lambda d: d["source"].update(commit="4d773a4"), "40-hex"),
        (lambda d: d["source"].pop("branch"), "branch"),
        (lambda d: d["source"].update(branch="../x"), "branch"),
        (
            lambda d: d["source"].update(repo_url="https://evil.example/x.git"),
            "repo_url",
        ),
        (lambda d: d.update(extra=1), "unknown plan keys"),
        (lambda d: d["inputs"].pop("feed"), "requires inputs"),
        (
            lambda d: d["inputs"].update(puf={"uri": "volume://x", "sha256": "d" * 64}),
            "takes no inputs",
        ),
        (lambda d: d["inputs"]["feed"].update(sha256="ABC"), "64 lowercase hex"),
        (lambda d: d["inputs"]["feed"].pop("sha256"), "exactly"),
        (
            lambda d: d["inputs"]["feed"].update(uri="s3://bucket/feed.jsonl"),
            "unsupported uri",
        ),
        (
            lambda d: d["inputs"]["feed"].update(uri="volume://../etc/passwd"),
            "clean relative",
        ),
        (
            lambda d: d["inputs"]["feed"].update(uri="volume:///abs/feed.jsonl"),
            "clean relative",
        ),
        (
            lambda d: d["inputs"]["feed"].update(
                uri=f"volume://cas/sha256/{'e' * 64}/consumer_facts.jsonl"
            ),
            "own digest",
        ),
        (
            lambda d: d["inputs"]["ladder"].update(
                uri="hf://datasets/policyengine/populace-us/inputs/x.npz"
            ),
            "@revision",
        ),
        (
            lambda d: d["inputs"]["ladder"].update(
                uri="hf://spaces/policyengine/populace-us@main/x.npz"
            ),
            "hf://datasets/",
        ),
        (lambda d: d["options"].update(allow_dirty=True), "not allowlisted"),
        (lambda d: d["options"].update(checkpoint_dir="/tmp"), "not allowlisted"),
        (lambda d: d["options"].update(hh_chunk="20000"), "must be int"),
        (lambda d: d["options"].update(hh_chunk=True), "must be int"),
        (lambda d: d["options"].update(resume="yes"), "must be bool"),
        (lambda d: d["options"].update(soi_mode="--out=/x"), "must be str"),
        (lambda d: d.update(env={"HF_TOKEN": "x"}), "not allowlisted"),
        (lambda d: d.update(env={"MICROCOSM_X": 1}), "must be a string"),
    ],
)
def test_loose_or_unsafe_plans_are_refused(mutate, message: str) -> None:
    data = copy.deepcopy(_plan_data())
    mutate(data)
    with pytest.raises(plan_lib.PlanError, match=message):
        plan_lib.parse_plan(data)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["source"].update(commit=COMMIT + "\n"), "40-hex"),
        (lambda d: d.update(run_id="acs-local-20260923\n"), "run_id"),
        (lambda d: d["source"].update(branch="us-modal-stage-runner\n"), "branch"),
        (
            lambda d: d["source"].update(repo_url=plan_lib.DEFAULT_REPO_URL + "\n"),
            "repo_url",
        ),
        (lambda d: d.update(env={"MICROCOSM_X\n": "1"}), "not allowlisted"),
        (lambda d: d["inputs"]["feed"].update(sha256=FEED_SHA + "\n"), "64 lowercase"),
        (
            lambda d: d["inputs"]["feed"].update(
                uri=f"volume://cas/sha256/{FEED_SHA}/consumer_facts.jsonl\n"
            ),
            "unsafe file name",
        ),
        (
            lambda d: d["inputs"]["ladder"].update(
                uri="hf://datasets/policyengine/populace-us@main\n/x.npz"
            ),
            "@revision",
        ),
    ],
)
def test_a_trailing_newline_never_matches(mutate, message: str) -> None:
    # re.match with a "$" anchor accepts "value\n"; every field uses fullmatch.
    data = copy.deepcopy(_plan_data())
    mutate(data)
    with pytest.raises(plan_lib.PlanError, match=message):
        plan_lib.parse_plan(data)


@pytest.mark.parametrize("field", ["tool", "stage"])
@pytest.mark.parametrize("value", [["materialize"], {"a": 1}, 3, None])
def test_non_string_tool_or_stage_is_a_plan_error(field: str, value) -> None:
    data = _plan_data()
    data[field] = value
    with pytest.raises(plan_lib.PlanError):
        plan_lib.parse_plan(data)


def test_validate_cli_refuses_a_non_string_stage(tmp_path: Path, capsys) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan_data(stage=["materialize"])))
    assert plan_lib.main(["validate", str(plan_path)]) == 2
    assert "REFUSED" in capsys.readouterr().err


@pytest.mark.parametrize(
    "key",
    [
        "POPULACE_LEDGER_API_KEY",
        "POPULACE_LEDGER_KEY",
        "POPULACE_LEDGER_EXPORT_KEY",
        "MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY",
        "MICROCOSM_HF_TOKEN",
        "POPULACE_DB_PASSWORD",
        "MICROCOSM_CLIENT_SECRET",
        "POPULACE_SERVICE_CREDENTIALS",
    ],
)
def test_credential_env_keys_are_refused_under_allowlisted_prefixes(key: str) -> None:
    with pytest.raises(plan_lib.PlanError, match="names a credential"):
        plan_lib.parse_plan(_plan_data(env={key: "x"}))


def test_non_credential_env_keys_still_pass() -> None:
    env = {
        "MICROCOSM_ACS_POOL_PEAK_LIMIT_BYTES": "100000000000",
        "POPULACE_FIT_N_JOBS": "1",
        "OMP_NUM_THREADS": "4",
    }
    assert dict(plan_lib.parse_plan(_plan_data(env=env)).env) == env


def test_the_committed_plans_parse() -> None:
    for name in (
        "us-modal-stage-example-plan.json",
        "us-modal-stage-smoke-plan.json",
        "us-modal-stage-acceptance-20260923-plan.json",
    ):
        plan_lib.parse_plan(json.loads((ROOT / "docs" / name).read_text()))


def test_later_stages_do_not_need_the_feed() -> None:
    data = _plan_data("calibrate")
    data["inputs"].pop("feed")
    plan = plan_lib.parse_plan(data)
    assert "--feed" not in plan_lib.planned_argv(plan)


def test_plan_digest_is_key_order_independent() -> None:
    data = _plan_data()
    reordered = json.loads(json.dumps(data, sort_keys=True))
    assert plan_lib.plan_digest(data) == plan_lib.plan_digest(reordered)
    changed = _plan_data(run_id="acs-local-20260924")
    assert plan_lib.plan_digest(data) != plan_lib.plan_digest(changed)


# --------------------------------------------------------------------------- #
# Argv                                                                         #
# --------------------------------------------------------------------------- #


def test_materialize_argv_is_exact() -> None:
    plan = plan_lib.parse_plan(_plan_data())
    assert plan_lib.planned_argv(plan) == [
        "/opt/venv/bin/python",
        "-B",
        "tools/build_us_acs_local_release.py",
        "--stage",
        "materialize",
        "--staging-h5",
        "/work/inputs/staging_h5/acs_multispine_staging.h5",
        "--staging-summary",
        "/work/inputs/staging_summary/acs_multispine_staging.summary.json",
        "--ladder",
        "/work/inputs/ladder/us_puma_ladder_2020.npz",
        "--feed",
        "/work/inputs/feed/consumer_facts.jsonl",
        "--feed-sha256",
        FEED_SHA,
        "--checkpoint-dir",
        "/work/state/checkpoints",
        "--out-h5",
        "/work/state/populace_us_2024_acs_local.h5",
        "--hh-chunk",
        "20000",
        "--soi-mode",
        "totals",
    ]


def test_package_argv_adds_release_root_and_bool_flags() -> None:
    data = _plan_data("package")
    data["options"] = {"resume": False, "allow_partial_geography": True}
    argv = plan_lib.planned_argv(plan_lib.parse_plan(data))
    assert argv[argv.index("--out") + 1] == "/work/state/out"
    assert "--allow-partial-geography" in argv
    assert "--resume" not in argv
    assert "--allow-dirty" not in argv


def test_argv_paths_are_stable_across_stages() -> None:
    # run_identity.json records the staging path and every later stage
    # re-verifies the staging digest, so staged paths must not move.
    argvs = [
        plan_lib.planned_argv(plan_lib.parse_plan(_plan_data(stage)))
        for stage in ("materialize", "calibrate", "qa", "finalize", "package")
    ]
    for argv in argvs:
        for flag in ("--staging-h5", "--checkpoint-dir", "--out-h5", "--ladder"):
            assert argv[argv.index(flag) + 1] == argvs[0][argvs[0].index(flag) + 1]


def test_no_option_maps_to_a_runner_owned_flag() -> None:
    for tool in plan_lib.TOOLS.values():
        flags = {option.flag for option in tool.options.values()}
        assert not flags & tool.owned_flags


def test_build_stage_argv_requires_every_staged_path() -> None:
    plan = plan_lib.parse_plan(_plan_data())
    with pytest.raises(plan_lib.PlanError, match="no staged path"):
        plan_lib.build_stage_argv(
            plan, python="python", input_paths={"staging_h5": "x"}, state_dir="/s"
        )


def test_image_pins_the_commit_and_the_lock() -> None:
    plan = plan_lib.parse_plan(_plan_data())
    commands = plan_lib.image_build_commands(plan)
    joined = "\n".join(commands)
    assert f"fetch -q --depth 1 origin {COMMIT}" in joined
    assert f'test "$(git -C /root/microcosm rev-parse HEAD)" = {COMMIT}' in joined
    assert "checkout -q -B us-modal-stage-runner FETCH_HEAD" in joined
    assert "uv sync --all-packages --extra us --frozen" in joined
    # The clean-tree assertion runs after the sync, so the release tool's
    # own dirty-tree refusal and recorded sha match the pushed commit.
    assert commands[-1].startswith('test -z "$(git')


# --------------------------------------------------------------------------- #
# Hashing, mirroring, receipts                                                 #
# --------------------------------------------------------------------------- #


def test_copy_with_sha256_hashes_in_the_same_pass(tmp_path: Path) -> None:
    payload = os.urandom(3 * 1024 * 1024 + 17)
    src = tmp_path / "src.bin"
    src.write_bytes(payload)
    sha, size = plan_lib.copy_with_sha256(src, tmp_path / "a" / "b" / "dst.bin")
    assert (sha, size) == (hashlib.sha256(payload).hexdigest(), len(payload))
    assert (tmp_path / "a" / "b" / "dst.bin").read_bytes() == payload
    assert not list(tmp_path.rglob("*.partial"))


def test_verify_digest_refuses_a_mismatch() -> None:
    ref = plan_lib.parse_plan(_plan_data()).inputs["feed"]
    plan_lib.verify_digest(ref, FEED_SHA)
    with pytest.raises(plan_lib.PlanError, match="plan pins"):
        plan_lib.verify_digest(ref, "0" * 64)


def test_mirror_actions_copy_changed_and_delete_removed() -> None:
    source = {"a": (1, 10), "b": (2, 20), "c": (3, 30)}
    destination = {"a": (1, 10), "b": (2, 99), "stale.mmap": (5, 50)}
    copy_, delete = plan_lib.mirror_actions(source, destination)
    assert copy_ == ["b", "c"]
    assert delete == ["stale.mmap"]


def test_mirror_tree_round_trip_is_a_no_op(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "checkpoints").mkdir(parents=True)
    (src / "checkpoints" / "targets.json").write_text("[]")
    (src / "artifact.h5").write_bytes(b"h5")
    counts = plan_lib.mirror_tree(src, dst)
    assert counts == {"copied": 2, "deleted": 0, "partials_removed": 0}
    counts = plan_lib.mirror_tree(src, dst)
    assert counts == {"copied": 0, "deleted": 0, "partials_removed": 0}
    (src / "artifact.h5").unlink()
    counts = plan_lib.mirror_tree(src, dst)
    assert counts == {"copied": 0, "deleted": 1, "partials_removed": 0}
    assert plan_lib.tree_listing(dst).keys() == {"checkpoints/targets.json"}


def _receipt(tmp_path: Path, returncode: int = 0) -> tuple[dict, Path]:
    state = tmp_path / "state"
    (state / "checkpoints").mkdir(parents=True)
    (state / "checkpoints" / "run_identity.json").write_text('{"staging_sha256": "x"}')
    (state / "populace_us_2024_acs_local.h5").write_bytes(b"\x89HDF" * 100)
    data = _plan_data()
    plan = plan_lib.parse_plan(data)
    receipt = plan_lib.build_receipt(
        plan,
        data,
        argv=plan_lib.planned_argv(plan),
        returncode=returncode,
        started_at="2026-09-23T03:00:00Z",
        finished_at="2026-09-23T04:24:27Z",
        wall_seconds=5067.1,
        peak_rss_bytes=74_760_110_080,
        inputs_verified=[{"name": "feed", "sha256": FEED_SHA, "bytes": 1}],
        outputs=plan_lib.hash_tree(state),
        git={"head": COMMIT, "tree_clean": True},
        runner={"python": "3.13"},
    )
    return json.loads(json.dumps(receipt)), state


def test_receipt_records_plan_source_outputs_and_cost(tmp_path: Path) -> None:
    receipt, state = _receipt(tmp_path)
    assert receipt["schema"] == plan_lib.RECEIPT_SCHEMA
    assert receipt["status"] == "COMPLETED"
    assert receipt["plan_sha256"] == plan_lib.plan_digest(_plan_data())
    assert receipt["source"]["commit"] == COMMIT
    assert receipt["resources"] == {
        "class": "heavy",
        "cpu": 4.0,
        "memory_mib": 131072,
        "timeout_s": 28800,
        "nonpreemptible": False,
    }
    paths = [item["path"] for item in receipt["outputs"]]
    assert paths == ["checkpoints/run_identity.json", "populace_us_2024_acs_local.h5"]
    h5 = receipt["outputs"][1]
    assert h5["sha256"] == hashlib.sha256(b"\x89HDF" * 100).hexdigest()
    # 4 cores + 128 GiB at list price for the measured materialize wall.
    assert receipt["estimated_usd_at_list_price"] == pytest.approx(1.71, abs=0.01)
    assert plan_lib.verify_receipt(receipt, state) == []


def test_receipt_prices_the_whole_container_when_given(tmp_path: Path) -> None:
    receipt, _ = _receipt(tmp_path)
    assert "container_wall_seconds" not in receipt
    data = _plan_data()
    plan = plan_lib.parse_plan(data)
    priced = plan_lib.build_receipt(
        plan,
        data,
        argv=plan_lib.planned_argv(plan),
        returncode=0,
        started_at="2026-09-23T03:00:00Z",
        finished_at="2026-09-23T04:24:27Z",
        wall_seconds=5067.1,
        peak_rss_bytes=None,
        inputs_verified=[],
        outputs=[],
        git={"head": COMMIT, "tree_clean": True},
        runner={},
        container_wall_seconds=5067.1 + 3600,
    )
    assert priced["container_wall_seconds"] == pytest.approx(8667.1)
    # One more hour of the heavy class at list price: about $1.21.
    assert priced["estimated_usd_container_at_list_price"] - priced[
        "estimated_usd_at_list_price"
    ] == pytest.approx(1.21, abs=0.01)


def test_failed_stage_receipt_is_marked_failed(tmp_path: Path) -> None:
    receipt, _ = _receipt(tmp_path, returncode=1)
    assert receipt["status"] == "FAILED"


def test_verify_receipt_reports_tampering(tmp_path: Path) -> None:
    receipt, state = _receipt(tmp_path)
    (state / "populace_us_2024_acs_local.h5").write_bytes(b"\x89HDF" * 99 + b"XXXX")
    (state / "checkpoints" / "run_identity.json").unlink()
    (state / "extra.json").write_text("{}")
    problems = plan_lib.verify_receipt(receipt, state)
    assert problems == [
        "missing: checkpoints/run_identity.json",
        "sha256 mismatch: populace_us_2024_acs_local.h5",
    ]
    strict = plan_lib.verify_receipt(receipt, state, strict=True)
    assert strict[-1] == "not in receipt: extra.json"
    assert plan_lib.verify_receipt({"schema": "x"}, state) == [
        f"not a {plan_lib.RECEIPT_SCHEMA} receipt"
    ]


def test_prior_state_check_refuses_missing_or_foreign_runs() -> None:
    materialize = plan_lib.parse_plan(_plan_data("materialize"))
    calibrate = plan_lib.parse_plan(_plan_data("calibrate"))
    assert plan_lib.prior_state_problems(materialize, None) == []
    assert "run materialize" in plan_lib.prior_state_problems(calibrate, None)[0]
    good = {"staging_sha256": STAGING_SHA, "ladder_sha256": LADDER_SHA}
    assert plan_lib.prior_state_problems(calibrate, good) == []
    foreign = {"staging_sha256": "f" * 64, "ladder_sha256": "e" * 64}
    problems = plan_lib.prior_state_problems(calibrate, foreign)
    assert [p.split(" sha256")[0] for p in problems] == ["staging_h5", "ladder"]


def test_cas_path_and_digest_cli(tmp_path: Path, capsys) -> None:
    path = tmp_path / "consumer_facts.jsonl"
    path.write_text('{"fact": 1}\n')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert plan_lib.cas_volume_path(sha, path.name) == (
        f"cas/sha256/{sha}/consumer_facts.jsonl"
    )
    assert plan_lib.main(["digest", str(path)]) == 0
    line = json.loads(capsys.readouterr().out)
    assert line["input"] == {
        "uri": f"volume://cas/sha256/{sha}/consumer_facts.jsonl",
        "sha256": sha,
    }
    assert line["upload"].startswith(f"modal volume put {plan_lib.INPUTS_VOLUME} ")
    # The emitted input parses as a plan input.
    ref = plan_lib.parse_input("feed", line["input"])
    assert ref.volume_path == f"cas/sha256/{sha}/consumer_facts.jsonl"


def test_validate_cli_prints_argv_and_estimate(tmp_path: Path, capsys) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan_data()))
    assert plan_lib.main(["validate", str(plan_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["resources"]["class"] == "heavy"
    assert summary["measured_locally"]["peak_rss_gb"] == 74.8
    assert summary["estimated_usd_at_measured_wall"] == pytest.approx(1.71, abs=0.01)
    bad = _plan_data()
    bad["source"]["commit"] = "abc"
    plan_path.write_text(json.dumps(bad))
    assert plan_lib.main(["validate", str(plan_path)]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_verify_receipt_cli_exit_codes(tmp_path: Path, capsys) -> None:
    receipt, state = _receipt(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    args = ["verify-receipt", str(receipt_path), "--state-root", str(state)]
    assert plan_lib.main(args) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True
    (state / "populace_us_2024_acs_local.h5").write_bytes(b"changed")
    assert plan_lib.main(args) == 1


def test_example_plan_in_the_runbook_parses() -> None:
    example = ROOT / "docs" / "us-modal-stage-example-plan.json"
    plan = plan_lib.parse_plan(json.loads(example.read_text()))
    assert plan.tool is plan_lib.US_ACS_LOCAL_RELEASE


def test_every_stage_uses_a_resource_class_the_app_can_run() -> None:
    # tools/modal_us_stage.py defines one Modal function per class.
    for tool in plan_lib.TOOLS.values():
        for stage in tool.stages.values():
            assert stage.resources is plan_lib.RESOURCE_CLASSES[stage.resources.name]


@pytest.mark.parametrize(
    ("value", "ok"),
    [(60, True), (10_800, True), (59, False), (8 * 3600, False), (True, False)],
)
def test_max_wall_seconds_is_bounded_below_the_hard_timeout(value, ok) -> None:
    data = _plan_data(max_wall_seconds=value)
    if ok:
        assert plan_lib.parse_plan(data).max_wall_seconds == value
    else:
        with pytest.raises(plan_lib.PlanError, match="max_wall_seconds"):
            plan_lib.parse_plan(data)


def test_budget_stop_marks_the_receipt_failed(tmp_path: Path) -> None:
    data = _plan_data(max_wall_seconds=3600)
    plan = plan_lib.parse_plan(data)
    receipt = plan_lib.build_receipt(
        plan,
        data,
        argv=["python"],
        returncode=0,
        started_at="t0",
        finished_at="t1",
        wall_seconds=3600.0,
        peak_rss_bytes=None,
        inputs_verified=[],
        outputs=[],
        git={},
        runner={},
        stopped_at_budget=True,
    )
    assert receipt["status"] == "FAILED"
    assert (receipt["max_wall_seconds"], receipt["stopped_at_budget"]) == (3600, True)


def test_runner_smoke_is_inline_and_check_sized() -> None:
    plan = plan_lib.parse_plan(_smoke_plan_data())
    assert plan.resources is plan_lib.CHECK
    argv = plan_lib.planned_argv(plan)
    assert argv[:3] == ["/opt/venv/bin/python", "-B", "-c"]
    assert "import microcosm.build" in argv[3]
    assert argv[4:] == [
        "/work/state",
        "/work/inputs/ladder/us_puma_ladder_2020.npz",
    ]


def test_preempted_attempts_are_charged_to_the_wall_budget() -> None:
    data = _plan_data(max_wall_seconds=20_000)
    plan = plan_lib.parse_plan(data)
    sha = plan_lib.plan_digest(data)

    def record(start: float, seen: float, **kw) -> dict:
        return plan_lib.attempt_record(
            plan,
            sha,
            attempt_id=f"a{start}",
            started_epoch=start,
            last_seen_epoch=seen,
            **kw,
        )

    preempted = record(0.0, 3_300.0)
    finished = record(10_000.0, 16_000.0, finished=True, receipt="receipts/x.json")
    other_plan = {**record(0.0, 9_000.0), "plan_sha256": "0" * 64}
    other_stage = {**record(0.0, 9_000.0), "stage": "calibrate"}
    prior = plan_lib.unfinished_attempts(
        [preempted, finished, other_plan, other_stage, {"schema": "junk"}], plan, sha
    )
    # Only this plan's own attempt that never wrote a receipt is charged.
    assert [item["attempt_id"] for item in prior] == [preempted["attempt_id"]]
    assert prior[0]["elapsed_seconds"] == 3_300.0
    assert plan_lib.remaining_wall_seconds(plan, prior) == 16_700
    assert plan_lib.remaining_wall_seconds(plan, []) == 20_000
    unbudgeted = plan_lib.parse_plan(_plan_data())
    assert plan_lib.remaining_wall_seconds(unbudgeted, prior) is None
    # Two preemptions that used the budget leave nothing to start with.
    spent = [record(0.0, 10_000.0), record(20_000.0, 30_000.0)]
    assert plan_lib.remaining_wall_seconds(plan, spent) < plan_lib.MIN_ATTEMPT_SECONDS


def test_receipt_prices_earlier_preempted_attempts() -> None:
    data = _plan_data(max_wall_seconds=20_000)
    plan = plan_lib.parse_plan(data)
    sha = plan_lib.plan_digest(data)
    preempted = plan_lib.attempt_record(
        plan, sha, attempt_id="a", started_epoch=0.0, last_seen_epoch=3_600.0
    )
    receipt = plan_lib.build_receipt(
        plan,
        data,
        argv=plan_lib.planned_argv(plan),
        returncode=0,
        started_at="2026-09-23T07:00:00Z",
        finished_at="2026-09-23T08:00:00Z",
        wall_seconds=3_000.0,
        peak_rss_bytes=None,
        inputs_verified=[],
        outputs=[],
        git={"head": COMMIT, "tree_clean": True},
        runner={},
        container_wall_seconds=3_600.0,
        prior_attempts=[preempted],
        budget_seconds=16_400,
    )
    assert receipt["budget_seconds_this_attempt"] == 16_400
    assert receipt["prior_unfinished_attempts"][0]["elapsed_seconds"] == 3_600.0
    # Two container-hours of the heavy class at list price: about $2.42.
    assert receipt["estimated_usd_all_attempts_at_list_price"] == pytest.approx(
        2.42, abs=0.01
    )
    assert receipt["estimated_usd_container_at_list_price"] == pytest.approx(
        1.21, abs=0.01
    )


# --------------------------------------------------------------------------- #
# Branch verification                                                          #
# --------------------------------------------------------------------------- #


_GIT_ENV = {
    "PATH": os.environ.get("PATH", ""),
    "HOME": os.environ.get("HOME", "/"),
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=_GIT_ENV,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def remote(tmp_path: Path) -> dict[str, str]:
    """A local remote: main c1-c2-c4 and feature/x c1-c2-c3."""

    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "remote"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "uploadpack.allowFilter", "true")
    shas = {}
    for name in ("c1", "c2"):
        _git(repo, "commit", "-q", "--allow-empty", "-m", name)
        shas[name] = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "feature/x")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "c3")
    shas["c3"] = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "c4")
    shas["c4"] = _git(repo, "rev-parse", "HEAD")
    return {"url": repo.as_uri(), **shas}


@pytest.mark.parametrize(
    ("branch", "commit", "verified", "check"),
    [
        ("feature/x", "c3", True, "is the branch tip"),
        ("feature/x", "c2", True, "is an ancestor of the tip"),
        ("feature/x", "c4", False, "not reachable"),
        ("main", "c3", False, "not reachable"),
        ("no-such-branch", "c3", False, "could not be fetched"),
    ],
)
def test_branch_is_verified_against_the_remote(
    app, remote, monkeypatch, branch, commit, verified, check
) -> None:
    for key, value in _GIT_ENV.items():
        monkeypatch.setenv(key, value)
    plan = dataclasses.replace(
        plan_lib.parse_plan(_plan_data()),
        repo_url=remote["url"],
        branch=branch,
        commit=remote[commit],
    )
    result = app._verify_branch(plan)
    assert result["branch_verified"] is verified
    assert check in result["branch_check"]
    if verified:
        assert result["branch_tip"] == remote["c3"]


def test_branch_check_argvs_never_touch_the_pinned_clone() -> None:
    argvs = plan_lib.branch_check_argvs(
        plan_lib.DEFAULT_REPO_URL, "us-modal-stage-runner", COMMIT, "/tmp/check.git"
    )
    for argv in argvs:
        assert plan_lib.IMAGE_REPO_ROOT not in argv
    assert argvs[1][-1] == "+refs/heads/us-modal-stage-runner:refs/branch-check/tip"
    assert "--filter=tree:0" in argvs[1]
    assert argvs[-1][-3:] == ["--is-ancestor", COMMIT, "refs/branch-check/tip"]


def test_branch_verdict_needs_a_proven_ancestry() -> None:
    verdict = plan_lib.branch_verdict
    assert verdict("b", COMMIT, fetch_returncode=0, tip=COMMIT, ancestor_returncode=0)[
        "branch_verified"
    ]
    for kwargs in (
        {"fetch_returncode": 128, "fetch_stderr": "fatal: couldn't find remote ref"},
        {"fetch_returncode": 0, "tip": "f" * 40, "ancestor_returncode": 1},
        {"fetch_returncode": 0, "tip": "f" * 40, "ancestor_returncode": 128},
        {"fetch_returncode": 0, "tip": None, "ancestor_returncode": None},
    ):
        assert verdict("b", COMMIT, **kwargs)["branch_verified"] is False


def test_nonpreemptible_placement_is_opt_in_and_priced_at_three_times_list() -> None:
    plain = plan_lib.parse_plan(_plan_data(max_wall_seconds=3600))
    assert plain.nonpreemptible is False
    assert plain.price_multiplier == 1.0
    data = _plan_data(max_wall_seconds=3600, nonpreemptible=True)
    plan = plan_lib.parse_plan(data)
    assert plan.nonpreemptible is True
    summary = plan_lib.summarize(plan)
    assert summary["resources"]["nonpreemptible"] is True
    # The heavy class for the #974 materialize wall: $1.71 at list, x3.
    assert summary["estimated_usd_at_measured_wall"] == pytest.approx(5.12, abs=0.01)
    # The budget plus the runner's 15 minutes of staging and mirroring.
    assert plan_lib.summarize(plain)["estimated_usd_at_max_wall"] == pytest.approx(
        1.51, abs=0.01
    )
    assert summary["estimated_usd_at_max_wall"] == pytest.approx(4.54, abs=0.01)
    receipt = plan_lib.build_receipt(
        plan,
        data,
        argv=plan_lib.planned_argv(plan),
        returncode=0,
        started_at="2026-09-23T07:00:00Z",
        finished_at="2026-09-23T08:00:00Z",
        wall_seconds=3_600.0,
        peak_rss_bytes=None,
        inputs_verified=[],
        outputs=[],
        git={"head": COMMIT, "tree_clean": True},
        runner={},
        container_wall_seconds=3_600.0,
    )
    assert receipt["resources"]["nonpreemptible"] is True
    assert receipt["estimated_usd_at_list_price"] == pytest.approx(3.63, abs=0.01)
    with pytest.raises(plan_lib.PlanError, match="true or false"):
        plan_lib.parse_plan(_plan_data(nonpreemptible="yes"))


# --------------------------------------------------------------------------- #
# The tool's environment                                                       #
# --------------------------------------------------------------------------- #


def test_tool_environment_strips_credentials_and_stays_offline() -> None:
    base = {
        "PATH": "/usr/bin",
        "HOME": "/root",
        "HF_TOKEN": "hf_secret",
        "HUGGING_FACE_HUB_TOKEN": "hf_secret",
        "HF_HUB_OFFLINE": "0",
        "AWS_SECRET_ACCESS_KEY": "x",
        "MODAL_IDENTITY_TOKEN": "x",
        "OMP_NUM_THREADS": "8",
    }
    plan_env = {"OMP_NUM_THREADS": "4", "MICROCOSM_ACS_POOL_PEAK_LIMIT_BYTES": "1"}
    env, removed = plan_lib.tool_environment(base, plan_env)
    assert removed == [
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "MODAL_IDENTITY_TOKEN",
    ]
    assert not set(removed) & set(env)
    assert "hf_secret" not in env.values()
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["OMP_NUM_THREADS"] == "4"  # the plan's override wins
    assert env["PATH"] == "/usr/bin"
    assert env["MICROCOSM_ACS_POOL_PEAK_LIMIT_BYTES"] == "1"


@pytest.mark.parametrize("key", ["HF_HUB_OFFLINE", "HF_TOKEN", "POPULACE_LEDGER_KEY"])
def test_tool_environment_refuses_an_unvalidated_plan_env(key: str) -> None:
    with pytest.raises(plan_lib.PlanError, match="may not be passed"):
        plan_lib.tool_environment({}, {key: "x"})


# --------------------------------------------------------------------------- #
# Attempt ledger: outcomes, charging and the run's lock                        #
# --------------------------------------------------------------------------- #


def _attempt(
    attempt_id: str,
    *,
    seen: float,
    started: float = 0.0,
    plan_data: dict | None = None,
    **kw,
) -> dict:
    data = plan_data or _plan_data(max_wall_seconds=20_000)
    return plan_lib.attempt_record(
        plan_lib.parse_plan(data),
        plan_lib.plan_digest(data),
        attempt_id=attempt_id,
        started_epoch=started,
        last_seen_epoch=seen,
        **kw,
    )


def test_attempt_records_carry_outcome_note_and_modal_ids() -> None:
    record = _attempt(
        "20260923T050000000000Z",
        seen=60.0,
        outcome=plan_lib.OUTCOME_ERROR,
        note="PlanError: input 'feed' ...",
        modal={"input_id": "in-1", "function_call_id": "fc-1"},
    )
    assert record["outcome"] == "error"
    assert record["finished"] is False
    assert record["modal"] == {"input_id": "in-1", "function_call_id": "fc-1"}
    assert _attempt("a", seen=1.0)["outcome"] is None


def test_charging_counts_every_attempt_that_did_not_finish() -> None:
    data = _plan_data(max_wall_seconds=20_000)
    plan, sha = plan_lib.parse_plan(data), plan_lib.plan_digest(data)
    preempted = _attempt("a1", seen=3_000.0)
    errored = _attempt("a2", seen=500.0, outcome=plan_lib.OUTCOME_ERROR)
    refused = _attempt("a3", seen=270.0, finished=True, outcome="refused")
    stopped = _attempt(
        "a4", seen=9_000.0, finished=True, outcome="receipt", receipt="r.json"
    )
    own = _attempt("a5", seen=10.0)
    charged = plan_lib.unfinished_attempts(
        [preempted, errored, refused, stopped, own], plan, sha, exclude="a5"
    )
    assert [item["attempt_id"] for item in charged] == ["a1", "a2"]
    # A budget stop wrote a FAILED receipt, so the same plan starts afresh.
    assert plan_lib.remaining_wall_seconds(plan, charged) == 20_000 - 3_500


def test_recent_unfinished_attempts_are_the_run_s_possible_lock_holders() -> None:
    now = 10_000.0
    window = plan_lib.ATTEMPT_LIVE_WINDOW_SECONDS
    other_run = _attempt(
        "b", seen=now, plan_data=_plan_data(run_id="another-run", stage="calibrate")
    )
    records = [
        _attempt("fresh", seen=now - 10),
        _attempt("edge", seen=now - window + 1),
        _attempt("stale", seen=now - window - 1),
        _attempt("ended", seen=now - 10, outcome=plan_lib.OUTCOME_ERROR),
        _attempt("done", seen=now - 10, finished=True, outcome="receipt"),
        _attempt("mine", seen=now),
        other_run,
        # A record from the runner before outcomes existed.
        {
            key: value
            for key, value in _attempt("legacy", seen=now - 5).items()
            if key not in {"outcome", "note", "modal"}
        },
    ]
    recent = plan_lib.recent_unfinished_attempts(
        records, "acs-local-20260923", now=now, exclude="mine"
    )
    assert [item["attempt_id"] for item in recent] == ["fresh", "edge", "legacy"]


class _Ledger:
    """Attempt records as successive reads of the runs volume return them."""

    def __init__(self, *reads: list[dict]) -> None:
        self.reads = list(reads)
        self.sleeps: list[float] = []

    def read(self) -> list[dict]:
        return self.reads.pop(0) if len(self.reads) > 1 else self.reads[0]

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


OWN = "20260923T060000000000Z"
EARLIER = "20260923T050000000000Z"
LATER = "20260923T070000000000Z"


def _lock(ledger: _Ledger, now: float = 10_000.0) -> list[dict]:
    return plan_lib.live_attempts(
        ledger.read,
        "acs-local-20260923",
        own_attempt_id=OWN,
        now=lambda: now,
        sleep=ledger.sleep,
    )


def test_lock_is_free_without_recent_earlier_attempts() -> None:
    ledger = _Ledger([_attempt(EARLIER, seen=1_000.0), _attempt(OWN, seen=10_000.0)])
    assert _lock(ledger) == []
    assert ledger.sleeps == []


def test_lock_waits_out_a_preempted_predecessor_and_starts() -> None:
    # Modal restarted this input moments after preempting it: the earlier
    # attempt's record is recent but never moves again.
    dead = _attempt(EARLIER, seen=9_950.0)
    ledger = _Ledger([dead], [dead])
    assert _lock(ledger) == []
    assert ledger.sleeps == [plan_lib.ATTEMPT_RECHECK_SECONDS]


def test_lock_refuses_while_an_earlier_attempt_keeps_writing() -> None:
    ledger = _Ledger(
        [_attempt(EARLIER, seen=9_950.0)], [_attempt(EARLIER, seen=10_200.0)]
    )
    running = _lock(ledger)
    assert [item["attempt_id"] for item in running] == [EARLIER]


def test_lock_frees_when_the_earlier_attempt_finishes_during_the_wait() -> None:
    ledger = _Ledger(
        [_attempt(EARLIER, seen=9_950.0)],
        [_attempt(EARLIER, seen=10_200.0, finished=True, outcome="receipt")],
    )
    assert _lock(ledger) == []


def test_lock_leaves_a_later_attempt_to_refuse_itself() -> None:
    ledger = _Ledger([_attempt(LATER, seen=10_000.0)])
    assert _lock(ledger) == []
    assert ledger.sleeps == []


def test_lock_covers_every_stage_and_plan_of_the_run() -> None:
    calibrate = _attempt(
        EARLIER, seen=9_950.0, plan_data=_plan_data("calibrate", max_wall_seconds=600)
    )
    moved = {**calibrate, "last_seen_epoch": 10_100.0}
    assert len(_lock(_Ledger([calibrate], [moved]))) == 1


def test_app_attempt_record_cannot_be_overwritten_after_it_ends(
    app, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(plan_lib, "RUNS_MOUNT", str(tmp_path))
    data = _plan_data(max_wall_seconds=600)
    plan = plan_lib.parse_plan(data)
    attempt = app._Attempt(plan, plan_lib.plan_digest(data), started=1_000.0)
    attempt.write()
    record = json.loads(attempt.path.read_text())
    assert (record["finished"], record["outcome"]) == (False, None)
    assert record["modal"] == {"input_id": "in-test", "function_call_id": "fc-test"}
    attempt.end("receipt", finished=True, receipt="receipts/x.json")
    attempt.write()  # a heartbeat that lost the race
    record = json.loads(attempt.path.read_text())
    assert (record["finished"], record["outcome"]) == (True, "receipt")
    assert record["receipt"] == "receipts/x.json"
    assert not list(attempt.path.parent.glob("*.tmp"))
    assert app.runs_volume.commit.call_count == 2
    assert [item["attempt_id"] for item in attempt.records()] == [attempt.attempt_id]
    app.runs_volume.reload.assert_called()


@pytest.mark.parametrize(
    ("raised", "outcome", "finished"),
    [("refusal", "refused", True), ("error", "error", False)],
)
def test_app_run_stage_ends_the_attempt_with_its_outcome(
    app, tmp_path: Path, monkeypatch, raised: str, outcome: str, finished: bool
) -> None:
    monkeypatch.setattr(plan_lib, "RUNS_MOUNT", str(tmp_path))
    monkeypatch.setattr(
        app,
        "_git_state",
        lambda plan: {
            "head_matches_plan": True,
            "tree_clean": True,
            "branch_verified": True,
            "tool_present": True,
        },
    )
    error = app._Refusal("lock held") if raised == "refusal" else OSError("disk full")

    def attempt_stage(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(app, "_attempt_stage", attempt_stage)
    with pytest.raises(type(error)):
        app._run_stage(_plan_data(max_wall_seconds=600))
    (path,) = (tmp_path / "runs" / "acs-local-20260923" / "attempts").glob("*.json")
    record = json.loads(path.read_text())
    assert (record["outcome"], record["finished"]) == (outcome, finished)
    assert str(error) in record["note"]


# --------------------------------------------------------------------------- #
# Atomic mirroring and the pulled-state check                                  #
# --------------------------------------------------------------------------- #


def test_a_mirror_cut_short_never_leaves_a_half_written_file(
    tmp_path: Path, monkeypatch
) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "a.npz").write_bytes(b"old-a")
    (src / "b.h5").write_bytes(b"old-b")
    plan_lib.mirror_tree(src, dst)
    (src / "a.npz").write_bytes(b"new-a-longer")
    (src / "b.h5").write_bytes(b"new-b-longer")
    real_copy = plan_lib.shutil.copy2

    def copy_then_die(source, target):
        if Path(source).name == "b.h5":
            Path(target).write_bytes(b"new-")  # the partial copy, then preemption
            raise KeyboardInterrupt
        return real_copy(source, target)

    monkeypatch.setattr(plan_lib.shutil, "copy2", copy_then_die)
    with pytest.raises(KeyboardInterrupt):
        plan_lib.mirror_tree(src, dst)
    # a.npz was replaced whole, b.h5 still holds its old bytes, and the
    # partial copy is invisible to listings.
    assert (dst / "a.npz").read_bytes() == b"new-a-longer"
    assert (dst / "b.h5").read_bytes() == b"old-b"
    assert (dst / f".b.h5{plan_lib.MIRROR_PARTIAL_SUFFIX}").exists()
    assert set(plan_lib.tree_listing(dst)) == {"a.npz", "b.h5"}
    monkeypatch.setattr(plan_lib.shutil, "copy2", real_copy)
    counts = plan_lib.mirror_tree(src, dst)
    assert counts["partials_removed"] == 1
    assert (dst / "b.h5").read_bytes() == b"new-b-longer"
    assert not list(dst.rglob(f"*{plan_lib.MIRROR_PARTIAL_SUFFIX}"))


def _state_receipt(
    state: Path, finished_at: str, stage: str = "materialize", run_id: str = ""
) -> dict:
    data = _plan_data(stage, run_id=run_id or "acs-local-20260923")
    plan = plan_lib.parse_plan(data)
    return plan_lib.build_receipt(
        plan,
        data,
        argv=["python"],
        returncode=0,
        started_at=finished_at,
        finished_at=finished_at,
        wall_seconds=1.0,
        peak_rss_bytes=None,
        inputs_verified=[],
        outputs=plan_lib.hash_tree(state),
        git={},
        runner={},
    )


def test_latest_receipt_is_by_finish_time_not_by_name(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    materialize = _state_receipt(state, "2026-09-23T09:00:00Z")
    calibrate = _state_receipt(state, "2026-09-23T10:00:00Z", stage="calibrate")
    foreign = _state_receipt(state, "2026-09-23T11:00:00Z", run_id="other-run")
    receipts = [
        ("calibrate-2026-09-23T100000Z.json", calibrate),
        ("materialize-2026-09-23T090000Z.json", materialize),
        ("materialize-2026-09-23T110000Z.json", foreign),
        ("truncated.json", {}),
    ]
    name, receipt = plan_lib.latest_receipt(receipts, "acs-local-20260923")
    assert name == "calibrate-2026-09-23T100000Z.json"
    assert receipt["stage"] == "calibrate"
    assert plan_lib.latest_receipt([("x.json", {})], "acs-local-20260923") is None


def test_pulled_state_must_be_what_the_latest_receipt_lists(tmp_path: Path) -> None:
    state = tmp_path / "state"
    assert plan_lib.pulled_state_problems(state, None) == []
    (state / "checkpoints").mkdir(parents=True)
    (state / "checkpoints" / "targets.json").write_text("[]")
    # State with no receipt at all: a first stage cut short after mirroring.
    (problem,) = plan_lib.pulled_state_problems(state, None)
    assert "has no receipt" in problem
    receipt = _state_receipt(state, "2026-09-23T09:00:00Z")
    latest = ("materialize-2026-09-23T090000Z.json", receipt)
    assert plan_lib.pulled_state_problems(state, latest) == []
    # A later stage's mirror cut short: one file replaced, one added.
    (state / "checkpoints" / "targets.json").write_text("[1]")
    (state / "weights_latest.npz").write_bytes(b"w")
    problems = plan_lib.pulled_state_problems(state, latest)
    assert problems == [
        "state differs from receipt materialize-2026-09-23T090000Z.json: "
        "size mismatch: checkpoints/targets.json is 3 bytes, receipt 2",
        "state differs from receipt materialize-2026-09-23T090000Z.json: "
        "not in receipt: weights_latest.npz",
    ]


def test_receipt_names_the_receipt_the_pulled_state_matched(tmp_path: Path) -> None:
    data = _plan_data("calibrate")
    plan = plan_lib.parse_plan(data)
    receipt = plan_lib.build_receipt(
        plan,
        data,
        argv=["python"],
        returncode=0,
        started_at="t0",
        finished_at="t1",
        wall_seconds=1.0,
        peak_rss_bytes=None,
        inputs_verified=[],
        outputs=[],
        git={},
        runner={},
        attempt_id=OWN,
        prior_state_verified_against="materialize-2026-09-23T090000Z.json",
    )
    assert receipt["attempt_id"] == OWN
    assert receipt["prior_state_verified_against"] == (
        "materialize-2026-09-23T090000Z.json"
    )


def test_app_reads_receipts_even_when_one_is_truncated(app, tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "materialize-2026-09-23T090000Z.json").write_text('{"schema": "x"}')
    (receipts / "calibrate-2026-09-23T100000Z.json").write_text('{"sche')
    (receipts / ".calibrate-2026-09-23T110000Z.json.tmp").write_text("{}")
    rows = app._receipts(tmp_path)
    assert [(name, data) for name, data, _ in rows] == [
        ("calibrate-2026-09-23T100000Z.json", {}),
        ("materialize-2026-09-23T090000Z.json", {"schema": "x"}),
    ]


# --------------------------------------------------------------------------- #
# Non-preemptible placement                                                    #
# --------------------------------------------------------------------------- #


def test_nonpreemptible_is_refused_for_the_check_class() -> None:
    data = _smoke_plan_data()
    data["nonpreemptible"] = True
    with pytest.raises(plan_lib.PlanError, match="always runs preemptible"):
        plan_lib.parse_plan(data)


def test_every_valid_plan_has_a_runner_with_its_class_and_placement(app) -> None:
    keys = {
        (plan.resources.name, plan.nonpreemptible)
        for plan in [
            plan_lib.parse_plan(_plan_data(stage, nonpreemptible=flag))
            for stage in plan_lib.US_ACS_LOCAL_RELEASE.stages
            for flag in (False, True)
        ]
        + [plan_lib.parse_plan(_smoke_plan_data())]
    }
    assert keys <= set(app.RUNNERS)
    for (name, nonpreemptible), runner in app.RUNNERS.items():
        resources = plan_lib.RESOURCE_CLASSES[name]
        options = runner.modal_options
        assert (options["cpu"], options["memory"], options["timeout"]) == (
            resources.cpu,
            resources.memory_mib,
            resources.timeout_s,
        )
        assert options.get("nonpreemptible", False) is nonpreemptible
        assert options["retries"] == 0
    # The check never asks for non-preemptible placement.
    assert "nonpreemptible" not in app.check_stage.modal_options


def test_nonpreemptible_prices_every_attempt_at_three_times_list() -> None:
    data = _plan_data(max_wall_seconds=20_000, nonpreemptible=True)
    plan = plan_lib.parse_plan(data)
    preempted = _attempt("a1", seen=3_600.0, plan_data=data)
    receipt = plan_lib.build_receipt(
        plan,
        data,
        argv=["python"],
        returncode=0,
        started_at="t0",
        finished_at="t1",
        wall_seconds=3_600.0,
        peak_rss_bytes=None,
        inputs_verified=[],
        outputs=[],
        git={},
        runner={},
        container_wall_seconds=3_600.0,
        prior_attempts=[preempted],
    )
    # One heavy container-hour is $1.21 at list; x3, and two hours in all.
    assert receipt["estimated_usd_container_at_list_price"] == pytest.approx(3.63)
    assert receipt["estimated_usd_all_attempts_at_list_price"] == pytest.approx(7.27)
    # The flag is part of the plan, so switching it is a new plan digest.
    assert plan_lib.plan_digest(data) != plan_lib.plan_digest(
        _plan_data(max_wall_seconds=20_000)
    )


@pytest.mark.parametrize(
    ("status", "returncode", "stopped", "fails"),
    [
        ("COMPLETED", 0, False, False),
        ("FAILED", 0, True, True),  # stopped at the budget; the tool exited 0
        ("FAILED", 1, False, True),
    ],
)
def test_app_main_exits_nonzero_unless_the_stage_completed(
    app, monkeypatch, capsys, status, returncode, stopped, fails
) -> None:
    data = _plan_data(max_wall_seconds=600)
    monkeypatch.setattr(app, "_LOADED", (data, plan_lib.parse_plan(data)))
    receipt = {
        "status": status,
        "stage": "materialize",
        "run_id": "acs-local-20260923",
        "wall_seconds": 600.0,
        "peak_rss_bytes": 1,
        "returncode": returncode,
        "stopped_at_budget": stopped,
        "estimated_usd_at_list_price": 0.2,
        "receipt_path": "runs/acs-local-20260923/receipts/x.json",
        "outputs": [],
    }
    runner = MagicMock()
    runner.remote.return_value = receipt
    monkeypatch.setitem(app.RUNNERS, ("heavy", False), runner)
    if fails:
        with pytest.raises(SystemExit, match=f"STAGE {status}"):
            app.main(run=True)
    else:
        app.main(run=True)
    assert json.loads(capsys.readouterr().out)["stopped_at_budget"] is stopped


@pytest.mark.parametrize("key", plan_lib.THREAD_ENV_KEYS)
def test_every_documented_thread_count_is_allowlisted(key) -> None:
    assert plan_lib._ENV_KEY.fullmatch(key)


@pytest.mark.parametrize("key", ["VECLIB_MAXIMUM_THREADS", "FOO_NUM_THREADS"])
def test_an_undocumented_thread_variable_is_refused_by_name(key) -> None:
    assert not plan_lib._ENV_KEY.fullmatch(key)
