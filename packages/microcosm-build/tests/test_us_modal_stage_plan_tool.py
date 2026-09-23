"""Unit tests for the pure half of the US Modal stage runner.

``tools/modal_us_stage_plan.py`` builds the plan, argv and sha256 receipts
that ``tools/modal_us_stage.py`` executes on Modal. Nothing here imports
``modal`` or touches the network.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load():
    path = ROOT / "tools" / "modal_us_stage_plan.py"
    spec = importlib.util.spec_from_file_location("modal_us_stage_plan", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolve string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plan_lib = _load()

COMMIT = "4d773a4785a1e2c7f0b9d3e6a8c5b1f2e3d4c5b6"
STAGING_SHA = "a" * 64
SUMMARY_SHA = "b" * 64
FEED_SHA = "4d1dba8c1b6274877bf184fa6de5d99b13fc61f34709ccab1487db2b5c64a79f"
LADDER_SHA = "c" * 64


def _plan_data(stage: str = "materialize", **overrides) -> dict:
    data = {
        "schema": plan_lib.PLAN_SCHEMA,
        "tool": "us-acs-local-release",
        "stage": stage,
        "run_id": "acs-local-20260923",
        "source": {"commit": COMMIT, "branch": "us-modal-stage-runner"},
        "inputs": {
            "staging_h5": {
                "uri": f"volume://cas/sha256/{STAGING_SHA}/acs_multispine_staging.h5",
                "sha256": STAGING_SHA,
            },
            "staging_summary": {
                "uri": (
                    f"volume://cas/sha256/{SUMMARY_SHA}/"
                    "acs_multispine_staging.summary.json"
                ),
                "sha256": SUMMARY_SHA,
            },
            "feed": {
                "uri": f"volume://cas/sha256/{FEED_SHA}/consumer_facts.jsonl",
                "sha256": FEED_SHA,
            },
            "ladder": {
                "uri": (
                    "hf://datasets/policyengine/populace-us@"
                    "populace-us-2024-spm-receipts-20260923/"
                    "inputs/us_puma_ladder_2020.npz"
                ),
                "sha256": LADDER_SHA,
            },
        },
        "options": {"soi_mode": "totals", "hh_chunk": 20000},
    }
    data.update(overrides)
    return data


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
    assert plan_lib.mirror_tree(src, dst) == {"copied": 2, "deleted": 0}
    assert plan_lib.mirror_tree(src, dst) == {"copied": 0, "deleted": 0}
    (src / "artifact.h5").unlink()
    assert plan_lib.mirror_tree(src, dst) == {"copied": 0, "deleted": 1}
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


def _smoke_plan_data() -> dict:
    return {
        "schema": plan_lib.PLAN_SCHEMA,
        "tool": "runner-smoke",
        "stage": "smoke",
        "run_id": "runner-smoke-20260923",
        "source": {"commit": COMMIT, "branch": "us-modal-stage-runner"},
        "inputs": {
            "ladder": {
                "uri": f"volume://cas/sha256/{LADDER_SHA}/us_puma_ladder_2020.npz",
                "sha256": LADDER_SHA,
            },
        },
    }


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


@pytest.mark.skipif(
    importlib.util.find_spec("policyengine_us") is None,
    reason="the smoke records the policyengine-us version (the us extra)",
)
def test_runner_smoke_code_writes_its_state_file(tmp_path: Path) -> None:
    # The inline code runs as written: it writes the state file the receipt
    # lists (executed here against a local stand-in input).
    code = plan_lib.planned_argv(plan_lib.parse_plan(_smoke_plan_data()))[3]
    ladder = tmp_path / "inputs" / "ladder" / "us_puma_ladder_2020.npz"
    ladder.parent.mkdir(parents=True)
    ladder.write_bytes(b"npz")
    state = tmp_path / "state"
    subprocess.run([sys.executable, "-c", code, str(state), str(ladder)], check=True)
    payload = json.loads((state / "smoke" / "inputs.json").read_text())
    assert payload["inputs"] == [{"input": "ladder", "bytes": 3}]
    assert payload["policyengine_us"]
