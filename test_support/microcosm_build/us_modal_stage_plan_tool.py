"""Shared fixtures for US Modal stage-plan tests."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path

from test_support.paths import paths_for

_PATHS = paths_for("microcosm-build")
ROOT = _PATHS.repository
#: The committed plan for Route A's base stage.
ROUTE_A_BASE_PLAN = ROOT / "docs" / "us-modal-stage-route-a-base-plan.json"
#: Route A's base command (base-config.json for 4b57d15a287c), paths tokenized.
ROUTE_A_BASE_COMMAND = (
    _PATHS.tests
    / "fixtures"
    / "modal_us_stage"
    / "route_a_base_command_4b57d15a287c.json"
)


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
#: A collector origin under a reserved name (RFC 2606), never a real collector.
TEST_COLLECTOR_ORIGIN = "https://collector.invalid"
#: What the image's emitter answers to ``microcosm-emitter version``. Tests
#: get this answer instead of running a command at the image's emitter path.
RECORDED_EMITTER_VERSION = {
    "emitter_version": "0.13.0",
    "protocol_versions": [1],
    "collector_origin": TEST_COLLECTOR_ORIGIN,
}
#: What the runner records of the image emitter's ``microcosm-emitter status``
#: after the tool. Tests get this record instead of running a command at the
#: image's emitter path.
RECORDED_EMITTER_STATUS = {
    "runs": [
        {
            "run_id": "fake-run.run",
            "producer_id": "recorded-producer",
            "upload_state": "pending",
            "reason": "collector_token_expired",
            "undelivered_events": 3,
        }
    ],
    "problem": None,
}
#: A made-up collector token. Tests never use a real credential.
DUMMY_COLLECTOR_TOKEN = "dummy-collector-token-not-a-credential"
#: Long after any test runs.
TEST_TOKEN_EXPIRES_AT = "2099-01-01T00:00:00+00:00"
#: What the secret a launch with ``--run`` makes for the container holds: the
#: runner's own names for the collector token, its expiry and its origin.
RUNNER_TOKEN_SECRET = {
    "MICROCOSM_MODAL_TELEMETRY_TOKEN": DUMMY_COLLECTOR_TOKEN,
    "MICROCOSM_MODAL_TELEMETRY_TOKEN_EXPIRES_AT": TEST_TOKEN_EXPIRES_AT,
    "MICROCOSM_MODAL_TELEMETRY_ORIGIN": TEST_COLLECTOR_ORIGIN,
}


def load_app(
    monkeypatch,
    *,
    real_emitter_version: bool = False,
    real_emitter_status: bool = False,
):
    """``tools/modal_us_stage.py`` imported against a stub ``modal`` module.

    The stub answers ``is_local()`` with False, so the module defines its
    image and functions without reading a plan or contacting Modal; volume
    calls (commit, reload) are recorded, not performed. Each
    ``@app.function`` keeps its function and records its Modal options as
    ``modal_options``. ``_emitter_version`` answers
    ``RECORDED_EMITTER_VERSION`` and ``_emitter_status`` answers
    ``RECORDED_EMITTER_STATUS``, so no test runs an emitter installed at the
    image's path (``/usr/local/bin/microcosm-emitter``) on the test machine;
    ``real_emitter_version`` and ``real_emitter_status`` keep the module's
    own, for a test that replaces what they run. The environment holds
    ``RUNNER_TOKEN_SECRET``, as a
    container launched with ``--run`` does; a test of a container without
    the token removes it.
    """

    from unittest.mock import MagicMock

    stub = MagicMock(name="modal")
    stub.is_local.return_value = False
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
    if not real_emitter_version:
        monkeypatch.setattr(
            module,
            "_emitter_version",
            lambda: copy.deepcopy(RECORDED_EMITTER_VERSION),
        )
    if not real_emitter_status:
        monkeypatch.setattr(
            module,
            "_emitter_status",
            lambda env: copy.deepcopy(RECORDED_EMITTER_STATUS),
        )
    for name, value in RUNNER_TOKEN_SECRET.items():
        monkeypatch.setenv(name, value)
    return module


COMMIT = "4d773a4785a1e2c7f0b9d3e6a8c5b1f2e3d4c5b6"
STAGING_SHA = "a" * 64
SUMMARY_SHA = "b" * 64
FEED_SHA = "4d1dba8c1b6274877bf184fa6de5d99b13fc61f34709ccab1487db2b5c64a79f"
LADDER_SHA = "c" * 64


def plan_data(stage: str = "materialize", **overrides) -> dict:
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


def base_plan_data(**overrides) -> dict:
    """A deep copy of the committed Route A base plan, with top-level overrides."""

    data = copy.deepcopy(json.loads(ROUTE_A_BASE_PLAN.read_text()))
    data.update(overrides)
    return data


def smoke_plan_data() -> dict:
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


# --------------------------------------------------------------------------- #
# The run path end to end, against local directories                          #
# --------------------------------------------------------------------------- #

#: A stand-in for a resumable tool: its first run writes a checkpoint, a
#: later run of the same state finds it and only rewrites the output.
FAKE_TOOL_SCRIPT = """\
import os, pathlib, sys, time
state, data = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
ckpt = state / "ckpt" / "step1.bin"
resumed = ckpt.exists()
if not resumed:
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(data.read_bytes() * 3)
(state / "out").mkdir(parents=True, exist_ok=True)
(state / "out" / "result.txt").write_text(f"resumed={resumed}\\n")
print("fake tool", "resumed" if resumed else "fresh", flush=True)
time.sleep(float(os.environ.get("MICROCOSM_FAKE_SLEEP", "0")))
"""
FAKE_PAYLOAD = b"fake input"
FAKE_SHA = hashlib.sha256(FAKE_PAYLOAD).hexdigest()


def fake_tool(
    timeout_s: int = 3600, overhead: int = 60, **stage: object
) -> plan_lib.ToolSpec:
    """A registered tool whose one stage, ``run``, runs ``FAKE_TOOL_SCRIPT``."""

    resources = plan_lib.Resources("check", 2.0, 8 * 1024, timeout_s=timeout_s)
    return plan_lib.ToolSpec(
        name="fake-stage",
        script=None,
        inputs=("data",),
        stages={
            "run": plan_lib.StageSpec(
                "run",
                resources,
                ("data",),
                runner_overhead_seconds=overhead,
                **stage,  # type: ignore[arg-type]
            )
        },
        options={},
        owned_flags=frozenset(),
        argv_builder=lambda plan, paths, state: [
            "-c",
            FAKE_TOOL_SCRIPT,
            state,
            paths["data"],
        ],
    )


def fake_plan(**overrides) -> dict:
    """A plan for ``fake_tool``'s stage, run ``fake-run``, with top-level overrides."""

    data = {
        "schema": plan_lib.PLAN_SCHEMA,
        "tool": "fake-stage",
        "stage": "run",
        "run_id": "fake-run",
        "source": {"commit": "a" * 40, "branch": "main"},
        "inputs": {
            "data": {
                "uri": f"volume://cas/sha256/{FAKE_SHA}/data.bin",
                "sha256": FAKE_SHA,
            }
        },
    }
    data.update(overrides)
    return data


def install_fake_runner(app, tmp_path: Path, monkeypatch) -> Callable:
    """Point ``app``'s mounts, clone and venv at local directories under ``tmp_path``.

    Returns ``install(tool)``, which registers ``tool`` and returns ``app``.
    The venv's Python is this interpreter, and the clone's git state is a
    stand-in that passes every check.
    """

    def install(tool: plan_lib.ToolSpec):
        monkeypatch.setitem(plan_lib.TOOLS, tool.name, tool)
        return app

    for name in ("inputs", "runs", "work", "repo", "venv/bin"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "venv" / "bin" / "python").symlink_to(sys.executable)
    cas = tmp_path / "inputs" / "cas" / "sha256" / FAKE_SHA
    cas.mkdir(parents=True)
    (cas / "data.bin").write_bytes(FAKE_PAYLOAD)
    monkeypatch.setattr(plan_lib, "INPUTS_MOUNT", str(tmp_path / "inputs"))
    monkeypatch.setattr(plan_lib, "RUNS_MOUNT", str(tmp_path / "runs"))
    monkeypatch.setattr(plan_lib, "WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(plan_lib, "IMAGE_REPO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setattr(plan_lib, "IMAGE_VENV", str(tmp_path / "venv"))
    monkeypatch.setattr(
        app,
        "_git_state",
        lambda plan: {
            "head_matches_plan": True,
            "tree_clean": True,
            "branch_verified": True,
            "tool_present": True,
            "tree_files_verified": True,
        },
    )
    return install
