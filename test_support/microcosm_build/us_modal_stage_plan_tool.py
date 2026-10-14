"""Shared fixtures for US Modal stage-plan tests."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys

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


def load_app(monkeypatch):
    """``tools/modal_us_stage.py`` imported against a stub ``modal`` module.

    The stub answers ``is_local()`` with False, so the module defines its
    image and functions without reading a plan or contacting Modal; volume
    calls (commit, reload) are recorded, not performed. Each
    ``@app.function`` keeps its function and records its Modal options as
    ``modal_options``.
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
