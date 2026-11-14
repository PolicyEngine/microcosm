"""Exercise the actual Modal parser probe with the registered US tools.

The scorer imports the US engine through its sibling release tool, so this
subprocess regression belongs to ci_test_plan.py's engine-contract-us group.
It parses argv without calculating households or reading population H5 files.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from test_support.microcosm_build.us_modal_stage_plan_tool import (
    ROOT,
    base_plan_data,
    load_app,
    plan_data,
    plan_lib,
    smoke_plan_data,
)


@pytest.mark.parametrize(
    ("data", "stages"),
    [
        pytest.param(
            json.loads(
                (ROOT / "docs" / "us-modal-stage-route-a-h2h-plan.json").read_text()
            ),
            None,
            id="us-release-head-to-head",
        ),
        pytest.param(plan_data(), ["materialize"], id="us-acs-local-release"),
        pytest.param(base_plan_data(), None, id="us-puf-support-base"),
    ],
)
def test_registered_tool_parses_with_the_actual_subprocess_probe(
    data: dict, stages: list[str] | None, monkeypatch
) -> None:
    app = load_app(monkeypatch)
    monkeypatch.setattr(plan_lib, "IMAGE_REPO_ROOT", str(ROOT))
    plan = plan_lib.parse_plan(data)
    argv = plan_lib.planned_argv(plan)
    argv[0] = sys.executable
    env = {**os.environ, **plan.env}
    # A caller's PYTHONPATH must not hide missing sibling-import support.
    env.pop("PYTHONPATH", None)

    result = app._parse_check(plan, argv, env)

    assert result["returncode"] == 0, result
    assert json.loads(result["stdout_tail"].splitlines()[-1]) == {"stages": stages}


def test_inline_runner_smoke_still_skips_the_parser_probe(monkeypatch) -> None:
    app = load_app(monkeypatch)
    plan = plan_lib.parse_plan(smoke_plan_data())

    assert app._parse_check(plan, plan_lib.planned_argv(plan), {}) == {
        "returncode": 0,
        "skipped": "inline tool; nothing to parse",
    }
