"""US-engine coverage for the Modal stage runner smoke operation."""

import json
import subprocess
import sys
from pathlib import Path

from test_support.microcosm_build.us_modal_stage_plan_tool import (
    plan_lib,
    smoke_plan_data,
)


def test_runner_smoke_code_writes_its_state_file(tmp_path: Path) -> None:
    """The generated operation records its inputs and US engine version."""

    code = plan_lib.planned_argv(plan_lib.parse_plan(smoke_plan_data()))[3]
    ladder = tmp_path / "inputs" / "ladder" / "us_puma_ladder_2020.npz"
    ladder.parent.mkdir(parents=True)
    ladder.write_bytes(b"npz")
    state = tmp_path / "state"
    subprocess.run([sys.executable, "-c", code, str(state), str(ladder)], check=True)
    payload = json.loads((state / "smoke" / "inputs.json").read_text())
    assert payload["inputs"] == [{"input": "ladder", "bytes": 3}]
    assert payload["policyengine_us"]
