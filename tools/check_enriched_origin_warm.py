"""Fresh-process REQUIRED replay of the exact source-owned invented26 fixture.

Run through run_origin_group_cap_checks.py, which pins Python/source and limits
CPU/wall time. The fixture installs the same memory and policy-call tripwires.
Fixture restoration/target decoding remains setup work; graph kernel and source
codec calls are all fatal during the actual REQUIRED replay.
"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
root = Path(sys.argv[1])
assert root.is_absolute() and root.name.startswith("invented-enriched-")
spec = importlib.util.spec_from_file_location(
    "enriched_warm_fixture",
    ROOT / "packages/microcosm-build/tests/test_us_enriched_origin_calibration.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
guard = module.invented_only_guard.__wrapped__()
next(guard)
try:
    result = module.run_fixture(root, warm=True)
    before = json.loads((root / "cold.json").read_text())
    after = json.loads((root / "warm.json").read_text())
    assert result["kernel_calls"] == result["source_calls"] == []
    assert [p["hits"] for p in result["phases"]] == [17, 20, 23, 24, 26]
    for field in (
        "keys",
        "opaque",
        "budget",
        "parent",
        "successor",
        "successor_key",
        "content",
    ):
        assert before[field] == after[field], field
    print(
        json.dumps(
            {
                "status": "PASS",
                "phase_hits": [17, 20, 23, 24, 26],
                "total_restores": 110,
                "kernel_calls": 0,
                "graph_source_codec_calls": 0,
            }
        )
    )
finally:
    try:
        next(guard)
    except StopIteration:
        pass
