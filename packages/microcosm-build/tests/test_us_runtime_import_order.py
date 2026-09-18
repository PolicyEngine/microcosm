"""Registry/ABI imports must succeed whichever public module loads first."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.requires_us


@pytest.mark.parametrize("first", ["multispine_pool", "spine_agreement"])
def test_pool_registry_import_order_in_fresh_process(tmp_path, first: str) -> None:
    # A fresh interpreter catches the partially initialized module path that a
    # previously collected sibling test can otherwise hide in sys.modules.
    script = """
import importlib
import sys

importlib.import_module('microcosm.build.us_runtime.' + sys.argv[1])
pool = importlib.import_module('microcosm.build.us_runtime.multispine_pool')
agreement = importlib.import_module('microcosm.build.us_runtime.spine_agreement')
assert pool.default_spine_agreement_registry is agreement.default_spine_agreement_registry
assert pool.spine_agreement_gate is agreement.spine_agreement_gate
assert pool.POOL_SPINE_AGREEMENT_REGISTRY is not None
assert agreement.US_SPINE_AGREEMENT_REGISTRY is not None
assert callable(pool.pool_remaining_stage_input_manifest)
assert callable(pool.pool_remaining_stage_input_manifest_receipt)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, first],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
