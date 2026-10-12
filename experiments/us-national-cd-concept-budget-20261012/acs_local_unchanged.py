"""ACS local weights before and after this change, bit for bit, on real surfaces.

Loads the pre-change ``target_loss_weights.py`` (at ``BASE``) as a separate
module and compares ``us_acs_local_target_loss_weights`` with the current
module's on the 09-23 release registry (whose recorded digest is
``ba36f887…``) and on each ``--soi-mode`` surface of the pinned feed, with
the 09-23 ladder population rows. Run from the repository root, with the
pinned feed and the 09-23 registry on disk::

    uv run python experiments/us-national-cd-concept-budget-20261012/acs_local_unchanged.py \
        experiments/us-national-cd-concept-budget-20261012/acs_local_unchanged.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path.cwd()
#: The commit before this change (origin/main when it was written).
BASE = "51a263726"
sys.path.insert(0, str(REPO / "experiments/us-acs-local-target-loss-weights-20261004"))
import distribution as dist  # noqa: E402

from microcosm.build.us_runtime import target_loss_weights as new  # noqa: E402
from microcosm.calibrate import TargetRegistry  # noqa: E402

source = subprocess.run(
    [
        "git",
        "show",
        f"{BASE}:packages/microcosm-build/src/microcosm/build/us_runtime/target_loss_weights.py",
    ],
    capture_output=True,
    text=True,
    check=True,
    cwd=REPO,
).stdout
old = types.ModuleType("target_loss_weights_main")
old.__dict__["__name__"] = "target_loss_weights_main"
sys.modules["target_loss_weights_main"] = old
exec(compile(source, "target_loss_weights_main.py", "exec"), old.__dict__)

out = {}
release = TargetRegistry.from_json(dist.DEFAULT_REGISTRY).specs
a = old.us_acs_local_target_loss_weights(release)
b = new.us_acs_local_target_loss_weights(release)
names = [new.target_row_name(s) for s in release]
out["release_20260923"] = {
    "n": len(release),
    "bit_identical": bool(np.array_equal(a, b)),
    "new_sha256": new.target_loss_weights_sha256(names, b),
    "recorded_sha256": "ba36f887407181dc6641596b4a200e6b18017cd815d7806ac4f523da3204f1d1",
}
print(json.dumps(out["release_20260923"]))

tool = dist.load_tool()
feed = (
    Path.home() / "PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_51aa40f.jsonl"
)
population = [s for s in release if s.family == "census_population"]
for mode in ("state", "totals", "full", "state_cd"):
    surface = tool.state_admin_surface(feed, dist.FAMILIES, soi_mode=mode)
    specs = [*surface.registry.specs, *population]
    a = old.us_acs_local_target_loss_weights(specs)
    b = new.us_acs_local_target_loss_weights(specs)
    district = sum(
        1
        for s in specs
        if s.metadata.get("ledger_geography_level") == "congressional_district"
    )
    diag = sum(
        1
        for s in specs
        if s.metadata.get("ledger_geography_level") == "congressional_district"
        and "diagnostic_target_label" in s.metadata
    )
    out[mode] = {
        "n": len(specs),
        "district_ledger_rows": district,
        "district_rows_with_diagnostic_label": diag,
        "bit_identical": bool(np.array_equal(a, b)),
    }
    print(mode, json.dumps(out[mode]))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
