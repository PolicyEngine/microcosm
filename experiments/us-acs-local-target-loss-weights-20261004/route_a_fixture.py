"""Write the Route A loss-weight golden fixture the engine-free tests read.

Route A (``populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z``, built from
commit 4b57d15a2, published tag-only on 2026-10-02) is the first certified US
release. Its ``calibration_diagnostics.json`` records, for each of its 5,694
targets, the spec fields and the ``target_loss_weight`` the release passed to
``calibrate``, plus the loss basis hash over (name, weight, scale).

The fixture keeps, per target, exactly what the national weighting reads:
name, entity, period, family, filter, value, ``measure_mode``,
``source_measure_id`` and ``ledger_geography_level``, with the recorded
weight. Route A has no congressional-district rows, so no other metadata
reaches its concept keys. Before writing, this script checks that the shared
module reproduces every recorded weight bit for bit from the full metadata and
from the reduced metadata, and that the recorded loss basis hash matches. The
test then pins the same equalities from the fixture alone.

Run from the repository root::

    uv run python experiments/us-acs-local-target-loss-weights-20261004/route_a_fixture.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RELEASE_ID = "populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z"
DIAGNOSTICS = Path(
    "/Users/maxghenis/PolicyEngine/_recovered/scratch-backup/893/overnight-20260923/"
    "route-a/run-4b57d15a287c/release-out/"
    f"{RELEASE_ID}/releases/{RELEASE_ID}/calibration_diagnostics.json"
)
BUILD_COMMIT = "4b57d15a287c4a1729bad362a2f6d57c4de9059e"
DIAGNOSTICS_SHA256 = "64b55a02aaf573964cffee44d4a86bd63f33dc30c7f5012e4345348eecc06f80"
FIXTURE = (
    REPO / "packages/microcosm-build/tests/fixtures/us_route_a_target_loss_weights.json"
)
#: The only metadata the national weighting reads from a non-district row.
METADATA_KEYS = ("measure_mode", "source_measure_id", "ledger_geography_level")
COLUMNS = ("name", "entity", "period", "family", "filter", "value", *METADATA_KEYS)


def main() -> int:
    from microcosm.build.us_runtime import target_loss_weights as lw
    from microcosm.calibrate import TargetRegistry, TargetSpec
    from microcosm.calibrate._target_loss_attribution import target_loss_basis_hash

    raw = DIAGNOSTICS.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DIAGNOSTICS_SHA256:
        raise SystemExit(f"{DIAGNOSTICS} is not Route A's diagnostics.")
    diagnostics = json.loads(raw)
    targets = diagnostics["targets"]
    levels = {target["metadata"].get("ledger_geography_level") for target in targets}
    if "congressional_district" in levels:
        raise SystemExit("Route A has district rows; the reduced fixture is not exact.")

    def spec(target, metadata):
        return TargetSpec(
            name=target["target_name"],
            entity=target["entity"],
            measure=target["target_name"],
            value=target["target"],
            period=target["period"],
            family=target["registry"]["family"],
            filter=target["filter"],
            source=target["source"],
            metadata=metadata,
        )

    recorded = np.asarray(
        [target["target_loss_weight"] for target in targets], dtype=np.float64
    )
    full = lw.fiscal_target_loss_weights(
        TargetRegistry([spec(t, t["metadata"]) for t in targets], country="us")
    )
    reduced_specs = [
        spec(
            t,
            {key: t["metadata"][key] for key in METADATA_KEYS if key in t["metadata"]},
        )
        for t in targets
    ]
    reduced = lw.fiscal_target_loss_weights(TargetRegistry(reduced_specs, country="us"))
    if not (np.array_equal(full, recorded) and np.array_equal(reduced, recorded)):
        raise SystemExit("The shared module does not reproduce Route A's weights.")
    names = [target["name"] for target in targets]
    if names != [f"{t['target_name']}@{t['period']}" for t in targets]:
        raise SystemExit("Route A's diagnostic names are not <target_name>@<period>.")
    scales = np.asarray([t["target_loss_scale"] for t in targets], dtype=np.float64)
    basis = diagnostics["target_loss_basis"]
    if target_loss_basis_hash(names, recorded, scales) != basis["sha256"]:
        raise SystemExit("Route A's loss basis hash does not match its weights.")

    rows = [
        [
            t["target_name"],
            t["entity"],
            t["period"],
            t["registry"]["family"],
            t["filter"],
            t["target"],
            *(t["metadata"].get(key) for key in METADATA_KEYS),
        ]
        for t in targets
    ]
    fixture = {
        "description": (
            "Route A's 5,694 calibration targets: the fields the national "
            "target-loss weighting reads, and the weight the release recorded. "
            "Written by experiments/us-acs-local-target-loss-weights-20261004/"
            "route_a_fixture.py."
        ),
        "release_id": RELEASE_ID,
        "build_commit": BUILD_COMMIT,
        "calibration_diagnostics_sha256": hashlib.sha256(raw).hexdigest(),
        "target_loss_weighting": diagnostics["build"]["target_loss_weighting"],
        "target_loss_basis": {
            "hash_algorithm": basis["hash_algorithm"],
            "sha256": basis["sha256"],
            "names": "<name>@<period>",
            "scales": "max(abs(value), 1)",
        },
        "columns": list(COLUMNS),
        "rows": rows,
        "target_loss_weights": recorded.tolist(),
    }
    FIXTURE.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size:,} bytes, {len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
