"""Build J STEP 2 — selection carry-over verification (#330 identity join).

Re-derive the Build I rmloss100 57,240-record selection onto the NEW Build J base by
joining the manifest identities (join key: source_year, source_household_id,
household_support_channel, household_support_clone_index) onto the base frame in
"frozen_support" mode. That mode RAISES on any unmapped or ambiguous identity — the
task's STOP condition (">0 missing identities = STOP and report"). Reports n_selected
(expect 57,240), n_unmapped (expect 0), n_ambiguous (expect 0). STAGING/LOCAL ONLY.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

WT = Path("/Users/maxghenis/PolicyEngine/_worktrees/microcosm-build-j-recert")
sys.path.insert(0, str(WT / "tools"))
import build_us_fiscal_refresh_release as release  # noqa: E402

from microcosm.build.us_runtime.warm_start_selection import (  # noqa: E402
    load_selection_source_from_manifest,
)

BASE = Path("/Users/maxghenis/PolicyEngine/_buildj-runtime/out/base-j/base_populace_us_2024_puf_support.h5")
MANIFEST = Path("/Users/maxghenis/PolicyEngine/_buildi-runtime/inputs/buildi_rmloss100_selection_source.json")
EXPECT_N = 57_240


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print(f"base sha:     {_sha256(BASE)}")
    print(f"manifest sha: {_sha256(MANIFEST)[:16]}…")
    base_frame = release._load_frame(str(BASE))
    print(f"base frame n(household) = {base_frame.n('household')}")
    source = load_selection_source_from_manifest(MANIFEST)
    print(f"manifest: n_selected={source.size} join_key={source.join_key} "
          f"identities_sha256={getattr(source, 'identities_sha256', 'n/a')}")
    try:
        mask, report = source.base_selection_mask(base_frame, mode="frozen_support")
    except Exception as exc:  # frozen_support raises on unmapped/ambiguous
        print(f"CARRY-OVER FAIL (STOP): {type(exc).__name__}: {exc}")
        return 2
    print("=== CARRY-OVER REPORT ===")
    print(f"  n_selected  = {report.n_selected}  (expect {EXPECT_N})")
    print(f"  n_unmapped  = {report.n_unmapped}  (expect 0)")
    print(f"  n_ambiguous = {report.n_ambiguous}  (expect 0)")
    print(f"  mode        = {report.mode}")
    print(f"  mask.sum()  = {int(mask.sum())}")
    ok = (
        report.n_selected == EXPECT_N
        and report.n_unmapped == 0
        and report.n_ambiguous == 0
        and int(mask.sum()) == EXPECT_N
    )
    print(f"CARRY-OVER {'PASS' if ok else 'FAIL (STOP)'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
