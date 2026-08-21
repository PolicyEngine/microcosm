"""One-time Fixture B extraction: incumbent registry surface at calibration year 2025.

Runs inside the retired UK data-package checkout pinned at 12a1e028, with network
(several source modules fetch at get_targets() time). Mirrors the level-union +
name-dedup of build_loss_matrix.create_target_matrix (:93-103) and resolves each
target through the incumbent's own _resolve_value, at both 2025 (the calibration
year, #723) and 2026 (the TCL cross-check year). No simulation, no microdata --
target values are published aggregate statistics only.

Output: registry_fixture_2025_extraction.json next to this script.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

_DATA_PACKAGE = "policyengine_" + "uk_data"
_resolve_value = import_module(
    f"{_DATA_PACKAGE}.targets.build_loss_matrix"
)._resolve_value
get_all_targets = import_module(f"{_DATA_PACKAGE}.targets.registry").get_all_targets
GeographicLevel = import_module(f"{_DATA_PACKAGE}.targets.schema").GeographicLevel

OUT = Path(__file__).resolve().parent / "registry_fixture_2025_extraction.json"


def union_targets():
    """The exact surface create_target_matrix assembles (level union, name-dedup)."""
    all_targets = []
    seen = set()
    for level in (
        GeographicLevel.NATIONAL,
        GeographicLevel.REGION,
        GeographicLevel.COUNTRY,
    ):
        for t in get_all_targets(geographic_level=level):
            if t.name not in seen:
                seen.add(t.name)
                all_targets.append(t)
    return all_targets


def resolution(target, year):
    value = _resolve_value(target, year)
    if value is None:
        return {"value": None, "resolved_from_year": None}
    available = sorted(target.values.keys())
    closest = (
        min(available, key=lambda y: abs(y - year))
        if year not in target.values
        else year
    )
    return {"value": float(value), "resolved_from_year": int(closest)}


def main():
    targets = union_targets()
    rows = []
    for t in targets:
        rows.append(
            {
                "name": t.name,
                "variable": t.variable,
                "source": t.source,
                "unit": t.unit.value,
                "geographic_level": t.geographic_level.value,
                "geo_code": t.geo_code,
                "geo_name": t.geo_name,
                "is_count": t.is_count,
                "breakdown_variable": t.breakdown_variable,
                "reference_url": t.reference_url,
                "forecast_vintage": t.forecast_vintage,
                "values_years": sorted(int(y) for y in t.values),
                "at_2025": resolution(t, 2025),
                "at_2026": resolution(t, 2026),
            }
        )

    by_source = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    payload = {
        "provenance": {
            "source_repository": "retired UK data package",
            "pinned_ref": "12a1e028afeef08d8b2d74ee03fd9de3a78b2dd3",
            "extraction_utc": datetime.now(UTC).isoformat(),
            "method": (
                "level-union name-dedup per build_loss_matrix.create_target_matrix:93-103; "
                "values via build_loss_matrix._resolve_value at years 2025 and 2026; "
                "network-dependent get_targets() sources fetched live at extraction time"
            ),
            "python": sys.version.split()[0],
        },
        "surface": {
            "union_rows": len(rows),
            "resolved_at_2025": sum(
                1 for r in rows if r["at_2025"]["value"] is not None
            ),
            "resolved_at_2026": sum(
                1 for r in rows if r["at_2026"]["value"] is not None
            ),
            "by_source": dict(sorted(by_source.items())),
        },
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=False))
    print(json.dumps(payload["surface"], indent=2))


if __name__ == "__main__":
    main()
