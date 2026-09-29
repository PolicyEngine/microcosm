"""Measure the Ledger filter guards against a compiled US fiscal registry.

Compiles the US fiscal target registry from a Chronicle consumer-facts feed
and reports, for the whole registry and for the state-level surface
``tools/build_us_acs_local_release.py`` selects in both SOI modes:

* target counts by family and ``target_role``;
* every ``ledger_filter_*`` key carried, with its value histogram;
* what ``_unsupported_ledger_filter_metadata`` refuses (the fatal guard) and
  what ``_unsupported_soi_ledger_filters`` would skip silently;
* a sample of compiled metadata mappings, one per (family, ``target_role``)
  plus IRS SOI rows carrying an AGI band or an EITC child-count filter, which
  regenerates ``tests/fixtures/us_compiled_ledger_filter_specs.json``.

Usage::

    uv run python experiments/us-labelled-filter-support/\\
        census_compiled_ledger_filters.py <consumer_facts.jsonl> [out.json]

Needs no engine: the compile path reads the feed and the congressional-district
vintage crosswalk only. It does read the whole feed, so it is slow (order ten
minutes for the 39,158-fact US feed) and is a receipt, not a test.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _survey(builder, specs: list, label: str) -> dict[str, object]:
    unsupported = builder._unsupported_ledger_filter_metadata(specs)
    refusals = collections.Counter(
        entry for entries in unsupported.values() for entry in entries
    )
    skipped = collections.Counter()
    skipped_targets = 0
    for spec in specs:
        if getattr(spec, "family", "") != "irs_soi":
            continue
        keys = builder._unsupported_soi_ledger_filters(spec.metadata)
        if keys:
            skipped_targets += 1
            skipped.update(keys)
    keys = collections.Counter()
    values: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for spec in specs:
        for key, value in spec.metadata.items():
            if str(key).startswith("ledger_filter"):
                keys[str(key)] += 1
                values[str(key)][str(value)] += 1
    roles = collections.Counter(
        (spec.family, spec.metadata.get("target_role", "")) for spec in specs
    )
    return {
        "label": label,
        "targets": len(specs),
        "by_family": dict(collections.Counter(spec.family for spec in specs)),
        "by_family_role": {
            f"{family}|{role}": count for (family, role), count in sorted(roles.items())
        },
        "ledger_filter_key_census": {
            key: {"targets": count, "values": dict(values[key].most_common())}
            for key, count in keys.most_common()
        },
        "refused_targets": len(unsupported),
        "refusals": dict(refusals),
        "soi_silently_skipped_targets": skipped_targets,
        "soi_skipped_keys": dict(skipped),
    }


def _fixture_rows(specs: list) -> list[dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for spec in specs:
        role = (spec.family, spec.metadata.get("target_role", ""))
        if role in {
            (row["family"], row["metadata"].get("target_role", ""))
            for row in rows.values()
        }:
            continue
        rows[spec.name] = {
            "name": spec.name,
            "family": spec.family,
            "metadata": dict(sorted(spec.metadata.items())),
        }
    banded = 0
    for spec in specs:
        if spec.family != "irs_soi" or spec.name in rows:
            continue
        metadata = spec.metadata
        interesting = metadata.get("ledger_filter_eitc_child_count") or (
            metadata.get("agi_lower_bound") not in (None, "-inf")
            and metadata.get("agi_upper_bound") not in (None, "inf")
        )
        if not interesting:
            continue
        rows[spec.name] = {
            "name": spec.name,
            "family": spec.family,
            "metadata": dict(sorted(metadata.items())),
        }
        banded += 1
        if banded >= 4:
            break
    return sorted(rows.values(), key=lambda row: row["name"])


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    feed = argv[0]
    out_path = Path(argv[1]) if len(argv) > 1 else None

    builder = _load_tool("build_us_fiscal_refresh_release")
    acs = _load_tool("build_us_acs_local_release")

    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from microcosm.build.us_runtime import (
        default_congressional_district_vintage_crosswalk_path,
        load_congressional_district_vintage_crosswalk,
    )
    from microcosm.build.us_runtime.fiscal_targets import (
        compile_us_fiscal_target_registry,
    )

    crosswalk = load_congressional_district_vintage_crosswalk(
        default_congressional_district_vintage_crosswalk_path()
    )
    artifact = load_ledger_consumer_artifact(feed)
    registry = compile_us_fiscal_target_registry(
        artifact.facts,
        target_period=acs.PERIOD,
        congressional_district_vintage_crosswalk=crosswalk,
        age_targets=True,
    )
    whole = list(registry.specs)

    surveys = [_survey(builder, whole, "whole_registry")]
    for soi_mode in ("full", "totals"):
        selected, _substitutions = acs.state_admin_specs(
            feed, ["snap", "medicaid", "soi"], soi_mode=soi_mode
        )
        surveys.append(
            _survey(builder, list(selected.specs), f"state_surface_{soi_mode}")
        )

    payload = {
        "feed": feed,
        "target_period": acs.PERIOD,
        "surveys": surveys,
        "fixture_specs": _fixture_rows(whole),
    }
    text = json.dumps(payload, indent=2, default=str)
    if out_path is None:
        print(text)
    else:
        out_path.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
