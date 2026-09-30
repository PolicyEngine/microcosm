"""Build the 117th->119th CD vintage crosswalk artifact from the CD plan registry.

Reads the pinned US block -> congressional-district plan registry (every
populated 2020 block's district under each plan, with its 2020 P.L. 94-171
population), overlays its ``117th_congress`` and ``119th_congress`` plans on
the same blocks weighted by block population, and writes the
``source_geography_id,target_geography_id,weight`` crosswalk that
``congressional_district_vintage.translate_congressional_district_facts_to_current_vintage``
consumes, plus a sidecar provenance JSON recording the registry's SHA-256, its
117th-plan known deviations, the build metadata, and per-state population
conservation.

The registry carries the 117th plan correctly for North Carolina, whose 117th
districts (the 2019 remedial plan) differ from the 116th plan in the 2020
Block Assignment Files; an earlier version of this tool read the BAF layer
directly and so mapped North Carolina's SOI districts through the wrong plan.
The registry's sources and checks are documented in
``packages/microcosm-build/src/microcosm/build/us_runtime/US_CD_PLAN_REGISTRY.md``.

The method and rationale live in
``microcosm.build.us_runtime.congressional_district_vintage_crosswalk``. The
crosswalk is a regenerable build artifact, not a Ledger fact (the fact-vs-
computed boundary of PolicyEngine/ledger#71); it feeds the CD geography-vintage
translation that PolicyEngine/microcosm#205 requires.

Example:
    uv run --python 3.13 --package microcosm-build --group dev python \
        tools/build_us_congressional_district_vintage_crosswalk.py \
        --cd-plan-registry build/us/us_cd_plan_registry_2020.npz \
        --out packages/microcosm-build/src/microcosm/build/us_runtime/data/\
congressional_district_vintage_crosswalk.csv

    # Smoke run over a few states:
    uv run python tools/build_us_congressional_district_vintage_crosswalk.py \
        --cd-plan-registry build/us/us_cd_plan_registry_2020.npz \
        --out /tmp/cd_xwalk_smoke.csv --states 08,30,37
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from microcosm.build.us_runtime.block_ladder_sources import US_STATES
from microcosm.build.us_runtime.cd_plan_registry import (
    US_CD_PLAN_REGISTRY_PROVENANCE_RESOURCE,
    load_pinned_us_cd_plan_registry,
)
from microcosm.build.us_runtime.congressional_district_vintage import (
    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
)
from microcosm.build.us_runtime.congressional_district_vintage_crosswalk import (
    build_cd_vintage_crosswalk_rows,
)

SOURCE_CONGRESSIONAL_DISTRICT_VINTAGE = "117th_congress"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the 117th->119th CD vintage crosswalk CSV artifact."
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--cd-plan-registry",
        required=True,
        type=Path,
        help=(
            "The block -> CD-plan registry NPZ. It must be the artifact the "
            "package pins (us_cd_plan_registry.provenance.json)."
        ),
    )
    parser.add_argument(
        "--states",
        help=(
            "Optional comma-separated state FIPS subset (smoke runs only; a "
            "published crosswalk covers all 51)."
        ),
    )
    parser.add_argument(
        "--provenance-json",
        type=Path,
        help="Path for the provenance sidecar. Defaults to --out with a "
        "'.provenance.json' suffix.",
    )
    return parser.parse_args(argv)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    selected = (
        {value.strip() for value in args.states.split(",")} if args.states else None
    )
    states = [entry for entry in US_STATES if selected is None or entry[0] in selected]
    if selected is not None:
        unknown = selected - {entry[0] for entry in US_STATES}
        if unknown:
            raise SystemExit(f"Unknown state FIPS in --states: {sorted(unknown)}")
    selected_fips = {int(fips) for fips, _, _ in states}
    _log(f"Building CD117->CD119 crosswalk for {len(states)} state(s)")

    registry = load_pinned_us_cd_plan_registry(args.cd_plan_registry)
    for plan in (
        SOURCE_CONGRESSIONAL_DISTRICT_VINTAGE,
        CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
    ):
        if plan not in registry.plans:
            raise SystemExit(f"The CD plan registry has no {plan!r} plan.")
    blocks = registry.block_geoid.tolist()
    in_scope = [block // 10**13 in selected_fips for block in blocks]
    old_cd_by_block = {
        block: f"{district % 100:02d}"
        for block, district, keep in zip(
            blocks,
            registry.plans[SOURCE_CONGRESSIONAL_DISTRICT_VINTAGE].tolist(),
            in_scope,
            strict=True,
        )
        if keep
    }
    current_cd_by_block = {
        block: f"{district % 100:02d}"
        for block, district, keep in zip(
            blocks,
            registry.plans[CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE].tolist(),
            in_scope,
            strict=True,
        )
        if keep
    }
    block_population = {
        block: population
        for block, population, keep in zip(
            blocks, registry.population.tolist(), in_scope, strict=True
        )
        if keep
    }

    rows, diagnostics = build_cd_vintage_crosswalk_rows(
        old_cd_by_block=old_cd_by_block,
        current_cd_by_block=current_cd_by_block,
        block_population=block_population,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "source_geography_id",
                "target_geography_id",
                "pair_population",
                "weight",
            ],
            lineterminator="\n",  # Unix newlines; RFC 4180 CRLF trips git checks.
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    crosswalk_sha256 = _sha256(args.out)

    source_spec = registry.plan_sources[SOURCE_CONGRESSIONAL_DISTRICT_VINTAGE]
    provenance = {
        "schema_version": 1,
        "kind": "us_congressional_district_vintage_crosswalk",
        "source_geography_vintage": SOURCE_CONGRESSIONAL_DISTRICT_VINTAGE,
        "target_geography_vintage": CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
        "block_vintage": registry.metadata["block_vintage"],
        "crosswalk_sha256": crosswalk_sha256,
        "method": (
            "Population-weighted overlay of the 117th and 119th plans of the US "
            "block -> congressional-district plan registry on 2020 tabulation "
            "blocks, weighted by 2020 P.L. 94-171 block populations. Each old "
            "district's population is redistributed across current districts; "
            "totals conserve by construction. The registry's 117th plan is the "
            "2020 BAF CD layer except North Carolina, carried from its 2019 plan "
            "(see known_deviations)."
        ),
        "sources": {
            "cd_plan_registry": {
                "artifact": args.cd_plan_registry.name,
                "sha256": registry.sha256,
                "receipt": (
                    "microcosm.build.us_runtime/"
                    f"{US_CD_PLAN_REGISTRY_PROVENANCE_RESOURCE}"
                ),
                "plans": [
                    SOURCE_CONGRESSIONAL_DISTRICT_VINTAGE,
                    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
                ],
            }
        },
        "known_deviations": source_spec.get("known_deviations", {}),
        "diagnostics": diagnostics,
        "states": [fips for fips, _, _ in states],
    }
    provenance_path = args.provenance_json or args.out.with_suffix(
        args.out.suffix + ".provenance.json"
    )
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    _validate_conservation(diagnostics)

    _log(
        f"DONE {len(rows)} rows; "
        f"{diagnostics['source_district_count']} source -> "
        f"{diagnostics['target_district_count']} target districts; "
        f"crosswalk sha256 {crosswalk_sha256}"
    )
    _log(f"  crosswalk:  {args.out}")
    _log(f"  provenance: {provenance_path}")


def _validate_conservation(diagnostics: dict[str, object]) -> None:
    """Fail loudly if any state loses population beyond a small block-coverage gap.

    The registry assigns every populated block a district under every plan,
    so any uncovered population means the registry and this tool disagree on
    the block universe: a defect, not a silent redistribution.
    """

    state_conservation = diagnostics.get("state_conservation")
    if not isinstance(state_conservation, dict):
        raise ValueError("Crosswalk diagnostics missing state_conservation.")
    offenders: list[str] = []
    for state_fips, record in state_conservation.items():
        state_population = int(record["state_population"])
        unmatched = int(record["unmatched_population"])
        if state_population <= 0:
            continue
        if unmatched / state_population > 0.001:  # >0.1% of a state uncovered
            offenders.append(
                f"{state_fips}: {unmatched:,}/{state_population:,} "
                f"({unmatched / state_population:.2%}) uncovered"
            )
    if offenders:
        raise ValueError(
            "CD vintage crosswalk leaves too much population uncovered "
            "(possible source/layout drift): " + "; ".join(offenders)
        )


if __name__ == "__main__":
    main()
