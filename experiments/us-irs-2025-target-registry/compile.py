"""Compile an offline annual US target registry from the pinned base and IRS supplements."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from microcosm.build.us_runtime.congressional_district_vintage import (
    load_congressional_district_vintage_crosswalk,
)
from microcosm.build.us_runtime.fiscal_targets import (
    US_FISCAL_TARGET_SUPPORT_EXCLUSIONS,
    compile_us_fiscal_target_registry,
)

REPO = Path(__file__).resolve().parents[2]
GEO = REPO / "packages/microcosm-build/src/microcosm/build/us_runtime/data"
PINS = json.loads(Path(__file__).with_name("sources.json").read_text())
# The committed feed labels these older artifacts as TY2023. It also
# contains 1,020 genuine TY2023 state/AGI rows in the refreshed HT2 package.
REPLACE = {
    "20in04w2all.xlsx": 5,
    "22in05ira.xlsx": 2,
    "22in06ira.xlsx": 2,
    "22in54us.xlsx": 4,
    "22incd.csv": 26_880,
    "23in55cmcsv.csv": 1_020,
}
HELD = {
    f"irs_soi.ty2023.historic_table_2.state_agi.{state}.under_1.taxable_interest_{measure}"
    for state in ("ia", "nd", "vt")
    for measure in ("amount", "returns")
}


def read_rows(path, expected_hash):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for raw in stream:
            sha.update(raw)
            yield json.loads(raw)
    if sha.hexdigest() != expected_hash:
        raise ValueError(f"Source digest mismatch for {path}")


def record_id(row):
    return row["lineage"]["source_record_id"]


def superseded_interest_band(row):
    """The 2022 aggregate spans the two source bands separately published in 2023."""
    return (
        row["source"]["source_file"] == "22in55cmcsv.csv"
        and row["period"] == {"type": "tax_year", "value": 2022}
        and ".state_agi." in record_id(row)
        and row["layout"]["groupby_value_id"] == "500k_plus"
        and row["layout"]["measure_id"]
        in ("taxable_interest_returns", "taxable_interest_amount")
    )


def successor_bands(rid):
    prefix = rid.replace(".ty2022.", ".ty2023.")
    return [
        prefix.replace(".500k_plus.", f".{band}.") for band in ("500k_to_1m", "1m_plus")
    ]


def assemble(base, supplements):
    kept, removed, unchanged, held = [], Counter(), {}, {}
    retired = {}
    for row in base:
        source = row["source"]
        filename = source["source_file"]
        if (
            source["source_name"] == "irs_soi"
            and row["period"]["value"] == 2023
            and filename in REPLACE
        ):
            removed[filename] += 1
            if filename == "23in55cmcsv.csv":
                unchanged[record_id(row)] = row["value"]
        elif superseded_interest_band(row):
            retired[record_id(row)] = successor_bands(record_id(row))
        else:
            kept.append(row)
    if removed != REPLACE:
        raise ValueError(f"Base source coverage changed: {removed}")
    inputs = Counter()
    known_ids = {record_id(r) for r in kept}
    for row in supplements:
        source = row["source"]
        filename = source["source_file"]
        expected = PINS["publisher_artifacts"].get(filename)
        if (
            expected is None
            or source["source_sha256"] != expected["sha256"]
            or source["url"] != expected["url"]
            or not row["lineage"]["source_cell_keys"]
            or source["source_name"] != "irs_soi"
            or row["period"] != {"type": "tax_year", "value": expected["year"]}
        ):
            raise ValueError(f"Unexpected publisher, period or digest for {filename}")
        rid = record_id(row)
        if rid in known_ids or not rid.startswith(f"irs_soi.ty{expected['year']}."):
            raise ValueError(f"Unexpected or duplicate source ID {rid}")
        known_ids.add(rid)
        inputs[filename] += 1
        if rid in unchanged and unchanged.pop(rid) != row["value"]:
            raise ValueError(f"Existing TY2023 AGI observation changed: {rid}")
        if rid in HELD:
            # Data support was adjudicated for this cell; newer publisher
            # bytes alone do not establish positive support in microdata.
            prior = rid.replace(".ty2023.", ".ty2022.")
            held[rid] = {
                "value": row["value"],
                "reason": US_FISCAL_TARGET_SUPPORT_EXCLUSIONS[prior],
            }
        else:
            kept.append(row)
    if (
        unchanged
        or held.keys() != HELD
        or inputs
        != {f: spec["rows"] for f, spec in PINS["publisher_artifacts"].items()}
    ):
        raise ValueError("Incomplete successor data or support-exclusion coverage.")
    if len(retired) != 102 or any(
        not set(pair) <= known_ids for pair in retired.values()
    ):
        raise ValueError(
            "A superseded TY2022 interest band has incomplete TY2023 coverage."
        )
    return kept, {
        "superseded_interest_bands": retired,
        "replaced_base_rows": dict(removed),
        "observations": dict(inputs),
        "held": held,
    }


def compile_registry(rows, year, crosswalk_name):
    path = GEO / crosswalk_name
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != PINS["crosswalks"][crosswalk_name]
    ):
        raise ValueError(f"Crosswalk digest mismatch for {path}")
    return compile_us_fiscal_target_registry(
        rows,
        target_period=year,
        age_targets=True,
        congressional_district_vintage_crosswalk=load_congressional_district_vintage_crosswalk(
            path
        ),
    )


def non_irs_controls(registry):
    return {s.key: s for s in registry.specs if s.family != "irs_soi"}


def record_sets(registry):
    return dict(
        sorted(
            Counter(
                s.metadata.get("ledger_layout_record_set_spec_id", "derived")
                for s in registry.specs
                if s.family == "irs_soi"
            ).items()
        )
    )


def families(registry):
    return dict(sorted(Counter(s.family for s in registry.specs).items()))


def write_specs(path, specs):
    # Compress directly, retaining each native TargetSpec and its source metadata.
    with (
        path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream,
    ):
        for s in specs:
            stream.write(
                (
                    json.dumps(s.to_dict(), sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode()
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "annual", "geographic", "w2-2021", "w2-2022", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--year", type=int, choices=(2024, 2025), default=2025)
    parser.add_argument("--county-universe", type=Path)
    args = parser.parse_args()
    files = vars(args)
    base = list(read_rows(args.base, PINS["feeds"]["base"]))
    extra = (
        r
        for label in ("annual", "geographic", "w2_2021", "w2_2022")
        for r in read_rows(files[label], PINS["feeds"][label])
    )
    rows, receipt = assemble(base, extra)
    baseline = compile_registry(
        base, args.year, "congressional_district_vintage_crosswalk.csv"
    )
    candidate = compile_registry(
        rows, args.year, "congressional_district_vintage_crosswalk_118_119.csv"
    )
    selected_ids = {s.metadata.get("ledger_source_record_id") for s in candidate.specs}
    for old, children in receipt["superseded_interest_bands"].items():
        if old in selected_ids or not set(children) <= selected_ids:
            raise ValueError(f"Incomplete compiled interest coverage for {old}.")
    if non_irs_controls(baseline) != non_irs_controls(candidate):
        raise ValueError("The candidate changes a non-IRS control.")
    receipt.update(
        year=args.year,
        input_hashes=PINS["feeds"],
        base_count=len(baseline.specs),
        candidate_count=len(candidate.specs),
        base_families=families(baseline),
        candidate_families=families(candidate),
        base_record_sets=record_sets(baseline),
        candidate_record_sets=record_sets(candidate),
        base_version=baseline.version,
        candidate_version=candidate.version,
        unmodified_non_irs_controls=len(non_irs_controls(baseline)),
        input_observations=len(rows),
    )
    surface = None
    if args.county_universe:
        # The county adapter takes these codes to check complete blocks; the
        # county return and AGI allocations use the two published IRS measures.
        sys.path.insert(0, str(REPO / "tools"))
        from build_us_acs_local_release import SOI_MODE_STATE, soi_surface_predicate
        from us_acs_local_county_surface import state_county_soi_surface

        pops = {
            int(r["geography"]["id"].removeprefix("0500000US")): r["value"]
            for r in read_rows(args.county_universe, PINS["county_universe_sha256"])
            if r["layout"]["record_set_id"]
            == "census_pep.vintage2024.county_population"
            and r["period"]["value"] == 2024
            and r["layout"]["measure_id"] == "resident_population"
        }
        if not pops:
            raise ValueError("The independent county universe is empty.")
        county = [r for r in rows if r["source"]["source_file"] == "23incyallnoagi.csv"]
        parents = candidate.select(
            family="irs_soi", predicate=soi_surface_predicate(SOI_MODE_STATE)
        ).specs
        surface = state_county_soi_surface(parents, county, pops)
        receipt["county"] = surface.receipt
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as temp:
        stage = Path(temp)
        write_specs(stage / "registry.jsonl.gz", candidate.specs)
        if surface is not None:
            write_specs(stage / "county-registry.jsonl.gz", surface.specs)
        (stage / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        # An existing directory may hold outputs for a different year or run.
        # Publish a complete new run atomically and leave existing outputs intact.
        stage.rename(args.out)


if __name__ == "__main__":
    main()
