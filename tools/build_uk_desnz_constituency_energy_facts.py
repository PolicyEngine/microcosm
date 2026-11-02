"""Build the UK constituency domestic energy resource (microcosm#1095).

Reads DESNZ's 2024 domestic electricity and gas consumption and meter counts
by local authority and by small area (2021 LSOAs, 2022 Scottish data zones)
from the pinned Chronicle consumer feed, verified against
``uk/chronicle_feed.json``, and the committed England-and-Wales and Scotland
atomic supports, verified against their provenance. It writes
``uk/desnz_constituency_energy_facts.json``: each authority's published total
split over its July 2024 constituencies by DESNZ's own small-area figures
(``uk_runtime/desnz_constituency_energy.py`` holds the method), in
consumer-fact row shape under the Microcosm-owned ``microcosm.derived_fact.``
key namespace.

The feed is streamed: every line is hashed and only the DESNZ record sets and
the census constituency households (for the seats' names) are parsed. Every
number the resource carries is an aggregate of public administrative data;
nothing here reads licensed microdata.

Example:
    uv run --no-sync python tools/build_uk_desnz_constituency_energy_facts.py \
        --ledger-facts <artifact>/consumer_facts.jsonl \
        --ledger-manifest <artifact>/manifest.json \
        --ew-support <supports>/uk_ew_output_area_2021_support.npz \
        --scotland-support <supports>/uk_scotland_output_area_2022_support.npz \
        --built-on 2026-10-25
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_energy_module():
    """Load the pure library module by path.

    Importing it through ``microcosm.build`` would load the UK country spec,
    which refuses a package whose declared resources are missing: the very
    resource this tool writes on a first run.
    """

    path = (
        REPO_ROOT
        / "packages/microcosm-build/src/microcosm/build/uk_runtime/desnz_constituency_energy.py"
    )
    spec = importlib.util.spec_from_file_location("desnz_constituency_energy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


energy = _load_energy_module()

UK_PACKAGE = REPO_ROOT / "packages/microcosm-build/src/microcosm/build/uk"
DEFAULT_FEED_PIN = UK_PACKAGE / "chronicle_feed.json"
DEFAULT_SUPPORT_PROVENANCE = UK_PACKAGE / energy.SUPPORT_PROVENANCE_RESOURCE
DEFAULT_CROSSWALK = UK_PACKAGE / energy.CROSSWALK_RESOURCE
DEFAULT_OUT = UK_PACKAGE / energy.FACTS_RESOURCE
#: The census households by constituency, read only for the seats' names.
SEAT_NAMES_SPEC_ID = "uk.local_geography.households.by_constituency.v1"
#: The inputs' licences and the attribution statements their publishers ask
#: for on derived data. The resource carries area aggregates only.
LICENCE = {
    "licence": "Open Government Licence v3.0 (every input)",
    "attribution": [
        (
            "Contains public sector information licensed under the Open Government "
            "Licence v3.0 (Department for Energy Security and Net Zero, sub-national "
            "and LSOA / data zone electricity and gas consumption statistics)."
        ),
        "Source: Office for National Statistics licensed under the Open Government Licence v.3.0",
        "Contains OS data © Crown copyright and database right 2024",
        "Contains NRS data © Crown copyright and database right 2024",
    ],
    "inputs": {
        "desnz_subnational": "DESNZ sub-national electricity and gas consumption statistics, Great Britain, 2005 to 2024, OGL v3.0",
        "desnz_small_area": "DESNZ LSOA and data zone domestic electricity and gas consumption, 2010 to 2024, OGL v3.0",
        "ons_lookups": "ONS output area (2021) lookups to LSOAs, local authorities and July 2024 constituencies, and Census 2021 TS041 households, OGL v3.0",
        "nrs_lookups": "NRS output area (2022) lookups to data zones, council areas and 2024 constituencies, and the Census 2022 index households, OGL v3.0",
    },
}


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        required=True,
        help="consumer_facts.jsonl of the pinned feed",
    )
    parser.add_argument(
        "--ledger-manifest",
        type=Path,
        required=True,
        help="manifest.json of the pinned feed",
    )
    parser.add_argument("--ew-support", type=Path, required=True)
    parser.add_argument("--scotland-support", type=Path, required=True)
    parser.add_argument("--feed-pin", type=Path, default=DEFAULT_FEED_PIN)
    parser.add_argument(
        "--support-provenance", type=Path, default=DEFAULT_SUPPORT_PROVENANCE
    )
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--built-on", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _feed_identity(pin_path: Path, manifest_path: Path) -> dict[str, object]:
    """The pin, after the manifest's bytes and its facts digest are checked."""

    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    manifest_bytes = manifest_path.read_bytes()
    if _sha256(manifest_bytes) != pin["manifest_sha256"]:
        raise ValueError(
            f"{manifest_path} is not the manifest uk/chronicle_feed.json pins."
        )
    manifest = json.loads(manifest_bytes)
    if manifest.get("facts_sha256") != pin["facts_sha256"]:
        raise ValueError("The feed manifest names other facts than the pin.")
    return {
        key: pin[key]
        for key in (
            "source_commit",
            "facts_sha256",
            "manifest_sha256",
            "fact_row_count",
        )
    }


def _read_feed(
    path: Path, *, markers: tuple[bytes, ...], identity: Mapping[str, object]
) -> list[dict]:
    """Every row carrying a marker; the whole file's digest and rows are checked."""

    digest = hashlib.sha256()
    count = 0
    facts = []
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if line.strip():
                count += 1
            if any(marker in line for marker in markers):
                facts.append(json.loads(line))
    if digest.hexdigest() != identity["facts_sha256"]:
        raise ValueError(f"{path} is not the feed uk/chronicle_feed.json pins.")
    if count != identity["fact_row_count"]:
        raise ValueError(
            f"{path} carries {count} rows; the pin records {identity['fact_row_count']}."
        )
    return facts


def _load_support(
    path: Path, name: str, provenance: Mapping[str, object]
) -> tuple[dict[str, np.ndarray], str]:
    payload = path.read_bytes()
    digest = _sha256(payload)
    if digest != provenance["supports"][name]["sha256"]:
        raise ValueError(
            f"The {name} support's sha256 differs from the committed provenance."
        )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {column: np.asarray(archive[column]) for column in metadata["columns"]}
    return arrays, digest


def _aliases(crosswalk: Mapping[str, object]) -> dict[str, str]:
    """A publisher's recoded authority to its roster code (microcosm#929)."""

    level = crosswalk["levels"]["local_authority"]
    return {
        str(code): str(area_id)
        for area_id, alias in (level.get("code_aliases") or {}).items()
        for code in alias.get("alias_codes", ())
    }


def _seat_names(facts: list[dict]) -> dict[str, str]:
    names: dict[str, str] = {}
    for row in facts:
        layout = row.get("layout") or {}
        geography = row.get("geography") or {}
        if layout.get("record_set_spec_id") != SEAT_NAMES_SPEC_ID:
            continue
        if geography.get("level") != energy.CONSTITUENCY_LEVEL:
            continue
        code = str(geography.get("id") or "")
        name = str(geography.get("name") or "").strip()
        if code and name:
            if names.setdefault(code, name) != name:
                raise ValueError(f"Constituency {code} carries two names in the feed.")
    return names


def _check_gas_shares(splits: Mapping[tuple[str, str], object]) -> dict[str, object]:
    """Each seat's gas meters over its electricity meters, as the applier reads them."""

    electricity = splits[("electricity", "meter_count")].seats
    gas = splits[("gas", "meter_count")].seats
    missing = sorted(set(gas) - set(electricity))
    if missing:
        raise ValueError(
            f"Seats {missing[:3]} carry gas meters but no electricity meters."
        )
    shares = {seat: gas[seat] / electricity[seat] for seat in gas}
    outside = sorted(seat for seat, share in shares.items() if not 0.0 < share <= 1.0)
    if outside:
        raise ValueError(
            f"The gas-connected share of seats {outside[:3]} is outside (0, 1]."
        )
    values = np.array(list(shares.values()))
    return {
        "seats_with_gas": len(shares),
        "seats_without_gas": sorted(set(electricity) - set(gas)),
        "share_min": round(float(values.min()), 4),
        "share_median": round(float(np.median(values)), 4),
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.write_text(rendered, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    identity = _feed_identity(args.feed_pin, args.ledger_manifest)
    provenance = json.loads(args.support_provenance.read_text(encoding="utf-8"))
    crosswalk = json.loads(args.crosswalk.read_text(encoding="utf-8"))
    record_sets = sorted(
        {energy.authority_record_set_id(fuel, args.year) for fuel in energy.FUELS}
        | {energy.small_area_record_set_id(fuel, args.year) for fuel in energy.FUELS}
    )
    markers = tuple(
        f'"{marker}"'.encode() for marker in (*record_sets, SEAT_NAMES_SPEC_ID)
    )
    _log("Streaming the pinned feed ...")
    facts = _read_feed(args.ledger_facts, markers=markers, identity=identity)
    _log("Loading the atomic supports ...")
    ew, ew_sha = _load_support(args.ew_support, "uk_ew_output_area_2021", provenance)
    scotland, scotland_sha = _load_support(
        args.scotland_support, "uk_scotland_output_area_2022", provenance
    )
    support = energy.support_frame([ew, scotland])
    series = energy.select_series(facts, year=args.year, aliases=_aliases(crosswalk))
    _log("Splitting the authorities' totals ...")
    splits = {key: energy.split_series(support, value) for key, value in series.items()}
    gas_shares = _check_gas_shares(splits)
    inputs = {
        "ledger_feed": identity,
        "support_sha256": {
            "uk_ew_output_area_2021": ew_sha,
            "uk_scotland_output_area_2022": scotland_sha,
        },
        "record_sets": record_sets,
        "method": energy.METHOD_VERSION,
    }
    inputs_sha256 = _sha256(json.dumps(inputs, sort_keys=True).encode())
    seat_names = _seat_names(facts)
    rows = energy.seat_fact_rows(
        series,
        splits,
        year=args.year,
        seat_names=seat_names,
        inputs_sha256=inputs_sha256,
    )
    payload = {
        "schema_version": 1,
        "kind": energy.RESOURCE_KIND,
        "resource": args.out.name,
        "description": (
            "Microcosm-derived domestic electricity and gas consumption and meter counts by "
            "July 2024 constituency, Great Britain: each authority's DESNZ sub-national total "
            "split over its constituencies by DESNZ's own LSOA and data zone figures for the "
            "same year, through the committed atomic supports (microcosm#1095). Not Chronicle "
            "facts."
        ),
        "generator": {
            "tool": "tools/build_uk_desnz_constituency_energy_facts.py",
            "version": energy.METHOD_VERSION,
        },
        "built_on": args.built_on,
        "licence": LICENCE,
        "inputs": inputs,
        "inputs_sha256": inputs_sha256,
        "parameters": {
            "year": args.year,
            "split_rule": (
                "Each output area's key is its LSOA's (data zone's) published value times the "
                "output area's share of the LSOA's census households; each authority's "
                "published total is shared among its constituency pieces in proportion to "
                "their keys, so every authority keeps its total and DESNZ's unallocated "
                "meters follow the split. A small area DESNZ publishes no row for carries "
                "no key."
            ),
            "small_area_sum_tripwire": energy.SMALL_AREA_SUM_TRIPWIRE,
            "scope": (
                "Great Britain: DESNZ publishes no small areas for Northern Ireland, which "
                "the family leaves out."
            ),
        },
        "receipts": {
            "series": {
                f"{fuel}.{metric}": split.receipt
                for (fuel, metric), split in sorted(splits.items())
            },
            "gas_connected_share": gas_shares,
        },
        "row_count": len(rows),
        "rows_sha256": energy.rows_sha256(rows),
        "rows": rows,
    }
    _write_json(args.out, payload)
    _log(f"Wrote {len(rows)} rows to {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
