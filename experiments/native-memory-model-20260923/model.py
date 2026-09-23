"""Scale the invented-data retention census to real fractions.

Reads ``retention-census.json`` (written by ``test_retention_census.py``) and
prints, for three points of ``run_atomic_survey_financial`` -- the prefix
runner's own peak (after its fresh geography reconstruction), just after the
19-node ``run_graph`` returns, and after the independent replay (the call of
``atomic._states``) -- the unique physical bytes each runner local holds,
scaled from invented row counts to real row counts at 1/1000, 1/15 and full
source. It prints the measured whole-process peaks of the same 19-node
configuration beside them and writes ``model-summary.json``.

What is measured and what is assumed is printed with the numbers:

* measured (invented data, this tree): which objects are live at the
  post-replay point, which of them share physical buffers, and bytes per row
  per entity table (column count and dtype mix are declared by the graph, so
  they transfer; Python ``str`` cell sizes depend on string lengths);
* measured (real data, receipts): cloned entity counts at 1/15 and 1/1000,
  the documented full-source household/person counts, and the two peaks;
* assumed: the stacked grain has exactly half the cloned rows (true of the
  invented clone and of the documented full-source counts); the ASEC donor
  grain is 55,117 / 1,587,376 = 3.47% of the stacked grain (the ACS share in
  the recovered 1/1000 preparation receipt), scaled by fraction; unscaled
  byte payloads (preparation payload, artifacts) are omitted and reported
  separately, because invented payload sizes do not transfer.

Usage: ``python model.py [retention-census.json]``
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

GiB = 2**30

# Real cloned-grain entity counts at 1/15 (receipt a36f5c69...,
# runner.households) and at 1/1000 (native-attribution-9af56aa8a results.md).
CLONED_1_15 = {
    "person": 475_728,
    "household": 211_638,
    "tax_unit": 289_972,
    "spm_unit": 211_992,
    "family": 212_576,
    "marital_unit": 380_690,
}
CLONED_1_1000 = {
    "person": 6_928,
    "household": 3_168,
    "tax_unit": 4_256,
    "spm_unit": 3_170,
    "family": 3_188,
    "marital_unit": 5_548,
}
# docs/us-native-row-ceilings.md:82-94 (catalogue-derived): full source cloned
# 3,174,752 households / 7,130,026 people. Other entities scale with households
# at their 1/15 ratio to households.
FULL_HH, FULL_PERSON = 3_174_752, 7_130_026
CLONED_FULL = {
    e: round(n * FULL_HH / CLONED_1_15["household"]) for e, n in CLONED_1_15.items()
}
CLONED_FULL["person"] = FULL_PERSON
ASEC_SHARE_OF_STACKED = 55_117 / 1_587_376

TARGETS = {"1/1000": CLONED_1_1000, "1/15": CLONED_1_15, "full": CLONED_FULL}
MEASURED = {
    "1/1000": "peak 11,900,780,544 B (11.08 GiB) whole process; cold-phase sampled"
    " peak 11.20 GB; RSS after cold phase 6.54 GB (9af56aa8a, all retention)",
    "1/15": "peak 46,665,105,408 B (43.46 GiB); RSS at runner return"
    " 34,444,591,104 B (32.08 GiB) (9af56aa8a, all retention)",
}

# Column-count signatures of the invented tables, used to name DataFrames the
# census reached outside a Frame (the manifest's attached views).
GROUP_BY_COLUMNS = {5: "family", 6: "tax_unit"}


def _classify(row, fixture):
    stacked, cloned = fixture["stacked"], fixture["cloned"]
    rows, table, columns = row["rows"], row["table"], row["columns"]
    if table is None and row["kind"] == "dataframe":
        if columns >= 90:
            table = "person"
        elif columns >= 20:
            table = "household"
        elif rows in (stacked["marital_unit"], cloned["marital_unit"]):
            table = "marital_unit"
        else:
            table = GROUP_BY_COLUMNS.get(columns, "household")
    if table is None:
        if row["kind"] in ("weights", "ndarray", "series"):
            # Weight vectors and loose series: household- or person-length.
            if rows in (cloned["person"], stacked["person"]):
                table = "person"
            else:
                table = "household"
        else:
            return None, None
    if rows == cloned.get(table):
        return "cloned", table
    if rows == stacked.get(table):
        return "stacked", table
    return "donor", table


def scale(census_rows, fixture, target):
    by_root = defaultdict(float)
    by_root_grain = defaultdict(float)
    unscaled = defaultdict(int)
    for row in census_rows:
        if row["new_bytes"] == 0:
            continue
        grain, table = _classify(row, fixture)
        if grain is None:
            unscaled[row["root"]] += row["new_bytes"]
            continue
        per_row = row["new_bytes"] / row["rows"]
        real_cloned = target[table]
        if grain == "cloned":
            real = real_cloned
        elif grain == "stacked":
            real = real_cloned / 2
        else:
            real = real_cloned / 2 * ASEC_SHARE_OF_STACKED
        by_root[row["root"]] += per_row * real
        by_root_grain[row["root"], grain] += per_row * real
    return by_root, by_root_grain, unscaled


POINTS = (
    ("prefix_peak_census", "prefix peak: 9-node prefix after its fresh reconstruction"),
    ("post_graph_census", "financial runner just after the 19-node run_graph returns"),
    ("states_census", "financial runner after the independent replay (atomic._states)"),
)


def _print_point(record, key, label):
    fixture = record["fixture_counts"]
    rows = record[key]["rows"]
    scaled = {k: scale(rows, fixture, t) for k, t in TARGETS.items()}
    roots = sorted(
        {r for k in scaled for r in scaled[k][0]},
        key=lambda r: -scaled["full"][0][r],
    )
    print(f"\n--- {label} [{key}] ---")
    print(
        f"{'runner local':22s}"
        + "".join(f"{k:>12s}" for k in TARGETS)
        + "   (GiB, unique physical bytes, first owner wins)"
    )
    for root in roots:
        if scaled["full"][0][root] < 0.005:
            continue
        print(
            f"{root:22s}"
            + "".join(f"{scaled[k][0][root] / GiB:12.2f}" for k in TARGETS)
        )
    totals = {k: sum(scaled[k][0].values()) / GiB for k in TARGETS}
    print(f"{'TOTAL':22s}" + "".join(f"{totals[k]:12.2f}" for k in TARGETS))
    print(
        "  unscaled payload bytes at invented scale (not extrapolated):",
        dict(scaled["full"][2]),
    )
    return totals


def main(path):
    census = json.loads(Path(path).read_text())
    summary = {}
    for profile in ("all", "compact"):
        record = census[profile]
        fixture = record["fixture_counts"]
        print(f"\n=== retention profile {profile!r} ===")
        summary[profile] = {}
        for key, label in POINTS:
            if key in record:
                summary[profile][key] = _print_point(record, key, label)
        returned = record["returned_census"]["rows"]
        ret = {k: scale(returned, fixture, t) for k, t in TARGETS.items()}
        summary[profile]["returned_run"] = {
            k: sum(ret[k][0].values()) / GiB for k in TARGETS
        }
        print(
            f"\n{'returned run object':22s}"
            + "".join(f"{summary[profile]['returned_run'][k]:12.2f}" for k in TARGETS)
        )
        snaps = record["snapshots"]
        print("per-node observer snapshot, full-source GiB:")
        per_node = []
        for snap in snaps:
            total = 0.0
            for name, table in snap["tables"].items():
                rows_fixture = table["rows"]
                grain = (
                    "cloned"
                    if rows_fixture == fixture["cloned"].get(name)
                    else "stacked"
                    if rows_fixture == fixture["stacked"].get(name)
                    else "donor"
                )
                real = CLONED_FULL[name] * (
                    1
                    if grain == "cloned"
                    else 0.5
                    if grain == "stacked"
                    else 0.5 * ASEC_SHARE_OF_STACKED
                )
                total += table["physical_bytes"] / rows_fixture * real
            per_node.append((snap["node"], snap["graph_nodes"], total / GiB))
            print(
                f"   {snap['node']:45s} ({snap['graph_nodes']:2d}-node graph) {total / GiB:7.2f}"
            )
        summary[profile]["snapshots_full_gib"] = per_node
    print("\nmeasured whole-process figures (same 19-node call, 'all' retention):")
    for k, v in MEASURED.items():
        print(f"  {k}: {v}")
    out = Path(path).with_name("model-summary.json")
    out.write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else Path(__file__).with_name("retention-census.json")
    )
