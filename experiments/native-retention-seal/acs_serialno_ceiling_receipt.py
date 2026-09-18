"""The ceiling that actually binds the native path, and where it sits.

The native-scale lane's 1/10 run reached its memory gate, ran 573 CPU-s and
stopped with ``PREPARATION_ISSUANCE_REFUSED`` -- a catch-all that discards its
cause. A ``sys.monitoring`` RAISE trace (``harness19_diag.py``, this lane)
names it: ``ACSCoverageAuthenticationError: CANONICAL_SIZE``, raised through
``acs_person_coverage_authentication._json``'s ``charge`` with exactly two
``visit`` frames beneath it -- a FLAT list, not the nested evidence receipt --
called from ``acs_native_coverage_binding.issue_acs_native_coverage``. That is
the guard at ``acs_native_coverage_binding.py:562-564``:

    coverage._json(serialnos, min(MAX_EVIDENCE_BYTES, housing.ACS_HU_RECEIPT_MAX_BYTES))

``MAX_EVIDENCE_BYTES`` is 2 MiB and ``ACS_HU_RECEIPT_MAX_BYTES`` is 1 MiB, so
the bound on the selected-ACS-SERIALNO list is **1 MiB**, and
``survey_population_preparation.py:1959-1961`` always passes
``serialnos=acs_keys``, never ``None``.

This script computes where that binds, from measured inputs only: the ACS
SERIALNO width read off the staged source, and the ACS share of a selection
read off the recovered 1/1000 pilot's own committed preparation receipt.

Nothing here is a build, a certification or a release artifact. It reads two
local files and writes one JSON.
"""

import csv
import io
import json
import pathlib
import sys
import zipfile

ACS_HOUSING_ZIP = pathlib.Path(
    "/Users/maxghenis/PolicyEngine/_recovered/pilot-runs/native45-v5/run"
    "/sources/acs/csv_hus.zip"
)
PILOT_RECEIPT = pathlib.Path(
    "/Users/maxghenis/PolicyEngine/_recovered/pilot-runs/native19-required-20260912"
    "/run/financial-artifacts/preparation.json"
)
CAP = 1024**2  # acs_housing_universe_source.ACS_HU_RECEIPT_MAX_BYTES
SAMPLE_ROWS = 20_000


def serialno_width():
    with zipfile.ZipFile(ACS_HOUSING_ZIP) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        with archive.open(name) as handle:
            reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8"))
            column = next(reader).index("SERIALNO")
            widths = set()
            for index, row in enumerate(reader):
                widths.add(len(row[column]))
                if index >= SAMPLE_ROWS:
                    break
    return sorted(widths)


def main():
    widths = serialno_width()
    receipt = json.loads(PILOT_RECEIPT.read_text())
    households = receipt["origins"]["households"]
    acs = sum(1 for row in households if row["source"] == "acs")
    supplied = receipt["selection"]["supplied_households"]
    # ``["2024GQ0000926",`` -- two quotes and one separator per element.
    per_entry = max(widths) + 3
    admitted = (CAP - 2) // per_entry
    share = acs / len(households)
    limit = admitted / share
    fractions = {}
    for denominator in (1000, 30, 24, 20, 15, 10, 1):
        selected = supplied / denominator
        fractions[f"1/{denominator}"] = {
            "selected_households": round(selected),
            "acs_serialnos": round(selected * share),
            "canonical_json_bytes": round(selected * share * per_entry),
            "refuses": selected * share * per_entry > CAP,
        }
    document = {
        "scope": "diagnostic; not a build, certification or release artifact",
        "release_eligible": False,
        "cap_bytes": CAP,
        "cap_name": "acs_housing_universe_source.ACS_HU_RECEIPT_MAX_BYTES",
        "guard": "acs_native_coverage_binding.py:562-564",
        "serialno_widths_observed": widths,
        "json_bytes_per_serialno": per_entry,
        "serialnos_admitted": admitted,
        "pilot_selection": {
            "selected_households": len(households),
            "acs": acs,
            "asec": len(households) - acs,
            "acs_share": round(share, 6),
            "supplied_households": supplied,
        },
        "binds_above_selected_households": round(limit),
        "binds_above_share_of_source": round(limit / supplied, 6),
        "transport_lane_lifted_ceiling_1": {
            "households": 96860,
            "share_of_source": 0.061,
            "note": "this bound is LOWER, so it refuses first",
        },
        "by_fraction": fractions,
    }
    pathlib.Path(sys.argv[1]).write_text(json.dumps(document, indent=2) + "\n")
    print(f"serialno widths observed: {widths}")
    print(f"json bytes per serialno : {per_entry}")
    print(f"1 MiB admits            : {admitted:,} ACS serialnos")
    print(
        f"ACS share of a selection: {acs}/{len(households)} = {share:.4f} "
        "(recovered 1/1000 pilot)"
    )
    print(
        f"=> refuses above ~{round(limit):,} selected households "
        f"({limit / supplied * 100:.2f}% of source), against the transport lane's "
        "lifted 96,860 (6.10%)"
    )
    for name, row in fractions.items():
        print(
            f"   {name:>7}: {row['selected_households']:>10,} selected, "
            f"{row['acs_serialnos']:>10,} ACS -> {row['canonical_json_bytes']:>12,} B  "
            f"{'REFUSES' if row['refuses'] else 'fits'}"
        )


if __name__ == "__main__":
    main()
