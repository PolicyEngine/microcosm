"""Create a new, entirely invented 40-SSUID pipe-delimited fixture and binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for source_root in sorted((REPO / "packages").glob("microcosm-*/src")):
    sys.path.insert(0, str(source_root))
from microcosm.build import sipp_annual_panel as panel  # noqa: E402


def person_rows(ssuid="00000000000001", pnum=101, amount="10", weight="2"):
    rows = []
    for month in range(1, 13):
        row = dict.fromkeys(panel.FIELDS, "")
        row.update(
            SSUID=ssuid,
            PNUM=str(pnum),
            MONTHCODE=str(month),
            THHLDSTATUS="1",
            RIN_UNIV="1",
            ERESIDENCEID="100001",
            ARESIDENCEID="1",
            TAGE_EHC="30",
            WPFINWGT=weight,
            RSSI_MNYN="2",
            ASSI_MNYN="1",
            TPTOTINC=amount,
            APTOTINC="1",
            THTOTINC=amount,
            AHTOTINC="1",
            RHNUMPER="1",
            AHNUMPER="1",
            RHNUMPERWT2="1",
            AHNUMPERWT2="1",
            TLMPAMT="120",
            ALMPAMT="1",
            TROLLAMT="50",
            AROLLAMT="1",
            TMOVER="" if month == 1 else "1",
            AMOVER="0" if month == 1 else "1",
            RREL_PNUM1=str(pnum),
            RREL1="99",
        )
        rows.append(row)
    return rows


def write_rows(path, rows):
    columns = panel.FIELDS
    with path.open("x") as stream:
        stream.write("|".join(columns) + "\n")
        for row in rows:
            stream.write("|".join(row[f] for f in columns) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir()  # Exclusive; an existing output is never reused.
    source = args.output_dir.resolve() / "invented.psv"
    rows = [r for i in reversed(range(1, 41)) for r in person_rows(ssuid=f"{i:014d}")]
    write_rows(source, rows)
    raw = source.read_bytes()  # Only the invented bytes just created.
    binding = {
        "schema": "sipp-annual-panel-binding/1",
        "source_kind": "invented",
        "source_path": str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_size": len(raw),
        "survey_year": 2023,
        "income_year": 2022,
        "selection": panel.SELECTION,
        "contract_sha256": panel.CONTRACT_SHA256,
    }
    with (args.output_dir / "binding.json").open("x") as stream:
        json.dump(binding, stream, indent=2)
        stream.write("\n")
    print(
        "Invented fixture created: 40 SSUIDs, 480 person-month rows; pilot selects first32."
    )


if __name__ == "__main__":
    main()
