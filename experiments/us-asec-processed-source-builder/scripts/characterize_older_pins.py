"""Explain how a rebuild differs from the pinned income-year 2022 and 2023 files.

    uv run python experiments/us-asec-processed-source-builder/scripts/characterize_older_pins.py \\
        --year 2022 --rebuilt /tmp/census_cps_2022.h5 --pinned <pinned census_cps_2022.h5> \\
        --receipt experiments/us-asec-processed-source-builder/receipts/characterize_2022.json

The pinned 2022 and 2023 files predate loader commit ``40314a75`` (#824): they
were written with an older person column list and Census's own tax units. A
rebuild follows ``40314a75`` and so differs by design. This script checks, and
records, that the difference is exactly that:

* ``family``, ``household`` and ``spm_unit`` are equal (the tool's comparison);
* every person column the pinned file carries is equal, except ``TAX_ID``;
* the rebuild's ``CENSUS_TAX_ID`` equals the pinned ``TAX_ID``;
* the pinned ``person`` columns keep their order in the rebuild, and the
  rebuild's extra columns are listed;
* the pinned ``tax_unit`` table equals the pre-#824 loader's rule, applied to
  the pinned person table: ``person[TAX_UNIT_COLUMNS].groupby(TAX_ID).sum()``
  with the ``TAX_ID`` index copied into a column.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from microcosm.build.us_runtime.asec_processed_source import (
    TAX_UNIT_COLUMNS,
    compare_asec_processed_stores,
    read_asec_processed_store,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit_members(tax_id: pd.Series) -> pd.Series:
    """Each row's tax-unit members, as a frozenset of row positions."""

    positions = pd.Series(range(len(tax_id)), index=tax_id.index)
    members = positions.groupby(tax_id.to_numpy()).agg(frozenset)
    return pd.Series(members.loc[tax_id.to_numpy()].to_numpy(), index=tax_id.index)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--rebuilt", type=Path, required=True)
    parser.add_argument("--pinned", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)

    rebuilt = read_asec_processed_store(args.rebuilt)
    pinned = read_asec_processed_store(args.pinned)
    comparison = compare_asec_processed_stores(args.rebuilt, args.pinned)

    new, old = rebuilt["person"], pinned["person"]
    shared = [column for column in old.columns if column != "TAX_ID"]
    unequal_shared = [
        column for column in shared if not new[column].equals(old[column])
    ]
    legacy_tax_unit = old[list(TAX_UNIT_COLUMNS)].groupby(old["TAX_ID"]).sum()
    legacy_tax_unit["TAX_ID"] = legacy_tax_unit.index
    try:
        pd.testing.assert_frame_equal(
            legacy_tax_unit, pinned["tax_unit"], check_exact=True
        )
        legacy_rule_holds = True
    except AssertionError:
        legacy_rule_holds = False

    receipt = {
        "income_year": args.year,
        "pinned_sha256": _sha256(args.pinned),
        "rebuilt_sha256": _sha256(args.rebuilt),
        "tables_equal": {
            key: table["equal"] for key, table in comparison["tables"].items()
        },
        "person": {
            "rows": [int(len(new)), int(len(old))],
            "pinned_columns": int(len(old.columns)),
            "rebuilt_columns": int(len(new.columns)),
            "pinned_columns_other_than_tax_id_equal": not unequal_shared,
            "unequal_pinned_columns_other_than_tax_id": unequal_shared,
            "census_tax_id_equals_pinned_tax_id": bool(
                new["CENSUS_TAX_ID"].equals(old["TAX_ID"])
            ),
            "pinned_column_order_kept": [
                c for c in new.columns if c in set(old.columns)
            ]
            == list(old.columns),
            "rebuilt_only_columns": [
                c for c in new.columns if c not in set(old.columns)
            ],
            # Constructed units against Census's: people whose unit members
            # differ.
            "persons_whose_tax_unit_members_differ": int(
                (_unit_members(new["TAX_ID"]) != _unit_members(old["TAX_ID"])).sum()
            ),
            "tax_units": [int(new["TAX_ID"].nunique()), int(old["TAX_ID"].nunique())],
        },
        "tax_unit": {
            "pinned_shape": list(pinned["tax_unit"].shape),
            "rebuilt_shape": list(rebuilt["tax_unit"].shape),
            "pinned_equals_pre_824_rule_on_pinned_person": legacy_rule_holds,
        },
    }
    receipt["explained"] = bool(
        receipt["tables_equal"].get("family")
        and receipt["tables_equal"].get("household")
        and receipt["tables_equal"].get("spm_unit")
        and receipt["person"]["pinned_columns_other_than_tax_id_equal"]
        and receipt["person"]["census_tax_id_equals_pinned_tax_id"]
        and receipt["person"]["pinned_column_order_kept"]
        and legacy_rule_holds
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=1))
    return 0 if receipt["explained"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
