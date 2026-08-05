"""Preflight UK H5 artifacts against populace Frame linkage invariants (#612).

The #612 carrier swap makes ``populace.frame.Frame`` validate the UK national
tables at every stage boundary. Frame enforces invariants the UK path never
checked — group ids sorted ascending (Frame raises, it does not reorder), and
membership equality in *both* directions (a benunit or household with no
member persons fails, not just a dangling reference) — so this tool answers,
before any migration code runs, whether the real artifacts already satisfy
them. Run it against the raw single-year input, the latest staging output,
and the certified candidate.

Disclosure control: the output is written to be publishable on the tracking
issue under the UK Data Service End User Licence (CD137 v16.00 clause 8 /
CD171 §5.2.1). It reports classifications, booleans, aggregates, and
threshold-guarded counts — never unit-record values. Frame's own exception
messages embed real ids, so construction failures are reported as a boolean
plus the pre-computed violation classes, and exception text is never echoed.

Read-only by construction: tables are read with ``pandas.HDFStore`` in mode
``"r"`` and root attributes with ``h5py`` in mode ``"r"``. Never open these
artifacts through ``UKSingleYearDataset(file_path=...)`` — that path opens
mode ``"a"`` (a write open) and would invalidate the certified-candidate
file fingerprint on licensed data.

Example::

    uv run python tools/preflight_uk_frame_linkage.py \
        /path/to/populace_uk_2023.h5 \
        /path/to/staging_uk_2023.h5 \
        --json-out preflight_uk_frame_linkage.json

Exit code: 1 if any artifact fails Frame construction, 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "packages" / "populace-frame" / "src")
)
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "packages" / "populace-build" / "src")
)

from populace.build.uk_runtime.rowwise_dataset import (  # noqa: E402
    read_uk_single_year_weight_metadata,
)
from populace.frame import EntitySchema, Frame, Weights  # noqa: E402

GROUP_ENTITIES = ("benunit", "household")
UK_SCHEMA = EntitySchema(group_entities=GROUP_ENTITIES)
_TABLES = ("person", *GROUP_ENTITIES)


def sdc_count(count: int, *, minimum: int) -> int | str:
    """Mask small nonzero counts: zero and counts >= minimum are safe."""

    count = int(count)
    if count == 0 or count >= minimum:
        return count
    return f"< {minimum}"


def _column_missing(table: pd.DataFrame, column: str) -> bool:
    return column not in table.columns


def classify_group_linkage(
    person: pd.DataFrame,
    group_table: pd.DataFrame,
    group: str,
    *,
    minimum: int,
) -> dict[str, Any]:
    """Classify one group entity against Frame's linkage invariants."""

    id_column = f"{group}_id"
    membership_column = f"person_{group}_id"
    missing = [
        column
        for table, column in ((group_table, id_column), (person, membership_column))
        if _column_missing(table, column)
    ]
    if missing:
        return {"missing_columns": missing}

    ids = group_table[id_column]
    memberships = person[membership_column].dropna()
    id_set = set(ids.dropna())
    membership_set = set(memberships)
    non_na = ids.dropna()
    return {
        "missing_columns": [],
        "id_na": sdc_count(ids.isna().sum(), minimum=minimum),
        "id_duplicated": sdc_count(non_na.duplicated().sum(), minimum=minimum),
        "ids_sorted_ascending": bool(non_na.is_monotonic_increasing),
        "orphaned_group_rows": sdc_count(
            len(id_set - membership_set), minimum=minimum
        ),
        "dangling_memberships": sdc_count(
            len(membership_set - id_set), minimum=minimum
        ),
    }


def classify_linkage(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    *,
    minimum: int,
) -> dict[str, Any]:
    """Classify all Frame linkage invariants over the three UK tables."""

    report: dict[str, Any] = {
        group: classify_group_linkage(
            person,
            {"benunit": benunit, "household": household}[group],
            group,
            minimum=minimum,
        )
        for group in GROUP_ENTITIES
    }
    if _column_missing(person, "person_id"):
        report["person"] = {"missing_columns": ["person_id"]}
    else:
        person_ids = person["person_id"]
        report["person"] = {
            "missing_columns": [],
            "id_na": sdc_count(person_ids.isna().sum(), minimum=minimum),
            "id_duplicated": sdc_count(
                person_ids.dropna().duplicated().sum(), minimum=minimum
            ),
        }
    if not _column_missing(person, "person_benunit_id") and not _column_missing(
        person, "person_household_id"
    ):
        households_per_benunit = person.groupby("person_benunit_id")[
            "person_household_id"
        ].nunique()
        report["split_benunits"] = sdc_count(
            int((households_per_benunit > 1).sum()), minimum=minimum
        )
    return report


def classify_household_weights(
    household: pd.DataFrame, *, minimum: int
) -> dict[str, Any]:
    """Classify the weight vector against Frame's weight invariants."""

    if _column_missing(household, "household_weight"):
        return {"missing_columns": ["household_weight"]}
    weights = household["household_weight"]
    values = weights.to_numpy(dtype="float64", na_value=np.nan)
    return {
        "missing_columns": [],
        "dtype": str(weights.dtype),
        "non_finite": sdc_count(int((~np.isfinite(values)).sum()), minimum=minimum),
        "negative": sdc_count(int((values < 0).sum()), minimum=minimum),
        "all_zero": bool(np.nansum(np.abs(values)) == 0.0),
        # A population aggregate: sums every carrier, publishable.
        "total_mass": float(np.nansum(values)),
    }


def column_collisions(
    person: pd.DataFrame, benunit: pd.DataFrame, household: pd.DataFrame
) -> list[str]:
    """Column names appearing on more than one table (schema, not records)."""

    tables = (person, benunit, household)
    seen: dict[str, int] = {}
    for table in tables:
        for column in table.columns:
            seen[column] = seen.get(column, 0) + 1
    return sorted(name for name, uses in seen.items() if uses > 1)


def payload_probe(table: pd.DataFrame) -> dict[str, Any]:
    """Shape and dtype histogram — the payload facts a rewrite must preserve."""

    dtypes: dict[str, int] = {}
    for dtype in table.dtypes:
        dtypes[str(dtype)] = dtypes.get(str(dtype), 0) + 1
    return {
        "rows": int(len(table)),
        "columns": int(table.shape[1]),
        "dtypes": dict(sorted(dtypes.items())),
        "index": type(table.index).__name__,
        "is_range_index": bool(isinstance(table.index, pd.RangeIndex)),
    }


def attempt_frame_construction(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    *,
    weight_kind: Any,
) -> bool:
    """True iff Frame construction succeeds. Exception text is never surfaced:
    Frame's messages embed real ids, and the classification above already
    names every violation class this could fail on."""

    try:
        weights = Weights(
            values=household["household_weight"].to_numpy(dtype="float64"),
            kind=weight_kind,
        )
        Frame(
            tables={"person": person, "benunit": benunit, "household": household},
            schema=UK_SCHEMA,
            weights={"household": weights},
        )
    except Exception:
        return False
    return True


def preflight_artifact(path: Path, *, minimum: int) -> dict[str, Any]:
    """Run every check against one UK single-year H5 artifact."""

    weight_kind, _mass_log = read_uk_single_year_weight_metadata(path)
    with pd.HDFStore(path, mode="r") as store:
        keys = {key.lstrip("/") for key in store.keys()}
        missing = sorted(set(_TABLES) - keys)
        if missing:
            raise ValueError(
                f"UK artifact {path.name} is missing table(s): {missing}."
            )
        person = store["person"]
        benunit = store["benunit"]
        household = store["household"]
        time_period = (
            str(store["time_period"].iloc[0]) if "time_period" in keys else None
        )
    return {
        "path": str(path),
        "time_period": time_period,
        "household_weight_kind": weight_kind.value,
        "tables": {
            "person": payload_probe(person),
            "benunit": payload_probe(benunit),
            "household": payload_probe(household),
        },
        "linkage": classify_linkage(person, benunit, household, minimum=minimum),
        "household_weights": classify_household_weights(household, minimum=minimum),
        "column_collisions": column_collisions(person, benunit, household),
        "frame_constructed": attempt_frame_construction(
            person, benunit, household, weight_kind=weight_kind
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify UK H5 artifacts against populace Frame linkage "
            "invariants (#612), with SDC-safe output."
        )
    )
    parser.add_argument(
        "h5",
        nargs="+",
        type=Path,
        help="UK single-year H5 artifacts (opened read-only).",
    )
    parser.add_argument(
        "--sdc-minimum-count",
        type=int,
        default=10,
        help=(
            "Counts below this (other than zero) are reported as a "
            "threshold, never exactly (default: 10)."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Also write the report JSON to this path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = {
        "sdc_minimum_count": args.sdc_minimum_count,
        "artifacts": [
            preflight_artifact(path, minimum=args.sdc_minimum_count)
            for path in args.h5
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.write_text(rendered + "\n")
    failed = [
        artifact["path"]
        for artifact in report["artifacts"]
        if not artifact["frame_constructed"]
    ]
    if failed:
        print(
            f"FAIL: {len(failed)} artifact(s) do not construct a Frame.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
