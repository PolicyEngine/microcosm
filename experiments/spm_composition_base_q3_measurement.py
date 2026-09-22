#!/usr/bin/env python3
"""Measure SPM measurement composition on a US base pool H5 (read-only).

The receipt behind ``docs/us-spm-role-for-a-fresh-base.md``. It answers, for one
base, the questions a fresh lineage has to settle before the SPM independence
role can be delivered:

1. How many SPM units have no member aged 18 or over, and how many still have no
   classified adult once the engine's own role resolution is applied?
2. Are those units resolvable at all — does each contain a 15-to-17-year-old?
3. Could the raw ASEC columns the base carries supply ``SPM_ROLE_RULE`` directly?
4. Does every person carry an ASEC origin, so the role can be joined without
   synthesizing a default (which ``policyengine_us/spm.py`` forbids)?
5. Does the base meet every structural precondition of
   ``microcosm.build.us_runtime.spm_role_source.derive_spm_role_source``?

Read-only: it opens the H5 for reading and writes nothing. Run:

    uv run python experiments/spm_composition_base_q3_measurement.py \
        ~/PolicyEngine/_buildq-runtime/out/base-q3/base_populace_us_2024_puf_support.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tables

#: The parent columns ``derive_spm_role_source`` requires, read from its own
#: ``required`` tuple and ``_REQUIRED_RAW_CHECKS`` rather than re-invented.
DERIVE_REQUIRED_COLUMNS = (
    "person_id",
    "person_spm_unit_id",
    "source_year",
    "PERIDNUM",
    "age",
    "source_household_id",
    "source_person_id",
    "source_row_id",
    "A_AGE",
    "A_LINENO",
    "P_SEQ",
    "SPM_HAGE",
    "SPM_NUMADULTS",
    "SPM_NUMKIDS",
    "SPM_NUMPER",
)

#: The columns this measurement reads. Kept narrow so a 2.3 GB pool is a
#: seconds-long read rather than a full-frame load.
_READ_COLUMNS = (
    "person_spm_unit_id",
    "age",
    "is_household_head",
    "person_support_channel",
    "source_person_id",
    "source_year",
    "A_FAMTYP",
    "A_FAMREL",
)


def _sum_by_unit(
    flags: np.ndarray, membership: np.ndarray, unit_ids: np.ndarray
) -> np.ndarray:
    return (
        pd.Series(np.asarray(flags).astype(np.int64), index=membership)
        .groupby(level=0)
        .sum()
        .reindex(unit_ids, fill_value=0)
        .to_numpy(dtype=np.int64)
    )


def _decode(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: value.decode() if isinstance(value, bytes) else value
    )


def measure(path: Path) -> dict:
    with tables.open_file(str(path), "r") as handle:
        person_table = handle.get_node("/person/table")
        columns = list(person_table.colnames)
        person = pd.DataFrame(
            {name: person_table.col(name) for name in _READ_COLUMNS if name in columns}
        )
        unit_ids = handle.get_node("/spm_unit/table").col("spm_unit_id")
        household_ids = handle.get_node("/household/table").col("household_id")
        person_ids = person_table.col("person_id")

    membership = person["person_spm_unit_id"].to_numpy()
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(dtype=np.float64)
    head = (
        person["is_household_head"].fillna(False).to_numpy(dtype=bool)
        if "is_household_head" in person
        else np.zeros(len(person), dtype=bool)
    )
    # is_household_spouse has no producer in microcosm, so the engine's fallback
    # formula reduces to the head column wherever that column is exported.
    role_fallback = head

    no_adult_18 = _sum_by_unit(age >= 18.0, membership, unit_ids) < 1
    unclassified = (
        _sum_by_unit(
            (age >= 18.0) | ((age >= 15.0) & role_fallback), membership, unit_ids
        )
        < 1
    )

    offending = unit_ids[no_adult_18]
    member_of_offending = np.isin(membership, offending)
    teen = member_of_offending & (age >= 15.0) & (age < 18.0)

    # SPM_ROLE_RULE's second leg; its SPM_HEAD leg needs the pinned Census CSV.
    leg = np.zeros(len(person), dtype=bool)
    if {"A_FAMTYP", "A_FAMREL"} <= set(person.columns):
        leg = np.isin(person["A_FAMTYP"].to_numpy(), [1, 4]) & np.isin(
            person["A_FAMREL"].to_numpy(), [1, 2]
        )

    origin = _decode(person["source_person_id"].astype(object)).astype(str)
    has_origin = origin.str.fullmatch(r"\d{22}").to_numpy()
    unit_sizes = _sum_by_unit(np.ones(len(person)), membership, unit_ids)
    units_without_origin = int(
        (_sum_by_unit(has_origin, membership, unit_ids) < 1).sum()
    )
    within_unit_repeats = int(
        pd.DataFrame({"sid": origin.to_numpy(), "unit": membership})
        .groupby(["unit", "sid"])
        .size()
        .max()
    )

    return {
        "base_h5": str(path),
        "households": int(len(household_ids)),
        "persons": int(len(person_ids)),
        "spm_units": int(len(unit_ids)),
        "defect": {
            "units_without_member_aged_18_or_over": int(no_adult_18.sum()),
            "units_without_classified_adult": int(unclassified.sum()),
            "offending_units_containing_a_15_to_17_year_old": int(
                (_sum_by_unit(teen, membership, unit_ids)[no_adult_18] > 0).sum()
            ),
            "members_aged_15_to_17_in_offending_units": int(teen.sum()),
            "offending_unit_sizes": {
                str(int(size)): int(count)
                for size, count in zip(
                    *np.unique(unit_sizes[no_adult_18], return_counts=True),
                    strict=True,
                )
            },
            "offending_member_channels": (
                _decode(person.loc[member_of_offending, "person_support_channel"])
                .value_counts()
                .to_dict()
                if "person_support_channel" in person
                else {}
            ),
        },
        "fallback": {
            "is_household_head_present": "is_household_head" in columns,
            "is_household_spouse_present": "is_household_spouse" in columns,
            "is_spm_independent_minor_role_present": (
                "is_spm_independent_minor_role" in columns
            ),
        },
        "raw_rule_leg": {
            "SPM_HEAD_present": "SPM_HEAD" in columns,
            "A_FAMTYP_null_share": (
                float(pd.isna(person["A_FAMTYP"]).mean())
                if "A_FAMTYP" in person
                else None
            ),
            "persons_flagged_by_famtyp_famrel_leg": int(leg.sum()),
            "offending_teens_flagged_by_leg": int((teen & leg).sum()),
            "units_unresolved_by_leg_alone": int(
                (
                    _sum_by_unit(
                        (age >= 18.0) | ((age >= 15.0) & leg), membership, unit_ids
                    )
                    < 1
                ).sum()
            ),
        },
        "asec_origin": {
            "persons_with_22_digit_source_person_id": int(has_origin.sum()),
            "spm_units_with_no_asec_origin_member": units_without_origin,
            "distinct_source_persons": int(origin.nunique()),
            "max_repeats_of_one_source_person_within_one_spm_unit": within_unit_repeats,
        },
        "derive_spm_role_source_preconditions": {
            "missing_required_columns": [
                name for name in DERIVE_REQUIRED_COLUMNS if name not in columns
            ],
            "source_years": sorted({int(year) for year in person["source_year"]}),
            "person_id_unique": bool(pd.Series(person_ids).is_unique),
            "spm_unit_id_unique": bool(pd.Series(unit_ids).is_unique),
            "membership_exactly_covers_spm_table": bool(
                set(membership.tolist()) == set(unit_ids.tolist())
            ),
            "no_source_person_repeats_within_a_unit": within_unit_repeats == 1,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_h5", type=Path, help="US base pool H5 (read-only).")
    args = parser.parse_args(argv)
    print(json.dumps(measure(args.base_h5), indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
