#!/usr/bin/env python3
"""Validate the ACS arm of the SPM independence role, with aggregate counts only.

The ACS spine has no Census SPM fields and no SPM partition: every ACS household
is one SPM unit. ``with_acs_spm_independence_role`` therefore reads the ASEC
rule (``SPM_HEAD == 1 OR (A_FAMTYP in {1,4} AND A_FAMREL in {1,2})``) off
``RELSHIPP``: the reference person (the unit's head), the spouse of the
reference person (the primary family's spouse), or a group-quarters record's
sole person. This script measures two things:

1. ``asec``: on every pinned Census ASEC person file, in households that are one
   SPM unit, whether "reference person or spouse of the reference person"
   (``A_EXPRRP`` 1-4, the CPS encoding of the ACS rule) equals the ASEC rule
   and reproduces Census's own ``SPM_NUMADULTS`` / ``SPM_NUMKIDS``.
2. ``acs``: the receipt of the real ``with_acs_spm_independence_role`` run on
   the full pinned ACS 2024 1-year PUMS, loaded by the real PUMS loader, on the
   one-SPM-unit-per-household partition the staging build uses.

Every input is verified against its pin before it is read. The receipt holds
aggregate counts and shares only, never a record.

    .venv/bin/python experiments/acs_spm_role_asec_validation.py \\
        --acs-inputs-dir PATH/inputs/acs_2024_1yr \\
        --out experiments/acs-spm-role-asec-validation-receipt.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.us_runtime import acs_pums, acs_sources
from microcosm.build.us_runtime.acs_inputs import with_acs_spm_independence_role
from microcosm.build.us_runtime.education_assistance_source import (
    fetch_asec_education_assistance_source,
)
from microcosm.build.us_runtime.spm_composition import check_spm_composition
from microcosm.build.us_runtime.spm_role_source import (
    ASEC_SPM_ROLE_SOURCES,
    SPM_ROLE_RULE,
    independent_minor_role,
)
from microcosm.frame import US_SCHEMA, Frame

_ASEC_COLUMNS = (
    "PH_SEQ",
    "SPM_ID",
    "SPM_HEAD",
    "A_FAMTYP",
    "A_FAMREL",
    "A_EXPRRP",
    "A_AGE",
    "SPM_NUMADULTS",
    "SPM_NUMKIDS",
    "MARSUPWT",
)
#: CPS ``A_EXPRRP``: 1/2 reference person (with/without relatives), 3/4 spouse.
_REFERENCE = (1, 2)
_REFERENCE_OR_SPOUSE = (1, 2, 3, 4)


def _share(weights: pd.Series, mask: pd.Series, among: pd.Series) -> float:
    denominator = float(weights[among].sum())
    return float(weights[mask & among].sum()) / denominator if denominator else 0.0


def _adult_counts_match(
    frame: pd.DataFrame, role: pd.Series, units: pd.Series
) -> dict[str, int]:
    adult = (frame["A_AGE"] >= 18) | ((frame["A_AGE"] >= 15) & role)
    grouped = pd.DataFrame(
        {
            "unit": units,
            "adult": adult,
            "expected_adults": frame["SPM_NUMADULTS"],
            "expected_children": frame["SPM_NUMKIDS"],
        }
    ).groupby("unit", sort=False)
    unit = grouped.agg(
        adults=("adult", "sum"),
        persons=("adult", "size"),
        expected_adults=("expected_adults", "first"),
        expected_children=("expected_children", "first"),
    )
    matched = unit["adults"].eq(unit["expected_adults"]) & (
        unit["persons"] - unit["adults"]
    ).eq(unit["expected_children"])
    return {"units": int(len(unit)), "units_matching_census_counts": int(matched.sum())}


def _crosstab(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, int]]:
    if frame.empty:
        return []
    counts = frame.groupby(columns, sort=True).size()
    return [
        {**dict(zip(columns, (int(v) for v in key), strict=True)), "persons": int(n)}
        for key, n in counts.items()
    ]


def measure_asec(income_year: int) -> dict[str, Any]:
    """Compare the ACS rule's CPS analog with the ASEC rule on one vintage."""

    path = fetch_asec_education_assistance_source(income_year)
    person = pd.read_csv(path, usecols=list(_ASEC_COLUMNS))
    rule = independent_minor_role(person).astype(bool)
    analog = person["A_EXPRRP"].isin(_REFERENCE_OR_SPOUSE)
    reference = person["A_EXPRRP"].isin(_REFERENCE)
    teen = (person["A_AGE"] >= 15) & (person["A_AGE"] < 18)
    weights = person["MARSUPWT"].astype(float)
    units_per_household = person.groupby("PH_SEQ")["SPM_ID"].transform("nunique")
    one_unit = units_per_household.eq(1)
    everyone = pd.Series(True, index=person.index)

    # Premise: in a one-SPM-unit household the SPM head is the reference person.
    head = person["SPM_HEAD"].eq(1)
    one_unit_households = person.loc[one_unit, "PH_SEQ"].nunique()
    head_is_reference = (
        person.loc[one_unit & head & reference, "PH_SEQ"].nunique()
        if one_unit.any()
        else 0
    )

    restricted = person[one_unit]
    disagree = restricted[rule[one_unit] != analog[one_unit]].assign(
        rule=rule[one_unit].astype(int), analog=analog[one_unit].astype(int)
    )
    hidden_teens = person[teen & rule & ~analog].assign(
        multi_unit_household=(~one_unit[teen & rule & ~analog]).astype(int)
    )
    return {
        "income_year": income_year,
        "source": {
            "csv_sha256": ASEC_SPM_ROLE_SOURCES[income_year].csv_sha256,
            "archive_sha256": ASEC_SPM_ROLE_SOURCES[income_year].archive_sha256,
            "official_archive_url": ASEC_SPM_ROLE_SOURCES[
                income_year
            ].official_archive_url,
        },
        "persons": int(len(person)),
        "households": int(person["PH_SEQ"].nunique()),
        "one_unit_households": int(one_unit_households),
        "one_unit_persons": int(one_unit.sum()),
        "one_unit_households_whose_spm_head_is_the_reference_person": int(
            head_is_reference
        ),
        "one_unit_agreement": {
            "persons_agree": int((rule == analog)[one_unit].sum()),
            "weighted_agreement": _share(weights, rule == analog, one_unit),
            "teens_aged_15_to_17": int((one_unit & teen).sum()),
            "teens_agree": int(((rule == analog) & one_unit & teen).sum()),
        },
        "one_unit_disagreements": _crosstab(
            disagree, ["A_EXPRRP", "A_FAMTYP", "A_FAMREL", "rule", "analog"]
        ),
        "analog_true_where_rule_false_all_households": int((analog & ~rule).sum()),
        "census_adult_counts_one_unit": {
            "asec_rule": _adult_counts_match(
                restricted, rule[one_unit], restricted["SPM_ID"]
            ),
            "acs_analog": _adult_counts_match(
                restricted, analog[one_unit], restricted["SPM_ID"]
            ),
        },
        "teens_holding_the_rule_but_not_reference_or_spouse": {
            "persons": int(len(hidden_teens)),
            "by_relationship": _crosstab(
                hidden_teens, ["A_EXPRRP", "A_FAMTYP", "multi_unit_household"]
            ),
        },
        "weighted_role_shares": {
            "rule_all": _share(weights, rule, everyone),
            "analog_all": _share(weights, analog, everyone),
            "rule_one_unit": _share(weights, rule, one_unit),
            "analog_one_unit": _share(weights, analog, one_unit),
            "rule_teens_all": _share(weights, rule, teen),
            "analog_teens_all": _share(weights, analog, teen),
            "rule_teens_one_unit": _share(weights, rule, teen & one_unit),
            "analog_teens_one_unit": _share(weights, analog, teen & one_unit),
        },
    }


def measure_acs(inputs_dir: Path) -> dict[str, Any]:
    """Run the real derivation on the full pinned ACS 2024 1-year spine."""

    manifest = acs_sources.load_acs_source_manifest()
    source = acs_sources.fetch_acs_pums_sources(inputs_dir, manifest=manifest)
    tables, loader = acs_pums.load_acs_pums_tables(source)
    household, person = tables["household"], tables["person"]
    household_ids = pd.Series(
        np.arange(1, len(household) + 1, dtype=np.int64), index=household["SERIALNO"]
    )
    person_household = person["SERIALNO"].map(household_ids).to_numpy(np.int64)
    ids = np.arange(1, len(person) + 1, dtype=np.int64)
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": ids,
                    "person_household_id": person_household,
                    "person_tax_unit_id": ids,
                    # assign_us_unit_structure's partition without SPM_ID.
                    "person_spm_unit_id": person_household,
                    "person_family_id": person_household,
                    "person_marital_unit_id": ids,
                    "RELSHIPP": pd.to_numeric(person["RELSHIPP"]),
                    "age": pd.to_numeric(person["AGEP"]).astype(float),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": household_ids.to_numpy(),
                    "TYPEHUGQ": pd.to_numeric(household["TYPEHUGQ"]).to_numpy(),
                }
            ),
            "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
            "spm_unit": pd.DataFrame({"spm_unit_id": household_ids.to_numpy()}),
            "family": pd.DataFrame({"family_id": household_ids.to_numpy()}),
            "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
        },
        US_SCHEMA,
        {"household": acs_pums._household_weights(household, person)},
    )
    result = with_acs_spm_independence_role(frame)
    composition = check_spm_composition(result.frame)
    return {
        "source": {
            artifact.role: {"sha256": artifact.sha256, "url": artifact.url}
            for artifact in manifest.artifacts
        },
        "loader": {
            key: loader[key]
            for key in (
                "household_rows",
                "person_rows",
                "vacant_household_rows_dropped",
            )
        },
        "receipt": result.provenance,
        "composition_check": {
            "status": composition.status,
            "n_units": int(composition.details["n_units"]),
            "n_units_without_classified_adult": int(
                composition.details["n_units_without_classified_adult"]
            ),
        },
    }


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acs-inputs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = {
        "artifact_kind": "acs_spm_independence_role_validation",
        "code_commit": _git_head(),
        "asec_rule": SPM_ROLE_RULE,
        "acs_analog_in_cps_codes": "A_EXPRRP in {1, 2, 3, 4}",
        "asec": [measure_asec(year) for year in sorted(ASEC_SPM_ROLE_SOURCES)],
        "acs": measure_acs(args.acs_inputs_dir),
    }
    args.out.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
