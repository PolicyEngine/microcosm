"""Build an experimental US base H5 with an added prior-year ASEC support block.

This is an experiment, not a release builder. It starts from a complete
Populace base H5, maps a raw prior-year Census CPS ASEC H5 onto the same entity
schema, appends those rows, and normalizes household weights so the resolved
person population stays at the original base scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from policyengine_us.data import USSingleYearDataset

ENTITY_ORDER = ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")
GROUP_ENTITIES = ("household", "tax_unit", "spm_unit", "family", "marital_unit")
WEIGHT_SCALE = 0.01


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-h5", type=Path, required=True)
    parser.add_argument("--prior-asec-h5", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--period", type=int, default=2024)
    parser.add_argument("--base-share", type=float, default=0.5)
    parser.add_argument(
        "--max-prior-households",
        type=int,
        help="Optional smoke limit on prior-year ASEC households.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not (0.0 < args.base_share < 1.0):
        raise ValueError("--base-share must be strictly between 0 and 1.")

    base = USSingleYearDataset(file_path=str(args.base_h5))
    prior = USSingleYearDataset(file_path=str(args.prior_asec_h5))
    base_tables = {entity: getattr(base, entity).copy() for entity in ENTITY_ORDER}
    prior_raw = {entity: getattr(prior, entity).copy() for entity in ENTITY_ORDER[:-1]}

    if args.max_prior_households is not None:
        keep_h = set(
            prior_raw["household"]["H_SEQ"].head(args.max_prior_households).tolist()
        )
        prior_raw["household"] = prior_raw["household"][
            prior_raw["household"]["H_SEQ"].isin(keep_h)
        ].copy()
        prior_raw["person"] = prior_raw["person"][
            prior_raw["person"]["PH_SEQ"].isin(keep_h)
        ].copy()
        keep_tax = set(prior_raw["person"]["TAX_ID"].dropna().astype(int).tolist())
        keep_spm = set(prior_raw["person"]["SPM_ID"].dropna().astype(int).tolist())
        prior_raw["tax_unit"] = prior_raw["tax_unit"][
            prior_raw["tax_unit"]["TAX_ID"].isin(keep_tax)
        ].copy()
        prior_raw["spm_unit"] = prior_raw["spm_unit"][
            prior_raw["spm_unit"]["SPM_ID"].isin(keep_spm)
        ].copy()

    offsets = {
        entity: int(base_tables[entity][f"{entity}_id"].max())
        if f"{entity}_id" in base_tables[entity] and len(base_tables[entity])
        else 0
        for entity in GROUP_ENTITIES
    }
    offsets["person"] = int(base_tables["person"]["person_id"].max())

    prior_tables = _build_prior_tables(base_tables, prior_raw, offsets)
    _normalize_weights(base_tables, prior_tables, base_share=float(args.base_share))

    out_tables = {
        entity: pd.concat(
            [base_tables[entity], prior_tables[entity]], ignore_index=True
        ).reindex(columns=base_tables[entity].columns)
        for entity in ENTITY_ORDER
    }
    _coerce_to_template_dtypes(out_tables, base_tables)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset = USSingleYearDataset(
        person=out_tables["person"],
        household=out_tables["household"],
        tax_unit=out_tables["tax_unit"],
        spm_unit=out_tables["spm_unit"],
        family=out_tables["family"],
        marital_unit=out_tables["marital_unit"],
        time_period=int(args.period),
    )
    dataset.save(str(args.out))

    summary = _summary(base_tables, prior_tables, out_tables, args)
    summary_path = args.summary or args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _build_prior_tables(
    base_tables: dict[str, pd.DataFrame],
    prior_raw: dict[str, pd.DataFrame],
    offsets: dict[str, int],
) -> dict[str, pd.DataFrame]:
    person_raw = prior_raw["person"].copy().reset_index(drop=True)
    household_raw = prior_raw["household"].copy().reset_index(drop=True)
    spm_raw = prior_raw["spm_unit"].copy().reset_index(drop=True)

    household_map = _offset_map(household_raw["H_SEQ"], offsets["household"])
    tax_unit_map = _offset_map(person_raw["TAX_ID"], offsets["tax_unit"])
    spm_unit_map = _offset_map(person_raw["SPM_ID"], offsets["spm_unit"])
    family_keys = pd.Series(
        list(
            zip(
                person_raw["PH_SEQ"].astype(str),
                person_raw["PF_SEQ"].astype(str),
                strict=True,
            )
        ),
        index=person_raw.index,
    )
    family_map = _offset_map(family_keys, offsets["family"])
    marital_map = _offset_map(person_raw["TAX_ID"], offsets["marital_unit"])

    person = _empty_like(base_tables["person"], len(person_raw))
    _copy_common_columns(person, person_raw)
    person["person_id"] = offsets["person"] + 1 + np.arange(len(person_raw))
    person["person_household_id"] = person_raw["PH_SEQ"].map(household_map).astype(int)
    person["person_tax_unit_id"] = person_raw["TAX_ID"].map(tax_unit_map).astype(int)
    person["person_spm_unit_id"] = person_raw["SPM_ID"].map(spm_unit_map).astype(int)
    person["person_family_id"] = family_keys.map(family_map).astype(int)
    person["person_marital_unit_id"] = person_raw["TAX_ID"].map(marital_map).astype(int)
    person["new_tax_unit_id"] = person["person_tax_unit_id"]
    person["_orig_household_id"] = person_raw["PH_SEQ"].to_numpy()
    person["_half"] = "prior_asec"
    person["person_is_puf_clone"] = False
    _map_prior_person_columns(person, person_raw)

    tax_unit = _empty_like(base_tables["tax_unit"], len(tax_unit_map))
    tax_ids = pd.Series(list(tax_unit_map.keys()), dtype=object)
    tax_unit["tax_unit_id"] = tax_ids.map(tax_unit_map).astype(int)
    tax_unit["tax_unit_is_puf_clone"] = False
    tax_unit["filing_status_input"] = (
        _derive_filing_status(person_raw)
        .reindex(tax_ids.astype(int))
        .fillna("SINGLE")
        .to_numpy()
    )
    for flag in ("takes_up_eitc", "would_file_taxes_voluntarily"):
        if flag in tax_unit:
            tax_unit[flag] = True

    person["tax_unit_role_input"] = _derive_tax_unit_roles(
        person_raw,
        tax_unit["tax_unit_id"],
        tax_ids.astype(int).to_numpy(),
        tax_unit_map,
    )

    household = _empty_like(base_tables["household"], len(household_raw))
    household["household_id"] = household_raw["H_SEQ"].map(household_map).astype(int)
    _copy_common_columns(household, household_raw)
    _map_prior_household_columns(household, household_raw)

    spm_unit = _empty_like(base_tables["spm_unit"], len(spm_unit_map))
    spm_ids = pd.Series(list(spm_unit_map.keys()), dtype=object)
    spm_unit["spm_unit_id"] = spm_ids.map(spm_unit_map).astype(int)
    spm_unit["spm_unit_is_puf_clone"] = False
    _map_prior_spm_columns(spm_unit, spm_raw, spm_ids, spm_unit_map)

    family = _empty_like(base_tables["family"], len(family_map))
    family["family_id"] = list(family_map.values())
    family["family_is_puf_clone"] = False

    marital_unit = _empty_like(base_tables["marital_unit"], len(marital_map))
    marital_unit["marital_unit_id"] = list(marital_map.values())

    return {
        "person": person,
        "household": household,
        "tax_unit": tax_unit,
        "spm_unit": spm_unit,
        "family": family,
        "marital_unit": marital_unit,
    }


def _empty_like(template: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    data: dict[str, Any] = {}
    for column in template.columns:
        data[column] = _default_series(template[column], n_rows)
    return pd.DataFrame(data, columns=template.columns)


def _default_series(template: pd.Series, n_rows: int) -> pd.Series:
    if pd.api.types.is_bool_dtype(template):
        return pd.Series(False, index=range(n_rows), dtype=bool)
    if pd.api.types.is_integer_dtype(template):
        return pd.Series(0, index=range(n_rows), dtype=template.dtype)
    if pd.api.types.is_float_dtype(template):
        return pd.Series(0.0, index=range(n_rows), dtype=template.dtype)
    mode = template.dropna().astype(str).mode()
    default = mode.iloc[0] if len(mode) else ""
    return pd.Series(default, index=range(n_rows), dtype=object)


def _copy_common_columns(target: pd.DataFrame, source: pd.DataFrame) -> None:
    for column in sorted(set(target.columns).intersection(source.columns)):
        target[column] = source[column].to_numpy()


def _offset_map(values: pd.Series, offset: int) -> dict[object, int]:
    unique = pd.Series(values).dropna().drop_duplicates().tolist()
    return {value: offset + index + 1 for index, value in enumerate(unique)}


def _map_prior_person_columns(person: pd.DataFrame, raw: pd.DataFrame) -> None:
    num = lambda c: _num(raw, c)  # noqa: E731
    person["age"] = num("A_AGE")
    person["sex"] = num("A_SEX").astype(int)
    person["marital_status"] = num("A_MARITL").astype(int)
    person["family_relationship"] = num("A_EXPRRP").astype(int)
    person["person_number"] = num("P_SEQ").astype(int)
    person["hispanic"] = (num("PRDTHSP") > 0).astype(int)
    person["cps_race"] = num("PRDTRACE").astype(int)
    person["is_female"] = (num("A_SEX") == 2).astype(bool)
    person["is_hispanic"] = (num("PRDTHSP") > 0).astype(bool)
    person["is_household_head"] = (num("P_SEQ") == 1).astype(bool)
    person["is_separated"] = (num("A_MARITL") == 5).astype(bool)
    person["is_surviving_spouse"] = (num("A_MARITL") == 3).astype(bool)
    person["is_blind"] = (num("PEDISEYE") == 1).astype(bool)
    disability_flags = (
        "PEDISDRS",
        "PEDISEAR",
        "PEDISEYE",
        "PEDISOUT",
        "PEDISPHY",
        "PEDISREM",
    )
    person["is_disabled"] = pd.concat(
        [num(column) == 1 for column in disability_flags],
        axis=1,
    ).any(axis=1)
    person["class_of_worker"] = num("PEIO1COW").astype(int)
    person["hours_worked"] = num("HRSWK").where(num("HRSWK") > 0, num("A_HRS1"))
    person["hours_worked_last_week"] = person["hours_worked"]
    person["employment_income"] = num("WSAL_VAL")
    person["employment_income_before_lsr"] = person["employment_income"]
    person["self_employment_income"] = num("SEMP_VAL") + num("FRSE_VAL")
    person["self_employment_income_before_lsr"] = person["self_employment_income"]
    person["taxable_interest_income"] = num("INT_VAL")
    person["dividend_income"] = num("DIV_VAL")
    person["rental_income"] = num("RNT_VAL")
    person["social_security"] = num("SS_VAL")
    person["taxable_pension_income"] = num("PNSN_VAL")
    person["taxable_private_pension_income"] = person["taxable_pension_income"]
    person["unemployment_compensation"] = num("UC_VAL")
    person["taxable_unemployment_compensation"] = person["unemployment_compensation"]
    person["public_assistance"] = num("PAW_VAL")
    person["veterans_benefits"] = num("VET_VAL")
    person["workers_compensation"] = num("WC_VAL")
    person["child_support_received"] = num("CSP_VAL")
    person["child_support_expense"] = num("CHSP_VAL")
    person["disability_benefits"] = num("DIS_VAL1") * (num("DIS_SC1") != 1) + num(
        "DIS_VAL2"
    ) * (num("DIS_SC2") != 1)
    person["has_medicare"] = (num("MCARE") == 1).astype(bool)
    person["has_medicaid"] = (num("CAID") == 1).astype(bool)
    person["has_esi"] = (num("NOW_GRP") == 1).astype(bool)
    person["has_marketplace_health_coverage"] = (num("NOW_MRK") == 1).astype(bool)
    person["has_marketplace_health_coverage_at_interview"] = person[
        "has_marketplace_health_coverage"
    ]
    person["health_insurance_premiums_without_medicare_part_b"] = num("PHIP_VAL")
    person["over_the_counter_health_expenses"] = num("POTC_VAL")
    person["other_medical_expenses"] = num("PMED_VAL")
    person["receives_wic"] = (num("WICYN") == 1).astype(bool)
    person["taxable_401k_distributions"] = _retirement_distribution(raw, 1)
    person["taxable_403b_distributions"] = _retirement_distribution(raw, 2)
    person["taxable_sep_distributions"] = _retirement_distribution(raw, 6)
    person["keogh_distributions"] = _retirement_distribution(raw, 5)
    _map_social_security_splits(person, raw)

    age = num("A_AGE")
    household_key = raw["PH_SEQ"].astype(str)
    me = household_key + ":" + num("A_LINENO").astype(int).astype(str)
    par1 = household_key + ":" + num("PEPAR1").astype(int).astype(str)
    par2 = household_key + ":" + num("PEPAR2").astype(int).astype(str)
    counts = par1.value_counts().add(par2.value_counts(), fill_value=0)
    person["own_children_in_household"] = me.map(counts).fillna(0).astype(float)
    person["count_under_18"] = (age < 18).groupby(raw["PH_SEQ"]).transform("sum")
    person["count_under_6"] = (age < 6).groupby(raw["PH_SEQ"]).transform("sum")
    person["meets_ssi_disability_criteria"] = (
        person["is_disabled"].astype(bool)
        | (num("SSI_VAL") > 0)
        | (person["social_security_disability"] > 0)
        | (person["disability_benefits"] > 0)
    )
    for column in ("takes_up_ssi_if_eligible", "has_valid_ssn"):
        if column in person:
            person[column] = True


def _retirement_distribution(raw: pd.DataFrame, code: int) -> pd.Series:
    total = pd.Series(0.0, index=raw.index)
    for suffix in ("1", "2", "1_YNG", "2_YNG"):
        source = _num(raw, f"DST_SC{suffix}")
        value = _num(raw, f"DST_VAL{suffix}")
        total += (source == code) * value
    return total.astype(float)


def _map_social_security_splits(person: pd.DataFrame, raw: pd.DataFrame) -> None:
    num = lambda c: _num(raw, c)  # noqa: E731
    ss = person["social_security"].astype(float)
    r1 = num("RESNSS1")
    r2 = num("RESNSS2")
    retired = (r1 == 1) | (r2 == 1)
    disabled = ((r1 == 2) | (r2 == 2)) & ~retired
    survivor = (r1.isin([3, 5]) | r2.isin([3, 5])) & ~retired & ~disabled
    dependent = (
        (r1.isin([4, 6, 7]) | r2.isin([4, 6, 7])) & ~retired & ~disabled & ~survivor
    )
    unclassified = (ss > 0) & ~(retired | disabled | survivor | dependent)
    age = num("A_AGE")
    retired = retired | (unclassified & (age >= 62))
    disabled = disabled | (unclassified & (age < 62))
    person["social_security_retirement"] = ss * retired
    person["social_security_disability"] = ss * disabled
    person["social_security_survivors"] = ss * survivor
    person["social_security_dependents"] = ss * dependent


def _map_prior_household_columns(household: pd.DataFrame, raw: pd.DataFrame) -> None:
    num = lambda c: _num(raw, c)  # noqa: E731
    household["state_fips"] = num("GESTFIPS").astype(int)
    household["cbsa"] = num("GTCBSA").astype(int)
    household["GTCO"] = num("GTCO").astype(int)
    household["household_type"] = num("HRHTYPE").astype(int)
    household["household_total_income"] = num("HTOTVAL")
    household["household_income_bracket"] = num("HEFAMINC").astype(int)
    household["household_is_puf_clone"] = False
    household["tenure_type"] = (
        num("H_TENURE")
        .astype(int)
        .map({1: "OWNED_WITH_MORTGAGE", 2: "RENTED", 3: "NONE"})
        .fillna("NONE")
    )
    raw_weight = num("HSUP_WGT") * WEIGHT_SCALE
    household["household_weight"] = raw_weight.astype(float)


def _map_prior_spm_columns(
    spm_unit: pd.DataFrame,
    raw: pd.DataFrame,
    spm_ids: pd.Series,
    spm_unit_map: dict[object, int],
) -> None:
    if raw.empty:
        return
    raw_indexed = raw.drop_duplicates("SPM_ID").set_index("SPM_ID")
    raw_rows = raw_indexed.reindex(spm_ids.astype(int)).reset_index(drop=True)
    num = lambda c: _num(raw_rows, c)  # noqa: E731
    spm_unit["spm_unit_pre_subsidy_childcare_expenses"] = num("SPM_CAPWKCCXPNS")
    spm_unit["spm_unit_energy_subsidy"] = num("SPM_ENGVAL")
    spm_unit["receives_housing_assistance"] = (num("SPM_CAPHOUSESUB") > 0).astype(bool)
    spm_unit["spm_unit_tenure_type"] = (
        num("SPM_TENMORTSTATUS")
        .astype(int)
        .map({1: "OWNER_WITH_MORTGAGE", 2: "OWNER_WITHOUT_MORTGAGE", 3: "RENTER"})
        .fillna("RENTER")
    )
    for column in (
        "takes_up_housing_assistance_if_eligible",
        "takes_up_snap_if_eligible",
        "takes_up_tanf_if_eligible",
    ):
        if column in spm_unit:
            spm_unit[column] = True


def _derive_filing_status(raw: pd.DataFrame) -> pd.Series:
    grouped = raw.groupby("TAX_ID", sort=False)
    head = grouped.head(1).set_index("TAX_ID")
    adult_count = grouped["A_AGE"].apply(
        lambda s: (pd.to_numeric(s, errors="coerce") >= 18).sum()
    )
    child_count = grouped["A_AGE"].apply(
        lambda s: (pd.to_numeric(s, errors="coerce") < 19).sum()
    )
    marital = pd.to_numeric(head["A_MARITL"], errors="coerce").fillna(0).astype(int)
    status = pd.Series("SINGLE", index=head.index, dtype=object)
    status.loc[(marital == 1) & (adult_count.reindex(status.index).fillna(0) >= 2)] = (
        "JOINT"
    )
    status.loc[(marital == 3) & (child_count.reindex(status.index).fillna(0) > 0)] = (
        "SURVIVING_SPOUSE"
    )
    status.loc[marital == 5] = "SEPARATE"
    status.loc[
        (status == "SINGLE") & (child_count.reindex(status.index).fillna(0) > 0)
    ] = "HEAD_OF_HOUSEHOLD"
    return status


def _derive_tax_unit_roles(
    raw: pd.DataFrame,
    _new_tax_unit_ids: pd.Series,
    raw_tax_ids: np.ndarray,
    tax_unit_map: dict[object, int],
) -> pd.Series:
    statuses = _derive_filing_status(raw)
    roles = pd.Series("DEPENDENT", index=raw.index, dtype=object)
    order = raw.assign(_order=np.arange(len(raw))).sort_values(["TAX_ID", "_order"])
    for tax_id, group in order.groupby("TAX_ID", sort=False):
        idx = group.index.to_list()
        if not idx:
            continue
        roles.loc[idx[0]] = "HEAD"
        if statuses.get(tax_id) == "JOINT" and len(idx) > 1:
            adult = group.index[
                pd.to_numeric(group["A_AGE"], errors="coerce").fillna(0) >= 18
            ]
            spouse_candidates = [item for item in adult.to_list() if item != idx[0]]
            if spouse_candidates:
                roles.loc[spouse_candidates[0]] = "SPOUSE"
    return roles


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _normalize_weights(
    base_tables: dict[str, pd.DataFrame],
    prior_tables: dict[str, pd.DataFrame],
    *,
    base_share: float,
) -> None:
    base_population = _resolved_person_population(base_tables)
    prior_population = _resolved_person_population(prior_tables)
    if prior_population <= 0:
        raise ValueError("Prior ASEC support has non-positive resolved population.")
    prior_scale = base_population / prior_population
    base_tables["household"]["household_weight"] = (
        pd.to_numeric(base_tables["household"]["household_weight"], errors="coerce")
        .fillna(0.0)
        .astype(float)
        * base_share
    )
    prior_tables["household"]["household_weight"] = (
        pd.to_numeric(prior_tables["household"]["household_weight"], errors="coerce")
        .fillna(0.0)
        .astype(float)
        * prior_scale
        * (1.0 - base_share)
    )


def _resolved_person_population(tables: dict[str, pd.DataFrame]) -> float:
    weights = tables["household"].set_index("household_id")["household_weight"]
    person_weights = tables["person"]["person_household_id"].map(weights)
    if person_weights.isna().any():
        raise ValueError("Some persons do not map to household weights.")
    return float(person_weights.sum())


def _coerce_to_template_dtypes(
    out_tables: dict[str, pd.DataFrame],
    template_tables: dict[str, pd.DataFrame],
) -> None:
    for entity, table in out_tables.items():
        template = template_tables[entity]
        for column in template.columns:
            if pd.api.types.is_bool_dtype(template[column]):
                table[column] = table[column].fillna(False).astype(bool)
            elif pd.api.types.is_integer_dtype(template[column]):
                table[column] = (
                    pd.to_numeric(table[column], errors="coerce")
                    .fillna(0)
                    .round()
                    .astype(template[column].dtype)
                )
            elif pd.api.types.is_float_dtype(template[column]):
                table[column] = pd.to_numeric(table[column], errors="coerce").fillna(
                    0.0
                )
            else:
                table[column] = table[column].fillna("").astype(str)


def _summary(
    base_tables: dict[str, pd.DataFrame],
    prior_tables: dict[str, pd.DataFrame],
    out_tables: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "base_h5": str(args.base_h5),
        "prior_asec_h5": str(args.prior_asec_h5),
        "output_h5": str(args.out),
        "period": int(args.period),
        "base_share": float(args.base_share),
        "rows": {
            entity: {
                "base": int(len(base_tables[entity])),
                "prior": int(len(prior_tables[entity])),
                "combined": int(len(out_tables[entity])),
            }
            for entity in ENTITY_ORDER
        },
        "resolved_person_population": {
            "base_after_scaling": _resolved_person_population(base_tables),
            "prior_after_scaling": _resolved_person_population(prior_tables),
            "combined": _resolved_person_population(out_tables),
        },
        "household_weight_sum": {
            "base_after_scaling": float(
                base_tables["household"]["household_weight"].sum()
            ),
            "prior_after_scaling": float(
                prior_tables["household"]["household_weight"].sum()
            ),
            "combined": float(out_tables["household"]["household_weight"].sum()),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
