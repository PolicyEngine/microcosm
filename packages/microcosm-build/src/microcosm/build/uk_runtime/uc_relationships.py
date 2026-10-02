"""Relationship inputs shared by UC measurement and diagnostic model replays."""

from __future__ import annotations

import numpy as np
import pandas as pd

UC_COUPLE_DEFINITION = (
    "count(is_benunit_head OR is_parent OR (dependent_children == 0)) "
    "over benefit-unit members == 2"
)


def frs_uc_claimant_mask(person: pd.DataFrame, benunit: pd.DataFrame) -> np.ndarray:
    """Return FRS claimant/partner roles in person-row order.

    The FRS adult-file members of a unit with dependent children are retained
    as ``is_parent``; the head is retained separately. A childless FRS benefit
    unit consists of its claimant or couple. This source role does not assert
    eligibility, receipt of a couple standard allowance, or legal marriage.
    Dependent child-file members do not become partners on their 18th birthday.
    Membership is never changed here.

    Persist this mask as the Boolean ``is_uc_claimant`` input. It is
    authoritative over PolicyEngine-UK's calculator fallback: a 17-year-old
    partner or an older nonqualifying dependent cannot reliably be inferred
    from age and education alone. Reporter payment landing uses a separate
    established convention; this mask identifies every claimant or partner.
    """
    required_person = {"person_benunit_id", "is_benunit_head", "is_parent"}
    required_benunit = {"benunit_id", "dependent_children"}
    for table, required, label in (
        (person, required_person, "person"),
        (benunit, required_benunit, "benunit"),
    ):
        missing = required - set(table.columns)
        if missing:
            raise KeyError(f"UC relationship {label} inputs missing: {sorted(missing)}")
    ids = benunit["benunit_id"]
    members = person["person_benunit_id"]
    if ids.isna().any() or ids.duplicated().any() or members.isna().any():
        raise ValueError("UC relationship membership IDs must be present and unique.")
    if set(ids) != set(members):
        raise ValueError("UC relationship membership must cover every benefit unit.")
    for name in ("is_benunit_head", "is_parent"):
        values = person[name]
        if values.isna().any() or not values.isin([0, 1, False, True]).all():
            raise ValueError(f"{name} must contain complete boolean source roles.")
    dependent = pd.to_numeric(benunit["dependent_children"], errors="coerce")
    if (
        dependent.isna().any()
        or not np.isfinite(dependent).all()
        or (dependent < 0).any()
        or not np.equal(dependent, np.floor(dependent)).all()
    ):
        raise ValueError("dependent_children must contain nonnegative integer counts.")
    child_count = pd.Series(dependent.to_numpy(), index=ids).reindex(members).to_numpy()
    claimants = (
        person["is_benunit_head"].to_numpy(dtype=bool)
        | person["is_parent"].to_numpy(dtype=bool)
        | (child_count == 0)
    )
    counts = pd.Series(claimants).groupby(members.to_numpy()).sum().reindex(ids)
    if not counts.isin([1, 2]).all():
        raise ValueError("FRS UC roles require one or two claimants per benefit unit.")
    return claimants


def frs_uc_couple_mask(person: pd.DataFrame, benunit: pd.DataFrame) -> np.ndarray:
    """Return claimant/partner couple status in benefit-unit row order.

    UC treats cohabiting claimants as a couple regardless of legal marriage.
    Source roles, rather than an adult-age count, also keep an older dependent
    from becoming the lone claimant's partner.
    """
    claimants = frs_uc_claimant_mask(person, benunit)
    counts = (
        pd.Series(claimants)
        .groupby(person["person_benunit_id"].to_numpy(), sort=False)
        .sum()
        .reindex(benunit["benunit_id"])
    )
    return counts.eq(2).to_numpy(dtype=bool)


#: Benefit-unit financial investment income: interest (tax-free included),
#: dividends and other investment income. TOTCAPB4 counts accounts and assets
#: and has no property code (SN 9563), so let-property income is not part of
#: the capital it predicts (microcosm#1095).
UC_FINANCIAL_INVESTMENT_INCOME_COLUMNS = (
    "savings_interest_income",
    "dividend_income",
    "other_investment_income",
)


def benunit_financial_investment_income(
    person: pd.DataFrame, benunit: pd.DataFrame
) -> np.ndarray:
    """Sum each member's financial investment income in benefit-unit row order."""

    missing = sorted(
        {"person_benunit_id", *UC_FINANCIAL_INVESTMENT_INCOME_COLUMNS} - set(person)
    )
    if missing:
        raise KeyError(f"benefit-unit investment income inputs missing: {missing}")
    values = (
        person[list(UC_FINANCIAL_INVESTMENT_INCOME_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .sum(axis=1, min_count=len(UC_FINANCIAL_INVESTMENT_INCOME_COLUMNS))
    )
    if not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError("benefit-unit investment income inputs must be finite.")
    if (values < 0.0).any():
        raise ValueError("benefit-unit investment income inputs must be nonnegative.")
    totals = values.groupby(person["person_benunit_id"].to_numpy(), sort=False).sum()
    return (
        totals.reindex(benunit["benunit_id"].to_numpy())
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def household_family_role_counts(
    person: pd.DataFrame, household: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Claimants and partners, and every other member, per household.

    In household row order. The persisted FRS adult-file role
    ``is_uc_claimant`` splits each household into its adults and its dependent
    children the way survey household grids do (WAS ``NumAdultR8`` and
    ``NumChildR8`` sum to the household size), rather than at the engine's
    age-18 ``is_adult`` (uk-data#486, microcosm#1095).
    """

    missing = sorted({"person_household_id", "is_uc_claimant"} - set(person))
    if missing:
        raise KeyError(f"household role-count inputs missing: {missing}")
    role = person["is_uc_claimant"]
    if not pd.api.types.is_bool_dtype(role.dtype):
        raise ValueError("is_uc_claimant must be a boolean column.")
    claimant = role.to_numpy(dtype=bool)
    members = person["person_household_id"].to_numpy()
    ids = household["household_id"].to_numpy()
    adults = (
        pd.Series(claimant.astype(float)).groupby(members, sort=False).sum()
    ).reindex(ids, fill_value=0.0)
    children = (
        pd.Series((~claimant).astype(float)).groupby(members, sort=False).sum()
    ).reindex(ids, fill_value=0.0)
    return adults.to_numpy(dtype=float), children.to_numpy(dtype=float)
