"""Measure how the SPI support half of a UK spine relates to its FRS parents.

Development diagnostic for the SPI-before-donor-imputations reorder. It reads a
built spine H5 and pairs every SPI support row with the FRS row it was copied
from (``person_source_id`` / ``household_source_id``), leaving out the CGT
clones and CGT band donors, which are stacked later. The JSON output is
aggregate only; no record is written. Cells under ``MIN_CELL`` records report
``null``.

Four surfaces:

* ``income_vs_parent``: SPI adults' incomes against their FRS parent's
  (Spearman, zero/non-zero agreement, share within 10 %), next to a placebo
  that pairs each SPI adult with a random FRS adult of the same SPI age band,
  sex and region. Stage 1 conditions on age, sex and region only, so the
  actual correlation should equal the placebo.
* ``inherited_from_parent``: per donor-imputed column, the share of SPI rows
  whose value equals the parent's, overall and where either side is non-zero.
* ``coherence``: income against wealth and spending on each half.
* ``band_donors``: wealth and spending by #1006 income band.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN_CELL = 10
INCOME_COLUMNS = (
    "employment_income",
    "self_employment_income",
    "savings_interest_income",
    "dividend_income",
    "private_pension_income",
    "property_income",
)
INVESTMENT_INCOME_COLUMNS = (
    "savings_interest_income",
    "dividend_income",
    "property_income",
)
#: Household columns written by was_wealth, lcfs_consumption, etb_vat and
#: etb_services (regional_property_uprating rewrites two of them).
HOUSEHOLD_DONOR_COLUMNS = (
    "owned_land",
    "property_wealth",
    "corporate_wealth",
    "private_pension_wealth",
    "gross_financial_wealth",
    "net_financial_wealth",
    "main_residence_value",
    "other_residential_property_value",
    "non_residential_property_value",
    "savings",
    "num_vehicles",
    "cash_isa",
    "stocks_and_shares_isa",
    "mortgage_debt",
    "consumer_debt",
    "food_and_non_alcoholic_beverages_consumption",
    "alcohol_and_tobacco_consumption",
    "clothing_and_footwear_consumption",
    "housing_water_and_electricity_consumption",
    "household_furnishings_consumption",
    "health_consumption",
    "transport_consumption",
    "communication_consumption",
    "recreation_consumption",
    "education_consumption",
    "restaurants_and_hotels_consumption",
    "miscellaneous_consumption",
    "petrol_spending",
    "diesel_spending",
    "bus_fare_spending",
    "domestic_energy_consumption",
    "electricity_consumption",
    "gas_consumption",
    "full_rate_vat_expenditure_rate",
    "dfe_education_spending",
    "rail_subsidy_spending",
    "bus_subsidy_spending",
    "rail_usage",
)
#: Person columns written by was_wealth, nts_bus_travel and etb_services.
PERSON_DONOR_COLUMNS = (
    "student_loan_balance",
    "local_bus_trips",
    "bus_in_london_trips",
    "other_local_bus_trips",
    "a_and_e_visits",
    "admitted_patient_visits",
    "outpatient_visits",
    "nhs_a_and_e_spending",
    "nhs_admitted_patient_spending",
    "nhs_outpatient_spending",
)
WEALTH_COLUMNS = (
    "gross_financial_wealth",
    "corporate_wealth",
    "property_wealth",
    "savings",
)
#: The SPI 2022-23 tape's AGERANGE lower bounds (stage-1 donor age bands).
SPI_AGE_BAND_EDGES = (16, 25, 35, 45, 55, 65, 74, 200)


def load_spine(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with pd.HDFStore(path, mode="r") as store:
        return store["person"], store["household"]


def split_halves(
    person: pd.DataFrame, household: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """FRS base and SPI support rows, without the later CGT copies."""

    flags = household.set_index("household_id")[
        [
            column
            for column in (
                "household_is_capital_gains_clone",
                "household_is_cgt_band_donor",
            )
            if column in household.columns
        ]
    ]
    later_copy = flags.astype(bool).any(axis=1)
    plain = household.loc[~household["household_id"].map(later_copy).to_numpy()]
    frs_households = plain.loc[plain["household_support_channel"] == "frs"]
    spi_households = plain.loc[plain["household_support_channel"] == "spi"]
    person = person.loc[person["person_household_id"].isin(plain["household_id"])]
    return {
        "frs_households": frs_households,
        "spi_households": spi_households,
        "frs_persons": person.loc[person["person_support_channel"] == "frs"],
        "spi_persons": person.loc[person["person_support_channel"] == "spi"],
    }


def _paired(child: pd.DataFrame, parent: pd.DataFrame, *, key: str, parent_id: str):
    parent = parent.set_index(parent_id)
    child = child.loc[child[key].isin(parent.index)].reset_index(drop=True)
    return child, parent.loc[child[key].to_numpy()].copy().reset_index()


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < MIN_CELL or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    return float(spearmanr(left, right).correlation)


def _income_metrics(child: pd.DataFrame, parent: pd.DataFrame) -> dict[str, object]:
    result = {}
    for column in INCOME_COLUMNS:
        a = child[column].to_numpy(dtype=float)
        b = parent[column].to_numpy(dtype=float)
        positive = b > 0
        result[column] = {
            "spearman": _spearman(a, b),
            "zero_status_agreement": float(((a > 0) == (b > 0)).mean()),
            "within_10pct_of_parent_share": (
                float((np.abs(a[positive] - b[positive]) <= 0.1 * b[positive]).mean())
                if positive.sum() >= MIN_CELL
                else None
            ),
        }
    return result


def income_vs_parent(
    halves: dict[str, pd.DataFrame],
    household: pd.DataFrame,
    *,
    placebo_reps: int,
    seed: int,
) -> dict[str, object]:
    spi = halves["spi_persons"]
    spi = spi.loc[spi["age"] >= SPI_AGE_BAND_EDGES[0]]
    child, parent = _paired(
        spi, halves["frs_persons"], key="person_source_id", parent_id="person_id"
    )
    actual = _income_metrics(child, parent)
    region = household.set_index("household_id")["region"].astype(str)
    band = pd.cut(parent["age"], SPI_AGE_BAND_EDGES, right=False).astype(str)
    cell = (
        band
        + "|"
        + parent["gender"].astype(str)
        + "|"
        + parent["person_household_id"].map(region).to_numpy()
    )
    rng = np.random.default_rng(seed)
    groups = pd.Series(np.arange(len(parent))).groupby(cell.to_numpy()).indices
    placebo: dict[str, list[float]] = {column: [] for column in INCOME_COLUMNS}
    for _ in range(placebo_reps):
        permutation = np.arange(len(parent))
        for positions in groups.values():
            permutation[positions] = rng.permutation(positions)
        shuffled = parent.iloc[permutation].reset_index(drop=True)
        for column in INCOME_COLUMNS:
            value = _spearman(
                child[column].to_numpy(dtype=float),
                shuffled[column].to_numpy(dtype=float),
            )
            if value is not None:
                placebo[column].append(value)
    for column in INCOME_COLUMNS:
        values = placebo[column]
        actual[column]["placebo_spearman_mean"] = (
            float(np.mean(values)) if values else None
        )
        actual[column]["placebo_spearman_sd"] = (
            float(np.std(values)) if values else None
        )
    return {"spi_adults": len(child), "items": actual}


def _identical_shares(
    child: pd.DataFrame, parent: pd.DataFrame, columns: Sequence[str]
) -> dict[str, object]:
    result = {}
    for column in columns:
        if column not in child.columns or column not in parent.columns:
            continue
        a = child[column].to_numpy(dtype=float)
        b = parent[column].to_numpy(dtype=float)
        either = (a != 0) | (b != 0)
        result[column] = {
            "identical_share": float((a == b).mean()),
            "identical_share_where_either_nonzero": (
                float((a[either] == b[either]).mean())
                if either.sum() >= MIN_CELL
                else None
            ),
        }
    return result


def inherited_from_parent(halves: dict[str, pd.DataFrame]) -> dict[str, object]:
    household_child, household_parent = _paired(
        halves["spi_households"],
        halves["frs_households"],
        key="household_source_id",
        parent_id="household_id",
    )
    person_child, person_parent = _paired(
        halves["spi_persons"],
        halves["frs_persons"],
        key="person_source_id",
        parent_id="person_id",
    )
    return {
        "spi_households": len(household_child),
        "spi_persons": len(person_child),
        "household": _identical_shares(
            household_child, household_parent, HOUSEHOLD_DONOR_COLUMNS
        ),
        "person": _identical_shares(person_child, person_parent, PERSON_DONOR_COLUMNS),
    }


def _household_income(person: pd.DataFrame) -> pd.DataFrame:
    grouped = person.groupby("person_household_id")[list(INCOME_COLUMNS)].sum()
    return pd.DataFrame(
        {
            "total_income": grouped.sum(axis=1),
            "investment_income": grouped[list(INVESTMENT_INCOME_COLUMNS)].sum(axis=1),
        }
    )


def _total_spending(household: pd.DataFrame) -> pd.Series:
    columns = [
        column
        for column in household.columns
        if column.endswith("_consumption") and column != "domestic_energy_consumption"
    ]
    return household[columns].sum(axis=1)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, q * cumulative[-1])])


def _median_or_none(values: pd.Series) -> float | None:
    return float(values.median()) if len(values) >= MIN_CELL else None


def _half_coherence(household: pd.DataFrame, person: pd.DataFrame) -> dict:
    joined = household.join(_household_income(person), on="household_id").fillna(
        {"total_income": 0.0, "investment_income": 0.0}
    )
    spending = _total_spending(joined)
    top = joined["total_income"] >= joined["total_income"].quantile(0.99)
    weights = joined["household_weight"].to_numpy(dtype=float)
    return {
        "households": len(joined),
        "spearman_investment_income_gross_financial_wealth": _spearman(
            joined["investment_income"].to_numpy(dtype=float),
            joined["gross_financial_wealth"].to_numpy(dtype=float),
        ),
        "spearman_total_income_gross_financial_wealth": _spearman(
            joined["total_income"].to_numpy(dtype=float),
            joined["gross_financial_wealth"].to_numpy(dtype=float),
        ),
        "spearman_total_income_corporate_wealth": _spearman(
            joined["total_income"].to_numpy(dtype=float),
            joined["corporate_wealth"].to_numpy(dtype=float),
        ),
        "spearman_total_income_spending": _spearman(
            joined["total_income"].to_numpy(dtype=float), spending.to_numpy(dtype=float)
        ),
        "top_1pct_income_threshold": float(joined["total_income"].quantile(0.99)),
        "top_1pct_median": {
            column: _median_or_none(joined.loc[top, column])
            for column in WEALTH_COLUMNS
        }
        | {"total_spending": _median_or_none(spending[top])},
        "prior_weighted": {
            column: {
                "mean": float(np.average(joined[column], weights=weights)),
                "p50": _weighted_quantile(
                    joined[column].to_numpy(dtype=float), weights, 0.5
                ),
                "p90": _weighted_quantile(
                    joined[column].to_numpy(dtype=float), weights, 0.9
                ),
                "p99": _weighted_quantile(
                    joined[column].to_numpy(dtype=float), weights, 0.99
                ),
            }
            for column in WEALTH_COLUMNS
        },
        "prior_weight_total": float(weights.sum()),
    }


def coherence(halves: dict[str, pd.DataFrame]) -> dict[str, object]:
    return {
        "frs_half": _half_coherence(halves["frs_households"], halves["frs_persons"]),
        "spi_half": _half_coherence(halves["spi_households"], halves["spi_persons"]),
    }


def band_donors(halves: dict[str, pd.DataFrame]) -> dict[str, object] | None:
    household = halves["spi_households"]
    if "household_is_spi_income_band_donor" not in household.columns:
        return None
    donors = household.loc[household["household_is_spi_income_band_donor"].astype(bool)]
    joined = donors.join(_household_income(halves["spi_persons"]), on="household_id")
    joined["total_spending"] = _total_spending(joined)
    result = {}
    for lower, rows in joined.groupby("spi_income_band_donor_lower_bound"):
        result[str(int(lower))] = {
            "households": len(rows),
            "median_total_income": _median_or_none(rows["total_income"]),
            **{
                f"median_{column}": _median_or_none(rows[column])
                for column in (*WEALTH_COLUMNS, "total_spending")
            },
        }
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spine-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--placebo-reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    person, household = load_spine(args.spine_h5)
    halves = split_halves(person, household)
    receipt = {
        "spine_h5": str(args.spine_h5),
        "spine_sha256": _sha256(args.spine_h5),
        "min_cell": MIN_CELL,
        "income_vs_parent": income_vs_parent(
            halves, household, placebo_reps=args.placebo_reps, seed=args.seed
        ),
        "inherited_from_parent": inherited_from_parent(halves),
        "coherence": coherence(halves),
        "band_donors": band_donors(halves),
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
