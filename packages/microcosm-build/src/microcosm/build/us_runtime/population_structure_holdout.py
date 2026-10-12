"""Held-out population-structure checks, never calibration targets.

The US target surface pins people by state, age and sex, tax returns, and
Social Security dollars by payment type, but no household-type, marital-status
or beneficiary-count series. A release can therefore meet every target while
its older population is mostly married couples and its Social Security dollars
sit on too few, or synthetic, beneficiaries (microcosm#880, #1212, #1213,
#1214). This module reports those margins beside
the other post-export diagnostics. It reads the written H5 directly; nothing
here is simulated, fitted, or enforced.

Four comparisons, each against fixed published observations:

- Marital status of people 65 and over, by sex: ACS 2024 1-year table B12002.
- People 65 and over who live alone: ACS 2024 1-year table B09021.
- People receiving Social Security, total, 65 and over, and under 65:
  SSA Monthly Statistical Snapshot, December 2024, Table 1.
- Beneficiaries by component against the benefit types behind the release's
  SSA payment targets (retired-worker, disabled-worker, survivor, spouse and
  child): the same snapshot's Table 2.

A fifth, structural field counts people with more than one positive Social
Security component. The CPS-carried split assigns each reporter's whole
benefit to one component, so a positive count points to imputed records
(microcosm#1214). It is not an exact entitlement check: SSA counts a dually
entitled beneficiary once in Table 1 and under each benefit in Table 2.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

SOCIAL_SECURITY_COMPONENTS = (
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
)
MARITAL_STATUSES = ("married", "widowed", "divorced", "separated", "never_married")
#: CPS ASEC A_MARITL -> the ACS B12002 status rows. Code 3 (spouse absent,
#: not separated) is "married, spouse absent: other" in B12002, so it counts
#: as married in both sources.
A_MARITL_STATUS = MappingProxyType(
    {
        1: "married",
        2: "married",
        3: "married",
        4: "widowed",
        5: "divorced",
        6: "separated",
        7: "never_married",
    }
)
OLDER_AGE = 65


@dataclass(frozen=True)
class MaritalCounts:
    married_spouse_present: int
    married_spouse_absent_other: int
    separated: int
    widowed: int
    divorced: int
    never_married: int

    @property
    def total(self) -> int:
        return (
            self.married_spouse_present
            + self.married_spouse_absent_other
            + self.separated
            + self.widowed
            + self.divorced
            + self.never_married
        )

    def counts(self) -> dict[str, int]:
        return {
            "married": self.married_spouse_present + self.married_spouse_absent_other,
            "widowed": self.widowed,
            "divorced": self.divorced,
            "separated": self.separated,
            "never_married": self.never_married,
        }

    def shares(self) -> dict[str, float]:
        return {status: n / self.total for status, n in self.counts().items()}

    def __add__(self, other: MaritalCounts) -> MaritalCounts:
        return MaritalCounts(
            *(
                getattr(self, field) + getattr(other, field)
                for field in self.__dataclass_fields__
            )
        )


#: ACS 2024 1-year, national, B12002 (sex by marital status by age; universe:
#: population 15 years and over). Each count sums the 65-74, 75-84 and 85+
#: cells: men B12002_015-017 (never married), 031-033 (spouse present),
#: 047-049 (separated), 062-064 (spouse absent, other), 077-079 (widowed),
#: 092-094 (divorced); women the same rows offset by 93.
ACS_2024_B12002_65_PLUS = MappingProxyType(
    {
        "male": MaritalCounts(
            married_spouse_present=17_915_854,
            married_spouse_absent_other=891_428,
            separated=359_868,
            widowed=2_834_778,
            divorced=3_671_510,
            never_married=2_009_873,
        ),
        "female": MaritalCounts(
            married_spouse_present=14_820_691,
            married_spouse_absent_other=766_485,
            separated=392_837,
            widowed=9_531_373,
            divorced=5_794_197,
            never_married=2_256_389,
        ),
    }
)
ACS_2024_B12002_SOURCE_URL = (
    "https://api.census.gov/data/2024/acs/acs1?get=group(B12002)&for=us:1"
)

#: ACS 2024 1-year, national, B09021 (living arrangements of adults 18 years
#: and over; universe: population 18+ in households): 65 years and over
#: (B09021_022E) and its "lives alone" row (B09021_023E).
ACS_2024_B09021_65_PLUS_ADULTS = 59_418_120
ACS_2024_B09021_65_PLUS_LIVES_ALONE = 15_784_869
ACS_2024_B09021_SOURCE_URL = (
    "https://api.census.gov/data/2024/acs/acs1?get=group(B09021)&for=us:1"
)

SSA_SNAPSHOT_2024_12_SOURCE_URL = (
    "https://www.ssa.gov/policy/docs/quickfacts/stat_snapshot/2024-12.html"
)


@dataclass(frozen=True)
class SocialSecurityPeople:
    """SSA snapshot Table 1 people receiving Social Security (thousands)."""

    social_security_only: int
    both_social_security_and_ssi: int

    @property
    def total(self) -> int:
        return 1_000 * (self.social_security_only + self.both_social_security_and_ssi)


#: SSA Monthly Statistical Snapshot, December 2024, Table 1 (thousands).
#: Dual entitlement is counted once. "Under 65" adds the "disabled, under age
#: 65" and "other" rows; SSI-only recipients are excluded throughout.
SSA_SNAPSHOT_2024_12_TABLE_1 = MappingProxyType(
    {
        "all": SocialSecurityPeople(65_522, 2_533),
        "65_plus": SocialSecurityPeople(53_595, 1_403),
        "under_65": SocialSecurityPeople(6_387 + 5_540, 1_130),
    }
)

#: SSA Monthly Statistical Snapshot, December 2024, Table 2 beneficiaries
#: (thousands), grouped as the release's SSA payment targets label them
#: (Annual Statistical Supplement payment types): retired-worker,
#: disabled-worker, survivor, and spouse and child benefits.
SSA_SNAPSHOT_2024_12_TABLE_2_BY_COMPONENT = MappingProxyType(
    {
        "social_security_retirement": MappingProxyType({"retired_workers": 51_773}),
        "social_security_disability": MappingProxyType({"disabled_workers": 7_231}),
        "social_security_survivors": MappingProxyType(
            {
                "children_of_deceased_workers": 2_051,
                "widowed_mothers_and_fathers": 104,
                "nondisabled_widowers": 3_434,
                "disabled_widowers": 196,
                "parents_of_deceased_workers": 1,
            }
        ),
        "social_security_dependents": MappingProxyType(
            {
                "spouses_of_retired_workers": 1_862,
                "children_of_retired_workers": 714,
                "spouses_of_disabled_workers": 86,
                "children_of_disabled_workers": 1_005,
            }
        ),
    }
)
SSA_SNAPSHOT_2024_12_TABLE_2_TOTAL = 68_456


def _benchmark_people(component: str) -> int:
    return 1_000 * sum(SSA_SNAPSHOT_2024_12_TABLE_2_BY_COMPONENT[component].values())


def _share(numerator: float, denominator: float) -> float | None:
    """A weighted share, clipped to [0, 1] against float summation residue."""
    if denominator <= 0:
        return None
    return min(1.0, max(0.0, numerator / denominator))


def _difference_pp(model: float | None, benchmark: float) -> float | None:
    return None if model is None else 100.0 * (model - benchmark)


def _relative(model: float, benchmark: float) -> float:
    return model / benchmark - 1.0


def _flag(values: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    """A boolean person flag; missing values (NaN, None, NA) are False."""
    if values is None:
        return np.zeros(shape, dtype=bool)
    import pandas as pd

    return pd.Series(values).fillna(False).astype(bool).to_numpy()


def _marital_status(
    marital_code: np.ndarray | None,
    *,
    is_surviving_spouse: np.ndarray | None,
    is_separated: np.ndarray | None,
    spouse_present: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Status labels per person and the measure used to derive them.

    With ``A_MARITL``, any value outside CPS codes 1-7 (missing, 0, a donor
    record without a CPS code) is labelled ``unknown`` and reported, so one
    bad value never blanks the report.
    """
    if marital_code is not None:
        codes = np.asarray(marital_code, dtype=np.float64)
        labels = np.full(codes.shape, "unknown", dtype=object)
        for code, status in A_MARITL_STATUS.items():
            labels[codes == code] = status
        return labels, "A_MARITL (CPS ASEC marital status)"
    # Releases without the raw code: married = a resident spouse in the
    # person's marital unit; widowed and separated from the engine inputs the
    # relationship stage derives from A_MARITL (codes 4 and 6). Divorced,
    # never married and spouse-absent people cannot be told apart, so they
    # are reported together.
    shape = spouse_present.shape
    labels = np.full(shape, "other_unmarried_or_spouse_absent", dtype=object)
    labels[_flag(is_separated, shape)] = "separated"
    labels[_flag(is_surviving_spouse, shape)] = "widowed"
    labels[np.asarray(spouse_present, dtype=bool)] = "married_spouse_present"
    return labels, "marital-unit spouse, is_surviving_spouse, is_separated"


def population_structure_holdout_payload(
    *,
    age: np.ndarray,
    is_female: np.ndarray,
    household_id: np.ndarray,
    marital_unit_id: np.ndarray,
    components: Mapping[str, np.ndarray],
    weights: np.ndarray,
    period: int,
    marital_code: np.ndarray | None = None,
    is_surviving_spouse: np.ndarray | None = None,
    is_separated: np.ndarray | None = None,
    release_id: str | None = None,
) -> dict[str, Any]:
    """Population-structure comparisons from aligned person arrays.

    ``weights`` are each person's household weight. A person lives alone when
    no other person shares their ``household_id``; a spouse is present when
    another adult shares their ``marital_unit_id``. Empty weighted groups
    report null shares, never zero or a failure.
    """
    age = np.asarray(age, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    is_female = np.asarray(is_female, dtype=bool)
    household_id = np.asarray(household_id)
    marital_unit_id = np.asarray(marital_unit_id)
    arrays = {
        "is_female": is_female,
        "household_id": household_id,
        "marital_unit_id": marital_unit_id,
        "weights": weights,
    }
    missing = [name for name in SOCIAL_SECURITY_COMPONENTS if name not in components]
    if missing:
        raise ValueError(f"Population structure needs component(s) {missing}.")
    component_values = {
        name: np.asarray(components[name], dtype=np.float64)
        for name in SOCIAL_SECURITY_COMPONENTS
    }
    arrays.update(component_values)
    for name, values in (
        ("marital_code", marital_code),
        ("is_surviving_spouse", is_surviving_spouse),
        ("is_separated", is_separated),
    ):
        if values is not None:
            arrays[name] = np.asarray(values)
    if age.ndim != 1 or any(values.shape != age.shape for values in arrays.values()):
        raise ValueError(
            "Population structure needs aligned one-dimensional person arrays."
        )
    numeric = [age, weights, *component_values.values()]
    if any(not np.isfinite(values).all() for values in numeric) or np.any(weights < 0):
        raise ValueError(
            "Population structure needs finite values and nonnegative weights."
        )

    _, household_index = np.unique(household_id, return_inverse=True)
    household_size = np.bincount(household_index)[household_index]
    adult = age >= 18
    _, unit_index = np.unique(marital_unit_id, return_inverse=True)
    unit_adults = np.bincount(unit_index, weights=adult.astype(np.float64))[unit_index]
    spouse_present = adult & (unit_adults >= 2)
    status, marital_measure = _marital_status(
        marital_code,
        is_surviving_spouse=is_surviving_spouse,
        is_separated=is_separated,
        spouse_present=spouse_present,
    )
    older = age >= OLDER_AGE

    marital_rows = []
    benchmark_all = ACS_2024_B12002_65_PLUS["male"] + ACS_2024_B12002_65_PLUS["female"]
    for sex, mask, benchmark in (
        ("male", older & ~is_female, ACS_2024_B12002_65_PLUS["male"]),
        ("female", older & is_female, ACS_2024_B12002_65_PLUS["female"]),
        ("all", older, benchmark_all),
    ):
        unknown = float(weights[mask & (status == "unknown")].sum())
        total = float(weights[mask & (status != "unknown")].sum())
        benchmark_shares = benchmark.shares()
        if marital_code is None:
            n = benchmark.total
            benchmark_shares = {
                "married_spouse_present": benchmark.married_spouse_present / n,
                "widowed": benchmark.widowed / n,
                "separated": benchmark.separated / n,
                "other_unmarried_or_spouse_absent": (
                    benchmark.divorced
                    + benchmark.never_married
                    + benchmark.married_spouse_absent_other
                )
                / n,
            }
        rows = []
        for label in benchmark_shares:
            share = _share(float(weights[mask & (status == label)].sum()), total)
            rows.append(
                {
                    "status": label,
                    "share": share,
                    "benchmark_share": benchmark_shares[label],
                    "difference_percentage_points": _difference_pp(
                        share, benchmark_shares[label]
                    ),
                }
            )
        marital_rows.append(
            {
                "sex": sex,
                "weighted_people": total,
                "weighted_people_unknown_status": unknown,
                "benchmark_people": benchmark.total,
                "statuses": rows,
            }
        )

    older_people = float(weights[older].sum())
    alone_share = _share(
        float(weights[older & (household_size == 1)].sum()), older_people
    )
    alone_benchmark = (
        ACS_2024_B09021_65_PLUS_LIVES_ALONE / ACS_2024_B09021_65_PLUS_ADULTS
    )

    positive = np.column_stack(
        [component_values[name] > 0 for name in SOCIAL_SECURITY_COMPONENTS]
    )
    n_positive = positive.sum(axis=1)
    beneficiary = n_positive > 0
    people_rows = []
    for key, mask in (
        ("all", np.ones_like(beneficiary)),
        ("65_plus", older),
        ("under_65", ~older),
    ):
        model = float(weights[beneficiary & mask].sum())
        benchmark = SSA_SNAPSHOT_2024_12_TABLE_1[key].total
        people_rows.append(
            {
                "age_group": key,
                "weighted_beneficiaries": model,
                "benchmark_people": benchmark,
                "relative_difference": _relative(model, benchmark),
            }
        )
    component_rows = []
    for index, name in enumerate(SOCIAL_SECURITY_COMPONENTS):
        model = float(weights[positive[:, index]].sum())
        benchmark = _benchmark_people(name)
        single = positive[:, index] & (n_positive == 1)
        component_rows.append(
            {
                "component": name,
                "weighted_people_with_component": model,
                "weighted_people_with_only_this_component": float(
                    weights[single].sum()
                ),
                "benchmark_beneficiaries": benchmark,
                "benchmark_types": {
                    label: 1_000 * value
                    for label, value in SSA_SNAPSHOT_2024_12_TABLE_2_BY_COMPONENT[
                        name
                    ].items()
                },
                "relative_difference": _relative(model, benchmark),
            }
        )
    multiple = n_positive > 1
    structure = {
        "people_by_positive_component_count": {
            str(k): {
                "records": int((n_positive == k).sum()),
                "weighted_people": float(weights[n_positive == k].sum()),
            }
            for k in range(len(SOCIAL_SECURITY_COMPONENTS) + 1)
        },
        "records_with_multiple_components": int(multiple.sum()),
        "weighted_people_with_multiple_components": float(weights[multiple].sum()),
        "share_of_beneficiaries_with_multiple_components": _share(
            float(weights[multiple].sum()), float(weights[beneficiary].sum())
        ),
        "note": (
            "The CPS-carried split assigns each reporter's whole benefit to one "
            "component, so a positive count points to imputed records spread "
            "across components. Dual entitlement exists (SSA Table 2 counts a "
            "dually entitled beneficiary under each benefit), so this is not "
            "an exact entitlement check."
        ),
    }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "available",
        "period": int(period),
        "target_role": "validation",
        "enforced": False,
        "used_for_calibration": False,
        "weight_basis": "person's calibrated household_weight",
        "marital_status_65_plus": {
            "measure": marital_measure,
            "benchmark": {
                "source": "ACS 2024 1-year, table B12002, national",
                "source_url": ACS_2024_B12002_SOURCE_URL,
                "universe": "population 15 years and over, 65+ cells",
                "married": (
                    "now married, spouse present or absent (not separated)"
                    if marital_code is not None
                    else "now married, spouse present (the only marriage the "
                    "marital unit shows); spouse-absent married people are in "
                    "other_unmarried_or_spouse_absent"
                ),
            },
            "by_sex": marital_rows,
        },
        "living_alone_65_plus": {
            "measure": "people 65+ who are the only person in their household",
            "weighted_people_65_plus": older_people,
            "share": alone_share,
            "benchmark_share": alone_benchmark,
            "difference_percentage_points": _difference_pp(
                alone_share, alone_benchmark
            ),
            "benchmark": {
                "source": "ACS 2024 1-year, table B09021, national",
                "source_url": ACS_2024_B09021_SOURCE_URL,
                "universe": "adults 18+ in households, 65 years and over",
                "lives_alone": ACS_2024_B09021_65_PLUS_LIVES_ALONE,
                "adults_65_plus": ACS_2024_B09021_65_PLUS_ADULTS,
            },
        },
        "social_security_beneficiaries": {
            "measure": "people with any positive Social Security component",
            "by_age": people_rows,
            "by_component": component_rows,
            "benchmark": {
                "source": "SSA Monthly Statistical Snapshot, December 2024",
                "source_url": SSA_SNAPSHOT_2024_12_SOURCE_URL,
                "people": "Table 1: Social Security only plus both Social Security and SSI",
                "by_component": (
                    "Table 2 beneficiaries grouped as the release's SSA payment "
                    "targets: retired workers; disabled workers; survivors; "
                    "spouses and children of retired and disabled workers"
                ),
                "table_2_total": 1_000 * SSA_SNAPSHOT_2024_12_TABLE_2_TOTAL,
            },
        },
        "social_security_component_structure": structure,
        "comparability": (
            "SSA counts people in current-payment status in December 2024; the "
            f"model counts people with a positive annual benefit in {int(period)}, "
            "which runs somewhat above a December count. The CPS ASEC asks "
            "income only of people 15 and over, so child beneficiaries under 15 "
            "(within SSA's 3.8M children) enter the model only through "
            "imputation. ACS counts the 2024 resident population; the model's "
            "household universe follows the CPS ASEC. These are held-out "
            "diagnostics, not exact population or period matches, calibration "
            "targets, or release gates."
        ),
    }
    if release_id is not None:
        payload["release_id"] = release_id
    return payload


def population_structure_holdout_from_h5(
    dataset_path: Path | str, *, period: int, release_id: str | None = None
) -> dict[str, Any]:
    """Read the written release H5's person and household tables (no engine)."""
    import pandas as pd

    wanted = [
        "person_household_id",
        "person_marital_unit_id",
        "age",
        "A_AGE",
        "is_female",
        "A_SEX",
        "A_MARITL",
        "is_surviving_spouse",
        "is_separated",
        *SOCIAL_SECURITY_COMPONENTS,
    ]
    with pd.HDFStore(str(dataset_path), "r") as store:
        storer = store.get_storer("person")
        if storer.is_table:
            available = set(storer.non_index_axes[0][1])
            person = store.select(
                "person", columns=[column for column in wanted if column in available]
            )
        else:
            person = store["person"]
        household = (
            store.select("household", columns=["household_id", "household_weight"])
            if store.get_storer("household").is_table
            else store["household"]
        )
    weight_by_household = pd.Series(
        household["household_weight"].to_numpy(dtype=np.float64),
        index=household["household_id"].to_numpy(),
    )
    weights = person["person_household_id"].map(weight_by_household)
    if weights.isna().any():
        raise ValueError(
            f"{int(weights.isna().sum())} person(s) have no household weight."
        )
    age_column = "age" if "age" in person else "A_AGE"
    if "is_female" in person:
        is_female = person["is_female"].to_numpy(dtype=bool)
    else:
        is_female = pd.to_numeric(person["A_SEX"]).to_numpy() == 2

    def optional(column: str) -> np.ndarray | None:
        return person[column].to_numpy() if column in person else None

    return population_structure_holdout_payload(
        age=pd.to_numeric(person[age_column]).to_numpy(dtype=np.float64),
        is_female=is_female,
        household_id=person["person_household_id"].to_numpy(),
        marital_unit_id=person["person_marital_unit_id"].to_numpy(),
        components={
            name: pd.to_numeric(person[name]).fillna(0.0).to_numpy(dtype=np.float64)
            for name in SOCIAL_SECURITY_COMPONENTS
        },
        weights=weights.to_numpy(dtype=np.float64),
        period=period,
        marital_code=optional("A_MARITL"),
        is_surviving_spouse=optional("is_surviving_spouse"),
        is_separated=optional("is_separated"),
        release_id=release_id,
    )


def population_structure_holdout_unavailable(
    error: Exception, *, period: int, release_id: str | None = None
) -> dict[str, Any]:
    """An observational receipt when the H5 cannot be read or measured."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "unavailable",
        "period": int(period),
        "target_role": "validation",
        "enforced": False,
        "used_for_calibration": False,
        "error": f"{type(error).__name__}: {error}",
    }
    if release_id is not None:
        payload["release_id"] = release_id
    return payload


def write_population_structure_holdout(
    payload: dict[str, Any], path: Path | str
) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return path
