"""SLD per-district targets from witnessed ledger facts (populace#625).

Ingests ledger-emitted ACS 5-year SLD facts (summary levels 610/620) and
binds them to the local artifact's household grain:

- ``census_acs.person_count`` facts with age constraints bind to
  per-household counts of members in the age band;
- ``census_acs.household_count`` facts bind to household indicators —
  unconstrained rows are the district household count, income-constrained
  rows are B19001 bracket indicators on the declared ACS money-income
  analog below;
- ``census_acs.median_household_income`` facts (aggregation ``median``) are
  **validation-only**: a linear reweighting operator can honestly hit
  bracket counts; it cannot honestly target a median. They are returned on
  the validation surface and never compiled into a solve.

The ACS money-income analog is a declared recipe over artifact input
columns (:data:`SLD_ACS_MONEY_INCOME_RECIPE`): the ACS "money income"
components mapped to the input surface the artifact actually carries, with
engine-computed transfers (SSI, TANF cash assistance) recorded as declared
omissions rather than silently approximated, and ACS-excluded flows
(capital gains, in-kind transfers) recorded as exclusions. The resolved
recipe — which columns were present, which were absent — is part of the
layer's provenance and the honest-boundaries statement.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from populace.build.us_runtime.sld_local_solver import SldDistrictProblem

__all__ = [
    "SLD_ACS_MONEY_INCOME_DECLARED_OMISSIONS",
    "SLD_ACS_MONEY_INCOME_EXCLUSIONS",
    "SLD_ACS_MONEY_INCOME_RECIPE",
    "MoneyIncomeRecipeResolution",
    "SldProblemBuild",
    "SldTargetFacts",
    "build_sld_district_problems",
    "household_acs_money_income",
    "load_sld_target_facts",
    "resolve_money_income_recipe",
]


@dataclass(frozen=True)
class MoneyIncomeComponent:
    """One ACS money-income component mapped to artifact input columns."""

    name: str
    columns: tuple[str, ...]
    acs_definition: str
    required: bool


#: The declared ACS money-income analog. Component definitions follow the
#: Census money-income concept the B19001/B19013 universes tabulate
#: (income in the past 12 months of household members). ``required``
#: components must resolve to at least one present column or the recipe
#: refuses to bind.
SLD_ACS_MONEY_INCOME_RECIPE: tuple[MoneyIncomeComponent, ...] = (
    MoneyIncomeComponent(
        name="wages_salary_tips",
        columns=("employment_income_before_lsr", "tip_income"),
        acs_definition="wage or salary income (incl. commissions, bonuses, tips)",
        required=True,
    ),
    MoneyIncomeComponent(
        name="self_employment",
        columns=("self_employment_income_before_lsr", "farm_income"),
        acs_definition="net self-employment income (own nonfarm and farm business)",
        required=True,
    ),
    MoneyIncomeComponent(
        name="interest_dividends_rental_estates",
        columns=(
            "taxable_interest_income",
            "tax_exempt_interest_income",
            "qualified_dividend_income",
            "non_qualified_dividend_income",
            "rental_income",
            "farm_rent_income",
            "estate_income",
        ),
        acs_definition=(
            "interest, dividends, net rental income, royalty income, and "
            "income from estates and trusts"
        ),
        required=True,
    ),
    MoneyIncomeComponent(
        name="social_security",
        columns=(
            "social_security_retirement",
            "social_security_disability",
            "social_security_survivors",
            "social_security_dependents",
        ),
        acs_definition="Social Security or Railroad Retirement income",
        required=True,
    ),
    MoneyIncomeComponent(
        name="retirement_pensions",
        columns=(
            "taxable_private_pension_income",
            "tax_exempt_private_pension_income",
            "taxable_ira_distributions",
            "tax_exempt_ira_distributions",
            "taxable_401k_distributions",
            "taxable_403b_distributions",
            "taxable_sep_distributions",
            "keogh_distributions",
            "survivor_benefits",
            "disability_benefits",
        ),
        acs_definition="retirement, survivor, or disability pensions",
        required=False,
    ),
    MoneyIncomeComponent(
        name="unemployment_workers_comp_veterans",
        columns=(
            "unemployment_compensation",
            "workers_compensation",
            "veterans_benefits",
        ),
        acs_definition=(
            "unemployment compensation, workers' compensation, and VA payments"
        ),
        required=False,
    ),
    MoneyIncomeComponent(
        name="other_regular_income",
        columns=(
            "alimony_income",
            "child_support_received",
            "financial_assistance",
        ),
        acs_definition=(
            "alimony, child support, and regular contributions from persons "
            "outside the household"
        ),
        required=False,
    ),
)

#: ACS money-income components the artifact carries only as engine-computed
#: outputs, so the input-native recipe omits them. Declared, never silent.
SLD_ACS_MONEY_INCOME_DECLARED_OMISSIONS: Mapping[str, str] = {
    "supplemental_security_income": (
        "SSI is engine-computed (seeded take-up), not an artifact input; "
        "omitted from the bracket instrument"
    ),
    "public_assistance_cash": (
        "TANF cash assistance is engine-computed, not an artifact input; "
        "omitted from the bracket instrument"
    ),
}

#: Flows the ACS money-income concept itself excludes.
SLD_ACS_MONEY_INCOME_EXCLUSIONS: Mapping[str, str] = {
    "capital_gains": "ACS money income excludes capital gains and losses",
    "in_kind_transfers": (
        "SNAP, housing subsidies, and other in-kind transfers are not money income"
    ),
    "tax_credits": "refundable tax credits are not ACS money income",
}


@dataclass(frozen=True)
class MoneyIncomeRecipeResolution:
    """Which recipe columns the frame actually carries, per component."""

    person_columns: tuple[str, ...]
    household_columns: tuple[str, ...]
    absent_columns: tuple[str, ...]
    components_present: tuple[str, ...]
    components_absent: tuple[str, ...]

    def as_record(self) -> dict:
        """JSON-ready provenance record, including the declared caveats."""
        return {
            "person_columns": list(self.person_columns),
            "household_columns": list(self.household_columns),
            "absent_columns": list(self.absent_columns),
            "components_present": list(self.components_present),
            "components_absent": list(self.components_absent),
            "declared_omissions": dict(SLD_ACS_MONEY_INCOME_DECLARED_OMISSIONS),
            "declared_exclusions": dict(SLD_ACS_MONEY_INCOME_EXCLUSIONS),
        }


def resolve_money_income_recipe(
    person_columns: Iterable[str],
    household_columns: Iterable[str],
) -> MoneyIncomeRecipeResolution:
    """Resolve the declared recipe against the frame's actual columns."""
    person_set = set(person_columns)
    household_set = set(household_columns)
    resolved_person: list[str] = []
    resolved_household: list[str] = []
    absent: list[str] = []
    present_components: list[str] = []
    absent_components: list[str] = []
    for component in SLD_ACS_MONEY_INCOME_RECIPE:
        any_present = False
        for column in component.columns:
            if column in person_set:
                resolved_person.append(column)
                any_present = True
            elif column in household_set:
                resolved_household.append(column)
                any_present = True
            else:
                absent.append(column)
        if any_present:
            present_components.append(component.name)
        else:
            absent_components.append(component.name)
            if component.required:
                raise ValueError(
                    f"required money-income component {component.name!r} has "
                    "no artifact column; the bracket instrument cannot bind "
                    f"honestly (candidates: {list(component.columns)})."
                )
    return MoneyIncomeRecipeResolution(
        person_columns=tuple(resolved_person),
        household_columns=tuple(resolved_household),
        absent_columns=tuple(absent),
        components_present=tuple(present_components),
        components_absent=tuple(absent_components),
    )


def household_acs_money_income(
    households: pd.DataFrame,
    persons: pd.DataFrame,
    resolution: MoneyIncomeRecipeResolution,
) -> np.ndarray:
    """The ACS money-income analog per household row, in household order."""
    if "household_id" not in households.columns:
        raise ValueError("households must carry household_id.")
    if "person_household_id" not in persons.columns:
        raise ValueError("persons must carry person_household_id.")
    total = pd.Series(0.0, index=households["household_id"].to_numpy())
    if resolution.person_columns:
        person_sum = (
            persons[list(resolution.person_columns)]
            .astype(np.float64)
            .sum(axis=1)
            .groupby(persons["person_household_id"].to_numpy())
            .sum()
        )
        total = total.add(person_sum, fill_value=0.0)
    if resolution.household_columns:
        household_sum = pd.Series(
            households[list(resolution.household_columns)]
            .astype(np.float64)
            .sum(axis=1)
            .to_numpy(),
            index=households["household_id"].to_numpy(),
        )
        total = total.add(household_sum, fill_value=0.0)
    return total.loc[households["household_id"].to_numpy()].to_numpy()


_AREA_TYPE_BY_LEVEL = {
    "state_legislative_district_upper": "sldu",
    "state_legislative_district_lower": "sldl",
}
_CALIBRATION_CONCEPTS = {
    "census_acs.person_count",
    "census_acs.household_count",
}
_VALIDATION_CONCEPTS = {"census_acs.median_household_income"}


@dataclass(frozen=True)
class SldTargetFacts:
    """Tidy SLD facts split into calibration and validation surfaces."""

    calibration: pd.DataFrame
    validation: pd.DataFrame
    source_path: str
    source_sha256: str
    geography_vintages: tuple[str, ...]


def _fact_concept(row: dict) -> str:
    measure = row.get("measure")
    if isinstance(measure, dict) and measure.get("concept"):
        return str(measure["concept"])
    alignment = row.get("concept_alignment")
    if isinstance(alignment, dict) and alignment.get("canonical_concept"):
        return str(alignment["canonical_concept"])
    raise ValueError(
        "fact row carries no concept (measure.concept or "
        "concept_alignment.canonical_concept)."
    )


def _fact_constraints(row: dict) -> list[dict]:
    constraints = row.get("constraints")
    if constraints is None and isinstance(row.get("universe_constraints"), dict):
        constraints = row["universe_constraints"].get("constraints")
    return list(constraints or ())


def _bounds(
    constraints: list[dict], variable: str
) -> tuple[float | None, float | None]:
    lower: float | None = None
    upper: float | None = None
    for constraint in constraints:
        if str(constraint.get("variable")) != variable:
            continue
        operator = str(constraint.get("operator"))
        value = float(constraint.get("value"))
        if operator == ">=":
            lower = value
        elif operator == "<":
            upper = value
        else:
            raise ValueError(
                f"unsupported {variable} constraint operator {operator!r}; "
                "the SLD surface binds [lower, upper) bands only."
            )
    return lower, upper


def _band_metric(prefix: str, lower: float | None, upper: float | None) -> str:
    if lower is None and upper is None:
        raise ValueError(f"{prefix} band needs at least one bound.")
    if lower is None:
        return f"{prefix}_under_{int(upper)}"
    if upper is None:
        return f"{prefix}_{int(lower)}_and_over"
    return f"{prefix}_{int(lower)}_to_{int(upper) - 1}"


def _metric_for(concept: str, entity: str, constraints: list[dict]) -> str:
    if concept == "census_acs.person_count":
        lower, upper = _bounds(constraints, "age")
        return _band_metric("age", lower, upper)
    if concept == "census_acs.household_count":
        lower, upper = _bounds(constraints, "household_income")
        if lower is None and upper is None:
            return "households"
        return _band_metric("income", lower, upper)
    raise ValueError(f"no metric rule for concept {concept!r} (entity {entity!r}).")


def _parse_sld_geo_id(geo_id: str) -> tuple[str, str]:
    marker = geo_id.find("US")
    if marker < 0 or len(geo_id) < marker + 7:
        raise ValueError(f"unparseable SLD GEO_ID {geo_id!r}.")
    state_fips = geo_id[marker + 2 : marker + 4]
    district_code = geo_id[marker + 4 :]
    if not state_fips.isdigit() or not district_code:
        raise ValueError(f"unparseable SLD GEO_ID {geo_id!r}.")
    return state_fips, district_code


def load_sld_target_facts(path: str | Path) -> SldTargetFacts:
    """Load SLD facts from a ledger facts/consumer-facts JSONL export.

    Non-SLD geography levels pass through untouched (the file may carry a
    whole bundle); SLD rows with an unrecognized concept are a hard error —
    a new fact family must be routed deliberately, never dropped silently.
    """
    content = Path(path).read_bytes()
    calibration_rows: list[dict] = []
    validation_rows: list[dict] = []
    vintages: set[str] = set()
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        geography = row.get("geography") or {}
        level = str(geography.get("level", ""))
        if level not in _AREA_TYPE_BY_LEVEL:
            continue
        area_type = _AREA_TYPE_BY_LEVEL[level]
        geo_id = str(geography.get("id", ""))
        state_fips, district_code = _parse_sld_geo_id(geo_id)
        concept = _fact_concept(row)
        constraints = _fact_constraints(row)
        entity = str((row.get("entity") or {}).get("name", ""))
        aggregation = str((row.get("aggregation") or {}).get("method", ""))
        vintage = str(geography.get("vintage", ""))
        vintages.add(vintage)
        base = {
            "area_type": area_type,
            "area_code": geo_id,
            "state_fips": state_fips,
            "district_code": district_code,
            "geography_vintage": vintage,
            "entity": entity,
            "concept": concept,
            "aggregation": aggregation,
            "value": float(row["value"]),
            "period": (row.get("period") or {}).get("value"),
        }
        if concept in _VALIDATION_CONCEPTS:
            if aggregation != "median":
                raise ValueError(
                    f"line {line_number}: {concept} must aggregate as median, "
                    f"got {aggregation!r}."
                )
            validation_rows.append({**base, "metric": "median_household_income"})
            continue
        if concept not in _CALIBRATION_CONCEPTS:
            raise ValueError(
                f"line {line_number}: unrecognized SLD fact concept "
                f"{concept!r}; route it deliberately before compiling."
            )
        if aggregation != "sum":
            raise ValueError(
                f"line {line_number}: calibration facts must aggregate as "
                f"sum, got {aggregation!r} for {concept}."
            )
        metric = _metric_for(concept, entity, constraints)
        income_lower, income_upper = _bounds(constraints, "household_income")
        age_lower, age_upper = _bounds(constraints, "age")
        calibration_rows.append(
            {
                **base,
                "metric": metric,
                "age_lower": age_lower,
                "age_upper": age_upper,
                "income_lower": income_lower,
                "income_upper": income_upper,
            }
        )
    calibration = pd.DataFrame(calibration_rows)
    validation = pd.DataFrame(validation_rows)
    if not calibration.empty:
        duplicated = calibration.duplicated(["area_type", "area_code", "metric"])
        if duplicated.any():
            examples = (
                calibration.loc[duplicated, ["area_code", "metric"]]
                .head(5)
                .to_numpy()
                .tolist()
            )
            raise ValueError(
                f"duplicate (area, metric) fact rows {examples}; the target "
                "surface must be uniform."
            )
    return SldTargetFacts(
        calibration=calibration,
        validation=validation,
        source_path=str(path),
        source_sha256=hashlib.sha256(content).hexdigest(),
        geography_vintages=tuple(sorted(vintages)),
    )


def _metric_sort_key(row: pd.Series) -> tuple:
    order = {"age": 0, "households": 1, "income": 2}
    metric = str(row["metric"])
    family = metric.split("_", 1)[0] if metric != "households" else "households"
    lower = row.get("age_lower")
    if pd.isna(lower) or lower is None:
        lower = row.get("income_lower")
    if pd.isna(lower) or lower is None:
        lower = -1.0
    return (order.get(family, 9), float(lower), metric)


@dataclass(frozen=True)
class SldProblemBuild:
    """Compiled district problems plus everything deliberately not compiled."""

    problems: tuple[SldDistrictProblem, ...]
    validation_facts: pd.DataFrame
    zero_support_districts: tuple[str, ...]
    money_income: np.ndarray
    recipe_resolution: MoneyIncomeRecipeResolution


def build_sld_district_problems(
    households: pd.DataFrame,
    persons: pd.DataFrame,
    base_weights: np.ndarray,
    facts: SldTargetFacts,
    *,
    area_type: str,
) -> SldProblemBuild:
    """Compile one chamber's per-district problems from facts + frame.

    ``households`` must carry ``household_id``, ``state_fips``, and the
    membership column for the chamber (``sld_upper_code``/``sld_lower_code``
    from :func:`assign_us_sld_membership`); ``persons`` must carry
    ``person_household_id`` and ``age``. Districts with facts but no
    assigned households are returned as ``zero_support_districts`` — a
    support finding, never a silent drop.
    """
    membership_column = {
        "sldu": "sld_upper_code",
        "sldl": "sld_lower_code",
    }.get(area_type)
    if membership_column is None:
        raise ValueError(f"area_type must be 'sldu' or 'sldl', got {area_type!r}.")
    for column in ("household_id", "state_fips", membership_column):
        if column not in households.columns:
            raise ValueError(f"households must carry {column}.")
    if "age" not in persons.columns:
        raise ValueError("persons must carry age.")
    base_weights = np.asarray(base_weights, dtype=np.float64)
    if len(base_weights) != len(households):
        raise ValueError(
            f"base_weights ({len(base_weights)}) must align with households "
            f"({len(households)})."
        )

    chamber_facts = facts.calibration[facts.calibration["area_type"] == area_type]
    validation_facts = (
        facts.validation[facts.validation["area_type"] == area_type]
        if not facts.validation.empty
        else facts.validation
    )
    resolution = resolve_money_income_recipe(
        persons.columns,
        households.columns,
    )
    money_income = household_acs_money_income(households, persons, resolution)

    ages = persons["age"].to_numpy(dtype=np.float64)
    person_household = persons["person_household_id"].to_numpy()
    household_ids = households["household_id"].to_numpy()
    state_by_row = households["state_fips"].map(lambda value: f"{int(value):02d}")
    membership = households[membership_column].astype(str)

    problems: list[SldDistrictProblem] = []
    zero_support: list[str] = []
    for (state_fips, district_code), district_facts in sorted(
        chamber_facts.groupby(["state_fips", "district_code"]),
        key=lambda item: item[0],
    ):
        ordered = district_facts.loc[
            sorted(
                district_facts.index,
                key=lambda index: _metric_sort_key(district_facts.loc[index]),
            )
        ]
        area_code = str(ordered["area_code"].iloc[0])
        mask = ((state_by_row == state_fips) & (membership == district_code)).to_numpy()
        if not mask.any():
            zero_support.append(area_code)
            continue
        district_household_ids = household_ids[mask]
        id_set = set(district_household_ids.tolist())
        person_mask = np.fromiter(
            (household in id_set for household in person_household),
            dtype=bool,
            count=len(person_household),
        )
        district_income = money_income[mask]
        rows: list[np.ndarray] = []
        for fact in ordered.itertuples():
            metric = str(fact.metric)
            if metric.startswith("age_"):
                lower = 0.0 if pd.isna(fact.age_lower) else float(fact.age_lower)
                upper = np.inf if pd.isna(fact.age_upper) else float(fact.age_upper)
                in_band = person_mask & (ages >= lower) & (ages < upper)
                counts = (
                    pd.Series(1.0, index=person_household[in_band])
                    .groupby(level=0)
                    .sum()
                )
                rows.append(
                    counts.reindex(district_household_ids, fill_value=0.0).to_numpy()
                )
            elif metric == "households":
                rows.append(np.ones(int(mask.sum())))
            elif metric.startswith("income_"):
                lower = (
                    -np.inf if pd.isna(fact.income_lower) else float(fact.income_lower)
                )
                upper = (
                    np.inf if pd.isna(fact.income_upper) else float(fact.income_upper)
                )
                rows.append(
                    ((district_income >= lower) & (district_income < upper)).astype(
                        np.float64
                    )
                )
            else:
                raise ValueError(f"no binding rule for metric {metric!r}.")
        problems.append(
            SldDistrictProblem(
                area_type=area_type,
                area_code=area_code,
                state_fips=str(state_fips),
                district_code=str(district_code),
                matrix=np.vstack(rows),
                targets=ordered["value"].to_numpy(dtype=np.float64),
                target_frame=ordered[
                    ["area_type", "area_code", "metric", "entity", "concept"]
                ].reset_index(drop=True),
                household_ids=district_household_ids,
                base_weights=base_weights[mask],
            )
        )
    return SldProblemBuild(
        problems=tuple(problems),
        validation_facts=validation_facts.reset_index(drop=True),
        zero_support_districts=tuple(zero_support),
        money_income=money_income,
        recipe_resolution=resolution,
    )
