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

from microcosm.build.us_runtime.sld_local_solver import SldDistrictProblem

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
#: (income in the past 12 months of household members aged 15 and over —
#: the ACS universe; the under-15 exclusion is applied at measurement).
#: ``required`` components must resolve to at least one present column or
#: the recipe refuses to bind. Column choices verified against the
#: policyengine-us variable definitions: ``employment_income`` already
#: includes tips (``tip_income`` is a memo leg — adding it would double
#: count); SSTB self-employment is disjoint from non-SSTB
#: (``sstb_self_employment_income`` "treated separately"); farm
#: self-employment is ``farm_operations_income`` (Schedule F) — the model's
#: ``farm_income`` is Schedule J income averaging, "separate from
#: self-employment income", and is deliberately not a component.
SLD_ACS_MONEY_INCOME_RECIPE: tuple[MoneyIncomeComponent, ...] = (
    MoneyIncomeComponent(
        name="wages_salary_tips",
        columns=("employment_income_before_lsr",),
        acs_definition=(
            "wage or salary income (incl. commissions, bonuses, tips; the "
            "model input already includes tip income)"
        ),
        required=True,
    ),
    MoneyIncomeComponent(
        name="self_employment",
        columns=(
            "self_employment_income_before_lsr",
            "sstb_self_employment_income_before_lsr",
            "farm_operations_income",
        ),
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

#: Flows the ACS money-income concept itself excludes, plus model columns
#: deliberately kept out of the recipe with the reason stated.
SLD_ACS_MONEY_INCOME_EXCLUSIONS: Mapping[str, str] = {
    "capital_gains": "ACS money income excludes capital gains and losses",
    "in_kind_transfers": (
        "SNAP, housing subsidies, and other in-kind transfers are not money income"
    ),
    "tax_credits": "refundable tax credits are not ACS money income",
    "tip_income_column": (
        "the model's employment_income input already includes tips; adding "
        "the tip_income memo column would double count"
    ),
    "farm_income_column": (
        "the model's farm_income is Schedule J income averaging, separate "
        "from self-employment; farm self-employment enters via "
        "farm_operations_income"
    ),
    "under_15_income": (
        "ACS money income counts persons aged 15 and over; income carried "
        "by younger household members is excluded from the analog"
    ),
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


#: ACS money income counts income of persons aged 15 and over.
ACS_MONEY_INCOME_MINIMUM_AGE = 15.0


def household_acs_money_income(
    households: pd.DataFrame,
    persons: pd.DataFrame,
    resolution: MoneyIncomeRecipeResolution,
) -> np.ndarray:
    """The ACS money-income analog per household row, in household order.

    Person-level components count only members aged 15 and over — the ACS
    money-income universe. ``persons`` must carry ``age`` for that filter.
    """
    if "household_id" not in households.columns:
        raise ValueError("households must carry household_id.")
    if "person_household_id" not in persons.columns:
        raise ValueError("persons must carry person_household_id.")
    total = pd.Series(0.0, index=households["household_id"].to_numpy())
    if resolution.person_columns:
        if "age" not in persons.columns:
            raise ValueError(
                "persons must carry age: ACS money income counts members "
                "aged 15 and over only."
            )
        of_income_age = (
            persons["age"].to_numpy(dtype=np.float64) >= ACS_MONEY_INCOME_MINIMUM_AGE
        )
        person_sum = (
            persons.loc[of_income_age, list(resolution.person_columns)]
            .astype(np.float64)
            .sum(axis=1)
            .groupby(persons.loc[of_income_age, "person_household_id"].to_numpy())
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
        unsupported = sorted(
            {
                str(constraint.get("variable"))
                for constraint in constraints
                if str(constraint.get("variable")) not in ("age", "household_income")
            }
        )
        if unsupported:
            raise ValueError(
                f"line {line_number}: constraint variable(s) {unsupported} "
                "have no binding rule; compiling the fact without them would "
                "silently widen its universe."
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


def _group_sort_key(item: tuple) -> tuple:
    """Sort groupby items by their (state, district) key, subscript-free."""
    key, _frame = item
    return key


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


#: ACS PUMS TYPEHUGQ codes for group-quarters placeholder records. These
#: rows are population (S0101's universe) but not households (the
#: B19001/B19013 universe).
GROUP_QUARTERS_TYPEHUGQ_CODES = (2, 3)


def household_universe_mask(households: pd.DataFrame) -> np.ndarray:
    """True where a row is a housing-unit household in the ACS sense.

    ACS-spine group-quarters placeholder rows (``TYPEHUGQ`` 2/3) carry one
    person each and belong to the population universe only. Rows without a
    ``TYPEHUGQ`` column (donor spine, older artifact shapes) are housing
    units by construction.
    """
    if "TYPEHUGQ" not in households.columns:
        return np.ones(len(households), dtype=bool)
    kind = pd.to_numeric(households["TYPEHUGQ"], errors="coerce")
    return ~kind.isin(GROUP_QUARTERS_TYPEHUGQ_CODES).to_numpy()


@dataclass(frozen=True)
class SldProblemBuild:
    """Compiled district problems plus everything deliberately not compiled."""

    problems: tuple[SldDistrictProblem, ...]
    validation_facts: pd.DataFrame
    zero_support_districts: tuple[str, ...]
    money_income: np.ndarray
    recipe_resolution: MoneyIncomeRecipeResolution
    is_household: np.ndarray
    n_group_quarters_rows: int
    gq_marker_present: bool


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

    Universes: age-band rows count every member (total population, the
    S0101 universe); the household count and income-bracket rows are
    indicators on housing-unit households only (the B19001 universe), so
    group-quarters rows carry zero in them while still supporting the
    population targets. Every district of a chamber must carry the same
    metric set — an asymmetric fact surface is a broken input, refused.
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
    if not chamber_facts.empty:
        metric_sets = chamber_facts.groupby("area_code")["metric"].agg(
            lambda values: tuple(sorted(values))
        )
        if metric_sets.nunique() > 1:
            counts = metric_sets.value_counts()
            minority = metric_sets[metric_sets != counts.index[0]]
            raise ValueError(
                "asymmetric fact surface: district(s) "
                f"{sorted(minority.index.tolist())[:5]} carry a different "
                "metric set than the rest of the chamber."
            )
    resolution = resolve_money_income_recipe(
        persons.columns,
        households.columns,
    )
    money_income = household_acs_money_income(households, persons, resolution)
    is_household = household_universe_mask(households)
    gq_marker_present = "TYPEHUGQ" in households.columns

    ages = persons["age"].to_numpy(dtype=np.float64)
    person_household = persons["person_household_id"].to_numpy()
    household_ids = households["household_id"].to_numpy()
    state_by_row = households["state_fips"].map(lambda value: f"{int(value):02d}")
    membership = households[membership_column].astype(str)

    # One pass over the frame: positional indices per (state, district), and
    # person rows resolved to household positions once — district compiles
    # then slice, never rescan (the national run is ~6.8k districts).
    district_key = state_by_row.str.cat(membership, sep=":")
    rows_by_district = {
        key: indices.to_numpy()
        for key, indices in pd.RangeIndex(len(households))
        .to_series()
        .groupby(district_key.to_numpy())
        .groups.items()
    }
    position_by_household_id = pd.Series(
        np.arange(len(households)),
        index=household_ids,
    )
    person_household_position = position_by_household_id.reindex(
        person_household
    ).to_numpy()
    if np.isnan(person_household_position).any():
        raise ValueError(
            "persons reference household ids absent from the household table."
        )
    person_household_position = person_household_position.astype(np.int64)

    # Precompute the canonical metric ordering as sortable scalar columns —
    # the spine-blindness guard requires statically resolvable selectors, so
    # no dynamic .loc/lambda subscripts inside the district loop.
    sort_keys = [_metric_sort_key(row) for _, row in chamber_facts.iterrows()]
    chamber_facts = chamber_facts.assign(
        _order_family=[key[0] for key in sort_keys],
        _order_lower=[key[1] for key in sort_keys],
        _order_metric=[key[2] for key in sort_keys],
    )

    problems: list[SldDistrictProblem] = []
    zero_support: list[str] = []
    grouped = sorted(
        chamber_facts.groupby(["state_fips", "district_code"]),
        key=_group_sort_key,
    )
    for (state_fips, district_code), district_facts in grouped:
        ordered = district_facts.sort_values(
            ["_order_family", "_order_lower", "_order_metric"]
        )
        area_code = str(ordered["area_code"].iloc[0])
        indices = rows_by_district.get(f"{state_fips}:{district_code}")
        if indices is None or len(indices) == 0:
            zero_support.append(area_code)
            continue
        district_household_ids = household_ids[indices]
        district_income = money_income[indices]
        district_is_household = is_household[indices].astype(np.float64)
        n_rows = len(indices)
        # Person counts per (district row, age band) via one bincount per
        # band over positions remapped into the district's row space.
        position_in_district = np.full(len(households), -1, dtype=np.int64)
        position_in_district[indices] = np.arange(n_rows)
        person_district_position = position_in_district[person_household_position]
        in_district_person = person_district_position >= 0
        rows: list[np.ndarray] = []
        for fact in ordered.itertuples():
            metric = str(fact.metric)
            if metric.startswith("age_"):
                lower = 0.0 if pd.isna(fact.age_lower) else float(fact.age_lower)
                upper = np.inf if pd.isna(fact.age_upper) else float(fact.age_upper)
                in_band = in_district_person & (ages >= lower) & (ages < upper)
                rows.append(
                    np.bincount(
                        person_district_position[in_band],
                        minlength=n_rows,
                    ).astype(np.float64)
                )
            elif metric == "households":
                rows.append(district_is_household.copy())
            elif metric.startswith("income_"):
                lower = (
                    -np.inf if pd.isna(fact.income_lower) else float(fact.income_lower)
                )
                upper = (
                    np.inf if pd.isna(fact.income_upper) else float(fact.income_upper)
                )
                rows.append(
                    district_is_household
                    * ((district_income >= lower) & (district_income < upper))
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
                base_weights=base_weights[indices],
            )
        )
    return SldProblemBuild(
        problems=tuple(problems),
        validation_facts=validation_facts.reset_index(drop=True),
        zero_support_districts=tuple(zero_support),
        money_income=money_income,
        recipe_resolution=resolution,
        is_household=is_household,
        n_group_quarters_rows=int((~is_household).sum()),
        gq_marker_present=gq_marker_present,
    )
