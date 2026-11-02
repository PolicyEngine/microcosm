"""Generate Belgium's reviewed reference documents from a pinned Chronicle artifact.

Run with ``BE_CHRONICLE_ARTIFACT`` or ``--artifact``. References carry selectors
only: observations stay in Chronicle. The separate inventory resource records the
diagnostics the method card asks to publish (ageing ratios, stand-in gaps, the
B.1 checks) and every exclusion. Each choice that shapes numbers is a proposal
of the Belgian method card (BE-MC0 to BE-MC6), pending Max's ruling d1902;
nothing here publishes a number, and Belgium stays inactive.

``--pin-feed SOURCE_COMMIT`` rewrites ``chronicle_feed.json`` from the loaded
artifact's verified identity before generating; use it only when Chronicle
re-pins (for example once PolicyEngine/chronicle#401 merges).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

from microcosm.build.ledger_artifact import (
    LexicalFloat,
    chronicle_feed_identity,
    load_ledger_consumer_artifact,
    validate_chronicle_feed_pin,
)
from microcosm.build.ledger_targets import fact_matches_selector

SPEC = (
    Path(__file__).resolve().parents[1]
    / "packages/microcosm-build/src/microcosm/build/be"
)
POLICY_YEAR = 2026
METHOD_CARD_STATUS = "proposed_pending_d1902"
REGIONS = {
    "BE1": "Brussels-Capital Region",
    "BE2": "Flemish Region",
    "BE3": "Walloon Region",
}
AGE_BANDS = ("0_17", "18_64", "65_plus")
SEXES = ("male", "female")
FPB = "fpb.economic_outlook_2026_2031."
STATBEL_POPULATION = "statbel.population_structure.cy{year}.people.by_nuts1_age_sex"
COMMUNE_POPULATION = (
    "statbel.population_structure.cy2025.people.by_commune_nis2025.total"
)
A1 = "statbel.fiscal_income_distribution.income_year2023.{component}.by_income_class_eur1000"
A1_CLASS = "tax_return.net_taxable_income_class"
A1_COMPONENTS = (
    ("professional_income", "professional_income_eur", "in"),
    ("immovable_property_income", "immovable_property_income_eur", "gap"),
    (
        "capital_and_movable_property_income",
        "capital_and_movable_property_income_eur",
        "gap",
    ),
    ("miscellaneous_income", "miscellaneous_income_eur", "gap"),
    ("deductible_expenditures", "deductible_expenditures_eur", "gap_subtracted"),
)
B1_VALUES = "statbel.fiscal_income_distribution.b1.decile_values.v1"
B1_PERCENTILES = "statbel.fiscal_income_distribution.b1.top_decile_percentile_values.v1"
B1_BOUNDS = "statbel.fiscal_income_distribution.b1.decile_bounds.v1"
B1_PERCENTILE_BOUNDS = (
    "statbel.fiscal_income_distribution.b1.top_decile_percentile_bounds.v1"
)
B1_GROUPS = (
    *(f"decile_{decile:02d}" for decile in range(1, 10)),
    *(f"percentile_{percentile:03d}" for percentile in range(91, 101)),
)
B4 = "statbel.fiscal_income_distribution.b4.declaration_type_age_decile.v1"

# The d1952 stand-in bands, fixed in advance and before any candidate exists
# (BE-MC5; d1952 option A). Edges are income-year-2023 euros of total net
# taxable income, chosen on A.1's own class edges so every band is an exact sum
# of published EUR 1,000 classes and no cut is interpolated. EUR 10,000 steps
# cover the dense middle of the declaration distribution, which holds B.1's
# deciles 3 to 7 whole (their bounds run from EUR 16,867 to EUR 44,639); above
# EUR 50,000 counts thin, so the steps widen; A.1's open top class (EUR 100,000
# and more) stays whole. Eight bands give sixteen count and amount
# constraints, the "short list" d1952 asks for, not 101 classes. The
# lowest band is kept so the bands partition A.1's whole positive range and sum
# exactly to B.4's joint plus individual declarations.
STANDIN_BANDS = (
    ("under_10k", 0, 10_000),
    ("10k_20k", 10_000, 20_000),
    ("20k_30k", 20_000, 30_000),
    ("30k_40k", 30_000, 40_000),
    ("40k_50k", 40_000, 50_000),
    ("50k_75k", 50_000, 75_000),
    ("75k_100k", 75_000, 100_000),
    ("100k_plus", 100_000, None),
)
STANDIN_MEASUREMENT = "statbel_net_taxable_income_proxy_v1"
STANDIN_MONEY_COLUMN = f"{STANDIN_MEASUREMENT}_eur"
STANDIN_INCOME_COLUMN = f"{STANDIN_MEASUREMENT}_cents"
STANDIN_UNIVERSE_COLUMN = f"{STANDIN_MEASUREMENT}_universe_status"
STANDIN_TABLE = "statbel_a1_standin_bands_iy2023"
STANDIN_REASON = (
    "B4/W3: the d1952 stand-in measurement statbel_net_taxable_income_proxy_v1 "
    "(rule-computed net professional income per declaration on the S2-d "
    "tax_unit) has no producer, and legal pensions, part of professional "
    "income, are not modelled until d1902. The band cuts are income-year-2023 "
    "euros: ageing them to 2026 needs a shared operation that scales interval "
    "cuts by a declared annual ratio pair. scale_money_and_intervals needs a "
    "G8-r2 factor and singleton identity money rows, and G8-r2's resolver needs "
    "dated observations through the target period; neither fits FPB's annual "
    "2026 projections or summed bands, so reference_profile:aging_index stays "
    "deferred. The stand-in is never described as reproducing official "
    "taxable income."
)
B4_REASON = (
    "S1/S2-d: the declared tax-unit partition and the BE-MC4 legal-status proxy "
    "(partner pointers calibrated to these counts) are not built; joint and "
    "individual declaration indicators have no producer. B.4, like A.1, counts "
    "only declarations with positive total net taxable income (Statbel's 523,770 "
    "zero-income declarations of income year 2023 are outside it), so the "
    "indicators must count the same declared filing universe as the A.1 bands "
    "(d1952 build item 2), or the target overshoots by up to 7%."
)
DECLARATION_UNIVERSE = (
    "filed declarations with positive total net taxable income; zero-income "
    "declarations are outside A.1, B.1 and B.4"
)
B1_REASON = (
    "BE-MC5 activates B.1 only after its tax definition, income ranking and "
    "policy year are checked. All three checks fail or cannot be completed from "
    "the documents (reference_inventory.json b1_checks), so B.1 stays inactive "
    "and unaged at its source period."
)
RULES_REASON = (
    "W3: needs a pinned rulespec-be binding for {module} on the transported "
    "frame, the S5 pre-calibration allowlist entry for it, and B4's euro "
    "conversion of the donor inputs it reads."
)

# Declared ratio pairs from the FPB June 2026 outlook. Each pair is one
# published annex row at two periods; its numerator is FPB's 2026 projection
# and its denominator the observed value at the base fact's own period, so
# rolled_forward_by_ratio applies base * numerator / denominator. The reviewed
# lexical tokens are assertions of the feed, never arithmetic inputs. The
# series choice for each pair is a B3 proposal (reported as a finding).
AGEING_PAIRS = {
    "fpb_t07_total_population": {
        "row": "labour_market.levels.total_population",
        "operands": {2023: "11730600", 2026: "11896000"},
        "use": "calibration",
        "reason": "Declaration counts follow resident population (August's alignment); A.1 band counts and B.4 type counts share this pair so the bands still sum to B.4's joint plus individual total.",
    },
    "fpb_t11_total_resources": {
        "row": "household_account.total_resources",
        "operands": {2023: "518286000000", 2026: "571010000000"},
        "use": "calibration",
        "reason": "Net professional income mixes wages, replacement incomes, pensions and profits; the household account's total resources is the one FPB row spanning them (August's resource alignment).",
    },
    "fpb_t11_actual_social_contributions": {
        "row": "household_account.actual_social_contributions",
        "operands": {2024: "94613000000", 2026: "101300000000"},
        "use": "calibration",
        "reason": "Employee personal contributions follow FPB's actual social contributions; gross wages stay free for the wage-headline holdout.",
    },
    "fpb_t24_social_security_cash_unemployment": {
        "row": "social_benefits_detail.social_security_cash_unemployment",
        "operands": {2025: "5602000000", 2026: "4255000000"},
        "use": "calibration",
        "reason": "The target's own series, FPB T24 line a., rolled by its 2026/2025 step. unemployment_series_choice in the inventory compares it with the larger line 4.",
    },
    "fpb_t24_federal_cash_income_guarantee_for_older_people": {
        "row": "social_benefits_detail.federal_government_cash_income_guarantee_for_older_people",
        "operands": {2025: "1025000000", 2026: "1034000000"},
        "use": "calibration",
        "reason": "The target's own series (GRAPA).",
    },
    "fpb_t11_net_mixed_income": {
        "row": "household_account.net_mixed_income",
        "operands": {2025: "39834000000", 2026: "41044000000"},
        "use": "calibration",
        "reason": "The target's own series (BE-MC3 self-employment aggregate).",
    },
    "fpb_t17_direct_taxes_households": {
        "row": "general_government_account.direct_taxes_households",
        "operands": {2025: "77771000000", 2026: "81561000000"},
        "use": "holdout",
        "reason": "Government direct taxes on households, the FPB counterpart of D51A_C1.",
    },
    "fpb_t06_net_national_disposable_income": {
        "row": "national_income_account.net_national_disposable_income",
        "operands": {2024: "500699000000", 2026: "534156000000"},
        "use": "holdout",
        "reason": "FPB publishes no household disposable income; national net disposable income is the nearest disposable-income row (a proxy).",
    },
    "fpb_t11_social_benefits": {
        "row": "household_account.social_benefits",
        "operands": {2024: "124608000000", 2026: "132674000000"},
        "use": "holdout",
        "reason": "The household account row matching Eurostat D62 received.",
    },
    "fpb_t11_social_contributions": {
        "row": "household_account.social_contributions",
        "operands": {2024: "112346000000", 2026: "120094000000"},
        "use": "holdout",
        "reason": "The household account row matching Eurostat D61 paid.",
    },
    "fpb_t11_current_taxes_on_income_and_wealth": {
        "row": "household_account.current_taxes_on_income_and_wealth",
        "operands": {2024: "77033000000", 2026: "83113000000"},
        "use": "holdout",
        "reason": "The household account row matching Eurostat D5 paid.",
    },
    "fpb_t11_gross_wages_and_salaries": {
        "row": "household_account.gross_wages_and_salaries",
        "operands": {2025: "240536000000", 2026: "247221000000"},
        "use": "holdout",
        "reason": "The card's wage-headline comparator (FPB salaires bruts), rolled by its own 2026/2025 step.",
    },
}
# Observed FPB levels the inventory splits each pair at (2025 is the last
# observed year); reviewed like the pair operands.
FPB_OBSERVED_2025 = {
    "fpb_t07_total_population": "11850400",
    "fpb_t11_total_resources": "554117000000",
    "fpb_t11_actual_social_contributions": "98402000000",
    "fpb_t06_net_national_disposable_income": "519140000000",
    "fpb_t11_social_benefits": "128659000000",
    "fpb_t11_social_contributions": "116379000000",
    "fpb_t11_current_taxes_on_income_and_wealth": "79273000000",
}
# G8-r2 factors need operands with dated period coverage. FPB's annex facts
# carry none, so the only Belgian operands the resolver can authenticate are
# RSZ's annual wage totals: diagnostic cross-checks of the observed segment of
# the wage-driven pairs, never profile or money-normalization inputs.
# FPB T24 prints unemployment twice. The pair takes line a.; these reviewed 2025
# levels keep the inventory's comparison with line 4 tied to the feed.
T24_UNEMPLOYMENT_2025 = {
    "chosen": ("social_security_cash_unemployment", "5602000000"),
    "alternative": ("social_security_cash_unemployment_total", "6510000000"),
    "company_supplement": (
        "social_security_cash_unemployment_with_company_supplement",
        "167000000",
    ),
    "career_breaks": (
        "social_security_cash_career_breaks_and_time_credit",
        "740000000",
    ),
}
RSZ_SOURCE_SHA256 = "7f97f510d332a0966b1afba110ce448f4a4a83d9d58fd5c3bb4d8d36d1aeca97"
RSZ_OPERANDS = {2023: "160796591.5", 2024: "165739460", 2025: "171074878.6"}
RSZ_FACTORS = {
    "rsz_ordinary_remuneration_cy2023_to_cy2025_observed": (2023, 1063921),
    "rsz_ordinary_remuneration_cy2024_to_cy2025_observed": (2024, 1032192),
}
FACTOR_VALIDITY = {
    "declared_on": "2026-11-01",
    "valid_from": "2026-11-01",
    "valid_until": "2027-06-30",
    "as_of": "2026-11-01",
}
# Chronicle labels Statbel's regions NUTS_2024, while the typed geography layer
# names vintage:be_nuts1_2025 and its closed alias set admits no NUTS label.
# Each selector keeps Chronicle's label, and its row names the layer's typed
# authority, so an unreviewed label refuses at generation.
GEOGRAPHY_VINTAGES = {
    "nuts1": ("NUTS_2024", "vintage:be_nuts1_2025"),
    "commune": ("nis_2025", "vintage:be_nis_2025"),
}
TRANSPORTED_INPUTS = {"employment_income", "nonfarm_self_employment_income"}
PROVIDERS = {
    "statbel": "Statbel",
    "fpb": "Federal Planning Bureau",
    "onss": "National Social Security Office",
    "onem": "National Employment Office",
    "eurostat": "Eurostat",
    "nbb": "National Bank of Belgium",
}
CATEGORIES = {
    "statbel.demography": ("statbel", "Demography"),
    "statbel.geography_support": ("statbel", "Geography support"),
    "statbel.fiscal_income": ("statbel", "Fiscal income"),
    "statbel.income_tax": ("statbel", "Assessed income tax"),
    "onss.social_security": ("onss", "Social security contributions"),
    "onem.caseloads": ("onem", "Benefit caseloads"),
    "fpb.benefit_expenditure": ("fpb", "Benefit expenditure"),
    "fpb.self_employment": ("fpb", "Self-employment income"),
    "fpb.household_account": ("fpb", "Household account"),
    "eurostat.income_tax": ("eurostat", "Income tax revenue"),
    "eurostat.household_accounts": ("eurostat", "Household accounts"),
    "nbb.national_accounts": ("nbb", "National accounts"),
}
DEFERRED_JRC = (
    "jrc-euromod-be-baseline-statistics-2025 is outside the Chronicle be suite "
    "while d1822 (move it to the scorecard, its only consumer) is open; no "
    "reference selects JRC. August took two calibration rows from it: the "
    "unemployment recipient count, now read from ONEM/RVA's own 2024 count "
    "(a different concept, parked), and the RIS amount and recipient count, "
    "for which the suite holds no source, so no RIS reference exists."
)


def _token(value):
    if isinstance(value, LexicalFloat):
        return value.lexical_token
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Non-numeric fact value {value!r}")
    return repr(value) if isinstance(value, float) else str(value)


def _belgian(fact, communes=None):
    """Refuse any fact outside Belgium before a selector can name it."""

    geography = fact["geography"]
    level, area = geography["level"], geography["id"]
    if (
        (level == "country" and area == "BE")
        or (level == "nuts1" and area in REGIONS)
        or (
            level == "commune"
            and re.fullmatch(r"[1-9][0-9]{4}", area)
            and (communes is None or area in communes)
        )
    ):
        return fact
    raise ValueError(
        f"Belgian selectors read Belgian facts only; got {level} {area!r} "
        f"({fact['aggregate_fact_key']})."
    )


# _selector, Inventory and _typed_equal follow tools/generate_nz_target_references.py
# (copied, not imported: Belgium must not bind to NZ's generator module).
def _selector(fact, *, pin_key=True):
    _belgian(fact)
    measure, geography = fact["observed_measure"], fact["geography"]
    result = {
        "source_name": measure["source_name"],
        "source_measure_id": measure["source_measure_id"],
        "source_concept": measure["source_concept"],
        "entity_name": fact["entity"]["name"],
        "period_type": fact["period"]["type"],
        "period_value": fact["period"]["value"],
        "geography_level": geography["level"],
        "geography_id": geography["id"],
        "geography_vintage": geography["vintage"],
        "record_set_id": fact["layout"]["record_set_id"],
        "dimensions": list(fact["dimensions"]),
    }
    if fact["dimensions"]:
        result["dimension_values"] = dict(fact["dimensions"])
    if pin_key:
        result["aggregate_fact_key"] = fact["aggregate_fact_key"]
    return result


class Inventory:
    def __init__(self, facts):
        self.by_record_set, self.by_spec = {}, {}
        for fact in facts:
            layout = fact.get("layout", {})
            self.by_record_set.setdefault(layout.get("record_set_id"), []).append(fact)
            self.by_spec.setdefault(layout.get("record_set_spec_id"), []).append(fact)

    def cells(
        self,
        record_set=None,
        *,
        spec=None,
        measure=None,
        geography=None,
        period=None,
        dimensions=None,
    ):
        result = (
            self.by_record_set.get(record_set, [])
            if spec is None
            else self.by_spec.get(spec, [])
        )
        if measure is not None:
            result = [
                f
                for f in result
                if f["observed_measure"]["source_measure_id"] == measure
            ]
        if geography is not None:
            result = [f for f in result if f["geography"]["id"] == geography]
        if period is not None:
            result = [f for f in result if f["period"]["value"] == period]
        if dimensions is not None:
            result = [f for f in result if _typed_equal(f["dimensions"], dimensions)]
        return result

    def one(self, *args, **kwargs):
        matches = self.cells(*args, **kwargs)
        if len(matches) != 1:
            raise ValueError(
                f"Expected one source cell for {args!r} {kwargs!r}; found {len(matches)}"
            )
        return _belgian(matches[0])


def _typed_equal(actual, expected):
    return actual.keys() == expected.keys() and all(
        type(actual[k]) is type(v) and actual[k] == v for k, v in expected.items()
    )


def _eurostat(inventory, spec, period, **dimensions):
    matches = [
        fact
        for fact in inventory.cells(spec=spec, geography="BE", period=period)
        if all(
            fact["dimensions"].get(key) == value for key, value in dimensions.items()
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one Eurostat cell {spec} {period} {dimensions}")
    return _belgian(matches[0])


def _fpb(inventory, row, period):
    fact = inventory.one(
        f"{FPB}cy{period}.{row}", geography="BE", dimensions={}, period=period
    )
    if fact["observed_measure"]["source_name"] != "fpb_economic_outlook":
        raise ValueError(f"FPB row {row} is not an FPB outlook fact")
    return fact


def _pair(inventory, pair_id, base_period):
    """Return the reviewed numerator/denominator facts of one declared pair."""

    pair = AGEING_PAIRS[pair_id]
    if set(pair["operands"]) != {base_period, POLICY_YEAR}:
        raise ValueError(f"{pair_id} is declared for {sorted(pair['operands'])}")
    numerator = _fpb(inventory, pair["row"], POLICY_YEAR)
    denominator = _fpb(inventory, pair["row"], base_period)
    for fact, period, assertion in (
        (numerator, POLICY_YEAR, "source_projection"),
        (denominator, base_period, "observation"),
    ):
        if (
            _token(fact["value"]) != pair["operands"][period]
            or fact["assertion"] != assertion
        ):
            raise ValueError(f"Reviewed FPB operand {pair_id} {period} changed")
    if (
        numerator["layout"]["record_set_spec_id"]
        != denominator["layout"]["record_set_spec_id"]
        or numerator["layout"]["source_row_id"]
        != denominator["layout"]["source_row_id"]
        or numerator["observed_measure"]["source_concept"]
        != denominator["observed_measure"]["source_concept"]
    ):
        raise ValueError(f"{pair_id} numerator and denominator are not one row")
    return numerator, denominator


def _operand(role, fact):
    selector = _selector(fact)
    selector["assertion"] = fact["assertion"]
    return {"role": role, **selector}


def _roll(reference, pair_id, numerator, denominator, *, base_fact=None, members=None):
    base = {"role": "base"}
    selector = reference["ledger_selector"]
    selector["assertion"] = "observation"
    if members is not None:
        dimension, values = members
        base["expected_member_count"] = len(values)
        selector.pop("aggregate_fact_key")
        selector["dimension_values"][dimension] = list(values)
    reference.update(
        {
            "value_operation": "rolled_forward_by_ratio",
            "value_operands": [
                base,
                _operand("numerator", numerator),
                _operand("denominator", denominator),
            ],
            "period": POLICY_YEAR,
            "period_match_policy": "latest_not_after",
            "assertion_policy": "allow_source_projection",
        }
    )
    if base_fact is not None and _value(base_fact) == _value(denominator):
        # A base equal to the denominator makes the target FPB's 2026
        # projection of the row: declared, not hidden.
        reference["metadata"]["ageing_target_equals"] = (
            "FPB's 2026 projection of the base's own row"
            if base_fact["aggregate_fact_key"] == denominator["aggregate_fact_key"]
            else "FPB's 2026 projection of the pair's row: the base equals its denominator"
        )
    reference["metadata"].update(
        {
            "ageing": "rolled_forward_by_ratio",
            "ageing_pair": pair_id,
            "ageing_source": "FPB Economic Outlook 2026-2031, June 2026 annex",
            "source_period": str(denominator["period"]["value"]),
            "target_period": str(POLICY_YEAR),
        }
    )


def _vintage_authority(fact):
    """Return a subnational fact's typed authority; refuse an unreviewed label."""

    geography = fact["geography"]
    if geography["level"] == "country":
        return None
    vintage, authority = GEOGRAPHY_VINTAGES[geography["level"]]
    if geography["vintage"] != vintage:
        raise ValueError(
            f"{geography['level']} fact carries vintage {geography['vintage']!r}; "
            f"the reviewed mapping names {vintage!r}."
        )
    return authority


def _measure_kind(measure):
    if measure.startswith("belgium_"):
        return "rules_output"
    return "transported_input" if measure in TRANSPORTED_INPUTS else "prepared_column"


def _reference(
    fact,
    *,
    name,
    contract,
    category,
    entity,
    measure,
    family,
    reference_set,
    status,
    method_card,
    measure_kind,
    reason=None,
    filter_column=None,
    period=None,
    tolerance=None,
):
    provider = CATEGORIES[category][0]
    role = {
        "calibration": "calibration",
        "precal": "pre_calibration",
        "holdout": "validation",
    }[reference_set]
    result = {
        "name": name,
        "ledger_selector": _selector(fact),
        "entity": entity,
        "measure": measure,
        "period": fact["period"]["value"] if period is None else period,
        "family": family,
        "category_id": category,
        "assertion_policy": "observed_only",
        "metadata": {
            "contract_target_id": contract,
            "activation_status": status,
            "criticality": "release_blocking"
            if reference_set != "holdout"
            else "diagnostic",
            "measure_kind": measure_kind,
            "publisher": PROVIDERS[provider],
            "target_role": role,
            "reference_set": reference_set,
            "method_card_row": method_card,
            "method_card_status": METHOD_CARD_STATUS,
        },
    }
    authority = _vintage_authority(fact)
    if authority is not None:
        result["metadata"].update(
            geography_vintage=fact["geography"]["vintage"],
            geography_vintage_authority=authority,
        )
    if tolerance is not None:
        result["metadata"]["fit_tolerance"] = tolerance
    if filter_column is not None:
        result["filter"] = filter_column
    if reason:
        result["metadata"]["parking_reason"] = reason
        result["notes"] = reason
    return result


def _hierarchy(rows, labels):
    used = sorted({row["category_id"] for row in rows})
    return {
        "providers": {
            provider: {"label": PROVIDERS[provider]}
            for provider in sorted({CATEGORIES[category][0] for category in used})
        },
        "categories": {
            category: {
                "provider_id": CATEGORIES[category][0],
                "label": CATEGORIES[category][1],
            }
            for category in used
        },
        "target_labels": {
            contract: labels[contract]
            for contract in sorted(
                {row["metadata"]["contract_target_id"] for row in rows}
            )
        },
    }


def _a1_classes(inventory):
    """Read A.1's EUR 1,000 classes and their bounds from Chronicle's constraints."""

    facts = inventory.cells(
        A1.format(component="taxable_income"), measure="declarations", geography="BE"
    )
    classes = []
    for fact in facts:
        code = fact["dimensions"][A1_CLASS]
        if code == "all":
            continue
        bounds = {"lower": None, "upper": None}
        for row in fact["universe_constraints"]["constraints"]:
            if row["variable"] != "tax_return.net_taxable_income":
                continue
            if row.get("unit") != "eur":
                raise ValueError(f"A.1 class {code} bound is not in euros")
            if row["operator"] in {">", ">="}:
                bounds["lower"] = (row["value"], row["operator"])
            elif row["operator"] == "<":
                bounds["upper"] = row["value"]
            else:
                raise ValueError(f"A.1 class {code} has operator {row['operator']}")
        classes.append((bounds["lower"], bounds["upper"], code))
    classes.sort(key=lambda row: row[0][0])
    previous = None
    for index, ((lower, operator), upper, code) in enumerate(classes):
        if index == 0:
            if (lower, operator) != (0, ">"):
                raise ValueError("A.1's first class must exclude zero income")
        elif (lower, operator) != (previous, ">="):
            raise ValueError(f"A.1 class {code} leaves a gap or overlap")
        previous = upper
    if classes[-1][1] is not None or len(classes) != 101:
        raise ValueError("A.1 must hold 101 classes ending in an open top class")
    return [(lower, upper, code) for (lower, _), upper, code in classes]


def _band_members(classes):
    members = {}
    ends = {code: end for _, end, code in classes}
    for band, lower, upper in STANDIN_BANDS:
        inside = [
            code
            for start, end, code in classes
            if start >= lower and (upper is None or (end is not None and end <= upper))
        ]
        edges = [start for start, _, code in classes if code in inside]
        if not inside or edges[0] != lower or ends[inside[-1]] != upper:
            raise ValueError(f"Band {band} does not sit on A.1 class edges")
        members[band] = inside
    flat = [code for band in members for code in members[band]]
    if flat != [code for _, _, code in classes]:
        raise ValueError("Stand-in bands must partition A.1's positive classes")
    return members


def calibration_references(inventory, pairs, members):
    rows, labels = [], {}
    for region in REGIONS:
        for sex in SEXES:
            for band in AGE_BANDS:
                fact = inventory.one(
                    STATBEL_POPULATION.format(year=POLICY_YEAR),
                    geography=region,
                    dimensions={"person.age_band": band, "person.sex": sex},
                )
                code = region.lower()
                rows.append(
                    _reference(
                        fact,
                        name=f"statbel_population_by_region_sex_age__{code}__{sex}__a{band}",
                        contract="statbel_population_by_region_sex_age",
                        category="statbel.demography",
                        entity="person",
                        measure="people",
                        family="demography",
                        reference_set="calibration",
                        status="active",
                        method_card="BE-MC6",
                        measure_kind="prepared_column",
                        filter_column=f"population_region_{code}_{sex}_age_{band}_indicator",
                        tolerance="0.02",
                    )
                )
                rows[-1]["period_match_policy"] = "exact"
                rows[-1]["metadata"].update(
                    {
                        "ageing": "none: observation for the target year",
                        "source_snapshot": "1 January 2026 resident population",
                        "target_period": str(POLICY_YEAR),
                    }
                )
    labels["statbel_population_by_region_sex_age"] = (
        "Statbel resident population on 1 January 2026 by region, sex and age band"
    )
    count_num, count_den = pairs["fpb_t07_total_population"]
    money_num, money_den = pairs["fpb_t11_total_resources"]
    for band, lower, upper in STANDIN_BANDS:
        anchor = inventory.one(
            A1.format(component="taxable_income"),
            measure="declarations",
            geography="BE",
            dimensions={A1_CLASS: members[band][0]},
        )
        money_anchor = inventory.one(
            A1.format(component="taxable_income"),
            measure="taxable_income_eur",
            geography="BE",
            dimensions={A1_CLASS: members[band][0]},
        )
        for kind, fact, contract, measure, numerator, denominator, pair_id in (
            (
                "count",
                anchor,
                "statbel_a1_standin_declarations_by_band",
                "tax_units",
                count_num,
                count_den,
                "fpb_t07_total_population",
            ),
            (
                "money",
                money_anchor,
                "statbel_a1_standin_taxable_income_by_band",
                STANDIN_MONEY_COLUMN,
                money_num,
                money_den,
                "fpb_t11_total_resources",
            ),
        ):
            row = _reference(
                fact,
                name=f"{contract.removesuffix('_by_band')}__{band}",
                contract=contract,
                category="statbel.fiscal_income",
                entity="tax_unit",
                measure=measure,
                family="fiscal_income",
                reference_set="calibration",
                status="requires_prerequisites",
                method_card="BE-MC5",
                measure_kind="prepared_column",
                reason=STANDIN_REASON,
                filter_column=f"statbel_a1_standin_band_{band}_indicator",
                tolerance="0.05",
            )
            _roll(
                row, pair_id, numerator, denominator, members=(A1_CLASS, members[band])
            )
            row["metadata"].update(
                {
                    "reference_value_kind": kind,
                    "required_reference_profile": "aging_index",
                    "required_measurement_binding": STANDIN_MEASUREMENT,
                    "reference_prerequisites": json.dumps(
                        [
                            "reference_profile:aging_index",
                            f"measurement:{STANDIN_MEASUREMENT}",
                        ],
                        separators=(",", ":"),
                    ),
                    "interval_table_id": STANDIN_TABLE,
                    "declaration_universe": DECLARATION_UNIVERSE,
                    "standin_band": band,
                    "standin_band_lower_eur_iy2023": str(lower),
                    "standin_band_upper_eur_iy2023": "open"
                    if upper is None
                    else str(upper),
                }
            )
            rows.append(row)
    labels["statbel_a1_standin_declarations_by_band"] = (
        "Stand-in income (d1952): Statbel A.1 declarations by broad band of total "
        "net taxable income, rolled to 2026 (parked)"
    )
    labels["statbel_a1_standin_taxable_income_by_band"] = (
        "Stand-in income (d1952): Statbel A.1 total net taxable income by broad "
        "band, rolled to 2026 (parked)"
    )
    for declaration_type in ("joint", "individual"):
        fact = inventory.one(
            spec=B4,
            measure="declarations",
            geography="BE",
            dimensions={
                "tax_return.age_band": "all",
                "tax_return.decile": 1,
                "tax_return.declaration_type": declaration_type,
            },
        )
        row = _reference(
            fact,
            name=f"statbel_b4_declarations__{declaration_type}",
            contract="statbel_b4_declarations_by_type",
            category="statbel.fiscal_income",
            entity="tax_unit",
            measure="tax_units",
            family="fiscal_income",
            reference_set="calibration",
            status="requires_declared_tax_units",
            method_card="BE-MC4",
            measure_kind="prepared_column",
            reason=B4_REASON,
            filter_column=f"{declaration_type}_declaration_indicator",
            tolerance="0.05",
        )
        _roll(
            row,
            "fpb_t07_total_population",
            count_num,
            count_den,
            members=("tax_return.decile", list(range(1, 11))),
        )
        row["metadata"].update(
            reference_value_kind="count", declaration_universe=DECLARATION_UNIVERSE
        )
        rows.append(row)
    labels["statbel_b4_declarations_by_type"] = (
        "Statbel B.4 joint and individual declarations (all ages, deciles 1-10 "
        "summed), rolled to 2026 (parked)"
    )
    for group in B1_GROUPS:
        fact = inventory.one(
            spec=B1_VALUES if group.startswith("decile") else B1_PERCENTILES,
            measure="total_tax_eur",
            geography="BE",
            dimensions={"tax_return.rank_group": group},
        )
        row = _reference(
            fact,
            name=f"statbel_b1_enrolled_tax__{group}",
            contract="statbel_b1_enrolled_tax_by_rank_group",
            category="statbel.income_tax",
            entity="tax_unit",
            measure="statbel_b1_enrolled_tax_proxy_v1_eur",
            family="income_tax",
            reference_set="calibration",
            status="inactive_pending_b1_checks",
            method_card="BE-MC5",
            measure_kind="prepared_column",
            reason=B1_REASON,
            filter_column=f"statbel_b1_rank_group_{group}_indicator",
            tolerance="0.05",
        )
        row["signed"] = True
        row["period_match_policy"] = "exact"
        row["metadata"].update(
            {
                "ageing": "none declared: B.1 checks failed",
                "reference_value_kind": "money",
                "source_period": "2023",
            }
        )
        rows.append(row)
    labels["statbel_b1_enrolled_tax_by_rank_group"] = (
        "Statbel B.1 total enrolled tax by decile 1-9 and top percentiles 91-100, "
        "income year 2023 (inactive: checks failed)"
    )
    onss = inventory.one(
        spec="onss.contributions.worker_personal_contributions.country.v1",
        geography="BE",
    )
    row = _reference(
        onss,
        name="onss_worker_personal_contributions",
        contract="onss_worker_personal_contributions",
        category="onss.social_security",
        entity="person",
        measure="belgium_worker_article_17_uncapped_component_contribution",
        family="social_security",
        reference_set="calibration",
        status="requires_rules_binding",
        method_card="BE-MC6",
        measure_kind="rules_output",
        reason=RULES_REASON.format(
            module="be/statutes/social_security/workers/contribution_rates.yaml"
        ),
        tolerance="0.05",
    )
    _roll(
        row,
        "fpb_t11_actual_social_contributions",
        *pairs["fpb_t11_actual_social_contributions"],
        base_fact=onss,
    )
    rows.append(row)
    labels["onss_worker_personal_contributions"] = (
        "ONSS/RSZ worker-borne personal contributions, 2024, rolled to 2026 (parked)"
    )
    onem = inventory.one(
        spec="onem_rva.unemployment.complete_unemployment.country.v1", geography="BE"
    )
    rows.append(
        _reference(
            onem,
            name="onem_complete_unemployment_recipients",
            contract="onem_complete_unemployment_recipients",
            category="onem.caseloads",
            entity="person",
            measure="receives_unemployment_benefit",
            family="caseloads",
            reference_set="calibration",
            status="requires_ageing_ruling_and_receipt_binding",
            method_card="BE-MC6",
            measure_kind="receipt_flag",
            reason=(
                "No declared ageing pair: FPB publishes jobseeker levels (BFP "
                "unemployment, +0.6% 2026/2025), not benefit recipients, and its "
                "unemployment benefit spending falls 24% in 2026, so the two "
                "series point opposite ways and neither ages a recipient count. "
                "Kept unaged at its 2024 source period. Also needs W3's receipt "
                "contract for unemployment."
            ),
            tolerance="0.05",
        )
    )
    rows[-1]["period_match_policy"] = "exact"
    rows[-1]["metadata"]["ageing"] = "none declared: no recipient series"
    labels["onem_complete_unemployment_recipients"] = (
        "ONEM/RVA complete-unemployment benefit recipients, 2024 monthly average "
        "(parked, unaged)"
    )
    for (
        name,
        pair_id,
        category,
        entity,
        measure,
        family,
        status,
        method_card,
        reason,
        label,
    ) in (
        (
            "fpb_unemployment_benefit_expenditure",
            "fpb_t24_social_security_cash_unemployment",
            "fpb.benefit_expenditure",
            "household",
            "belgium_unemployment_ordinary_annual_payable_amount",
            "benefit_expenditure",
            "requires_receipt_contract_and_rules_binding",
            "BE-MC6",
            RULES_REASON.format(
                module="be/regulations/unemployment/pilot_oracle_pipeline.yaml"
            )
            + " Unemployment history inputs and W3's receipt contract are also missing.",
            "FPB social-security unemployment benefits, 2025, rolled to 2026 (parked)",
        ),
        (
            "fpb_grapa_expenditure",
            "fpb_t24_federal_cash_income_guarantee_for_older_people",
            "fpb.benefit_expenditure",
            "person",
            "belgium_grapa_income_guarantee_for_elderly_annual_amount",
            "benefit_expenditure",
            "requires_pension_route_d1902",
            "BE-MC6",
            "GRAPA is means-tested on pension income, and legal pensions are not "
            "modelled until d1902 settles BE-MC3's pension route. "
            + RULES_REASON.format(
                module="be/statutes/income_guarantee_for_elderly/payable_amount.yaml"
            ),
            "FPB income guarantee for older people (GRAPA), 2025, rolled to 2026 (parked)",
        ),
        (
            "fpb_net_mixed_income",
            "fpb_t11_net_mixed_income",
            "fpb.self_employment",
            "person",
            "nonfarm_self_employment_income",
            "self_employment",
            "requires_currency_bridge_b4",
            "BE-MC3",
            "B4: BE-MC3 gives self-employment income FX and uprating only, then "
            "calibrates its aggregate; the euro column has no producer until "
            "B4's currency bridge. FPB's net mixed income also covers farm "
            "income, which the donor carries separately.",
            "FPB household net mixed income, 2025, rolled to 2026 (parked)",
        ),
    ):
        numerator, denominator = pairs[pair_id]
        fact = denominator
        row = _reference(
            fact,
            name=name,
            contract=name,
            category=category,
            entity=entity,
            measure=measure,
            family=family,
            reference_set="calibration",
            status=status,
            method_card=method_card,
            measure_kind=_measure_kind(measure),
            reason=reason,
            tolerance="0.05",
        )
        _roll(row, pair_id, numerator, denominator, base_fact=fact)
        rows.append(row)
        labels[name] = label
    return rows, labels


def precal_references(communes):
    rows = []
    for code in sorted(communes):
        fact = _belgian(communes[code], communes)
        row = _reference(
            fact,
            name=f"statbel_population_by_commune__nis{code}",
            contract="statbel_population_by_commune",
            category="statbel.geography_support",
            entity="person",
            measure="people",
            family="geography_support",
            reference_set="precal",
            status="active",
            method_card="geography_spine",
            measure_kind="prepared_column",
        )
        row["metadata"].update(
            {
                "precal_use": "atomic_geography_support",
                "ageing": "none: support shares at 1 January 2025",
            }
        )
        rows.append(row)
    labels = {
        "statbel_population_by_commune": "Statbel resident population by NIS 2025 commune, 1 January 2025 (atomic geography support)",
    }
    return rows, labels


def holdout_references(inventory, pairs):
    rows, labels = [], {}
    d51 = _eurostat(
        inventory,
        "eurostat.gov_10a_taxag.income_tax_revenue_by_taxpayer.country.v1",
        2025,
        na_item="D51A_C1",
        sector="S13",
    )
    nbb = inventory.one(
        spec="nbb.national_accounts.household_disposable_income.country.v1",
        geography="BE",
    )
    wages = pairs["fpb_t11_gross_wages_and_salaries"][1]

    def nasa(item, direction):
        return _eurostat(
            inventory,
            "eurostat.nasa_10_nf_tr.household_non_financial_transactions.direction.country.v1",
            2024,
            na_item=item,
            direct=direction,
            sector="S14_S15",
        )

    declarations = (
        (
            d51,
            "eurostat_d51a_c1_individual_income_tax",
            "fpb_t17_direct_taxes_households",
            "eurostat.income_tax",
            "person",
            "belgium_pit_federal_and_local_tax_before_withholding",
            "income_tax",
            "Eurostat D51A_C1 taxes on individual or household income, 2025, rolled to 2026",
            "Accrual national-accounts revenue held out against assessed tax (BE-MC5). "
            + RULES_REASON.format(
                module="be/statutes/income_tax/individual/final_tax.yaml"
            ),
        ),
        (
            nbb,
            "nbb_household_gross_disposable_income",
            "fpb_t06_net_national_disposable_income",
            "nbb.national_accounts",
            "household",
            "household_gross_disposable_income_eur",
            "national_accounts",
            "NBB household gross disposable income (B6G), 2024, rolled to 2026",
            "No producer: rulespec-be's disposable-income policy is a narrow worker bridge, not integrated disposable income.",
        ),
        (
            nasa("B6G", "RECV"),
            "eurostat_household_gross_disposable_income",
            "fpb_t06_net_national_disposable_income",
            "eurostat.household_accounts",
            "household",
            "household_gross_disposable_income_eur",
            "household_accounts",
            "Eurostat household and NPISH gross disposable income (B6G), 2024, rolled to 2026",
            "No producer, as for NBB's B6G. Eurostat's S14_S15 sector adds non-profit institutions to NBB's household figure.",
        ),
        (
            nasa("D62", "RECV"),
            "eurostat_household_social_benefits_received",
            "fpb_t11_social_benefits",
            "eurostat.household_accounts",
            "household",
            "household_social_benefits_received_eur",
            "household_accounts",
            "Eurostat household social benefits received (D62), 2024, rolled to 2026",
            "No producer: pensions (d1902), child benefits (not modelled at level) and unemployment receipts (W3) are missing.",
        ),
        (
            nasa("D61", "PAID"),
            "eurostat_household_social_contributions_paid",
            "fpb_t11_social_contributions",
            "eurostat.household_accounts",
            "household",
            "household_social_contributions_paid_eur",
            "household_accounts",
            "Eurostat household social contributions paid (D61), 2024, rolled to 2026",
            "No producer: employer and self-employed contributions are not bound (W3).",
        ),
        (
            nasa("D5", "PAID"),
            "eurostat_household_current_taxes_paid",
            "fpb_t11_current_taxes_on_income_and_wealth",
            "eurostat.household_accounts",
            "household",
            "household_current_taxes_on_income_paid_eur",
            "household_accounts",
            "Eurostat household current taxes on income and wealth paid (D5), 2024, rolled to 2026",
            "No producer: needs the PIT binding (W3) aggregated to households.",
        ),
        (
            wages,
            "fpb_gross_wages_and_salaries",
            "fpb_t11_gross_wages_and_salaries",
            "fpb.household_account",
            "person",
            "employment_income",
            "household_accounts",
            "FPB gross wages and salaries (salaires bruts), 2025, rolled to 2026",
            "The card's wage-headline comparator (BE-MC5): gross wages, not compensation including employer contributions. The euro wage column has no producer until B4's currency bridge.",
        ),
    )
    for (
        fact,
        name,
        pair_id,
        category,
        entity,
        measure,
        family,
        label,
        reason,
    ) in declarations:
        row = _reference(
            fact,
            name=name,
            contract=name,
            category=category,
            entity=entity,
            measure=measure,
            family=family,
            reference_set="holdout",
            status="requires_holdout_measure",
            method_card="BE-MC5",
            measure_kind=_measure_kind(measure),
            reason="Held out of calibration; compared, never fitted. " + reason,
        )
        _roll(row, pair_id, *pairs[pair_id], base_fact=fact)
        row["metadata"]["holdout_use"] = "validation_comparator"
        rows.append(row)
        labels[name] = label
    return rows, labels


def _interval_document(classes, members):
    bands = []
    bands.append(
        {
            "id": "nil",
            "kind": "nil",
            "indicator": "statbel_a1_standin_nil_indicator",
            "source_label": "zero total net taxable income (outside A.1)",
        }
    )
    edges = {code: (lower, upper) for lower, upper, code in classes}
    for band, lower, upper in STANDIN_BANDS:
        if edges[members[band][0]][0] != lower or (
            edges[members[band][-1]][1] != upper
        ):
            raise ValueError(f"Band {band} edges differ from its member classes")
        bands.append(
            {
                "id": band,
                "kind": "positive",
                "upper_cent": None if upper is None else upper * 100,
                "indicator": f"statbel_a1_standin_band_{band}_indicator",
                "source_label": f"A.1 classes {members[band][0]} to {members[band][-1]}",
            }
        )
    return {
        "schema_version": 1,
        "country": "be",
        "description": "Generated by tools/generate_be_target_references.py from the class bounds Chronicle publishes on Statbel A.1 (universe constraints), checked to tile A.1's positive range without gaps.",
        "disclosure": (
            "Statbel's labels ('Minder dan 1', '1-2', '100 en meer') do not state a "
            "closure; Chronicle's A.1 universe constraints do: 0 < t < 1,000, then "
            "lower <= t < upper, and t >= 100,000 for the open top class. The "
            "table follows them: left-closed positive bands, a nil band for "
            "t == 0 (Statbel counts zero-income declarations outside A.1, and "
            "no reference fits the nil band, because the proxy's zeros are not "
            "Statbel's zeros), negative proxy income floored to zero, integer "
            "cents. Cuts are income-year-2023 euros and are not aged: no shared "
            "operation scales them by a declared annual ratio pair, so the A.1 "
            "references stay parked (BE-MC5, d1952; proposed pending d1902)."
        ),
        "tables": [
            {
                "id": STANDIN_TABLE,
                "entity": "tax_unit",
                "currency": "EUR",
                "input_unit": "integer_cent",
                "income_column": STANDIN_INCOME_COLUMN,
                "universe_column": STANDIN_UNIVERSE_COLUMN,
                "source_period": 2023,
                "target_period": POLICY_YEAR,
                "positive_band_closure": "left",
                "nil_band": "present",
                "negative_income_policy": "floor_zero",
                "first_lower_cut_cents": 0,
                "source_locator": "Chronicle statbel.fiscal_income_distribution.a1.by_income_class_eur1000.v1, Belgium, taxable income classes",
                "description": "The eight d1952 stand-in bands of total net taxable income, income year 2023, each an exact sum of A.1 EUR 1,000 classes.",
                "diagnostics": {
                    "boundary_window_cents": 100,
                    "alternative_positive_band_closure": "right",
                },
                "bands": bands,
            }
        ],
    }


def _reference_profiles():
    return {
        "schema_version": 1,
        "factor_context_resource": "reference_aging.json",
        "interval_resource": "reference_intervals.json",
        "dimensions": {
            "aging_index": {
                "default": "central",
                "profiles": {
                    "central": {
                        "deferred": True,
                        "notes": (
                            "Declared but not executable: ageing the A.1 stand-in "
                            "band cuts to 2026 needs a shared operation that scales "
                            "interval cuts by a declared annual ratio pair, and "
                            "reference_aging.json holds no profile_input factor for "
                            "it. The profile is deferred, so the resolver applies "
                            "no transformation and the A.1 rows keep both "
                            "prerequisites."
                        ),
                    }
                },
            }
        },
    }


def _factor_validation(fact):
    return {
        "unit": fact["observed_measure"]["unit"],
        "source_sha256": fact["source"]["source_sha256"],
        "source_vintage": fact["source"]["vintage"],
        "entity_role": fact["entity"]["role"],
        "provenance_class": fact["provenance_class"],
        "assertion": fact["assertion"],
        "period_coverage": {
            key: fact["period_coverage"][key] for key in ("start_date", "end_date")
        },
        "source_column_id": fact["layout"]["source_column_id"],
    }


def _rsz(inventory, year):
    fact = inventory.one(
        f"rsz.annual_wages.year{year}.ordinary_remuneration",
        geography="BE",
        dimensions={"rsz.annual_worker_type_sex": "total"},
    )
    if (
        _token(fact["value"]) != RSZ_OPERANDS[year]
        or fact["source"]["source_sha256"] != RSZ_SOURCE_SHA256
        or fact["assertion"] != "observation"
    ):
        raise ValueError(f"Reviewed RSZ operand changed for {year}")
    return fact


def build_aging_declarations(inventory, feed_identity):
    """Declare the G8-r2 factors whose operands carry dated coverage."""

    declarations = []
    target = _rsz(inventory, 2025)
    for factor_id, (source_year, expected) in RSZ_FACTORS.items():
        source = _rsz(inventory, source_year)

        def slot(fact, *, is_target=False):
            row = {
                "selector": {**_selector(fact), "assertion": "observation"},
                "validation": _factor_validation(fact),
                "weight": "1",
                "sequence": 0,
            }
            if is_target:
                row["projection_offset"] = 0
            return row

        declarations.append(
            {
                "schema_version": 1,
                "id": factor_id,
                "operation": "projected_period_average_ratio",
                "operation_version": 1,
                "purpose": "diagnostic",
                "feed_identity": copy.deepcopy(feed_identity),
                "source_period": source_year,
                "target_period": 2025,
                "source_schedule": [slot(source)],
                "target_schedule": [slot(target, is_target=True)],
                "growth": None,
                "subperiods_per_year": 1,
                "projection_policy": "observed_preferred_latest_contiguous",
                "rounding": {"digits": 6, "mode": "nearest_half_even"},
                "expected_operational_factor": {
                    "numerator": expected,
                    "denominator": 1000000,
                },
                "validity": dict(FACTOR_VALIDITY),
                "evidence": [
                    "RSZ/ONSS annual wages key figures (Kerncijfers), ordinary "
                    "remuneration, all employees, calendar years; pre-publication "
                    "raw data, 2024-2025 not final",
                    "Diagnostic cross-check of the observed segment of FPB "
                    "wage-driven ratio pairs; never a profile or money "
                    "normalization input",
                ],
                "decision_references": [
                    "BE-MC0 (proposed pending d1902): targets rolled to 2026 by FPB June 2026 ratio pairs",
                    "G8-r2 factor resolver: projected_period_average_ratio@1",
                ],
            }
        )
    return declarations


def build_reference_aging(inventory, feed_identity):
    from microcosm.build.reference_factors import resolve_reference_factor

    factors = []
    for declaration in build_aging_declarations(inventory, feed_identity):
        factor, receipt = resolve_reference_factor(
            declaration,
            [fact for rows in inventory.by_spec.values() for fact in rows],
            feed_identity,
        )
        factors.append(
            {
                "declaration": declaration,
                "operational_factor": factor,
                "receipt": receipt,
            }
        )
    return {"schema_version": 1, "feed_identity": feed_identity, "factors": factors}


def _value(fact):
    return Fraction(_token(fact["value"]))


def _share(part, whole):
    return float(part / whole)


def _standin_gaps(inventory, members):
    def component(name, measure, codes):
        return sum(
            (
                _value(
                    inventory.one(
                        A1.format(component=name),
                        measure=measure,
                        geography="BE",
                        dimensions={A1_CLASS: code},
                    )
                )
                for code in codes
            ),
            Fraction(),
        )

    def row(codes, label):
        total = component("taxable_income", "taxable_income_eur", codes)
        declarations = component("taxable_income", "declarations", codes)
        shares = {
            name: _share(component(name, measure, codes), total)
            for name, measure, _ in A1_COMPONENTS
        }
        identity = component("professional_income", "professional_income_eur", codes)
        for name, measure, role in A1_COMPONENTS[1:]:
            value = component(name, measure, codes)
            identity += -value if role == "gap_subtracted" else value
        return {
            "band": label,
            "declarations_iy2023": float(declarations),
            "taxable_income_eur_iy2023": float(total),
            "component_shares_of_taxable_income": shares,
            "omitted_share_gross": sum(
                shares[name] for name, _, role in A1_COMPONENTS if role != "in"
            ),
            "omitted_share_net": 1 - shares["professional_income"],
            "identity_residual_eur": float(identity - total),
        }

    national = row(["all"], "all")
    bands = [row(members[band], band) for band, _, _ in STANDIN_BANDS]
    professional = _value(
        inventory.one(
            A1.format(component="professional_income"),
            measure="declarations",
            geography="BE",
            dimensions={A1_CLASS: "all"},
        )
    )
    zero = _value(
        inventory.one(
            "statbel.fiscal_income_distribution.income_year2023.zero_income_declarations",
            geography="BE",
        )
    )
    return {
        "measure": STANDIN_MEASUREMENT,
        "unit": "tax declaration (S2-d tax_unit); joint declarations count once",
        "in_measure": "net professional income, rule-computed from transported gross amounts",
        "published_gaps": "Components of Statbel's total net taxable income left out of the stand-in, as shares of it (method card BE-MC5; d1952). Within bands they can be larger than nationally.",
        "national": national,
        "bands": bands,
        "declarations_with_taxable_but_no_professional_income": float(
            Fraction(national["declarations_iy2023"]) - professional
        ),
        "zero_income_declarations_outside_a1": float(zero),
        "zero_handling": "The proxy's zeros (no professional income, or losses floored to zero) are not Statbel's zero-income declarations; the nil band is not fitted.",
        "stress_test": "Declared, not executable in B3: rerun with unemployment and sickness benefits left out of the measure (d1952); a measurement variant owed by B4/W3.",
        "pension_gap": "Legal pensions are part of net professional income but are not modelled until d1902 settles BE-MC3's pension route, so the proxy cannot yet represent pensioner declarations.",
    }


def _b1_checks(inventory):
    """Run the three BE-MC5 checks the artifact allows; B.1 activates only if all pass."""

    def b1(spec, measure, group):
        return _value(
            inventory.one(
                spec=spec,
                measure=measure,
                geography="BE",
                dimensions={"tax_return.rank_group": group},
            )
        )

    total_tax = b1(B1_VALUES, "total_tax_eur", "all")
    spf = _value(
        inventory.one(
            spec="spf_finances.pit.tax_before_withholding.country.v1", geography="BE"
        )
    )
    deciles = [f"decile_{decile:02d}" for decile in range(1, 11)]
    decile_tax = sum((b1(B1_VALUES, "total_tax_eur", g) for g in deciles), Fraction())
    percentile_tax = sum(
        (
            b1(B1_PERCENTILES, "total_tax_eur", f"percentile_{p:03d}")
            for p in range(91, 101)
        ),
        Fraction(),
    )
    negative = [g for g in deciles if b1(B1_VALUES, "total_tax_eur", g) < 0]
    bounds = [b1(B1_BOUNDS, "upper_bound_eur", f"decile_{d:02d}") for d in range(1, 10)]
    top_bounds = [
        b1(B1_PERCENTILE_BOUNDS, "upper_bound_eur", f"percentile_{p:03d}")
        for p in range(91, 100)
    ]
    b1_income = b1(B1_VALUES, "taxable_income_eur", "all")
    a1_income = _value(
        inventory.one(
            A1.format(component="taxable_income"),
            measure="taxable_income_eur",
            geography="BE",
            dimensions={A1_CLASS: "all"},
        )
    )
    shares = sum(
        (b1(B1_VALUES, "taxable_income_share", g) for g in deciles), Fraction()
    )
    decile_10 = b1(B1_VALUES, "total_tax_eur", "decile_10")
    note = inventory.one(
        spec=B1_VALUES,
        measure="total_tax_eur",
        geography="BE",
        dimensions={"tax_return.rank_group": "all"},
    )["concept_alignment"]["evidence_notes"]
    cent = Fraction(1, 10)
    asserted = {
        "b1_total_matches_spf_total": abs(total_tax - spf) <= cent,
        "b1_deciles_sum_to_total": abs(decile_tax - total_tax) <= cent,
        "top_percentiles_sum_to_decile_10": abs(percentile_tax - decile_10) <= cent,
        "decile_bounds_strictly_increase": all(
            a < b for a, b in zip(bounds, bounds[1:], strict=False)
        ),
        "percentile_bounds_strictly_increase": all(
            a < b for a, b in zip(top_bounds, top_bounds[1:], strict=False)
        ),
        "decile_9_bound_below_percentile_91_bound": bounds[-1] < top_bounds[0],
        "b1_and_a1_taxable_income_totals_match": abs(b1_income - a1_income) <= cent,
        "chronicle_states_zero_income_exclusion": (
            "excludes declarations with zero total net taxable income" in note
        ),
    }
    if not all(asserted.values()):
        raise ValueError(f"Reviewed B.1 evidence changed: {asserted}")
    checks = {
        "tax_definition": {
            "evidence": {
                "b1_total_enrolled_tax_eur": float(total_tax),
                "spf_finances_total_taxes_eur": float(spf),
                "difference_eur": float(total_tax - spf),
                "b1_deciles_sum_minus_total_eur": float(decile_tax - total_tax),
                "top_percentiles_sum_minus_decile_10_eur": float(
                    percentile_tax - decile_10
                ),
                "negative_deciles": negative,
            },
            "passed": [
                "B.1's total enrolled tax equals the total taxes in Chronicle's spf-finances-pit-2023 package to EUR 0.06. That fact carries Statbel's concept and evidence URL, so the match reads one administrative total twice and corroborates nothing independently.",
                "B.1's deciles sum to its total, and percentiles 91-100 sum to decile 10, within EUR 0.10.",
            ],
            "missing": [
                "rulespec-be's candidate output belgium_pit_federal_and_local_tax_before_withholding (final_tax.yaml at 530849e) adds article 134 federal tax and the articles 465-466 local additions but not regional_surcharge.yaml's regional additional tax. No document in the suite says whether Statbel's total taxes include the regional additional tax.",
                "B.1 counts declarations (joint = one); every rulespec-be PIT output is per Person, and the couple pipeline needs relation bindings (S2-d) that are not built.",
                "That output is also the D51A_C1 holdout's measure: a B.1 producer must be a separately bound declaration-level measure, because S5 refuses a pre-calibration stage whose module produces a holdout measure.",
            ],
        },
        "income_ranking": {
            "evidence": {
                "decile_upper_bounds_strictly_increase": asserted[
                    "decile_bounds_strictly_increase"
                ],
                "percentile_upper_bounds_strictly_increase": asserted[
                    "percentile_bounds_strictly_increase"
                ],
                "decile_9_bound_below_percentile_91_bound": asserted[
                    "decile_9_bound_below_percentile_91_bound"
                ],
                "b1_minus_a1_total_taxable_income_eur": float(b1_income - a1_income),
                "decile_taxable_income_shares_sum": float(shares),
                "chronicle_concept_note": note,
            },
            "passed": [
                "Chronicle's B.1 facts state that B.1 excludes zero-income declarations, as A.1 does; B.1's and A.1's taxable-income totals agree to EUR 0.02, and the published bounds increase.",
            ],
            "missing": [
                "The stand-in ranks declarations by net professional income only; 42,007 declarations with taxable but no professional income, and every pensioner while pensions are unmodelled (d1902), would rank differently from Statbel's total net taxable income.",
                "B.1 publishes no declaration counts by decile; classifying model declarations needs either model-side ranking or the published bounds, and the bounds are income-year-2023 euros that need the same unavailable cut ageing as A.1.",
            ],
        },
        "policy_year": {
            "evidence": {
                "b1_legal_vintage": "income_year_2023_assessment_year_2024",
                "model_policy_year": POLICY_YEAR,
                "rulespec_be_final_tax_effective_from": "2025-01-01",
                "negative_deciles": negative,
            },
            "passed": [],
            "missing": [
                "B.1 is assessed under income-year-2023 law; the model applies 2026 law (BE-MC0). rulespec-be defines its PIT outputs from 2025-01-01 (its article 134 module states income year 2025 parameters), so B.1's own year cannot be reproduced, and no document bridges 2023 to 2026 tax law: ageing a 2023 tax mass by an FPB ratio would mix growth with law changes.",
                "Deciles 1 and 2 are negative (refundable credits) and rolled_forward_by_ratio refuses a negative base, so the declared ageing route cannot carry them to 2026.",
            ],
        },
    }
    for check in checks.values():
        check["result"] = "fail" if check["missing"] else "pass"
    passing = all(check["result"] == "pass" for check in checks.values())
    return {
        "verdict": "B.1 may activate."
        if passing
        else "B.1 stays inactive: no check passes completely from the documents.",
        "asserted_evidence": asserted,
        **checks,
    }


def _digest(payload):
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def selector_digest(row):
    """Digest everything a reference row selects, however it is spelled."""

    return _digest(
        {key: row[key] for key in ("ledger_selector", "value_operands") if key in row}
    )


def factor_digest(declaration):
    """Digest the selectors of one factor declaration's two schedules."""

    return _digest(
        [
            slot["selector"]
            for schedule in ("source_schedule", "target_schedule")
            for slot in declaration[schedule]
        ]
    )


def _matching(inventory, selector):
    pool = inventory.by_record_set.get(selector["record_set_id"], [])
    return [fact for fact in pool if fact_matches_selector(fact, selector)]


def _partition_total(inventory, row):
    """Return the published total a row's bands or rank groups add up to."""

    if row["metadata"].get("interval_table_id") == STANDIN_TABLE:
        return inventory.one(
            A1.format(component="taxable_income"),
            measure=row["ledger_selector"]["source_measure_id"],
            geography="BE",
            dimensions={A1_CLASS: "all"},
        )
    if row["metadata"]["contract_target_id"] == "statbel_b1_enrolled_tax_by_rank_group":
        return inventory.one(
            spec=B1_VALUES,
            measure="total_tax_eur",
            geography="BE",
            dimensions={"tax_return.rank_group": "all"},
        )
    return None


def _keyed(facts, field):
    """List distinct keys under the field name package scans admit as fact keys."""

    return [{field: key} for key in sorted({fact[field] for fact in facts})]


def _resolved_row(inventory, row, reference_set):
    base = _matching(inventory, row["ledger_selector"])
    operands = [
        fact
        for operand in row.get("value_operands", [])
        if operand.get("role") != "base"
        for fact in _matching(
            inventory, {key: value for key, value in operand.items() if key != "role"}
        )
    ]
    entry = {
        "name": row["name"],
        "reference_set": reference_set,
        "selector_sha256": selector_digest(row),
        "base_facts": _keyed(base, "aggregate_fact_key"),
        "operand_facts": _keyed(operands, "aggregate_fact_key"),
        "semantic_facts": _keyed([*base, *operands], "semantic_fact_key"),
    }
    total = _partition_total(inventory, row)
    if total is not None:
        entry["partition_total"] = {"semantic_fact_key": total["semantic_fact_key"]}
    return entry


def _resolved_factor(inventory, declaration):
    facts = [
        fact
        for schedule in ("source_schedule", "target_schedule")
        for slot in declaration[schedule]
        for fact in _matching(inventory, slot["selector"])
    ]
    return {
        "name": f"factor:{declaration['id']}",
        "reference_set": "factor_context",
        "selector_sha256": factor_digest(declaration),
        "base_facts": [],
        "operand_facts": _keyed(facts, "aggregate_fact_key"),
        "semantic_facts": _keyed(facts, "semantic_fact_key"),
    }


def resolved_reads(inventory, documents):
    """Bind each row and factor to the facts its selectors resolve to in the feed.

    CI holds no artifact, so it checks each digest against the committed row and
    the listed keys for disjointness. The feed audit proves the listed keys.
    """

    rows = [
        _resolved_row(inventory, row, document["reference_set"])
        for filename in (
            "target_references.json",
            "precal_references.json",
            "holdout_references.json",
        )
        for document in (documents[filename],)
        for row in document["target_references"]
    ]
    factors = [
        _resolved_factor(inventory, entry["declaration"])
        for entry in documents["reference_aging.json"]["factors"]
    ]
    return [*rows, *factors]


def _unemployment_series_choice(inventory):
    lines = {
        key: _fpb(inventory, f"social_benefits_detail.{row}", 2025)
        for key, (row, _) in T24_UNEMPLOYMENT_2025.items()
    }
    for key, (_, token) in T24_UNEMPLOYMENT_2025.items():
        if _token(lines[key]["value"]) != token:
            raise ValueError(f"Reviewed FPB T24 level {key} changed")
    return {
        "lines_2025": {
            key: {
                "row": fact["layout"]["source_row_id"],
                "label": fact["layout"]["groupby_value_label"],
                "eur": float(_value(fact)),
            }
            for key, fact in lines.items()
        },
        "choice": (
            "The target takes line a. Line 4 also holds benefits with a company "
            "supplement and career breaks and time credit, about 14% of the level. "
            "This row choice is proposed with d1902."
        ),
        "open_question": (
            "No document in the suite says whether line a. includes temporary "
            "unemployment, which the ordinary-benefit output does not compute."
        ),
    }


def _ageing_receipts(pairs, documents):
    used = {}
    for filename in (
        "target_references.json",
        "precal_references.json",
        "holdout_references.json",
    ):
        for row in documents[filename]["target_references"]:
            pair = row["metadata"].get("ageing_pair")
            if pair:
                used.setdefault(pair, []).append(row["name"])
    receipts = []
    for pair_id, (numerator, denominator) in sorted(pairs.items()):
        num, den = _value(numerator), _value(denominator)
        receipt = {
            "id": pair_id,
            "use": AGEING_PAIRS[pair_id]["use"],
            "fpb_row": AGEING_PAIRS[pair_id]["row"],
            "record_set_spec_id": numerator["layout"]["record_set_spec_id"],
            "source_row_id": numerator["layout"]["source_row_id"],
            "row_label": numerator["layout"]["groupby_value_label"],
            "numerator_selector": _selector(numerator),
            "denominator_selector": _selector(denominator),
            "numerator": float(num),
            "denominator": float(den),
            "ratio": float(num / den),
            "base_period": denominator["period"]["value"],
            "target_period": POLICY_YEAR,
            "numerator_assertion": numerator["assertion"],
            "denominator_assertion": denominator["assertion"],
            "reason": AGEING_PAIRS[pair_id]["reason"],
            "status": "proposed (BE-MC0, pending d1902); series choice is a B3 finding",
            "used_by": sorted(used.get(pair_id, [])),
        }
        if pair_id in FPB_OBSERVED_2025:
            observed = Fraction(FPB_OBSERVED_2025[pair_id])
            receipt.update(
                observed_segment_to_2025=float(observed / den),
                projected_step_2025_to_2026=float(num / observed),
            )
        receipts.append(receipt)
    return receipts


def build_reference_documents(artifact, *, package_dir=SPEC):
    """Return generated file-name/document pairs; verify the full feed identity first."""

    package_dir = Path(package_dir)
    feed_pin = json.loads((package_dir / "chronicle_feed.json").read_text())
    validate_chronicle_feed_pin(artifact, feed_pin)
    feed_identity = chronicle_feed_identity(feed_pin)
    inventory = Inventory(artifact.facts)
    pairs = {}
    for pair_id, pair in AGEING_PAIRS.items():
        base = min(pair["operands"])
        pairs[pair_id] = _pair(inventory, pair_id, base)
    for pair_id, token in FPB_OBSERVED_2025.items():
        if (
            _token(_fpb(inventory, AGEING_PAIRS[pair_id]["row"], 2025)["value"])
            != token
        ):
            raise ValueError(f"Reviewed FPB 2025 level {pair_id} changed")
    classes = _a1_classes(inventory)
    members = _band_members(classes)
    communes = {
        fact["geography"]["id"]: fact for fact in inventory.cells(COMMUNE_POPULATION)
    }
    if len(communes) != 565 or any(
        fact["geography"]["vintage"] != "nis_2025" for fact in communes.values()
    ):
        raise ValueError("Expected the 565 NIS 2025 communes")
    targets, target_labels = calibration_references(inventory, pairs, members)
    precal, precal_labels = precal_references(communes)
    holdouts, holdout_labels = holdout_references(inventory, pairs)
    documents = {}
    description = {
        "calibration": "Belgian calibration references (reference set calibration), generated by tools/generate_be_target_references.py from the pinned Chronicle be feed. Selectors only: no observation is carried. Demography targets Statbel's 1 January 2026 population directly; every other row rolls a 2023-2025 fact to 2026 by a declared FPB June 2026 ratio pair or stays parked with a named reason. Method-card choices are proposed pending d1902. Belgium stays inactive.",
        "precal": "Belgian pre-calibration references: Statbel commune population on 1 January 2025 as the atomic geography support mass, unaged (support shares). Identity only; no observation is carried.",
        "holdout": "Belgian held-out references: compared with the model, never fitted. No calibration-side node may read this resource. Every row is parked until its measure has a producer; JRC's EUROMOD-BE statistics are deferred while d1822 is open.",
    }
    for reference_set, rows, labels, filename, operations in (
        (
            "calibration",
            targets,
            target_labels,
            "target_references.json",
            ["identity", "rolled_forward_by_ratio"],
        ),
        ("precal", precal, precal_labels, "precal_references.json", ["identity"]),
        (
            "holdout",
            holdouts,
            holdout_labels,
            "holdout_references.json",
            ["rolled_forward_by_ratio"],
        ),
    ):
        documents[filename] = {
            "country": "be",
            "schema_version": 2,
            "reference_set": reference_set,
            "description": description[reference_set],
            "allowed_value_operations": operations,
            "hierarchy": _hierarchy(rows, labels),
            "target_references": rows,
            "target_profile": {"schema_version": 1, "hierarchy_reconciliations": []},
        }
    documents["reference_intervals.json"] = _interval_document(classes, members)
    documents["reference_profiles.json"] = _reference_profiles()
    documents["reference_aging.json"] = build_reference_aging(inventory, feed_identity)
    documents["reference_inventory.json"] = _inventory(
        inventory, pairs, documents, feed_pin, members, communes
    )
    return documents


def _inventory(inventory, pairs, documents, feed_pin, members, communes):
    counts = {}
    for filename in (
        "target_references.json",
        "precal_references.json",
        "holdout_references.json",
    ):
        rows = documents[filename]["target_references"]
        counts[filename] = {
            "total": len(rows),
            "by_family": dict(Counter(row["family"] for row in rows)),
            "by_contract": dict(
                Counter(row["metadata"]["contract_target_id"] for row in rows)
            ),
            "by_activation_status": dict(
                Counter(row["metadata"]["activation_status"] for row in rows)
            ),
        }
    nuts = {
        year: sum(
            (
                _value(fact)
                for fact in inventory.cells(STATBEL_POPULATION.format(year=year))
            ),
            Fraction(),
        )
        for year in (2025, POLICY_YEAR)
    }
    commune_total = sum((_value(fact) for fact in communes.values()), Fraction())
    b4 = {
        kind: sum(
            (
                _value(
                    inventory.one(
                        spec=B4,
                        measure="declarations",
                        geography="BE",
                        dimensions={
                            "tax_return.age_band": "all",
                            "tax_return.decile": decile,
                            "tax_return.declaration_type": kind,
                        },
                    )
                )
                for decile in range(1, 11)
            ),
            Fraction(),
        )
        for kind in ("joint", "individual")
    }
    a1_total = _value(
        inventory.one(
            A1.format(component="taxable_income"),
            measure="declarations",
            geography="BE",
            dimensions={A1_CLASS: "all"},
        )
    )
    aging = documents["reference_aging.json"]
    rsz = {
        entry["declaration"]["id"]: entry["operational_factor"]["numerator"]
        / entry["operational_factor"]["denominator"]
        for entry in aging["factors"]
    }
    wage_pair = {
        year: Fraction(token)
        for year, token in (
            (2023, "226018000000"),
            (2024, "233079000000"),
            (2025, "240536000000"),
        )
    }
    for year, token in wage_pair.items():
        if (
            _value(_fpb(inventory, "household_account.gross_wages_and_salaries", year))
            != token
        ):
            raise ValueError("Reviewed FPB gross wages changed")
    excluded = [
        ("jrc_euromod_be_statistics", DEFERRED_JRC),
        (
            "ris_amount_and_recipients",
            "August's RIS amount and count came from JRC; the suite holds no RIS administrative source, so the gap is named, not filled.",
        ),
        (
            "legal_pensions",
            "FPB pension amounts and SFPD beneficiary counts and bands: legal pensions are not modelled until d1902 settles BE-MC3's pension route; the SFPD bands are B4's quantile-map surface.",
        ),
        (
            "family_allowances",
            "FPB family allowances and Opgroeien/Ostbelgien caseloads: child benefits are reported as not modelled at level until indexed amounts are encoded (BE-MC0).",
        ),
        (
            "rsz_wage_classes",
            "RSZ daily-wage classes (B2f) are B4's wage quantile-map input under d1902, not a reference.",
        ),
        (
            "usd_eur_rate",
            "Eurostat ert_bil_eur_a is B4's currency-bridge input (BE-MC3), not a reference.",
        ),
        (
            "regional_a1_and_b1",
            "National bands only: the short list d1952 asks for; v0.5.1's 606 regional A.1 rows failed to fit.",
        ),
        (
            "a1_component_tables",
            "Professional, property, movable, miscellaneous and deduction tables are gap diagnostics in this inventory, not fitted rows.",
        ),
        (
            "a1_and_b1_totals",
            "Published totals duplicate the band and decile partitions; B.1's taxable-income masses duplicate A.1.",
        ),
        (
            "zero_income_declarations",
            "Statbel's 523,770 zero-income declarations are a diagnostic; the proxy's zeros differ, so the nil band is not fitted.",
        ),
        (
            "b3_b5_tables",
            "B.3 professional-category and B.5 dependant tables are outside the card's targets; v0.5.1's wage-only B.3 proxy destroyed weights.",
        ),
        (
            "b4_age_and_decile_cells",
            "Only the all-age joint and individual totals are targets (BE-MC4); age and decile cells would multiply weak cells.",
        ),
        (
            "statbel_commune_fiscal_income",
            "Commune fiscal income is a diagnostic surface for a later commune spine gate, not a B3 reference.",
        ),
        (
            "spf_finances_pit_total",
            "SPF Finances' total taxes equal B.1's total: one administrative total, read twice. The tax-definition check records it, and no row targets it.",
        ),
        (
            "spf_municipal_surcharge_rates",
            "Communal surcharge rates are W3 rule inputs, not references.",
        ),
        (
            "statbel_population_2025_nuts1",
            "Statbel's 1 January 2026 cells are the target-year observation; the 2025 cells would duplicate them.",
        ),
        (
            "movable_income",
            "Movable income stays out of the stand-in (art. 313 CIR 92 withholding dispensation) and out of the targets until the investment-income match check (d1952 build item 5).",
        ),
        (
            "eurostat_d11_wages",
            "Eurostat D11 duplicates FPB's gross wages, the card's wage-headline comparator.",
        ),
        (
            "eurostat_b3g_mixed_income",
            "Mixed income is a calibration aggregate (FPB net mixed income), so it cannot also be held out.",
        ),
        (
            "eurostat_survey_and_social_protection",
            "Eurostat ilc_* survey aggregates and spr_exp_func expenditure are excluded: surveys are validation-only and outside the card's holdouts, and ESSPROS overlaps FPB's T24 rows.",
        ),
        (
            "fpb_projections_as_direct_targets",
            "No reference takes an FPB projection as its base: every base is an observation. Seven rolled rows still equal FPB's 2026 projection, because each base equals its pair's denominator: unemployment benefits, GRAPA and net mixed income in calibration, and gross wages and the Eurostat D61, D62 and D5 rows in the holdouts. Those rows say so in ageing_target_equals.",
        ),
        (
            "german_and_french_rows",
            "Four Eurostat members carry DE and FR rows (72 of 87,112); the generator refuses any non-Belgian fact before it can name one.",
        ),
    ]
    return {
        "schema_version": 1,
        "country": "be",
        "generated_by": "tools/generate_be_target_references.py",
        "feed_identity": feed_pin,
        "policy_year": POLICY_YEAR,
        "method_card_status": "BE-MC0, BE-MC3 to BE-MC6 proposed pending d1902; nothing publishes a number",
        "reference_counts": counts,
        "regions": list(REGIONS),
        "age_bands": list(AGE_BANDS),
        "standin_bands": [
            {
                "id": band,
                "lower_eur_iy2023": lower,
                "upper_eur_iy2023": upper,
                "a1_classes": members[band],
            }
            for band, lower, upper in STANDIN_BANDS
        ],
        "standin_band_reason": (
            "Fixed before any candidate (d1952): EUR 10,000 steps over the dense "
            "middle, wider steps above EUR 50,000, A.1's open top class whole; "
            "every edge is an A.1 class edge; eight bands, sixteen constraints."
        ),
        "standin_gaps": _standin_gaps(inventory, members),
        "ageing_pairs": _ageing_receipts(pairs, documents),
        "rsz_wage_cross_checks": [
            {
                "factor_id": factor_id,
                "rsz_observed_factor": factor,
                "fpb_gross_wages_observed_ratio": float(
                    wage_pair[2025] / wage_pair[RSZ_FACTORS[factor_id][0]]
                ),
                "use": "diagnostic only",
            }
            for factor_id, factor in rsz.items()
        ],
        "b4_identity": {
            "joint": float(b4["joint"]),
            "individual": float(b4["individual"]),
            "a1_declarations": float(a1_total),
            "closes_exactly": b4["joint"] + b4["individual"] == a1_total,
            "use": "A.1 band counts and B.4 type counts share fpb_t07_total_population, so the identity survives ageing.",
        },
        "destination_mass": {
            "commune_sum_2025": float(commune_total),
            "nuts1_sum_2025": float(nuts[2025]),
            "nuts1_sum_2026": float(nuts[POLICY_YEAR]),
            "commune_minus_nuts1_2025": float(commune_total - nuts[2025]),
            "create_mass_reference": "Not in B3. CREATE needs exactly one unfiltered national person count. The suite has no national Statbel population fact, and a national row composed from the 565 communes refuses: the shared composed-geography path labels only UK areas. W3 chooses between a shared Belgian geography label (NZ hub) and FPB's annual-average population.",
        },
        "b1_checks": _b1_checks(inventory),
        "unemployment_series_choice": _unemployment_series_choice(inventory),
        "geography_vintage_bindings": [
            {
                "layer": "nuts1",
                "chronicle_vintage": GEOGRAPHY_VINTAGES["nuts1"][0],
                "typed_authority": GEOGRAPHY_VINTAGES["nuts1"][1],
                "relation": "crosswalk",
                "basis": "Chronicle labels Statbel's regions NUTS_2024. The typed nuts1 layer records no code list, so the mapping rests on destination_mass: the three regions' 2025 population equals the 565 communes' sum. A package test holds every nuts1 row to this binding. This choice is proposed with d1902.",
            },
            {
                "layer": "commune",
                "chronicle_vintage": GEOGRAPHY_VINTAGES["commune"][0],
                "typed_authority": GEOGRAPHY_VINTAGES["commune"][1],
                "relation": "exact_alias",
                "basis": "nis_2025 is an exact typed alias of vintage:be_nis_2025.",
            },
        ],
        "holdout_information_overlaps": [
            "Separation is checked at the fact and measure level; holdouts still share information with calibration through economics, not facts:",
            "FPB gross wages (wage-headline holdout): ONSS's calibrated worker contributions are a fixed share of gross pay.",
            "Eurostat D62 social benefits received contain the calibrated unemployment and GRAPA aggregates.",
            "Eurostat D61 social contributions paid contain the calibrated ONSS worker contributions.",
            "Eurostat D51A_C1 and D5 measure the same personal income tax that B.1 would calibrate once active.",
            "The A.1 stand-in is 98% net professional income, and its amount pair, FPB T11 total resources, contains gross wages (line 3), the wage holdout's own series.",
        ],
        "grammar_gaps": [
            "A national row composed from commune facts refuses for Belgium: the composed-geography path labels only UK areas (ledger_targets._geography_fallback_label), so CREATE's single national mass reference cannot be the commune sum.",
            "A schema-2 target_profile requires period_match_policy 'exact', which refuses rolled_forward_by_ratio; the documents keep B1's schema-2 hierarchy catalogs with a schema-1 target_profile (NZ's documents use the same schema-1 profile, with inline hierarchies).",
            "No shared operation ages interval cuts by a declared annual ratio pair: scale_money_and_intervals needs a G8-r2 factor and singleton identity money rows, and G8-r2's projected_period_average_ratio cannot project an annual series past its last observation.",
            "G8-r2's resolver needs dated period_coverage on every operand; Chronicle's FPB, Statbel NUTS1, NBB, ONSS, ONEM and most Eurostat facts carry none.",
            "rolled_forward_by_ratio refuses a negative base, so signed targets (B.1 low deciles) cannot be aged by it.",
            "rolled_forward_by_ratio's series key treats every FPB annex row of one table as one series (same concept, entity, geography and empty dimensions); the generator and tests enforce row identity themselves.",
            "The typed vintage alias set admits only a layer's full id, its country-shortened id and its authority value, so Chronicle's NUTS_2024 label cannot be an alias of vintage:be_nuts1_2025; geography_vintage_bindings carries the mapping instead.",
            "The schema-1 target_profile drops B1's required_families, basis_periods, tolerance tiers and subnational vintage rule. Package tests restore required families, the vintage rule and each row's periods. Caseload tolerance moves from B1's 15% to BE-MC6's 5%.",
        ],
        "exclusions": [{"id": name, "reason": reason} for name, reason in excluded],
        "resolved_reads": resolved_reads(inventory, documents),
    }


def pin_feed(artifact, source_commit):
    """Return a feed pin for a reviewed artifact; identities are the loader's own."""

    manifest = artifact.manifest or {}
    return {
        "version": 1,
        "country": "be",
        "scope": "be_calibration",
        "source_repo": "PolicyEngine/chronicle",
        "source_commit": source_commit,
        "build": "build-bundle --suite be -> build-consumer-artifact",
        "artifact_schema_version": artifact.schema_version,
        "consumer_fact_schema_versions": list(artifact.fact_schema_versions),
        "consumer_fact_schema_sha256": manifest.get("consumer_fact_schema_sha256"),
        "fact_row_count": artifact.fact_row_count,
        "facts_sha256": artifact.facts_sha256,
        "manifest_sha256": artifact.manifest_sha256,
    }


def _encode(document):
    return json.dumps(document, ensure_ascii=False, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=os.environ.get("BE_CHRONICLE_ARTIFACT"))
    parser.add_argument("--out", type=Path, default=SPEC)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--pin-feed", metavar="SOURCE_COMMIT")
    args = parser.parse_args()
    if not args.artifact:
        parser.error("--artifact or BE_CHRONICLE_ARTIFACT is required")
    if args.check and args.pin_feed:
        parser.error("--check and --pin-feed are exclusive")
    artifact = load_ledger_consumer_artifact(args.artifact, lexical_floats=True)
    if args.pin_feed:
        (args.out / "chronicle_feed.json").write_text(
            _encode(pin_feed(artifact, args.pin_feed))
        )
    documents = build_reference_documents(artifact, package_dir=args.out)
    stale = []
    for filename, document in documents.items():
        encoded = _encode(document)
        destination = args.out / filename
        if args.check:
            if not destination.exists() or destination.read_text() != encoded:
                stale.append(str(destination))
        else:
            destination.write_text(encoded)
    if stale:
        raise SystemExit(f"Stale generated resource(s): {stale}")


if __name__ == "__main__":
    main()
