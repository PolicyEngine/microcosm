"""Reconcile available MSD headline-rate evidence before NZ activation.

Only the pinned Chronicle artifact and the packaged published MSD rates enter
this calculation. No donor, engine, assigned receipt or active graph enters it.
The feed lacks payment-category joint counts and individual abatement/exposure
data, so the report preserves that gap instead of manufacturing paid totals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path

from microcosm.build.ledger_artifact import (
    load_ledger_consumer_artifact,
    validate_chronicle_feed_pin,
)

SPEC = (
    Path(__file__).resolve().parents[1]
    / "packages/microcosm-build/src/microcosm/build/nz"
)
OUTPUT = "msd_pre_activation_reconciliation.json"
RATE_RESOURCE = "msd_benefit_rates_20260401.json"
AGE_RECORD_SET = "msd_benefit_statistics.month2026_06.working_age_by_benefit_and_age"
SUPER_RECORD_SET = "msd_nzs_vp.june2026.recipient_total"
EXPENSE_RECORD_SET = "msd.benefit_expenses.fy2024"
AGES = ("18-24 years", "25-39 years", "40-54 years", "55-64 years")
PROGRAMMES = (
    "jobseeker_support",
    "sole_parent_support",
    "supported_living_payment",
    "nz_super",
)
EXPENSE_KEYS = {
    "jobseeker_support_and_emergency_benefit": "ledger.aggregate_fact.v2:ee2fbceca03ea6cf897aeb87",
    "new_zealand_superannuation": "ledger.aggregate_fact.v2:0bbf770a6b0baa84853d9c18",
    "orphans_unsupported_childs_benefit": "ledger.aggregate_fact.v2:c6846c99892eccdf29a4a695",
    "sole_parent_support": "ledger.aggregate_fact.v2:48d697bf0900434af3924605",
    "student_allowances": "ledger.aggregate_fact.v2:7ee2e4e9979352ac3989a5b7",
    "supported_living_payment": "ledger.aggregate_fact.v2:da9daf8ba1407b2e8a76dce6",
}
EXPENSE_COMPONENT = {
    "jobseeker_support": "jobseeker_support_and_emergency_benefit",
    "sole_parent_support": "sole_parent_support",
    "supported_living_payment": "supported_living_payment",
    "nz_super": "new_zealand_superannuation",
}
EXPOSURES = ("1/2", "3/4", "1")


def _decimal(value):
    """Render a terminating Fraction exactly, without a floating-point step."""
    value = Fraction(value)
    scale = 0
    while (value * 10**scale).denominator != 1:
        scale += 1
        if scale > 12:
            raise ValueError("Diagnostic amount does not have a short exact decimal")
    scaled = int(value * 10**scale)
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).zfill(scale + 1)
    if not scale:
        return sign + digits + ".00"
    return sign + digits[:-scale] + "." + digits[-scale:].ljust(2, "0")


def _one(rows, **fields):
    matches = [row for row in rows if all(row.get(k) == v for k, v in fields.items())]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one disclosed source row for {fields}: {len(matches)}"
        )
    return matches[0]


def _source_rows(artifact):
    selected = []
    for fact in artifact.facts:
        record_set = fact["layout"]["record_set_id"]
        dimensions = fact["dimensions"]
        if not (
            fact["aggregate_fact_key"] in EXPENSE_KEYS.values()
            or (
                record_set == AGE_RECORD_SET
                and dimensions.get("msd_main_benefit_type") in PROGRAMMES
            )
            or (
                record_set == SUPER_RECORD_SET
                and dimensions.get("msd_nzs_vp.benefit_type") == "nzs"
            )
        ):
            continue
        selected.append(
            {
                "aggregate_fact_key": fact["aggregate_fact_key"],
                "record_set_id": record_set,
                "source_row_id": fact["layout"]["source_row_id"],
                "dimensions": dimensions,
                "period": fact["period"],
                "period_coverage": {
                    key: fact["period_coverage"][key]
                    for key in (
                        "basis",
                        "start_date",
                        "end_date",
                        "source_period_label",
                    )
                },
                "unit": fact["observed_measure"]["unit"],
                "value": fact["value"],
            }
        )
    return selected


def _dimension_audit(artifact):
    """Search every MSD row, including subgroup and NZS marginal tables."""
    record_sets = {}
    for fact in artifact.facts:
        if fact["observed_measure"]["source_name"] == "msd":
            record_sets.setdefault(fact["layout"]["record_set_id"], set()).update(
                fact["dimensions"]
            )
    return {key: sorted(value) for key, value in sorted(record_sets.items())}


def _rate_options(rates, programme, age):
    """Retain alternative categories; never posit their unknown joint shares."""
    if programme == "nz_super":
        return [
            {
                "section": row["section"],
                "category": row["category"],
                "gross_weekly_nzd": _decimal(Fraction(row["before_tax"]) / 2),
            }
            for row in rates["fortnightly_nz_super"]
            if "combined" not in row["category"] and "(total)" not in row["category"]
        ]
    result = []
    for row in rates["weekly_main_benefits"]:
        if row["programme"] != programme or not row["individual_validation_point"]:
            continue
        category = row["category"]
        if "16-17" in category:
            continue
        if programme == "jobseeker_support":
            young = "18 to 19" in category or "20 to 24" in category
            older = "25 years or over" in category
            if (age == AGES[0] and older) or (age != AGES[0] and young):
                continue
        result.append(
            {
                "section": row["section"],
                "category": category,
                "gross_weekly_nzd": row["before_tax"],
            }
        )
    if not result:
        raise ValueError(f"No individual rate options for {programme} / {age}")
    return result


def reconcile(source_rows, rates, feed_pin, rates_sha256, dimension_audit):
    """Replay the report from its disclosed aggregate inputs and published rates."""
    if len(source_rows) != 19:
        raise ValueError(
            "Expected 12 working-age cells, one NZS total and six expenses"
        )
    expenses = []
    for component, key in EXPENSE_KEYS.items():
        source = _one(source_rows, aggregate_fact_key=key)
        if (
            source["record_set_id"] != EXPENSE_RECORD_SET
            or source["source_row_id"] != component
            or source["unit"] != "nzd"
        ):
            raise ValueError(f"Expense source identity changed for {component}")
        expenses.append(
            {
                "component": component,
                "aggregate_fact_key": key,
                "published_actual_expense_nzd": _decimal(source["value"]),
                "comparison_status": (
                    "headline_envelope_only"
                    if component in EXPENSE_COMPONENT.values()
                    else "unmodelled_programme_no_comparison"
                ),
            }
        )
    comparisons = []
    for programme in PROGRAMMES:
        strata = []
        for age in ("all",) if programme == "nz_super" else AGES:
            if programme == "nz_super":
                source = _one(source_rows, record_set_id=SUPER_RECORD_SET)
            else:
                dimensions = {
                    "msd_main_benefit_age_group_source_label": age,
                    "msd_main_benefit_age_status": "working_age",
                    "msd_main_benefit_type": programme,
                }
                source = _one(
                    source_rows, record_set_id=AGE_RECORD_SET, dimensions=dimensions
                )
            if source["unit"] != "count" or source["period"]["value"] != "2026-06":
                raise ValueError(f"Recipient source identity changed for {programme}")
            count = source["value"]
            if type(count) is not int or count < 0:
                raise ValueError("Recipient counts must be nonnegative integers")
            options = _rate_options(rates, programme, age)
            weekly = [Fraction(row["gross_weekly_nzd"]) for row in options]
            strata.append(
                {
                    "age_stratum": age,
                    "recipient_selector": {
                        "aggregate_fact_key": source["aggregate_fact_key"]
                    },
                    "recipient_count": count,
                    "available_individual_rate_categories": options,
                    "annual_headline_lower_nzd": _decimal(count * 52 * min(weekly)),
                    "annual_headline_upper_nzd": _decimal(count * 52 * max(weekly)),
                }
            )
        lower = sum(Fraction(row["annual_headline_lower_nzd"]) for row in strata)
        upper = sum(Fraction(row["annual_headline_upper_nzd"]) for row in strata)
        expense = _one(expenses, component=EXPENSE_COMPONENT[programme])[
            "published_actual_expense_nzd"
        ]
        published = Fraction(expense)
        comparisons.append(
            {
                "programme": programme,
                "recipient_count": sum(row["recipient_count"] for row in strata),
                "strata": strata,
                "headline_rate_total_nzd": _decimal(lower) if lower == upper else None,
                "category_weighted_after_abatement_total_nzd": None,
                "published_expense_component": EXPENSE_COMPONENT[programme],
                "published_actual_expense_nzd": expense,
                "exposure_sensitivity": [
                    {
                        "full_year_exposure": exposure,
                        "headline_lower_nzd": _decimal(lower * Fraction(exposure)),
                        "headline_upper_nzd": _decimal(upper * Fraction(exposure)),
                        "lower_minus_published_nzd": _decimal(
                            lower * Fraction(exposure) - published
                        ),
                        "upper_minus_published_nzd": _decimal(
                            upper * Fraction(exposure) - published
                        ),
                    }
                    for exposure in EXPOSURES
                ],
            }
        )
    return {
        "schema_version": 1,
        "stage": "pre_activation",
        "status": "after_abatement_reconciliation_unavailable",
        "purpose": "Microcosm reviews the available aggregate evidence before activating the receipt references. This report grants no release approval and runs no active graph.",
        "feed_identity": feed_pin,
        "rates_resource": RATE_RESOURCE,
        "rates_sha256": rates_sha256,
        "source_rows": source_rows,
        "exhaustive_msd_dimension_audit": dimension_audit,
        "method": {
            "headline": "Microcosm multiplies each published June 2026 age-cell recipient count by 52 and each compatible published April 2026 individual gross weekly rate. Microcosm sums the lower and upper category endpoints across age cells; it assigns no category shares. NZS uses its unstratified stock and all published individual fortnightly rates divided by two, including hospital and legacy categories.",
            "category_weighting": "The pinned feed supplies programme-by-age counts and separate subgroup, gender and ethnicity marginals. None supplies the joint payment categories for single, couple, children, home or sharing status. Microcosm reports an age-weighted headline envelope; it cannot compute the required category-weighted paid total from these inputs.",
            "after_abatement": "Published headline rates precede income abatement. The pinned feed and rates omit recipient earnings, partner earnings and person-level paid amounts. Microcosm cannot compute an after-abatement total without person-level data; it leaves that total null and makes no abatement assumption.",
            "exposure": "Microcosm shows mechanical exposure fractions 1/2, 3/4 and 1 to disclose sensitivity. These fractions describe scenarios, not observed durations or a fitted central estimate. Each fraction scales both headline endpoints exactly.",
            "period_and_universe": "June 2026 stocks and April 2026 rates describe a different period from actual expenditure in the year ended 30 June 2025. Microcosm applies no uprating or period alignment. JS expenditure also includes Emergency Benefit; the modelled JS count excludes it. Working-age main-benefit counts omit persons outside 18-64. Independently rounded source cells may disagree with published totals.",
            "arithmetic": "Microcosm uses exact rational arithmetic and decimal text. It applies no intermediate cent rounding and never applies a two-recipient couple total to one recipient.",
        },
        "comparisons": comparisons,
        "published_expenses": expenses,
        "closure_requirements": [
            "The hub needs person-level or published joint payment-category counts to replace the headline envelopes with category-weighted totals.",
            "The hub needs the pinned engine, person and partner income inputs, paid gross amounts after abatement, and measured annual exposure to compute paid programme totals; G8-r2c owns the declared annual-transfer exposure sensitivity.",
            "The hub needs a reviewed common expenditure period, rate vintage and recipient universe, including Emergency Benefit and non-working-age differences, before it can interpret the remaining published-total gap.",
        ],
    }


def generate(artifact, *, package_dir=SPEC):
    package_dir = Path(package_dir)
    feed_pin = json.loads((package_dir / "chronicle_feed.json").read_text())
    validate_chronicle_feed_pin(artifact, feed_pin)
    rates_bytes = (package_dir / RATE_RESOURCE).read_bytes()
    return reconcile(
        _source_rows(artifact),
        json.loads(rates_bytes),
        feed_pin,
        hashlib.sha256(rates_bytes).hexdigest(),
        _dimension_audit(artifact),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=os.environ.get("NZ_CHRONICLE_ARTIFACT"))
    parser.add_argument("--out", type=Path, default=SPEC / OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if not args.artifact:
        parser.error("--artifact or NZ_CHRONICLE_ARTIFACT is required")
    artifact = load_ledger_consumer_artifact(args.artifact)
    rendered = json.dumps(generate(artifact), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if args.out.read_text() != rendered:
            raise SystemExit(f"Stale MSD pre-activation reconciliation: {args.out}")
        print("MSD pre-activation reconciliation: current")
    else:
        args.out.write_text(rendered)


if __name__ == "__main__":
    main()
