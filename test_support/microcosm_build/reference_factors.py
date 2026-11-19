"""Small engine-free declarations for exercising public factor interfaces."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal
from fractions import Fraction

from microcosm.build.ledger_artifact import CONSUMER_ARTIFACT_SCHEMA_VERSION
from microcosm.build.reference_factors import resolve_reference_factor


def factor_fixture(
    *,
    feed_identity=None,
    frequency=2,
    source_level="100",
    anchor_level="110",
    growth="0",
    digits=6,
    expected_numerator=None,
):
    """Return a declaration, exact numeric facts and a matching feed identity.

    Slots are single days: period coverage, rather than calendar conventions,
    establishes contiguity. Callers may serialize facts into their toy artifact.
    """
    identity = (
        {
            "source_commit": "c" * 40,
            "facts_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "artifact_schema_version": CONSUMER_ARTIFACT_SCHEMA_VERSION,
            "consumer_fact_schema_versions": ["chronicle.consumer_fact.v4"],
            "consumer_fact_schema_sha256": "d" * 64,
            "fact_row_count": frequency + 2,
        }
        if feed_identity is None
        else deepcopy(feed_identity)
    )
    facts, source, target = [], [], []
    validation = {
        "unit": "money_per_week",
        "source_sha256": "a" * 64,
        "source_vintage": "reviewed",
        "entity_role": "employee",
        "provenance_class": "survey_aggregate",
        "assertion": "observation",
    }
    for schedule, start, prefix, level in (
        (source, date(2001, 1, 1), "source", source_level),
        (target, date(2002, 1, 1), "target", anchor_level),
    ):
        for index in range(frequency):
            period = (start + timedelta(days=index)).isoformat()
            coverage = {"start_date": period, "end_date": period}
            selector = {
                "source_name": "toy_office",
                "source_measure_id": "weekly_pay",
                "entity_name": "person",
                "period_type": "day",
                "period_value": period,
                "assertion": "observation",
                "dimensions": [],
            }
            slot = {
                "selector": selector,
                "validation": {**validation, "period_coverage": coverage},
                "weight": "1",
                "sequence": index,
            }
            if schedule is target:
                slot["projection_offset"] = index
            schedule.append(slot)
            if schedule is source or index == 0:
                facts.append(
                    {
                        "aggregate_fact_key": f"{prefix}-{index}",
                        "value": Decimal(level),
                        "assertion": "observation",
                        "provenance_class": "survey_aggregate",
                        "period": {"type": "day", "value": period},
                        "period_coverage": coverage,
                        "entity": {"name": "person", "role": "employee"},
                        "dimensions": {},
                        "source": {
                            "source_name": "toy_office",
                            "source_sha256": "a" * 64,
                            "vintage": "reviewed",
                        },
                        "observed_measure": {
                            "source_measure_id": "weekly_pay",
                            "unit": "money_per_week",
                        },
                    }
                )
    growth_period = "2002"
    growth_validation = {
        **validation,
        "unit": "percent",
        "assertion": "source_projection",
        "provenance_class": "model_output",
        "period_coverage": {"start_date": "2002-01-01", "end_date": "2002-12-31"},
        "source_column_id": "annual_projection",
    }
    facts.append(
        {
            "aggregate_fact_key": "growth",
            "value": Decimal(growth),
            "assertion": "source_projection",
            "provenance_class": "model_output",
            "period": {"type": "year", "value": growth_period},
            "period_coverage": growth_validation["period_coverage"],
            "entity": {"name": "person", "role": "employee"},
            "dimensions": {},
            "source": {
                "source_name": "toy_office",
                "source_sha256": "a" * 64,
                "vintage": "reviewed",
            },
            "observed_measure": {"source_measure_id": "growth", "unit": "percent"},
            "layout": {"source_column_id": "annual_projection"},
        }
    )
    if expected_numerator is None:
        expected_numerator = int(
            Decimal(anchor_level) / Decimal(source_level) * 10**digits
        )
    declaration = {
        "schema_version": 1,
        "id": "toy_factor",
        "operation": "projected_period_average_ratio",
        "operation_version": 1,
        "purpose": "profile_input",
        "feed_identity": identity,
        "source_period": 2001,
        "target_period": 2002,
        "source_schedule": source,
        "target_schedule": target,
        "growth": {
            "selector": {"aggregate_fact_key": "growth", "period_value": growth_period},
            "validation": growth_validation,
            "unit_multiplier": "0.01",
            "scenario_delta": "0",
            "sensitivity_provenance": ["toy reviewed sensitivity"],
            "assertion_policy": "allow_source_projection",
        },
        "subperiods_per_year": frequency,
        "projection_policy": "observed_preferred_latest_contiguous",
        "rounding": {"digits": digits, "mode": "nearest_half_even"},
        "expected_operational_factor": {
            "numerator": expected_numerator,
            "denominator": 10**digits,
        },
        "validity": {
            "declared_on": "2000-01-01",
            "valid_from": "2000-01-01",
            "valid_until": "2099-12-31",
            "as_of": "2002-01-01",
        },
        "evidence": ["toy published levels"],
        "decision_references": ["toy reviewed decision"],
    }
    return declaration, facts, identity


def factor_context_fixture(**kwargs):
    """Return a context with a factor genuinely resolved through the public API."""
    declaration, facts, identity = factor_fixture(**kwargs)
    factor, receipt = resolve_reference_factor(declaration, facts, identity)
    return (
        {
            "schema_version": 1,
            "feed_identity": identity,
            "factors": [
                {
                    "declaration": declaration,
                    "operational_factor": factor,
                    "receipt": receipt,
                }
            ],
        },
        facts,
        identity,
    )


def published_projection_fixture(
    *,
    feed_identity=None,
    frequency=1,
    source_levels=("100",),
    target_levels=("110",),
    weights=None,
    digits=6,
    purpose="money_normalization_input",
    source_year=2001,
    target_year=2002,
    expected_numerator=None,
):
    """Return a published-projection declaration, its facts and feed identity.

    Source facts are observations; target facts are projections the source
    published (``source_projection``, ``model_output``). At one subperiod a
    slot covers its whole calendar year; above one, slots are single days, as
    in :func:`factor_fixture`. A single level or weight applies to every slot.
    """
    facts = []

    def levels(values):
        values = tuple(values)
        values = values * frequency if len(values) == 1 else values
        if len(values) != frequency:
            raise ValueError("Give one level or weight per subperiod.")
        return values

    weights = levels(("1",) if weights is None else weights)
    identity = (
        {
            "source_commit": "c" * 40,
            "facts_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "artifact_schema_version": CONSUMER_ARTIFACT_SCHEMA_VERSION,
            "consumer_fact_schema_versions": ["chronicle.consumer_fact.v4"],
            "consumer_fact_schema_sha256": "d" * 64,
            "fact_row_count": 2 * frequency,
        }
        if feed_identity is None
        else deepcopy(feed_identity)
    )
    schedules = {}
    for name, year, prefix, values, assertion, provenance in (
        (
            "source",
            source_year,
            "source",
            source_levels,
            "observation",
            "survey_aggregate",
        ),
        (
            "target",
            target_year,
            "projection",
            target_levels,
            "source_projection",
            "model_output",
        ),
    ):
        schedule = schedules[name] = []
        for index, level in enumerate(levels(values)):
            if frequency == 1:
                period_type, period = "year", str(year)
                coverage = {"start_date": f"{year}-01-01", "end_date": f"{year}-12-31"}
            else:
                period_type = "day"
                period = (date(year, 1, 1) + timedelta(days=index)).isoformat()
                coverage = {"start_date": period, "end_date": period}
            schedule.append(
                {
                    "selector": {
                        "source_name": "toy_office",
                        "source_measure_id": "weekly_pay",
                        "entity_name": "person",
                        "period_type": period_type,
                        "period_value": period,
                        "assertion": assertion,
                        "dimensions": [],
                    },
                    "validation": {
                        "unit": "money_per_week",
                        "source_sha256": "a" * 64,
                        "source_vintage": "reviewed",
                        "entity_role": "employee",
                        "provenance_class": provenance,
                        "assertion": assertion,
                        "period_coverage": dict(coverage),
                    },
                    "weight": weights[index],
                    "sequence": index,
                }
            )
            facts.append(
                {
                    "aggregate_fact_key": f"{prefix}-{index}",
                    "value": Decimal(level),
                    "assertion": assertion,
                    "provenance_class": provenance,
                    "period": {"type": period_type, "value": period},
                    "period_coverage": dict(coverage),
                    "entity": {"name": "person", "role": "employee"},
                    "dimensions": {},
                    "source": {
                        "source_name": "toy_office",
                        "source_sha256": "a" * 64,
                        "vintage": "reviewed",
                    },
                    "observed_measure": {
                        "source_measure_id": "weekly_pay",
                        "unit": "money_per_week",
                    },
                }
            )
    if expected_numerator is None:

        def mean(values):
            pairs = [
                (Fraction(level), Fraction(weight))
                for level, weight in zip(levels(values), weights, strict=True)
            ]
            return sum(v * w for v, w in pairs) / sum(w for _, w in pairs)

        # The built-in round takes ties to even; no production helper.
        expected_numerator = round(
            mean(target_levels) / mean(source_levels) * 10**digits
        )
    declaration = {
        "schema_version": 1,
        "id": "toy_published_projection_factor",
        "operation": "published_projection_period_average_ratio",
        "operation_version": 1,
        "purpose": purpose,
        "feed_identity": identity,
        "source_period": source_year,
        "target_period": target_year,
        "source_schedule": schedules["source"],
        "target_schedule": schedules["target"],
        "subperiods_per_year": frequency,
        "rounding": {"digits": digits, "mode": "nearest_half_even"},
        "expected_operational_factor": {
            "numerator": expected_numerator,
            "denominator": 10**digits,
        },
        "validity": {
            "declared_on": "2000-01-01",
            "valid_from": "2000-01-01",
            "valid_until": "2099-12-31",
            "as_of": "2002-01-01",
        },
        "evidence": ["toy published levels and projections"],
        "decision_references": ["toy reviewed decision"],
    }
    return declaration, facts, identity
