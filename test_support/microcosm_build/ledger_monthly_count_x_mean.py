"""Stat-Xplore-shaped count and mean facts for the count x mean window tests."""

from __future__ import annotations

from microcosm.build.ledger_targets import (
    MONTHLY_WINDOW_COUNT_X_MEAN,
    LedgerTargetReference,
)

COUNT = "dwp.state_pension_recipients"
MEAN = "dwp.state_pension_mean_weekly_amount"
MONTHS = ("2025-02", "2025-05", "2025-08", "2025-11")


def statx_fact(
    month: str,
    role: str,
    value: float,
    *,
    group: str = "all",
    release: str = "stat_xplore_2026_09_29",
    sha: str = "a" * 64,
) -> dict[str, object]:
    stamp = month.replace("-", "_")
    mean = role == "mean"
    record_set = "dwp.statx.state_pension.type" + (".mean" if mean else "")
    return {
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{role}-{group}-{stamp}",
        "semantic_fact_key": f"ledger.semantic_fact.v2:{role}-{group}",
        "label": f"Great Britain {month} State Pension {role}",
        "source_release_key": release,
        "value": value,
        "period": {"type": "month", "value": month},
        "geography": {
            "level": "country",
            "id": "K03000001",
            "name": "Great Britain",
            "vintage": "current",
        },
        "entity": {"name": "person", "role": "claimant"},
        "observed_measure": {
            "source_name": "dwp",
            "source_measure_id": "mean_weekly_amount" if mean else "recipients",
            "source_concept": "dwp.state_pension_mean_weekly_amount"
            if mean
            else "dwp.state_pension_caseload",
            "unit": "gbp_per_week" if mean else "count",
        },
        "concept_alignment": {
            "source_concept": "dwp.state_pension_mean_weekly_amount"
            if mean
            else "dwp.state_pension_caseload",
            "canonical_concept": MEAN if mean else COUNT,
            "relation": "source_label",
            "authority": "dwp",
        },
        "aggregation": {"method": "mean" if mean else "sum"},
        "source": {
            "source_name": "dwp",
            "source_table": "State Pension caseload and mean weekly amount",
            "source_file": "state_pension_by_type.json",
            "source_sha256": sha,
            "vintage": "stat_xplore_2026_09_29",
        },
        "dimensions": {"category_of_pension": group},
        "universe_constraints": {"domain": "social_security"},
        "layout": {
            "record_set_id": f"{record_set}.month{stamp}",
            "record_set_spec_id": f"{record_set}.month{stamp}.v1",
            "groupby_dimension": "dwp.state_pension_type",
            "groupby_value_id": group,
            "measure_id": "mean_weekly_amount" if mean else "recipients",
        },
    }


def statx_window(counts, means, **kwargs) -> list[dict[str, object]]:
    return [
        fact
        for month, count, mean in zip(MONTHS, counts, means, strict=True)
        for fact in (
            statx_fact(month, "count", count, **kwargs),
            statx_fact(month, "mean", mean, **kwargs),
        )
    ]


def window_reference(**overrides) -> LedgerTargetReference:
    fields = {
        "name": "dwp/state_pension/amount",
        "ledger_selector": {
            "source_name": "dwp",
            "source_concept": [
                "dwp.state_pension_caseload",
                "dwp.state_pension_mean_weekly_amount",
            ],
            "geography_id": "K03000001",
            "dimension_values": {"category_of_pension": "all"},
            "period_type": "month",
            "period_value": list(MONTHS),
        },
        "value_operation": MONTHLY_WINDOW_COUNT_X_MEAN,
        "period_match_policy": "source_window",
        "value_operands": (
            {"role": "count", "concept": COUNT},
            {"role": "mean", "concept": MEAN, "period_factor": 52},
        ),
        "entity": "person",
        "measure": "state_pension",
        "period": 2025,
        "family": "dwp_state_pension",
    }
    fields.update(overrides)
    return LedgerTargetReference(**fields)
