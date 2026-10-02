"""The monthly count x mean window and DWP's half-open amount bands (microcosm#1069)."""

import json
from types import SimpleNamespace

import pytest

from microcosm.build.ledger_targets import (
    MONTHLY_WINDOW_COUNT_X_MEAN,
    LedgerTargetReference,
    compile_ledger_target_references,
)
from microcosm.build.target_materialization import (
    _band_bounds,
    _band_label_upper_edge,
    _band_lower_edge,
)
from test_support.microcosm_build.ledger_monthly_count_x_mean import (
    COUNT,
    MEAN,
    MONTHS,
    statx_fact,
    statx_window,
    window_reference,
)


def test_window_averages_count_times_mean_times_the_period_factor() -> None:
    counts = (12_000_000, 12_050_000, 12_100_000, 12_150_000)
    means = (221.0, 230.0, 230.5, 231.0)

    registry = compile_ledger_target_references(
        list(reversed(statx_window(counts, means))), [window_reference()], country="uk"
    )

    (spec,) = registry.specs
    expected = sum(c * m * 52 for c, m in zip(counts, means, strict=True)) / 4
    assert spec.value == pytest.approx(expected)
    assert spec.metadata["ledger_value_operation"] == MONTHLY_WINDOW_COUNT_X_MEAN
    assert spec.metadata["ledger_member_fact_count"] == "8"
    assert spec.metadata["ledger_window_count_x_mean_period_factor"] == "52"
    assert spec.metadata["ledger_window_count_x_mean_count_concept"] == COUNT
    assert spec.metadata["ledger_window_count_x_mean_mean_concept"] == MEAN
    assert spec.metadata["ledger_measure_unit"] == "gbp_per_week x 52"
    members = json.loads(spec.metadata["ledger_window_count_x_mean_members"])
    assert [member["month"] for member in members] == list(MONTHS)
    assert float(members[0]["count"]) == counts[0]
    assert float(members[0]["mean"]) == means[0]
    assert spec.metadata["ledger_source_release_key"] == "stat_xplore_2026_09_29"


def test_window_refuses_a_month_without_its_mean() -> None:
    facts = statx_window((1.0,) * 4, (2.0,) * 4)
    facts = [
        fact
        for fact in facts
        if not (fact["period"]["value"] == "2025-08" and "mean" in str(fact["label"]))
    ]

    with pytest.raises(ValueError, match=r"missing \['2025-08 mean'\]"):
        compile_ledger_target_references(facts, [window_reference()], country="uk")


def test_window_refuses_a_count_and_mean_from_different_cells() -> None:
    facts = statx_window((1.0,) * 4, (2.0,) * 4)
    # Every mean describes another group, so each role is one clean series
    # and only the pairing can catch it.
    for fact in facts[1::2]:
        fact["layout"]["groupby_value_id"] = "new_state_pension"

    with pytest.raises(ValueError, match="different published cells in 2025-02"):
        compile_ledger_target_references(facts, [window_reference()], country="uk")


def test_window_refuses_a_role_that_changes_series_inside_the_window() -> None:
    facts = statx_window((1.0,) * 4, (2.0,) * 4)
    facts[3]["layout"]["groupby_value_id"] = "new_state_pension"

    with pytest.raises(ValueError, match="more than one series for a role"):
        compile_ledger_target_references(facts, [window_reference()], country="uk")


def test_window_refuses_members_from_two_publications() -> None:
    facts = statx_window((1.0,) * 4, (2.0,) * 4)
    facts[-1]["source"]["source_sha256"] = "b" * 64

    with pytest.raises(ValueError, match="one publication"):
        compile_ledger_target_references(facts, [window_reference()], country="uk")


def test_window_refuses_a_selected_fact_of_neither_concept() -> None:
    reference = window_reference(
        value_operands=(
            {"role": "count", "concept": "dwp.pension_credit_benefit_units"},
            {"role": "mean", "concept": MEAN, "period_factor": 52},
        )
    )

    with pytest.raises(ValueError, match="neither operand's"):
        compile_ledger_target_references(
            statx_window((1.0,) * 4, (2.0,) * 4), [reference], country="uk"
        )


def test_window_refuses_a_count_fact_published_as_a_mean() -> None:
    facts = statx_window((1.0,) * 4, (2.0,) * 4)
    for fact in facts[0::2]:
        fact["aggregation"] = {"method": "mean"}

    with pytest.raises(ValueError, match="unsupported aggregation"):
        compile_ledger_target_references(facts, [window_reference()], country="uk")


@pytest.mark.parametrize(
    ("operands", "match"),
    [
        (
            (
                {"role": "mean", "concept": MEAN},
                {"role": "count", "concept": COUNT},
            ),
            "exactly ordered count/mean",
        ),
        (
            (
                {"role": "count", "concept": COUNT},
                {"role": "mean", "concept": COUNT},
            ),
            "different concepts",
        ),
        (
            (
                {"role": "count", "concept": COUNT, "period_factor": 52},
                {"role": "mean", "concept": MEAN},
            ),
            "takes only",
        ),
        (
            (
                {"role": "count", "concept": COUNT},
                {"role": "mean", "concept": MEAN, "period_factor": 0},
            ),
            "finite positive",
        ),
        (
            (
                {"role": "count", "concept": COUNT},
                {"role": "mean", "concept": MEAN, "period_factor": True},
            ),
            "finite positive",
        ),
        (
            ({"role": "count", "concept": COUNT}, {"role": "mean", "concept": ""}),
            "must name its concept",
        ),
    ],
)
def test_window_operands_are_validated_on_thewindow_reference(operands, match) -> None:
    with pytest.raises(ValueError, match=match):
        window_reference(value_operands=operands)


def _band_spec(label: str) -> SimpleNamespace:
    return SimpleNamespace(
        measure="dwp/state_pension/recipients_by_amount",
        metadata={"ledger_filter_grouped_amount_of_benefit": label},
    )


@pytest.mark.parametrize(
    ("label", "lower", "upper"),
    [
        ("Under £20.00", 0.0, 1_040.0),
        ("£20.00 to under £40.00", 1_040.0, 2_080.0),
        ("£300.00 and over", 15_600.0, None),
        ("£500.01 to £600.00", 26_000.52, None),
        ("Under 25", None, None),
        ("Nil payment", None, None),
        ("all", None, None),
    ],
)
def test_half_open_amount_labels_publish_their_own_upper_edge(
    label, lower, upper
) -> None:
    binding = {"groupby_variable": "state_pension", "band_period_factor": 52}

    assert _band_lower_edge(_band_spec(label), binding) == (
        pytest.approx(lower) if lower is not None else None
    )
    assert _band_label_upper_edge(_band_spec(label), binding) == (
        pytest.approx(upper) if upper is not None else None
    )


def test_a_half_open_band_keeps_its_published_edge_when_a_sibling_is_missing() -> None:
    binding = {"groupby_variable": "state_pension", "band_period_factor": 52}
    # The £40 to £60 band is not bound, so the sibling rule would stretch the
    # £20 to £40 band up to £60.
    edges = [0.0, 1_040.0, 3_120.0]

    assert _band_bounds(_band_spec("£20.00 to under £40.00"), binding, edges) == (
        pytest.approx(1_040.0),
        pytest.approx(2_080.0),
    )
    assert _band_bounds(_band_spec("Under £20.00"), binding, edges) == (
        0.0,
        pytest.approx(1_040.0),
    )


def test_window_requires_the_source_window_period_policy() -> None:
    # Its months are declared, like the single-series windows, so the target
    # period never re-selects them (and never holds them for uprating).
    with pytest.raises(ValueError, match="paired source_window period policy"):
        window_reference(period_match_policy="latest_not_after")


def _single_age_fact(month: str, age: int, value: float) -> dict[str, object]:
    fact = statx_fact(month, "count", value, group="new_state_pension")
    fact["aggregate_fact_key"] = f"{fact['aggregate_fact_key']}-{age}"
    fact["semantic_fact_key"] = f"{fact['semantic_fact_key']}-{age}"
    fact["dimensions"] = {
        "age_bands_and_single_year": age,
        "category_of_pension": "New State Pension",
    }
    fact["universe_constraints"] = {
        "domain": "social_security",
        "constraints": [
            {"variable": "age", "operator": ">=", "value": age, "unit": "years"},
            {"variable": "age", "operator": "<", "value": age + 1, "unit": "years"},
            {
                "variable": "age_bands_and_single_year",
                "operator": "==",
                "value": age,
            },
            {
                "variable": "category_of_pension",
                "operator": "==",
                "value": "New State Pension",
            },
        ],
    }
    return fact


def _single_age_window(**overrides) -> LedgerTargetReference:
    return LedgerTargetReference(
        name="dwp/state_pension/recipients_66_67",
        ledger_selector={
            "source_name": "dwp",
            "source_concept": "dwp.state_pension_caseload",
            "dimension_values": {
                "age_bands_and_single_year": [66, 67],
                "category_of_pension": "New State Pension",
            },
            "period_type": "month",
            "period_value": list(MONTHS),
        },
        value_operation="monthly_window_sum_average",
        value_operands=tuple(
            {
                "dimension_values": {
                    "age_bands_and_single_year": age,
                    "category_of_pension": "New State Pension",
                }
            }
            for age in (66, 67)
        ),
        period_match_policy="source_window",
        entity="person",
        measure="person_count",
        period=2025,
        family="dwp_state_pension",
        **overrides,
    )


def test_a_sum_window_over_single_years_accepts_each_cells_published_age_edges() -> (
    None
):
    facts = [
        _single_age_fact(month, age, value)
        for month in MONTHS
        for age, value in ((66, 100.0), (67, 50.0))
    ]

    (spec,) = compile_ledger_target_references(
        facts, [_single_age_window()], country="uk"
    ).specs

    assert spec.value == 150.0


def test_a_sum_window_still_refuses_cells_from_different_universes() -> None:
    facts = [_single_age_fact(month, age, 1.0) for month in MONTHS for age in (66, 67)]
    for fact in facts:
        if fact["dimensions"]["age_bands_and_single_year"] == 67:
            fact["universe_constraints"]["constraints"].append(
                {"variable": "residence", "operator": "==", "value": "abroad"}
            )

    with pytest.raises(ValueError, match="cells must share source publication"):
        compile_ledger_target_references(facts, [_single_age_window()], country="uk")
