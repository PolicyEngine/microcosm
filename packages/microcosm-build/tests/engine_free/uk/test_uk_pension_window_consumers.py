"""UK consumers of the pension count x mean window (microcosm#1069, c2)."""

import json

import pytest

from microcosm.build.ledger_targets import (
    MONTHLY_WINDOW_COUNT_X_MEAN,
    compile_ledger_target_references,
)
from microcosm.build.uk_runtime.hmrc_uprating import (
    UK_ENGINE_PARAMETER_INDEX_PREFIX,
    align_hmrc_row_by_engine_index,
)
from microcosm.build.uk_runtime.uc_source_periods import (
    EXPECTED_SOURCE_MONTHS,
    SOURCE_MONTH_FAMILIES,
    uc_source_month_metadata,
    validate_uc_source_month_coverage,
)
from test_support.microcosm_build.ledger_monthly_count_x_mean import (
    MONTHS,
    statx_window,
    window_reference,
)
from tools.generate_uk_target_references import _geography_id_for_target

NSP = "gov.dwp.state_pension.new_state_pension.amount"
NSP_INDEX = f"{UK_ENGINE_PARAMETER_INDEX_PREFIX}{NSP}"
# The engine keys each tax year's weekly rate at 1 January of the year it
# opens: 221.20 for 2024-25 and 230.25 for 2025-26.
RATES = {"2024-01-01": 221.20, "2025-01-01": 230.25}
COUNTS = (12_000_000.0, 12_050_000.0, 12_100_000.0, 12_150_000.0)
MEANS = (221.0, 230.0, 230.5, 231.0)


def _rate(path: str, instant: str) -> float:
    assert path == NSP
    return RATES[instant]


def _compiled(**overrides):
    reference = window_reference(uprating_index=NSP_INDEX, **overrides)
    registry = compile_ledger_target_references(
        statx_window(COUNTS, MEANS), [reference], country="uk"
    )
    return reference, registry


def test_each_window_month_moves_to_the_rate_the_engine_pays_in_2025() -> None:
    reference, compiled = _compiled()

    (spec,) = align_hmrc_row_by_engine_index(
        reference, compiled, parameter_path=NSP, parameter_value=_rate
    ).specs

    # Only February is paid at the 2024-25 rate; May onwards already at 2025-26.
    factors = (230.25 / 221.20, 1.0, 1.0, 1.0)
    expected = (
        sum(
            count * mean * 52 * factor
            for count, mean, factor in zip(COUNTS, MEANS, factors, strict=True)
        )
        / 4
    )
    assert spec.value == pytest.approx(expected, rel=1e-12)
    members = json.loads(spec.metadata["uprating_window_members"])
    assert [member["uprating_from_instant"] for member in members] == [
        "2024-01-01",
        "2025-01-01",
        "2025-01-01",
        "2025-01-01",
    ]
    assert float(members[0]["uprating_factor"]) == pytest.approx(factors[0])
    assert spec.metadata["uprating_index_to_instant"] == "2025-01-01"
    assert spec.metadata["ledger_value_before_alignment"] == (
        f"{compiled.specs[0].value:.15g}"
    )
    assert "R1" in spec.metadata["uprating_adjudication"]


def test_a_window_already_at_the_target_rate_is_unchanged() -> None:
    reference, compiled = _compiled()
    flat = {"2024-01-01": 230.25, "2025-01-01": 230.25}

    (spec,) = align_hmrc_row_by_engine_index(
        reference,
        compiled,
        parameter_path=NSP,
        parameter_value=lambda _path, instant: flat[instant],
    ).specs

    assert spec.value == compiled.specs[0].value
    assert spec.metadata["uprating_factor"] == "1"


def test_a_window_month_after_the_calibration_year_is_refused() -> None:
    _, compiled = _compiled()

    with pytest.raises(ValueError, match="never moves a value backwards"):
        align_hmrc_row_by_engine_index(
            window_reference(uprating_index=NSP_INDEX, period=2024),
            compiled,
            parameter_path=NSP,
            parameter_value=_rate,
        )


def test_the_pension_families_declare_their_source_months() -> None:
    assert {
        "dwp_state_pension",
        "dwp_pension_credit",
        "dwp_attendance_allowance",
    } <= SOURCE_MONTH_FAMILIES
    metadata = uc_source_month_metadata(
        list(MONTHS), value_operation=MONTHLY_WINDOW_COUNT_X_MEAN
    )
    reference = window_reference(metadata=metadata)
    facts = statx_window(COUNTS, MEANS)
    registry = compile_ledger_target_references(facts, [reference], country="uk")

    (spec,) = validate_uc_source_month_coverage(reference, registry, facts).specs

    # Each month resolves its count and its mean.
    assert spec.metadata["uk_uc_source_month_count"] == "4"
    assert spec.metadata["uk_uc_source_period_basis"] == (
        "mean_of_declared_monthly_count_x_mean"
    )
    assert json.loads(spec.metadata[EXPECTED_SOURCE_MONTHS]) == list(MONTHS)


@pytest.mark.parametrize(
    ("target_id", "geography_id"),
    [
        ("dfc_ni.state_pension.recipients", "N92000002"),
        ("dfc_ni.pension_credit.benefit_units", "N92000002"),
        ("dwp.winter_fuel_payment.recipients", "K04000001"),
        ("dwp.state_pension.recipients", "K03000001"),
    ],
)
def test_pension_targets_pin_their_publishers_geography(
    target_id: str, geography_id: str
) -> None:
    target = {"target_id": target_id, "ledger_selector": {}}

    assert _geography_id_for_target(target) == geography_id
