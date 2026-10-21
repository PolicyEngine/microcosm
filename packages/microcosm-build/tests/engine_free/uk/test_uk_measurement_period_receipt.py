"""The measurement-period receipt names every off-period stored read (#1324)."""

from __future__ import annotations

from microcosm.build.uk_runtime.measurement_period import uk_value_read_routes

CONTRACT = {
    "spi": {"bindings": {"policyengine": {"value_variable": "dividend_income"}}},
    "fuel": {
        "bindings": {
            "policyengine": {
                "value_expression": "petrol_spending + diesel_spending",
                "measurement_period": 2024,
            }
        }
    },
    "tax": {"bindings": {"policyengine": {"value_variable": "income_tax"}}},
    "count": {"bindings": {"policyengine": {"value_variable": "person_count"}}},
    "after_relief": {
        "bindings": {
            "policyengine": {
                "value_variable": "self_employment_income_after_loss_relief"
            }
        }
    },
    "gated": {
        "bindings": {
            "policyengine": {
                "kind": "parameter_gated_threshold",
                "value_variable": "dividend_income",
            }
        }
    },
}
ROWS = [
    ("spi@2025", "spi", "person"),
    ("fuel@2025", "fuel", "household"),
    ("tax@2025", "tax", "person"),
    ("count@2025", "count", "person"),
    ("after_relief@2025", "after_relief", "person"),
    ("gated@2025", "gated", "person"),
]
STORED = {
    "person": ["dividend_income", "self_employment_income"],
    "household": ["petrol_spending", "diesel_spending"],
}


def _receipt(attached):
    return uk_value_read_routes(
        CONTRACT,
        ROWS,
        stored_columns=STORED,
        frame_period="2024",
        calibration_period=2025,
        attached=attached,
    )


def test_before_the_fix_the_stored_reads_are_named():
    receipt = _receipt(
        {
            "person.income_tax": "native",
            "person.self_employment_income_after_loss_relief": (
                "frame_columns:self_employment_income_less_loss_relief"
            ),
        }
    )

    assert receipt["stored_reads_off_period"] == ["after_relief@2025", "spi@2025"]
    routes = {row["target"]: row["value_routes"] for row in receipt["rows"]}
    assert routes["fuel@2025"] == {
        "petrol_spending": "frame_period_declared",
        "diesel_spending": "frame_period_declared",
    }
    assert routes["tax@2025"] == {"income_tax": "provider:native"}
    assert routes["count@2025"] == {"person_count": "count"}
    # Provider kinds measure their own population and are left out.
    assert "gated@2025" not in routes


def test_after_the_fix_no_stored_read_is_off_period():
    receipt = _receipt(
        {
            "person.income_tax": "native",
            "person.dividend_income": "native",
            "person.self_employment_income_after_loss_relief": (
                "engine_components:self_employment_income_less_loss_relief"
            ),
        }
    )

    assert receipt["stored_reads_off_period"] == []
    routes = {row["target"]: row["value_routes"] for row in receipt["rows"]}
    assert routes["spi@2025"] == {"dividend_income": "provider:native"}
    # The receipt carries names, periods and route labels, never a figure.
    assert set(receipt) == {
        "schema_version",
        "frame_period",
        "calibration_period",
        "rows",
        "stored_reads_off_period",
    }
