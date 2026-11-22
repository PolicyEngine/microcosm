"""The UK firm rules seam: outputs, alignment, periods and the legacy adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_rules import (
    UK_FIRM_RULE_PERIOD_BY_VINTAGE,
    UK_FIRM_VINTAGE_BY_RULE_PERIOD,
    LegacyVATEvaluatorRules,
    UKFirmRuleOutputs,
    align_rule_outputs,
    rule_period_for_vintage,
)
from microcosm.build.uk_runtime.firm_table import (
    ASSUMED_INPUT_VAT_RATE,
    build_candidate_rules_table,
)


class _RecordingEvaluator:
    """A VAT double: 20% of supplies less 20% of purchases, in £k."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def net_liability_k(
        self, *, turnover_k, input_cost_k, vat_registered, data_vintage
    ):
        self.calls.append(
            {
                "turnover_k": np.asarray(turnover_k),
                "input_cost_k": np.asarray(input_cost_k),
                "vat_registered": np.asarray(vat_registered),
                "data_vintage": data_vintage,
            }
        )
        liability = 0.2 * np.asarray(turnover_k) - 0.2 * np.asarray(input_cost_k)
        return np.where(vat_registered, liability, 0.0).astype(np.float32)


def _outputs(ids, values) -> UKFirmRuleOutputs:
    return UKFirmRuleOutputs(
        firm_id=np.asarray(ids, dtype=np.int64),
        vat_liability_if_registered=np.asarray(values, dtype=np.float64),
        provenance={"rules": "test"},
    )


def _candidates() -> pd.DataFrame:
    return build_candidate_rules_table(
        pd.DataFrame(
            {
                "firm_id": [3, 1, 2],
                "sic_code": ["00011", "00047", "00086"],
                "annual_turnover_k": [100.0, 50.0, 2000.0],
                "annual_input_k": [60.0, 10.0, 1500.0],
                "employment": [4, 1, 30],
            }
        )
    )


@pytest.mark.parametrize(("vintage", "period"), [("2024-25", 2024), ("2023-24", 2023)])
def test_rule_period_is_the_tax_years_start_year(vintage, period) -> None:
    assert rule_period_for_vintage(vintage) == period
    assert UK_FIRM_VINTAGE_BY_RULE_PERIOD[period] == vintage


def test_the_inverse_period_view_is_derived_from_the_vintages() -> None:
    assert dict(UK_FIRM_VINTAGE_BY_RULE_PERIOD) == {
        period: vintage for vintage, period in UK_FIRM_RULE_PERIOD_BY_VINTAGE.items()
    }


def test_an_unknown_vintage_has_no_rule_period() -> None:
    with pytest.raises(ValueError, match="2025-26"):
        rule_period_for_vintage("2025-26")


def test_outputs_coerce_and_freeze() -> None:
    outputs = UKFirmRuleOutputs(
        firm_id=np.array([1, 2], dtype=np.int32),
        vat_liability_if_registered=[1.5, -2.0],
        provenance={"rules": "test"},
    )
    assert outputs.firm_id.dtype == np.int64
    assert outputs.vat_liability_if_registered.dtype == np.float64
    with pytest.raises(TypeError):
        outputs.provenance["rules"] = "other"  # type: ignore[index]


@pytest.mark.parametrize(
    ("ids", "values", "match"),
    [
        ([1.0, 2.0], [1.0, 2.0], "integer"),
        ([1, 2], [1.0], "liabilities"),
        ([1, 2], [1.0, np.nan], "finite"),
    ],
)
def test_outputs_refuse_malformed_arrays(ids, values, match) -> None:
    with pytest.raises(ValueError, match=match):
        UKFirmRuleOutputs(
            firm_id=np.asarray(ids),
            vat_liability_if_registered=np.asarray(values),
            provenance={},
        )


def test_align_reorders_by_firm_id() -> None:
    outputs = _outputs([3, 1, 2], [30.0, 10.0, 20.0])
    np.testing.assert_array_equal(
        align_rule_outputs(outputs, np.array([1, 2, 3])), [10.0, 20.0, 30.0]
    )
    np.testing.assert_array_equal(
        align_rule_outputs(outputs, pd.Series([2, 3, 1])), [20.0, 30.0, 10.0]
    )


@pytest.mark.parametrize(
    ("produced", "requested", "match"),
    [
        ([1, 2], [1, 2, 3], "miss 1 firm ids"),
        ([1, 2, 3], [1, 2], "unrequested"),
        ([1, 1, 2], [1, 2], "outputs repeat"),
        ([1, 2], [1, 1, 2], "Requested UK firm ids repeat"),
    ],
)
def test_align_refuses_mismatched_ids(produced, requested, match) -> None:
    outputs = _outputs(produced, np.zeros(len(produced)))
    with pytest.raises(ValueError, match=match):
        align_rule_outputs(outputs, np.asarray(requested))


def test_legacy_adapter_converts_units_and_registers_every_firm() -> None:
    evaluator = _RecordingEvaluator()
    candidates = _candidates()
    outputs = LegacyVATEvaluatorRules(evaluator).evaluate(candidates, period=2024)

    (call,) = evaluator.calls
    assert call["data_vintage"] == "2024-25"
    assert call["vat_registered"].all()
    np.testing.assert_allclose(call["turnover_k"], [50.0, 2000.0, 100.0])
    np.testing.assert_allclose(call["input_cost_k"], [10.0, 1500.0, 60.0])
    np.testing.assert_array_equal(outputs.firm_id, [1, 2, 3])
    np.testing.assert_allclose(
        outputs.vat_liability_if_registered, [8_000.0, 100_000.0, 8_000.0], rtol=1e-6
    )
    assert outputs.provenance["evaluator"] == "_RecordingEvaluator"
    assert outputs.provenance["data_vintage"] == "2024-25"
    assert outputs.provenance["period"] == "2024"


def test_legacy_adapter_agrees_with_the_tables_input_vat() -> None:
    """The legacy double's liability equals output VAT less ``firm_input_vat``."""

    candidates = _candidates()
    outputs = LegacyVATEvaluatorRules(_RecordingEvaluator()).evaluate(
        candidates, period=2023
    )
    expected = (
        ASSUMED_INPUT_VAT_RATE * candidates["firm_standard_rated_supplies"]
        - candidates["firm_input_vat"]
    ).to_numpy()
    np.testing.assert_allclose(
        align_rule_outputs(outputs, candidates["firm_id"]), expected, rtol=1e-6
    )


def test_legacy_adapter_refuses_an_unknown_period_and_missing_columns() -> None:
    rules = LegacyVATEvaluatorRules(_RecordingEvaluator())
    with pytest.raises(ValueError, match="rule period 2026"):
        rules.evaluate(_candidates(), period=2026)
    with pytest.raises(ValueError, match="firm_input_costs"):
        rules.evaluate(_candidates().drop(columns="firm_input_costs"), period=2024)


def test_legacy_adapter_refuses_a_wrong_shape() -> None:
    class _Short:
        def net_liability_k(self, **_kwargs):
            return np.zeros(1)

    with pytest.raises(ValueError, match="returned shape"):
        LegacyVATEvaluatorRules(_Short()).evaluate(_candidates(), period=2024)
