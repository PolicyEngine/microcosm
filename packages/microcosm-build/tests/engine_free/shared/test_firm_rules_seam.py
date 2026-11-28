"""The country-neutral firm rules seam: outputs, alignment and totals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.firms import (
    FirmRuleOutputs,
    FirmRules,
    align_rule_outputs,
    weighted_totals,
)


def _outputs(ids, **values) -> FirmRuleOutputs:
    return FirmRuleOutputs(
        firm_id=np.asarray(ids, dtype=np.int64),
        values={name: np.asarray(v, dtype=np.float64) for name, v in values.items()},
        provenance={"rules": "test"},
    )


class _FlatTax:
    """A rules double: 10% of turnover, and a count of one per firm."""

    def evaluate(self, firms: pd.DataFrame, *, period: int) -> FirmRuleOutputs:
        return FirmRuleOutputs(
            firm_id=firms["firm_id"].to_numpy(),
            values={
                "tax": 0.1 * firms["turnover"].to_numpy(np.float64),
                "payers": np.ones(len(firms)),
            },
            provenance={"rules": "flat", "period": str(period)},
        )


def test_outputs_coerce_and_freeze() -> None:
    outputs = FirmRuleOutputs(
        firm_id=np.array([1, 2], dtype=np.int32),
        values={"tax": [1.5, -2.0]},
        provenance={"rules": "test"},
    )
    assert outputs.firm_id.dtype == np.int64
    assert outputs.values["tax"].dtype == np.float64
    with pytest.raises(TypeError):
        outputs.provenance["rules"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        outputs.values["tax"] = np.zeros(2)  # type: ignore[index]


@pytest.mark.parametrize(
    ("ids", "values", "match"),
    [
        ([1.0, 2.0], {"tax": [1.0, 2.0]}, "integer"),
        ([1, 2], {}, "at least one"),
        ([1, 2], {"tax": [1.0]}, "shape"),
        ([1, 2], {"tax": [1.0, np.nan]}, "finite"),
        ([1, 2], {"firm_id": [1.0, 2.0]}, "not allowed"),
        ([1, 2], {"": [1.0, 2.0]}, "not allowed"),
    ],
)
def test_outputs_refuse_malformed_arrays(ids, values, match) -> None:
    with pytest.raises(ValueError, match=match):
        FirmRuleOutputs(firm_id=np.asarray(ids), values=values, provenance={})


def test_align_reorders_every_output_by_firm_id() -> None:
    outputs = _outputs([3, 1, 2], tax=[30.0, 10.0, 20.0], payers=[3.0, 1.0, 2.0])
    aligned = align_rule_outputs(outputs, np.array([1, 2, 3]))
    assert list(aligned) == ["tax", "payers"]
    np.testing.assert_array_equal(aligned["tax"], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(aligned["payers"], [1.0, 2.0, 3.0])
    by_series = align_rule_outputs(outputs, pd.Series([2, 3, 1]))
    np.testing.assert_array_equal(by_series["tax"], [20.0, 30.0, 10.0])


@pytest.mark.parametrize(
    ("produced", "requested", "match"),
    [
        ([1, 2], [1, 2, 3], "miss 1 firm ids"),
        ([1, 2, 3], [1, 2], "unrequested"),
        ([1, 1, 2], [1, 2], "outputs repeat"),
        ([1, 2], [1, 1, 2], "requested firm ids repeat"),
    ],
)
def test_align_refuses_mismatched_ids(produced, requested, match) -> None:
    outputs = _outputs(produced, tax=np.zeros(len(produced)))
    with pytest.raises(ValueError, match=match):
        align_rule_outputs(outputs, np.asarray(requested))


def test_a_rules_double_satisfies_the_protocol_and_aligns() -> None:
    rules: FirmRules = _FlatTax()
    firms = pd.DataFrame({"firm_id": [7, 5], "turnover": [100.0, 40.0]})
    outputs = rules.evaluate(firms, period=2024)
    assert outputs.provenance["period"] == "2024"
    aligned = align_rule_outputs(outputs, np.array([5, 7]))
    np.testing.assert_allclose(aligned["tax"], [4.0, 10.0])


def test_weighted_totals_sum_each_output_alone() -> None:
    table = pd.DataFrame(
        {
            "firm_id": [1, 2],
            "firm_weight": [2.0, 3.0],
            "tax": [10.0, -1.0],
            "payers": [1.0, 0.0],
        }
    )
    assert weighted_totals(table) == {"tax": 17.0, "payers": 2.0}
    renamed = table.rename(columns={"firm_weight": "w", "firm_id": "id"})
    assert weighted_totals(renamed, weight_column="w", id_column="id") == {
        "tax": 17.0,
        "payers": 2.0,
    }
    with pytest.raises(ValueError, match="firm_weight"):
        weighted_totals(table.drop(columns="firm_weight"))
