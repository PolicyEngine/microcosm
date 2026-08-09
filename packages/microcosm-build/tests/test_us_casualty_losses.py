"""Contracts for the IRS PUF casualty-loss input restoration."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import puf_support as puf_support_module
from microcosm.build.us_runtime.casualty_losses import (
    derive_us_casualty_loss_from_puf,
    us_casualty_loss_signal_gate,
    us_casualty_loss_stage_spec,
)
from microcosm.build.us_runtime.puf_aggregate_records import (
    _reconcile_puf_casualty_loss_from_source,
)

policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)


class _ResolvedWeights:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _PersonFrame:
    def __init__(
        self,
        person: pd.DataFrame,
        weights: np.ndarray | None = None,
    ) -> None:
        self._person = person
        self._weights = np.ones(len(person)) if weights is None else weights

    def table(self, entity: str) -> pd.DataFrame:
        assert entity == "person"
        return self._person

    def resolve_weights(self, entity: str) -> _ResolvedWeights:
        assert entity == "person"
        return _ResolvedWeights(np.asarray(self._weights, dtype=np.float64))


def test_archived_puf_mapping_is_an_exact_carry() -> None:
    source = pd.DataFrame({"E20500": [0.0, 125.5, 9_000.0]})

    result = derive_us_casualty_loss_from_puf(source)

    assert "casualty_loss" not in source.columns
    assert result["casualty_loss"].tolist() == [0.0, 125.5, 9_000.0]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (pd.DataFrame({"other": [1.0]}), "requires source column"),
        (pd.DataFrame({"E20500": ["not numeric"]}), "nonnumeric or nonfinite"),
        (pd.DataFrame({"E20500": [np.inf]}), "nonnumeric or nonfinite"),
        (pd.DataFrame({"E20500": [-1.0]}), "negative value"),
    ],
)
def test_source_derivation_fails_closed(
    source: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        derive_us_casualty_loss_from_puf(source)


def test_post_disaggregation_reconciliation_uses_final_e20500() -> None:
    source = pd.DataFrame(
        {
            "E20500": [0.0, 3_000.0, 7_500.0],
            "casualty_loss": [999.0, 999.0, 999.0],
        }
    )

    result = _reconcile_puf_casualty_loss_from_source(source)

    assert result["casualty_loss"].tolist() == [0.0, 3_000.0, 7_500.0]


def test_shared_puf_stage_declares_exact_source_and_output() -> None:
    stage = us_casualty_loss_stage_spec()
    operation = next(
        operation
        for operation in stage.operations
        if operation.kind == "derive_puf_policyengine_variables"
    )

    assert operation.parameters["casualty_loss_source"] == "E20500"
    assert operation.parameters["casualty_loss_output"] == "casualty_loss"
    assert "casualty_loss" in stage.outputs
    assert "casualty_loss" in stage.nonnegative_outputs
    assert any("E20500" in str(artifact.get("locator")) for artifact in stage.artifacts)


def test_puf_support_keeps_casualty_loss_sparse_and_earnings_distributed() -> None:
    assert "casualty_loss" in puf_support_module.PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "casualty_loss" in puf_support_module._PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS
    assert "casualty_loss" in puf_support_module._PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS
    assert puf_support_module._PERSON_OUTPUT_DISTRIBUTION_BASIS["casualty_loss"] == (
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
    )


def test_signal_gate_accepts_plausibly_sparse_nondefault_values() -> None:
    values = np.zeros(1_000)
    values[[10, 20, 30]] = [1_000.0, 2_000.0, 3_000.0]
    frame = _PersonFrame(pd.DataFrame({"casualty_loss": values}))

    result = us_casualty_loss_signal_gate(frame)  # type: ignore[arg-type]

    assert result.passed, result.failures
    assert result.details["positive_share"] == pytest.approx(0.003)


@pytest.mark.parametrize(
    "person",
    [
        pd.DataFrame({"other": [0.0, 1.0]}),
        pd.DataFrame({"casualty_loss": [0.0, 0.0]}),
        pd.DataFrame({"casualty_loss": [0.0, -1.0]}),
        pd.DataFrame({"casualty_loss": [0.0, np.nan]}),
    ],
)
def test_signal_gate_rejects_missing_default_or_invalid_surface(
    person: pd.DataFrame,
) -> None:
    result = us_casualty_loss_signal_gate(  # type: ignore[arg-type]
        _PersonFrame(person)
    )

    assert not result.passed


@requires_us
def test_policyengine_us_contract_is_a_person_year_input_leaf() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    variable = CountryTaxBenefitSystem().variables["casualty_loss"]

    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"
    assert variable.default_value == 0
