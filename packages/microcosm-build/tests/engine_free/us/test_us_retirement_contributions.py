"""ASEC retirement-contribution restoration and PUF-half QRF treatment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.retirement_contributions as module
from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime.one_population_contract import ACS_SUPPORT_CHANNEL
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.retirement_contributions import (
    US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
    US_RETIREMENT_CONTRIBUTION_REQUIRED_SOURCE_COLUMNS,
    derive_us_retirement_contributions_from_manifest,
    us_retirement_contributions_signal_gate,
    us_retirement_contributions_stage_spec,
    us_retirement_contributions_summary,
    with_us_retirement_contribution_inputs,
)
from microcosm.build.us_runtime.support_provenance import (
    support_channel_column,
    support_clone_index_column,
)
from microcosm.frame import Frame
from test_support.microcosm_build.us_one_population_contract import (
    mixed_origin_reader_population,
)
from test_support.microcosm_build.us_retirement_contributions import (
    _frame,
    _person_source,
    _stacked_frame,
)


def _derive(frame: pd.DataFrame) -> pd.DataFrame:
    operation = next(
        operation
        for operation in us_retirement_contributions_stage_spec().operations
        if operation.kind == "derive_retirement_contributions"
    )
    return derive_us_retirement_contributions_from_manifest(frame, operation, None)


def test_stage_manifest_pins_sources_operations_and_five_desired_leaves() -> None:
    spec = us_retirement_contributions_stage_spec()

    assert spec.stage == "retirement_contributions"
    assert spec.grain == "person"
    assert tuple(spec.outputs) == US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "derive_retirement_contributions",
        "impute_retirement_contributions_to_puf_support",
    ]
    derive = spec.operations[1]
    assert derive.parameters == {
        "se_pension_share": 0.046,
        "dc_share_of_remainder": 0.908,
        "roth_dc_share": 0.15,
        "traditional_ira_share": 0.392,
    }


def test_direct_split_matches_archived_wage_and_self_employment_cases() -> None:
    result = _derive(_person_source())

    np.testing.assert_allclose(
        result[list(US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS)].to_numpy(),
        np.asarray(
            [
                [0.7718, 0.1362, 0.036064, 0.055936, 0.0],
                [0.0, 0.0, 0.373968, 0.580032, 0.046],
                [0.7362972, 0.1299348, 0.034405056, 0.053362944, 0.046],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        rtol=0,
        atol=1e-12,
    )


def test_direct_split_fails_closed_without_measured_asec_source() -> None:
    with pytest.raises(SourceRuntimeError, match="RETCB_VAL"):
        _derive(_person_source().drop(columns=["RETCB_VAL"]))


@pytest.mark.parametrize(
    ("bad_value", "message"),
    [(np.nan, "nonfinite"), (-1.0, "negative")],
)
def test_direct_split_rejects_invalid_measured_totals(
    bad_value: float,
    message: str,
) -> None:
    person = _person_source()
    person.loc[0, "RETCB_VAL"] = bad_value

    with pytest.raises(SourceRuntimeError, match=message):
        _derive(person)


def test_with_inputs_materializes_signal_and_preserves_reported_total() -> None:
    result = with_us_retirement_contribution_inputs(
        _frame(),
        seed=0,
        time_period=2024,
    )

    gate = us_retirement_contributions_signal_gate(result)
    assert gate.passed, gate.failures
    person = result.table("person")
    allocated = person[list(US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS)].sum(axis=1)
    np.testing.assert_allclose(allocated.iloc[:3], person["RETCB_VAL"].iloc[:3])
    assert allocated.iloc[3] == 0.0


def test_stacked_gate_validates_physical_source_and_reconciles_direct_role() -> None:
    stacked = _stacked_frame()

    gate = us_retirement_contributions_signal_gate(stacked)

    assert gate.passed, gate.failures
    assert gate.details["source_rows"] == 4
    assert gate.details["source_reconciliation_rows"] == 2
    assert gate.details["allocation_mismatch_rows"] == 0

    person = stacked.table("person")
    asec_puf_role = person["person_support_channel"].eq("asec") & person[
        "person_support_clone_index"
    ].eq(1)
    person.loc[person.index[asec_puf_role][0], "RETCB_VAL"] = np.nan
    with pytest.raises(SourceRuntimeError, match="RETCB_VAL"):
        us_retirement_contributions_summary(stacked)


def test_acs_retcb_cells_never_move_contribution_reconciliation() -> None:
    """On a mixed-origin frame, no ACS ``RETCB_VAL`` reaches the source read
    or the reconciliation.

    d2283's one-population join adds ACS rows, which hold no ``RETCB_VAL``,
    ``WSAL_VAL`` or ``SEMP_VAL``, to Route A's CPS rows. The assembled source
    mask takes every CPS row, native or clone, and leaves the ACS rows out.
    The reconciliation takes each source row whose stored channel equals its
    role, and here every CPS row qualifies. A null, an infinite, a negative
    or a positive ``RETCB_VAL`` in every ACS row, beside any wage and
    self-employment amounts, leaves the verdict and every statistic
    unchanged. Inside the source mask the null and the infinity would raise
    and any value would move ``source_rows``. Inside the reconciliation the
    negative and positive amounts would move ``source_total``, and the
    positive one would count as a positive row and a mismatch, since every
    ACS row holds 0 in each output.
    """

    mixed = mixed_origin_reader_population().mixed_origin
    person = mixed.table("person")
    acs = person[support_channel_column("person")].eq(ACS_SUPPORT_CHANNEL).to_numpy()
    native = person[support_clone_index_column("person")].to_numpy() == 0
    sources = list(US_RETIREMENT_CONTRIBUTION_REQUIRED_SOURCE_COLUMNS)
    outputs = list(US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS)
    assert 0 < int(acs.sum()) < len(person)
    assert native[acs].all()
    assert person.loc[acs, sources].isna().all().all()
    assert (person.loc[acs, outputs] == 0.0).all().all()

    baseline = us_retirement_contributions_signal_gate(mixed)
    assert baseline.passed, baseline.failures
    cps_rows = int(np.count_nonzero(~acs))
    assert baseline.details["source_rows"] == cps_rows
    assert baseline.details["source_reconciliation_rows"] == cps_rows
    assert baseline.details["source_positive_rows"] > 0

    for retcb, wages, self_employment in (
        (np.nan, np.nan, np.nan),
        (np.inf, 40_000.0, 0.0),
        (-5.0, 0.0, 20_000.0),
        (1_000.0, 40_000.0, 20_000.0),
    ):
        tables = {entity: mixed.table(entity).copy() for entity in mixed.entities}
        tables["person"].loc[acs, "RETCB_VAL"] = retcb
        tables["person"].loc[acs, "WSAL_VAL"] = wages
        tables["person"].loc[acs, "SEMP_VAL"] = self_employment
        changed = Frame(
            tables,
            mixed.schema,
            {entity: mixed.weights_for(entity) for entity in mixed.weighted_entities},
            mixed.strata,
            mass_log=mixed.mass_log,
            metadata=mixed.metadata,
        )

        summary = us_retirement_contributions_summary(changed)
        gate = us_retirement_contributions_signal_gate(changed)

        assert summary == baseline.details, retcb
        assert (gate.passed, gate.failures, gate.details) == (
            baseline.passed,
            baseline.failures,
            baseline.details,
        ), retcb


def test_puf_half_uses_qrf_predictions_and_applies_income_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = with_us_retirement_contribution_inputs(
        _frame(),
        seed=0,
        time_period=2024,
    )
    expanded = clone_us_frame_for_puf_support(direct)
    calls: dict[str, object] = {}

    class FakeFitted:
        def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
            calls["test"] = test.copy()
            return pd.DataFrame(
                {
                    column: np.full(len(test), index + 1.0)
                    for index, column in enumerate(
                        US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS
                    )
                },
                index=test.index,
            )

    class FakeQRF:
        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs

        def fit(
            self,
            training: pd.DataFrame,
            predictors: list[str],
            targets: list[str],
            *,
            weights: np.ndarray,
        ) -> FakeFitted:
            calls["training"] = training.copy()
            calls["predictors"] = predictors
            calls["targets"] = targets
            calls["weights"] = weights.copy()
            return FakeFitted()

    monkeypatch.setattr(module, "QRF", FakeQRF)

    result = with_us_retirement_contribution_inputs(
        expanded,
        seed=7,
        time_period=2024,
    )

    assert calls["init"] == {"n_estimators": 100, "seed": 7}
    assert len(calls["training"]) == 4
    assert len(calls["test"]) == 4
    person = result.table("person")
    puf = person[person["person_support_channel"] == "puf_tax_detail"]
    # Fake predictions are 1,2,3,4,5. The no-wage rows lose both 401(k)
    # values, and the no-self-employment rows lose the SE-pension value.
    np.testing.assert_allclose(
        puf[list(US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS)].to_numpy(),
        np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 0.0],
                [0.0, 0.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [0.0, 0.0, 3.0, 4.0, 0.0],
            ]
        ),
    )


def test_signal_gate_rejects_a_default_leaf() -> None:
    result = with_us_retirement_contribution_inputs(
        _frame(),
        seed=0,
        time_period=2024,
    )
    result.table("person")["roth_ira_contributions_desired"] = 0.0

    gate = us_retirement_contributions_signal_gate(result)

    assert not gate.passed
    assert any("roth_ira_contributions_desired" in failure for failure in gate.failures)
