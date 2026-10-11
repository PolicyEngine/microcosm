"""ASEC retirement-contribution restoration and PUF-half QRF treatment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.retirement_contributions import (
    US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
    derive_us_retirement_contributions_from_manifest,
    us_retirement_contributions_signal_gate,
    us_retirement_contributions_stage_spec,
    us_retirement_contributions_summary,
    with_us_retirement_contribution_inputs,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _person_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3, 4], dtype="int64"),
            "person_household_id": np.asarray([10, 20, 30, 40], dtype="int64"),
            "person_tax_unit_id": np.asarray([100, 200, 300, 400], dtype="int64"),
            "person_spm_unit_id": np.asarray([1_000, 2_000, 3_000, 4_000]),
            "person_family_id": np.asarray([10_000, 20_000, 30_000, 40_000]),
            "person_marital_unit_id": np.asarray([100_000, 200_000, 300_000, 400_000]),
            # Wage-only, self-employment-only, both, and neither.
            "RETCB_VAL": [1.0, 1.0, 1.0, 1.0],
            "WSAL_VAL": [50_000.0, 0.0, 50_000.0, 0.0],
            "SEMP_VAL": [0.0, 30_000.0, 30_000.0, 0.0],
            "employment_income_before_lsr": [50_000.0, 0.0, 50_000.0, 0.0],
            "self_employment_income_before_lsr": [0.0, 30_000.0, 30_000.0, 0.0],
            "age": [35, 45, 55, 25],
            "is_female": [False, True, False, True],
            "has_esi": [True, False, True, False],
            "tax_unit_role_input": ["HEAD", "HEAD", "HEAD", "HEAD"],
            "social_security_retirement": [0.0, 0.0, 2_000.0, 0.0],
            "social_security_disability": [0.0, 0.0, 0.0, 0.0],
            "social_security_dependents": [0.0, 0.0, 0.0, 0.0],
            "social_security_survivors": [0.0, 0.0, 0.0, 0.0],
        }
    )


def _frame() -> Frame:
    person = _person_source()
    ids = {
        "household": [10, 20, 30, 40],
        "tax_unit": [100, 200, 300, 400],
        "spm_unit": [1_000, 2_000, 3_000, 4_000],
        "family": [10_000, 20_000, 30_000, 40_000],
        "marital_unit": [100_000, 200_000, 300_000, 400_000],
    }
    tables = {
        entity: pd.DataFrame({f"{entity}_id": np.asarray(values, dtype="int64")})
        for entity, values in ids.items()
    }
    tables["person"] = person
    tables["tax_unit"]["filing_status_input"] = [
        "SINGLE",
        "SINGLE",
        "SINGLE",
        "SINGLE",
    ]
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([1.0, 1.0, 1.0, 1.0]),
                WeightKind.DESIGN,
            )
        },
    )


def _donor() -> pd.DataFrame:
    values = [
        [0.85, 0.15, 0, 0, 0],
        [0, 0, 0.4, 0.6, 0],
        [0.4, 0.1, 0.2, 0.1, 0.2],
        [0, 0, 0.4, 0.6, 0],
    ]
    donor = pd.DataFrame(values, columns=US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS)
    return donor.assign(
        donor_id=range(4),
        age=[35, 45, 55, 25],
        has_wages=[True, False, True, False],
        has_self_employment=[False, True, True, False],
        scf_weight=1.0,
    )


def _derive(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    operation = us_retirement_contributions_stage_spec().operations[1]
    frame = frame.assign(person_weight=1.0)
    context = SourceRuntimeContext(
        SourceRuntimeConfig(seed=seed), {"scf_retirement": _donor()}
    )
    return derive_us_retirement_contributions_from_manifest(frame, operation, context)


def _with(frame: Frame, *, seed: int = 0, time_period: int = 2024) -> Frame:
    return with_us_retirement_contribution_inputs(
        frame, seed=seed, time_period=time_period, donor=_donor()
    )


def _stacked_frame() -> Frame:
    direct = _with(
        _frame(),
        seed=0,
        time_period=2024,
    )
    stacked = clone_us_frame_for_puf_support(direct)
    person = stacked.table("person")
    source_record = np.tile(np.arange(4, dtype=np.int64), 2)
    person["person_spine_source_id"] = source_record
    person["person_support_channel"] = np.where(
        source_record < 2,
        "asec",
        "acs",
    )
    acs = person["person_support_channel"].eq("acs")
    person.loc[acs, ["RETCB_VAL", "WSAL_VAL", "SEMP_VAL"]] = np.nan
    return stacked


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
        "donor_table": "scf_retirement",
        "age_breaks": [35, 50, 65],
        "seed_from_build_config": True,
        "roth_dc_share": 0.15,
    }


def test_direct_allocation_preserves_source_joint_cases_and_cps_total() -> None:
    result = _derive(_person_source())

    np.testing.assert_allclose(
        result[list(US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS)].to_numpy(),
        np.asarray(
            [
                [0.85, 0.15, 0, 0, 0],
                [0, 0, 0.4, 0.6, 0],
                [0.4, 0.1, 0.2, 0.1, 0.2],
                [0, 0, 0.4, 0.6, 0],
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
    result = _with(
        _frame(),
        seed=0,
        time_period=2024,
    )

    gate = us_retirement_contributions_signal_gate(result)
    assert gate.passed, gate.failures
    person = result.table("person")
    allocated = person[list(US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS)].sum(axis=1)
    np.testing.assert_allclose(allocated, person["RETCB_VAL"])


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


def test_puf_half_draws_complete_asec_vectors_and_preserves_joint_participation() -> (
    None
):
    direct = _with(_frame())
    expanded = clone_us_frame_for_puf_support(direct)
    result = _with(expanded, seed=7)
    person = result.table("person")
    puf = person[person["person_support_channel"] == "puf_tax_detail"]
    np.testing.assert_allclose(
        puf[list(US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS)].to_numpy(),
        direct.table("person")[
            list(US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS)
        ].to_numpy(),
    )
    assert us_retirement_contributions_signal_gate(result).passed


def test_legacy_valid_split_is_rebuilt_instead_of_passing_through() -> None:
    frame = _frame()
    for column in US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS:
        frame.table("person")[column] = 0.2
    result = _with(frame)
    assert (
        result.metadata["retirement_contributions"]["method"]
        == "weighted_joint_hot_deck_v1"
    )
    assert us_retirement_contributions_signal_gate(result).passed
    assert (
        not result.table("person")["traditional_ira_contributions_desired"]
        .eq(0.2)
        .all()
    )


def test_fixed_overlap_surface_fails_joint_participation_gate() -> None:
    frame = _with(_frame())
    for column in US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS:
        frame.table("person")[column] = [0.2, 0.2, 0.2, 0]
    gate = us_retirement_contributions_signal_gate(frame)
    assert not gate.passed
    assert any("ira_only" in failure for failure in gate.failures)
    assert any("employer_only" in failure for failure in gate.failures)


@pytest.mark.parametrize("seed", range(5))
def test_bundled_totals_reconcile_for_varied_nonnegative_amounts(seed) -> None:
    person = _person_source()
    person["RETCB_VAL"] = np.random.default_rng(seed).uniform(0, 25000, len(person))
    result = _derive(person, seed)
    values = result[list(US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS)]
    assert (values.to_numpy() >= 0).all()
    np.testing.assert_allclose(values.sum(axis=1), person.RETCB_VAL, rtol=1e-12)


def test_signal_gate_rejects_a_default_leaf() -> None:
    result = _with(
        _frame(),
        seed=0,
        time_period=2024,
    )
    result.table("person")["roth_ira_contributions_desired"] = 0.0

    gate = us_retirement_contributions_signal_gate(result)

    assert not gate.passed
    assert any("roth_ira_contributions_desired" in failure for failure in gate.failures)


@pytest.mark.parametrize("source_total", [0.0, 1.0])
def test_signal_gate_rejects_corrupted_direct_bundled_total(source_total):
    result = _with(_frame())
    person = result.table("person")
    person.loc[0, "RETCB_VAL"] = source_total
    person.loc[0, "traditional_401k_contributions_desired"] = 5.0
    gate = us_retirement_contributions_signal_gate(result)
    assert not gate.passed
    assert gate.details["allocation_mismatch_rows"] == 1
    assert any("do not reconcile" in failure for failure in gate.failures)
