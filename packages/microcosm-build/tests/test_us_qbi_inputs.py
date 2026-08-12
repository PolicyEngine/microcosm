"""Contracts for the retired eCPS Section 199A input family."""

from __future__ import annotations

import copy
import importlib.util

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import puf_support as puf_support_module
from microcosm.build.us_runtime import qbi_inputs as qbi_inputs_module
from microcosm.build.us_runtime.qbi_inputs import (
    QBI_ARCHIVED_ASSUMPTIONS_URL,
    QBI_ARCHIVED_CLONE_URL,
    QBI_ARCHIVED_DERIVATION_URL,
    QBI_ARCHIVED_EXPORT_URL,
    QBI_ARCHIVED_IMPUTATION_URL,
    QBI_ARCHIVED_PUF_ARTIFACT_URL,
    QBI_ARCHIVED_SIMULATION_URL,
    US_QBI_BOOLEAN_OUTPUT_COLUMNS,
    US_QBI_NONNEGATIVE_OUTPUT_COLUMNS,
    US_QBI_OUTPUT_COLUMNS,
    us_qbi_inputs_signal_gate,
    us_qbi_inputs_stage_spec,
    us_qbi_inputs_summary,
    us_qbi_reconciliation_change_receipt,
    us_qbi_reconciliation_universe_receipt,
    validate_us_qbi_reconciliation_live_output,
    validate_us_qbi_reconciliation_receipt,
    with_us_qbi_input_reconciliation,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.frame.schema import EntitySchema

policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)


def _frame(person: pd.DataFrame) -> Frame:
    person = person.copy(deep=True).reset_index(drop=True)
    n = len(person)
    ids = np.arange(1, n + 1, dtype=np.int64)
    person.insert(0, "person_id", ids)
    for entity in ("household", "tax_unit", "spm_unit", "family", "marital_unit"):
        person[f"person_{entity}_id"] = ids
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
    )


def _qbi_person(n: int = 200) -> pd.DataFrame:
    person = pd.DataFrame(
        {
            column: np.zeros(n, dtype=bool)
            if column in US_QBI_BOOLEAN_OUTPUT_COLUMNS
            else np.zeros(n, dtype=np.float64)
            for column in US_QBI_OUTPUT_COLUMNS
        }
    )
    person["person_support_channel"] = np.where(
        np.arange(n) < n // 2, "asec", "puf_tax_detail"
    )
    person["self_employment_income_before_lsr"] = 0.0
    person["partnership_income"] = 0.0
    person["s_corp_income"] = 0.0
    person["estate_income"] = 0.0
    person["rental_income"] = 0.0
    person["non_qualified_dividend_income"] = 0.0

    puf = np.arange(n // 2, n)
    for index, column in enumerate(US_QBI_BOOLEAN_OUTPUT_COLUMNS):
        if column == "business_is_sstb":
            continue
        person.loc[puf[index % 5 :: 5], column] = False
        person.loc[puf, column] |= np.arange(len(puf)) % 5 != index % 5

    sstb = puf[:10]
    person.loc[sstb, "business_is_sstb"] = True
    person.loc[sstb, "self_employment_income_would_be_qualified"] = True
    person.loc[sstb, "self_employment_income_before_lsr"] = 10_000.0
    person.loc[sstb, "w2_wages_from_qualified_business"] = 2_000.0
    person.loc[sstb, "unadjusted_basis_qualified_property"] = 5_000.0

    person.loc[puf[:2], "non_qualified_dividend_income"] = 1_000.0
    person.loc[puf[:2], "qualified_bdc_income"] = 20.0
    person.loc[puf[:20], "non_qualified_dividend_income"] = 1_000.0
    person.loc[puf[:20], "qualified_reit_and_ptp_income"] = 40.0
    person.loc[puf[:25], "w2_wages_from_qualified_business"] = 2_000.0
    person.loc[puf[:30], "unadjusted_basis_qualified_property"] = 5_000.0
    return person


def test_archived_coordinates_pin_algorithms_export_clone_and_artifact() -> None:
    urls = (
        QBI_ARCHIVED_ASSUMPTIONS_URL,
        QBI_ARCHIVED_CLONE_URL,
        QBI_ARCHIVED_DERIVATION_URL,
        QBI_ARCHIVED_EXPORT_URL,
        QBI_ARCHIVED_IMPUTATION_URL,
        QBI_ARCHIVED_PUF_ARTIFACT_URL,
        QBI_ARCHIVED_SIMULATION_URL,
    )
    for url in urls:
        assert "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe" in url
        assert "#L" in url


def test_output_contract_is_exact_and_source_declared() -> None:
    expected = {
        "business_is_sstb",
        "estate_income_would_be_qualified",
        "farm_operations_income_would_be_qualified",
        "farm_rent_income_would_be_qualified",
        "partnership_s_corp_income_would_be_qualified",
        "rental_income_would_be_qualified",
        "self_employment_income_would_be_qualified",
        "sstb_self_employment_income_would_be_qualified",
        "qualified_bdc_income",
        "qualified_reit_and_ptp_income",
        "sstb_self_employment_income_before_lsr",
        "sstb_unadjusted_basis_qualified_property",
        "sstb_w2_wages_from_qualified_business",
        "unadjusted_basis_qualified_property",
        "w2_wages_from_qualified_business",
    }
    assert set(US_QBI_OUTPUT_COLUMNS) == expected
    assert set(US_QBI_BOOLEAN_OUTPUT_COLUMNS) == {
        name for name in expected if name.endswith("_would_be_qualified")
    } | {"business_is_sstb"}
    assert set(US_QBI_NONNEGATIVE_OUTPUT_COLUMNS) == {
        "qualified_bdc_income",
        "qualified_reit_and_ptp_income",
        "sstb_unadjusted_basis_qualified_property",
        "sstb_w2_wages_from_qualified_business",
        "unadjusted_basis_qualified_property",
        "w2_wages_from_qualified_business",
    }

    stage = us_qbi_inputs_stage_spec()
    assert expected <= set(stage.outputs)
    assert set(US_QBI_NONNEGATIVE_OUTPUT_COLUMNS) <= set(stage.nonnegative_outputs)
    assert "release://policyengine/irs-soi-puf/1.8.0/puf_2024.h5" in str(
        stage.artifacts
    )


def test_processed_puf_sstb_alias_is_carried_without_redrawing() -> None:
    source = {"sstb_self_employment_income": np.array([-20.0, 0.0, 300.0])}

    values = puf_support_module._person_source_values(
        source, "sstb_self_employment_income_before_lsr"
    )

    assert values.tolist() == [-20.0, 0.0, 300.0]


def test_boolean_tax_unit_counts_are_preserved_and_source_aligned() -> None:
    person = pd.DataFrame(
        {
            "person_tax_unit_id": [1, 1, 1, 2, 2],
            "self_employment_income_before_lsr": [0.0, 50.0, 100.0, 0.0, 25.0],
            "business_is_sstb": 0.0,
        }
    )

    puf_support_module._write_person_tax_unit_boolean_counts(
        person,
        mask=pd.Series(True, index=person.index),
        column="business_is_sstb",
        totals=pd.Series({1: 2.0, 2: 1.0}),
        fallback_basis_columns=("self_employment_income_before_lsr",),
    )

    assert person.groupby("person_tax_unit_id")["business_is_sstb"].sum().to_dict() == {
        1: 2.0,
        2: 1.0,
    }
    assert person["business_is_sstb"].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]


def test_reconciliation_restores_sstb_routes_total_pools_and_exposure_caps() -> None:
    person = _qbi_person(200)
    original_total = (
        person["self_employment_income_before_lsr"]
        + person["sstb_self_employment_income_before_lsr"]
    ).to_numpy(copy=True)
    person.loc[100, "qualified_bdc_income"] = 2_000.0
    person.loc[100, "qualified_reit_and_ptp_income"] = 4_000.0

    result = with_us_qbi_input_reconciliation(_frame(person))
    reconciled = result.table("person")

    np.testing.assert_allclose(
        reconciled["self_employment_income_before_lsr"]
        + reconciled["sstb_self_employment_income_before_lsr"],
        original_total,
    )
    sstb = reconciled["business_is_sstb"].to_numpy()
    assert np.all(reconciled.loc[sstb, "self_employment_income_before_lsr"] == 0)
    assert np.all(reconciled.loc[~sstb, "sstb_self_employment_income_before_lsr"] == 0)
    np.testing.assert_allclose(
        reconciled["sstb_w2_wages_from_qualified_business"],
        np.where(sstb, reconciled["w2_wages_from_qualified_business"], 0.0),
    )
    np.testing.assert_allclose(
        reconciled["sstb_unadjusted_basis_qualified_property"],
        np.where(sstb, reconciled["unadjusted_basis_qualified_property"], 0.0),
    )
    # The base W-2 and UBIA leaves remain total pools; they are not zeroed on
    # all-SSTB rows.
    assert reconciled.loc[sstb, "w2_wages_from_qualified_business"].sum() > 0
    assert reconciled.loc[sstb, "unadjusted_basis_qualified_property"].sum() > 0
    assert reconciled.loc[100, "qualified_bdc_income"] == 1_000.0
    assert reconciled.loc[100, "qualified_reit_and_ptp_income"] == 1_000.0


def _stacked_qbi_universe_frame(*, child_age: float = 12.0) -> Frame:
    person = _qbi_person(20)
    person["person_support_channel"] = np.where(
        np.arange(len(person)) < 10,
        "asec",
        "acs",
    )
    person["person_support_clone_index"] = 0
    person["person_spine_source_id"] = np.arange(len(person), dtype=np.int64)
    person["age"] = 40.0
    person["SEMP"] = person["self_employment_income_before_lsr"]
    child = 10
    person.loc[child, "age"] = child_age
    person.loc[child, "self_employment_income_before_lsr"] = 0.0
    person.loc[child, "SEMP"] = np.nan
    person.loc[child, "business_is_sstb"] = True
    person.loc[child, "qualified_bdc_income"] = 321.0
    return _frame(person)


def test_reconciliation_preserves_child_universe_zero_and_other_qbi_cells() -> None:
    frame = _stacked_qbi_universe_frame()

    result = with_us_qbi_input_reconciliation(frame)
    after = result.table("person").loc[10]
    receipt = us_qbi_reconciliation_universe_receipt(frame)

    assert after["self_employment_income_before_lsr"] == 0.0
    assert after["business_is_sstb"] == 0
    assert after["sstb_self_employment_income_before_lsr"] == 0.0
    assert after["sstb_w2_wages_from_qualified_business"] == 0.0
    assert receipt["rows_excluded_from_base_self_employment_rewrite"] == 1
    assert receipt["rows_included_in_other_qbi_reconciliation"] == 20
    assert (
        receipt["rules"]["self_employment_income_before_lsr"]["source_column"] == "SEMP"
    )
    assert receipt["structurally_absent_base_source_cells_mutated"] is False
    assert len(receipt["sha256"]) == 64


def test_reconciliation_rejects_native_acs_in_universe_null() -> None:
    frame = _stacked_qbi_universe_frame(child_age=15.0)

    with pytest.raises(
        ValueError,
        match="acs_2024_pums_semp_age_15_plus.*raw_in_universe_null_rows=1",
    ):
        with_us_qbi_input_reconciliation(frame)


def test_change_receipt_binds_read_only_qbi_input_columns() -> None:
    baseline = _stacked_qbi_universe_frame()
    changed = _stacked_qbi_universe_frame()
    changed.table("person").loc[0, "partnership_income"] = -500.0

    baseline_output = with_us_qbi_input_reconciliation(baseline)
    changed_output = with_us_qbi_input_reconciliation(changed)
    baseline_receipt = us_qbi_reconciliation_change_receipt(baseline, baseline_output)
    changed_receipt = us_qbi_reconciliation_change_receipt(changed, changed_output)

    pd.testing.assert_frame_equal(
        baseline_output.table("person").loc[:, US_QBI_OUTPUT_COLUMNS],
        changed_output.table("person").loc[:, US_QBI_OUTPUT_COLUMNS],
    )
    assert (
        baseline_receipt["output_declared_person_values_sha256"]
        == changed_receipt["output_declared_person_values_sha256"]
    )
    assert (
        baseline_receipt["input_person_table_sha256"]
        != changed_receipt["input_person_table_sha256"]
    )
    assert baseline_receipt["sha256"] != changed_receipt["sha256"]


def test_change_receipt_rejects_undeclared_output_mutation() -> None:
    frame = _stacked_qbi_universe_frame()
    reconciled = with_us_qbi_input_reconciliation(frame)
    reconciled.table("person").loc[0, "partnership_income"] = 999.0

    with pytest.raises(ValueError, match="undeclared person column"):
        us_qbi_reconciliation_change_receipt(frame, reconciled)


def test_change_receipt_rejects_structural_universe_zero_mutation() -> None:
    frame = _stacked_qbi_universe_frame()
    person = frame.table("person")
    person["self_employment_income_before_lsr"] = person[
        "self_employment_income_before_lsr"
    ].astype("Float64")
    reconciled = with_us_qbi_input_reconciliation(frame)
    reconciled.table("person").loc[10, "self_employment_income_before_lsr"] = 1.0

    with pytest.raises(ValueError, match="deterministic kernel"):
        us_qbi_reconciliation_change_receipt(frame, reconciled)


@pytest.mark.parametrize("tamper", ["sha256", "negative_count", "extra_field"])
def test_change_receipt_rejects_forged_envelopes(tamper: str) -> None:
    frame = _stacked_qbi_universe_frame()
    reconciled = with_us_qbi_input_reconciliation(frame)
    receipt = us_qbi_reconciliation_change_receipt(frame, reconciled)
    forged = copy.deepcopy(receipt)
    if tamper == "sha256":
        forged["sha256"] = "0" * 64
    elif tamper == "negative_count":
        forged["changed_person_rows"] = -1
    else:
        forged["tampered"] = True

    with pytest.raises(ValueError, match="QBI"):
        validate_us_qbi_reconciliation_receipt(
            forged,
            boundary="forged QBI receipt test",
        )


def test_change_receipt_rejects_mutated_output_even_when_regenerated() -> None:
    frame = _stacked_qbi_universe_frame()
    reconciled = with_us_qbi_input_reconciliation(frame)
    reconciled.table("person").loc[0, "qualified_bdc_income"] = 0.25

    with pytest.raises(ValueError, match="deterministic kernel"):
        us_qbi_reconciliation_change_receipt(frame, reconciled)


def _authorized_qbi_output(frame: Frame) -> tuple[Frame, dict[str, object]]:
    reconciled = with_us_qbi_input_reconciliation(frame)
    receipt = us_qbi_reconciliation_change_receipt(frame, reconciled)
    authorized = qbi_inputs_module.bind_us_qbi_reconciliation_transition_authority(
        reconciled,
        receipt,
    )
    return authorized, receipt


def test_live_output_validation_rejects_post_receipt_mutation() -> None:
    frame = _stacked_qbi_universe_frame()
    reconciled, receipt = _authorized_qbi_output(frame)
    reconciled.table("person").loc[0, "qualified_bdc_income"] = 0.25

    with pytest.raises(ValueError, match="output digest"):
        validate_us_qbi_reconciliation_live_output(
            reconciled,
            receipt,
            boundary="mutated live QBI output test",
            expected_transition_authority_sha256=receipt["sha256"],
        )


def _rehash_forged_stacked_universe_receipt(
    receipt: dict[str, object],
) -> None:
    universe = receipt["recipient_source_universe"]
    assert isinstance(universe, dict)
    source_receipt = {
        key: value
        for key, value in universe.items()
        if key
        not in {
            "source_universe_sha256",
            "source_universe_resolution_mutated_raw_pums_cells",
            "operation",
            "rows_excluded_from_base_self_employment_rewrite",
            "rows_included_in_other_qbi_reconciliation",
            "structurally_absent_base_source_cells_mutated",
            "sha256",
        }
    }
    source_receipt["raw_pums_source_cells_mutated"] = universe[
        "source_universe_resolution_mutated_raw_pums_cells"
    ]
    universe["source_universe_sha256"] = qbi_inputs_module._qbi_receipt_sha256(
        source_receipt
    )
    unsigned_universe = dict(universe)
    unsigned_universe.pop("sha256")
    universe["sha256"] = qbi_inputs_module._qbi_receipt_sha256(unsigned_universe)
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("sha256")
    receipt["sha256"] = qbi_inputs_module._qbi_receipt_sha256(unsigned_receipt)


def test_live_output_validation_rejects_rehashed_universe_semantic_forgery() -> None:
    frame = _stacked_qbi_universe_frame()
    reconciled, receipt = _authorized_qbi_output(frame)
    forged = copy.deepcopy(receipt)
    universe = forged["recipient_source_universe"]
    assert isinstance(universe, dict)
    universe["scoped_person_rows"] += 1
    _rehash_forged_stacked_universe_receipt(forged)

    with pytest.raises(ValueError, match="does not exactly match the live frame"):
        validate_us_qbi_reconciliation_live_output(
            reconciled,
            forged,
            boundary="rehashed QBI universe forgery test",
            expected_transition_authority_sha256=receipt["sha256"],
        )


def test_live_output_validation_rejects_rebound_driver_and_fixed_point_output() -> None:
    frame = _stacked_qbi_universe_frame()
    reconciled, receipt = _authorized_qbi_output(frame)
    person = reconciled.table("person")
    person.loc[0, "non_qualified_dividend_income"] = 100.0
    person.loc[0, "qualified_bdc_income"] = 50.0
    forged = copy.deepcopy(receipt)
    forged["output_declared_person_values_sha256"] = (
        qbi_inputs_module._qbi_person_values_sha256(
            person,
            columns=qbi_inputs_module.US_QBI_RECONCILED_PERSON_COLUMNS,
        )
    )
    unsigned = dict(forged)
    unsigned.pop("sha256")
    forged["sha256"] = qbi_inputs_module._qbi_receipt_sha256(unsigned)

    with pytest.raises(ValueError, match="driver-surface digest"):
        validate_us_qbi_reconciliation_live_output(
            reconciled,
            forged,
            boundary="rebound QBI driver/output test",
            expected_transition_authority_sha256=receipt["sha256"],
        )


def test_live_output_validation_rejects_fresh_receipt_for_alternate_fixed_point() -> (
    None
):
    frame = _stacked_qbi_universe_frame()
    reconciled, receipt = _authorized_qbi_output(frame)
    person = reconciled.table("person")
    person.loc[0, "non_qualified_dividend_income"] = 100.0
    person.loc[0, "qualified_bdc_income"] = 50.0
    fresh = us_qbi_reconciliation_change_receipt(reconciled, reconciled)

    with pytest.raises(ValueError, match="independently carried transition authority"):
        validate_us_qbi_reconciliation_live_output(
            reconciled,
            fresh,
            boundary="alternate QBI fixed-point receipt test",
            expected_transition_authority_sha256=receipt["sha256"],
        )


def test_qbi_universe_receipt_rejects_boolean_count() -> None:
    frame = _stacked_qbi_universe_frame()
    reconciled = with_us_qbi_input_reconciliation(frame)
    receipt = us_qbi_reconciliation_change_receipt(frame, reconciled)
    forged = copy.deepcopy(receipt)
    universe = forged["recipient_source_universe"]
    assert isinstance(universe, dict)
    universe["scoped_person_rows"] = True

    with pytest.raises(ValueError, match="scoped_person_rows.*nonnegative integer"):
        validate_us_qbi_reconciliation_receipt(
            forged,
            boundary="Boolean QBI count test",
        )


def test_live_output_validation_rejects_undeclared_added_person_column() -> None:
    frame = _stacked_qbi_universe_frame()
    reconciled, receipt = _authorized_qbi_output(frame)
    reconciled.table("person")["post_receipt_undeclared_tamper"] = 1

    with pytest.raises(ValueError, match="person-column inventory changed"):
        validate_us_qbi_reconciliation_live_output(
            reconciled,
            receipt,
            boundary="added QBI person-column test",
            expected_transition_authority_sha256=receipt["sha256"],
        )


def test_live_output_validation_rejects_receipt_laundered_added_column() -> None:
    frame = _stacked_qbi_universe_frame()
    reconciled, receipt = _authorized_qbi_output(frame)
    added = "post_receipt_undeclared_tamper"
    person = reconciled.table("person")
    person[added] = 1
    forged = copy.deepcopy(receipt)
    forged["input_person_columns"].append(added)
    forged["input_person_table_sha256"] = qbi_inputs_module._qbi_person_values_sha256(
        person,
        columns=tuple(forged["input_person_columns"]),
    )
    preservation = forged["undeclared_surface_preservation"]
    assert isinstance(preservation, dict)
    preservation["undeclared_person_columns_verified"] += 1
    unsigned = dict(forged)
    unsigned.pop("sha256")
    forged["sha256"] = qbi_inputs_module._qbi_receipt_sha256(unsigned)

    with pytest.raises(ValueError, match="independently carried transition authority"):
        validate_us_qbi_reconciliation_live_output(
            reconciled,
            forged,
            boundary="laundered QBI person-column test",
            expected_transition_authority_sha256=receipt["sha256"],
        )


def test_live_output_validation_rejects_rehashed_preservation_count() -> None:
    frame = _stacked_qbi_universe_frame()
    reconciled, receipt = _authorized_qbi_output(frame)
    forged = copy.deepcopy(receipt)
    preservation = forged["undeclared_surface_preservation"]
    assert isinstance(preservation, dict)
    preservation["undeclared_person_columns_verified"] += 1
    unsigned = dict(forged)
    unsigned.pop("sha256")
    forged["sha256"] = qbi_inputs_module._qbi_receipt_sha256(unsigned)

    with pytest.raises(ValueError, match="preservation receipt does not match"):
        validate_us_qbi_reconciliation_live_output(
            reconciled,
            forged,
            boundary="rehashed QBI preservation-count test",
            expected_transition_authority_sha256=receipt["sha256"],
        )


def test_live_output_validation_allows_only_canonical_seed_person_output() -> None:
    frame = _stacked_qbi_universe_frame()
    reconciled, receipt = _authorized_qbi_output(frame)
    seed_column = "takes_up_medicaid_if_eligible"
    reconciled.table("person")[seed_column] = False

    assert (
        qbi_inputs_module.validate_us_qbi_reconciliation_live_output(
            reconciled,
            receipt,
            boundary="canonical post-QBI seed output test",
            expected_transition_authority_sha256=receipt["sha256"],
            allowed_post_reconciliation_person_columns=(seed_column,),
        )
        == receipt
    )
    with pytest.raises(ValueError, match="allowed post-QBI person columns"):
        qbi_inputs_module.validate_us_qbi_reconciliation_live_output(
            reconciled,
            receipt,
            boundary="non-contract post-QBI seed output test",
            expected_transition_authority_sha256=receipt["sha256"],
            allowed_post_reconciliation_person_columns=(
                "post_receipt_undeclared_tamper",
            ),
        )


def test_qbi_seed_receipt_rejects_non_contract_person_output() -> None:
    with pytest.raises(ValueError, match="non-contract program"):
        qbi_inputs_module.us_qbi_post_reconciliation_person_columns(
            {
                "programs": {
                    "post_receipt_undeclared_tamper": {"entity": "person"},
                }
            }
        )


def test_sstb_requires_a_positive_qualified_mapped_source() -> None:
    person = _qbi_person(200)
    row = 100
    person.loc[row, "business_is_sstb"] = True
    person.loc[row, "self_employment_income_before_lsr"] = 500.0
    person.loc[row, "self_employment_income_would_be_qualified"] = False
    person.loc[row, "sstb_self_employment_income_would_be_qualified"] = False
    person.loc[row, "partnership_income"] = 0.0
    person.loc[row, "s_corp_income"] = 0.0
    person.loc[row, "estate_income"] = 0.0

    result = with_us_qbi_input_reconciliation(_frame(person)).table("person")

    assert not result.loc[row, "business_is_sstb"]
    assert result.loc[row, "self_employment_income_before_lsr"] == 500.0
    assert result.loc[row, "sstb_self_employment_income_before_lsr"] == 0.0


def test_asec_channel_policy_is_explicit_and_reconciliation_is_idempotent() -> None:
    first = with_us_qbi_input_reconciliation(_frame(_qbi_person()))
    second = with_us_qbi_input_reconciliation(first)
    asec = second.table("person").query("person_support_channel == 'asec'")

    assert not asec["business_is_sstb"].any()
    assert not asec["sstb_self_employment_income_would_be_qualified"].any()
    for column in US_QBI_BOOLEAN_OUTPUT_COLUMNS:
        if column not in {
            "business_is_sstb",
            "sstb_self_employment_income_would_be_qualified",
        }:
            assert asec[column].all()
    pd.testing.assert_frame_equal(first.table("person"), second.table("person"))


def test_signal_gate_passes_reconciled_nondefault_family() -> None:
    reconciled = with_us_qbi_input_reconciliation(_frame(_qbi_person()))

    result = us_qbi_inputs_signal_gate(reconciled)

    assert result.passed, result.failures
    assert all(value == 0 for value in result.details["invariants"].values())


def test_signal_gate_reports_missing_negative_and_identity_failures() -> None:
    person = _qbi_person()
    person.loc[100, "qualified_bdc_income"] = -1.0
    person.loc[100, "business_is_sstb"] = True
    person.loc[100, "self_employment_income_before_lsr"] = 100.0
    person.loc[100, "sstb_self_employment_income_before_lsr"] = 0.0
    frame = _frame(person)

    result = us_qbi_inputs_signal_gate(frame)

    assert not result.passed
    assert any("qualified_bdc_income" in failure for failure in result.failures)
    assert any(
        "sstb_rows_with_non_sstb_income" in failure for failure in result.failures
    )

    missing = person.drop(columns=["qualified_reit_and_ptp_income"])
    missing_result = us_qbi_inputs_signal_gate(_frame(missing))
    assert not missing_result.passed
    assert "qualified_reit_and_ptp_income" in missing_result.failures[0]


def test_reconciliation_rejects_wrong_schema_and_missing_inputs() -> None:
    wrong = Frame(
        {
            "person": pd.DataFrame({"person_id": [1], "person_household_id": [1]}),
            "household": pd.DataFrame({"household_id": [1]}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(1), WeightKind.DESIGN)},
    )
    with pytest.raises(ValueError, match="US schema"):
        with_us_qbi_input_reconciliation(wrong)

    with pytest.raises(ValueError, match="requires PUF-imputed"):
        with_us_qbi_input_reconciliation(
            _frame(_qbi_person().drop(columns=["qualified_bdc_income"]))
        )


@pytest.mark.parametrize(
    "consumer",
    (with_us_qbi_input_reconciliation, us_qbi_inputs_summary),
    ids=("reconciliation", "summary"),
)
def test_present_s_corp_column_retains_exact_whole_pool_nonfinite_check(
    consumer,
) -> None:
    person = _qbi_person(7)
    person["s_corp_income"] = np.nan

    with pytest.raises(
        ValueError,
        match=r"US QBI input 's_corp_income' contains 7 nonfinite value\(s\)\.",
    ):
        consumer(_frame(person))


def test_summary_does_not_mutate_source_frame() -> None:
    reconciled = with_us_qbi_input_reconciliation(_frame(_qbi_person()))
    before = reconciled.table("person").copy(deep=True)

    summary = us_qbi_inputs_summary(reconciled)

    assert set(summary) == {
        "columns",
        "invariants",
        "reconciliation_universe",
    }
    pd.testing.assert_frame_equal(before, reconciled.table("person"))


@requires_us
def test_all_qbi_outputs_are_live_policyengine_us_input_leaves() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    variables = CountryTaxBenefitSystem().variables
    for column in US_QBI_OUTPUT_COLUMNS:
        variable = variables[column]
        assert variable.entity.key == "person"
        assert not variable.formulas
        assert getattr(variable, "adds", None) is None
        assert getattr(variable, "subtracts", None) is None
