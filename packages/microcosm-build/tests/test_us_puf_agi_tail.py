"""Deterministic AGI own-tail selection and representability contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import puf_agi_tail as agi_tail
from microcosm.build.us_runtime.puf_support import (
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    attach_puf_tail_person_projection,
)
from microcosm.build.us_runtime.qbi_inputs import US_QBI_BOOLEAN_OUTPUT_COLUMNS


def _donors(agi: list[float] | np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    count = len(agi)
    donor = pd.DataFrame(
        {
            "tax_unit_id": np.arange(1, count + 1),
            "weight": np.arange(1, count + 1, dtype=float),
            "filing_status_code": np.ones(count),
            **{
                column: np.zeros(count)
                for column in (
                    *PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
                    *PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
                )
            },
        }
    )
    donor["employment_income_before_lsr"] = agi
    persons = donor[["tax_unit_id", *PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS]].copy()
    persons.insert(1, "person_id", np.arange(100_001, 100_001 + count))
    persons.insert(2, "role", "head")
    for column in US_QBI_BOOLEAN_OUTPUT_COLUMNS:
        persons[column] = False
    attach_puf_tail_person_projection(donor, persons)
    return donor, persons


def _select(donor: pd.DataFrame, cg: np.ndarray | None = None):
    return agi_tail.select_puf_agi_tail_donors(
        donor,
        np.zeros(len(donor), dtype=bool) if cg is None else cg,
        np.ones(len(donor), dtype=bool),
    )


def test_agi_floor_is_inclusive_and_both_arm_keeps_full_vector() -> None:
    donor, _persons = _donors([4_999_999.0, 5_000_000.0, 6_000_000.0, 1.0])
    selected, receipt = _select(donor, np.asarray([False, False, True, True]))
    assert selected.tax_unit_id.tolist() == [2, 3, 4]
    assert selected._puf_tail_arm.tolist() == [2, 3, 1]
    assert selected.weight.tolist() == [2.0, 3.0, 4.0]
    assert (
        selected.iloc[0]._puf_tail_person_vectors["head"][
            "employment_income_before_lsr"
        ]
        == 5_000_000.0
    )
    assert selected.iloc[0]._puf_tail_person_dtypes["business_is_sstb"] == "bool"
    assert receipt["agi_candidate_count"] == 2
    assert receipt["skipped"]["count"] == 0


def test_ineligible_dependent_money_skips_and_records_both_arm_donor() -> None:
    donor, persons = _donors([6_000_000.0, 7_000_000.0])
    dependent = persons.iloc[[0]].copy()
    dependent["person_id"] = 999_999
    dependent["role"] = "dependent"
    dependent["employment_income_before_lsr"] = 1.0
    donor.loc[0, "employment_income_before_lsr"] += 1.0
    persons = pd.concat([persons, dependent], ignore_index=True)
    attach_puf_tail_person_projection(donor, persons)
    selected, receipt = _select(donor, np.asarray([True, False]))
    assert selected.tax_unit_id.tolist() == [2]
    assert receipt["skipped"]["count"] == 1
    assert receipt["skipped"]["weight"] == 1.0
    assert receipt["skipped"]["proxy_agi_sum"] == 6_000_001.0
    assert receipt["skipped"]["weighted_proxy_agi"] == 6_000_001.0
    assert receipt["skipped"]["by_reason"]["dependent_monetary_value"]["count"] == 1


def test_dependent_boolean_flags_are_ignored_and_spouse_role_is_retained() -> None:
    donor, persons = _donors([6_000_000.0])
    spouse = persons.copy()
    spouse["person_id"] = 999_998
    spouse["role"] = "spouse"
    spouse["employment_income_before_lsr"] = 0.0
    spouse["partnership_income"] = 12.0
    spouse["partnership_s_corp_income_would_be_qualified"] = True
    dependent = spouse.copy()
    dependent["person_id"] = 999_999
    dependent["role"] = "dependent"
    dependent["partnership_income"] = 0.0
    donor["partnership_income"] = 12.0
    donor["partnership_s_corp_income_would_be_qualified"] = 2.0
    persons = pd.concat([persons, spouse, dependent], ignore_index=True)
    attach_puf_tail_person_projection(donor, persons)
    selected, receipt = _select(donor)
    row = selected.iloc[0]
    assert row._puf_tail_needs_spouse
    assert set(row._puf_tail_person_vectors) == {"head", "spouse"}
    assert (
        row._puf_tail_person_vectors["spouse"][
            "partnership_s_corp_income_would_be_qualified"
        ]
        is True
    )
    assert receipt["skipped"]["count"] == 0


def test_original_dependent_qbi_counterexample_is_skipped_without_folding() -> None:
    donor, persons = _donors([6_000_000.0])
    spouse = persons.copy()
    spouse["person_id"] = 999_998
    spouse["role"] = "spouse"
    spouse["employment_income_before_lsr"] = 0.0
    dependent = spouse.copy()
    dependent["person_id"] = 999_999
    dependent["role"] = "dependent"
    persons["partnership_income"] = -1_000.0
    dependent["partnership_income"] = 2_000.0
    dependent["qualified_reit_and_ptp_income"] = 2_000.0
    persons = pd.concat([persons, spouse, dependent], ignore_index=True)
    persons["partnership_s_corp_income_would_be_qualified"] = True
    donor["partnership_income"] = 1_000.0
    donor["qualified_reit_and_ptp_income"] = 2_000.0
    donor["partnership_s_corp_income_would_be_qualified"] = 3.0
    attach_puf_tail_person_projection(donor, persons)
    selected, receipt = _select(donor)
    assert selected.empty
    assert receipt["skipped"]["by_reason"]["dependent_monetary_value"]["count"] == 1
    assert receipt["skipped"]["weighted_proxy_agi"] == 6_001_000.0


@pytest.mark.parametrize("role", ["unknown", "spouse", "dependent"])
def test_missing_or_unrepresentable_head_role_is_skipped(role: str) -> None:
    donor, persons = _donors([6_000_000.0])
    persons["role"] = role
    attach_puf_tail_person_projection(donor, persons)
    selected, receipt = _select(donor)
    assert selected.empty
    assert receipt["skipped"]["by_reason"]["unrepresentable_person_roles"]["count"] == 1


def test_numeric_boolean_projection_is_skipped_instead_of_coerced() -> None:
    donor, persons = _donors([6_000_000.0])
    persons["business_is_sstb"] = 0.0
    attach_puf_tail_person_projection(donor, persons)
    selected, receipt = _select(donor)
    assert selected.empty
    assert receipt["skipped"]["by_reason"]["unsupported_person_dtype"]["count"] == 1


def test_thinning_is_order_independent_preserves_cell_returns_and_cap() -> None:
    donor, _persons = _donors(5_000_000.0 + np.arange(6_005) * 1_000.0)
    donor["filing_status_code"] = 1.0 + np.arange(len(donor)) % 4
    cg = np.zeros(len(donor), dtype=bool)
    cg[:5] = True
    selected, receipt = _select(donor, cg)
    shuffled = donor.sample(frac=1, random_state=42)
    shuffled_cg = shuffled.tax_unit_id.le(5).to_numpy()
    other, other_receipt = _select(shuffled, shuffled_cg)
    pd.testing.assert_frame_equal(selected, other)
    assert receipt == other_receipt
    assert len(selected) == agi_tail.PUF_AGI_TAIL_MAX_COUNT + 5
    assert selected.loc[selected._puf_tail_arm.eq(3), "tax_unit_id"].tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert selected.loc[selected._puf_tail_arm.eq(3), "weight"].tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert sum(cell["kept_count"] for cell in receipt["thinning"]["cells"]) == 3_000
    for cell in receipt["thinning"]["cells"]:
        assert cell["weight_before"] == cell["weight_after"]
        assert cell["count_before"] == cell["kept_count"] + cell["dropped_count"]
        assert np.isfinite(cell["weighted_proxy_agi_before"])
        assert np.isfinite(cell["weighted_proxy_agi_after"])
        live = selected.loc[selected.tax_unit_id.isin(cell["kept_donor_source_ids"])]
        assert float(live.weight.sum()) == cell["weight_before"]


def test_no_thinning_preserves_original_weights_exactly() -> None:
    donor, _persons = _donors([5_000_000.0, 5_500_000.0, 9_000_000.0])
    donor["weight"] = [0.125, 3.7, 13.1]
    selected, receipt = _select(donor)
    assert np.array_equal(selected.weight.to_numpy(), donor.weight.to_numpy())
    assert receipt["thinning"]["dropped_count"] == 0


def test_complete_high_agi_source_requires_bound_person_projection() -> None:
    donor, _persons = _donors([5_000_000.0])
    donor.attrs.clear()
    with pytest.raises(ValueError, match="person projection"):
        _select(donor)


@pytest.mark.parametrize(
    "missing_output", ["charitable_cash_donations", "domestic_production_ald"]
)
def test_known_high_agi_requires_complete_vector_even_without_projection(
    missing_output: str,
) -> None:
    donor, _persons = _donors([5_000_000.0])
    donor.attrs.clear()
    donor.drop(columns=missing_output, inplace=True)
    with pytest.raises(
        ValueError, match=f"complete donor output vector.*{missing_output}"
    ):
        _select(donor, np.asarray([True]))


def test_partial_vector_with_known_below_floor_agi_is_not_legacy_unknown() -> None:
    donor, _persons = _donors([4_999_999.0])
    donor.attrs.clear()
    donor.drop(columns="charitable_cash_donations", inplace=True)
    selected, receipt = _select(donor, np.asarray([True]))
    assert selected._puf_tail_arm.tolist() == [1]
    assert selected._puf_tail_proxy_agi.tolist() == [4_999_999.0]
    assert receipt["agi_candidate_count"] == 0
    assert receipt["projection_status"] == "not_required_no_agi_candidates"


def test_legacy_capital_gains_only_fixture_retains_original_columns_and_values() -> (
    None
):
    donor = pd.DataFrame(
        {
            "tax_unit_id": [1, 2],
            "weight": [100.0, 5.0],
            "filing_status_code": [1.0, 2.0],
        }
    )
    selected, receipt = _select(donor, np.asarray([False, True]))
    pd.testing.assert_frame_equal(
        selected.loc[:, donor.columns], donor.iloc[[1]].reset_index(drop=True)
    )
    assert selected._puf_tail_arm.tolist() == [1]
    assert receipt["projection_status"] == "unavailable_legacy_capital_gains_only"


def test_selection_receipt_round_trip_and_resealed_arithmetic_tampering() -> None:
    import copy
    import json

    donor, _persons = _donors([5_000_000.0, 6_000_000.0])
    _selected, receipt = _select(donor)
    restored = json.loads(json.dumps(receipt))
    agi_tail.validate_puf_agi_tail_selection_receipt(restored)
    for path, value in (
        (("agi_selected_count",), 100),
        (("identity", "proxy_agi_floor"), 4_000_000),
        (("thinning", "cells", 0, "weight_after"), 123.0),
        (("thinning", "cells", 0, "kept_count"), True),
        (("skipped", "count"), 1),
    ):
        changed = copy.deepcopy(restored)
        node = changed
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        with pytest.raises(ValueError, match="AGI tail"):
            agi_tail.validate_puf_agi_tail_selection_receipt(changed)


def _mass_manifest() -> dict:
    from microcosm.build.us_runtime.puf_capital_gains_tail import puf_tail_owned_columns

    owned = puf_tail_owned_columns(2)
    cg_owned = puf_tail_owned_columns(1)
    cg_columns = (*cg_owned["person"], *cg_owned["tax_unit"])
    head = dict.fromkeys(owned["person"], 0.0)
    spouse = head.copy()
    head["employment_income_before_lsr"] = 6_000_000.0
    head["short_term_capital_gains"] = 10.0
    spouse["short_term_capital_gains"] = 5.0
    unit = dict.fromkeys(owned["tax_unit"], 0.0)
    records = [
        {
            "arm": 1,
            "donor_source_id": 1,
            "donor_weight": 5.0,
            "assigned_weight": 15.0,
            "joint_vector": dict.fromkeys(cg_columns, 3.0),
            "tail_person_count": 2,
        },
        {
            "arm": 2,
            "donor_source_id": 2,
            "donor_weight": 2.0,
            "assigned_weight": 6.0,
            "person_vectors": {"head": head, "spouse": spouse},
            "tax_unit_vector": unit,
            "joint_vector": {
                column: head[column] + spouse[column]
                if column in head
                else unit[column]
                for column in cg_columns
            },
            "tail_person_count": 3,
        },
    ]
    full_masses = {}
    for column in (*owned["person"], *owned["tax_unit"]):
        if column in cg_columns:
            continue
        value = head[column] + spouse[column] if column in head else unit[column]
        full_masses[column] = {
            "scope": "agi_arm",
            "donor_weighted_signed_mass": value * 2.0,
            "expected_frame_weighted_signed_mass": value * 6.0,
            "transferred_frame_weighted_signed_mass": value * 6.0,
            "difference": 0.0,
        }
    signed = {}
    for column in cg_columns:
        donor_mass = sum(
            row["joint_vector"][column] * row["donor_weight"] for row in records
        )
        transferred = sum(
            row["joint_vector"][column] * row["assigned_weight"] for row in records
        )
        signed[column] = {
            "donor_weighted_signed_mass": donor_mass,
            "design_weight_normalization": 3.0,
            "expected_frame_weighted_signed_mass": donor_mass * 3.0,
            "transferred_frame_weighted_signed_mass": transferred,
            "difference": transferred - donor_mass * 3.0,
        }
    return {
        "records": records,
        "weight_domain": {"design_weight_normalization": 3.0},
        "full_vector_reconciliation": {
            "passed": True,
            "owned_cell_count": len(owned["person"]) * 3 + len(owned["tax_unit"]),
            "signed_mass": full_masses,
        },
        "signed_leg_reconciliation": {**signed, **full_masses},
    }


def test_full_vector_mass_receipts_rederive_every_column_from_records() -> None:
    import copy

    manifest = _mass_manifest()
    agi_tail.validate_puf_tail_vector_mass_receipts(manifest)
    column = "employment_income_before_lsr"
    for path, value in (
        (("full_vector_reconciliation",), None),
        (("full_vector_reconciliation", "passed"), 1),
        (("full_vector_reconciliation", "owned_cell_count"), 1),
        (("full_vector_reconciliation", "signed_mass", column), None),
        (("signed_leg_reconciliation", "short_term_capital_gains", "difference"), 1.0),
        (
            (
                "signed_leg_reconciliation",
                column,
                "expected_frame_weighted_signed_mass",
            ),
            123.0,
        ),
        (("records", 1, "tail_person_count"), 4),
        (("records", 1, "person_vectors", "head", column), 7_000_000.0),
    ):
        changed = copy.deepcopy(manifest)
        node = changed
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        with pytest.raises(ValueError, match="PUF tail"):
            agi_tail.validate_puf_tail_vector_mass_receipts(changed)


def test_mass_receipt_allows_zero_source_spouse_on_head_only_recipient() -> None:
    from microcosm.build.us_runtime.puf_capital_gains_tail import puf_tail_owned_columns

    manifest = _mass_manifest()
    record = manifest["records"][1]
    record["person_vectors"]["head"]["short_term_capital_gains"] = 15.0
    record["person_vectors"]["spouse"]["short_term_capital_gains"] = 0.0
    record["tail_person_count"] = 1
    owned = puf_tail_owned_columns(2)
    manifest["full_vector_reconciliation"]["owned_cell_count"] = len(
        owned["person"]
    ) + len(owned["tax_unit"])
    agi_tail.validate_puf_tail_vector_mass_receipts(manifest)

    record["person_vectors"]["head"]["short_term_capital_gains"] = 10.0
    record["person_vectors"]["spouse"]["short_term_capital_gains"] = 5.0
    with pytest.raises(ValueError, match="person count/roles"):
        agi_tail.validate_puf_tail_vector_mass_receipts(manifest)
