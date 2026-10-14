"""Tests split from packages/microcosm-build/tests/test_us_stacked_spine.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_stacked_spine import *


def test_puf_finalize_masks_earnings_allocation_to_age_15_plus() -> None:
    __import__("policyengine_us")
    frame = _cloned_acs_earnings_universe_fixture()
    person = frame.table("person")
    for column in US_QBI_OUTPUT_COLUMNS:
        person[column] = False if column in US_QBI_BOOLEAN_OUTPUT_COLUMNS else 0.0
    person["long_term_capital_gains_before_response"] = 0.0
    person["non_sch_d_capital_gains"] = 0.0
    tax_unit = frame.table("tax_unit")
    detail_tax_units = tax_unit[support_clone_index_column("tax_unit")].eq(1)
    predictions = pd.DataFrame(
        {
            "employment_income_before_lsr": 1_000.0,
            "self_employment_income_before_lsr": 100.0,
        },
        index=tax_unit.index[detail_tax_units],
    )
    donor = pd.DataFrame(
        {
            "employment_income_before_lsr": [1_000.0, 2_000.0],
            "self_employment_income_before_lsr": [100.0, 200.0],
            "weight": [1.0, 1.0],
        }
    )

    finalized = finalize_us_puf_tax_detail_predictions(
        frame,
        donor,
        predictions,
        person_outputs=tuple(predictions.columns),
        tax_unit_outputs=(),
        absent_cells=PUF_ABSENT_CELLS_PRESERVE_NULLS,
    )
    finalized_person = finalized.table("person")
    acs = finalized_person[support_channel_column("person")].eq("acs")
    child = finalized_person["age"].lt(15)
    native_child = (
        acs & child & finalized_person[support_clone_index_column("person")].eq(0)
    )
    detail_child = (
        acs & child & finalized_person[support_clone_index_column("person")].eq(1)
    )
    assert int(native_child.sum()) == int(detail_child.sum()) == 4
    for column in predictions:
        assert finalized_person.loc[native_child, column].eq(0.0).all()
        assert finalized_person.loc[detail_child, column].eq(0.0).all()

    detail = finalized_person[support_clone_index_column("person")].eq(1)
    detail_units = finalized_person.loc[
        detail, ["person_tax_unit_id", "age", *predictions.columns]
    ]
    mixed = detail_units.groupby("person_tax_unit_id", sort=False).filter(
        lambda group: group["age"].lt(15).any() and group["age"].ge(15).any()
    )
    assert not mixed.empty
    mixed_child_counts = mixed.groupby("person_tax_unit_id", sort=False)["age"].apply(
        lambda age: int(age.lt(15).sum())
    )
    assert mixed_child_counts.eq(2).all()
    assert mixed.loc[mixed["age"].lt(15), list(predictions)].eq(0.0).all().all()
    mixed_adult_totals = (
        mixed.loc[mixed["age"].ge(15)]
        .groupby("person_tax_unit_id", sort=False)[list(predictions)]
        .sum()
    )
    assert mixed_adult_totals["employment_income_before_lsr"].eq(1_000.0).all()
    assert mixed_adult_totals["self_employment_income_before_lsr"].eq(100.0).all()
    all_child = detail_units.groupby("person_tax_unit_id", sort=False).filter(
        lambda group: group["age"].lt(15).all()
    )
    assert len(all_child) == 2
    assert all_child[list(predictions)].eq(0.0).all().all()

    derived = derive_multispine_pool_inputs(finalized)
    derived_person = derived.frame.table("person")
    qbi_receipt = derived.receipt["qbi_input_reconciliation"]

    assert (
        derived_person.loc[native_child, "self_employment_income_before_lsr"]
        .eq(0.0)
        .all()
    )
    assert (
        derived_person.loc[detail_child, "self_employment_income_before_lsr"]
        .eq(0.0)
        .all()
    )
    assert (
        qbi_receipt["recipient_source_universe"][
            "rows_excluded_from_base_self_employment_rewrite"
        ]
        == 8
    )
    assert qbi_receipt["structurally_absent_base_source_changed_rows"] == 0
    assert len(qbi_receipt["input_person_table_sha256"]) == 64
    assert len(qbi_receipt["output_declared_person_values_sha256"]) == 64
