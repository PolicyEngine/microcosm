"""UK firm tables in policyengine-uk names: candidates, export and contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_table import (
    ASSUMED_INPUT_VAT_RATE,
    UK_FIRM_ATTRIBUTE_POLICYENGINE_UK_COLUMNS,
    UK_FIRM_CANDIDATE_RULES_COLUMNS,
    UK_FIRM_ENGLISH_REGION_CODES,
    UK_FIRM_EXTENDED_POLICYENGINE_UK_COLUMNS,
    UK_FIRM_EXTENDED_TABLE_COLUMNS,
    UK_FIRM_INPUT_STATUSES,
    UK_FIRM_LEGAL_FORMS,
    UK_FIRM_MICROCOSM_COLUMNS,
    UK_FIRM_MICROCOSM_ONLY_COLUMNS,
    UK_FIRM_ONS_ATTRIBUTE_COLUMNS,
    UK_FIRM_ONS_LEGAL_STATUS_TO_LEGAL_FORM,
    UK_FIRM_POLICYENGINE_UK_COLUMNS,
    UK_FIRM_POLICYENGINE_UK_INPUT_STATUS,
    UK_FIRM_PUBLIC_BODY_ONS_LEGAL_STATUSES,
    UK_FIRM_REGION_CODE_TO_REGION,
    UK_FIRM_SIC_SECTIONS,
    UK_FIRM_TABLE_COLUMNS,
    build_candidate_rules_table,
    to_policyengine_uk_firm_table,
    uk_firm_sic_sections,
    validate_uk_firm_table,
)
from microcosm.build.uk_runtime.occupation_constants import (
    UK_SIC2007_SECTION_DIVISIONS,
)

#: policyengine-uk ``SICSection`` member names (branch firm-entity-engine,
#: policyengine_uk/variables/firm/firm_sic_section.py), in declaration order.
_POLICYENGINE_UK_SIC_SECTION_MEMBERS = ("UNKNOWN", *"ABCDEFGHIJKLMNOPQRSTU")

#: policyengine-uk ``FirmLegalForm`` members (policyengine_uk/variables/firm/
#: firm_legal_form.py) and ``Region`` members (policyengine_uk/variables/
#: household/demographic/geography.py), in declaration order.
_POLICYENGINE_UK_FIRM_LEGAL_FORM_MEMBERS = (
    "COMPANY",
    "SOLE_TRADER",
    "PARTNERSHIP",
    "OTHER",
)
_POLICYENGINE_UK_REGION_MEMBERS = (
    "UNKNOWN",
    "NORTH_EAST",
    "NORTH_WEST",
    "YORKSHIRE",
    "EAST_MIDLANDS",
    "WEST_MIDLANDS",
    "EAST_OF_ENGLAND",
    "LONDON",
    "SOUTH_EAST",
    "SOUTH_WEST",
    "WALES",
    "SCOTLAND",
    "NORTHERN_IRELAND",
)


def _firms() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "firm_id": np.array([4, 2, 1, 3], dtype=np.int64),
            "source_firm_id": np.array([4, 2, 1, 3], dtype=np.int64),
            "source_firm_key": [
                f"2024-25:experimental_uk_firm:{i}" for i in (4, 2, 1, 3)
            ],
            "data_vintage": "2024-25",
            "sic_code": ["00011", "00047", "00004", "00086"],
            "annual_turnover_k": np.array([120.0, 90.0, 0.0, 40.0], dtype=np.float32),
            "annual_input_k": np.array([80.0, 30.0, 0.0, 10.0], dtype=np.float32),
            "employment": np.array([12, 3, 0, 1]),
            "firm_weight": [2.5, 10.0, 0.0, 4.0],
            "vat_scope": [True, True, False, False],
            "vat_registered": [True, True, False, True],
            "in_frame": [True, True, False, True],
            "base_weight": [1.0, 2.0, 1.0, 1.0],
            "generation_status": "experimental",
        }
    )


def test_sic_sections_match_policyengine_uk_members() -> None:
    assert UK_FIRM_SIC_SECTIONS == _POLICYENGINE_UK_SIC_SECTION_MEMBERS
    assert UK_FIRM_SIC_SECTIONS[1:] == tuple(UK_SIC2007_SECTION_DIVISIONS)


def test_sic_sections_read_divisions_as_ints_or_padded_strings() -> None:
    assert list(uk_firm_sic_sections(["00001", "00011", "00047", "00099"])) == [
        "A",
        "C",
        "G",
        "U",
    ]
    assert list(uk_firm_sic_sections(np.array([4, 35, 68, 86]))) == [
        "UNKNOWN",
        "D",
        "L",
        "Q",
    ]


@pytest.mark.parametrize("bad", [["47110"], [100], [-1], ["x"], [11.5]])
def test_sic_sections_refuse_values_that_are_not_divisions(bad) -> None:
    with pytest.raises(ValueError, match="SIC 2007 divisions"):
        uk_firm_sic_sections(bad)


def test_candidate_table_is_pounds_all_standard_rated_ordered_by_id() -> None:
    table = build_candidate_rules_table(_firms())
    assert tuple(table.columns) == UK_FIRM_CANDIDATE_RULES_COLUMNS
    assert table["firm_id"].tolist() == [1, 2, 3, 4]
    assert table["firm_turnover"].tolist() == [0.0, 90_000.0, 40_000.0, 120_000.0]
    assert table["firm_input_costs"].tolist() == [0.0, 30_000.0, 10_000.0, 80_000.0]
    assert table["firm_employees"].tolist() == [0, 3, 1, 12]
    assert table["firm_sic_section"].tolist() == ["UNKNOWN", "G", "Q", "C"]
    pd.testing.assert_series_equal(
        table["firm_standard_rated_supplies"],
        table["firm_turnover"],
        check_names=False,
    )
    assert (table["firm_exempt_supplies"] == 0).all()
    np.testing.assert_allclose(
        table["firm_input_vat"], ASSUMED_INPUT_VAT_RATE * table["firm_input_costs"]
    )
    assert ASSUMED_INPUT_VAT_RATE == 0.20


def test_candidate_table_needs_only_the_pre_calibration_columns() -> None:
    firms = _firms()[
        ["firm_id", "sic_code", "annual_turnover_k", "annual_input_k", "employment"]
    ]
    assert len(build_candidate_rules_table(firms)) == 4
    with pytest.raises(ValueError, match="firm_id"):
        build_candidate_rules_table(firms.drop(columns="firm_id"))


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("annual_turnover_k", -1.0, "annual_turnover_k"),
        ("annual_input_k", np.nan, "annual_input_k"),
        ("employment", 1.5, "whole numbers"),
        ("firm_id", 2, "repeats"),
    ],
)
def test_candidate_table_refuses_bad_rows(column, value, match) -> None:
    firms = _firms()
    if column == "employment":
        firms[column] = firms[column].astype(float)
    firms.loc[0, column] = value
    with pytest.raises(ValueError, match=match):
        build_candidate_rules_table(firms)


def test_export_follows_the_contract() -> None:
    table = to_policyengine_uk_firm_table(_firms(), threshold_k=90.0)
    assert tuple(table.columns) == UK_FIRM_TABLE_COLUMNS
    assert UK_FIRM_TABLE_COLUMNS == (
        *UK_FIRM_POLICYENGINE_UK_COLUMNS,
        *UK_FIRM_MICROCOSM_COLUMNS,
    )
    for absent in ("firm_region", "firm_legal_form", "firm_paybill"):
        assert absent not in table.columns
    assert table["firm_id"].tolist() == [1, 2, 3, 4]
    assert table["firm_weight"].tolist() == [0.0, 10.0, 4.0, 2.5]
    assert table["source_firm_key"].tolist() == [
        f"2024-25:experimental_uk_firm:{i}" for i in (1, 2, 3, 4)
    ]
    validate_uk_firm_table(table)


def test_export_splits_supplies_by_scope_and_flags_voluntary_registration() -> None:
    table = to_policyengine_uk_firm_table(_firms(), threshold_k=90.0).set_index(
        "firm_id"
    )
    assert table["firm_standard_rated_supplies"].tolist() == [
        0.0,
        90_000.0,
        0.0,
        120_000.0,
    ]
    assert table["firm_exempt_supplies"].tolist() == [0.0, 0.0, 40_000.0, 0.0]
    # Firm 2 sits on the threshold, which registration must exceed; firm 3 is
    # registered below it; firm 4 is above it.
    assert table["firm_vat_voluntarily_registered"].tolist() == [
        False,
        True,
        True,
        False,
    ]
    np.testing.assert_allclose(
        table["firm_input_vat"], [0.0, 6_000.0, 2_000.0, 16_000.0]
    )


def test_export_threshold_moves_the_voluntary_flag() -> None:
    table = to_policyengine_uk_firm_table(_firms(), threshold_k=85.0)
    assert table["firm_vat_voluntarily_registered"].tolist() == [
        False,
        False,
        True,
        False,
    ]
    with pytest.raises(ValueError, match="threshold_k"):
        to_policyengine_uk_firm_table(_firms(), threshold_k=0.0)


def test_export_needs_the_calibrated_columns() -> None:
    with pytest.raises(ValueError, match="vat_scope"):
        to_policyengine_uk_firm_table(
            _firms().drop(columns="vat_scope"), threshold_k=90.0
        )


def test_export_refuses_flags_that_are_not_boolean() -> None:
    firms = _firms().astype({"vat_registered": object})
    firms.loc[0, "vat_registered"] = np.nan
    with pytest.raises(ValueError, match="vat_registered must be boolean"):
        to_policyengine_uk_firm_table(firms, threshold_k=90.0)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda t: t.assign(firm_weight=[-1.0, 1.0, 1.0, 1.0]), "firm_weight"),
        (lambda t: t.assign(firm_weight=[np.inf, 1.0, 1.0, 1.0]), "firm_weight"),
        (lambda t: t.assign(firm_id=[1, 1, 3, 4]), "repeats"),
        (lambda t: t.assign(firm_id=[2, 1, 3, 4]), "ordered"),
        (lambda t: t.drop(columns="base_weight"), "columns"),
        (lambda t: t[list(reversed(t.columns))], "columns"),
        (lambda t: t.assign(firm_sic_section=["Z", "A", "A", "A"]), "SICSection"),
        (lambda t: t.assign(vat_scope=[1, 1, 0, 1]), "boolean"),
        (lambda t: t.assign(firm_exempt_supplies=[0.0, 1.0, 40_000.0, 0.0]), "add up"),
        (
            lambda t: t.assign(
                firm_vat_voluntarily_registered=[True, True, True, False]
            ),
            "needs vat_registered",
        ),
    ],
)
def test_validation_refuses_tables_outside_the_contract(mutate, match) -> None:
    table = to_policyengine_uk_firm_table(_firms(), threshold_k=90.0)
    with pytest.raises(ValueError, match=match):
        validate_uk_firm_table(mutate(table))


def test_validation_refuses_supplies_against_scope() -> None:
    table = to_policyengine_uk_firm_table(_firms(), threshold_k=90.0)
    swapped = table.assign(
        firm_standard_rated_supplies=table["firm_exempt_supplies"],
        firm_exempt_supplies=table["firm_standard_rated_supplies"],
    )
    with pytest.raises(ValueError, match="vat_scope"):
        validate_uk_firm_table(swapped)


def _attributed_firms() -> pd.DataFrame:
    # Firms 4, 2, 1, 3: a Welsh company, a local authority in the North East, a
    # Scottish partnership and a London sole trader.
    return _firms().assign(
        ons_legal_status=np.array([1, 6, 3, 2], dtype=np.int64),
        ons_region_code=["W92000004", "E12000001", "S92000003", "E12000007"],
    )


def test_base_export_columns_are_unchanged() -> None:
    assert UK_FIRM_POLICYENGINE_UK_COLUMNS == (
        "firm_id",
        "firm_weight",
        "firm_turnover",
        "firm_input_costs",
        "firm_employees",
        "firm_sic_section",
        "firm_standard_rated_supplies",
        "firm_exempt_supplies",
        "firm_vat_voluntarily_registered",
        "firm_input_vat",
    )
    assert UK_FIRM_TABLE_COLUMNS == (
        *UK_FIRM_POLICYENGINE_UK_COLUMNS,
        "in_frame",
        "vat_scope",
        "vat_registered",
        "base_weight",
        "data_vintage",
        "source_firm_id",
        "source_firm_key",
    )
    table = to_policyengine_uk_firm_table(_firms(), threshold_k=90.0)
    assert tuple(table.columns) == UK_FIRM_TABLE_COLUMNS


def test_extended_columns_add_the_attributes_after_the_base_ones() -> None:
    assert UK_FIRM_EXTENDED_POLICYENGINE_UK_COLUMNS == (
        *UK_FIRM_POLICYENGINE_UK_COLUMNS,
        *UK_FIRM_ATTRIBUTE_POLICYENGINE_UK_COLUMNS,
    )
    assert UK_FIRM_MICROCOSM_ONLY_COLUMNS == (
        *UK_FIRM_MICROCOSM_COLUMNS,
        "ons_legal_status",
        "ons_region_code",
    )
    assert UK_FIRM_EXTENDED_TABLE_COLUMNS == (
        *UK_FIRM_EXTENDED_POLICYENGINE_UK_COLUMNS,
        *UK_FIRM_MICROCOSM_ONLY_COLUMNS,
    )
    assert "region" not in UK_FIRM_EXTENDED_TABLE_COLUMNS
    assert len(set(UK_FIRM_EXTENDED_TABLE_COLUMNS)) == len(
        UK_FIRM_EXTENDED_TABLE_COLUMNS
    )


def test_attribute_maps_match_policyengine_uk_members() -> None:
    assert UK_FIRM_LEGAL_FORMS == _POLICYENGINE_UK_FIRM_LEGAL_FORM_MEMBERS
    assert set(UK_FIRM_ONS_LEGAL_STATUS_TO_LEGAL_FORM) == set(range(1, 8))
    assert set(UK_FIRM_ONS_LEGAL_STATUS_TO_LEGAL_FORM.values()) == set(
        UK_FIRM_LEGAL_FORMS
    )
    assert UK_FIRM_PUBLIC_BODY_ONS_LEGAL_STATUSES == {5, 6}
    # Every Region member but UNKNOWN, each from exactly one GSS code.
    regions = list(UK_FIRM_REGION_CODE_TO_REGION.values())
    assert sorted(regions) == sorted(_POLICYENGINE_UK_REGION_MEMBERS[1:])
    assert len(set(regions)) == len(regions)
    assert UK_FIRM_ENGLISH_REGION_CODES == {f"E1200000{i}" for i in range(1, 10)}
    assert set(UK_FIRM_REGION_CODE_TO_REGION) == UK_FIRM_ENGLISH_REGION_CODES | {
        "W92000004",
        "S92000003",
        "N92000002",
    }


def test_extended_export_maps_the_ons_attributes() -> None:
    table = to_policyengine_uk_firm_table(_attributed_firms(), threshold_k=90.0)
    assert tuple(table.columns) == UK_FIRM_EXTENDED_TABLE_COLUMNS
    validate_uk_firm_table(table)
    # Ordered by firm_id: 1, 2, 3, 4.
    assert table["ons_legal_status"].tolist() == [3, 6, 2, 1]
    assert pd.api.types.is_integer_dtype(table["ons_legal_status"])
    assert table["ons_region_code"].tolist() == [
        "S92000003",
        "E12000001",
        "E12000007",
        "W92000004",
    ]
    assert table["firm_legal_form"].tolist() == [
        "PARTNERSHIP",
        "OTHER",
        "SOLE_TRADER",
        "COMPANY",
    ]
    assert table["firm_region"].tolist() == [
        "SCOTLAND",
        "NORTH_EAST",
        "LONDON",
        "WALES",
    ]
    assert table["firm_in_england"].tolist() == [False, True, True, False]
    assert table["firm_is_public_body"].tolist() == [False, True, False, False]
    assert pd.api.types.is_bool_dtype(table["firm_in_england"])
    assert pd.api.types.is_bool_dtype(table["firm_is_public_body"])
    base = to_policyengine_uk_firm_table(_firms(), threshold_k=90.0)
    pd.testing.assert_frame_equal(table[list(UK_FIRM_TABLE_COLUMNS)], base)


@pytest.mark.parametrize("column", UK_FIRM_ONS_ATTRIBUTE_COLUMNS)
def test_export_refuses_one_ons_attribute_without_the_other(column) -> None:
    with pytest.raises(ValueError, match="needs both"):
        to_policyengine_uk_firm_table(
            _attributed_firms().drop(columns=column), threshold_k=90.0
        )


@pytest.mark.parametrize(
    ("column", "values", "match"),
    [
        ("ons_legal_status", np.array([1, 8, 3, 2]), "legal status codes"),
        ("ons_legal_status", np.array([1, 0, 3, 2]), "legal status codes"),
        ("ons_legal_status", [1.0, np.nan, 3.0, 2.0], "integers"),
        ("ons_region_code", ["W92000004", "E12000010", "S92000003", ""], "GSS"),
        ("ons_region_code", ["W92000004", None, "S92000003", "E12000007"], "str"),
        ("ons_region_code", ["W92000004", "LONDON", "S92000003", "E12000007"], "GSS"),
    ],
)
def test_export_never_falls_back_to_policyengine_uk_defaults(
    column, values, match
) -> None:
    firms = _attributed_firms().assign(**{column: values})
    with pytest.raises(ValueError, match=match):
        to_policyengine_uk_firm_table(firms, threshold_k=90.0)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda t: t.assign(firm_legal_form="LIMITED"), "FirmLegalForm"),
        (lambda t: t.assign(firm_region="UNKNOWN"), "Region member"),
        (lambda t: t.assign(firm_region="E12000007"), "Region member"),
        (lambda t: t.assign(firm_legal_form="COMPANY"), "follow ons_legal_status"),
        (lambda t: t.assign(firm_region="LONDON"), "follow ons_region_code"),
        (lambda t: t.assign(firm_in_england=True), "follow ons_region_code"),
        (lambda t: t.assign(firm_is_public_body=False), "follow ons_legal_status"),
        (lambda t: t.assign(firm_in_england=[0, 1, 1, 0]), "boolean"),
        (lambda t: t.assign(ons_legal_status=[3.0, 6.0, 2.0, 1.0]), "integers"),
        (lambda t: t.assign(ons_legal_status=[3, 9, 2, 1]), "legal status codes"),
        (lambda t: t.assign(ons_region_code=[1, 2, 3, 4]), "strings"),
        (lambda t: t.assign(ons_region_code="E92000001"), "GSS"),
        (lambda t: t.drop(columns="ons_region_code"), "columns"),
        (lambda t: t.drop(columns="firm_region"), "columns"),
        (lambda t: t.rename(columns={"ons_region_code": "region"}), "columns"),
    ],
)
def test_validation_refuses_extended_tables_outside_the_contract(mutate, match) -> None:
    table = to_policyengine_uk_firm_table(_attributed_firms(), threshold_k=90.0)
    with pytest.raises(ValueError, match=match):
        validate_uk_firm_table(mutate(table))


@pytest.mark.parametrize("extended", [False, True])
def test_validation_refuses_missing_values_in_any_column(extended) -> None:
    firms = _attributed_firms() if extended else _firms()
    table = to_policyengine_uk_firm_table(firms, threshold_k=90.0)
    for column in table.columns:
        broken = table.astype({column: object})
        broken.loc[1, column] = np.nan
        with pytest.raises(ValueError, match=f"missing values in .*{column}"):
            validate_uk_firm_table(broken)


def test_input_status_covers_every_exported_policyengine_uk_column() -> None:
    assert set(UK_FIRM_POLICYENGINE_UK_INPUT_STATUS.values()) <= set(
        UK_FIRM_INPUT_STATUSES
    )
    assert set(UK_FIRM_EXTENDED_POLICYENGINE_UK_COLUMNS) <= set(
        UK_FIRM_POLICYENGINE_UK_INPUT_STATUS
    )
    exported = {
        name
        for name, status in UK_FIRM_POLICYENGINE_UK_INPUT_STATUS.items()
        if status in ("set", "derived", "assumed")
    }
    assert exported == set(UK_FIRM_EXTENDED_POLICYENGINE_UK_COLUMNS)
    # firm_in_england is the enterprise's region standing in for its
    # property's location, so it is an assumption, not a derivation.
    assert {
        name
        for name, status in UK_FIRM_POLICYENGINE_UK_INPUT_STATUS.items()
        if status == "assumed"
    } == {"firm_input_vat", "firm_legal_form", "firm_is_public_body", "firm_in_england"}
    assert all(
        name.startswith("firm_") for name in UK_FIRM_POLICYENGINE_UK_INPUT_STATUS
    )
    assert not set(UK_FIRM_MICROCOSM_ONLY_COLUMNS) & set(
        UK_FIRM_POLICYENGINE_UK_INPUT_STATUS
    )
    with pytest.raises(TypeError):
        UK_FIRM_POLICYENGINE_UK_INPUT_STATUS["firm_paybill"] = "set"  # type: ignore[index]
