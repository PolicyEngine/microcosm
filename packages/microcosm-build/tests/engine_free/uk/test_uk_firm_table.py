"""UK firm tables in policyengine-uk names: candidates, export and contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_table import (
    ASSUMED_INPUT_VAT_RATE,
    UK_FIRM_CANDIDATE_RULES_COLUMNS,
    UK_FIRM_MICROCOSM_COLUMNS,
    UK_FIRM_POLICYENGINE_UK_COLUMNS,
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
