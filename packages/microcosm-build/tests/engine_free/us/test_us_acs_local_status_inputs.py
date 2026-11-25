"""ACS local lane: marital, military and occupation status (microcosm#1720)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.us_runtime import acs_multispine
from microcosm.build.us_runtime.acs_local_status_inputs import (
    ACS_LOCAL_MARITAL_STATUS_COLUMNS,
    ACS_LOCAL_OCCUPATION_FLAG_COLUMNS,
    ACS_LOCAL_STATUS_INPUTS_COLUMNS,
    ACS_LOCAL_STATUS_INPUTS_ISSUE,
    ACS_LOCAL_STATUS_INPUTS_METHOD,
    map_acs_local_status_inputs,
)
from microcosm.build.us_runtime.acs_pums import (
    _ACS_TO_CPS_MARITAL_STATUS,
    ACS_2024_1YR_SPINE,
    AcsPumsSource,
)
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.org_wages import (
    FLSA_EXECUTIVE_ADMINISTRATIVE_PROFESSIONAL_OCCUPATION_CODES,
    derive_us_org_occupation_inputs,
    us_flsa_occupation_flags,
)
from microcosm.build.us_runtime.relationship_inputs import (
    derive_us_relationship_inputs_from_manifest,
    us_marital_status_inputs,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

TAG = spine_column("person")


# ---------------------------------------------------------------------------
# The donor's own rules, shared with the ACS rows
# ---------------------------------------------------------------------------


def test_the_loader_maps_separated_and_widowed_to_the_donor_codes() -> None:
    """The premise of the marital rule: ACS MAR 4 is A_MARITL 6 and MAR 2 is 4."""

    assert _ACS_TO_CPS_MARITAL_STATUS[4] == 6
    assert _ACS_TO_CPS_MARITAL_STATUS[2] == 4


def test_the_donor_stage_still_derives_its_marital_leaves_from_the_shared_rule():
    codes = np.asarray([1, 2, 3, 4, 5, 6, 7])
    frame = pd.DataFrame({"PH_SEQ": np.arange(1, 8), "P_SEQ": 1, "A_MARITL": codes})
    derived = derive_us_relationship_inputs_from_manifest(
        frame,
        SourceOperationSpec(kind="derive_relationship_inputs", parameters={}),
        None,
    )
    shared = us_marital_status_inputs(codes)
    assert derived["is_separated"].tolist() == [False] * 5 + [True, False]
    assert derived["is_surviving_spouse"].tolist() == [False] * 3 + [True] + [False] * 3
    for column in ACS_LOCAL_MARITAL_STATUS_COLUMNS:
        np.testing.assert_array_equal(derived[column].to_numpy(), shared[column])


def test_the_shared_marital_rule_refuses_an_unknown_code() -> None:
    with pytest.raises(ValueError, match="A_MARITL codes 1-7"):
        us_marital_status_inputs(np.asarray([1.0, 8.0]))
    with pytest.raises(ValueError, match="A_MARITL codes 1-7"):
        us_marital_status_inputs(np.asarray([np.nan]))


def test_the_donor_stage_still_derives_its_occupation_flags_from_the_shared_rule():
    occupation = [0, 8, 41, 52, 53, 1, 50, 20]
    person = pd.DataFrame(
        {
            "PRDTRACE": 1,
            "PRDTHSP": 0,
            "POCCU2": occupation,
            "WEIND": 23,
            "WEMIND": 15,
            "WKSWORK": 0,
        }
    )
    derived = derive_us_org_occupation_inputs(person)
    shared = us_flsa_occupation_flags(pd.Series(occupation))
    assert list(shared.columns) == [
        "has_never_worked",
        "is_military",
        "is_computer_scientist",
        "is_farmer_fisher",
        "is_executive_administrative_professional",
    ]
    for column in shared.columns:
        np.testing.assert_array_equal(derived[column], shared[column])
    assert shared["is_computer_scientist"].tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    assert shared["is_farmer_fisher"].tolist() == [0, 0, 1, 0, 0, 0, 0, 0]
    assert shared["is_military"].tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    assert shared["has_never_worked"].tolist() == [0, 0, 0, 0, 1, 0, 0, 0]
    assert shared["is_executive_administrative_professional"].tolist() == [
        code in FLSA_EXECUTIVE_ADMINISTRATIVE_PROFESSIONAL_OCCUPATION_CODES
        for code in occupation
    ]


# ---------------------------------------------------------------------------
# The ACS mapping
# ---------------------------------------------------------------------------


def _acs_frame(**overrides) -> Frame:
    """Eight ACS persons in four households, with the stage's source columns."""

    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 9),
            "person_household_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "person_tax_unit_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "person_spm_unit_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "person_family_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "person_marital_unit_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "AGEP": [30, 28, 45, 10, 70, 40, 22, 16],
            # A_MARITL: separated, married, widowed, never married (a child),
            # widowed, separated, never married, never married.
            "A_MARITL": [6, 1, 4, 7, 4, 6, 7, 7],
            # MIL: on active duty, never, past duty (a veteran), blank under
            # 17, never, active duty, reserves only, blank under 17 (at 16).
            "ACS_MIL": [1, 4, 2, np.nan, 4, 1, 3, np.nan],
            "ESR": [4, 1, 1, np.nan, 6, 6, 1, 6],
            # POCCU2: military, an EAP code, computer, 0 under 16, never
            # worked, military (a recent veteran: off duty, no job), farm, 53.
            "detailed_occupation_recode": [52, 1, 8, 0, 53, 52, 41, 53],
            **overrides,
        }
    )
    households = [1, 2, 3, 4]
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame({"household_id": households}),
            "tax_unit": pd.DataFrame({"tax_unit_id": households}),
            "spm_unit": pd.DataFrame({"spm_unit_id": households}),
            "family": pd.DataFrame({"family_id": households}),
            "marital_unit": pd.DataFrame({"marital_unit_id": households}),
        },
        US_SCHEMA,
        {"household": Weights(np.asarray([10.0, 20.0, 30.0, 40.0]), WeightKind.DESIGN)},
        pd.Series(ACS_2024_1YR_SPINE, index=person.index, dtype=object),
    )


def test_the_acs_rows_get_every_status_column_under_its_rule() -> None:
    frame = _acs_frame()
    result = map_acs_local_status_inputs(frame)
    person = result.frame.table("person")

    assert person["is_separated"].tolist() == [1, 0, 0, 0, 0, 1, 0, 0]
    assert person["is_surviving_spouse"].tolist() == [0, 0, 1, 0, 1, 0, 0, 0]
    # MIL == 1 only; a veteran, a reservist and a blank under 17 are not.
    assert person["is_military"].tolist() == [1, 0, 0, 0, 0, 1, 0, 0]
    assert person["has_never_worked"].tolist() == [0, 0, 0, 0, 1, 0, 0, 1]
    assert person["is_computer_scientist"].tolist() == [0, 0, 1, 0, 0, 0, 0, 0]
    assert person["is_farmer_fisher"].tolist() == [0, 0, 0, 0, 0, 0, 1, 0]
    assert person["is_executive_administrative_professional"].tolist() == [
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    for column in ACS_LOCAL_STATUS_INPUTS_COLUMNS:
        assert person[column].dtype == bool, column
    # The input frame is not modified.
    assert "is_military" not in frame.table("person")

    receipt = result.receipt
    assert receipt["issue"] == ACS_LOCAL_STATUS_INPUTS_ISSUE == "microcosm#1720"
    assert receipt["method"] == ACS_LOCAL_STATUS_INPUTS_METHOD
    assert receipt["acs_persons"] == 8
    assert set(receipt["columns"]) == set(ACS_LOCAL_STATUS_INPUTS_COLUMNS)
    military = receipt["columns"]["is_military"]
    assert military["filled_rows"] == 8
    assert military["preserved_rows"] == 0
    assert military["true_rows"] == 2
    # Household weights 10 and 30 carry the two active-duty persons.
    assert military["weighted_true_share"] == pytest.approx(40.0 / 200.0)
    assert military["donor_rule"] == "org_wages: POCCU2 == 52"
    assert receipt["a_maritl_rows"] == {"1": 1, "4": 2, "6": 2, "7": 3}
    assert receipt["military"]["mil_code_rows"] == {
        "1": 2,
        "2": 1,
        "3": 1,
        "4": 2,
        "blank": 2,
    }
    assert receipt["military"]["blank_rows_age_17_plus"] == 0
    assert receipt["military"]["coded_rows_under_17"] == 0
    agreement = receipt["military"]["agreement"]
    # Both active-duty persons hold a military occupation (POCCU2 52), the
    # donor's rule; only one of them is ESR 4 (Armed Forces, at work).
    assert agreement["donor_rule_poccu2_52"] == {
        "both_rows": 2,
        "mil_only_rows": 0,
        "other_only_rows": 0,
        "other_weighted_true_share": pytest.approx(40.0 / 200.0),
    }
    assert agreement["esr_armed_forces_4_or_5"] == {
        "both_rows": 1,
        "mil_only_rows": 1,
        "other_only_rows": 0,
        "other_weighted_true_share": pytest.approx(10.0 / 200.0),
    }
    json.dumps(receipt, allow_nan=False)

    assert set(result.native_inputs) == set(ACS_LOCAL_STATUS_INPUTS_COLUMNS)
    for entry in result.native_inputs.values():
        assert entry["provenance"] == "acs_2024_1yr_native"
        assert entry["missing_rows"] == 0
        assert entry["observed_rows"] == 8


def test_the_recent_veteran_is_military_only_under_the_donor_occupation_rule():
    """Person 6 is off duty with no job (ESR 6) and a military last job."""

    result = map_acs_local_status_inputs(
        _acs_frame(ACS_MIL=[1, 4, 2, np.nan, 4, 2, 3, np.nan])
    )
    agreement = result.receipt["military"]["agreement"]["donor_rule_poccu2_52"]
    assert agreement == {
        "both_rows": 1,
        "mil_only_rows": 0,
        "other_only_rows": 1,
        "other_weighted_true_share": pytest.approx(40.0 / 200.0),
    }
    assert result.frame.table("person")["is_military"].tolist() == [1] + [0] * 7


def test_stored_acs_values_stand_and_only_missing_cells_are_filled() -> None:
    stored = pd.Series([True, np.nan, np.nan, np.nan, np.nan, False, np.nan, np.nan])
    result = map_acs_local_status_inputs(_acs_frame(is_separated=stored))
    person = result.frame.table("person")
    # Row 6's stored False stands against A_MARITL 6; row 1's stored True too.
    assert person["is_separated"].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    entry = result.receipt["columns"]["is_separated"]
    assert (entry["filled_rows"], entry["preserved_rows"]) == (6, 2)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"ACS_MIL": [1, 4, 2, np.nan, 4, 1, 5, np.nan]}, r"MIL requires blank"),
        ({"A_MARITL": [6, 1, 4, 7, 4, 6, 7, 0]}, "A_MARITL codes 1-7"),
        ({"ESR": [4, 1, 1, np.nan, 6, 6, 1, 9]}, r"ESR requires blank"),
        ({"is_military": [True, 2, 0, 0, 0, 0, 0, 0]}, "must be boolean"),
    ],
)
def test_malformed_sources_are_refused(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        map_acs_local_status_inputs(_acs_frame(**overrides))


@pytest.mark.parametrize(
    "column", ["A_MARITL", "ACS_MIL", "detailed_occupation_recode", "ESR", "AGEP"]
)
def test_a_missing_source_column_is_refused(column) -> None:
    frame = _acs_frame()
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = tables["person"].drop(columns=[column])
    reduced = Frame(
        tables,
        US_SCHEMA,
        {"household": frame.weights_for("household")},
        frame.strata,
    )
    with pytest.raises(ValueError, match=r"microcosm#1720"):
        map_acs_local_status_inputs(reduced)


# ---------------------------------------------------------------------------
# The multispine seam
# ---------------------------------------------------------------------------


def _donor() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_household_id": [1, 1, 2, 2],
            "person_tax_unit_id": [1, 1, 2, 2],
            "person_spm_unit_id": [1, 1, 2, 2],
            "person_family_id": [1, 1, 2, 2],
            "person_marital_unit_id": [1, 1, 2, 2],
            "age": [42.0, 38.0, 65.0, 12.0],
            "is_female": [False, True, True, False],
            "qualified_dividend_income": [100.0, 20.0, 2_000.0, 0.0],
            **{
                column: [True, False, False, False]
                for column in ACS_LOCAL_STATUS_INPUTS_COLUMNS
            },
        }
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame({"household_id": [1, 2], "state_fips": [6, 36]}),
            "tax_unit": pd.DataFrame({"tax_unit_id": [1, 2]}),
            "spm_unit": pd.DataFrame({"spm_unit_id": [1, 2]}),
            "family": pd.DataFrame({"family_id": [1, 2]}),
            "marital_unit": pd.DataFrame({"marital_unit_id": [1, 2]}),
        },
        US_SCHEMA,
        {"household": Weights(np.asarray([100.0, 300.0]), WeightKind.DESIGN)},
        pd.Series("asec_puf", index=person.index, dtype=object),
    )


def _raw_acs() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": [10, 11],
            "person_household_id": [10, 10],
            "person_tax_unit_id": [10, 10],
            "person_spm_unit_id": [10, 10],
            "person_family_id": [10, 10],
            "person_marital_unit_id": [10, 10],
            "AGEP": [40, 15],
            "SEX": [1, 2],
            "RELSHIPP": [20, 25],
            "ADJINC": [1_100_000, 1_100_000],
            "WAGP": [50_000.0, 0.0],
            "SEMP": [0.0, np.nan],
            "SSP": [0.0, 0.0],
            "SSIP": [0.0, 0.0],
            "RETP": [0.0, 0.0],
            "INTP": [100.0, 0.0],
            "A_MARITL": [6, 7],
            "ACS_MIL": [1, np.nan],
            "ESR": [4, np.nan],
            "detailed_occupation_recode": [52, 0],
        }
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {
                    "household_id": [10],
                    "state_fips": [6],
                    "puma": ["0600100"],
                    "ADJHSG": [1_000_000],
                    "TEN": [3],
                    "RNTP": [1_000.0],
                    "GRNTP": [1_200.0],
                    "TAXAMT": [np.nan],
                }
            ),
            "tax_unit": pd.DataFrame({"tax_unit_id": [10]}),
            "spm_unit": pd.DataFrame({"spm_unit_id": [10]}),
            "family": pd.DataFrame({"family_id": [10]}),
            "marital_unit": pd.DataFrame({"marital_unit_id": [10]}),
        },
        US_SCHEMA,
        {"household": Weights(np.asarray([50.0]), WeightKind.DESIGN)},
        pd.Series(ACS_2024_1YR_SPINE, index=person.index, dtype=object),
    )


def _run(monkeypatch, tmp_path, *, status_inputs: bool):
    monkeypatch.setattr(
        acs_multispine,
        "build_acs_pums_unit_frame",
        lambda source, *, chunksize: (_raw_acs(), {"spine": ACS_2024_1YR_SPINE}),
    )
    return acs_multispine.build_optional_acs_multispine(
        _donor(),
        AcsPumsSource(tmp_path / "csv_hus.zip", tmp_path / "csv_pus.zip"),
        acs_share=0.25,
        target_families={"person": {"tax_detail": ("qualified_dividend_income",)}},
        status_inputs=status_inputs,
        seed=4,
        n_estimators=2,
    )


def test_the_seam_maps_the_acs_rows_before_the_transfer_and_keeps_the_donor(
    monkeypatch, tmp_path
) -> None:
    with_stage = _run(monkeypatch, tmp_path, status_inputs=True)
    without_stage = _run(monkeypatch, tmp_path, status_inputs=False)

    person = with_stage.frame.table("person")
    acs = person.loc[person[TAG].eq(ACS_2024_1YR_SPINE)].reset_index(drop=True)
    donor = person.loc[~person[TAG].eq(ACS_2024_1YR_SPINE)].reset_index(drop=True)
    assert acs["is_separated"].tolist() == [True, False]
    assert acs["is_military"].tolist() == [True, False]
    for column in ACS_LOCAL_STATUS_INPUTS_COLUMNS:
        assert acs[column].notna().all(), column
        # The donor rows keep the donor's own values.
        assert donor[column].tolist() == [True, False, False, False], column
    for column in ACS_LOCAL_OCCUPATION_FLAG_COLUMNS:
        assert not acs[column].any(), column
    receipt = with_stage.provenance["acs_local_status_inputs"]
    assert receipt["acs_persons"] == 2
    assert set(ACS_LOCAL_STATUS_INPUTS_COLUMNS) <= set(
        with_stage.provenance["native_inputs"]
    )
    json.dumps(with_stage.provenance, allow_nan=False)

    # Without the flag (the pool lane) nothing is mapped and nothing else moves.
    assert "acs_local_status_inputs" not in without_stage.provenance
    other = without_stage.frame.table("person")
    other_acs = other.loc[other[TAG].eq(ACS_2024_1YR_SPINE)]
    for column in ACS_LOCAL_STATUS_INPUTS_COLUMNS:
        assert other_acs[column].isna().all(), column
    pd.testing.assert_frame_equal(
        person.drop(columns=list(ACS_LOCAL_STATUS_INPUTS_COLUMNS)),
        other.drop(columns=list(ACS_LOCAL_STATUS_INPUTS_COLUMNS)),
        check_dtype=False,
        check_like=True,
    )
    assert (
        with_stage.provenance["imputed_inputs"]
        == without_stage.provenance["imputed_inputs"]
    )
