"""Native ACS SNAP receipt anchor in the ACS local lane (microcosm#1022)."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.acs_local_receipt_anchors import (
    ACS_LOCAL_RECEIPT_ANCHOR_GATE_NAME,
    ACS_LOCAL_RECEIPT_ANCHOR_ISSUE,
    ACS_LOCAL_RECEIPT_ANCHOR_METHOD,
    ACS_SNAP_HOUSEHOLD_SHARE_BAND,
    acs_local_receipt_anchor_signal_gate,
    require_acs_receipt_anchor_sources,
    with_acs_local_snap_receipt_anchor,
)
from microcosm.build.us_runtime.acs_local_spm_units import (
    split_acs_adult_nonrelative_spm_units,
)
from microcosm.build.us_runtime.acs_pums import (
    ACS_2024_1YR_SPINE,
    AcsPumsSource,
    build_acs_pums_unit_frame,
)
from microcosm.build.us_runtime.base_pool import spine_column, with_optional_acs_spine
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

pytest.importorskip("microunit")  # sanctioned tax-unit constructor (us extra)

UNIT_TAG = spine_column("spm_unit")
HOUSEHOLD_TAG = spine_column("household")


def _write_csv_zip(path: Path, member: str, rows: list[dict[str, object]]) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(member, pd.DataFrame(rows).to_csv(index=False))


def _household(serialno: str, persons: int, **overrides: object) -> dict:
    row: dict[str, object] = {
        "SERIALNO": serialno,
        "ST": "06",
        "PUMA": "00100",
        "WGTP": 10,
        "NP": persons,
        "ADJHSG": 1_000_000,
        "TEN": 3,
        "RNTP": 1_000,
        "GRNTP": 1_200,
        "TAXAMT": None,
        "TYPEHUGQ": 1,
        "FS": 2,
    }
    row.update(overrides)
    return row


def _person(serialno: str, sporder: int, relationship: int, **overrides) -> dict:
    row: dict[str, object] = {
        "SERIALNO": serialno,
        "SPORDER": sporder,
        "RELSHIPP": relationship,
        "AGEP": 40,
        "SEX": 1,
        "MAR": 5,
        "ADJINC": 1_000_000,
        "WAGP": 30_000,
        "SEMP": 0,
        "SSP": 0,
        "SSIP": 0,
        "RETP": 0,
        "INTP": 0,
        "PWGTP": 10,
        "PAP": 0,
    }
    row.update(overrides)
    return row


#: Seven ACS records (sorted SERIALNO order = household ids 1-7):
#: A  FS 1: reference, a 30-year-old roommate (own unit), a nonrelative child;
#:    the QRF names neither unit, so the reference unit is forced
#: B  FS 2: reference with public assistance income, a 50-year-old roommate
#: C  noninstitutional group quarters, FS blank
#: D  FS 1: a married couple and their child (one unit)
#: E  FS 2: a reference person alone
#: F  FS 1: reference and a 45-year-old roommate; the QRF names the roommate
#: G  FS 1: reference and a 33-year-old roommate; the QRF names the reference
_HOUSEHOLDS = [
    _household("A", 3, WGTP=10, FS=1),
    _household("B", 2, WGTP=20, FS=2),
    _household("C", 1, WGTP=0, TYPEHUGQ=3, TEN=None, RNTP=None, GRNTP=None, FS=None),
    _household("D", 3, WGTP=30, FS=1),
    _household("E", 1, WGTP=40, FS=2),
    _household("F", 2, WGTP=50, FS=1),
    _household("G", 2, WGTP=60, FS=1),
]
_PERSONS = [
    _person("A", 1, 20),
    _person("A", 2, 34, AGEP=30),
    _person("A", 3, 36, AGEP=8, WAGP=None, PAP=None),
    _person("B", 1, 20, PAP=1_200),
    _person("B", 2, 34, AGEP=50),
    _person("C", 1, 38, AGEP=70, PWGTP=7),
    _person("D", 1, 20, MAR=1),
    _person("D", 2, 21, MAR=1, SEX=2),
    _person("D", 3, 25, AGEP=5, WAGP=None, PAP=None),
    _person("E", 1, 20),
    _person("F", 1, 20),
    _person("F", 2, 34, AGEP=45),
    _person("G", 1, 20),
    _person("G", 2, 34, AGEP=33),
]
#: Split units, in id order: A reference (+ child), A roommate, B reference,
#: B roommate, C, D family, E, F reference, F roommate, G reference,
#: G roommate.
_RULE = [True, False, False, False, False, True, False, False, True, True, False]
#: What the QRF transfer is made to have written.
_TRANSFERRED_SNAP = [
    False,
    False,
    True,
    False,
    True,
    True,
    False,
    False,
    True,
    True,
    False,
]
_TRANSFERRED_TANF = [
    False,
    False,
    False,
    False,
    False,
    True,
    True,
    False,
    True,
    False,
    False,
]


def _source(tmp_path: Path, households=None, persons=None) -> AcsPumsSource:
    household_zip = tmp_path / "csv_hus.zip"
    person_zip = tmp_path / "csv_pus.zip"
    _write_csv_zip(household_zip, "psam_husa.csv", households or _HOUSEHOLDS)
    _write_csv_zip(person_zip, "psam_pusa.csv", persons or _PERSONS)
    return AcsPumsSource(household_zip=household_zip, person_zip=person_zip)


def _with_tables(frame: Frame, **tables: pd.DataFrame) -> Frame:
    replaced = {entity: frame.table(entity) for entity in frame.entities}
    replaced.update(tables)
    return Frame(
        replaced,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
    )


def _transferred(split: Frame, snap=None, tanf=None) -> Frame:
    """The split ACS frame with the QRF transfer's SPM-unit receipt leaves."""

    spm_unit = split.table("spm_unit").assign(
        receives_snap=np.asarray(snap or _TRANSFERRED_SNAP, dtype=bool),
        receives_tanf=np.asarray(tanf or _TRANSFERRED_TANF, dtype=bool),
    )
    return _with_tables(split, spm_unit=spm_unit)


@pytest.fixture(scope="module")
def loader_frame(tmp_path_factory) -> Frame:
    frame, _metadata = build_acs_pums_unit_frame(
        _source(tmp_path_factory.mktemp("acs"))
    )
    return frame


@pytest.fixture(scope="module")
def transferred(loader_frame) -> Frame:
    split, _receipt = split_acs_adult_nonrelative_spm_units(loader_frame)
    return _transferred(split)


@pytest.fixture(scope="module")
def anchored(transferred) -> tuple[Frame, dict]:
    return with_acs_local_snap_receipt_anchor(transferred)


# --- the loader --------------------------------------------------------------


def test_loader_carries_fs_on_households_and_pap_on_persons(loader_frame) -> None:
    household = loader_frame.table("household")
    assert household["FS"].tolist()[:2] == [1, 2]
    assert pd.isna(household["FS"].iloc[2])  # group quarters: blank
    person = loader_frame.table("person").sort_values("source_row_id")
    assert person["PAP"].iloc[3] == 1_200
    # Under 15 the Census blank stays missing, never a zero.
    assert pd.isna(person["PAP"].iloc[2])


# --- the rule ----------------------------------------------------------------


def test_fs_constrains_the_housing_unit_and_the_qrf_names_the_recipient(
    anchored,
) -> None:
    frame, _receipt = anchored
    assert frame.table("spm_unit")["receives_snap"].tolist() == _RULE
    assert frame.table("spm_unit")["receives_snap"].dtype == np.dtype(bool)
    # FS == 2 clears the transferred True on B's reference unit; group
    # quarters are False even where the transfer said True; D's only unit is
    # True.
    assert frame.n("spm_unit") == 11


def test_a_roommate_unit_is_no_longer_forced(anchored) -> None:
    """microcosm#1062 review: FS == 1 says someone in the housing unit got
    SNAP, not that every SPM unit did. A's two units hold FS == 1 and the QRF
    named neither: the reference unit takes the receipt and the roommate's
    unit, which may buy and prepare food apart, stays a non-reporter for the
    take-up draw."""

    frame, receipt = anchored
    snap = frame.table("spm_unit")["receives_snap"].tolist()
    assert snap[:2] == [True, False]  # A: reference forced, roommate not
    # F: the QRF named the roommate's unit, which keeps it; the constraint is
    # met, so the reference unit is not forced.
    assert snap[7:9] == [False, True]
    # G: the QRF named the reference unit; the roommate's unit stays False.
    assert snap[9:11] == [True, False]
    assert receipt["snap"]["resolution"] == {
        "single_unit_households": 1,
        "several_unit_households_met_by_qrf": 2,
        "several_unit_households_reference_forced": 1,
        "reference_units_true_via_qrf": 1,
        "reference_units_forced": 1,
        "non_reference_units_true_via_qrf": 1,
        "non_reference_units_false": 2,
        "reference_units_false": 1,
        "weighted": {
            "single_unit_households": 30.0,
            "several_unit_households_met_by_qrf": 110.0,
            "several_unit_households_reference_forced": 10.0,
            "reference_units_true_via_qrf": 60.0,
            "reference_units_forced": 10.0,
            "non_reference_units_true_via_qrf": 50.0,
            "non_reference_units_false": 70.0,
        },
    }


def test_a_housing_unit_with_no_reference_unit_to_force_is_refused(
    transferred,
) -> None:
    person = transferred.table("person").copy()
    reference = (person["person_household_id"] == 1) & (person["RELSHIPP"] == 20)
    person.loc[reference, "RELSHIPP"] = 21
    with pytest.raises(ValueError, match="no unit holding the reference person"):
        with_acs_local_snap_receipt_anchor(_with_tables(transferred, person=person))


def test_only_receives_snap_changes(anchored, transferred) -> None:
    frame, _receipt = anchored
    pd.testing.assert_series_equal(
        frame.table("spm_unit")["receives_tanf"],
        transferred.table("spm_unit")["receives_tanf"],
    )
    for entity in ("household", "person", "tax_unit", "family", "marital_unit"):
        pd.testing.assert_frame_equal(frame.table(entity), transferred.table(entity))
    assert frame.weights_for("household").values.tolist() == (
        transferred.weights_for("household").values.tolist()
    )


def test_receipt_counts_the_anchor_and_the_overrides(anchored) -> None:
    _frame, receipt = anchored
    assert receipt["issue"] == ACS_LOCAL_RECEIPT_ANCHOR_ISSUE
    assert receipt["method"] == ACS_LOCAL_RECEIPT_ANCHOR_METHOD
    snap = receipt["snap"]
    assert {
        key: snap[key]
        for key in (
            "acs_households",
            "housing_unit_households",
            "group_quarters_households",
            "fs_yes_households",
            "fs_no_households",
            "group_quarters_fs_coded_ignored",
            "fs_yes_households_with_several_units",
            "acs_spm_units",
            "units_anchored",
            "group_quarters_units",
            "non_reference_units_anchored",
            "several_unit_households_reference_only",
            "several_unit_households_with_a_reporter",
        )
    } == {
        "acs_households": 7,
        "housing_unit_households": 6,
        "group_quarters_households": 1,
        "fs_yes_households": 4,
        "fs_no_households": 2,
        "group_quarters_fs_coded_ignored": 0,
        "fs_yes_households_with_several_units": 3,
        "acs_spm_units": 11,
        "units_anchored": 4,
        "group_quarters_units": 1,
        "non_reference_units_anchored": 1,
        "several_unit_households_reference_only": 2,
        "several_unit_households_with_a_reporter": 3,
    }
    # Unit weights: A 10+10, B 20+20, C 7, D 30, E 40, F 50+50, G 60+60.
    assert snap["transferred"] == {
        "true_units": 5,
        "missing_units": 0,
        "false_to_true": 1,
        "true_to_false": 2,
        "group_quarters_true_to_false": 1,
        "unchanged": 8,
        "weighted": {
            "true_unit_share": pytest.approx(167 / 357),
            "false_to_true_unit_share": pytest.approx(10 / 357),
            "true_to_false_unit_share": pytest.approx(27 / 357),
        },
    }
    weighted = snap["weighted"]
    # Housing units A (10), B (20), D (30), E (40), F (50), G (60); FS == 1
    # in A, D, F and G.
    assert weighted["housing_unit_households"] == 210.0
    assert weighted["fs_yes_households"] == 150.0
    assert weighted["fs_yes_household_share"] == pytest.approx(150 / 210)
    assert weighted["anchored_unit_share"] == pytest.approx(150 / 357)
    reference = snap["household_share_reference"]
    assert reference["value"] == 0.122
    assert reference["band"] == list(ACS_SNAP_HOUSEHOLD_SHARE_BAND)
    assert reference["within_band"] is False
    assert reference["graded"] is False
    assert "S2201" in reference["source"]


def test_receipt_records_pap_against_transferred_tanf(anchored) -> None:
    _frame, receipt = anchored
    tanf = receipt["tanf"]
    assert "not TANF receipt" in tanf["decision"]
    assert "general assistance" in tanf["source"]
    assert {key: value for key, value in tanf.items() if type(value) is int} == {
        "pap_recipients": 1,
        "pap_blank_under_min_age": 2,
        "pap_units": 1,
        "transferred_tanf_units": 3,
        "transferred_tanf_units_with_pap": 0,
        "transferred_tanf_units_without_pap": 3,
        "pap_units_without_transferred_tanf": 1,
        "transferred_tanf_units_not_snap_anchored": 1,
    }
    assert tanf["weighted"]["pap_recipients"] == 20.0
    assert tanf["weighted"]["pap_unit_share"] == pytest.approx(20 / 357)


def test_receipt_carries_the_snap_tanf_crosstab(anchored) -> None:
    """microcosm#1062 review: the override touches receives_snap alone, so the
    receipt shows the joint distribution with receives_tanf both ways."""

    _frame, receipt = anchored
    crosstab = receipt["snap_tanf_crosstab"]
    assert crosstab["informational"] is True
    before, after = crosstab["before_override"], crosstab["after_override"]
    assert before["units"] == {
        "snap_tanf": 2,
        "snap_no_tanf": 3,
        "no_snap_tanf": 1,
        "no_snap_no_tanf": 5,
    }
    assert after["units"] == {
        "snap_tanf": 2,
        "snap_no_tanf": 2,
        "no_snap_tanf": 1,
        "no_snap_no_tanf": 6,
    }
    assert after["weighted_unit_shares"] == {
        "snap_tanf": pytest.approx(80 / 357),
        "snap_no_tanf": pytest.approx(70 / 357),
        "no_snap_tanf": pytest.approx(40 / 357),
        "no_snap_no_tanf": pytest.approx(167 / 357),
    }
    assert before["weighted_unit_shares"]["snap_no_tanf"] == pytest.approx(87 / 357)
    assert before["tanf_share_of_snap_units"] == pytest.approx(80 / 167)
    assert after["tanf_share_of_snap_units"] == pytest.approx(80 / 150)
    assert after["snap_share_of_tanf_units"] == pytest.approx(80 / 120)


def test_the_anchor_is_idempotent(anchored) -> None:
    frame, receipt = anchored
    again, again_receipt = with_acs_local_snap_receipt_anchor(frame)
    assert again.table("spm_unit")["receives_snap"].tolist() == _RULE
    assert again_receipt["snap"]["transferred"]["false_to_true"] == 0
    assert again_receipt["snap"]["transferred"]["true_to_false"] == 0
    assert again_receipt["snap"]["units_anchored"] == receipt["snap"]["units_anchored"]


def test_a_missing_transferred_cell_is_anchored_too(transferred) -> None:
    spm_unit = transferred.table("spm_unit").copy()
    spm_unit["receives_snap"] = spm_unit["receives_snap"].astype(object)
    spm_unit.loc[0, "receives_snap"] = None
    frame, receipt = with_acs_local_snap_receipt_anchor(
        _with_tables(transferred, spm_unit=spm_unit)
    )
    assert frame.table("spm_unit")["receives_snap"].tolist() == _RULE
    assert receipt["snap"]["transferred"]["missing_units"] == 1


def test_a_coded_group_quarters_record_is_ignored(transferred) -> None:
    household = transferred.table("household").copy()
    household.loc[household["TYPEHUGQ"] == 3, "FS"] = 1
    frame, receipt = with_acs_local_snap_receipt_anchor(
        _with_tables(transferred, household=household)
    )
    assert frame.table("spm_unit")["receives_snap"].tolist() == _RULE
    assert receipt["snap"]["group_quarters_fs_coded_ignored"] == 1


# --- refusals ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("fs", "message"),
    [
        (np.nan, "1 ACS housing-unit household"),
        (3, "outside 1 \\(yes\\) / 2 \\(no\\)"),
    ],
    ids=["blank", "unknown-code"],
)
def test_a_housing_unit_without_fs_1_or_2_is_refused(transferred, fs, message):
    household = transferred.table("household").copy()
    household["FS"] = household["FS"].astype(float)
    household.loc[0, "FS"] = fs
    frame = _with_tables(transferred, household=household)
    with pytest.raises(ValueError, match=message):
        with_acs_local_snap_receipt_anchor(frame)
    with pytest.raises(ValueError, match=message):
        require_acs_receipt_anchor_sources(frame)


@pytest.mark.parametrize(
    ("pap", "message"),
    [
        (np.nan, "PAP is blank on 1 person\\(s\\) aged 15 or over"),
        (-5.0, "PAP is negative on 1 row"),
    ],
    ids=["blank-adult", "negative"],
)
def test_an_unreadable_pap_is_refused(transferred, pap, message) -> None:
    person = transferred.table("person").copy()
    person["PAP"] = person["PAP"].astype(float)
    person.loc[person["RELSHIPP"] == 20, "PAP"] = [pap] + [0.0] * 5
    frame = _with_tables(transferred, person=person)
    with pytest.raises(ValueError, match=message):
        with_acs_local_snap_receipt_anchor(frame)
    with pytest.raises(ValueError, match=message):
        require_acs_receipt_anchor_sources(frame)


def test_the_anchor_runs_after_the_transfer_on_acs_rows_only(
    loader_frame, transferred
) -> None:
    # Before the transfer there is no receives_snap to override.
    with pytest.raises(ValueError, match="runs after the transfer"):
        with_acs_local_snap_receipt_anchor(loader_frame)
    # A source without FS fails the early check, before any fit.
    household = loader_frame.table("household").drop(columns=["FS"])
    with pytest.raises(ValueError, match=r"household column\(s\) \['FS'\]"):
        require_acs_receipt_anchor_sources(
            _with_tables(loader_frame, household=household)
        )
    # A pooled frame's donor households carry no TYPEHUGQ/FS: refused, so the
    # stage can never reach an ASEC row.
    with pytest.raises(ValueError, match="have no TYPEHUGQ 1/2/3"):
        with_acs_local_snap_receipt_anchor(_pooled(transferred))


# --- the release gate ----------------------------------------------------------


def _donor() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "person_household_id": [1, 1],
            "person_tax_unit_id": [1, 1],
            "person_spm_unit_id": [1, 1],
            "person_family_id": [1, 1],
            "person_marital_unit_id": [1, 1],
            "age": [42.0, 38.0],
        }
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame({"household_id": [1], "state_fips": [6]}),
            "tax_unit": pd.DataFrame({"tax_unit_id": [1]}),
            # The donor's own ASEC receipt, with no FS behind it.
            "spm_unit": pd.DataFrame(
                {
                    "spm_unit_id": [1],
                    "receives_snap": [True],
                    "receives_tanf": [False],
                }
            ),
            "family": pd.DataFrame({"family_id": [1]}),
            "marital_unit": pd.DataFrame({"marital_unit_id": [1]}),
        },
        US_SCHEMA,
        {"household": Weights(np.asarray([100.0]), WeightKind.DESIGN)},
        pd.Series("asec_puf", index=person.index, dtype=object),
    )


def _pooled(acs: Frame) -> Frame:
    return with_optional_acs_spine(_donor(), acs, acs_share=0.5)


@pytest.fixture(scope="module")
def pooled(anchored) -> Frame:
    return _pooled(anchored[0])


def test_gate_passes_the_anchored_pool_and_leaves_donor_rows_alone(
    pooled, anchored
) -> None:
    gate = acs_local_receipt_anchor_signal_gate(pooled, receipt=anchored[1])
    assert gate.name == ACS_LOCAL_RECEIPT_ANCHOR_GATE_NAME
    assert gate.passed, gate.failures
    spm_unit = pooled.table("spm_unit")
    donor = spm_unit[UNIT_TAG].ne(ACS_2024_1YR_SPINE)
    # The donor unit keeps its ASEC receipt, though no FS stands behind it.
    assert spm_unit.loc[donor, "receives_snap"].tolist() == [True]
    assert gate.details["donor_spm_units"] == 1
    assert gate.details["donor_receives_snap_unit_share"] == 1.0
    assert gate.details["snap"]["units_anchored"] == 4
    assert gate.details["tanf"]["pap_units"] == 1
    # The FS == 1 share (0.71 here) sits outside the reference band; the band
    # is informational and never fails the gate.
    snap = gate.details["snap"]
    assert snap["weighted"]["fs_yes_household_share"] == pytest.approx(150 / 210)
    assert snap["household_share_reference"]["within_band"] is False
    # The joint SNAP x TANF distribution: the ACS units' after the override,
    # beside the donor spine's (its one unit reports SNAP, not TANF).
    crosstab = gate.details["snap_tanf_crosstab"]
    assert crosstab["informational"] is True
    assert crosstab[ACS_2024_1YR_SPINE]["weighted_unit_shares"] == pytest.approx(
        anchored[1]["snap_tanf_crosstab"]["after_override"]["weighted_unit_shares"]
    )
    assert crosstab["donor"]["units"] == {
        "snap_tanf": 0,
        "snap_no_tanf": 1,
        "no_snap_tanf": 0,
        "no_snap_no_tanf": 0,
    }


def test_gate_fails_the_transferred_receipt(anchored, transferred) -> None:
    gate = acs_local_receipt_anchor_signal_gate(
        _pooled(transferred), receipt=anchored[1]
    )
    assert not gate.passed
    assert any(
        "2 SPM unit(s) report SNAP receipt outside an FS == 1 housing unit" in f
        for f in gate.failures
    ), gate.failures
    assert any(
        "1 FS == 1 housing unit(s) with several SPM units have no unit reporting" in f
        for f in gate.failures
    ), gate.failures


def test_gate_fails_the_pre_review_every_unit_anchor(pooled, anchored) -> None:
    """Marking every unit of an FS == 1 housing unit (the pre-review rule)
    meets the at-least-one constraint, but the receipt does not reconcile:
    the frame's roommate reporters are not the ones the QRF named."""

    spm_unit = pooled.table("spm_unit").copy()
    acs = spm_unit[UNIT_TAG].eq(ACS_2024_1YR_SPINE)
    every = [True, True, False, False, False, True, False, True, True, True, True]
    spm_unit.loc[acs, "receives_snap"] = every
    gate = acs_local_receipt_anchor_signal_gate(
        _with_spm_unit(pooled, spm_unit), receipt=anchored[1]
    )
    assert not gate.passed
    assert any(
        "snap.units_anchored = 4; the frame holds 7" in f for f in gate.failures
    ), gate.failures
    assert any(
        "snap.non_reference_units_anchored = 1; the frame holds 3" in f
        for f in gate.failures
    ), gate.failures


def test_gate_fails_a_single_unit_fs_1_household_without_receipt(
    pooled, anchored
) -> None:
    spm_unit = pooled.table("spm_unit").copy()
    acs = spm_unit[UNIT_TAG].eq(ACS_2024_1YR_SPINE)
    # D's family is the only SPM unit of its FS == 1 housing unit.
    spm_unit.loc[spm_unit.index[acs][5], "receives_snap"] = False
    gate = acs_local_receipt_anchor_signal_gate(
        _with_spm_unit(pooled, spm_unit), receipt=anchored[1]
    )
    assert any(
        "1 SPM unit(s) alone in an FS == 1 housing unit do not report" in f
        for f in gate.failures
    ), gate.failures


def _with_spm_unit(frame: Frame, spm_unit: pd.DataFrame) -> Frame:
    return Frame(
        {**{e: frame.table(e) for e in frame.entities}, "spm_unit": spm_unit},
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
    )


def test_gate_fails_a_missing_acs_cell(pooled, anchored) -> None:
    spm_unit = pooled.table("spm_unit").copy()
    spm_unit["receives_snap"] = spm_unit["receives_snap"].astype(object)
    acs = spm_unit[UNIT_TAG].eq(ACS_2024_1YR_SPINE)
    spm_unit.loc[spm_unit.index[acs][0], "receives_snap"] = None
    gate = acs_local_receipt_anchor_signal_gate(
        _with_spm_unit(pooled, spm_unit), receipt=anchored[1]
    )
    assert not gate.passed
    assert any("missing on 1 SPM unit" in f for f in gate.failures), gate.failures


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (None, "no ACS receipt-anchor receipt"),
        ({"issue": "microcosm#1019"}, "not microcosm#1022's"),
        ({"method": "reference_unit_only"}, "records method 'reference_unit_only'"),
        ({"snap": None}, "no snap/tanf counts"),
        ({"snap": {"units_anchored": 5}}, "snap.units_anchored = 5"),
        (
            {"snap": {"non_reference_units_anchored": 3}},
            "snap.non_reference_units_anchored = 3",
        ),
        ({"snap": {"resolution": None}}, "does not count how each FS == 1"),
        (
            {"snap": {"resolution": {"several_unit_households_reference_forced": 2}}},
            "resolution does not reconcile",
        ),
        (
            {"snap": {"resolution": {"non_reference_units_true_via_qrf": 3}}},
            "3 non-reference unit(s) True through the QRF",
        ),
        (
            {"snap": {"resolution": {"single_unit_households": 2}}},
            "counts 2 single-unit FS == 1",
        ),
        ({"snap": {"fs_yes_households": "2"}}, "fs_yes_households is not an integer"),
        ({"tanf": {"pap_recipients": 2}}, "tanf.pap_recipients = 2"),
        ({"snap": {"transferred": None}}, "does not count the transferred"),
        (
            {"snap": {"transferred": {"false_to_true": 3}}},
            "overrides do not reconcile",
        ),
    ],
    ids=[
        "missing",
        "wrong-issue",
        "wrong-method",
        "no-counts",
        "stale-anchors",
        "stale-non-reference",
        "no-resolution",
        "unreconciled-households",
        "unreconciled-non-reference",
        "unreconciled-single",
        "untyped",
        "stale-pap",
        "no-overrides",
        "unreconciled",
    ],
)
def test_gate_grades_the_staging_receipt(pooled, anchored, override, message) -> None:
    receipt = anchored[1]
    if override is None:
        graded = None
    else:
        graded = {**receipt}
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(receipt.get(key), dict):
                section = {**receipt[key]}
                for inner, inner_value in value.items():
                    if isinstance(inner_value, dict):
                        section[inner] = {**section[inner], **inner_value}
                    else:
                        section[inner] = inner_value
                graded[key] = section
            else:
                graded[key] = value
    gate = acs_local_receipt_anchor_signal_gate(pooled, receipt=graded)
    assert not gate.passed
    assert any(message in failure for failure in gate.failures), gate.failures


def test_gate_requires_origin_tags_and_readable_sources(pooled, anchored) -> None:
    spm_unit = pooled.table("spm_unit").drop(columns=[UNIT_TAG])
    gate = acs_local_receipt_anchor_signal_gate(
        _with_spm_unit(pooled, spm_unit), receipt=anchored[1]
    )
    assert gate.failures == (f"Missing spm_unit origin tags: {UNIT_TAG}.",)

    household = pooled.table("household").copy()
    acs = household[HOUSEHOLD_TAG].eq(ACS_2024_1YR_SPINE)
    household.loc[household.index[acs][0], "FS"] = np.nan
    frame = Frame(
        {**{e: pooled.table(e) for e in pooled.entities}, "household": household},
        pooled.schema,
        {entity: pooled.weights_for(entity) for entity in pooled.weighted_entities},
        pooled.strata,
    )
    gate = acs_local_receipt_anchor_signal_gate(frame, receipt=anchored[1])
    assert not gate.passed
    assert "1 ACS housing-unit household(s) carry FS" in gate.failures[0]

    spm_unit = pooled.table("spm_unit").drop(columns=["receives_snap"])
    gate = acs_local_receipt_anchor_signal_gate(
        _with_spm_unit(pooled, spm_unit), receipt=anchored[1]
    )
    assert "['receives_snap'] are absent" in gate.failures[0]
