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


#: Five ACS records (sorted SERIALNO order = household ids 1-5):
#: A  FS 1: reference, a 30-year-old roommate (own unit), a nonrelative child
#: B  FS 2: reference with public assistance income, a 50-year-old roommate
#: C  noninstitutional group quarters, FS blank
#: D  FS 1: a married couple and their child (one unit)
#: E  FS 2: a reference person alone
_HOUSEHOLDS = [
    _household("A", 3, WGTP=10, FS=1),
    _household("B", 2, WGTP=20, FS=2),
    _household("C", 1, WGTP=0, TYPEHUGQ=3, TEN=None, RNTP=None, GRNTP=None, FS=None),
    _household("D", 3, WGTP=30, FS=1),
    _household("E", 1, WGTP=40, FS=2),
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
]
#: Split units, in id order: A reference (+ child), A roommate, B reference,
#: B roommate, C, D family, E.
_RULE = [True, True, False, False, False, True, False]
#: What the QRF transfer is made to have written.
_TRANSFERRED_SNAP = [False, False, True, False, True, True, False]
_TRANSFERRED_TANF = [False, False, False, False, False, True, True]


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


def test_fs_marks_every_unit_of_the_household(anchored, transferred) -> None:
    frame, _receipt = anchored
    assert frame.table("spm_unit")["receives_snap"].tolist() == _RULE
    assert frame.table("spm_unit")["receives_snap"].dtype == np.dtype(bool)
    # FS == 1 reaches the roommate's own unit (A); FS == 2 clears the
    # transferred True on B's reference unit; group quarters are False even
    # where the transfer said True.
    assert frame.n("spm_unit") == 7


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
        )
    } == {
        "acs_households": 5,
        "housing_unit_households": 4,
        "group_quarters_households": 1,
        "fs_yes_households": 2,
        "fs_no_households": 2,
        "group_quarters_fs_coded_ignored": 0,
        "fs_yes_households_with_several_units": 1,
        "acs_spm_units": 7,
        "units_anchored": 3,
        "group_quarters_units": 1,
        "non_reference_units_anchored": 1,
    }
    assert snap["transferred"] == {
        "true_units": 3,
        "missing_units": 0,
        "false_to_true": 2,
        "true_to_false": 2,
        "group_quarters_true_to_false": 1,
        "unchanged": 3,
        "weighted": {
            "true_unit_share": pytest.approx(57 / 137),
            "false_to_true_unit_share": pytest.approx(20 / 137),
            "true_to_false_unit_share": pytest.approx(27 / 137),
        },
    }
    weighted = snap["weighted"]
    # Housing units A (10), B (20), D (30), E (40); FS == 1 in A and D.
    assert weighted["housing_unit_households"] == 100.0
    assert weighted["fs_yes_households"] == 40.0
    assert weighted["fs_yes_household_share"] == pytest.approx(0.4)
    assert weighted["anchored_unit_share"] == pytest.approx(50 / 137)
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
        "transferred_tanf_units": 2,
        "transferred_tanf_units_with_pap": 0,
        "transferred_tanf_units_without_pap": 2,
        "pap_units_without_transferred_tanf": 1,
        "transferred_tanf_units_not_snap_anchored": 1,
    }
    assert tanf["weighted"]["pap_recipients"] == 20.0
    assert tanf["weighted"]["pap_unit_share"] == pytest.approx(20 / 137)


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
    person.loc[person["RELSHIPP"] == 20, "PAP"] = [pap, 0.0, 0.0, 0.0]
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
    assert gate.details["snap"]["units_anchored"] == 3
    assert gate.details["tanf"]["pap_units"] == 1
    # The FS == 1 share (0.4 here) sits outside the reference band; the band
    # is informational and never fails the gate.
    snap = gate.details["snap"]
    assert snap["weighted"]["fs_yes_household_share"] == pytest.approx(0.4)
    assert snap["household_share_reference"]["within_band"] is False


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
        "2 SPM unit(s) of an FS == 1 housing unit do not report" in f
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
        ({"snap": {"units_anchored": 4}}, "snap.units_anchored = 4"),
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
