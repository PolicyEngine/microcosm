"""ACS adult-nonrelative SPM units in the ACS local lane (microcosm#1023)."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.acs_inputs import map_acs_native_inputs
from microcosm.build.us_runtime.acs_local_spm_units import (
    ACS_LOCAL_SPM_UNIT_GATE_NAME,
    ACS_LOCAL_SPM_UNIT_ISSUE,
    ACS_LOCAL_SPM_UNIT_METHOD,
    acs_adult_nonrelative_mask,
    acs_local_spm_unit_signal_gate,
    acs_spm_unit_mover_mask,
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

TAG = spine_column("person")


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
    }
    row.update(overrides)
    return row


#: Six ACS records (sorted SERIALNO order = household ids 1-6):
#: A  reference, roommate (25), other nonrelative (60), nonrelative child (8)
#: B  reference, unmarried partner, nonrelative child (6), foster child (10)
#: C  one noninstitutional group-quarters person
#: D  married reference couple and a 19-year-old roommate with no income
#: E  reference, same-sex unmarried partner, and two other nonrelatives (16
#:    and 45): the review's case, possibly the partner's child and relative
#: F  reference and an other nonrelative (50), no partner
_HOUSEHOLDS = [
    _household("A", 4, WGTP=10),
    _household("B", 4, WGTP=20),
    _household("C", 1, WGTP=0, TYPEHUGQ=3, TEN=None, RNTP=None, GRNTP=None),
    _household("D", 3, WGTP=30),
    _household("E", 4, WGTP=40),
    _household("F", 2, WGTP=50),
]
_PERSONS = [
    _person("A", 1, 20),
    _person("A", 2, 34, AGEP=25),
    _person("A", 3, 36, AGEP=60, WAGP=0),
    _person("A", 4, 36, AGEP=8, WAGP=0),
    _person("B", 1, 20),
    _person("B", 2, 22, AGEP=35),
    _person("B", 3, 36, AGEP=6, WAGP=0),
    _person("B", 4, 35, AGEP=10, WAGP=0),
    _person("C", 1, 38, PWGTP=7),
    _person("D", 1, 20, MAR=1),
    _person("D", 2, 21, MAR=1, SEX=2),
    _person("D", 3, 34, AGEP=19, WAGP=0),
    _person("E", 1, 20),
    _person("E", 2, 24, AGEP=38),
    _person("E", 3, 36, AGEP=16, WAGP=0),
    _person("E", 4, 36, AGEP=45),
    _person("F", 1, 20),
    _person("F", 2, 36, AGEP=50),
]


def _source(tmp_path: Path, households=None, persons=None) -> AcsPumsSource:
    household_zip = tmp_path / "csv_hus.zip"
    person_zip = tmp_path / "csv_pus.zip"
    _write_csv_zip(household_zip, "psam_husa.csv", households or _HOUSEHOLDS)
    _write_csv_zip(person_zip, "psam_pusa.csv", persons or _PERSONS)
    return AcsPumsSource(household_zip=household_zip, person_zip=person_zip)


@pytest.fixture(scope="module")
def loader_frame(tmp_path_factory) -> Frame:
    frame, _metadata = build_acs_pums_unit_frame(
        _source(tmp_path_factory.mktemp("acs"))
    )
    return frame


@pytest.fixture(scope="module")
def split(loader_frame) -> tuple[Frame, dict]:
    return split_acs_adult_nonrelative_spm_units(loader_frame)


def _members(frame: Frame, entity: str) -> list[list[tuple[int, int]]]:
    """Each unit's members as sorted (household id, SPORDER) pairs."""

    person = frame.table("person")
    keys = list(
        zip(
            person["person_household_id"].astype(int),
            person["SPORDER"].astype(int),
            strict=True,
        )
    )
    grouped: dict[int, list[tuple[int, int]]] = {}
    for key, unit in zip(keys, person[f"person_{entity}_id"], strict=True):
        grouped.setdefault(int(unit), []).append(key)
    return sorted(sorted(members) for members in grouped.values())


# --- the loader (shared with the pool lane) is unchanged ---------------------


def test_loader_keeps_one_spm_unit_per_household(loader_frame) -> None:
    """The pool lane calls the loader directly: its SPM units stay whole
    households, roommates included, and never reach the split."""

    person = loader_frame.table("person")
    assert person["person_spm_unit_id"].tolist() == (
        person["person_household_id"].tolist()
    )
    assert loader_frame.n("spm_unit") == loader_frame.n("household") == 6


# --- the rule ----------------------------------------------------------------


def test_roommates_and_unpartnered_other_nonrelatives_get_their_own_unit(
    split,
) -> None:
    frame, _receipt = split
    assert _members(frame, "spm_unit") == [
        # Household A: the reference person keeps the nonrelative child (8).
        [(1, 1), (1, 4)],
        [(1, 2)],  # roommate, 25
        [(1, 3)],  # other nonrelative, 60, no partner in the household
        # Household B: partner, the nonrelative child and the foster child
        # all stay with the reference person.
        [(2, 1), (2, 2), (2, 3), (2, 4)],
        [(3, 1)],  # group quarters: already one person
        [(4, 1), (4, 2)],  # the married reference couple
        [(4, 3)],  # the 19-year-old roommate, even with no income
        # Household E: with an unmarried partner present, the 16-year-old
        # and the 45-year-old other nonrelatives stay (the review's case).
        [(5, 1), (5, 2), (5, 3), (5, 4)],
        [(6, 1)],  # household F's reference person
        [(6, 2)],  # other nonrelative, 50, no partner in the household
    ]


def test_partners_child_coded_36_stays_with_the_partner(split) -> None:
    """microcosm#1061 review: a reference person, an unmarried partner and a
    16-year-old coded 36 (possibly the partner's child) share one unit."""

    frame, _receipt = split
    person = frame.table("person")
    household_e = person[person["person_household_id"] == 5]
    assert household_e["RELSHIPP"].tolist() == [20, 24, 36, 36]
    assert household_e["person_spm_unit_id"].nunique() == 1
    child = household_e[household_e["AGEP"] == 16]
    partner = household_e[household_e["RELSHIPP"] == 24]
    assert child["person_spm_unit_id"].iloc[0] == partner["person_spm_unit_id"].iloc[0]


def test_mover_mask_keeps_other_nonrelatives_with_a_partner() -> None:
    person = pd.DataFrame(
        {
            "person_household_id": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5],
            "RELSHIPP": [20, 22, 36, 20, 24, 36, 20, 36, 20, 34, 36, 20, 22, 34],
            "AGEP": [40, 38, 16, 40, 38, 45, 40, 45, 40, 30, 17, 40, 38, 30],
        }
    )
    assert acs_spm_unit_mover_mask(person).tolist() == [
        # A partner's 16-year-old coded 36 stays (the review's case).
        False,
        False,
        False,
        # An adult coded 36 with a same-sex partner present stays.
        False,
        False,
        False,
        # An adult coded 36 with no partner present moves.
        False,
        True,
        # A roommate moves; a 17-year-old 36 beside them moves too (no
        # partner), though they may be the roommate's child: counted.
        False,
        True,
        True,
        # A roommate moves even when the household has a partner.
        False,
        False,
        True,
    ]
    with pytest.raises(ValueError, match="'person_household_id' is absent"):
        acs_spm_unit_mover_mask(person.drop(columns=["person_household_id"]))


def test_mask_reads_relationship_and_age() -> None:
    person = pd.DataFrame(
        {
            "RELSHIPP": [20, 34, 34, 36, 36, 35, 22, 38],
            "AGEP": [40, 15, 14, 90, 3, 30, 30, 40],
        }
    )
    assert acs_adult_nonrelative_mask(person).tolist() == [
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    with pytest.raises(ValueError, match="'AGEP' is blank on 1 row"):
        acs_adult_nonrelative_mask(person.assign(AGEP=[40, None, 1, 1, 1, 1, 1, 1]))
    with pytest.raises(ValueError, match="'RELSHIPP' is absent"):
        acs_adult_nonrelative_mask(person.drop(columns=["RELSHIPP"]))


def test_tax_units_nest_and_other_units_are_unchanged(loader_frame, split) -> None:
    frame, _receipt = split
    person = frame.table("person")
    # Every tax unit sits inside one SPM unit; a moved person files alone.
    assert (
        person.groupby("person_tax_unit_id")["person_spm_unit_id"].nunique() == 1
    ).all()
    moved = acs_spm_unit_mover_mask(person)
    tax_sizes = person.groupby("person_tax_unit_id").size()
    assert (tax_sizes.loc[person.loc[moved, "person_tax_unit_id"]] == 1).all()
    before = loader_frame.table("person")
    for column in (
        "person_household_id",
        "person_tax_unit_id",
        "person_family_id",
        "person_marital_unit_id",
    ):
        assert person[column].tolist() == before[column].tolist()
    for entity in ("household", "tax_unit", "family", "marital_unit"):
        pd.testing.assert_frame_equal(frame.table(entity), loader_frame.table(entity))
    assert frame.weights_for("household").values.tolist() == (
        loader_frame.weights_for("household").values.tolist()
    )


def test_ids_are_dense_sorted_and_deterministic(loader_frame, split) -> None:
    frame, receipt = split
    ids = frame.table("spm_unit")["spm_unit_id"].tolist()
    assert ids == list(range(1, 11))
    again, again_receipt = split_acs_adult_nonrelative_spm_units(loader_frame)
    assert again.table("person")["person_spm_unit_id"].tolist() == (
        frame.table("person")["person_spm_unit_id"].tolist()
    )
    assert again_receipt == receipt


def test_a_household_without_nonrelatives_keeps_its_ids(tmp_path) -> None:
    households = [_household("B", 2), _household("C", 1, WGTP=0, TYPEHUGQ=3)]
    persons = [_person("B", 1, 20), _person("B", 2, 25, AGEP=9), _person("C", 1, 37)]
    loader, _ = build_acs_pums_unit_frame(_source(tmp_path, households, persons))
    frame, receipt = split_acs_adult_nonrelative_spm_units(loader)
    pd.testing.assert_frame_equal(frame.table("person"), loader.table("person"))
    pd.testing.assert_frame_equal(frame.table("spm_unit"), loader.table("spm_unit"))
    assert receipt["persons_moved"] == receipt["units_created"] == 0


def test_receipt_counts_the_split(split) -> None:
    _frame, receipt = split
    assert receipt["issue"] == ACS_LOCAL_SPM_UNIT_ISSUE
    assert receipt["method"] == ACS_LOCAL_SPM_UNIT_METHOD
    assert receipt["moved_relationship_codes"] == [34, 36]
    assert receipt["partner_relationship_codes"] == [22, 24]
    assert receipt["min_age"] == 15
    assert receipt["acs_households"] == 6
    assert receipt["households_affected"] == 3
    assert receipt["persons_moved"] == receipt["units_created"] == 4
    assert receipt["persons_moved_by_relationship"] == {"34": 2, "36": 2}
    assert receipt["nonrelatives_under_min_age_kept"] == 2
    assert (receipt["spm_units_before"], receipt["spm_units_after"]) == (6, 10)
    weighted = receipt["weighted"]
    # Households A (10), D (30) and F (50); movers weigh 10 + 10 + 30 + 50.
    assert weighted["households_affected"] == 90.0
    assert weighted["persons_moved"] == 100.0
    # Persons: 4x10 + 4x20 + 1x7 + 3x30 + 4x40 + 2x50 = 477 over 157 household
    # weight before, and 257 unit weight after.
    assert weighted["persons_per_spm_unit"]["before"] == pytest.approx(477 / 157)
    assert weighted["persons_per_spm_unit"]["after"] == pytest.approx(477 / 257)
    assert weighted["single_person_spm_unit_share"]["before"] == pytest.approx(7 / 157)
    # Single-person units: A's two movers (10 each), C (7), D's roommate (30),
    # F's reference person and mover (50 each).
    assert weighted["single_person_spm_unit_share"]["after"] == pytest.approx(157 / 257)
    assert weighted["spm_units_per_household"] == {
        "before": 1.0,
        "after": pytest.approx(257 / 157),
    }


def test_receipt_sizes_the_code_36_ambiguity(split) -> None:
    """E's two 36s stay for the partner; A's and F's adult 36s move, and A's
    shares its household with a roommate, so it may be the roommate's child."""

    _frame, receipt = split
    other = receipt["other_nonrelatives"]
    assert other["kept_with_partner"] == 2
    assert other["moved"] == 2
    assert other["moved_from_roommate_households"] == 1
    assert other["weighted"] == {
        "kept_with_partner": 80.0,
        "moved": 60.0,
        "moved_from_roommate_households": 10.0,
    }
    assert "roommate's own child" in other["ambiguity"]
    sensitivity = receipt["sensitivity"]
    assert sensitivity["rule"] == {
        "persons_per_spm_unit": pytest.approx(477 / 257),
        "single_person_spm_unit_share": pytest.approx(157 / 257),
        "spm_units_per_household": pytest.approx(257 / 157),
    }
    # 34 only: A and D lose their roommates; every 36 stays.
    assert sensitivity["roommates_only"] == {
        "persons_per_spm_unit": pytest.approx(477 / 197),
        "single_person_spm_unit_share": pytest.approx(47 / 197),
        "spm_units_per_household": pytest.approx(197 / 157),
    }
    # 34 and every 36 (the pre-review rule): E's 16- and 45-year-old move too.
    assert sensitivity["roommates_and_every_other_nonrelative"] == {
        "persons_per_spm_unit": pytest.approx(477 / 337),
        "single_person_spm_unit_share": pytest.approx(237 / 337),
        "spm_units_per_household": pytest.approx(337 / 157),
    }


# --- downstream SPM-unit inputs ----------------------------------------------


def test_new_units_take_the_household_tenure(split) -> None:
    """The split lands before the native mapping, so tenure maps through the
    new SPM membership; measured housing amounts stay on the household."""

    frame, _receipt = split
    mapped = map_acs_native_inputs(frame).frame
    tenure = mapped.table("spm_unit").set_index("spm_unit_id")["spm_unit_tenure_type"]
    person = mapped.table("person")
    household_of_unit = person.groupby("person_spm_unit_id")[
        "person_household_id"
    ].first()
    group_quarters = household_of_unit.index[household_of_unit == 3]
    assert tenure.drop(group_quarters).eq("RENTER").all()
    assert tenure.loc[group_quarters].isna().all()


# --- refusals ----------------------------------------------------------------


def _with_tables(frame: Frame, **tables: pd.DataFrame) -> Frame:
    replaced = {entity: frame.table(entity) for entity in frame.entities}
    replaced.update(tables)
    return Frame(
        replaced,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
    )


def test_split_runs_only_on_the_loader_frame(loader_frame, split) -> None:
    with pytest.raises(ValueError, match="already split"):
        split_acs_adult_nonrelative_spm_units(split[0])
    mapped = _with_tables(
        loader_frame,
        spm_unit=loader_frame.table("spm_unit").assign(receives_snap=False),
    )
    with pytest.raises(ValueError, match=r"carries \['receives_snap'\]"):
        split_acs_adult_nonrelative_spm_units(mapped)
    person = loader_frame.table("person").drop(columns=["RELSHIPP"])
    with pytest.raises(ValueError, match=r"requires person column\(s\) \['RELSHIPP'\]"):
        split_acs_adult_nonrelative_spm_units(_with_tables(loader_frame, person=person))


def test_split_refuses_a_tax_unit_it_would_straddle(loader_frame) -> None:
    person = loader_frame.table("person").copy()
    roommate = (person["person_household_id"] == 1) & (person["SPORDER"] == 2)
    reference_tax_unit = person.loc[
        (person["person_household_id"] == 1) & (person["SPORDER"] == 1),
        "person_tax_unit_id",
    ].iloc[0]
    person.loc[roommate, "person_tax_unit_id"] = reference_tax_unit
    tax_unit = pd.DataFrame(
        {"tax_unit_id": np.sort(person["person_tax_unit_id"].unique())}
    )
    frame = _with_tables(loader_frame, person=person, tax_unit=tax_unit)
    with pytest.raises(ValueError, match="1 ACS tax unit.*span two SPM units"):
        split_acs_adult_nonrelative_spm_units(frame)


# --- the release gate ---------------------------------------------------------


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
            "spm_unit": pd.DataFrame({"spm_unit_id": [1]}),
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
def pooled(split) -> Frame:
    return _pooled(split[0])


def test_gate_passes_the_split_pool(pooled, split) -> None:
    gate = acs_local_spm_unit_signal_gate(pooled, receipt=split[1])
    assert gate.name == ACS_LOCAL_SPM_UNIT_GATE_NAME
    assert gate.passed, gate.failures
    assert gate.details["acs_person_rows"] == 18
    assert gate.details["adult_nonrelatives"] == 6
    assert gate.details["movers"] == 4
    assert gate.details["other_nonrelatives_kept_with_partner"] == 2
    assert gate.details["other_nonrelatives_moved"] == 2
    assert gate.details["other_nonrelatives_moved_from_roommate_households"] == 1
    assert gate.details["acs_spm_units"] == 10
    assert gate.details["acs_households"] == 6
    assert gate.details["weighted_persons_per_spm_unit"] == pytest.approx(477 / 257)
    assert gate.details["weighted_spm_units_per_household"] == pytest.approx(257 / 157)


def test_gate_fails_the_unsplit_loader_partition(loader_frame, split) -> None:
    gate = acs_local_spm_unit_signal_gate(_pooled(loader_frame), receipt=split[1])
    assert not gate.passed
    assert any(
        "4 adult roommate(s) or unpartnered other nonrelative(s)" in failure
        for failure in gate.failures
    ), gate.failures
    assert any("not one unit per mover" in f for f in gate.failures)


def test_gate_fails_a_partners_child_split_from_the_partner(pooled, split) -> None:
    """The pre-review rule's partition (every 36 aged 15+ alone) no longer
    passes: the gate reads the partner clause itself."""

    person = pooled.table("person").copy()
    acs = person[TAG].eq(ACS_2024_1YR_SPINE)
    household = person.loc[acs & (person["RELSHIPP"] == 24), "person_household_id"]
    others = (
        acs
        & (person["RELSHIPP"] == 36)
        & (person["AGEP"] >= 15)
        & person["person_household_id"].isin(household)
    )
    assert int(others.sum()) == 2
    # The pre-review rule gave each of them a unit of their own.
    fresh = person["person_spm_unit_id"].max() + np.arange(1, 3)
    person.loc[others, "person_spm_unit_id"] = fresh
    spm_units = pd.DataFrame(
        {"spm_unit_id": np.sort(person["person_spm_unit_id"].unique())}
    )
    frame = Frame(
        {
            **{e: pooled.table(e) for e in pooled.entities},
            "person": person,
            "spm_unit": spm_units,
        },
        pooled.schema,
        {entity: pooled.weights_for(entity) for entity in pooled.weighted_entities},
        pooled.strata,
    )
    gate = acs_local_spm_unit_signal_gate(frame, receipt=split[1])
    assert not gate.passed
    assert any(
        "2 other nonrelative(s) (RELSHIPP 36, 15 or over) in a household with an "
        "unmarried partner" in failure
        for failure in gate.failures
    ), gate.failures
    assert any("not one unit per mover" in f for f in gate.failures)


@pytest.mark.parametrize(
    ("receipt", "message"),
    [
        (None, "no ACS SPM-unit receipt"),
        ({"issue": "microcosm#1022"}, "not microcosm#1023's"),
        ({"method": "household"}, "records method 'household'"),
        ({"persons_moved": 2, "units_created": 2}, "moved 2 person"),
        ({"units_created": "3"}, "counts are not integers"),
        ({"other_nonrelatives": {"kept_with_partner": 0}}, "keeps 0 other"),
        ({"other_nonrelatives": None}, "keeps None other"),
    ],
    ids=[
        "missing",
        "wrong-issue",
        "wrong-method",
        "stale-count",
        "untyped",
        "stale-kept",
        "pre-review",
    ],
)
def test_gate_grades_the_staging_receipt(pooled, split, receipt, message) -> None:
    graded = None if receipt is None else {**split[1], **receipt}
    gate = acs_local_spm_unit_signal_gate(pooled, receipt=graded)
    assert not gate.passed
    assert any(message in failure for failure in gate.failures), gate.failures


def _with_person(frame: Frame, person: pd.DataFrame) -> Frame:
    return Frame(
        {**{e: frame.table(e) for e in frame.entities}, "person": person},
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
    )


def test_gate_fails_a_straddling_tax_unit_or_a_shared_donor_unit(pooled, split) -> None:
    person = pooled.table("person")
    acs = person[TAG].eq(ACS_2024_1YR_SPINE)
    roommate = acs & (person["RELSHIPP"] == 34) & (person["AGEP"] == 25)
    household = person.loc[roommate, "person_household_id"].iloc[0]
    reference = (
        acs & (person["RELSHIPP"] == 20) & (person["person_household_id"] == household)
    )
    straddled = person.copy()
    straddled.loc[roommate, "person_tax_unit_id"] = straddled.loc[
        reference, "person_tax_unit_id"
    ].iloc[0]
    tax_units = pd.DataFrame(
        {"tax_unit_id": np.sort(straddled["person_tax_unit_id"].unique())}
    )
    frame = Frame(
        {
            **{e: pooled.table(e) for e in pooled.entities},
            "person": straddled,
            "tax_unit": tax_units,
        },
        pooled.schema,
        {entity: pooled.weights_for(entity) for entity in pooled.weighted_entities},
        pooled.strata,
    )
    gate = acs_local_spm_unit_signal_gate(frame, receipt=split[1])
    assert any("tax unit(s) span more than one SPM unit" in f for f in gate.failures)

    shared = person.copy()
    donor_unit = shared.loc[~acs, "person_spm_unit_id"].iloc[0]
    shared.loc[roommate, "person_spm_unit_id"] = donor_unit
    spm_units = pd.DataFrame(
        {"spm_unit_id": np.sort(shared["person_spm_unit_id"].unique())}
    )
    frame = Frame(
        {
            **{e: pooled.table(e) for e in pooled.entities},
            "person": shared,
            "spm_unit": spm_units,
        },
        pooled.schema,
        {entity: pooled.weights_for(entity) for entity in pooled.weighted_entities},
        pooled.strata,
    )
    gate = acs_local_spm_unit_signal_gate(frame, receipt=split[1])
    assert any(
        "share an SPM unit with a row from another spine" in f for f in gate.failures
    )


def test_gate_requires_origin_tags_and_acs_relationships(pooled, split) -> None:
    person = pooled.table("person")
    gate = acs_local_spm_unit_signal_gate(
        _with_person(pooled, person.drop(columns=[TAG])), receipt=split[1]
    )
    assert gate.failures == (f"Missing person origin tags: {TAG}.",)
    blank = person.copy()
    blank.loc[blank[TAG].eq(ACS_2024_1YR_SPINE).idxmax(), "AGEP"] = np.nan
    gate = acs_local_spm_unit_signal_gate(_with_person(pooled, blank), receipt=split[1])
    assert not gate.passed
    assert "'AGEP' is blank on 1 row" in gate.failures[0]
