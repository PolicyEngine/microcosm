"""ACS vehicle availability, the owned vehicle count grade and Head Start
take-up for the ACS rows of the retained ACS local lane (microcosm#1022).

The helpers are tag-free: these tests hand them spine masks directly. The
take-up stage that wires them in is covered in test_us_acs_local_take_up.py.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.acs_local_vehicles_head_start import (
    HEAD_START_AGES,
    US_HEAD_START_TAKE_UP_COLUMN,
    US_VEHICLES_OWNED_COLUMN,
    US_VEHICLES_VALUE_COLUMN,
    donor_head_start_share,
    grade_head_start,
    grade_vehicles_owned,
    head_start_fill,
    vehicles_available_summary,
)
from microcosm.build.us_runtime.sipp_head_start import (
    US_SIPP_HEAD_START_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.sipp_vehicles import US_SIPP_VEHICLE_OUTPUT_COLUMNS

OWNED = US_VEHICLES_OWNED_COLUMN
HEAD_START = US_HEAD_START_TAKE_UP_COLUMN


def test_columns_are_the_donor_stages_outputs() -> None:
    assert (OWNED, US_VEHICLES_VALUE_COLUMN) == US_SIPP_VEHICLE_OUTPUT_COLUMNS
    assert (HEAD_START,) == US_SIPP_HEAD_START_OUTPUT_COLUMNS
    assert HEAD_START_AGES == (3, 5)


# ---------------------------------------------------------------------------
# Vehicle availability and the owned count
# ---------------------------------------------------------------------------


def _households(n_donor: int = 20, n_acs: int = 80) -> tuple[pd.DataFrame, np.ndarray]:
    """Donor households carry a count and no VEH; ACS ones VEH and no count.

    Every fifth ACS household is group quarters (TYPEHUGQ 2 or 3) with a
    blank VEH; ACS housing units cycle through VEH 0-6.
    """

    n = n_donor + n_acs
    acs = np.arange(n) >= n_donor
    position = np.arange(n) - n_donor
    group_quarters = acs & (position % 5 == 4)
    kinds = np.where(
        acs, np.where(group_quarters, np.where(position % 10 == 4, 2, 3), 1), np.nan
    )
    veh = np.where(acs & ~group_quarters, position % 7, np.nan)
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, n + 1),
            "TYPEHUGQ": kinds.astype(float),
            "VEH": veh.astype(float),
            OWNED: pd.Series(np.arange(n) % 4, dtype=float).where(~acs, np.nan),
        }
    )
    return household, acs


def _summary(household: pd.DataFrame, rows: np.ndarray) -> dict[str, object]:
    return vehicles_available_summary(household, rows, np.ones(len(household)))


def test_availability_counts_veh_codes_in_housing_units_only() -> None:
    household, acs = _households()
    entry = _summary(household, acs)
    kinds = household["TYPEHUGQ"].to_numpy()
    housing_units = acs & (kinds == 1)
    group_quarters = acs & np.isin(kinds, (2, 3))
    veh = household["VEH"].to_numpy()
    assert entry["households"] == int(acs.sum())
    assert entry["housing_units"] == int(housing_units.sum())
    assert entry["group_quarters"] == int(group_quarters.sum())
    assert entry["unknown_kind_rows"] == 0
    assert entry["group_quarters_coded_rows"] == 0
    assert entry["housing_units_without_veh"] == 0
    assert entry["top_coded_rows"] == int((veh[housing_units] == 6).sum())
    assert entry["code_counts"] == {
        str(code): int((veh[housing_units] == code).sum()) for code in range(7)
    }
    assert entry["weighted_housing_unit_share_with_vehicle_available"] == pytest.approx(
        float((veh[housing_units] > 0).mean())
    )
    assert entry["reference_owned_share_band"] == [0.40, 0.99]
    assert entry["within_reference_band"] is True
    assert "leased and employer-provided" in entry["source"]
    assert "TVEH_NUM" in entry["owned_definition"]
    assert "written to no engine input" in entry["treatment"]
    assert "microcosm#1064 review" in entry["treatment"]


def test_two_vehicles_available_are_recorded_and_no_count_is_written() -> None:
    """microcosm#1064 review: VEH 2 (say, two leased cars) is two vehicles
    available. The summary records it and leaves the owned count alone."""

    household, acs = _households()
    housing_units = acs & (household["TYPEHUGQ"] == 1).to_numpy()
    household["VEH"] = household["VEH"].where(~housing_units, 0.0)
    unit = int(np.flatnonzero(housing_units)[0])
    household.loc[unit, "VEH"] = 2.0
    before = household.copy(deep=True)
    entry = _summary(household, acs)
    pd.testing.assert_frame_equal(household, before)
    assert pd.isna(household.loc[unit, OWNED])
    assert entry["code_counts"]["2"] == 1
    assert entry["code_counts"]["0"] == int(housing_units.sum()) - 1
    assert entry["weighted_housing_unit_share_with_vehicle_available"] == pytest.approx(
        1 / int(housing_units.sum())
    )
    # The donor band is a reference, not a grade.
    assert entry["within_reference_band"] is False


def test_availability_ignores_a_group_quarters_code_and_counts_unknown_kinds() -> None:
    household, acs = _households()
    gq = int(np.flatnonzero(acs & (household["TYPEHUGQ"] == 3).to_numpy())[0])
    household.loc[gq, "VEH"] = 3.0
    household.loc[int(np.flatnonzero(acs)[0]), "TYPEHUGQ"] = np.nan
    entry = _summary(household, acs)
    assert entry["group_quarters_coded_rows"] == 1
    assert entry["unknown_kind_rows"] == 1
    assert sum(entry["code_counts"].values()) == entry["housing_units"]


@pytest.mark.parametrize("code", [np.nan, 7.0, -1.0, 2.5])
def test_availability_counts_a_housing_unit_without_a_veh_code(code) -> None:
    """Counted for review, not refused: VEH reaches no engine input."""

    household, acs = _households()
    unit = int(np.flatnonzero(acs & (household["TYPEHUGQ"] == 1).to_numpy())[0])
    household.loc[unit, "VEH"] = code
    entry = _summary(household, acs)
    assert entry["housing_units_without_veh"] == 1
    assert sum(entry["code_counts"].values()) == entry["housing_units"] - 1


def test_availability_records_a_missing_column_instead_of_refusing() -> None:
    household, acs = _households()
    entry = _summary(household.drop(columns="VEH"), acs)
    assert entry["missing_columns"] == ["VEH"]
    assert "code_counts" not in entry


def _grade(household: pd.DataFrame, rows: np.ndarray, *, graded: bool):
    entry: dict[str, object] = {}
    failures = grade_vehicles_owned(
        household.loc[rows], np.ones(int(rows.sum())), entry, graded=graded
    )
    return failures, entry


def test_owned_grade_passes_on_acs_rows_at_the_reviewed_default() -> None:
    """Missing before the reviewed-null fill and 0 after it; VEH is
    recorded beside the count as availability."""

    household, acs = _households()
    failures, entry = _grade(household, acs, graded=True)
    assert failures == []
    assert entry["not_default_rows"] == 0
    assert entry["missing_rows"] == int(acs.sum())
    housing_units = acs & (household["TYPEHUGQ"] == 1).to_numpy()
    assert entry["vehicles_available"]["housing_units"] == int(housing_units.sum())
    household[OWNED] = household[OWNED].where(~acs, 0.0)
    failures, entry = _grade(household, acs, graded=True)
    assert failures == []
    assert entry["unique_values"] == 1
    assert entry["weighted_share_with_vehicle"] == 0.0


def test_owned_grade_fails_on_veh_written_as_the_owned_count() -> None:
    """The #1064 review: VEH also counts leased and employer-provided cars."""

    household, acs = _households()
    household[OWNED] = household[OWNED].where(~acs, household["VEH"].fillna(0.0))
    written = int((acs & (household["VEH"] > 0).to_numpy()).sum())
    failures, entry = _grade(household, acs, graded=True)
    assert entry["not_default_rows"] == written
    assert len(failures) == 1
    assert failures[0].startswith(
        f"{OWNED} is not the reviewed default 0 in {written} household(s)"
    )
    assert "must not be written as the owned count" in failures[0]


def test_owned_grade_holds_donor_rows_to_complete_varying_counts() -> None:
    household, acs = _households()
    failures, entry = _grade(household, ~acs, graded=False)
    assert failures == []
    assert "vehicles_available" not in entry
    assert "not_default_rows" not in entry
    fractional = household.copy()
    fractional.loc[0, OWNED] = 1.5
    assert _grade(fractional, ~acs, graded=False)[0] == [
        f"{OWNED} has 1 negative or fractional value(s)."
    ]
    constant = household.copy()
    constant.loc[~acs, OWNED] = 0.0
    assert f"{OWNED} is constant" in " ".join(_grade(constant, ~acs, graded=False)[0])
    missing = household.copy()
    missing.loc[0, OWNED] = np.nan
    assert f"{OWNED} has missing rows" in " ".join(
        _grade(missing, ~acs, graded=False)[0]
    )


# ---------------------------------------------------------------------------
# Head Start take-up
# ---------------------------------------------------------------------------


def _persons(
    n_donor: int = 400, n_acs: int = 4000, donor_rate: float = 0.12
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Donor persons carry a flag, True only at ages 3-5; ACS persons none."""

    n = n_donor + n_acs
    rng = np.random.default_rng(3)
    acs = np.arange(n) >= n_donor
    age = rng.integers(0, 12, n).astype(float)
    domain = (age >= 3) & (age <= 5)
    donor_flags = domain & (rng.random(n) < donor_rate)
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1),
            "age": age,
            HEAD_START: pd.Series(donor_flags, dtype=object).where(~acs, np.nan),
        }
    )
    weights = rng.uniform(0.5, 2.0, n)
    return person, acs, ~acs, weights


def _keys(person: pd.DataFrame):
    """``acs_2024_1yr:SERIALNO:SPORDER`` stand-ins keyed on person_id."""

    def draw_keys(rows: np.ndarray) -> pd.Series:
        return (
            "acs_2024_1yr:HU" + person.loc[rows, "person_id"].astype(str) + ":1"
        ).reset_index(drop=True)

    return draw_keys


def _fill(person, acs, donor, weights, *, seed: int = 0):
    return head_start_fill(
        person, acs, donor, weights, seed=seed, draw_keys=_keys(person)
    )


def _donor_share(person, donor, weights) -> float:
    domain = donor & person["age"].between(3, 5).to_numpy()
    flags = person[HEAD_START].to_numpy(dtype=object)[domain].astype(bool)
    return float(weights[domain][flags].sum() / weights[domain].sum())


def test_head_start_draws_at_the_donor_share_within_ages_3_to_5() -> None:
    person, acs, donor, weights = _persons()
    missing, assigned, entry = _fill(person, acs, donor, weights)
    assert np.array_equal(missing, acs)
    rate = _donor_share(person, donor, weights)
    assert entry["rate"] == pytest.approx(rate)
    domain = acs & person["age"].between(3, 5).to_numpy()
    assert not assigned[acs & ~domain].any()
    share = weights[domain & assigned].sum() / weights[domain].sum()
    assert share == pytest.approx(rate, abs=0.03)
    assert entry["age_domain_filled_rows"] == int(domain.sum())
    assert entry["take_up_filled_rows"] == int(assigned.sum())
    assert entry["share_band"] == pytest.approx([rate / 2, rate * 1.5])
    assert entry["draw_key"] == "acs_2024_1yr:SERIALNO:SPORDER"


def test_head_start_draws_are_keyed_and_seeded() -> None:
    person, acs, donor, weights = _persons()
    _, first, _ = _fill(person, acs, donor, weights, seed=5)
    _, again, _ = _fill(person, acs, donor, weights, seed=5)
    _, other, _ = _fill(person, acs, donor, weights, seed=6)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    # The draw is the seeded blake2b uniform of the person's own key.
    rate = _fill(person, acs, donor, weights, seed=5)[2]["rate"]
    domain = acs & person["age"].between(3, 5).to_numpy()
    keys = _keys(person)(domain)
    uniforms = np.array(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"5:acs_local_head_start_take_up:{key}".encode(), digest_size=8
                ).digest(),
                "big",
            )
            / 2**64
            for key in keys
        ]
    )
    assert np.array_equal(first[domain], uniforms < rate)


def test_a_persons_draw_does_not_depend_on_other_acs_rows() -> None:
    person, acs, donor, weights = _persons()
    _, full, _ = _fill(person, acs, donor, weights, seed=2)
    # Store half of the ACS cells: the other half keep their draws.
    stored = acs & (np.arange(len(person)) % 2 == 0)
    partial = person.copy()
    partial.loc[stored, HEAD_START] = False
    missing, assigned, entry = _fill(partial, acs, donor, weights, seed=2)
    assert np.array_equal(missing, acs & ~stored)
    assert np.array_equal(assigned[missing], full[missing])
    assert entry["preserved_rows"] == int(stored.sum())


def test_head_start_leaves_donor_rows_to_the_donor() -> None:
    person, acs, donor, weights = _persons()
    missing, assigned, _ = _fill(person, acs, donor, weights)
    assert not missing[donor].any()
    assert not assigned[donor].any()


@pytest.mark.parametrize(
    "donor_value, match",
    [
        (True, "outside the sipp_head_start stage's band"),
        (False, "outside the sipp_head_start stage's band"),
        (np.nan, "carry no boolean"),
    ],
    ids=["universal", "never", "missing"],
)
def test_head_start_refuses_a_donor_without_the_sipp_stage(donor_value, match) -> None:
    person, acs, donor, weights = _persons()
    domain = donor & person["age"].between(3, 5).to_numpy()
    person.loc[domain, HEAD_START] = donor_value
    with pytest.raises(ValueError, match=match):
        _fill(person, acs, donor, weights)


def test_head_start_refuses_a_donor_without_children() -> None:
    person, acs, donor, weights = _persons()
    person.loc[donor, "age"] = 30.0
    with pytest.raises(ValueError, match="no weighted person aged 3-5"):
        _fill(person, acs, donor, weights)


def test_head_start_passes_key_refusals_through() -> None:
    person, acs, donor, weights = _persons()

    def refuse(rows: np.ndarray) -> pd.Series:
        raise ValueError("draw key repeats")

    with pytest.raises(ValueError, match="draw key repeats"):
        head_start_fill(person, acs, donor, weights, seed=0, draw_keys=refuse)


def test_donor_share_is_weighted_over_ages_3_to_5() -> None:
    person, _, donor, weights = _persons()
    share = donor_head_start_share(
        person[HEAD_START], person["age"].to_numpy(), donor, weights
    )
    assert share == pytest.approx(_donor_share(person, donor, weights))


def _filled_persons():
    person, acs, donor, weights = _persons()
    missing, assigned, entry = _fill(person, acs, donor, weights)
    person[HEAD_START] = person[HEAD_START].where(~missing, assigned).astype(bool)
    return person, acs, weights, entry["rate"]


def _grade_head_start(person, rows, weights, *, donor_share, graded: bool):
    entry: dict[str, object] = {}
    failures = grade_head_start(
        person.loc[rows],
        person.loc[rows, HEAD_START].to_numpy(dtype=bool),
        weights[rows],
        entry,
        donor_share=donor_share,
        graded=graded,
    )
    return failures, entry


def test_head_start_grade_passes_on_the_filled_draws() -> None:
    person, acs, weights, rate = _filled_persons()
    failures, entry = _grade_head_start(
        person, acs, weights, donor_share=rate, graded=True
    )
    assert failures == []
    assert entry["take_up_outside_ages"] == 0
    assert entry["donor_share"] == rate


def test_head_start_grade_fails_outside_the_ages_and_the_band() -> None:
    person, acs, weights, rate = _filled_persons()
    adult = int(np.flatnonzero(acs & (person["age"] > 5).to_numpy())[0])
    person.loc[adult, HEAD_START] = True
    failures = " ".join(
        _grade_head_start(person, acs, weights, donor_share=rate, graded=True)[0]
    )
    assert "1 person(s) outside ages 3-5 carry" in failures
    # Universal take-up at ages 3-5 is far above the donor share.
    domain = acs & person["age"].between(3, 5).to_numpy()
    person.loc[domain, HEAD_START] = True
    failures = " ".join(
        _grade_head_start(person, acs, weights, donor_share=rate, graded=True)[0]
    )
    assert "share among ages 3-5 is 1.000, outside" in failures


def test_head_start_grade_reports_donor_rows_and_needs_a_donor_share() -> None:
    person, acs, weights, rate = _filled_persons()
    person.loc[
        int(np.flatnonzero(~acs & (person["age"] > 5).to_numpy())[0]), HEAD_START
    ] = True
    failures, entry = _grade_head_start(
        person, ~acs, weights, donor_share=rate, graded=False
    )
    assert failures == []
    assert entry["take_up_outside_ages"] == 1
    failures, _ = _grade_head_start(person, acs, weights, donor_share=None, graded=True)
    assert failures == [f"no donor share to grade {HEAD_START} against."]
