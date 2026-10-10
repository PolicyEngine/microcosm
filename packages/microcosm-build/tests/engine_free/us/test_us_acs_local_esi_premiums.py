"""ESI employer premiums on the ACS local lane's ACS spine (microcosm #454).

The lane pools a dense ASEC-by-PUF donor release with the ACS spine
(``with_optional_acs_spine``): the donor keeps ``1 - acs_share`` of its
household mass and ACS takes the rest. The donor carries the premium from the
``meps_esi_premiums`` stage; the lane transfers it onto the ACS rows from the
donor's ASEC observation role and anchors the pooled frame.

Invariants of :func:`with_acs_local_esi_premium_anchor`, for every lane frame:

* **Conservation**: the lane-wide weighted employer premium equals the donor
  rows' total divided by their share of household mass. Pooling multiplies
  every donor household weight by that share, so this is the donor release's
  own weighted column.
* **Equal intensity**: both spines carry the same weighted premium per unit of
  household mass.
* **Donor rows are immutable**: no donor cell changes by a byte.
* **Structural zeros**: an ACS person who reports zero wages carries no
  premium, and neither does one below age 15, whose wage ACS leaves blank.
* **No invented zeros**: a blank wage at age 15 or older is refused, and the
  frame's own blanks are never overwritten.
* **Proportionality**: every other ACS premium is one common multiple of the
  value the transfer drew.
* **Idempotence**: an anchored frame is returned unchanged.
* **Identity independence**: ACS ids that repeat the donor's change nothing.
"""

from __future__ import annotations

import importlib.util
import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime import acs_multispine
from microcosm.build.us_runtime import esi_premiums as esi
from microcosm.build.us_runtime.acs_inputs import map_acs_native_inputs
from microcosm.build.us_runtime.acs_local_esi_premiums import (
    ACS_LOCAL_ESI_PREMIUM_FAMILY,
    acs_local_esi_premium_gate_payload,
    acs_local_esi_premium_gates,
    acs_local_esi_premium_transfer_target_families,
    prepare_acs_local_esi_premium_donor,
    with_acs_local_esi_premium_anchor,
)
from microcosm.build.us_runtime.acs_local_hours import (
    acs_local_transfer_target_families,
)
from microcosm.build.us_runtime.acs_pums import AcsPumsSource
from microcosm.build.us_runtime.acs_transfer import (
    declared_acs_transfer_target_families,
    transfer_acs_inputs,
)
from microcosm.build.us_runtime.esi_premiums import (
    US_ESI_PREMIUMS_OUTPUT_COLUMNS,
    refuse_unassigned_us_esi_premiums,
    us_esi_premiums_anchor_gate,
    us_esi_premiums_signal_gate,
    with_us_esi_premium_pool_anchor,
)
from microcosm.build.us_runtime.h5_io import (
    LEGACY_NULLABLE_STAGING_ARTIFACT_KIND,
    load_legacy_calibrated_us_h5,
    write_nullable_us_h5,
)
from microcosm.build.us_runtime.multispine_pool import pool_transfer_target_families
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_acs_local_esi_premiums import (
    ACS,
    ASEC_PUF,
    EMPLOYER,
    WAGES,
    build_lane,
    dense_donor,
    donor_rows,
    employer_total,
    household_mass_share,
    person_weights,
    pooled,
    raw_acs,
    reweighted,
    transferred_acs,
)

#: CMS NHE Table 24 employer contribution, CY2024: the stage's anchor.
ANCHOR = float(esi.EMPLOYER_PREMIUM_ANCHOR["values"]["2024"])
RAW = tuple(
    column
    for column in esi.US_ESI_PREMIUMS_REQUIRED_SOURCE_COLUMNS
    if column != "state_fips"
)
#: Above every donor id, so ACS ids do not repeat the donor's.
_DISJOINT_FIRST_ID = 5_000_000

requires_pytables = pytest.mark.skipif(
    importlib.util.find_spec("tables") is None,
    reason="requires pytables (the build environment)",
)


def _column(frame: Frame, column: str) -> np.ndarray:
    return frame.table("person")[column].to_numpy(dtype=float)


def _replace_person(frame: Frame, person: pd.DataFrame) -> Frame:
    return Frame(
        {
            entity: person if entity == "person" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )


# --- The lane's plan leaves the shared plans alone ----------------------------


def test_the_lane_plan_names_the_premium_and_changes_no_shared_plan() -> None:
    declared_before = declared_acs_transfer_target_families()
    pool_before = pool_transfer_target_families()

    local = acs_local_esi_premium_transfer_target_families()

    assert local == {"person": {ACS_LOCAL_ESI_PREMIUM_FAMILY: (EMPLOYER,)}}
    assert local["person"][ACS_LOCAL_ESI_PREMIUM_FAMILY] == (
        US_ESI_PREMIUMS_OUTPUT_COLUMNS
    )
    assert declared_acs_transfer_target_families() == declared_before
    assert pool_transfer_target_families() == pool_before
    # The declared plan still does not own the column, so the stacked pool
    # keeps it under its own pool-local family, where the early fill reads it.
    declared_targets = {
        target
        for families in declared_before.values()
        for targets in families.values()
        for target in targets
    }
    assert EMPLOYER not in declared_targets
    assert pool_before["person"][ACS_LOCAL_ESI_PREMIUM_FAMILY] == (EMPLOYER,)
    # The hours plan the staging tool composes with is untouched too.
    assert EMPLOYER not in {
        target
        for families in acs_local_transfer_target_families().values()
        for targets in families.values()
        for target in targets
    }


# --- Donor qualification ------------------------------------------------------


def test_the_donor_is_its_asec_observation_role_with_measured_wages() -> None:
    donor = dense_donor()
    before = {entity: donor.table(entity).copy() for entity in donor.entities}

    selected, receipt = prepare_acs_local_esi_premium_donor(donor)

    person = selected.table("person")
    assert len(person) == 320
    assert set(person["person_support_channel"]) == {"asec"}
    assert set(person["person_support_clone_index"]) == {0}
    # Measured wages: a premium only where the person has wages.
    assert not ((person[EMPLOYER] > 0) & ~(person[WAGES] > 0)).any()
    assert receipt["donor_channel"] == "asec"
    assert receipt["donor_rows"] == 640
    assert receipt["asec_observation_rows"] == 320
    assert receipt["asec_observation_premium_share_without_wages"] == 0.0
    assert receipt["donor_employer_premium_total"] == pytest.approx(
        employer_total(donor), rel=1e-12
    )
    assert receipt["donor_signal_gate"] == {"passed": True, "failures": []}
    for entity in donor.entities:
        pd.testing.assert_frame_equal(donor.table(entity), before[entity])


def _donor_without(*columns: str) -> Frame:
    donor = dense_donor(80)
    return _replace_person(donor, donor.table("person").drop(columns=list(columns)))


@pytest.mark.parametrize("column", [EMPLOYER, "NOW_OWNGRP", "PEIO1COW"])
def test_a_donor_built_without_the_stage_is_refused(column: str) -> None:
    with pytest.raises(ValueError, match="meps_esi_premiums stage") as error:
        prepare_acs_local_esi_premium_donor(_donor_without(column))
    assert column in str(error.value)
    assert "never defaults" in str(error.value)


def test_a_donor_without_the_cps_record_id_is_refused() -> None:
    with pytest.raises(ValueError, match="lacks 'PERIDNUM'"):
        prepare_acs_local_esi_premium_donor(_donor_without("PERIDNUM"))


def test_a_donor_row_with_raw_fields_and_no_record_id_is_refused() -> None:
    # The record id and the raw fields must agree on every row.
    donor = dense_donor(80)
    person = donor.table("person").copy()
    person["PERIDNUM"] = person["PERIDNUM"].astype(object)
    person.loc[person.index[3], "PERIDNUM"] = None
    with pytest.raises(ValueError, match="without a CPS record id"):
        prepare_acs_local_esi_premium_donor(_replace_person(donor, person))


def test_a_donor_without_wages_is_refused() -> None:
    with pytest.raises(ValueError, match="conditions the premium on"):
        prepare_acs_local_esi_premium_donor(_donor_without(WAGES))


def test_a_donor_whose_observation_role_has_missing_wages_is_refused() -> None:
    donor = dense_donor(80)
    person = donor.table("person").copy()
    person.loc[person["person_support_clone_index"].eq(0).idxmax(), WAGES] = np.nan
    with pytest.raises(ValueError, match="never treats a missing wage as zero"):
        prepare_acs_local_esi_premium_donor(_replace_person(donor, person))


def test_a_donor_without_support_roles_is_refused() -> None:
    donor = dense_donor(80)
    tables = {
        entity: donor.table(entity).drop(
            columns=[
                f"{entity}_source_id",
                f"{entity}_support_channel",
                f"{entity}_support_clone_index",
            ]
        )
        for entity in donor.entities
    }
    unclassified = Frame(
        tables, US_SCHEMA, {"household": donor.weights_for("household")}
    )
    with pytest.raises(ValueError, match="explicit legacy ASEC observation roles"):
        prepare_acs_local_esi_premium_donor(unclassified)


def test_an_assembled_multispine_donor_is_refused() -> None:
    donor = dense_donor(80)
    person = donor.table("person").copy()
    person["person_spine_source_id"] = person["person_source_id"]
    with pytest.raises(ValueError, match="multispine records need their own"):
        prepare_acs_local_esi_premium_donor(_replace_person(donor, person))


def test_a_donor_failing_the_stage_signal_gate_is_refused() -> None:
    # A premium outside its MEPS-IC cell: no longer one common multiple.
    donor = dense_donor(80)
    person = donor.table("person").copy()
    positive = person[EMPLOYER] > 0
    person.loc[positive.idxmax(), EMPLOYER] *= 3.0
    with pytest.raises(ValueError, match="fails the meps_esi_premiums signal gate"):
        prepare_acs_local_esi_premium_donor(_replace_person(donor, person))


# --- The anchor on every lane frame -------------------------------------------

_amount = st.one_of(
    st.just(0.0),
    st.floats(min_value=1.0, max_value=40_000.0, allow_nan=False),
)
_weight = st.floats(min_value=0.5, max_value=5_000.0, allow_nan=False)


@cache
def _small_donor() -> Frame:
    return dense_donor(40)


@st.composite
def _lane_cases(draw) -> dict[str, object]:
    """Any donor weights, any share, and any ACS households and draws.

    The donor's household weights are rescaled one by one, so the donor rows
    do not share one weight profile; ACS households hold one to several
    people.
    """

    people = draw(st.integers(min_value=2, max_value=10))
    premiums = np.asarray(draw(st.lists(_amount, min_size=people, max_size=people)))
    wages = np.asarray(draw(st.lists(_amount, min_size=people, max_size=people)))
    # Some ACS people are children: PUMS leaves their wage blank.
    child = np.asarray(draw(st.lists(st.booleans(), min_size=people, max_size=people)))
    wages = np.where(child, np.nan, wages)
    ages = np.where(child, 9.0, 30.0 + np.arange(people))
    joins = draw(st.lists(st.booleans(), min_size=people - 1, max_size=people - 1))
    households = np.concatenate([[0], np.cumsum(~np.asarray(joins, dtype=bool))])
    count = int(households.max()) + 1
    weights = np.asarray(draw(st.lists(_weight, min_size=count, max_size=count)))
    assume(((premiums > 0) & (wages > 0)).any())
    donor = _small_donor()
    factors = np.asarray(
        draw(
            st.lists(
                st.floats(min_value=0.05, max_value=20.0),
                min_size=donor.n("household"),
                max_size=donor.n("household"),
            )
        )
    )
    scale = draw(st.floats(min_value=0.01, max_value=100.0))
    return {
        "premiums": premiums,
        "wages": wages,
        "ages": ages,
        "weights": weights,
        "households": households,
        "acs_share": draw(st.floats(min_value=0.05, max_value=0.95)),
        "donor": reweighted(donor, factors * scale),
    }


def _lane_frame(case: dict[str, object], *, first_id: int = 1) -> Frame:
    acs = transferred_acs(
        case["premiums"],
        case["wages"],
        case["weights"],
        households=case["households"],
        ages=case["ages"],
        first_id=first_id,
    )
    return pooled(case["donor"], acs, acs_share=case["acs_share"])


_SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


@given(_lane_cases())
@_SETTINGS
def test_lane_total_is_the_donor_rows_total_over_their_mass_share(case) -> None:
    frame = _lane_frame(case)
    donor = donor_rows(frame)
    share = household_mass_share(frame, donor)

    anchored, receipt = with_acs_local_esi_premium_anchor(frame)

    # Pooling leaves the donor spine 1 - acs_share of the household mass.
    assert share == pytest.approx(1.0 - case["acs_share"], rel=1e-9)
    assert receipt["source_household_mass_share"] == pytest.approx(share, rel=1e-12)
    expected = employer_total(frame, donor) / share
    assert employer_total(anchored) == pytest.approx(expected, rel=1e-9)
    assert receipt["employer_premium_total"] == pytest.approx(expected, rel=1e-9)
    # Differential: that is the donor release's column at its own weights.
    assert employer_total(anchored) == pytest.approx(
        employer_total(case["donor"]), rel=1e-9
    )


@given(_lane_cases())
@_SETTINGS
def test_both_spines_carry_the_same_premium_per_unit_of_household_mass(case) -> None:
    frame = _lane_frame(case)
    donor = donor_rows(frame)
    share = household_mass_share(frame, donor)

    anchored, _receipt = with_acs_local_esi_premium_anchor(frame)

    assert employer_total(anchored, ~donor) / (1.0 - share) == pytest.approx(
        employer_total(anchored, donor) / share, rel=1e-9
    )


@given(_lane_cases())
@_SETTINGS
def test_donor_rows_and_every_other_column_are_untouched(case) -> None:
    frame = _lane_frame(case)
    donor = donor_rows(frame)

    anchored, _receipt = with_acs_local_esi_premium_anchor(frame)

    before, after = frame.table("person"), anchored.table("person")
    assert list(after.columns) == list(before.columns)
    pd.testing.assert_frame_equal(after.loc[donor], before.loc[donor])
    untouched = [column for column in before.columns if column != EMPLOYER]
    pd.testing.assert_frame_equal(after[untouched], before[untouched])
    for entity in frame.entities:
        if entity != "person":
            pd.testing.assert_frame_equal(anchored.table(entity), frame.table(entity))
    np.testing.assert_array_equal(person_weights(anchored), person_weights(frame))
    assert anchored.weights_for("household").kind is WeightKind.IMPORTANCE
    assert list(anchored.strata) == list(frame.strata)
    assert anchored.mass_log == frame.mass_log


@given(_lane_cases())
@_SETTINGS
def test_acs_people_without_wages_carry_nothing_and_the_rest_one_multiple(
    case,
) -> None:
    frame = _lane_frame(case)
    donor = donor_rows(frame)
    workers = _column(frame, WAGES) > 0

    anchored, receipt = with_acs_local_esi_premium_anchor(frame)

    cleared = ~donor & ~workers
    assert not _column(anchored, EMPLOYER)[cleared].any()
    assert receipt["cleared_rows"][EMPLOYER] == int(
        (cleared & (_column(frame, EMPLOYER) > 0)).sum()
    )
    scaled = ~donor & workers
    np.testing.assert_allclose(
        _column(anchored, EMPLOYER)[scaled],
        _column(frame, EMPLOYER)[scaled] * receipt["scale_factor"],
        rtol=1e-12,
    )
    values = _column(anchored, EMPLOYER)
    assert np.isfinite(values).all() and (values >= 0).all()
    # Every row carries the premium: nothing for a null fill to refuse.
    refuse_unassigned_us_esi_premiums(anchored, consumer="lane property")


@given(_lane_cases())
@_SETTINGS
def test_an_anchored_lane_frame_is_returned_unchanged(case) -> None:
    anchored, first = with_acs_local_esi_premium_anchor(_lane_frame(case))

    again, second = with_acs_local_esi_premium_anchor(anchored)

    assert first["status"] in {"scaled", "already_on_anchor"}
    assert again is anchored
    assert second["status"] == "already_on_anchor"
    assert second["employer_premium_total"] == pytest.approx(
        first["employer_premium_total"], rel=1e-12
    )


@given(_lane_cases())
@_SETTINGS
def test_acs_ids_that_repeat_the_donors_change_nothing(case) -> None:
    # The pool kernel finds a clone's wages by person_source_id across the
    # whole frame, and refuses a frame where source records repeat one. In
    # this lane ACS rows keep their pre-remap ids as source ids.
    colliding = _lane_frame(case, first_id=1)
    disjoint = _lane_frame(case, first_id=_DISJOINT_FIRST_ID)
    acs = ~donor_rows(colliding)
    assert set(colliding.table("person").loc[acs, "person_source_id"]) & set(
        colliding.table("person").loc[~acs, "person_source_id"]
    )
    with pytest.raises(SourceRuntimeError, match="repeat a 'person_source_id'"):
        with_us_esi_premium_pool_anchor(colliding)

    anchored, receipt = with_acs_local_esi_premium_anchor(colliding)
    reference, reference_receipt = with_acs_local_esi_premium_anchor(disjoint)

    np.testing.assert_array_equal(
        _column(anchored, EMPLOYER), _column(reference, EMPLOYER)
    )
    assert receipt["scale_factor"] == reference_receipt["scale_factor"]
    if np.isnan(case["wages"]).any():
        # The kernel alone has no universe rule for a blank ACS wage.
        with pytest.raises(SourceRuntimeError, match="missing wage is not a zero"):
            with_us_esi_premium_pool_anchor(disjoint)
        return
    # Differential: where the ids are unique the kernel runs on the whole
    # frame, and the lane's narrow view gives the same bytes.
    whole, whole_receipt = with_us_esi_premium_pool_anchor(disjoint)
    np.testing.assert_array_equal(
        _column(reference, EMPLOYER), _column(whole, EMPLOYER)
    )
    assert reference_receipt["scale_factor"] == whole_receipt["scale_factor"]
    assert reference_receipt["cleared_rows"] == whole_receipt["cleared_rows"]


def test_the_worked_lane_frame() -> None:
    # Donor release: 40 one-person households cloned once. ACS: three people,
    # one without wages. acs_share 0.25 leaves the donor 0.75 of the mass.
    donor = _small_donor()
    release_total = employer_total(donor)
    acs = transferred_acs(
        np.asarray([10_000.0, 5_000.0, 8_000.0]),
        np.asarray([50_000.0, 0.0, 20_000.0]),
        np.asarray([1.0, 1.0, 2.0]),
    )
    frame = pooled(donor, acs, acs_share=0.25)
    rows = donor_rows(frame)

    anchored, receipt = with_acs_local_esi_premium_anchor(frame)

    assert receipt["source_household_mass_share"] == pytest.approx(0.75, rel=1e-12)
    assert employer_total(frame, rows) == pytest.approx(0.75 * release_total, rel=1e-9)
    assert receipt["transferred_employer_premium_target"] == pytest.approx(
        0.25 * release_total, rel=1e-9
    )
    assert receipt["cleared_rows"] == {EMPLOYER: 1}
    assert employer_total(anchored) == pytest.approx(release_total, rel=1e-9)
    premiums = _column(anchored, EMPLOYER)[~rows]
    # One factor on the two workers: 10,000 and 8,000 keep their ratio.
    assert premiums[1] == 0.0
    assert premiums[0] / premiums[2] == pytest.approx(10_000.0 / 8_000.0, rel=1e-12)


def test_a_transferred_support_clone_is_refused() -> None:
    frame = pooled(
        _small_donor(),
        transferred_acs(np.asarray([9_000.0]), np.asarray([30_000.0]), np.ones(1)),
    )
    person = frame.table("person").copy()
    person.loc[~donor_rows(frame), "person_support_clone_index"] = 1
    with pytest.raises(SourceRuntimeError, match="not clone-0 source records"):
        with_acs_local_esi_premium_anchor(_replace_person(frame, person))


@pytest.mark.parametrize("column", [EMPLOYER, "NOW_HIPAID", "PERIDNUM", WAGES])
def test_a_lane_frame_missing_an_anchor_input_is_refused(column: str) -> None:
    frame = pooled(
        _small_donor(),
        transferred_acs(np.asarray([9_000.0]), np.asarray([30_000.0]), np.ones(1)),
    )
    person = frame.table("person").drop(columns=[column])
    with pytest.raises(SourceRuntimeError, match="the person table lacks") as error:
        with_acs_local_esi_premium_anchor(_replace_person(frame, person))
    assert column in str(error.value)


def test_acs_rows_with_no_premium_on_any_worker_are_refused() -> None:
    # Nothing to scale: the anchor cannot put the ACS half on the donor's
    # scale, so staging fails rather than shipping half the column.
    frame = pooled(
        _small_donor(),
        transferred_acs(
            np.asarray([0.0, 7_000.0]), np.asarray([30_000.0, 0.0]), np.ones(2)
        ),
    )
    with pytest.raises(SourceRuntimeError, match="nothing to scale"):
        with_acs_local_esi_premium_anchor(frame)


def test_a_multi_person_household_scales_as_one_unit_of_mass() -> None:
    # Three ACS people in one household and one alone: the household's mass
    # counts once, and its non-worker is cleared.
    donor = _small_donor()
    acs = transferred_acs(
        np.asarray([6_000.0, 4_000.0, 5_000.0, 3_000.0]),
        np.asarray([40_000.0, 25_000.0, 0.0, 30_000.0]),
        np.asarray([3.0, 1.0]),
        households=np.asarray([0, 0, 0, 1]),
    )
    frame = pooled(donor, acs, acs_share=0.4)
    rows = donor_rows(frame)

    anchored, receipt = with_acs_local_esi_premium_anchor(frame)

    assert household_mass_share(frame, rows) == pytest.approx(0.6, rel=1e-12)
    assert receipt["cleared_rows"] == {EMPLOYER: 1}
    assert employer_total(anchored) == pytest.approx(employer_total(donor), rel=1e-9)
    premiums = _column(anchored, EMPLOYER)[~rows]
    assert premiums[2] == 0.0
    np.testing.assert_allclose(
        premiums[[0, 1, 3]] / receipt["scale_factor"], [6_000.0, 4_000.0, 3_000.0]
    )


def _lane_with_a_child(**changes) -> Frame:
    """Two ACS adults and one nine-year-old whose wage PUMS left blank."""

    columns = {
        "premiums": np.asarray([8_000.0, 5_000.0, 6_000.0]),
        "wages": np.asarray([40_000.0, np.nan, 30_000.0]),
        "ages": np.asarray([41.0, 9.0, 52.0]),
    } | changes
    acs = transferred_acs(
        columns["premiums"], columns["wages"], np.ones(3), ages=columns["ages"]
    )
    return pooled(_small_donor(), acs)


def test_a_blank_acs_wage_below_fifteen_is_a_zero_wage_in_the_anchor_only() -> None:
    frame = _lane_with_a_child()
    acs = ~donor_rows(frame)

    anchored, receipt = with_acs_local_esi_premium_anchor(frame)

    premiums = _column(anchored, EMPLOYER)[acs]
    assert premiums[1] == 0.0 and (premiums[[0, 2]] > 0).all()
    assert receipt["cleared_rows"] == {EMPLOYER: 1}
    assert receipt["wage_universe"] == {
        "rule_id": "acs_2024_pums_wagp_age_15_plus",
        "minimum_age": 15,
        "raw_source_column": "WAGP",
        "transferred_rows_with_blank_wage": 1,
        "transferred_rows_outside_universe_read_as_zero": 1,
        "frame_cells_written": 0,
    }
    # The staging frame keeps the source's blank: no zero is written to it.
    np.testing.assert_array_equal(
        _column(anchored, WAGES)[acs], [40_000.0, np.nan, 30_000.0]
    )
    assert employer_total(anchored) == pytest.approx(
        employer_total(_small_donor()), rel=1e-9
    )


@pytest.mark.parametrize("age", [15.0, 40.0, np.nan])
def test_a_blank_wage_at_fifteen_or_older_is_refused(age: float) -> None:
    # Fifteen is inside the PUMS earnings universe, and an unknown age is not
    # evidence of being outside it.
    frame = _lane_with_a_child(ages=np.asarray([41.0, age, 52.0]))
    with pytest.raises(SourceRuntimeError, match="missing wage is not a zero wage"):
        with_acs_local_esi_premium_anchor(frame)


def test_a_child_whose_raw_wage_is_not_blank_is_not_read_as_zero() -> None:
    # The mapped wage is null but PUMS reported one: that is a mapping fault,
    # not the universe rule.
    frame = _lane_with_a_child()
    person = frame.table("person").copy()
    acs = ~donor_rows(frame)
    person["WAGP"] = np.where(acs, 1_200.0, np.nan)
    with pytest.raises(SourceRuntimeError, match="missing wage is not a zero wage"):
        with_acs_local_esi_premium_anchor(_replace_person(frame, person))


def test_a_blank_raw_wage_below_fifteen_corroborates_the_universe_rule() -> None:
    frame = _lane_with_a_child()
    person = frame.table("person").copy()
    acs = ~donor_rows(frame)
    person["WAGP"] = np.nan
    person.loc[acs, "WAGP"] = [40_000.0, np.nan, 30_000.0]

    _anchored, receipt = with_acs_local_esi_premium_anchor(
        _replace_person(frame, person)
    )

    assert (
        receipt["wage_universe"]["transferred_rows_outside_universe_read_as_zero"] == 1
    )


def test_an_untransferred_acs_spine_is_refused_by_the_anchor() -> None:
    # A lane frame whose ACS rows never got the premium: null, never zero.
    frame = pooled(
        _small_donor(),
        transferred_acs(np.asarray([np.nan]), np.asarray([30_000.0]), np.ones(1)),
    )
    with pytest.raises(SourceRuntimeError, match="the transfer left a gap"):
        with_acs_local_esi_premium_anchor(frame)
    with pytest.raises(SourceRuntimeError, match="never filled"):
        refuse_unassigned_us_esi_premiums(frame, consumer="ACS local release")


# --- The staging orchestration, with the real transfer ------------------------


@pytest.fixture(scope="module")
def lane() -> dict[str, object]:
    donor = dense_donor()
    result = build_lane(donor, raw_acs())
    return {"donor": donor, "result": result, "frame": result.frame}


def test_every_lane_row_carries_the_premium(lane) -> None:
    frame = lane["frame"]
    person = frame.table("person")
    acs = person["person_spine"].eq(ACS).to_numpy()

    assert acs.sum() == 280 and (~acs).sum() == 640
    np.testing.assert_array_equal(acs, ~donor_rows(frame))
    assert set(person.loc[~acs, "person_spine"]) == {ASEC_PUF}
    values = person[EMPLOYER].to_numpy(dtype=float)
    assert np.isfinite(values).all() and (values >= 0).all()
    assert (values[acs] > 0).any() and np.unique(values[acs]).size > 2
    # The raw ASEC fields stay null on the ACS rows: they mark the row kind.
    assert person.loc[acs, list(RAW)].isna().all().all()
    assert person.loc[~acs, list(RAW)].notna().all().all()
    # fill_reviewed_nulls' refusal has nothing to refuse.
    refuse_unassigned_us_esi_premiums(
        frame, consumer="ACS local release (fill_reviewed_nulls)"
    )


def test_the_lane_total_is_the_donor_releases_total(lane) -> None:
    frame, donor = lane["frame"], lane["donor"]
    rows = donor_rows(frame)
    anchor = lane["result"].provenance["local_esi_premiums"]["anchor"]

    assert household_mass_share(frame, rows) == pytest.approx(0.5, rel=1e-12)
    assert employer_total(frame) == pytest.approx(employer_total(donor), rel=1e-9)
    assert employer_total(frame, rows) == pytest.approx(
        0.5 * employer_total(donor), rel=1e-9
    )
    assert employer_total(frame, ~rows) == pytest.approx(
        0.5 * employer_total(donor), rel=1e-9
    )
    assert anchor["status"] == "scaled"
    assert anchor["kernel"] == "with_us_esi_premium_pool_anchor"
    assert anchor["source_household_mass_share"] == pytest.approx(0.5, rel=1e-12)
    assert anchor["employer_premium_total"] == pytest.approx(
        employer_total(donor), rel=1e-9
    )


def test_the_donor_spine_is_the_release_byte_for_byte(lane) -> None:
    frame, donor = lane["frame"], lane["donor"]
    person = frame.table("person")
    rows = donor_rows(frame)

    np.testing.assert_array_equal(
        person.loc[rows, EMPLOYER].to_numpy(), _column(donor, EMPLOYER)
    )
    np.testing.assert_array_equal(
        person.loc[rows, WAGES].to_numpy(), _column(donor, WAGES)
    )


def test_acs_people_without_wages_carry_no_premium(lane) -> None:
    person = lane["frame"].table("person")
    acs = ~donor_rows(lane["frame"])
    workers = (person[WAGES] > 0).to_numpy()
    anchor = lane["result"].provenance["local_esi_premiums"]["anchor"]

    assert (acs & ~workers).sum() == 112
    assert not person.loc[acs & ~workers, EMPLOYER].any()
    assert (person.loc[acs & workers, EMPLOYER] > 0).any()
    # The fit saw measured wages, so no draw needed clearing.
    assert anchor["transferred_rows_without_source_wages"] == 112
    assert anchor["cleared_rows"] == {EMPLOYER: 0}
    assert 0.5 <= anchor["scale_factor"] <= 2.0


def test_the_premium_transfers_from_asec_observations_in_its_own_pass(lane) -> None:
    provenance = lane["result"].provenance
    entries = [
        item for item in provenance["imputed_inputs"] if item["column"] == EMPLOYER
    ]

    assert provenance["fit_configuration"]["esi_premium_donor_channel"] == "asec"
    assert provenance["local_esi_premiums"]["source"]["donor_channel"] == "asec"
    assert len(entries) == 1
    assert entries[0]["family"] == ACS_LOCAL_ESI_PREMIUM_FAMILY
    assert entries[0]["donor_channel"] == "asec"
    assert entries[0]["imputed_recipient_rows"] == 280
    assert entries[0]["unmodeled_recipient_rows"] == 0
    assert "__acs_transfer_employment_income" in entries[0]["predictors"]
    json.dumps(provenance, allow_nan=False)


def test_the_tax_detail_transfer_keeps_its_puf_donor() -> None:
    # The lane's other families still fit on the PUF-detail role.
    donor = dense_donor(120)
    person = donor.table("person").copy()
    puf = person["person_support_clone_index"].eq(1).to_numpy()
    person["taxable_interest_income"] = np.where(puf, 500.0, 5.0) + np.arange(
        len(person)
    )
    result = build_lane(
        _replace_person(donor, person),
        raw_acs(60),
        n_estimators=5,
        target_families={"person": {"tax_detail": ("taxable_interest_income",)}},
    )

    configuration = result.provenance["fit_configuration"]
    assert configuration["resolved_donor_channel"] == "puf_tax_detail"
    assert configuration["esi_premium_donor_channel"] == "asec"
    channels = {
        item["column"]: item["donor_channel"]
        for item in result.provenance["imputed_inputs"]
    }
    assert channels == {
        EMPLOYER: "asec",
        "taxable_interest_income": "puf_tax_detail",
    }
    acs = ~donor_rows(result.frame)
    assert (
        result.frame.table("person").loc[acs, "taxable_interest_income"] >= 500
    ).all()


def test_acs_children_get_no_premium_and_keep_their_blank_wage() -> None:
    # The real orchestration on an ACS spine with children: PUMS leaves their
    # WAGP blank, the native mapping keeps the blank, and the anchor reads it
    # as zero wages without writing a zero into the staging frame.
    donor = dense_donor(120)
    result = build_lane(donor, raw_acs(100, children=20), n_estimators=5)
    person = result.frame.table("person")
    acs = ~donor_rows(result.frame)
    child = acs & (person["age"] < 15).to_numpy()
    anchor = result.provenance["local_esi_premiums"]["anchor"]

    assert child.sum() == 20
    assert person.loc[child, WAGES].isna().all()
    assert person.loc[child, "WAGP"].isna().all()
    assert not person.loc[child, EMPLOYER].any()
    assert anchor["wage_universe"]["transferred_rows_with_blank_wage"] == 20
    assert (
        anchor["wage_universe"]["transferred_rows_outside_universe_read_as_zero"] == 20
    )
    assert employer_total(result.frame) == pytest.approx(
        employer_total(donor), rel=1e-9
    )
    signal, anchor_gate = acs_local_esi_premium_gates(result.frame, time_period=2024)
    assert signal.passed, signal.failures
    assert anchor_gate.passed, anchor_gate.failures
    refuse_unassigned_us_esi_premiums(result.frame, consumer="ACS local release")


def test_no_factory_leaves_the_orchestration_as_it_was() -> None:
    # A caller that does not ask for the premium gets none: no pass, no anchor.
    donor = dense_donor(80)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            acs_multispine,
            "build_acs_pums_unit_frame",
            lambda *a, **k: (raw_acs(40), {}),
        )
        result = acs_multispine.build_optional_acs_multispine(
            donor,
            AcsPumsSource(Path("unused-hus.zip"), Path("unused-pus.zip")),
            target_families={},
        )

    assert "local_esi_premiums" not in result.provenance
    assert "esi_premium_donor_channel" not in result.provenance["fit_configuration"]
    acs = ~donor_rows(result.frame)
    assert result.frame.table("person").loc[acs, EMPLOYER].isna().all()


# --- Which donor role, and what it does to the structural zero ----------------


def _acs_draws(donor: Frame, channel: str) -> tuple[pd.Series, pd.Series]:
    mapped = map_acs_native_inputs(raw_acs()).frame
    person = transfer_acs_inputs(
        mapped,
        donor,
        target_families=acs_local_esi_premium_transfer_target_families(),
        donor_channel=channel,
        seed=0,
        n_estimators=20,
    ).frame.table("person")
    return person[EMPLOYER], person[WAGES]


def test_the_puf_detail_role_would_put_premiums_on_people_without_wages() -> None:
    # On PUF-detail rows the premium is a copy of the source person's and the
    # wages are PUF-imputed, so wages no longer say who holds a job.
    donor = dense_donor()
    donor_person = donor.table("person")
    puf = donor_person["person_support_clone_index"].eq(1).to_numpy()
    premium, wages = _column(donor, EMPLOYER), _column(donor, WAGES)
    assert not ((premium > 0) & ~(wages > 0))[~puf].any()
    assert ((premium > 0) & ~(wages > 0))[puf].any()

    asec_premium, asec_wages = _acs_draws(donor, "asec")
    puf_premium, puf_wages = _acs_draws(donor, "puf_tax_detail")

    def share_without_wages_drawing_a_premium(drawn, measured) -> float:
        idle = ~(measured > 0)
        return float((drawn[idle] > 0).mean())

    def share_of_workers_drawing_a_premium(drawn, measured) -> float:
        return float((drawn[measured > 0] > 0).mean())

    # ASEC observations: the fit learns the structural zero from measured wages.
    assert share_without_wages_drawing_a_premium(asec_premium, asec_wages) == 0.0
    # PUF-detail role: people without wages draw premiums, and workers draw
    # fewer, because the fit cannot tell the two apart.
    assert share_without_wages_drawing_a_premium(puf_premium, puf_wages) > 0.1
    assert share_of_workers_drawing_a_premium(
        puf_premium, puf_wages
    ) < share_of_workers_drawing_a_premium(asec_premium, asec_wages)


# --- Both gates on the lane frame ---------------------------------------------


def test_both_gates_pass_on_the_lane_frame(lane) -> None:
    frame, donor = lane["frame"], lane["donor"]

    signal, anchor = acs_local_esi_premium_gates(frame, time_period=2024)

    assert signal.passed, signal.failures
    assert anchor.passed, anchor.failures
    assert signal.details["source_rows"] == 640
    assert signal.details["transferred_rows"] == 280
    assert signal.details["employer_premium_outside_universe_rows"] == 0
    assert signal.details["clone_disagreement_source_persons"] == 0
    assert signal.details["transferred_source_records_with_premium_and_no_wages"] == 0
    # The donor ran the stage at the whole anchor, and the lane keeps its
    # anchor-universe total and its scale factor.
    alone = us_esi_premiums_anchor_gate(donor, time_period=2024)
    assert alone.passed, alone.failures
    assert anchor.details["anchor_universe_employer_total"] == pytest.approx(
        ANCHOR, rel=1e-9
    )
    assert anchor.details["relative_error"] == pytest.approx(
        alone.details["relative_error"], abs=1e-9
    )
    assert anchor.details["scale_factor"] == pytest.approx(
        alone.details["scale_factor"], rel=1e-9
    )
    assert anchor.details["source_household_mass_share"] == pytest.approx(
        0.5, rel=1e-12
    )
    assert anchor.details[
        "transferred_to_source_per_household_mass_ratio"
    ] == pytest.approx(1.0, rel=1e-9)
    assert "employed share" in anchor.details["anchor_universe_assumption"]


def test_the_narrow_gate_view_gives_the_whole_frame_verdicts(lane) -> None:
    # Differential: the gates read a few columns but copy every table, so the
    # lane hands them a narrow view. It must not change a verdict or a detail.
    frame = lane["frame"]

    signal, anchor = acs_local_esi_premium_gates(frame, time_period=2024)
    whole_signal = us_esi_premiums_signal_gate(frame)
    whole_anchor = us_esi_premiums_anchor_gate(frame, time_period=2024)

    for narrow, whole in ((signal, whole_signal), (anchor, whole_anchor)):
        assert narrow.name == whole.name
        assert narrow.passed is whole.passed
        assert narrow.failures == whole.failures
        assert json.dumps(dict(narrow.details), sort_keys=True, default=str) == (
            json.dumps(dict(whole.details), sort_keys=True, default=str)
        )


def _with_acs_premium(frame: Frame, transform) -> Frame:
    person = frame.table("person").copy()
    acs = ~donor_rows(frame)
    person.loc[acs, EMPLOYER] = transform(person.loc[acs, EMPLOYER])
    return _replace_person(frame, person)


def test_an_acs_spine_without_the_premium_fails_the_signal_gate(lane) -> None:
    zeroed = _with_acs_premium(lane["frame"], lambda values: values * 0.0)

    signal, _anchor = acs_local_esi_premium_gates(zeroed, time_period=2024)

    assert not signal.passed
    assert any(
        "rows without raw ASEC columns" in failure for failure in signal.failures
    )


@pytest.mark.parametrize("factor", [0.7, 1.4])
def test_an_acs_spine_off_the_donor_rows_scale_fails_the_anchor_gate(
    lane, factor: float
) -> None:
    scaled = _with_acs_premium(lane["frame"], lambda values: values * factor)

    _signal, anchor = acs_local_esi_premium_gates(scaled, time_period=2024)

    assert not anchor.passed
    assert any("per unit of household mass" in failure for failure in anchor.failures)
    assert anchor.details[
        "transferred_to_source_per_household_mass_ratio"
    ] == pytest.approx(factor, rel=1e-9)


def test_a_lane_frame_from_a_donor_without_the_stage_fails_both_gates() -> None:
    # A waived diagnostic build: no row carries the column at all.
    acs = transferred_acs(np.asarray([9_000.0]), np.asarray([30_000.0]), np.ones(1))
    frame = pooled(
        _donor_without(EMPLOYER),
        _replace_person(acs, acs.table("person").drop(columns=[EMPLOYER])),
    )
    assert EMPLOYER not in frame.table("person")

    signal, anchor = acs_local_esi_premium_gates(frame, time_period=2024)

    assert not signal.passed and not anchor.passed
    assert signal.failures == (f"person column missing: {EMPLOYER}.",)
    assert anchor.failures == (f"person column missing: {EMPLOYER}.",)


def test_moving_mass_between_spines_moves_neither_the_total_nor_the_ratio(
    lane,
) -> None:
    # Calibration moves mass between spines. Both carry the same premium per
    # unit of household mass, so a shift that conserves the frame's mass
    # leaves the lane total where it was.
    frame = lane["frame"]
    household = frame.table("household")
    acs_household = household["household_spine"].eq(ACS).to_numpy()
    weights = np.asarray(frame.weights_for("household").values, dtype=float).copy()
    weights[acs_household] *= 1.3
    weights *= frame.weights_for("household").total / weights.sum()
    moved = Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {"household": Weights(weights, WeightKind.CALIBRATED)},
        frame.strata,
    )

    signal, anchor = acs_local_esi_premium_gates(moved, time_period=2024)

    assert signal.passed, signal.failures
    assert anchor.passed, anchor.failures
    assert anchor.details[
        "transferred_to_source_per_household_mass_ratio"
    ] == pytest.approx(1.0, rel=1e-9)
    assert employer_total(moved) == pytest.approx(employer_total(frame), rel=1e-9)


def test_a_gate_payload_is_strict_json_and_keeps_failures_verbatim(lane) -> None:
    signal, anchor = acs_local_esi_premium_gates(
        _with_acs_premium(lane["frame"], lambda values: values * 0.0),
        time_period=2024,
    )

    payload = acs_local_esi_premium_gate_payload(signal)

    assert payload["name"] == "esi_premiums_signal"
    assert payload["passed"] is False
    assert payload["failures"] == list(signal.failures)
    json.dumps(payload, allow_nan=False)
    json.dumps(acs_local_esi_premium_gate_payload(anchor), allow_nan=False)


@requires_pytables
def test_the_staging_h5_round_trip_keeps_the_verdicts(lane, tmp_path) -> None:
    # Finalize grades the calibrated H5 the lane wrote, not the frame in memory.
    frame = lane["frame"]
    path = tmp_path / "acs_multispine_staging.h5"

    write_nullable_us_h5(
        frame, path, period=2024, artifact_kind=LEGACY_NULLABLE_STAGING_ARTIFACT_KIND
    )
    loaded = load_legacy_calibrated_us_h5(path)

    signal, anchor = acs_local_esi_premium_gates(loaded, time_period=2024)
    assert signal.passed, signal.failures
    assert anchor.passed, anchor.failures
    assert employer_total(loaded) == pytest.approx(employer_total(frame), rel=1e-12)
    np.testing.assert_array_equal(donor_rows(loaded), donor_rows(frame))
    refuse_unassigned_us_esi_premiums(loaded, consumer="ACS local release")
    again, receipt = with_acs_local_esi_premium_anchor(loaded)
    assert again is loaded and receipt["status"] == "already_on_anchor"
