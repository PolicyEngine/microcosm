"""ACS-row SSI and Medicaid take-up for the retained ACS local lane
(microcosm#1022), with stubbed engine pre-pass values."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.acs_local_ssi_medicaid_take_up import (
    ACS_LOCAL_SSI_BAND_RELATIVE_TOLERANCE,
    ACS_LOCAL_SSI_MEDICAID_TAKE_UP_GATE_NAME,
    ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE,
    acs_local_ssi_medicaid_take_up_assignment,
    acs_local_ssi_medicaid_take_up_signal_gate,
    with_acs_local_ssi_medicaid_take_up,
    with_recorded_acs_local_ssi_medicaid_take_up,
)
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

SSI = "takes_up_ssi_if_eligible"
MEDICAID = "takes_up_medicaid_if_eligible"
TAG = spine_column("person")
STATES = (6, 36, 48)
#: Stub engine outputs ride the frame so the pre-pass view carries them.
UNCAPPED = "stub_uncapped_ssi"
ELIGIBLE = "stub_medicaid_eligible"


def _frame(*, n_donor: int = 400, n_acs_households: int = 6000, seed: int = 11):
    """Donor persons alone in their households, ACS households of two.

    Donor rows carry both take-up flags; ACS rows carry them missing, with
    ``SERIALNO``/``SPORDER`` keys, ``ssi_reported`` (blank below 15), ``HINS4``
    and the stub engine outputs the pre-pass callables return.
    """

    rng = np.random.default_rng(seed)
    donor_households = np.arange(1, n_donor + 1)
    acs_households = np.arange(n_donor + 1, n_donor + n_acs_households + 1)
    households = np.concatenate([donor_households, acs_households])
    person_household = np.concatenate([donor_households, np.repeat(acs_households, 2)])
    n = len(person_household)
    ids = np.arange(1, n + 1)
    acs = np.arange(n) >= n_donor
    person = pd.DataFrame({"person_id": ids, "person_household_id": person_household})
    tables = {"person": person}
    for entity in US_SCHEMA.entities:
        if entity in ("person", "household"):
            continue
        person[f"person_{entity}_id"] = ids
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    household_acs = households > n_donor
    tables["household"] = pd.DataFrame(
        {
            "household_id": households,
            spine_column("household"): np.where(
                household_acs, ACS_2024_1YR_SPINE, ASEC_PUF_DONOR_SPINE
            ),
            "SERIALNO": pd.Series(
                [f"2024HU{i:07d}" for i in households], dtype=object
            ).where(household_acs, np.nan),
            "state_fips": rng.choice(STATES, len(households)),
        }
    )
    person[TAG] = np.where(acs, ACS_2024_1YR_SPINE, ASEC_PUF_DONOR_SPINE)
    age = rng.integers(0, 91, n).astype(float)
    person["age"] = age
    person["SPORDER"] = np.concatenate(
        [np.full(n_donor, np.nan), np.tile([1.0, 2.0], n_acs_households)]
    )
    reporter = acs & (age >= 15) & (rng.random(n) < 0.05)
    person["ssi_reported"] = np.where(
        acs & (age >= 15), np.where(reporter, 9_000.0, 0.0), np.nan
    )
    candidate_rate = np.select([age < 18, age < 65], [0.05, 0.3], 0.4)
    candidate = (reporter & (rng.random(n) < 0.9)) | (rng.random(n) < candidate_rate)
    person[UNCAPPED] = np.where(candidate, rng.uniform(100, 900, n), 0.0)
    coverage = np.where(rng.random(n) < 0.15, 1.0, 2.0)
    coverage[rng.random(n) < 0.01] = np.nan
    person["HINS4"] = np.where(acs, coverage, np.nan)
    anchored = person["HINS4"].eq(1).to_numpy()
    person[ELIGIBLE] = np.where(anchored, rng.random(n) < 0.9, rng.random(n) < 0.35)
    position = np.arange(n)
    person[SSI] = pd.Series(position % 10 == 0, dtype=object).where(~acs, np.nan)
    person[MEDICAID] = pd.Series(position % 3 == 0, dtype=object).where(~acs, np.nan)
    weights = rng.uniform(0.5, 2.0, len(households))
    return Frame(
        tables, US_SCHEMA, {"household": Weights(weights, WeightKind.CALIBRATED)}
    )


def _with_person(frame: Frame, person: pd.DataFrame) -> Frame:
    return Frame(
        {
            entity: person if entity == "person" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )


def _masks(frame: Frame) -> dict[str, np.ndarray]:
    person = frame.table("person")
    age = person["age"].to_numpy()
    return {
        "acs": person[TAG].eq(ACS_2024_1YR_SPINE).to_numpy(),
        "weights": np.asarray(frame.resolve_weights("person").values),
        "under_18": age < 18,
        "18_64": (age >= 18) & (age <= 64),
        "65_plus": age >= 65,
        "state": frame.broadcast("state_fips").to_numpy(),
        "candidate": person[UNCAPPED].to_numpy() > 0,
        "eligible": person[ELIGIBLE].to_numpy(dtype=bool),
    }


def _donor_pooled(frame: Frame) -> tuple[dict[str, float], dict[int, float]]:
    """The donor rows' pooled SSI recipients per band and Medicaid enrollees
    per state: stored flag x stub candidacy or eligibility x frame weight."""

    masks = _masks(frame)
    person = frame.table("person")
    donor, weights = ~masks["acs"], masks["weights"]
    ssi = person[SSI].where(donor, False).astype(bool).to_numpy()
    medicaid = person[MEDICAID].where(donor, False).astype(bool).to_numpy()
    bands = {
        band: float(weights[donor & masks[band] & masks["candidate"] & ssi].sum())
        for band in ("under_18", "18_64", "65_plus")
    }
    states = {
        state: float(
            weights[
                donor & (masks["state"] == state) & masks["eligible"] & medicaid
            ].sum()
        )
        for state in np.unique(masks["state"])
    }
    return bands, states


def _targets(
    frame: Frame, *, ssi_fraction: float = 0.5, medicaid_fraction: float = 0.6
) -> tuple[dict[str, float], pd.DataFrame]:
    """Counts whose donor residual is a fraction of the ACS rows' capacity.

    Each count is the donor rows' pooled contribution plus that fraction, so
    the ACS residual is reachable. The under-18 count is far above the ACS
    rows' capacity, so that fenced band saturates, as it does on the donor.
    """

    masks = _masks(frame)
    acs, weights = masks["acs"], masks["weights"]
    donor_ssi, donor_medicaid = _donor_pooled(frame)
    ssi = {"under_18": 1_000_000.0}
    for band in ("18_64", "65_plus"):
        capacity = weights[acs & masks[band] & masks["candidate"]].sum()
        ssi[band] = float(donor_ssi[band] + ssi_fraction * capacity)
    rows = []
    for state in STATES:
        eligible = weights[acs & (masks["state"] == state) & masks["eligible"]].sum()
        rows.append(
            {
                "state_fips": state,
                "target": float(donor_medicaid[state] + medicaid_fraction * eligible),
            }
        )
    return ssi, pd.DataFrame(rows)


def _stubs(calls: list | None = None):
    def uncapped(view: Frame) -> np.ndarray:
        if calls is not None:
            calls.append(("uncapped_ssi", view.table("person").copy()))
        return view.table("person")[UNCAPPED].to_numpy()

    def eligible(view: Frame) -> np.ndarray:
        if calls is not None:
            calls.append(("is_medicaid_eligible", view.table("person").copy()))
        return view.table("person")[ELIGIBLE].to_numpy()

    return uncapped, eligible


def _assign(frame: Frame, *, seed: int = 5, calls: list | None = None, **targets):
    ssi, medicaid = _targets(frame, **targets)
    return _assign_to(frame, ssi, medicaid, seed=seed, calls=calls)


def _assign_to(
    frame: Frame,
    ssi: dict[str, float],
    medicaid: pd.DataFrame,
    *,
    seed: int = 5,
    calls: list | None = None,
):
    uncapped, eligible = _stubs(calls)
    return with_acs_local_ssi_medicaid_take_up(
        frame,
        seed=seed,
        ssi_band_targets=ssi,
        medicaid_state_targets=medicaid,
        uncapped_ssi=uncapped,
        medicaid_eligibility=eligible,
    )


def _rebuilt(
    frame: Frame,
    *,
    person: pd.DataFrame,
    household: pd.DataFrame,
    household_weights: np.ndarray,
) -> Frame:
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"], tables["household"] = person, household
    return Frame(
        tables,
        frame.schema,
        {"household": Weights(household_weights, WeightKind.CALIBRATED)},
    )


#: The count in María's #1060 review example.
EXAMPLE_COUNT = 100.0
#: SSA band counts for the example frame: 18-64 is the review example; the
#: donor delivers 10 of the 65+ count and 5 of the under-18 count.
EXAMPLE_SSI = {"under_18": 20.0, "18_64": EXAMPLE_COUNT, "65_plus": 40.0}


def _example_frame(*, reporters: float = 0.0, hins4: float = 0.0) -> Frame:
    """María's #1060 example on a pooled frame, one state.

    ``acs_share`` is 0.5: a donor person weighed 0.2 before pooling and 0.1
    after it. ACS persons (two per household) weigh 0.02 each and outnumber
    donor persons (one per household) ten to one, so the ACS rows hold 2/3 of
    every age band's and the state's pooled person weight, not 1/2. Both
    spines have the same age mix: two working-age persons per aged person and
    per child. The donor's 500 working-age recipients (candidate with the
    flag) delivered its count of 100 on its own weight and deliver 50 pooled;
    its 500 Medicaid enrollees (eligible with the flag) likewise. Donor
    candidates without the flag, and flagged non-candidates, deliver nothing.
    40% of ACS persons are SSI candidates and, independently, 40% are
    Medicaid-eligible. ``reporters`` is the share of ACS working-age
    candidates who report SSIP; ``hins4`` the share of ACS persons, eligible
    or not, with ``HINS4 == 1``.
    """

    n_donor, n_acs_households = 2_000, 10_000
    frame = _frame(n_donor=n_donor, n_acs_households=n_acs_households, seed=3)
    rng = np.random.default_rng(3)
    person = frame.table("person").copy()
    household = frame.table("household").copy()
    n = len(person)
    acs = person[TAG].eq(ACS_2024_1YR_SPINE).to_numpy()
    position = np.arange(n)
    age = np.select([position % 4 == 2, position % 4 == 3], [70.0, 10.0], 40.0)
    rank = np.zeros(n, dtype=np.int64)
    for value in (10.0, 40.0, 70.0):
        members = np.flatnonzero(~acs & (age == value))
        rank[members] = np.arange(len(members))
    donor_candidate = ~acs & (
        ((age == 40.0) & (rank < 700))
        | ((age == 70.0) & (rank < 100))
        | ((age == 10.0) & (rank < 50))
    )
    donor_ssi = ~acs & (
        ((age == 40.0) & ((rank < 500) | ((rank >= 700) & (rank < 800))))
        | ((age == 70.0) & (rank < 100))
        | ((age == 10.0) & (rank < 50))
    )
    candidate = np.where(acs, rng.random(n) < 0.4, donor_candidate)
    eligible = np.where(acs, rng.random(n) < 0.4, position < 800)
    donor_medicaid = ~acs & ((position < 500) | ((position >= 800) & (position < 1000)))
    reporter = acs & candidate & (age == 40.0) & (rng.random(n) < reporters)
    person["age"] = age
    person["ssi_reported"] = np.where(
        acs & (age >= 15), np.where(reporter, 9_000.0, 0.0), np.nan
    )
    person[UNCAPPED] = np.where(candidate, 500.0, 0.0)
    person[ELIGIBLE] = eligible
    person["HINS4"] = np.where(acs, np.where(rng.random(n) < hins4, 1.0, 2.0), np.nan)
    person[SSI] = pd.Series(donor_ssi, dtype=object).where(~acs, np.nan)
    person[MEDICAID] = pd.Series(donor_medicaid, dtype=object).where(~acs, np.nan)
    household["state_fips"] = 6
    household_weights = np.where(household["household_id"] > n_donor, 0.02, 0.1)
    return _rebuilt(
        frame,
        person=person,
        household=household,
        household_weights=household_weights.astype(np.float64),
    )


def _example_medicaid(count: float = EXAMPLE_COUNT) -> pd.DataFrame:
    return pd.DataFrame({"state_fips": [6], "target": [count]})


def _band(receipt: dict, key: str) -> dict:
    return next(row for row in receipt["ssi"]["bands"] if row["age_band"] == key)


def _state(receipt: dict, code: str = "06") -> dict:
    return next(
        row for row in receipt["medicaid"]["state_targets"] if row["state_fips"] == code
    )


@pytest.fixture(scope="module")
def assigned():
    frame = _frame()
    result, receipt = _assign(frame)
    return frame, result, receipt


def test_fills_only_missing_acs_cells_and_keeps_donor_rows(assigned) -> None:
    frame, result, receipt = assigned
    before = frame.table("person")
    after = result.table("person")
    donor = before[TAG].eq(ASEC_PUF_DONOR_SPINE).to_numpy()
    for column in (SSI, MEDICAID):
        assert after[column].notna().all()
        assert after[column].dtype == bool
        assert (
            after.loc[donor, column].to_numpy(dtype=bool)
            == before.loc[donor, column].to_numpy(dtype=bool)
        ).all()
    acs_rows = int((~donor).sum())
    assert receipt["acs_persons"] == acs_rows
    for program in ("ssi", "medicaid"):
        assert receipt[program]["filled_rows"] == acs_rows
        assert receipt[program]["unfilled_acs_rows"] == 0
    # The other tables and the weights are the input's.
    for entity in ("household", "spm_unit"):
        pd.testing.assert_frame_equal(result.table(entity), frame.table(entity))
    assert np.array_equal(
        result.weights_for("household").values, frame.weights_for("household").values
    )


def test_ssi_reporters_take_up_and_enforced_bands_hit_their_donor_residuals(
    assigned,
) -> None:
    frame, result, receipt = assigned
    masks = _masks(frame)
    acs, weights = masks["acs"], masks["weights"]
    after = result.table("person")
    ssi = after[SSI].to_numpy(dtype=bool)
    reporters = acs & (np.nan_to_num(after["ssi_reported"].to_numpy()) > 0)
    assert reporters.any()
    assert ssi[reporters].all()
    targets, _ = _targets(frame)
    donor_ssi, _ = _donor_pooled(frame)
    bands = {row["age_band"]: row for row in receipt["ssi"]["bands"]}
    assert list(bands) == ["under_18", "18_64", "65_plus"]
    for band in ("18_64", "65_plus"):
        row = bands[band]
        in_band = masks[band]
        assert donor_ssi[band] > 0
        assert row["donor_recipient_weight"] == pytest.approx(donor_ssi[band])
        assert row["residual_target"] == pytest.approx(targets[band] - donor_ssi[band])
        delivered = weights[acs & in_band & masks["candidate"] & ssi].sum()
        assert row["selected_recipient_weight"] == pytest.approx(delivered)
        assert abs(row["relative_error"]) <= ACS_LOCAL_SSI_BAND_RELATIVE_TOLERANCE
        # Donor plus ACS carries the SSA count before calibration moves weight.
        pooled = weights[in_band & masks["candidate"] & ssi].sum()
        assert row["pooled_recipient_weight"] == pytest.approx(pooled)
        assert pooled == pytest.approx(
            targets[band], rel=ACS_LOCAL_SSI_BAND_RELATIVE_TOLERANCE
        )
        assert 0.0 < row["assignment_prior"] < 1.0
        assert not row["saturated"]
        assert row["capacity_shortfall"] == 0.0
        # Non-candidates draw too, at the same prior, as on the donor.
        open_non_candidates = acs & in_band & ~masks["candidate"] & ~reporters
        drawn_share = ssi[open_non_candidates].mean()
        assert drawn_share == pytest.approx(row["assignment_prior"], abs=0.05)
    # The fenced under-18 band saturates here: every ACS candidate takes up
    # and the shortfall is recorded.
    under = bands["under_18"]
    assert under["saturated"] is True
    assert under["all_candidates_assigned"] is True
    assert ssi[acs & masks["under_18"] & masks["candidate"]].all()
    assert under["capacity_shortfall"] == pytest.approx(
        under["residual_target"] - under["candidate_capacity"]
    )


def test_medicaid_anchors_take_up_and_states_hit_their_donor_residuals(
    assigned,
) -> None:
    frame, result, receipt = assigned
    masks = _masks(frame)
    acs, weights = masks["acs"], masks["weights"]
    after = result.table("person")
    medicaid = after[MEDICAID].to_numpy(dtype=bool)
    anchors = acs & after["HINS4"].eq(1).to_numpy()
    assert anchors.any()
    # No state's anchors exceed its residual here, so none is thinned.
    assert medicaid[anchors].all()
    assert receipt["medicaid"]["hins4_thinned_rows"] == 0
    assert receipt["medicaid"]["thinned_states"] == []
    # A blank HINS4 reads as no coverage, never as an anchor.
    assert receipt["medicaid"]["blank_coverage_rows_as_false"] > 0
    _, counts = _targets(frame)
    _, donor_medicaid = _donor_pooled(frame)
    for state, count in zip(counts["state_fips"], counts["target"], strict=True):
        in_state = masks["state"] == state
        row = _state(receipt, f"{state:02d}")
        assert donor_medicaid[state] > 0
        assert row["donor_enrolled_weight"] == pytest.approx(donor_medicaid[state])
        assert row["residual_target"] == pytest.approx(count - donor_medicaid[state])
        assert row["hins4_keep_probability"] == 1.0
        assert row["anchor_excess"] == 0.0
        enrolled = weights[acs & in_state & masks["eligible"] & medicaid].sum()
        assert enrolled == pytest.approx(row["residual_target"], rel=0.02)
        assert row["acs_enrolled_weight"] == pytest.approx(enrolled)
        # Donor plus ACS carries the CMS count before calibration.
        pooled = weights[in_state & masks["eligible"] & medicaid].sum()
        assert row["pooled_enrolled_weight"] == pytest.approx(pooled)
        assert pooled == pytest.approx(count, rel=0.02)
        # Not the universal take-up landmine.
        assert enrolled < weights[acs & in_state & masks["eligible"]].sum()
    diagnostics = receipt["medicaid"]["diagnostics"]
    assert diagnostics["anchor"].startswith("ACS HINS4 == 1")
    assert ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE in diagnostics["issues"]
    assert diagnostics["saturated_states"] == []


def test_review_example_pools_to_the_count_not_above_it() -> None:
    """María's #1060 example: a count of 100, ``acs_share`` 0.5, and ACS rows
    holding 2/3 of the band's pooled person weight.

    The donor delivers 50 pooled. The share-scaled target sought
    100 x 2/3 = 66.7 from the ACS rows and pooled 116.7; the residual seeks
    50, and donor plus ACS pools to 100, for SSI by band and for Medicaid by
    state.
    """

    frame = _example_frame()
    masks = _masks(frame)
    acs, weights = masks["acs"], masks["weights"]
    band = masks["18_64"]
    assert weights[acs & band].sum() / weights[band].sum() == pytest.approx(2 / 3)
    assert weights[acs].sum() / weights.sum() == pytest.approx(2 / 3)
    share_scaled_pool = 50.0 + EXAMPLE_COUNT * 2 / 3
    assert share_scaled_pool > EXAMPLE_COUNT * (
        1 + ACS_LOCAL_SSI_BAND_RELATIVE_TOLERANCE
    )

    result, receipt = _assign_to(frame, EXAMPLE_SSI, _example_medicaid())
    after = result.table("person")
    ssi = after[SSI].to_numpy(dtype=bool)
    row = _band(receipt, "18_64")
    assert row["donor_recipient_weight"] == pytest.approx(50.0)
    assert row["donor_share_of_target"] == pytest.approx(0.5)
    assert row["residual_target"] == pytest.approx(50.0)
    pooled = weights[band & masks["candidate"] & ssi].sum()
    assert row["pooled_recipient_weight"] == pytest.approx(pooled)
    assert pooled == pytest.approx(
        EXAMPLE_COUNT, rel=ACS_LOCAL_SSI_BAND_RELATIVE_TOLERANCE
    )
    aged = _band(receipt, "65_plus")
    assert aged["donor_recipient_weight"] == pytest.approx(10.0)
    assert weights[masks["65_plus"] & masks["candidate"] & ssi].sum() == (
        pytest.approx(40.0, rel=ACS_LOCAL_SSI_BAND_RELATIVE_TOLERANCE)
    )

    medicaid = after[MEDICAID].to_numpy(dtype=bool)
    state = _state(receipt)
    assert state["donor_enrolled_weight"] == pytest.approx(50.0)
    assert state["residual_target"] == pytest.approx(50.0)
    pooled = weights[masks["eligible"] & medicaid].sum()
    assert state["pooled_enrolled_weight"] == pytest.approx(pooled)
    assert pooled == pytest.approx(EXAMPLE_COUNT, rel=0.02)

    gate = acs_local_ssi_medicaid_take_up_signal_gate(result, receipt=receipt)
    assert gate.passed, gate.failures


def test_capacity_below_the_residual_assigns_every_eligible_acs_person() -> None:
    """Where the ACS capacity cannot reach the residual, every candidate (SSI)
    or eligible person (Medicaid) takes up and the shortfall is recorded."""

    frame = _example_frame()
    masks = _masks(frame)
    acs, weights = masks["acs"], masks["weights"]
    band = acs & masks["18_64"]
    capacity = weights[band & masks["candidate"]].sum()
    eligible = weights[acs & masks["eligible"]].sum()
    ssi_counts = {**EXAMPLE_SSI, "18_64": 250.0}
    result, receipt = _assign_to(frame, ssi_counts, _example_medicaid(300.0))
    after = result.table("person")
    ssi = after[SSI].to_numpy(dtype=bool)
    row = _band(receipt, "18_64")
    assert row["residual_target"] == pytest.approx(200.0)
    assert row["saturated"] is True
    assert row["all_candidates_assigned"] is True
    assert ssi[band & masks["candidate"]].all()
    # No reporters, so the reporter-rate fallback gives non-candidates zero.
    assert not ssi[band & ~masks["candidate"]].any()
    assert row["selected_recipient_weight"] == pytest.approx(capacity)
    assert row["capacity_shortfall"] == pytest.approx(200.0 - capacity)
    assert row["pooled_recipient_weight"] == pytest.approx(50.0 + capacity)

    medicaid = after[MEDICAID].to_numpy(dtype=bool)
    state = _state(receipt)
    assert state["residual_target"] == pytest.approx(250.0)
    assert medicaid[acs & masks["eligible"]].all()
    assert state["acs_enrolled_weight"] == pytest.approx(eligible)
    assert state["capacity_shortfall"] == pytest.approx(250.0 - eligible)
    assert receipt["medicaid"]["diagnostics"]["saturated_states"] == ["06"]

    gate = acs_local_ssi_medicaid_take_up_signal_gate(result, receipt=receipt)
    statuses = {row["age_band"]: row["status"] for row in gate.details["ssi_bands"]}
    assert statuses["18_64"] == "saturated"
    # The saturated band and state are reported, not failed. The donor
    # stage's fill rate is 1 in a saturated state for eligible and off-domain
    # persons alike, so this one-state frame's ACS Medicaid column is
    # constant, which the landmine check refuses; a real build has 51 states.
    assert gate.failures == (
        f"{ACS_2024_1YR_SPINE}: {MEDICAID} is constant; universal take-up is "
        "the engine-default landmine.",
    )


def test_anchor_excess_is_recorded_and_hins4_anchors_are_thinned() -> None:
    """SSIP reporters stay a floor above the residual. HINS4 anchors whose
    eligible weight exceeds a state's residual are kept with probability
    residual / anchored weight, so donor plus ACS stays at the count. A band
    the donor already meets gets a residual of zero."""

    frame = _example_frame(reporters=0.75, hins4=0.6)
    masks = _masks(frame)
    acs, weights = masks["acs"], masks["weights"]
    ssi_counts = {**EXAMPLE_SSI, "65_plus": 5.0}
    result, receipt = _assign_to(frame, ssi_counts, _example_medicaid())
    after = result.table("person")
    ssi = after[SSI].to_numpy(dtype=bool)
    reporters = acs & (np.nan_to_num(after["ssi_reported"].to_numpy()) > 0)
    band = acs & masks["18_64"]
    floor = weights[band & reporters].sum()
    assert floor > 50.0
    row = _band(receipt, "18_64")
    assert row["reporter_candidate_floor"] == pytest.approx(floor)
    assert row["anchor_excess"] == pytest.approx(floor - 50.0)
    assert ssi[reporters].all()
    assert not ssi[band & ~reporters].any()
    aged = _band(receipt, "65_plus")
    assert aged["residual_target"] == 0.0
    assert aged["donor_excess"] == pytest.approx(5.0)
    assert not ssi[acs & masks["65_plus"]].any()

    medicaid = after[MEDICAID].to_numpy(dtype=bool)
    hins4 = acs & after["HINS4"].eq(1).to_numpy()
    state = _state(receipt)
    anchored = weights[hins4 & masks["eligible"]].sum()
    assert anchored > 50.0
    assert state["anchored_eligible_weight"] == pytest.approx(anchored)
    assert state["anchor_excess"] == pytest.approx(anchored - 50.0)
    assert state["hins4_keep_probability"] == pytest.approx(50.0 / anchored)
    assert receipt["medicaid"]["thinned_states"] == ["06"]
    dropped = int((hins4 & ~medicaid).sum())
    assert 0 < dropped <= receipt["medicaid"]["hins4_thinned_rows"]
    assert receipt["medicaid"]["hins4_thinned_rows"] == state["hins4_thinned_rows"]
    # About (1 - keep) of HINS4 records are released, eligible or not.
    assert state["hins4_thinned_rows"] / int(hins4.sum()) == pytest.approx(
        1 - state["hins4_keep_probability"], abs=0.02
    )
    pooled = weights[masks["eligible"] & medicaid].sum()
    assert pooled == pytest.approx(EXAMPLE_COUNT, rel=0.02)

    gate = acs_local_ssi_medicaid_take_up_signal_gate(result, receipt=receipt)
    assert gate.passed, gate.failures
    statuses = {row["age_band"]: row["status"] for row in gate.details["ssi_bands"]}
    assert statuses == {
        "under_18": "fenced",
        "18_64": "anchor_excess",
        "65_plus": "donor_meets_count",
    }
    anchors = gate.details["acs_anchors"][MEDICAID]
    assert anchors["thinned"] == dropped
    assert anchors["dropped"] == 0

    # A receipt that does not thin the state cannot explain the drops.
    unthinned = copy.deepcopy(receipt)
    _state(unthinned)["hins4_keep_probability"] = 1.0
    gate = acs_local_ssi_medicaid_take_up_signal_gate(result, receipt=unthinned)
    assert any("by HINS4 do not carry" in failure for failure in gate.failures)


def test_gate_refuses_targets_that_are_not_the_donor_residual() -> None:
    """A receipt carrying the old share-scaled targets fails the gate."""

    frame = _example_frame()
    result, receipt = _assign_to(frame, EXAMPLE_SSI, _example_medicaid())
    scaled = copy.deepcopy(receipt)
    _band(scaled, "18_64")["residual_target"] = EXAMPLE_COUNT * 2 / 3
    _state(scaled)["residual_target"] = EXAMPLE_COUNT * 2 / 3
    gate = acs_local_ssi_medicaid_take_up_signal_gate(result, receipt=scaled)
    assert any(
        "SSI band '18_64' targets 67, not the donor residual 50" in failure
        for failure in gate.failures
    )
    assert any(
        "Medicaid state 06 targets 67, not the donor residual 50" in failure
        for failure in gate.failures
    )


def test_engine_prepass_reads_donor_then_acs_households() -> None:
    """The donor households are evaluated first, with their stored flags, to
    measure the donor's pooled contribution. Then both passes see only ACS
    persons; missing take-up reads True, and the Medicaid pass sees the SSI
    flags the stage just assigned."""

    frame = _frame(n_donor=40, n_acs_households=600)
    calls: list = []
    result, _ = _assign(frame, calls=calls)
    assert [name for name, _ in calls] == [
        "uncapped_ssi",
        "is_medicaid_eligible",
        "uncapped_ssi",
        "is_medicaid_eligible",
    ]
    person = frame.table("person")
    acs = person[TAG].eq(ACS_2024_1YR_SPINE).to_numpy()
    donor_views = [view for _, view in calls[:2]]
    for view in donor_views:
        assert np.array_equal(
            view["person_id"].to_numpy(), person.loc[~acs, "person_id"].to_numpy()
        )
        assert view[TAG].eq(ASEC_PUF_DONOR_SPINE).all()
        for column in (SSI, MEDICAID):
            assert np.array_equal(
                view[column].to_numpy(dtype=bool),
                person.loc[~acs, column].to_numpy(dtype=bool),
            )
    acs_ids = person.loc[acs, "person_id"].to_numpy()
    first, second = (view for _, view in calls[2:])
    for view in (first, second):
        assert np.array_equal(view["person_id"].to_numpy(), acs_ids)
        assert view[TAG].eq(ACS_2024_1YR_SPINE).all()
        assert view[MEDICAID].astype(bool).all()
    assert first[SSI].astype(bool).all()
    final_ssi = result.table("person").loc[acs, SSI].to_numpy(dtype=bool)
    assert np.array_equal(second[SSI].to_numpy(dtype=bool), final_ssi)
    assert not final_ssi.all()


def test_refuses_incomplete_donor_take_up_cells() -> None:
    frame = _frame(n_donor=40, n_acs_households=300)
    person = frame.table("person").copy()
    person.loc[0, SSI] = np.nan
    with pytest.raises(ValueError, match="donor SSI/Medicaid take-up cell"):
        _assign(_with_person(frame, person))


def test_assignment_is_deterministic_in_the_seed() -> None:
    frame = _frame(n_donor=40, n_acs_households=1500)
    first, first_receipt = _assign(frame, seed=5)
    again, again_receipt = _assign(frame, seed=5)
    other, other_receipt = _assign(frame, seed=6)
    assert first_receipt["assigned_sha256"] == again_receipt["assigned_sha256"]
    pd.testing.assert_frame_equal(first.table("person"), again.table("person"))
    assert first_receipt["assigned_sha256"] != other_receipt["assigned_sha256"]


def test_stored_acs_cells_are_kept() -> None:
    frame = _frame(n_donor=40, n_acs_households=1500)
    person = frame.table("person").copy()
    acs = person[TAG].eq(ACS_2024_1YR_SPINE).to_numpy()
    stored = np.flatnonzero(acs)[:10]
    person[SSI] = person[SSI].astype(object)
    person[MEDICAID] = person[MEDICAID].astype(object)
    person.loc[stored, SSI] = [True, False] * 5
    person.loc[stored, MEDICAID] = [False, True] * 5
    frame = _with_person(frame, person)
    result, receipt = _assign(frame)
    after = result.table("person")
    assert after.loc[stored, SSI].tolist() == [True, False] * 5
    assert after.loc[stored, MEDICAID].tolist() == [False, True] * 5
    assert receipt["ssi"]["preserved_acs_rows"] == 10
    assert receipt["medicaid"]["preserved_acs_rows"] == 10
    assert receipt["ssi"]["filled_rows"] == int(acs.sum()) - 10


def test_gate_passes_on_the_assigned_frame(assigned) -> None:
    _, result, receipt = assigned
    gate = acs_local_ssi_medicaid_take_up_signal_gate(result, receipt=receipt)
    assert gate.name == ACS_LOCAL_SSI_MEDICAID_TAKE_UP_GATE_NAME
    assert gate.passed, gate.failures
    statuses = {row["age_band"]: row["status"] for row in gate.details["ssi_bands"]}
    assert statuses == {
        "under_18": "fenced",
        "18_64": "within_tolerance",
        "65_plus": "within_tolerance",
    }
    anchors = gate.details["acs_anchors"]
    assert anchors[SSI]["dropped"] == 0
    assert anchors[MEDICAID]["dropped"] == 0


def test_gate_fails_on_default_filled_constant_or_unanchored_acs_rows(
    assigned,
) -> None:
    _, result, receipt = assigned
    person = result.table("person")
    acs = person[TAG].eq(ACS_2024_1YR_SPINE).to_numpy()

    missing = person.copy()
    missing[SSI] = missing[SSI].astype(object)
    missing.loc[np.flatnonzero(acs)[:3], SSI] = np.nan
    gate = acs_local_ssi_medicaid_take_up_signal_gate(
        _with_person(result, missing), receipt=receipt
    )
    assert any("3 missing row(s)" in failure for failure in gate.failures)

    universal = person.copy()
    universal.loc[acs, MEDICAID] = True
    gate = acs_local_ssi_medicaid_take_up_signal_gate(
        _with_person(result, universal), receipt=receipt
    )
    assert any(
        f"{ACS_2024_1YR_SPINE}: {MEDICAID} is constant" in failure
        for failure in gate.failures
    )

    dropped = person.copy()
    reporter = np.flatnonzero(
        acs & (np.nan_to_num(person["ssi_reported"].to_numpy()) > 0)
    )[0]
    anchor = np.flatnonzero(acs & person["HINS4"].eq(1).to_numpy())[0]
    dropped.loc[reporter, SSI] = False
    dropped.loc[anchor, MEDICAID] = False
    gate = acs_local_ssi_medicaid_take_up_signal_gate(
        _with_person(result, dropped), receipt=receipt
    )
    assert any("by ssi_reported do not carry" in failure for failure in gate.failures)
    assert any("by HINS4 do not carry" in failure for failure in gate.failures)


@pytest.mark.parametrize(
    "edit",
    ["absent", "wrong_issue", "unfilled", "no_digest", "rows", "bands"],
)
def test_gate_fails_without_a_current_receipt(assigned, edit) -> None:
    _, result, receipt = assigned
    receipt = copy.deepcopy(receipt)
    if edit == "absent":
        receipt = None
    elif edit == "wrong_issue":
        receipt["issue"] = "microcosm#1019"
    elif edit == "unfilled":
        receipt["medicaid"]["unfilled_acs_rows"] = 4
    elif edit == "no_digest":
        del receipt["assigned_sha256"]
    elif edit == "rows":
        receipt["acs_persons"] += 1
    elif edit == "bands":
        receipt["ssi"]["bands"] = receipt["ssi"]["bands"][1:]
    gate = acs_local_ssi_medicaid_take_up_signal_gate(result, receipt=receipt)
    assert not gate.passed
    assert gate.failures


def test_gate_grades_only_count_truthful_enforced_ssi_bands(assigned) -> None:
    _, result, receipt = assigned
    missed = copy.deepcopy(receipt)
    band = missed["ssi"]["bands"][1]
    band["selected_recipient_weight"] = band["residual_target"] * 1.2
    gate = acs_local_ssi_medicaid_take_up_signal_gate(result, receipt=missed)
    assert any("SSI band '18_64'" in failure for failure in gate.failures)

    # A saturated band's miss of its residual is reported, not failed, when
    # it assigned every candidate; so is any miss in the fenced under-18 band.
    saturated = copy.deepcopy(receipt)
    band = saturated["ssi"]["bands"][1]
    band["candidate_capacity"] = band["residual_target"] * 0.9
    band["selected_recipient_weight"] = band["candidate_capacity"]
    short = copy.deepcopy(saturated)
    short["ssi"]["bands"][1]["selected_recipient_weight"] *= 0.9
    gate = acs_local_ssi_medicaid_take_up_signal_gate(result, receipt=short)
    assert any("every candidate must take up" in failure for failure in gate.failures)
    fenced = copy.deepcopy(receipt)
    fenced["ssi"]["bands"][0]["selected_recipient_weight"] = 0.0
    for edited, key, status in (
        (saturated, "18_64", "saturated"),
        (fenced, "under_18", "fenced"),
    ):
        gate = acs_local_ssi_medicaid_take_up_signal_gate(result, receipt=edited)
        assert gate.passed, gate.failures
        statuses = {row["age_band"]: row["status"] for row in gate.details["ssi_bands"]}
        assert statuses[key] == status


def test_gate_fails_a_medicaid_state_miss(assigned) -> None:
    _, result, receipt = assigned
    missed = copy.deepcopy(receipt)
    state = missed["medicaid"]["diagnostics"]["states"][0]
    state["enrolled_weight"] = state["target"] * 1.5
    gate = acs_local_ssi_medicaid_take_up_signal_gate(result, receipt=missed)
    assert any(
        failure.startswith("medicaid: state") and "misses the CMS count" in failure
        for failure in gate.failures
    )


def test_recorded_assignment_reproduces_the_frame(assigned) -> None:
    frame, result, receipt = assigned
    recorded = acs_local_ssi_medicaid_take_up_assignment(result)
    applied = with_recorded_acs_local_ssi_medicaid_take_up(
        frame, recorded, assigned_sha256=receipt["assigned_sha256"]
    )
    for column in (SSI, MEDICAID):
        assert np.array_equal(
            applied.table("person")[column].to_numpy(dtype=bool),
            result.table("person")[column].to_numpy(dtype=bool),
        )
    flipped = recorded.copy()
    flipped.loc[0, SSI] = not flipped.loc[0, SSI]
    with pytest.raises(ValueError, match="does not reproduce its digest"):
        with_recorded_acs_local_ssi_medicaid_take_up(
            frame, flipped, assigned_sha256=receipt["assigned_sha256"]
        )
    with pytest.raises(ValueError, match="in order"):
        with_recorded_acs_local_ssi_medicaid_take_up(
            frame,
            recorded.iloc[::-1].reset_index(drop=True),
            assigned_sha256=receipt["assigned_sha256"],
        )
    with pytest.raises(ValueError, match="complete boolean"):
        acs_local_ssi_medicaid_take_up_assignment(frame)


def _break(frame: Frame, how: str) -> Frame:
    person = frame.table("person").copy()
    acs = person[TAG].eq(ACS_2024_1YR_SPINE).to_numpy()
    first_adult = np.flatnonzero(acs & (person["age"].to_numpy() >= 15))[0]
    if how == "no_hins4":
        person = person.drop(columns="HINS4")
    elif how == "blank_ssip_at_15":
        person.loc[first_adult, "ssi_reported"] = np.nan
    elif how == "mixed_household":
        # The second member of the first ACS household carries a donor tag.
        person.loc[np.flatnonzero(acs)[1], TAG] = ASEC_PUF_DONOR_SPINE
    elif how == "repeated_key":
        person.loc[np.flatnonzero(acs)[1], "SPORDER"] = 1.0
    return _with_person(frame, person)


@pytest.mark.parametrize(
    "how,match",
    [
        ("no_hins4", "HINS4"),
        ("blank_ssip_at_15", "no ssi_reported"),
        ("mixed_household", "share a household with ACS"),
        ("repeated_key", "draw key"),
    ],
)
def test_refuses_inputs_it_cannot_assign_from(how, match) -> None:
    frame = _break(_frame(n_donor=40, n_acs_households=300), how)
    with pytest.raises(ValueError, match=match):
        _assign(frame)


def test_refuses_misaligned_engine_values() -> None:
    frame = _frame(n_donor=40, n_acs_households=300)
    ssi, medicaid = _targets(frame)
    with pytest.raises(ValueError, match="uncapped_ssi value"):
        with_acs_local_ssi_medicaid_take_up(
            frame,
            seed=1,
            ssi_band_targets=ssi,
            medicaid_state_targets=medicaid,
            uncapped_ssi=lambda view: np.ones(3),
            medicaid_eligibility=lambda view: np.ones(view.n("person"), dtype=bool),
        )
    with pytest.raises(ValueError, match="state enrollment targets"):
        with_acs_local_ssi_medicaid_take_up(
            frame,
            seed=1,
            ssi_band_targets=ssi,
            medicaid_state_targets=medicaid.iloc[:0],
            uncapped_ssi=lambda view: np.zeros(view.n("person")),
            medicaid_eligibility=lambda view: np.ones(view.n("person"), dtype=bool),
        )
