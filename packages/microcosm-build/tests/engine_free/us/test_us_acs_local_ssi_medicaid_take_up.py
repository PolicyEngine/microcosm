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


def _targets(
    frame: Frame, *, ssi_fraction: float = 0.5, medicaid_fraction: float = 0.6
) -> tuple[dict[str, float], pd.DataFrame]:
    """National counts whose ACS-scaled share is a fraction of ACS capacity.

    The under-18 count is far above the ACS rows' capacity, so that fenced
    band saturates, as it does on the donor.
    """

    masks = _masks(frame)
    acs, weights = masks["acs"], masks["weights"]
    ssi = {"under_18": 1_000_000.0}
    for band in ("18_64", "65_plus"):
        in_band = masks[band]
        share = weights[acs & in_band].sum() / weights[in_band].sum()
        capacity = weights[acs & in_band & masks["candidate"]].sum()
        ssi[band] = float(ssi_fraction * capacity / share)
    rows = []
    for state in STATES:
        in_state = masks["state"] == state
        share = weights[acs & in_state].sum() / weights[in_state].sum()
        eligible = weights[acs & in_state & masks["eligible"]].sum()
        rows.append(
            {"state_fips": state, "target": float(medicaid_fraction * eligible / share)}
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
    uncapped, eligible = _stubs(calls)
    return with_acs_local_ssi_medicaid_take_up(
        frame,
        seed=seed,
        ssi_band_targets=ssi,
        medicaid_state_targets=medicaid,
        uncapped_ssi=uncapped,
        medicaid_eligibility=eligible,
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


def test_ssi_reporters_take_up_and_enforced_bands_hit_scaled_counts(assigned) -> None:
    frame, result, receipt = assigned
    masks = _masks(frame)
    acs, weights = masks["acs"], masks["weights"]
    after = result.table("person")
    ssi = after[SSI].to_numpy(dtype=bool)
    reporters = acs & (np.nan_to_num(after["ssi_reported"].to_numpy()) > 0)
    assert reporters.any()
    assert ssi[reporters].all()
    targets, _ = _targets(frame)
    bands = {row["age_band"]: row for row in receipt["ssi"]["bands"]}
    assert list(bands) == ["under_18", "18_64", "65_plus"]
    for band in ("18_64", "65_plus"):
        row = bands[band]
        in_band = masks[band]
        share = weights[acs & in_band].sum() / weights[in_band].sum()
        assert row["acs_weight_share"] == pytest.approx(share)
        assert row["scaled_target"] == pytest.approx(targets[band] * share)
        delivered = weights[acs & in_band & masks["candidate"] & ssi].sum()
        assert row["selected_recipient_weight"] == pytest.approx(delivered)
        assert abs(row["relative_error"]) <= ACS_LOCAL_SSI_BAND_RELATIVE_TOLERANCE
        assert 0.0 < row["assignment_prior"] < 1.0
        assert not row["saturated"]
        # Non-candidates draw too, at the same prior, as on the donor.
        open_non_candidates = acs & in_band & ~masks["candidate"] & ~reporters
        drawn_share = ssi[open_non_candidates].mean()
        assert drawn_share == pytest.approx(row["assignment_prior"], abs=0.05)
    # The fenced under-18 band still draws, and saturates here.
    assert bands["under_18"]["saturated"] is True
    assert ssi[acs & masks["under_18"] & ~reporters].any()


def test_medicaid_anchors_take_up_and_states_hit_scaled_counts(assigned) -> None:
    frame, result, receipt = assigned
    masks = _masks(frame)
    acs, weights = masks["acs"], masks["weights"]
    after = result.table("person")
    medicaid = after[MEDICAID].to_numpy(dtype=bool)
    anchors = acs & after["HINS4"].eq(1).to_numpy()
    assert anchors.any()
    assert medicaid[anchors].all()
    # A blank HINS4 reads as no coverage, never as an anchor.
    assert receipt["medicaid"]["blank_coverage_rows_as_false"] > 0
    _, national = _targets(frame)
    scaled = {row["state_fips"]: row for row in receipt["medicaid"]["state_targets"]}
    for state, target in zip(national["state_fips"], national["target"], strict=True):
        in_state = masks["state"] == state
        share = weights[acs & in_state].sum() / weights[in_state].sum()
        row = scaled[f"{state:02d}"]
        assert row["acs_weight_share"] == pytest.approx(share)
        assert row["scaled_target"] == pytest.approx(target * share)
        enrolled = weights[acs & in_state & masks["eligible"] & medicaid].sum()
        assert enrolled == pytest.approx(row["scaled_target"], rel=0.02)
        # Not the universal take-up landmine.
        assert enrolled < weights[acs & in_state & masks["eligible"]].sum()
    diagnostics = receipt["medicaid"]["diagnostics"]
    assert diagnostics["anchor"].startswith("ACS HINS4 == 1")
    assert ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE in diagnostics["issues"]
    assert diagnostics["saturated_states"] == []


def test_engine_prepass_reads_acs_households_with_missing_flags_true() -> None:
    """Both passes see only ACS persons; missing take-up reads True, and the
    Medicaid pass sees the SSI flags the stage just assigned."""

    frame = _frame(n_donor=40, n_acs_households=600)
    calls: list = []
    result, _ = _assign(frame, calls=calls)
    assert [name for name, _ in calls] == ["uncapped_ssi", "is_medicaid_eligible"]
    acs = frame.table("person")[TAG].eq(ACS_2024_1YR_SPINE).to_numpy()
    acs_ids = frame.table("person").loc[acs, "person_id"].to_numpy()
    first, second = (view for _, view in calls)
    for view in (first, second):
        assert np.array_equal(view["person_id"].to_numpy(), acs_ids)
        assert view[TAG].eq(ACS_2024_1YR_SPINE).all()
        assert view[MEDICAID].astype(bool).all()
    assert first[SSI].astype(bool).all()
    final_ssi = result.table("person").loc[acs, SSI].to_numpy(dtype=bool)
    assert np.array_equal(second[SSI].to_numpy(dtype=bool), final_ssi)
    assert not final_ssi.all()


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
    band["selected_recipient_weight"] = band["scaled_target"] * 1.2
    gate = acs_local_ssi_medicaid_take_up_signal_gate(result, receipt=missed)
    assert any("SSI band '18_64'" in failure for failure in gate.failures)

    # The same miss in a saturated band, or in the fenced under-18 band, is
    # reported, not failed: the prior could not reach the count there.
    saturated = copy.deepcopy(missed)
    band = saturated["ssi"]["bands"][1]
    band["candidate_capacity"] = band["scaled_target"] * 0.9
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
