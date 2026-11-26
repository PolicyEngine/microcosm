"""The release tool's wrappers around the household pass reuse.

``tools/build_us_fiscal_refresh_release.py`` reuses its pre-calibration
December SSI and Medicaid eligibility passes after calibration through
``_reuse_household_pass``, and refuses the Medicaid reuse for a frame whose
Medicaid claiming links cross households (``_medicaid_claim_crossing``).
These tests use a stand-in for the pass and a record made with
``microcosm.build.household_pass_reuse`` directly, so they need no engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.household_pass_reuse import (
    HouseholdPassInputs,
    PassInputColumn,
    record_household_pass,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_batched_target_materialization import (
    _load_builder_module,
)

_INPUTS = HouseholdPassInputs(
    variable="is_medicaid_eligible",
    columns=(
        PassInputColumn("income", "person", default=0.0, absent_reads_default=True),
    ),
)


@pytest.fixture(scope="module")
def builder():
    return _load_builder_module()


def _frame(*, income=(10.0, 0.0, 20.0, 30.0), claims=None) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3, 4], dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2, 3], dtype="int64"),
            "person_tax_unit_id": np.asarray([10, 11, 20, 30], dtype="int64"),
            "person_spm_unit_id": np.asarray([100, 100, 200, 300], dtype="int64"),
            "person_family_id": np.asarray([1000, 1000, 2000, 3000], dtype="int64"),
            "person_marital_unit_id": np.asarray([1, 2, 3, 4], dtype="int64"),
            "income": np.asarray(income, dtype=np.float64),
        }
    )
    if claims is not None:
        person["medicaid_claiming_tax_unit_id"] = np.asarray(claims, dtype="int64")
    tables = {"person": person}
    for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
        ids = np.unique(person[f"person_{entity}_id"].to_numpy())
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    tables["household"] = pd.DataFrame(
        {"household_id": np.asarray([1, 2, 3], dtype="int64")}
    )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(3), WeightKind.DESIGN)},
    )


class _StandInPass:
    def __init__(self) -> None:
        self.household_sets: list[list[int]] = []

    def __call__(self, frame: Frame) -> np.ndarray:
        self.household_sets.append(
            sorted(frame.table("household")["household_id"].tolist())
        )
        return frame.table("person")["income"].to_numpy() > 15.0


class _RecordingTelemetry:
    def __init__(self, *, fail: bool = False) -> None:
        self.events: list[tuple[str, str, dict]] = []
        self.fail = fail

    def stage(self, stage, *, message=None, status="running", **details):
        if self.fail:
            raise OSError("telemetry write failed")
        self.events.append((stage, message, details))


def test_claims_inside_households_allow_reuse(builder) -> None:
    assert builder._medicaid_claim_crossing(_frame()) is None
    # Person 2 is claimed by tax unit 10, in their own household.
    assert builder._medicaid_claim_crossing(_frame(claims=(0, 10, 0, 0))) is None


def test_a_claim_across_households_refuses_reuse(builder) -> None:
    # Person 3, in household 2, is claimed by tax unit 10 in household 1.
    reason = builder._medicaid_claim_crossing(_frame(claims=(0, 0, 10, 0)))

    assert reason is not None and "own household" in reason


def test_reuse_reports_its_stage_before_and_after(builder) -> None:
    frame = _frame()
    stand_in = _StandInPass()
    record = record_household_pass(frame, stand_in(frame), _INPUTS)
    changed = _frame(income=(10.0, 0.0, 20.0, 5.0))
    telemetry = _RecordingTelemetry()

    reuse = builder._reuse_household_pass(
        record,
        changed,
        recompute=stand_in,
        maximum_microsim_batch_size=1,
        description="Medicaid eligibility",
        telemetry=telemetry,
        telemetry_stage="medicaid_final_diagnostics",
    )

    assert reuse.values.tobytes() == stand_in(changed).tobytes()
    assert [stage for stage, _, _ in telemetry.events] == [
        "medicaid_final_diagnostics",
        "medicaid_final_diagnostics",
    ]
    assert telemetry.events[1][2]["household_pass_reuse"] == {
        "variable": "is_medicaid_eligible",
        "reused_households": 2,
        "recomputed_households": 1,
        "checked_households": 1,
        "check_failed": False,
        "reuse_refused": None,
    }


def test_a_telemetry_failure_does_not_stop_the_reuse(builder, capsys) -> None:
    frame = _frame()
    stand_in = _StandInPass()
    record = record_household_pass(frame, stand_in(frame), _INPUTS)

    reuse = builder._reuse_household_pass(
        record,
        frame,
        recompute=stand_in,
        maximum_microsim_batch_size=2,
        description="Medicaid eligibility",
        telemetry=_RecordingTelemetry(fail=True),
    )

    assert reuse.values.tobytes() == stand_in(frame).tobytes()
    assert "did not reach telemetry" in capsys.readouterr().out


def test_an_unbatched_build_reruns_in_one_selection(builder) -> None:
    frame = _frame()
    stand_in = _StandInPass()
    record = record_household_pass(frame, stand_in(frame), _INPUTS)
    changed = _frame(income=(11.0, 0.0, 21.0, 31.0))

    reuse = builder._reuse_household_pass(
        record,
        changed,
        recompute=stand_in,
        maximum_microsim_batch_size=0,
        description="Medicaid eligibility",
    )

    assert stand_in.household_sets[1:] == [[1, 2, 3]]
    assert reuse.recomputed_households == 3


def test_a_refused_reuse_runs_the_whole_export_frame(builder, capsys) -> None:
    frame = _frame()
    stand_in = _StandInPass()
    record = record_household_pass(frame, stand_in(frame), _INPUTS)

    reuse = builder._reuse_household_pass(
        record,
        frame,
        recompute=stand_in,
        maximum_microsim_batch_size=2,
        description="Medicaid eligibility",
        reuse_refused="a claim crosses households",
    )

    assert reuse.reuse_refused == "a claim crosses households"
    assert (reuse.reused_households, reuse.recomputed_households) == (0, 3)
    assert stand_in.household_sets[1:] == [[1, 2, 3]]
    assert "without reuse: a claim crosses households" in capsys.readouterr().out


def test_one_family_garbage_collection_collects_once_and_restores(
    builder, monkeypatch
) -> None:
    collections: list[int] = []
    monkeypatch.setattr(builder.gc, "collect", lambda *args: collections.append(1))

    with builder._one_family_garbage_collection():
        builder._collect_family_garbage()
        with builder._one_family_garbage_collection():
            builder._collect_family_garbage()
        assert collections == []
    assert collections == [1]
    assert builder._FAMILY_GARBAGE_DEFERRED is False

    with pytest.raises(RuntimeError):
        with builder._one_family_garbage_collection():
            raise RuntimeError("pass failed")
    assert collections == [1, 1]
    assert builder._FAMILY_GARBAGE_DEFERRED is False
    builder._collect_family_garbage()
    assert collections == [1, 1, 1]


def test_a_reuse_runs_one_full_collection_for_all_its_batches(
    builder, monkeypatch
) -> None:
    """Each pass call ends with a full collection; a reuse of several batches
    runs one, as a full pass does."""

    collections: list[int] = []
    monkeypatch.setattr(builder.gc, "collect", lambda *args: collections.append(1))
    frame = _frame()
    stand_in = _StandInPass()
    record = record_household_pass(frame, stand_in(frame), _INPUTS)
    # Household 1 changes; household 2 is the one checked household.
    changed = _frame(income=(11.0, 0.0, 20.0, 30.0))

    def pass_with_collection(run_frame):
        values = stand_in(run_frame)
        builder._collect_family_garbage()
        return values

    builder._reuse_household_pass(
        record,
        changed,
        recompute=pass_with_collection,
        maximum_microsim_batch_size=1,
        description="Medicaid eligibility",
    )

    # Two pass calls, one per batch of one household, and one collection.
    assert stand_in.household_sets[1:] == [[1], [2]]
    assert collections == [1]
