"""Stored evidence is independent of the transform instance that produced it."""

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from microcosm.build.observation import ObservedTransform
from microcosm.build.stage_evidence import (
    decode_stage_evidence,
    encode_stage_evidence,
    release_stage_results,
    snapshot_stage_evidence,
)


def test_snapshot_preserves_checkpoint_replay_and_fit_weight_records():
    class Transform:
        sampling = {"fraction": 0.1, "seed": 42}
        fit_weight_records = (SimpleNamespace(fit_name="wealth", weight_kind="design"),)

        def checkpoint_metadata(self):
            return {"evidence": {"rows": 4}, "replay_payload": {"target": [1, 2]}}

    payload = encode_stage_evidence(snapshot_stage_evidence("wealth", Transform()))
    recovered = decode_stage_evidence(payload, stage="wealth")
    assert recovered["evidence"] == {"rows": 4}
    assert recovered["checkpoint_metadata"]["replay_payload"] == {"target": [1, 2]}
    assert recovered["fit_weight_records"] == [
        {"fit_name": "wealth", "weight_kind": "design"}
    ]
    assert recovered["sampling"] == {"fraction": 0.1, "seed": 42}


def test_unreadable_fit_records_remain_an_explicit_failed_audit_input():
    class Transform:
        @property
        def fit_weight_records(self):
            raise RuntimeError("fit did not expose its records")

    payload = snapshot_stage_evidence("wealth", Transform())
    assert payload["fit_weight_records"] == []
    assert payload["fit_weight_records_status"] == "unreadable"


def test_stage_evidence_refuses_wrong_identity_and_nonportable_data():
    payload = encode_stage_evidence(snapshot_stage_evidence("one", object()))
    with pytest.raises(ValueError, match="stage identity"):
        decode_stage_evidence(payload, stage="two")
    mutated = json.loads(payload)
    mutated["schema_version"] = 2
    with pytest.raises(ValueError, match="schema"):
        decode_stage_evidence(json.dumps(mutated).encode(), stage="one")
    with pytest.raises(ValueError):
        encode_stage_evidence({"value": float("nan")})


@dataclass(frozen=True)
class _FrozenStage:
    """Keeps its result as the UK stages do, in a frozen field set per run."""

    last_result: object | None = field(default=None, init=False)

    def __call__(self, frame):
        result = SimpleNamespace(frame=frame, rows=len(frame.person))
        object.__setattr__(self, "last_result", result)
        return frame

    def checkpoint_metadata(self):
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": {"rows": self.last_result.rows}}


class _SourceBuilt:
    """A graph source transform: it keeps the stage it built in its own
    ``__dict__`` and proxies every other read to it."""

    def __init__(self, transform):
        self.transform = transform

    def run_with_sources(self, frame, sources):
        return self.transform(frame)

    def __getattr__(self, name):
        return getattr(self.__dict__["transform"], name)


def test_release_clears_a_frozen_result_through_the_run_wrappers(small_frame):
    stage = _FrozenStage()
    source = _SourceBuilt(stage)
    wrapped = ObservedTransform(
        source, stage_id="wealth", produced_column_count=1, observer=None
    )
    wrapped.run_with_sources(small_frame, {})
    payload = encode_stage_evidence(snapshot_stage_evidence("wealth", wrapped))

    release_stage_results(wrapped)

    assert stage.last_result is None
    assert decode_stage_evidence(payload, stage="wealth")["evidence"] == {"rows": 4}
    # Neither wrapper gained an attribute of its own, so both still proxy the
    # stage, and a later run's result reads through them.
    assert "last_result" not in vars(wrapped)
    assert "last_result" not in vars(source)
    assert source.transform is stage
    wrapped.run_with_sources(small_frame, {})
    assert wrapped.last_result is stage.last_result is not None
    # The stage itself stays, so a read after the release fails loudly.
    release_stage_results(wrapped)
    with pytest.raises(RuntimeError, match="completed stage run"):
        wrapped.checkpoint_metadata()


def test_release_keeps_the_fit_weight_evidence_snapshotted_before_it():
    record = SimpleNamespace(fit_name="wealth", weight_kind="design")

    @dataclass(frozen=True)
    class ResultRecords:
        """Reads its records from its result, as the SPI income stage does."""

        last_result: object | None = field(default=None, init=False)

        @property
        def fit_weight_records(self):
            return () if self.last_result is None else self.last_result.records

    @dataclass
    class KeptRecords:
        """Keeps its records beside its result, as the WAS stage does."""

        last_fit_weight_records: tuple = ()
        last_result: object | None = None

        @property
        def fit_weight_records(self):
            return tuple(self.last_fit_weight_records)

    from_result = ResultRecords()
    object.__setattr__(from_result, "last_result", SimpleNamespace(records=(record,)))
    beside = KeptRecords((record,), SimpleNamespace())
    for name, stage in (("from_result", from_result), ("beside", beside)):
        payload = encode_stage_evidence(snapshot_stage_evidence(name, stage))
        release_stage_results(stage)
        recovered = decode_stage_evidence(payload, stage=name)
        assert recovered["fit_weight_records"] == [
            {"fit_name": "wealth", "weight_kind": "design"}
        ]
        assert recovered["fit_weight_records_status"] == "present"
        assert stage.last_result is None
    # Records read from the result go with it, so only the snapshot taken
    # before the release carries them; records kept beside it stay.
    assert from_result.fit_weight_records == ()
    assert beside.fit_weight_records == (record,)


def test_release_writes_only_a_result_an_object_holds_itself():
    class NeverRan:
        last_result = None

    class Derived:
        @property
        def last_result(self):
            return "derived"

    class Loop:
        def __init__(self):
            self.transform = self
            self.last_result = SimpleNamespace()
            self.last_receipt = {"rows": 1}
            self.sampling = {"fraction": 1.0}

    release_stage_results(None)
    never_ran = NeverRan()
    release_stage_results(never_ran)
    assert "last_result" not in vars(never_ran)
    derived = Derived()
    release_stage_results(derived)
    assert derived.last_result == "derived"
    loop = Loop()
    release_stage_results(loop)
    assert loop.last_result is None
    assert (loop.last_receipt, loop.sampling) == ({"rows": 1}, {"fraction": 1.0})
