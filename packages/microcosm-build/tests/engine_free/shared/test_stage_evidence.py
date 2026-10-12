"""Stored evidence is independent of the transform instance that produced it."""

import json
from types import SimpleNamespace

import pytest

from microcosm.build.stage_evidence import (
    decode_stage_evidence,
    encode_stage_evidence,
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
