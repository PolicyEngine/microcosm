"""The diagnostics artifact: a calibration's evidence ships with it.

Behavioral contracts: the payload carries every per-target row, the whole
loss trajectory, and every skipped target with its reason; it round-trips
through strict JSON (no NaN/Infinity tokens); and the file writer produces
the artifact a release directory publishes.
"""

from __future__ import annotations

import json
from pathlib import Path

from populace.calibrate import (
    CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
    Target,
    TargetSet,
    calibrate,
    diagnostics_payload,
    write_calibration_diagnostics,
)


def _result(feasible_frame, *, with_skip: bool = False, epochs: int = 120):
    frame, truths = feasible_frame()
    targets = [
        Target(
            name="population",
            entity="household",
            aggregation="count",
            value=truths["population"] * 1.2,
        ),
        Target(
            name="income",
            entity="household",
            aggregation="sum",
            value=truths["income"] * 1.2,
            measure="income",
            tolerance=truths["income"],
        ),
    ]
    if with_skip:
        targets.append(
            Target(
                name="ghost",
                entity="household",
                aggregation="sum",
                value=1.0,
                measure="no_such_column",
            )
        )
    return calibrate(frame, TargetSet(tuple(targets)), epochs=epochs, seed=0)


def test_payload_carries_full_evidence(feasible_frame) -> None:
    result = _result(feasible_frame, epochs=120)
    payload = diagnostics_payload(result)

    assert payload["schema_version"] == CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION
    assert payload["weight_entity"] == "household"
    assert len(payload["loss_trajectory"]) == 120
    assert len(payload["targets"]) == len(result.diagnostics)
    assert payload["n_records"] == result.weights.shape[0]
    assert payload["final_loss"] == result.final_loss
    assert payload["fraction_within_10pct"] == result.fraction_within_10pct
    assert payload["options"]["epochs"] == 120
    assert payload["options"]["seed"] == 0

    income = next(
        row for row in payload["targets"] if row["name"].startswith("income")
    )
    assert income["initial_estimate"] is not None
    assert income["final_estimate"] is not None
    assert income["within_tolerance"] is True  # tolerance was a whole truth wide
    population = next(
        row for row in payload["targets"] if row["name"].startswith("population")
    )
    assert population["within_tolerance"] is None  # no tolerance declared


def test_skipped_targets_ship_with_their_reason(feasible_frame) -> None:
    result = _result(feasible_frame, with_skip=True)
    payload = diagnostics_payload(result)
    assert len(payload["skipped"]) == 1
    skip = payload["skipped"][0]
    assert skip["name"] == "ghost"
    assert "no_such_column" in skip["reason"]
    # The skip never leaks into the compiled target rows.
    assert all(not row["name"].startswith("ghost") for row in payload["targets"])


def test_payload_is_strict_json(feasible_frame) -> None:
    result = _result(feasible_frame)
    payload = diagnostics_payload(result)
    encoded = json.dumps(payload, allow_nan=False)  # raises on NaN/inf
    assert json.loads(encoded) == payload


def test_writer_round_trips(feasible_frame, tmp_path: Path) -> None:
    result = _result(feasible_frame)
    path = write_calibration_diagnostics(
        result, tmp_path / "calibration_diagnostics.json"
    )
    loaded = json.loads(path.read_text())
    assert loaded == diagnostics_payload(result)
    assert loaded["schema_version"] == CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION
