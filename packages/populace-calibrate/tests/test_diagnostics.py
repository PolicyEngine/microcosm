"""The diagnostics artifact: a calibration's evidence ships with it.

Behavioral contracts: the payload carries every per-target row, the whole
loss trajectory, and every skipped target with its reason; it round-trips
through strict JSON (no NaN/Infinity tokens); and the file writer produces
the artifact a release directory publishes.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from populace.calibrate import (
    CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
    Target,
    TargetDiagnostic,
    TargetRegistry,
    TargetSet,
    TargetSpec,
    calibrate,
    calibrate_l0_refit,
    diagnostics_payload,
    past_cap_census,
    score_targets,
    write_calibration_diagnostics,
)


def _result(feasible_frame, *, with_skip: bool = False, epochs: int = 120):
    frame, truths = feasible_frame()
    targets = [
        Target(
            name="population",
            entity="household",
            value=truths["population"] * 1.2,
            measure="household_count",
        ),
        Target(
            name="income",
            entity="household",
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
    assert payload["target_surface"]["n_targets"] == len(result.diagnostics)
    assert len(payload["target_surface"]["sha256"]) == 64
    assert payload["target_surface"]["constraint_matrix"]["rows"] == len(
        result.diagnostics
    )
    assert len(payload["loss_trajectory"]) == 120
    assert len(payload["targets"]) == len(result.diagnostics)
    assert payload["n_records"] == result.weights.shape[0]
    assert payload["final_loss"] == result.final_loss
    assert payload["fraction_within_10pct"] == result.fraction_within_10pct
    assert payload["options"]["epochs"] == 120
    assert payload["options"]["seed"] == 0

    income = next(row for row in payload["targets"] if row["name"].startswith("income"))
    assert income["target_name"] == "income"
    assert income["entity"] == "household"
    assert income["measure"] == {"kind": "column", "name": "income"}
    assert income["metadata"] == {}
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


def test_payload_can_carry_target_registry_identity(feasible_frame) -> None:
    frame, truths = feasible_frame()
    registry = TargetRegistry(
        (
            TargetSpec(
                name="population",
                entity="household",
                measure="household_count",
                value=truths["population"] * 1.2,
                period=2024,
                source="Census PEP 2024",
                family="census_population",
            ),
            TargetSpec(
                name="income",
                entity="household",
                measure="income",
                value=truths["income"] * 1.2,
                period=2024,
                source="IRS SOI 2024",
                family="irs_soi",
            ),
        ),
        country="us",
    )
    result = calibrate(frame, registry.to_target_set(), epochs=120, seed=0)
    payload = diagnostics_payload(result, target_registry=registry)

    assert payload["target_registry"] == {
        "country": "us",
        "version": registry.version,
        "n_specs": 2,
    }
    income = next(row for row in payload["targets"] if row["target_name"] == "income")
    assert income["period"] == 2024
    assert income["source"] == "IRS SOI 2024"
    assert income["registry"]["family"] == "irs_soi"


def test_payload_reports_weight_concentration(feasible_frame) -> None:
    """The accuracy-vs-spread coordinates ship with every calibration."""
    result = _result(feasible_frame)
    payload = diagnostics_payload(result)

    assert payload["effective_sample_size"] == result.effective_sample_size
    assert payload["realized_max_weight_ratio"] == result.realized_max_weight_ratio
    assert payload["top_1pct_weight_share"] == result.top_1pct_weight_share
    assert payload["effective_sample_size"] > 0.0


def test_payload_accepts_l0_refit_result(feasible_frame) -> None:
    """Regression: the default sparse build hands an L0RefitResult to the writer.

    ``fraction_within_10pct`` was never delegated to the refit stage, so the
    payload raised AttributeError on the production default (L0+refit) path.
    """
    frame, truths = feasible_frame()
    targets = TargetSet(
        (
            Target(
                name="income",
                entity="household",
                value=truths["income"] * 1.2,
                measure="income",
            ),
            Target(
                name="population",
                entity="household",
                value=truths["population"],
                measure="household_count",
            ),
        )
    )
    result = calibrate_l0_refit(
        frame, targets, epochs=120, seed=0, l0_lambda=0.003, mass="conserve"
    )
    payload = diagnostics_payload(result)

    assert payload["options"]["post_l0_refit"] is True
    assert payload["fraction_within_10pct"] == result.refit.fraction_within_10pct
    assert payload["effective_sample_size"] == result.refit.effective_sample_size
    # The census reads the cap through the L0RefitResult's merged options.
    assert payload["past_cap_census"]["cap"] == 10.0
    json.dumps(payload, allow_nan=False)


def _census_stub(rows, options):
    """A duck-typed result for the census: (name, target, initial, final) rows."""
    diagnostics = tuple(
        TargetDiagnostic(
            name=name,
            target=target,
            initial_estimate=initial,
            final_estimate=final,
            relative_error=(final - target) / target if target else final - target,
            within_tolerance=None,
        )
        for name, target, initial, final in rows
    )
    return SimpleNamespace(diagnostics=diagnostics, options=options)


def test_past_cap_census_classifies_every_row_class() -> None:
    """Exact classification against the loss's own scale rule max(|t|, 1)."""
    result = _census_stub(
        [
            ("inside@2024", 100.0, 90.0, 100.0),  # never past
            ("escaped@2024", 100.0, 250.0, 105.0),  # 1.5 -> 0.05
            ("frozen@2024", 100.0, 350.0, 320.0),  # 2.5 -> 2.2
            ("pushed@2024", 100.0, 150.0, 240.0),  # 0.5 -> 1.4
            ("zero_target@2024", 0.0, 0.0, 1.2),  # scale 1: 0.0 -> 1.2
            # |target| < 1 uses scale 1, not the raw relative error (which
            # would be 0.8 -> 3.2 against the 0.5 target).
            ("small_target@2024", 0.5, 0.9, 2.1),  # 0.4 -> 1.6
            ("boundary@2024", 100.0, 100.0, 200.0),  # 0.0 -> exactly 1.0
        ],
        options={"target_loss_scales": {"cap": 1.0}},
    )
    census = past_cap_census(result)

    assert census["cap"] == 1.0
    assert census["n_targets"] == 7
    assert census["initial_past_cap"] == 2  # escaped, frozen
    assert census["final_past_cap"] == 5  # frozen + the four pushed out
    assert census["escaped"] == 1
    assert census["frozen"] == 1
    assert census["pushed_out"] == 4
    # Worst final miss first; the boundary row (exactly at the cap) counts.
    assert [row["name"] for row in census["pushed_out_rows"]] == [
        "small_target@2024",
        "pushed@2024",
        "zero_target@2024",
        "boundary@2024",
    ]
    pushed = census["pushed_out_rows"][1]
    assert pushed["init_rel"] == 0.5
    assert pushed["final_rel"] == 1.4
    # The identities every census must satisfy.
    assert census["initial_past_cap"] == census["escaped"] + census["frozen"]
    assert census["final_past_cap"] == census["frozen"] + census["pushed_out"]


def test_past_cap_census_without_a_recorded_cap_is_none() -> None:
    result = _census_stub([("row@2024", 100.0, 100.0, 100.0)], options={})
    assert past_cap_census(result) is None


def test_past_cap_census_reads_score_targets_options_shape(feasible_frame) -> None:
    """score_targets records a top-level target_loss_cap; a score has no motion."""
    frame, truths = feasible_frame()
    targets = TargetSet(
        (
            Target(
                name="income",
                entity="household",
                # The current weights overshoot this tiny target by far more
                # than the cap, so the row scores past-cap (frozen).
                value=truths["income"] * 0.001,
                measure="income",
            ),
            Target(
                name="population",
                entity="household",
                value=truths["population"],
                measure="household_count",
            ),
        )
    )
    result = score_targets(frame, targets, target_loss_cap=1.0)
    census = past_cap_census(result)

    assert census["cap"] == 1.0
    # Score-only results carry initial == final estimates: nothing moves.
    assert census["escaped"] == 0
    assert census["pushed_out"] == 0
    assert census["frozen"] == 1
    assert census["initial_past_cap"] == census["final_past_cap"] == 1
    payload = diagnostics_payload(result)
    assert payload["past_cap_census"] == census


def test_payload_census_from_a_real_solve(feasible_frame) -> None:
    """A hopeless row stays past the cap and the payload censuses it."""
    frame, truths = feasible_frame()
    targets = TargetSet(
        (
            Target(
                name="income",
                entity="household",
                # ~1000x below the starting estimate: past the cap at
                # initialization and unreachable in a short solve.
                value=truths["income"] * 0.001,
                measure="income",
            ),
            Target(
                name="population",
                entity="household",
                value=truths["population"] * 1.05,
                measure="household_count",
            ),
        )
    )
    result = calibrate(frame, targets, epochs=30, seed=0, target_loss_cap=1.0)
    payload = diagnostics_payload(result)
    census = payload["past_cap_census"]

    assert census["cap"] == 1.0
    assert census["n_targets"] == 2
    assert census["frozen"] >= 1
    assert census["initial_past_cap"] == census["escaped"] + census["frozen"]
    assert census["final_past_cap"] == census["frozen"] + census["pushed_out"]
    assert census == past_cap_census(result)
    json.dumps(payload, allow_nan=False)
