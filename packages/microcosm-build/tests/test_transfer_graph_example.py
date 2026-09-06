"""Synthetic transfer exercises real model reuse without learning its holdout."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.transfer_example import (
    ANNUAL_BASIS,
    CONVERSION_FACTOR,
    default_targets,
    make_synthetic_inputs,
    run_transfer_example,
)
from microcosm.frame import Frame, Weights


def _changed_frame(frame, *, column=None, delta=0, change_weights=False):
    tables = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    weights = frame.weights_for("household")
    if change_weights:
        values = weights.values.copy()
        values[0] *= 4
        weights = Weights(values, weights.kind)
    if column is not None:
        tables["household"].loc[0, column] += delta
    return Frame(tables, frame.schema, {"household": weights}, frame.strata)


def _assert_branch_hits(result, destination):
    for stage in ("source", "apply", "annualize", "calibrate", "evaluate"):
        assert result.manifest.node(f"{destination}.{stage}").hit


def test_one_real_fit_is_shared_by_two_destinations_and_warm_runs(
    tmp_path, monkeypatch
):
    import microcosm.fit.graph_train as models

    calls = []
    original = models.fit_qrf

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(models, "fit_qrf", counted)
    inputs = make_synthetic_inputs()
    cold = run_transfer_example(tmp_path, inputs=inputs)
    warm = run_transfer_example(tmp_path, inputs=inputs)
    assert len(calls) == 1
    assert not cold.manifest.node("donor.fit").hit
    assert warm.manifest.node("donor.fit").hit
    assert cold.report["model"] == warm.report["model"]
    for destination in ("alpha", "beta"):
        _assert_branch_hits(warm, destination)
        report = cold.report["destinations"][destination]
        assert report["model_artifact_key"] == cold.report["model"]["artifact_key"]
        assert report["calibration_passed"]
        assert report["heldout_passed"]
        assert report["mass_before"] > 0 and report["mass_after"] > 0
        assert report["effective_sample_size"] > 0
        assert report["weight_kind"] == "calibrated"
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["scope"] == "synthetic_engineering"
    assert (tmp_path / "run_manifest.json").is_file()
    assert (tmp_path / "graph.json").is_file()


def test_target_edit_only_recomputes_affected_calibration_and_evaluation(tmp_path):
    inputs = make_synthetic_inputs()
    cold = run_transfer_example(tmp_path, inputs=inputs)
    targets = default_targets()
    targets["alpha"] = replace(
        targets["alpha"], size_total=targets["alpha"].size_total * 1.05
    )
    changed = run_transfer_example(tmp_path, inputs=inputs, targets=targets)
    assert changed.manifest.node("donor.fit").hit
    for stage in ("source", "apply", "annualize"):
        assert changed.manifest.node(f"alpha.{stage}").hit
    for stage in ("calibrate", "evaluate"):
        assert not changed.manifest.node(f"alpha.{stage}").hit
    _assert_branch_hits(changed, "beta")
    assert changed.report["model"] == cold.report["model"]


def test_recipient_edit_preserves_model_and_other_destination(tmp_path):
    inputs = make_synthetic_inputs()
    cold = run_transfer_example(tmp_path, inputs=inputs)
    recipients = dict(inputs.recipients)
    recipients["alpha"] = _changed_frame(
        recipients["alpha"], column="household_size", delta=1
    )
    changed = run_transfer_example(
        tmp_path, inputs=replace(inputs, recipients=recipients)
    )
    assert changed.manifest.node("donor.fit").hit
    for stage in ("source", "apply", "annualize", "calibrate", "evaluate"):
        assert not changed.manifest.node(f"alpha.{stage}").hit
    _assert_branch_hits(changed, "beta")
    assert changed.report["model"] == cold.report["model"]


@pytest.mark.parametrize("change_weights", [False, True])
def test_donor_values_or_weights_invalidate_both_applications(tmp_path, change_weights):
    inputs = make_synthetic_inputs()
    cold = run_transfer_example(tmp_path, inputs=inputs)
    donor = _changed_frame(
        inputs.donor,
        column=None if change_weights else "monthly_consumption_minor",
        delta=3000,
        change_weights=change_weights,
    )
    changed = run_transfer_example(tmp_path, inputs=replace(inputs, donor=donor))
    assert not changed.manifest.node("donor.fit").hit
    assert (
        changed.report["model"]["artifact_key"] != cold.report["model"]["artifact_key"]
    )
    for destination in ("alpha", "beta"):
        assert changed.manifest.node(f"{destination}.source").hit
        assert not changed.manifest.node(f"{destination}.apply").hit


def test_conversion_is_exactly_the_recorded_float_operation(tmp_path):
    result = run_transfer_example(tmp_path)
    for destination in ("alpha", "beta"):
        table = result.manifest.population(f"{destination}.calibrate").table(
            "household"
        )
        expected = table["monthly_consumption_minor"].to_numpy() * CONVERSION_FACTOR
        actual = table["annual_consumption"].to_numpy()
        assert expected.tobytes() == actual.tobytes()
        receipt = result.manifest.node(f"{destination}.annualize").receipt
        assert receipt["source_convention"] == "monthly synthetic minor units"
        assert receipt["factor"] == 12 / 100
        assert receipt["prepared"]["basis"]["unit"] == "base_currency"
        assert receipt["prepared"]["basis"]["currency"] == "XXX"
        assert set(table) >= {"household_id", "household_size", "annual_consumption"}
        assert result.report["destinations"][destination]["weight_kind"] == "calibrated"


def test_incompatible_destination_basis_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="basis"):
        run_transfer_example(tmp_path, basis=replace(ANNUAL_BASIS, currency="USD"))


def test_holdout_only_edit_changes_evaluation_and_can_fail_despite_good_calibration(
    tmp_path,
):
    inputs = make_synthetic_inputs()
    cold = run_transfer_example(tmp_path, inputs=inputs)
    references = dict(inputs.references)
    reference = references["alpha"]
    tables = {
        entity: reference.table(entity).copy(deep=True) for entity in reference.entities
    }
    tables["household"]["annual_consumption"] *= 10
    references["alpha"] = Frame(
        tables,
        reference.schema,
        {"household": reference.weights_for("household")},
        reference.strata,
    )
    changed = run_transfer_example(
        tmp_path, inputs=replace(inputs, references=references)
    )
    assert changed.manifest.node("donor.fit").hit
    for stage in ("source", "apply", "annualize", "calibrate"):
        assert changed.manifest.node(f"alpha.{stage}").hit
    assert not changed.manifest.node("alpha.evaluate").hit
    _assert_branch_hits(changed, "beta")
    report = changed.report["destinations"]["alpha"]
    assert report["calibration_passed"]
    assert not report["heldout_passed"]
    assert report["heldout_relative_error"] > 0.5
    pd.testing.assert_frame_equal(
        cold.manifest.population("alpha.calibrate").table("household"),
        changed.manifest.population("alpha.calibrate").table("household"),
    )
    np.testing.assert_array_equal(
        cold.manifest.population("alpha.calibrate").weights_for("household").values,
        changed.manifest.population("alpha.calibrate").weights_for("household").values,
    )
