"""The UK local scorer compares one frozen surface on both sides."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.holdout import summarize_rotations
from microcosm.build.uk_runtime.local_doctrine import UK_LOCAL_TARGET_LOSS_CAP
from microcosm.calibrate import TargetRegistry, TargetSpec, relative_error_loss


def _load_scorer():
    path = Path(__file__).resolve().parents[3] / "tools/score_uk_local_candidate.py"
    spec = importlib.util.spec_from_file_location("score_uk_local_candidate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _case():
    specs = (
        TargetSpec(
            name="households@E1",
            entity="household",
            value=10.0,
            measure="households",
            period=2025,
            source="fixture",
            family="census_households",
            metadata={"ledger_geography_id": "E1"},
        ),
        TargetSpec(
            name="income@W1",
            entity="household",
            value=100.0,
            measure="income",
            period=2025,
            source="fixture",
            family="hmrc_income",
            metadata={"ledger_geography_id": "W1"},
        ),
    )
    registry = TargetRegistry(specs, country="uk")
    candidate = {
        "schema_version": 6,
        "targets": [
            {"name": specs[0].to_target().row_name, "final_estimate": 10.0},
            {"name": specs[1].to_target().row_name, "final_estimate": 110.0},
        ],
        "uk_diagnostics": {
            "rotated_holdout": {
                "report_only": True,
                "method": "rotated_folds",
                "n_folds": 5,
                "seed": 20260529,
                # Summary closes over the folds, as summarize_rotations
                # produces it: mean 0.4, worst 0.6.
                "mean_holdout_loss": 0.4,
                "worst_holdout_loss": 0.6,
                "fold_losses": [0.2, 0.3, 0.4, 0.5, 0.6],
            }
        },
    }
    # Deliberately asymmetric across households: a positional pairing of these
    # two tables gives a different, plausible-looking answer than the join,
    # which is what the permutation regression below pins.
    weights = pd.DataFrame(
        {
            "household_id": ["h1", "h2"],
            "E1": [4.0, 2.0],
            "W1": [1.0, 3.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "household_id": ["h1", "h2"],
            "households": [2.0, 1.0],
            "income": [10.0, 20.0],
        }
    )
    return registry, candidate, weights, metrics


def test_local_scorer_reports_per_family_wins_and_the_measured_holdout() -> None:
    scorer = _load_scorer()
    registry, candidate, weights, metrics = _case()

    result = scorer.score_uk_local_candidate(
        candidate_diagnostics=candidate,
        incumbent_weights=weights,
        incumbent_metrics=metrics,
        target_registry=registry,
        expected_reference_count=2,
    )

    assert result["holdout_basis"] == "rotated_folds:n_folds=5:seed=20260529"
    assert result["candidate_holdout_loss"] == pytest.approx(0.4)
    assert result["candidate_holdout"]["worst_holdout_loss"] == pytest.approx(0.6)
    # The incumbent is never re-solved, so it has no holdout to compare, and
    # the head-to-head counters must say which surface they ran on.
    assert result["incumbent_holdout_loss"] is None
    assert result["incumbent_holdout_basis"] == "none_available_incumbent_not_resolved"
    assert result["loss"]["head_to_head_surface"] == "candidate_fitted_surface"
    # Both aggregates come from the canonical objective at the doctrine cap,
    # so the holdout beside them is on the same scale.
    assert result["loss"]["objective"] == "microcosm.calibrate.relative_error_loss"
    assert result["loss"]["target_loss_cap"] == UK_LOCAL_TARGET_LOSS_CAP
    assert result["candidate_fitted_surface_loss"] == pytest.approx(
        relative_error_loss(
            np.array([10.0, 110.0]),
            np.array([10.0, 100.0]),
            target_loss_cap=UK_LOCAL_TARGET_LOSS_CAP,
        )
    )
    assert result["candidate_target_wins"] == 1
    assert result["incumbent_target_wins"] == 0
    assert result["target_wins_by_family"] == {
        "census_households": {
            "candidate_target_wins": 0,
            "incumbent_target_wins": 0,
            "ties": 1,
        },
        "hmrc_income": {
            "candidate_target_wins": 1,
            "incumbent_target_wins": 0,
            "ties": 0,
        },
    }


def test_local_scorer_refuses_a_non_frozen_surface() -> None:
    scorer = _load_scorer()
    registry, candidate, weights, metrics = _case()

    with pytest.raises(ValueError, match="frozen active reference count"):
        scorer.score_uk_local_candidate(
            candidate_diagnostics=candidate,
            incumbent_weights=weights,
            incumbent_metrics=metrics,
            target_registry=registry,
        )


def test_local_scorer_joins_the_incumbent_on_household_id() -> None:
    """A reordered metrics file must score identically, not plausibly wrong."""

    scorer = _load_scorer()
    registry, candidate, weights, metrics = _case()

    straight = scorer.score_uk_local_candidate(
        candidate_diagnostics=candidate,
        incumbent_weights=weights,
        incumbent_metrics=metrics,
        target_registry=registry,
        expected_reference_count=2,
    )
    shuffled = scorer.score_uk_local_candidate(
        candidate_diagnostics=candidate,
        incumbent_weights=weights,
        incumbent_metrics=metrics.iloc[::-1].reset_index(drop=True),
        target_registry=registry,
        expected_reference_count=2,
    )

    assert straight["target_drift"] == shuffled["target_drift"]
    assert straight["incumbent_fitted_surface_loss"] == pytest.approx(
        shuffled["incumbent_fitted_surface_loss"]
    )
    # Pin the joined values themselves, so a silent revert to positional
    # pairing (which would give 8.0 and 50.0 here) fails rather than drifts.
    incumbent = scorer._incumbent_estimates(registry, weights, metrics)
    assert incumbent["households@E1@2025"] == pytest.approx(10.0)
    assert incumbent["income@W1@2025"] == pytest.approx(70.0)


@pytest.mark.parametrize("side", ["weights", "metrics"])
def test_local_scorer_refuses_incumbent_tables_without_household_id(side: str) -> None:
    scorer = _load_scorer()
    registry, candidate, weights, metrics = _case()
    if side == "weights":
        weights = weights.drop(columns=["household_id"])
    else:
        metrics = metrics.drop(columns=["household_id"])

    with pytest.raises(ValueError, match="positional pairing is not a join"):
        scorer.score_uk_local_candidate(
            candidate_diagnostics=candidate,
            incumbent_weights=weights,
            incumbent_metrics=metrics,
            target_registry=registry,
            expected_reference_count=2,
        )


def test_local_scorer_refuses_a_partial_incumbent_join() -> None:
    scorer = _load_scorer()
    registry, candidate, weights, metrics = _case()
    metrics.loc[1, "household_id"] = "h3"

    with pytest.raises(ValueError, match="must cover the same households"):
        scorer.score_uk_local_candidate(
            candidate_diagnostics=candidate,
            incumbent_weights=weights,
            incumbent_metrics=metrics,
            target_registry=registry,
            expected_reference_count=2,
        )


def test_local_scorer_refuses_duplicate_incumbent_households() -> None:
    scorer = _load_scorer()
    registry, candidate, weights, metrics = _case()
    weights.loc[1, "household_id"] = "h1"
    metrics.loc[1, "household_id"] = "h1"

    with pytest.raises(ValueError, match="repeats 'household_id'"):
        scorer.score_uk_local_candidate(
            candidate_diagnostics=candidate,
            incumbent_weights=weights,
            incumbent_metrics=metrics,
            target_registry=registry,
            expected_reference_count=2,
        )


def test_local_scorer_refuses_a_candidate_with_no_measured_holdout() -> None:
    """The rotation ships in the same payload; scoring without it is refused."""

    scorer = _load_scorer()
    registry, candidate, weights, metrics = _case()
    candidate = {
        key: value for key, value in candidate.items() if key != "uk_diagnostics"
    }

    with pytest.raises(ValueError, match="must carry a uk_diagnostics block"):
        scorer.score_uk_local_candidate(
            candidate_diagnostics=candidate,
            incumbent_weights=weights,
            incumbent_metrics=metrics,
            target_registry=registry,
            expected_reference_count=2,
        )


def test_local_scorer_refuses_a_holdout_with_missing_fold_losses() -> None:
    scorer = _load_scorer()
    registry, candidate, weights, metrics = _case()
    candidate["uk_diagnostics"]["rotated_holdout"]["fold_losses"] = [0.2, 0.3]

    with pytest.raises(ValueError, match="one loss per declared fold"):
        scorer.score_uk_local_candidate(
            candidate_diagnostics=candidate,
            incumbent_weights=weights,
            incumbent_metrics=metrics,
            target_registry=registry,
            expected_reference_count=2,
        )


def test_local_scorer_refuses_a_headline_that_is_not_its_folds() -> None:
    """The substitution the required-holdout refusal exists to catch."""

    scorer = _load_scorer()
    registry, candidate, weights, metrics = _case()
    # A fitted-looking number swapped in, with the real folds left beside it.
    candidate["uk_diagnostics"]["rotated_holdout"]["mean_holdout_loss"] = 0.02

    with pytest.raises(ValueError, match="does not close over its fold losses"):
        scorer.score_uk_local_candidate(
            candidate_diagnostics=candidate,
            incumbent_weights=weights,
            incumbent_metrics=metrics,
            target_registry=registry,
            expected_reference_count=2,
        )


def test_local_scorer_refuses_a_worst_loss_that_is_not_its_worst_fold() -> None:
    scorer = _load_scorer()
    registry, candidate, weights, metrics = _case()
    candidate["uk_diagnostics"]["rotated_holdout"]["worst_holdout_loss"] = 0.61

    with pytest.raises(ValueError, match="is not its worst fold"):
        scorer.score_uk_local_candidate(
            candidate_diagnostics=candidate,
            incumbent_weights=weights,
            incumbent_metrics=metrics,
            target_registry=registry,
            expected_reference_count=2,
        )


@pytest.mark.parametrize("bad", [float("nan"), -0.1])
def test_local_scorer_refuses_invalid_fold_losses(bad: float) -> None:
    scorer = _load_scorer()
    registry, candidate, weights, metrics = _case()
    candidate["uk_diagnostics"]["rotated_holdout"]["fold_losses"] = [
        bad,
        0.3,
        0.4,
        0.5,
        0.6,
    ]

    with pytest.raises(ValueError, match="non-finite or negative fold loss"):
        scorer.score_uk_local_candidate(
            candidate_diagnostics=candidate,
            incumbent_weights=weights,
            incumbent_metrics=metrics,
            target_registry=registry,
            expected_reference_count=2,
        )


def test_local_scorer_accepts_a_summary_its_folds_actually_produce() -> None:
    """`summarize_rotations` output must pass the closure check unchanged."""

    scorer = _load_scorer()
    registry, candidate, weights, metrics = _case()
    folds = [0.11, 0.27, 0.4, 0.52, 0.63]
    summary = summarize_rotations(folds)
    holdout = candidate["uk_diagnostics"]["rotated_holdout"]
    holdout["fold_losses"] = list(summary.fold_losses)
    holdout["mean_holdout_loss"] = summary.mean_holdout_loss
    holdout["worst_holdout_loss"] = summary.worst_holdout_loss

    result = scorer.score_uk_local_candidate(
        candidate_diagnostics=candidate,
        incumbent_weights=weights,
        incumbent_metrics=metrics,
        target_registry=registry,
        expected_reference_count=2,
    )

    assert result["candidate_holdout_loss"] == pytest.approx(summary.mean_holdout_loss)
