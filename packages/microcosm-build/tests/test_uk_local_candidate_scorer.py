"""The UK local scorer compares one frozen surface on both sides."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from microcosm.calibrate import TargetRegistry, TargetSpec


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
    }
    weights = pd.DataFrame({"E1": [5.0, 5.0], "W1": [1.0, 1.0]})
    metrics = pd.DataFrame({"households": [1.0, 1.0], "income": [40.0, 40.0]})
    return registry, candidate, weights, metrics


def test_local_scorer_reports_per_family_wins_and_no_declared_holdout() -> None:
    scorer = _load_scorer()
    registry, candidate, weights, metrics = _case()

    result = scorer.score_uk_local_candidate(
        candidate_diagnostics=candidate,
        incumbent_weights=weights,
        incumbent_metrics=metrics,
        target_registry=registry,
        expected_reference_count=2,
    )

    assert result["holdout_basis"] == "none_declared"
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
