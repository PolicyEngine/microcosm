"""The housing holdout tool compares the imputation, matching and a forest draw."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.spi_housing_shell import (
    apply_structural_rules,
    fit_housing_model,
    household_housing_predictors,
    household_housing_targets,
)

_TOOL_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "validate_uk_housing_imputation.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "validate_uk_housing_imputation", _TOOL_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic(n: int = 400):
    spec = importlib.util.spec_from_file_location(
        "_housing_shell_fixture",
        Path(__file__).with_name("test_uk_spi_housing_shell.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    person, _, household, weights = fixture._tables(n_frs=n, n_spi=0)
    predictors = household_housing_predictors(person, household)
    targets = household_housing_targets(person, household)
    return predictors, targets, weights


def test_every_method_reports_the_same_surfaces() -> None:
    tool = _load_tool()
    predictors, targets, weights = _synthetic()
    held = np.arange(len(predictors)) % 5 == 0
    train, test = ~held, held
    model = fit_housing_model(
        predictors[train], targets[train], weights[train], seed=0, n_estimators=10
    )
    imputed, _ = apply_structural_rules(
        model.draw(predictors[test]),
        predictors.loc[test, "region"],
        model.structural_zeros,
    )
    matched, chosen = tool.match_housing(
        predictors[train], targets[train], weights[train], predictors[test]
    )
    forest, _ = tool.forest_joint_draw(
        predictors[train],
        targets[train],
        weights[train],
        predictors[test],
        n_estimators=10,
    )
    observed = targets[test]
    noise = tool.noise_floor(observed, predictors[test])
    reports = [
        tool.evaluate(observed, filled, predictors[test], weights[test])
        for filled in (imputed, matched, forest, noise)
    ]
    keys = set(reports[0])
    assert all(set(report) == keys for report in reports)
    assert {"categorical_tvd", "conditional_distributions", "shares"} <= keys
    assert len(chosen) == int(test.sum())
    # The noise floor reshuffles observed records, so it preserves every
    # structural rule the observed data hold.
    assert reports[3]["coherence_violations"]["mortgage>0 & not mortgaged"] == 0
    pd.testing.assert_index_equal(imputed.index, observed.index)
