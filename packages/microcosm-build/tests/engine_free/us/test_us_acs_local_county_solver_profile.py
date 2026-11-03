"""County opt-in settings restore a corrective gradient for distant targets."""

from __future__ import annotations

import importlib.util
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from microcosm.calibrate import solve as solve_module
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from test_support.paths import paths_for


@lru_cache
def _release():
    path = (
        paths_for("microcosm-build").repository
        / "tools"
        / "build_us_acs_local_release.py"
    )
    spec = importlib.util.spec_from_file_location("acs_local_county_solver_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(*options):
    return _release()._parse_args(
        [
            "--stage",
            "calibrate",
            "--staging-h5",
            "staging.h5",
            "--checkpoint-dir",
            "checkpoint",
            "--out-h5",
            "calibrated.h5",
            *options,
        ]
    )


@pytest.mark.parametrize(
    "options",
    [
        ("--geographies", "state,cd,county"),
        ("--geographies", "county"),
        ("--soi-mode", "state_county"),
    ],
)
def test_county_rows_select_the_county_solver_profile(options) -> None:
    args = _args(*options)
    assert args.target_loss_cap == 10.0
    assert args.max_weight_ratio == 5.0
    profile = {"scope": "run", "target_loss_cap": 10.0, "max_weight_ratio": 5.0}
    assert args.county_solver_profile == profile
    settings = _release()._solver_settings(args, {"fixture": "loss weights"})
    assert settings["target_loss_cap"] == 10.0
    assert settings["max_weight_ratio"] == 5.0
    assert settings["county_solver_profile"] == profile


def test_county_caps_override_generic_caps_and_record_the_effective_settings() -> None:
    args = _args(
        "--geographies",
        "state,county",
        "--target-loss-cap",
        "2",
        "--max-weight-ratio",
        "3",
        "--county-target-loss-cap",
        "12",
        "--county-max-weight-ratio",
        "20",
    )
    assert (args.target_loss_cap, args.max_weight_ratio) == (12.0, 20.0)
    settings = _release()._solver_settings(args, {})
    assert settings["county_solver_profile"] == {
        "scope": "run",
        "target_loss_cap": 12.0,
        "max_weight_ratio": 20.0,
    }


@pytest.mark.parametrize(
    "generic_caps", [(), ("--target-loss-cap", "2", "--max-weight-ratio", "7")]
)
def test_no_county_rows_preserve_the_historical_solver_settings(generic_caps) -> None:
    module = _release()
    baseline = _args(*generic_caps)
    # Supplying county knobs alone cannot change an ordinary release's solve.
    county_knobs = _args(
        *generic_caps,
        "--county-target-loss-cap",
        "12",
        "--county-max-weight-ratio",
        "20",
    )
    loss_weights = {"fixture": "loss weights"}
    historical = {
        "method": "adam",
        "learning_rate": 0.02,
        "mass": "conserve",
        "max_weight_ratio": 7.0 if generic_caps else 5.0,
        "target_loss_cap": 2.0 if generic_caps else 1.0,
        "l2_lambda": 0.0,
        "l2_basis": "record",
        "mass_parametrization": "projection",
        "seed": 0,
        "epoch_batch": 400,
        "target_loss": loss_weights,
    }
    assert module._solver_settings(baseline, loss_weights) == historical
    assert module._solver_settings(county_knobs, loss_weights) == historical


@pytest.mark.parametrize(
    "identity",
    [
        {"geographies": "state,county", "soi_mode": "state"},
        {"geographies": "state,cd", "soi_mode": "state_county"},
    ],
)
def test_later_stages_select_profile_from_the_checkpoint(identity) -> None:
    module = _release()
    args = _args("--county-max-weight-ratio", "20")
    assert args.target_loss_cap == 1.0
    module._apply_county_solver_profile(args, identity=identity)
    assert (args.target_loss_cap, args.max_weight_ratio) == (10.0, 20.0)
    assert module._solver_settings(args, {})["county_solver_profile"]["scope"] == "run"


def test_a_noncounty_checkpoint_restores_the_requested_generic_caps() -> None:
    module = _release()
    args = _args(
        "--geographies",
        "state,county",
        "--target-loss-cap",
        "2",
        "--max-weight-ratio",
        "7",
    )
    assert args.target_loss_cap == 10.0
    module._apply_county_solver_profile(
        args, identity={"geographies": "state,cd", "soi_mode": "state"}
    )
    assert (args.target_loss_cap, args.max_weight_ratio) == (2.0, 7.0)
    assert "county_solver_profile" not in module._solver_settings(args, {})


@pytest.mark.parametrize(
    "identity",
    [
        {"staging_sha256": "legacy"},
        {"staging_sha256": "legacy", "soi_mode": "state"},
    ],
)
def test_a_legacy_checkpoint_without_geographies_is_not_a_county_run(
    identity,
) -> None:
    # Identities from before the county rung record no geographies and hold
    # no county rows; the CLI's --geographies must not select the profile.
    module = _release()
    args = _args(
        "--geographies",
        "state,county",
        "--target-loss-cap",
        "2",
        "--max-weight-ratio",
        "7",
    )
    assert args.target_loss_cap == 10.0
    module._apply_county_solver_profile(args, identity=identity)
    assert (args.target_loss_cap, args.max_weight_ratio) == (2.0, 7.0)
    assert "county_solver_profile" not in module._solver_settings(args, {})


@pytest.mark.parametrize(
    "option", ["--county-target-loss-cap", "--county-max-weight-ratio"]
)
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_county_solver_caps_must_be_positive_and_finite(option, value) -> None:
    with pytest.raises(SystemExit):
        _args("--geographies", "county", option, value)


def _county_problem(module):
    # The unmeasured household receives the displaced mass, so the county
    # target is feasible without changing the release's conserved total.
    design = np.asarray([300.0, 300.0])
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": [1, 2], "person_household_id": [1, 2]}
            ),
            "household": pd.DataFrame(
                {
                    "household_id": [1, 2],
                    "state_fips": [1, 1],
                    "county_fips": [1001, 1003],
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(design, WeightKind.DESIGN)},
    )
    spec = module.county_surface.population_spec("pop_county_01001", 100.0)
    targets = module.cd_surface.calibration_target_set(
        [{**module.county_surface.target_record(spec), "role": "train"}],
        sparse.csr_array([[1.0, 0.0]], dtype=np.float32),
        2,
        specs=[spec],
    )
    return frame, targets, design


@pytest.mark.parametrize("county_profile", [False, True])
def test_county_profile_restores_gradient_and_corrects_a_200_percent_miss(
    county_profile,
) -> None:
    module = _release()
    args = _args(*(("--geographies", "county") if county_profile else ()))
    estimate = torch.tensor([300.0], dtype=torch.float64, requires_grad=True)
    loss = solve_module._relative_error_loss(
        estimate,
        torch.tensor([100.0], dtype=torch.float64),
        None,
        torch.tensor([100.0], dtype=torch.float64),
        args.target_loss_cap,
    )
    loss.backward()
    assert estimate.grad is not None
    if county_profile:
        assert estimate.grad.item() > 0
    else:
        assert estimate.grad.item() == 0

    frame, targets, design = _county_problem(module)
    result, done = module.calibrate_surface(
        frame,
        targets,
        epochs=80,
        epoch_batch=40,
        max_weight_ratio=args.max_weight_ratio,
        target_loss_cap=args.target_loss_cap,
        l2_lambda=0.0,
        seed=0,
    )
    assert done == 80
    assert result.weights.sum() == pytest.approx(design.sum())
    if county_profile:
        assert abs(result.weights[0] - 100.0) < abs(design[0] - 100.0)
    else:
        np.testing.assert_allclose(result.weights, design, rtol=0, atol=1e-10)
