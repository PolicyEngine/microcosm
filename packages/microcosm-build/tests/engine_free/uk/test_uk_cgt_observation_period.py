"""Tests split from packages/microcosm-build/tests/test_uk_cgt_observation_period.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_cgt_observation_period import *


@pytest.mark.parametrize("route", ["national", "local"])
def test_dated_cgt_fit_restores_base2024_values_with_fitted_weights(
    monkeypatch, tmp_path, route
):
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    original = _frame()
    simulation = _Simulation()
    monkeypatch.setattr(
        measure_simulation,
        "_policyengine_uk_module",
        lambda: SimpleNamespace(__version__="test"),
    )

    def resolver_factory(**kwargs):
        kwargs.setdefault("simulation_source", None)
        return UKMeasureResolver(
            **kwargs, microsimulation_factory=lambda **_: simulation
        )

    registry = _registry()
    if route == "national":
        from microcosm.calibrate import build_constraint_matrix, calibrate

        # The graph's national route: resolve on the bound frame, materialise,
        # compile the rows, solve under the doctrine, restore the pristine
        # tables around the calibrated weights.
        prepared, restore, rows, evidence = materialize_uk_national_rows(
            original,
            registry,
            period=2025,
            band_edge_registry=registry,
            resolver_factory=resolver_factory,
            scratch_dir=tmp_path / "engine",
        )
        problem = build_constraint_matrix(prepared, rows.targets, "household")
        assert not problem.skipped
        doctrine = UKNationalSolveDoctrine(epochs=2)
        result = calibrate(
            prepared,
            rows.targets,
            weight_entity="household",
            epochs=doctrine.epochs,
            learning_rate=doctrine.learning_rate,
            seed=doctrine.seed,
            mass="free",
            mass_reason="national doctrine calibration (test)",
            max_weight_ratio=doctrine.max_weight_ratio,
            target_loss_cap=doctrine.target_loss_cap,
        )
        fitted = restore(result.frame)
        receipt = evidence["cgt_period_contract"]
    else:
        from microcosm.build.uk_runtime import full_measure
        from microcosm.build.uk_runtime.local_rowwise import (
            build_uk_rowwise_local_matrix,
            solve_uk_rowwise_weights_under_doctrine,
        )

        monkeypatch.setattr(
            full_measure,
            "compute_household_metrics",
            lambda _sim, _area, *, period, household_ids: pd.DataFrame(
                {"households": np.ones(len(household_ids))}, index=household_ids
            ),
        )
        prepared, restore, national, metrics, engine_receipt = (
            full_measure.resolve_uk_full_measures(
                original,
                registry,
                period=2025,
                scratch_dir=tmp_path / "engine",
                resolver_factory=resolver_factory,
            )
        )
        receipt = engine_receipt["cgt_period_contract"]
        local = build_uk_rowwise_local_matrix(
            metrics["constituency"],
            pd.Series(["A", "A", "A"], index=[0, 1, 2]),
            pd.DataFrame({"code": ["A"], "households": [20.0]}),
        )
        result = solve_uk_rowwise_weights_under_doctrine(
            prepared,
            local,
            bound_families=["census_households/constituency", "national/hmrc_cgt"],
            national_rows=national,
            restore=restore,
            epochs=2,
        )
        fitted = result.frame
    assert all(spec.period == 2025 for spec in registry.specs)
    assert receipt["input_period"] == "2024"
    assert receipt["calibration_period"] == 2025
    assert receipt["bound_measurements"] == {
        "cgt_2024_gains": {
            "model_variable": "capital_gains",
            "measurement_period": 2024,
        },
        "cgt_2024_tax": {
            "model_variable": "capital_gains_tax",
            "measurement_period": 2024,
        },
    }
    assert set(simulation.calls) == {
        ("capital_gains", 2024),
        ("capital_gains_tax", 2024),
    }
    _assert_export(original, fitted, tmp_path / f"{route}.h5")
