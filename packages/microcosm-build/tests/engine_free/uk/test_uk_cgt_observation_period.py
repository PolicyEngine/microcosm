"""Tests split from packages/microcosm-build/tests/test_uk_cgt_observation_period.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_cgt_observation_period import *
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


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
        resolver = resolver_factory(
            frame=original, scratch_dir=tmp_path / "engine", year=2025
        )
        stage = UKNationalCalibrationStage(
            registry,
            band_edge_registry=registry,
            period=2025,
            doctrine=UKNationalSolveDoctrine(epochs=2),
            measure_resolver=resolver,
        )
        fitted = stage(original)
        receipt = stage.manifest["measure_resolution"]["provider"][
            "cgt_period_contract"
        ]
    else:
        from microcosm.build.uk_runtime.local_rowwise import (
            build_uk_rowwise_local_matrix,
            solve_uk_rowwise_weights_under_doctrine,
        )

        spec = importlib.util.spec_from_file_location(
            "cgt_rowwise_builder",
            _TEST_PATHS.repository / "tools/build_uk_rowwise_candidate.py",
        )
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        monkeypatch.setattr(
            builder,
            "compute_household_metrics",
            lambda _sim, _area, *, period, household_ids: pd.DataFrame(
                {"households": np.ones(len(household_ids))}, index=household_ids
            ),
        )
        prepared, restore, national, metrics, engine_receipt = (
            builder._resolve_candidate_engine_surface(
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
