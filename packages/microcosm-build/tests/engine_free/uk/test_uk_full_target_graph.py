"""Actual full graph on synthetic target and engine-source adapters."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_full_target_graph import *


def test_default_scope_contains_all_levels_and_never_depends_on_k_or_k_small():
    default = UKFullBuildConfig(calibration_year=2026)
    for config in (
        default,
        replace(default, n_clones=1),
        replace(
            default, calibration=replace(default.calibration, dataset_households=20)
        ),
    ):
        graph = uk_full_graph(config).graph
        assert graph.node("uk.full.target_selection").params["geography_levels"] is None


def test_local_surface_selection_keeps_exact_target_periods():
    rows = pd.DataFrame(
        {
            "target_name": ["same", "same", "different"],
            "period": [2025, 2026, 2026],
            "value": [2.0, 3.0, 4.0],
        }
    )
    selected = [
        TargetSpec(
            name="same",
            entity="household",
            measure="count",
            value=3.0,
            period=2026,
            source="fixture",
            family="fixture",
        )
    ]
    actual = graph_targets._selected_local_surface(rows, selected)
    assert actual.to_dict(orient="records") == [
        {"target_name": "same", "period": 2026, "value": 3.0}
    ]


def test_local_problem_targets_come_from_chronicle_with_assignment_rosters(
    target_inputs, toy_ladder
):
    from microcosm.build.uk_runtime.full_problem import build_uk_full_local_problem
    from microcosm.build.uk_runtime.rowwise_dataset import (
        clone_uk_dataset_with_ladder_geography,
    )
    from test_support.microcosm_build.uk_full_population_graph import source_frame

    ladder, _ = toy_ladder
    clone = clone_uk_dataset_with_ladder_geography(
        source_frame(),
        ladder,
        n_clones=10,
        seed=7,
        source_year=2023,
        expected_constituency_vintage="2024_pcon",
    )
    ids = clone.frame.table("household")["household_id"]
    metrics = {
        grain: pd.DataFrame({"households": np.ones(len(ids))}, index=ids)
        for grain in ("constituency", "la")
    }
    local = target_inputs["local"]
    uprating = ledger_targets.uk_census_household_uprating(
        local, {"value": 33.0, "period": 2026}, period=2026
    )
    _, problem, receipt, _, _ = build_uk_full_local_problem(
        SimpleNamespace(result=clone, ladder=ladder),
        local_registry=local,
        national_registry=TargetRegistry([], country="uk"),
        local_metrics=metrics,
        period=2026,
        sample_fraction=1.0,
        reviewed_unbound_higher_targets={},
        census_household_uprating=uprating,
    )
    expected = target_inputs["surface"]()[0].set_index("target_name")["value"]
    actual = problem.target_frame.set_index("target_name")["value"]
    pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index())
    assert set(actual.index) == {spec.name for spec in local}
    assert receipt["census_household_uprating"]["household_cells"]["cells"] == 10


def test_target_kernel_identity_includes_reconciliation_and_ladder_diagnostics(
    monkeypatch,
):
    hashed = []
    monkeypatch.setattr(
        graph_targets, "source_hash", lambda *objects: hashed.extend(objects) or "test"
    )
    graph_targets.UKFullTargetCompilationKernel().implementation_hash()
    assert graph_targets.cross_grain in hashed
    assert graph_targets.ladder_targets in hashed


def test_joint_surface_keeps_region_rows_as_cross_grain_controls():
    """The joint surface carries the whole national register, region rows
    included, as main's rowwise tool does; a country-only filter left the
    region_over_constituency legs unparented on the first licensed graph build."""
    from microcosm.build.uk_runtime import full_problem
    from microcosm.calibrate import TargetRegistry, TargetSpec

    def spec(name, level, geography_id):
        return TargetSpec(
            name=name,
            entity="person",
            value=1.0,
            measure="age",
            period=2025,
            source="test",
            family="age",
            metadata={
                "contract_target_id": name,
                "geography_level": level,
                "geography_id": geography_id,
            },
        )

    national = TargetRegistry(
        [spec("age_uk", "country", "K02000001"), spec("age_ne", "region", "E12000001")],
        country="uk",
    )
    local = TargetRegistry(
        [spec("age_pcon", "constituency", "E14000530")], country="uk"
    )
    joint = full_problem._joint_surface_registry(local, national)
    assert [s.name for s in joint.specs] == ["age_pcon", "age_uk", "age_ne"]
    assert full_problem._national_contract_target_ids(national) == ("age_ne", "age_uk")
