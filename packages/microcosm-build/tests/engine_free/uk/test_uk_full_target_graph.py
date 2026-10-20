"""Actual full graph on synthetic target and engine-source adapters."""

# ruff: noqa: F403, F405
from microcosm.build.uk_runtime.atomic_area_support import (
    uk_atomic_assignment_definition,
)
from test_support.microcosm_build.uk_atomic_support_fixtures import (
    toy_support_payloads,
)
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
        graph = uk_full_graph(
            config,
            atomic_geography_definition=uk_atomic_assignment_definition(
                toy_support_payloads(), seed=config.seed
            ),
        ).graph
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


def test_the_local_problem_binds_scotlands_brma_rows_on_their_partial_grain(
    toy_ladder,
):
    """microcosm#1305 re-review (1305-20): BRMA rows through the problem builder.

    A Scottish household joins its BRMA's cell through its BRMA's S33 code;
    every other household belongs to no BRMA area, so the BRMA row's matrix
    row holds the Scottish clones only, and the full-scale coverage check
    reads Scotland's assigned BRMAs alone.
    """
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
    household = clone.frame.table("household").copy()
    scottish = household["region"].astype(str).to_numpy() == "SCOTLAND"
    household["brma"] = np.where(scottish, "LOTHIAN", "MAIDSTONE")
    ids = household["household_id"]
    metrics = {
        "constituency": pd.DataFrame({"households": np.ones(len(ids))}, index=ids),
        "brma": pd.DataFrame(
            {"rent/private_rent_brma": np.full(len(ids), 9_000.0)}, index=ids
        ),
    }
    rows = [
        {
            "target_name": f"ons.census.households@{code}",
            "value": 10.0,
            "family": "census_households",
            "area_type": "constituency",
            "area_code": code,
            "metric": "households",
        }
        for code in sorted(set(household["constituency_code"].astype(str)))
    ]
    rows.append(
        {
            "target_name": "ons.rent.private_rent_brma@S33000011",
            "value": 90_000.0,
            "family": "private_rent",
            "area_type": "brma",
            "area_code": "S33000011",
            "metric": "rent/private_rent_brma",
        }
    )
    assignment = SimpleNamespace(
        result=SimpleNamespace(frame=SimpleNamespace(table=lambda entity: household)),
        ladder=ladder,
    )
    _, problem, _, local_bound, _ = build_uk_full_local_problem(
        assignment,
        local_registry=TargetRegistry([], country="uk"),
        national_registry=TargetRegistry([], country="uk"),
        local_metrics=metrics,
        period=2025,
        sample_fraction=1.0,
        reviewed_unbound_higher_targets={},
        selected_surface=pd.DataFrame(rows),
    )
    assert "private_rent/brma" in local_bound
    assert problem.assigned_areas_by_grain["brma"] == tuple(
        "S33000011" if flag else "" for flag in scottish
    )
    brma = problem.target_frame.set_index("target_name").loc[
        "ons.rent.private_rent_brma@S33000011"
    ]
    assert brma["area_type"] == "brma"
    matrix = problem.matrix.tocsr()
    (row,) = np.flatnonzero(
        problem.target_frame["target_name"].to_numpy()
        == "ons.rent.private_rent_brma@S33000011"
    )
    columns = matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]]
    assert set(columns) == set(np.flatnonzero(scottish)) and scottish.sum() == 10


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


def test_stored_surface_round_trip_restores_the_hierarchy():
    """The local surface crosses the node boundary as JSON; its schema-8
    hierarchy must come back as a CalibrationHierarchy, or the problem assembly
    refuses every hierarchy-bearing target (first licensed graph build)."""
    import json

    import pandas as pd

    from microcosm.build.uk_runtime import graph_targets
    from microcosm.calibrate.hierarchy import CalibrationHierarchy
    from test_support.microcosm_build.uk_hierarchy_fixtures import uk_fixture_hierarchy

    name = "hmrc.self_employment_income.amount@E14001063"
    hierarchy = uk_fixture_hierarchy(
        name, level="constituency", geography_id="E14001063"
    )
    frame = pd.DataFrame(
        [
            {"target_name": name, "value": 1.5, "hierarchy": hierarchy},
            {"target_name": "plain@E14001063", "value": 2.0, "hierarchy": None},
        ]
    )
    payload = json.loads(
        json.dumps(
            {
                "surface": graph_targets._surface_records(frame),
                "surface_columns": frame.columns.tolist(),
            }
        )
    )
    assert isinstance(payload["surface"][0]["hierarchy"], dict)
    restored = graph_targets._surface_frame(payload)
    assert restored.columns.tolist() == frame.columns.tolist()
    assert isinstance(restored["hierarchy"][0], CalibrationHierarchy)
    assert restored["hierarchy"][0] == hierarchy
    assert restored["hierarchy"][1] is None
    assert restored["value"].tolist() == [1.5, 2.0]


def test_the_compiles_unsupported_report_crosses_the_surface_to_the_parity_gate(
    target_inputs, toy_ladder, monkeypatch
):
    """The full-build path of the report, end to end: the target-compilation
    payload writes it beside the compiled registries, the full gate kernel's
    evidence reads it back with integer period keys, and the parity gate then
    reads the unresolved reference as ledger_absent."""

    from microcosm.build.gate_battery import EvidenceContext
    from microcosm.build.uk_runtime import battery_bindings, graph_terminal

    national = target_inputs["national"]
    missing, *resolved = national.specs
    unresolved = {"name": missing.name, "period": 2026, "reason": "no fact"}
    # The compile resolved every reference but one, and reported that one.
    inputs = {
        **target_inputs["inputs"],
        "uk_ledger_compiled_registries": {2026: TargetRegistry(resolved, country="uk")},
        "uk_ledger_compile_unsupported": {2026: (unresolved,)},
    }
    _, ladder_path = toy_ladder

    payload = graph_targets.compile_uk_full_target_surface(inputs, ladder_path)

    # The surface crosses the node boundary as JSON, with string period keys.
    stored = json.dumps(payload)
    assert json.loads(stored)["uk_ledger_compile_unsupported"] == {"2026": [unresolved]}
    assert json.loads(stored)["uk_ledger_compile_local_unsupported"] == {"2026": []}
    evidence = graph_terminal._source_gate_evidence(
        SimpleNamespace(
            artifacts={"surface": SimpleNamespace(payload=stored)},
            params={"spine_stage_names": ("frs_spine",), "review_date": "2026-09-10"},
        ),
        engine=None,
    )
    assert evidence["uk_ledger_compile_unsupported"] == {2026: (unresolved,)}
    assert evidence["uk_ledger_compile_local_unsupported"] == {2026: ()}

    # The fixture still carries the row, signed as ledger_absent.
    monkeypatch.setattr(
        battery_bindings,
        "_load_ledger_compile_parity_fixture",
        lambda resource: national,
    )
    parameters = {
        "fixture_resource": "synthetic.json",
        "target_period": 2026,
        "signed_differences": (
            {
                "fixture_value": missing.value,
                "kind": "ledger_absent",
                "name": missing.name,
                "period": missing.period,
                "reason": "reviewed: the pinned feed no longer resolves it",
            },
        ),
    }
    context = EvidenceContext(artifacts=evidence)
    result = battery_bindings._evaluate_ledger_compile_parity(context, parameters)
    assert result.passed, result.failures
    assert result.details["unsupported_reference_count"] == 1
    # Without the report on the surface, the same signed row changes kind.
    dropped = {**evidence, "uk_ledger_compile_unsupported": {2026: ()}}
    refused = battery_bindings._evaluate_ledger_compile_parity(
        EvidenceContext(artifacts=dropped), parameters
    )
    assert not refused.passed
    assert "changed kind" in refused.failures[0]
