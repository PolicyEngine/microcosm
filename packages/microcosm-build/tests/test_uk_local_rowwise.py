"""Rowwise local solve surface (#495 increment 6b).

The US Build-N shape for the UK: one weight vector over the cloned rowwise
households, where each household supports only its assigned constituency's
target rows. The matrix builder fails closed on an assigned area the target
surface does not cover (the 650/650 requirement), and the solve runs under
the #503 doctrine — no per-target knobs, declared bounds, past-cap census on
every result, and initial weights that are the household base weights
directly (never split across areas: a rowwise household exists in exactly
one area).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import (
    UK_LOCAL_BINDING_ADJUDICATION_REGISTER_RESOURCE,
    UK_LOCAL_MAX_WEIGHT_RATIO,
    UK_LOCAL_TARGET_LOSS_CAP,
    UKRowwiseNationalRows,
    build_uk_rowwise_local_matrix,
    build_uk_rowwise_local_surface_matrix,
    load_uk_local_target_census,
    require_adjudicated_uk_local_binding,
    rotated_uk_local_holdout,
    rowwise_area_support_summary,
    rowwise_calibration_mass_reason,
    solve_uk_rowwise_weights_under_doctrine,
    uk_area_support_summary,
    uk_household_weight_kind,
    uk_ladder_area_support_summary,
    uk_national_frame,
)
from microcosm.build.uk_runtime.weighted_integrity import (
    load_uk_reviewed_exclusion_register,
)
from microcosm.calibrate.registry import TargetRegistry, TargetSpec
from microcosm.calibrate.target import TargetSet
from microcosm.frame import WeightKind


def _clone_frame(weights=(1.0, 1.0, 1.0)):
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "person_household_id": [101, 102, 103],
                "person_benunit_id": [11, 12, 13],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [11, 12, 13]}),
        household=pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "household_weight": list(weights),
            }
        ),
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
    )


def _metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "households": [2.0, 1.0, 3.0],
            "tenure/social_rent": [1.0, 0.0, 1.0],
        },
        index=[101, 102, 103],
    )


def _assigned() -> pd.Series:
    return pd.Series(["E001", "E001", "S001"], index=[101, 102, 103])


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["E001", "S001"],
            "households": [4.0, 2.0],
            "tenure/social_rent": [1.0, 1.0],
        }
    )


def _uc_problem():
    metrics = pd.DataFrame(
        {"uc_households": [1.0, 0.0, 1.0]},
        index=[101, 102, 103],
    )
    targets = pd.DataFrame(
        {
            "code": ["E001", "S001"],
            "uc_households": [1.0, 1.0],
        }
    )
    return build_uk_rowwise_local_matrix(metrics, _assigned(), targets)


def _reviewed_register_entry(
    *,
    approved_on: str = "2026-01-01",
    expires_on: str = "2027-01-01",
) -> dict[str, str]:
    return {
        "reason": (
            "Accept the stated basis for test evidence. Evidence: "
            "uk_local_target_census.json#/binding_fences/"
            "census_disclosure_control_noise."
        ),
        "approved_by": "tester",
        "adjudication": "microcosm#760-test",
        "approved_on": approved_on,
        "expires_on": expires_on,
    }


def test_matrix_builder_places_support_only_in_assigned_area() -> None:
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    assert problem.matrix.shape == (4, 3)
    assert problem.area_codes == ("E001", "S001")
    assert problem.metric_names == ("households", "tenure/social_rent")
    assert problem.household_ids == (101, 102, 103)

    dense = problem.matrix.toarray()
    frame = problem.target_frame
    # E001 rows carry only households 101/102; S001 rows only household 103.
    e_pop = int(
        frame[(frame["area_code"] == "E001") & (frame["metric"] == "households")][
            "target_index"
        ].iloc[0]
    )
    assert dense[e_pop].tolist() == [2.0, 1.0, 0.0]
    s_pop = int(
        frame[(frame["area_code"] == "S001") & (frame["metric"] == "households")][
            "target_index"
        ].iloc[0]
    )
    assert dense[s_pop].tolist() == [0.0, 0.0, 3.0]
    np.testing.assert_allclose(problem.targets, [4.0, 1.0, 2.0, 1.0])
    assert frame["area_type"].unique().tolist() == ["constituency"]


def test_surface_builder_omits_absent_cells_and_uses_canonical_order() -> None:
    constituency_metrics = _metrics()
    la_metrics = pd.DataFrame(
        {
            "households": [1.0, 1.0, 1.0],
            "income/employment": [10.0, 20.0, 30.0],
        },
        index=constituency_metrics.index,
    )
    surface = pd.DataFrame(
        [
            {
                "area_type": "la",
                "area_code": "L2",
                "metric": "households",
                "value": 1.0,
                "target_name": "external:census_households/households@L2",
                "family": "census_households",
            },
            {
                "area_type": "constituency",
                "area_code": "S001",
                "metric": "households",
                "value": 1.0,
                "target_name": "external:census_households/households@S001",
                "family": "census_households",
            },
            {
                "area_type": "constituency",
                "area_code": "E001",
                "metric": "tenure/social_rent",
                "value": 1.0,
                "target_name": "contract:tenure@E001",
                "family": "tenure",
            },
            {
                "area_type": "constituency",
                "area_code": "E001",
                "metric": "households",
                "value": 2.0,
                "target_name": "external:census_households/households@E001",
                "family": "census_households",
            },
            {
                "area_type": "la",
                "area_code": "L1",
                "metric": "income/employment",
                "value": 30.0,
                "target_name": "contract:income@L1",
                "family": "income",
            },
            {
                "area_type": "la",
                "area_code": "L1",
                "metric": "households",
                "value": 2.0,
                "target_name": "external:census_households/households@L1",
                "family": "census_households",
            },
        ]
    )

    problem = build_uk_rowwise_local_surface_matrix(
        {
            "constituency": constituency_metrics,
            "la": la_metrics,
        },
        {
            "constituency": _assigned(),
            "la": pd.Series(["L1", "L1", "L2"], index=constituency_metrics.index),
        },
        surface,
        area_codes_by_grain={
            "constituency": ("E001", "S001"),
            "la": ("L1", "L2"),
        },
    )

    assert problem.matrix.shape == (6, 3)
    assert list(
        problem.target_frame[["area_type", "metric", "area_code"]].itertuples(
            index=False, name=None
        )
    ) == [
        ("constituency", "households", "E001"),
        ("constituency", "households", "S001"),
        ("constituency", "tenure/social_rent", "E001"),
        ("la", "households", "L1"),
        ("la", "households", "L2"),
        ("la", "income/employment", "L1"),
    ]
    assert not (
        (problem.target_frame["area_type"] == "constituency")
        & (problem.target_frame["area_code"] == "S001")
        & (problem.target_frame["metric"] == "tenure/social_rent")
    ).any()


def test_surface_builder_refuses_roster_coverage_and_unreachable_rows() -> None:
    base = pd.DataFrame(
        [
            {
                "area_type": "constituency",
                "area_code": "E001",
                "metric": "households",
                "value": 1.0,
                "target_name": "households@E001",
                "family": "census_households",
            }
        ]
    )
    kwargs = {
        "metrics_by_grain": {"constituency": _metrics()},
        "assigned_by_grain": {"constituency": _assigned()},
        "area_codes_by_grain": {"constituency": ("E001", "S001")},
    }
    with pytest.raises(ValueError, match="assigned area.*S001"):
        build_uk_rowwise_local_surface_matrix(surface=base, **kwargs)

    relaxed = build_uk_rowwise_local_surface_matrix(
        surface=base,
        require_every_assigned_area_covered=False,
        **kwargs,
    )
    assert relaxed.matrix.shape == (1, 3)

    off_roster = base.assign(area_code="X999")
    with pytest.raises(ValueError, match="roster"):
        build_uk_rowwise_local_surface_matrix(
            surface=off_roster,
            require_every_assigned_area_covered=False,
            **kwargs,
        )

    unreachable = base.assign(
        area_code="S001",
        metric="tenure/social_rent",
        value=1.0,
    )
    zero_metrics = _metrics().assign(**{"tenure/social_rent": 0.0})
    with pytest.raises(ValueError, match="zero household support"):
        build_uk_rowwise_local_surface_matrix(
            metrics_by_grain={"constituency": zero_metrics},
            assigned_by_grain={"constituency": _assigned()},
            surface=unreachable,
            area_codes_by_grain={"constituency": ("E001", "S001")},
            require_every_assigned_area_covered=False,
        )


def test_registry_surface_names_periods_and_lazy_masks_match_sparse_matrix() -> None:
    import microcosm.build.uk_runtime.local_rowwise as local_rowwise

    spec = TargetSpec(
        name="contract:ons.households@E001",
        entity="household",
        value=4.0,
        measure="households",
        period=2025,
        source="ONS",
        family="census_households",
        metadata={"contract_target_id": "ons.households"},
    )
    surface = pd.DataFrame(
        [
            {
                "area_type": "constituency",
                "area_code": "E001",
                "metric": "households",
                "value": spec.value,
                "target_name": spec.name,
                "family": spec.family,
                "period": spec.period,
                "source": spec.source,
                "contract_target_id": spec.metadata["contract_target_id"],
            }
        ]
    )
    problem = build_uk_rowwise_local_surface_matrix(
        {"constituency": _metrics()},
        {"constituency": _assigned()},
        surface,
        area_codes_by_grain={"constituency": ("E001", "S001")},
        require_every_assigned_area_covered=False,
    )
    target = tuple(local_rowwise._rowwise_target_set(problem))[0]

    assert target.row_name == spec.to_target().row_name
    assert target.metadata == {
        "area_type": "constituency",
        "area_code": "E001",
        "metric": "households",
        "family": "census_households",
        "target_name": spec.name,
        "contract_target_id": "ons.households",
    }
    expected = np.asarray(target.measure(_clone_frame())) * np.asarray(
        target.filter(_clone_frame())
    )
    np.testing.assert_allclose(expected, problem.matrix.toarray()[0])


def test_bound_family_derivation_accepts_multiple_families_per_grain() -> None:
    import microcosm.build.uk_runtime.local_rowwise as local_rowwise

    frame = pd.DataFrame(
        {
            "area_type": ["constituency", "constituency", "la"],
            "metric": ["households", "households", "households"],
            "family": ["census_households", "tenure", "census_households"],
        }
    )
    families = local_rowwise._derive_uk_local_bound_families_from_target_frame(
        frame,
        family_rows={
            "census_households": {},
            "tenure": {},
        },
    )
    assert families == [
        "census_households/constituency",
        "census_households/la",
        "tenure/constituency",
    ]


def test_dense_builder_delegates_without_changing_legacy_result() -> None:
    dense = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    surface_rows = []
    for _, row in _targets().iterrows():
        for metric in _metrics().columns:
            surface_rows.append(
                {
                    "area_type": "constituency",
                    "area_code": row["code"],
                    "metric": metric,
                    "value": row[metric],
                    "target_name": f"constituency/{row['code']}/{metric}",
                    "family": (
                        "census_households" if metric == "households" else "tenure"
                    ),
                }
            )
    surface = build_uk_rowwise_local_surface_matrix(
        {"constituency": _metrics()},
        {"constituency": _assigned()},
        pd.DataFrame(surface_rows),
        area_codes_by_grain={"constituency": ("E001", "S001")},
    )

    # The long-format core is metric-major; the legacy dense wrapper preserves
    # its established area-major row order while sharing the same assembler.
    dense_by_cell = dense.target_frame.set_index(["area_type", "area_code", "metric"])
    surface_by_cell = surface.target_frame.set_index(
        ["area_type", "area_code", "metric"]
    )
    pd.testing.assert_series_equal(
        dense_by_cell["value"].sort_index(),
        surface_by_cell["value"].sort_index(),
    )
    np.testing.assert_allclose(
        dense.matrix.toarray()[
            dense.target_frame.sort_values(["area_type", "metric", "area_code"]).index
        ],
        surface.matrix.toarray(),
    )


def test_matrix_builder_fails_closed_on_uncovered_assigned_area() -> None:
    assigned = pd.Series(["E001", "E001", "X999"], index=[101, 102, 103])
    with pytest.raises(ValueError, match="X999"):
        build_uk_rowwise_local_matrix(_metrics(), assigned, _targets())


def test_matrix_builder_validates_alignment_and_finiteness() -> None:
    misaligned = pd.Series(["E001", "S001"], index=[101, 999])
    with pytest.raises(ValueError, match="align"):
        build_uk_rowwise_local_matrix(_metrics(), misaligned, _targets())

    bad = _metrics()
    bad.loc[101, "households"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_uk_rowwise_local_matrix(bad, _assigned(), _targets())


def test_rowwise_doctrine_solve_uses_base_weights_directly() -> None:
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    base = [1.0, 1.0, 1.0]
    frame = _clone_frame(base)
    result = solve_uk_rowwise_weights_under_doctrine(
        frame,
        problem,
        bound_families=[
            "census_households/constituency",
            "tenure/constituency",
        ],
        epochs=60,
        learning_rate=0.2,
        seed=1,
    )
    # Rowwise initial weights ARE the frame's typed base weights, never
    # split across areas.
    np.testing.assert_allclose(result.initial_weights, base)
    assert result.weights.shape == (3,)
    assert np.isfinite(result.final_loss)
    assert result.past_cap_census is not None
    assert result.past_cap_census["target_loss_cap"] == UK_LOCAL_TARGET_LOSS_CAP
    stretched = result.weights / result.initial_weights
    assert float(np.max(stretched)) <= UK_LOCAL_MAX_WEIGHT_RATIO * (1 + 1e-6)

    # The declarative target expression and the hand-assembled sparse matrix
    # derive from the same numbers: the compiled initial estimates equal the
    # COO assembly's matvec row for row.
    np.testing.assert_allclose(
        result.diagnostics["initial_estimate"].to_numpy(dtype=np.float64),
        problem.matrix @ np.asarray(base, dtype=np.float64),
    )

    # The kernel product: CALIBRATED typed weights, the calibration mass
    # record naming the bound family, and the refreshed persisted column.
    assert uk_household_weight_kind(result.frame) is WeightKind.CALIBRATED
    record = result.frame.mass_log[-1]
    assert "census_households/constituency" in record.reason
    assert record.old_total == pytest.approx(float(np.sum(base)))
    assert record.new_total == pytest.approx(float(np.sum(result.weights)))
    np.testing.assert_allclose(
        result.frame.weights_for("household").values,
        result.weights,
    )
    receipt = result.binding_adjudications
    assert (
        receipt["register_resource"] == UK_LOCAL_BINDING_ADJUDICATION_REGISTER_RESOURCE
    )
    assert receipt["dormant"] == [
        "full_frs_tei_band_unavailable",
        "hmrc_spi_frame_model_proxy",
        "population_universe_private_households",
        "uc_unit_vs_household_grain",
        "voa_dwellings_vs_household_frame",
    ]
    stood_on = receipt["stood_on"]["census_households/constituency"]
    seed = stood_on["census_disclosure_control_noise"]
    assert seed["approved_by"] == "juaristi22"
    assert seed["adjudication"] == "microcosm#802"
    assert seed["approved_on"] == "2026-08-31"
    assert seed["expires_on"] == "2026-11-30"
    assert receipt["stood_on"]["tenure/constituency"] == {}


def test_joint_doctrine_solve_compiles_local_then_national_once_and_restores() -> None:
    import microcosm.build.uk_runtime.local_rowwise as local_rowwise

    base = _clone_frame()
    household = base.table("household").copy()
    household["national/ones"] = 1.0
    prepared = uk_national_frame(
        person=base.table("person"),
        benunit=base.table("benunit"),
        household=household,
        household_weights=base.weights_for("household").values,
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
    )
    spec = TargetSpec(
        name="national/households",
        entity="household",
        value=3.0,
        measure="national/ones",
        period=0,
        source="synthetic national fixture",
        family="national_fixture",
    )
    registry = TargetRegistry([spec], country="uk")
    national = UKRowwiseNationalRows(
        targets=registry.to_target_set(),
        registry=registry,
        families=("national_fixture",),
    )
    calls = 0
    real_calibrate = local_rowwise.calibrate

    def spy(frame, targets, **kwargs):
        nonlocal calls
        calls += 1
        assert [target.name for target in targets][-1] == "national/households"
        return real_calibrate(frame, targets, **kwargs)

    def restore(frame):
        restored_household = frame.table("household").drop(columns=["national/ones"])
        restored_household["household_weight"] = frame.weights_for("household").values
        return uk_national_frame(
            person=frame.table("person"),
            benunit=frame.table("benunit"),
            household=restored_household,
            household_weights=frame.weights_for("household").values,
            time_period="2023",
            weight_kind=WeightKind.CALIBRATED,
            mass_log=frame.mass_log,
        )

    local_rowwise.calibrate = spy
    try:
        result = solve_uk_rowwise_weights_under_doctrine(
            prepared,
            build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets()),
            bound_families=[
                "census_households/constituency",
                "tenure/constituency",
                "national/national_fixture",
            ],
            national_rows=national,
            restore=restore,
            epochs=2,
        )
    finally:
        local_rowwise.calibrate = real_calibrate

    assert calls == 1
    assert len(result.diagnostics) == 4
    assert result.national_diagnostics["name"].tolist() == [spec.to_target().row_name]
    assert result.all_past_cap_census["n_targets"] == 5
    assert result.national_past_cap_census["n_targets"] == 1
    assert "national/national_fixture" in result.frame.mass_log[-1].reason
    assert "national/ones" not in result.frame.table("household").columns


def test_joint_doctrine_solve_refuses_national_registry_misalignment() -> None:
    spec = TargetSpec(
        name="national/households",
        entity="household",
        value=3.0,
        measure="household_weight",
        source="synthetic national fixture",
        family="national_fixture",
    )
    registry = TargetRegistry([spec], country="uk")
    wrong = TargetSpec(
        name="national/wrong",
        entity="household",
        value=3.0,
        measure="household_weight",
        source="synthetic national fixture",
        family="national_fixture",
    )
    national = UKRowwiseNationalRows(
        targets=TargetSet([wrong.to_target()]),
        registry=registry,
        families=("national_fixture",),
    )
    with pytest.raises(ValueError, match="not aligned"):
        solve_uk_rowwise_weights_under_doctrine(
            _clone_frame(),
            build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets()),
            bound_families=[
                "census_households/constituency",
                "tenure/constituency",
                "national/national_fixture",
            ],
            national_rows=national,
            epochs=1,
        )


def test_rotated_holdout_keeps_national_rows_in_every_training_fold(
    monkeypatch,
) -> None:
    import microcosm.build.uk_runtime.local_rowwise as local_rowwise

    metrics = pd.DataFrame(
        {
            "households": [1.0, 1.0, 1.0],
            "tenure/social_rent": [1.0, 1.0, 1.0],
        },
        index=[101, 102, 103],
    )
    assigned = pd.Series(["A", "B", "C"], index=metrics.index)
    targets = pd.DataFrame(
        {
            "code": ["A", "B", "C"],
            "households": [1.0, 1.0, 1.0],
            "tenure/social_rent": [1.0, 1.0, 1.0],
        }
    )
    problem = build_uk_rowwise_local_matrix(metrics, assigned, targets)
    spec = TargetSpec(
        name="national/households",
        entity="household",
        value=3.0,
        measure="household_weight",
        source="synthetic national fixture",
        family="national_fixture",
    )
    registry = TargetRegistry([spec], country="uk")
    national = UKRowwiseNationalRows(
        targets=registry.to_target_set(),
        registry=registry,
        families=("national_fixture",),
    )
    calls = []

    def fake_solve(frame, training_problem, **kwargs):
        calls.append(kwargs["national_rows"])
        return type("Solve", (), {"weights": np.ones(3)})()

    monkeypatch.setattr(
        local_rowwise, "solve_uk_rowwise_weights_under_doctrine", fake_solve
    )
    receipt = rotated_uk_local_holdout(
        _clone_frame(),
        problem,
        bound_families=[
            "census_households/constituency",
            "tenure/constituency",
            "national/national_fixture",
        ],
        national_rows=national,
        epochs=1,
        learning_rate=0.1,
        conserve_mass=False,
        target_records=None,
        l0_lambda=0.0,
        budget_iters=1,
        solve_seed=7,
    )

    assert calls == [national] * 5
    assert receipt["n_folds"] == 5
    assert all(fold["training_national_rows"] == 1 for fold in receipt["folds"])


def test_rowwise_binding_refuses_unadjudicated_committed_fence() -> None:
    problem = _uc_problem()
    with pytest.raises(ValueError, match="uc_unit_vs_household_grain"):
        require_adjudicated_uk_local_binding(
            ["uc_households/constituency"],
            problem.target_frame,
            register={},
        )


def test_rowwise_binding_refuses_declared_derived_mismatch() -> None:
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    with pytest.raises(ValueError, match="missing.*tenure/constituency"):
        require_adjudicated_uk_local_binding(
            ["census_households/constituency"],
            problem.target_frame,
        )
    with pytest.raises(ValueError, match="extra.*private_rent/constituency"):
        require_adjudicated_uk_local_binding(
            [
                "census_households/constituency",
                "tenure/constituency",
                "private_rent/constituency",
            ],
            problem.target_frame,
        )


def test_rowwise_binding_refuses_unknown_family_and_bad_area_type() -> None:
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    with pytest.raises(ValueError, match="unknown census family"):
        require_adjudicated_uk_local_binding(
            ["not_a_family/constituency"],
            problem.target_frame,
        )
    with pytest.raises(ValueError, match="unsupported area_type"):
        require_adjudicated_uk_local_binding(
            ["census_households/ward"],
            problem.target_frame,
        )


def test_rowwise_binding_refuses_expired_and_premature_adjudications() -> None:
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    expired_register = {
        "census_disclosure_control_noise": _reviewed_register_entry(
            approved_on="2026-01-01",
            expires_on="2026-02-01",
        )
    }
    with pytest.raises(ValueError, match="correct the underlying gap or renew"):
        require_adjudicated_uk_local_binding(
            [
                "census_households/constituency",
                "tenure/constituency",
            ],
            problem.target_frame,
            register=expired_register,
            now=date(2026, 3, 1),
        )

    premature_register = {
        "census_disclosure_control_noise": _reviewed_register_entry(
            approved_on="2026-04-01",
            expires_on="2027-04-01",
        )
    }
    with pytest.raises(ValueError, match="correct the underlying gap or renew"):
        require_adjudicated_uk_local_binding(
            [
                "census_households/constituency",
                "tenure/constituency",
            ],
            problem.target_frame,
            register=premature_register,
            now=date(2026, 3, 1),
        )


def test_rowwise_binding_warns_within_week_of_expiry() -> None:
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    register = {
        "census_disclosure_control_noise": _reviewed_register_entry(
            approved_on="2026-01-01",
            expires_on="2026-03-05",
        )
    }
    with pytest.warns(UserWarning, match="within one week"):
        receipt = require_adjudicated_uk_local_binding(
            ["census_households/constituency", "tenure/constituency"],
            problem.target_frame,
            register=register,
            now=date(2026, 3, 1),
        )
    assert (
        "census_disclosure_control_noise"
        in receipt["stood_on"]["census_households/constituency"]
    )

    # Far from expiry the same in-force entry passes silently.
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        require_adjudicated_uk_local_binding(
            ["census_households/constituency", "tenure/constituency"],
            problem.target_frame,
            register=register,
            now=date(2026, 1, 15),
        )


def test_committed_local_binding_register_references_committed_census() -> None:
    census = load_uk_local_target_census()
    fence_ids = {row["fence_id"] for row in census["binding_fences"]}
    register = load_uk_reviewed_exclusion_register(
        None,
        resource=UK_LOCAL_BINDING_ADJUDICATION_REGISTER_RESOURCE,
    )
    assert set(register) <= fence_ids
    for fence_id, record in register.items():
        assert f"uk_local_target_census.json#/binding_fences/{fence_id}" in (
            record.reason
        )
        assert len(record.reason) > 100


def test_rowwise_doctrine_solve_exposes_no_knobs() -> None:
    import inspect

    parameters = inspect.signature(solve_uk_rowwise_weights_under_doctrine).parameters
    for forbidden in (
        "target_loss_weights",
        "target_loss_scales",
        "target_loss_cap",
        "max_weight_ratio",
        "doctrine",
    ):
        assert forbidden not in parameters

    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    with pytest.raises(TypeError):
        solve_uk_rowwise_weights_under_doctrine(
            _clone_frame(),
            problem,
            bound_families=[
                "census_households/constituency",
                "tenure/constituency",
            ],
            target_loss_weights=[1.0] * 4,
        )


def test_rowwise_doctrine_solve_refuses_duplicate_surface() -> None:
    import dataclasses

    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    doctored = dataclasses.replace(
        problem,
        target_frame=pd.concat(
            [problem.target_frame, problem.target_frame.iloc[[0]]],
            ignore_index=True,
        ),
    )
    with pytest.raises(ValueError, match="per-target weights"):
        solve_uk_rowwise_weights_under_doctrine(
            _clone_frame(),
            doctored,
            bound_families=[
                "census_households/constituency",
                "tenure/constituency",
            ],
            epochs=1,
        )


def test_rowwise_area_support_summary_reports_all_target_areas() -> None:
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    weights = np.array([2.0, 0.0, 5.0])
    support = rowwise_area_support_summary(
        problem,
        weights,
        source_household_ids=[1, 1, 2],
    )
    rows = {row.area_code: row for row in support.itertuples(index=False)}
    assert rows["E001"].nonzero_households == 1
    assert rows["E001"].nonzero_source_households == 1
    assert rows["E001"].weight_sum == pytest.approx(2.0)
    assert rows["S001"].nonzero_households == 1
    assert rows["S001"].weight_sum == pytest.approx(5.0)
    assert rows["E001"].effective_sample_size == pytest.approx(1.0)


def test_rowwise_area_support_wrapper_equals_frame_agnostic_core() -> None:
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    weights = np.array([2.0, 0.0, 5.0])
    source_household_ids = [(2023, 1), (2023, 1), (2023, 2)]

    wrapped = rowwise_area_support_summary(
        problem,
        weights,
        source_household_ids=source_household_ids,
    )
    core = uk_area_support_summary(
        problem.assigned_areas,
        weights,
        area_codes=problem.area_codes,
        source_household_ids=source_household_ids,
    )

    pd.testing.assert_frame_equal(wrapped, core)


def test_ladder_area_support_includes_zeros_and_distinct_cloned_sources() -> None:
    ladder = type(
        "Ladder",
        (),
        {
            "constituency_code": np.asarray(["C1", "C2", "C3"]),
            "local_authority_code": np.asarray(["L1", "L2", "L3"]),
        },
    )()
    household = pd.DataFrame(
        {
            "household_id": [101, 102, 201, 202],
            "source_household_id": [10, 10, 20, 20],
            "household_weight": [2.0, 3.0, 4.0, 0.0],
            "constituency_code": ["C1", "C1", "C2", "C2"],
            "local_authority_code": ["L1", "L1", "L2", "L2"],
        }
    )

    summaries = uk_ladder_area_support_summary(household, ladder)

    constituency = summaries["constituency"].set_index("area_code")
    assert constituency.loc["C1", "assigned_households"] == 2
    assert constituency.loc["C1", "nonzero_households"] == 2
    assert constituency.loc["C1", "nonzero_source_households"] == 1
    assert constituency.loc["C2", "nonzero_source_households"] == 1
    assert constituency.loc["C3", "assigned_households"] == 0
    assert constituency.loc["C3", "effective_sample_size"] == 0.0
    assert summaries["la"]["area_code"].tolist() == ["L1", "L2", "L3"]


def test_ladder_area_support_refuses_missing_source_column() -> None:
    ladder = type(
        "Ladder",
        (),
        {
            "constituency_code": np.asarray(["C1"]),
            "local_authority_code": np.asarray(["L1"]),
        },
    )()
    household = pd.DataFrame(
        {
            "household_id": [1],
            "household_weight": [1.0],
            "constituency_code": ["C1"],
            "local_authority_code": ["L1"],
        }
    )

    with pytest.raises(ValueError, match="distinct-source honesty requires it"):
        uk_ladder_area_support_summary(household, ladder)


def test_matrix_builder_fails_closed_on_unreachable_nonzero_targets() -> None:
    # A target area with no assigned households cannot be hit; a nonzero
    # target there must refuse at build time, while a zero target is fine.
    targets = pd.DataFrame(
        {
            "code": ["E001", "S001", "W001"],
            "households": [4.0, 2.0, 5.0],
            "tenure/social_rent": [1.0, 1.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="W001/households"):
        build_uk_rowwise_local_matrix(_metrics(), _assigned(), targets)

    zero_ok = targets.copy()
    zero_ok.loc[zero_ok["code"] == "W001", "households"] = 0.0
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), zero_ok)
    assert problem.n_areas == 3


def test_matrix_builder_refuses_duplicate_and_metadata_metric_labels() -> None:
    duplicated = _metrics()
    duplicated.columns = ["households", "households"]
    with pytest.raises(ValueError, match="duplicate column label"):
        build_uk_rowwise_local_matrix(duplicated, _assigned(), _targets())

    metadata = _metrics().rename(columns={"tenure/social_rent": "area_index"})
    targets = _targets().rename(columns={"tenure/social_rent": "area_index"})
    with pytest.raises(ValueError, match="metadata"):
        build_uk_rowwise_local_matrix(metadata, _assigned(), targets)


def test_rowwise_solve_refuses_dead_rows_and_misaligned_frames() -> None:
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    with pytest.raises(ValueError, match="zero"):
        solve_uk_rowwise_weights_under_doctrine(
            _clone_frame([1.0, 0.0, 1.0]),
            problem,
            bound_families=[
                "census_households/constituency",
                "tenure/constituency",
            ],
            epochs=1,
        )

    # The same households in a different order are refused with the ordering
    # named, not a useless equal-counts message (and never realigned).
    reordered = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "person_household_id": [101, 102, 103],
                "person_benunit_id": [11, 12, 13],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [11, 12, 13]}),
        household=pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "household_weight": [1.0, 1.0, 1.0],
            }
        ),
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
    )
    reordered_problem = build_uk_rowwise_local_matrix(
        _metrics().reindex([102, 101, 103]),
        _assigned().reindex([102, 101, 103]),
        _targets(),
    )
    with pytest.raises(ValueError, match="different order.*row 0.*101.*102"):
        solve_uk_rowwise_weights_under_doctrine(
            reordered,
            reordered_problem,
            bound_families=[
                "census_households/constituency",
                "tenure/constituency",
            ],
            epochs=1,
        )

    # A frame whose household rows do not match the problem's households
    # cannot express the declared surface and is refused, not realigned.
    misaligned = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1],
                "person_household_id": [999],
                "person_benunit_id": [11],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [11]}),
        household=pd.DataFrame({"household_id": [999], "household_weight": [1.0]}),
        time_period="2023",
    )
    with pytest.raises(ValueError, match="match the problem"):
        solve_uk_rowwise_weights_under_doctrine(
            misaligned,
            problem,
            bound_families=[
                "census_households/constituency",
                "tenure/constituency",
            ],
            epochs=1,
        )


def test_calibration_mass_reason_names_families() -> None:
    reason = rowwise_calibration_mass_reason(["census_households/constituency"])
    assert "census_households/constituency" in reason
    assert "calibration" in reason

    with pytest.raises(ValueError, match="bound_families"):
        rowwise_calibration_mass_reason([])


def test_support_summary_normalizes_inputs() -> None:
    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    with pytest.raises(ValueError, match="one-dimensional"):
        rowwise_area_support_summary(problem, np.ones((3, 1)))
    # Composite (tuple) source ids must be handled, not coerced into 2-D.
    support = rowwise_area_support_summary(
        problem,
        [1.0, 1.0, 1.0],
        source_household_ids=[(2023, 1), (2023, 1), (2023, 2)],
    )
    rows = {row.area_code: row for row in support.itertuples(index=False)}
    assert rows["E001"].nonzero_source_households == 1


def test_doctrine_solve_forwards_declared_bounds_to_the_front_door(
    monkeypatch,
) -> None:
    """The doctrine's reviewed constants ride into calibrate() explicitly.

    The pre-migration solver defaults (epochs 512, learning rate 0.15) and
    the doctrine bounds (ratio 100.0, cap 10.0) differ from calibrate()'s
    own defaults, so silent default-drift would change solve behaviour;
    this pin fails if any of them stops being forwarded.
    """

    import microcosm.build.uk_runtime.local_rowwise as local_rowwise

    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    forwarded: dict[str, object] = {}
    real_calibrate = local_rowwise.calibrate

    def spy(frame, targets, **kwargs):
        forwarded.update(kwargs)
        return real_calibrate(frame, targets, **kwargs)

    monkeypatch.setattr(local_rowwise, "calibrate", spy)
    solve_uk_rowwise_weights_under_doctrine(
        _clone_frame(),
        problem,
        bound_families=[
            "census_households/constituency",
            "tenure/constituency",
        ],
        seed=3,
    )

    assert forwarded["epochs"] == 512
    assert forwarded["learning_rate"] == 0.15
    assert forwarded["max_weight_ratio"] == UK_LOCAL_MAX_WEIGHT_RATIO
    assert forwarded["target_loss_cap"] == UK_LOCAL_TARGET_LOSS_CAP
    assert forwarded["mass"] == "free"
    assert forwarded["seed"] == 3
    assert "census_households/constituency" in forwarded["mass_reason"]


def test_doctrine_solve_refuses_reordered_diagnostics(monkeypatch) -> None:
    """A front-door result whose diagnostics are reordered is refused by name.

    The evidence tables consume diagnostics positionally, and target values
    legitimately repeat on a local surface, so value equality alone could
    pass a reordering by coincidence; the solve asserts per-row name
    alignment against the declared surface instead.
    """

    import dataclasses

    import microcosm.build.uk_runtime.local_rowwise as local_rowwise

    problem = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    real_calibrate = local_rowwise.calibrate

    def reordering(frame, targets, **kwargs):
        result = real_calibrate(frame, targets, **kwargs)
        return dataclasses.replace(
            result, diagnostics=tuple(reversed(result.diagnostics))
        )

    monkeypatch.setattr(local_rowwise, "calibrate", reordering)
    with pytest.raises(ValueError, match="not aligned.*row 0"):
        solve_uk_rowwise_weights_under_doctrine(
            _clone_frame(),
            problem,
            bound_families=[
                "census_households/constituency",
                "tenure/constituency",
            ],
            epochs=1,
        )
