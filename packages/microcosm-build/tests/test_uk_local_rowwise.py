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

import json
from datetime import date
from pathlib import Path

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
                "target_name": "ons.census.households@L2",
                "family": "census_households",
            },
            {
                "area_type": "constituency",
                "area_code": "S001",
                "metric": "households",
                "value": 1.0,
                "target_name": "ons.census.households@S001",
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
                "target_name": "ons.census.households@E001",
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
                "target_name": "ons.census.households@L1",
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


def test_dense_wrapper_and_long_format_builder_agree() -> None:
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
        return type("Solve", (), {"weights": np.ones(3), "selected_support": None})()

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


def test_dense_builder_result_equals_frozen_pre_pr_fixture() -> None:
    """Exact equality against the result main's builder produced on this fixture.

    The golden was generated from ``origin/main``'s ``local_rowwise.py`` before
    the long-format assembler landed, so this pins the bit-for-bit claim rather
    than restating the delegation.
    """

    golden = json.loads(
        (
            Path(__file__).parent / "golden" / "uk_local_rowwise_dense_fixture.json"
        ).read_text(encoding="utf-8")
    )
    dense = build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets())
    matrix = dense.matrix.tocsr()
    assert matrix.indptr.tolist() == golden["indptr"]
    assert matrix.indices.tolist() == golden["indices"]
    assert matrix.data.tolist() == golden["data"]
    assert list(dense.targets) == golden["targets"]
    assert list(dense.area_codes) == golden["area_codes"]
    assert list(dense.assigned_areas) == golden["assigned_areas"]
    assert list(dense.household_ids) == golden["household_ids"]
    assert list(dense.metric_names) == golden["metric_names"]
    assert np.asarray(dense.metric_values).tolist() == golden["metric_values"]
    frozen = pd.DataFrame(golden["target_frame"])
    common = [column for column in frozen.columns if column in dense.target_frame]
    assert common == list(frozen.columns)
    # Exact, like the matrix: the golden's values round-trip through JSON
    # without loss, so a tolerance would only hide a drift.
    pd.testing.assert_frame_equal(
        dense.target_frame[common].reset_index(drop=True),
        frozen[common].reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )


def test_doctrine_solve_refuses_a_reordering_restore() -> None:
    from types import SimpleNamespace

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

    def reordering_restore(frame):
        # uk_national_frame refuses an unsorted id column, so a reordered
        # frame cannot be built through it; the guard must catch the stand-in
        # before anything else reads the restored result.
        reversed_household = frame.table("household").iloc[::-1].reset_index(drop=True)
        return SimpleNamespace(
            table=lambda entity: reversed_household,
            entities=("household",),
        )

    with pytest.raises(ValueError, match="same ids, same order"):
        solve_uk_rowwise_weights_under_doctrine(
            prepared,
            build_uk_rowwise_local_matrix(_metrics(), _assigned(), _targets()),
            bound_families=["census_households/constituency", "tenure/constituency"],
            restore=reordering_restore,
            epochs=2,
        )


def test_exact_size_preserves_targets_and_household_links():
    from microcosm.build.uk_runtime.dataset_size import refit_uk_dataset_size
    from microcosm.calibrate import Target, TargetSet, calibrate

    frame = _clone_frame()
    dense = calibrate(
        frame,
        TargetSet(
            [Target("count", "household", lambda f: np.ones(f.n("household")), 3)]
        ),
        epochs=2,
    )
    sized = refit_uk_dataset_size(
        frame, dense, households=2, epochs=2, learning_rate=0.02, seed=7
    )
    assert sized.result.frame.n("household") == 2
    assert sized.result.frame.n("person") == 2
    assert sized.result.frame.n("benunit") == 2
    assert sized.result.problem.names == dense.problem.names
    assert sized.result.initial_weights.sum() == pytest.approx(
        dense.initial_weights.sum()
    )
    assert (sized.result.weights <= 10 * sized.result.initial_weights).all()
    again = refit_uk_dataset_size(
        frame, dense, households=2, epochs=2, learning_rate=0.02, seed=7
    )
    np.testing.assert_array_equal(sized.support, again.support)


@pytest.mark.parametrize("size", [0, -1, True, 4, 1.5])
def test_exact_size_refuses_invalid_sizes(size):
    from microcosm.build.uk_runtime.dataset_size import refit_uk_dataset_size
    from microcosm.calibrate import Target, TargetSet, calibrate

    frame = _clone_frame()
    dense = calibrate(
        frame,
        TargetSet(
            [Target("count", "household", lambda f: np.ones(f.n("household")), 3)]
        ),
        epochs=1,
    )
    with pytest.raises(ValueError, match="integer"):
        refit_uk_dataset_size(
            frame, dense, households=size, epochs=1, learning_rate=0.02, seed=7
        )


def test_size_solve_restores_full_prepared_tables_before_subsetting():
    frame = _clone_frame()
    metrics = pd.DataFrame({"households": [1.0, 1.0, 1.0]}, index=[101, 102, 103])
    problem = build_uk_rowwise_local_matrix(
        metrics,
        _assigned(),
        pd.DataFrame({"code": ["E001", "S001"], "households": [2.0, 1.0]}),
    )
    restored_counts = []

    def restore(full):
        restored_counts.append(full.n("household"))
        return full

    result = solve_uk_rowwise_weights_under_doctrine(
        frame,
        problem,
        bound_families=["census_households/constituency"],
        dataset_households=2,
        epochs=2,
        seed=7,
        restore=restore,
    )
    assert restored_counts == [3]
    assert result.frame.n("household") == 2
    assert result.frame.n("person") == 2
    assert len(result.diagnostics) == 2
    assert "census_households/constituency" in result.frame.mass_log[-1].reason
    assert result.selected_support.tolist() == [0, 2]


def test_size_solve_keeps_its_dense_reference_and_selection_seed_moves_only_the_draw():
    frame = _clone_frame()
    metrics = pd.DataFrame({"households": [1.0, 1.0, 1.0]}, index=[101, 102, 103])
    problem = build_uk_rowwise_local_matrix(
        metrics,
        _assigned(),
        pd.DataFrame({"code": ["E001", "S001"], "households": [2.0, 1.0]}),
    )
    common = dict(
        bound_families=["census_households/constituency"],
        epochs=2,
        seed=7,
    )
    dense = solve_uk_rowwise_weights_under_doctrine(frame, problem, **common)
    sized = solve_uk_rowwise_weights_under_doctrine(
        frame, problem, dataset_households=2, **common
    )
    assert dense.dense_reference is None
    reference = sized.dense_reference
    assert reference is not None
    # The reference is the standalone dense solve, byte for byte.
    np.testing.assert_array_equal(reference.weights, dense.weights)
    np.testing.assert_array_equal(reference.initial_weights, dense.initial_weights)
    assert reference.final_loss == dense.final_loss
    assert reference.final_loss == sized.size_receipt["dense_loss"]
    assert reference.initial_loss == dense.initial_loss
    assert reference.n_nonzero == dense.n_nonzero
    pd.testing.assert_frame_equal(reference.diagnostics, dense.diagnostics)
    pd.testing.assert_frame_equal(
        reference.national_diagnostics, dense.national_diagnostics
    )
    assert dict(reference.past_cap_census) == dict(dense.past_cap_census)
    assert dict(reference.all_past_cap_census) == dict(dense.all_past_cap_census)
    # The shipped product is the compact refit, not the reference.
    assert sized.weights.size == 2
    assert reference.weights.size == 3
    # A selection seed moves the draw only: same pool, same dense reference.
    other = solve_uk_rowwise_weights_under_doctrine(
        frame, problem, dataset_households=2, selection_seed=11, **common
    )
    np.testing.assert_array_equal(other.dense_reference.weights, reference.weights)
    assert other.size_receipt["seed"] == 11
    assert sized.size_receipt["seed"] == 7
    assert other.size_receipt["dense_loss"] == sized.size_receipt["dense_loss"]


def test_selection_feasibility_measures_boundary_mass_and_the_ways_out():
    from microcosm.build.uk_runtime.dataset_size import selection_feasibility

    # Two protected certainties, then a boundary whose largest gate (0.9)
    # cannot be scaled to a draw of 5 out of mass 0.9 + 4 * 0.2 = 1.7.
    pi = np.array([1.0, 1.0, 0.9, 0.2, 0.2, 0.2, 0.2, 0.0])
    protected = np.array([True, True, False, False, False, False, False, False])
    report = selection_feasibility(
        pi, 7, protected=protected, n_nonzero=7, l0_lambda=0.5
    )
    assert report["certainties_at_pi_hi_1"] == 2
    assert report["boundary_draw"] == 5
    assert report["boundary_positive_gates"] == 5
    assert report["boundary_mass"] == pytest.approx(1.7)
    assert report["boundary_max"] == pytest.approx(0.9)
    assert report["feasible_at_pi_hi_1"] is False
    # At pi_hi=1 the boundary supports floor(1.7 / 0.9) = 1 draw: k <= 3.
    assert report["max_feasible_households_at_pi_hi_1"] == 3
    # Promoting the 0.9 gate to a certainty (pi_hi <= 0.9) leaves a draw of 4
    # over four equal 0.2 gates: feasible; 0.99 and above are not.
    assert report["pi_hi_scan"]["0.99"]["feasible"] is False
    assert report["pi_hi_scan"]["0.99"]["reason"] == "boundary_mass_short"
    assert report["pi_hi_scan"]["0.9"]["feasible"] is True
    assert report["pi_hi_scan"]["0.9"]["reason"] == "feasible"
    assert report["smallest_feasible_pi_hi_on_grid"] == 0.9
    assert report["requested_pi_hi_verdict"] == "boundary_mass_short"
    assert report["budget_search"] is None
    assert report["gates_above"]["0.5"] == 3
    assert report["budget_search_n_nonzero"] == 7
    assert report["selection_l0_lambda"] == 0.5
    assert report["requested_pi_hi"] == 1.0
    assert report["feasible_at_requested_pi_hi"] is False
    at_half = selection_feasibility(
        pi, 7, protected=protected, n_nonzero=7, l0_lambda=0.5, requested_pi_hi=0.9
    )
    assert at_half["requested_pi_hi"] == 0.9
    assert at_half["feasible_at_requested_pi_hi"] is True

    feasible = selection_feasibility(
        np.array([1.0, 0.5, 0.5, 0.5, 0.5]),
        3,
        protected=np.array([True, False, False, False, False]),
        n_nonzero=5,
        l0_lambda=0.1,
    )
    assert feasible["feasible_at_pi_hi_1"] is True
    assert feasible["smallest_feasible_pi_hi_on_grid"] == 1.0


def test_size_refit_pi_hi_promotes_learned_certainties_and_is_recorded():
    from microcosm.build.uk_runtime.dataset_size import refit_uk_dataset_size
    from microcosm.calibrate import Target, TargetSet, calibrate

    frame = _clone_frame()
    dense = calibrate(
        frame,
        TargetSet(
            [Target("count", "household", lambda f: np.ones(f.n("household")), 3)]
        ),
        epochs=2,
    )
    sized = refit_uk_dataset_size(
        frame, dense, households=2, epochs=2, learning_rate=0.02, seed=7, pi_hi=0.5
    )
    receipt = sized.receipt
    assert receipt["selection_pi_hi"] == 0.5
    assert receipt["selection_receipt"]["pi_hi"] == 0.5
    assert receipt["selection_budget_basis"] == "open_probability_mass"
    assert (
        receipt["selection_feasibility"]["budget_search_basis"]
        == "open_probability_mass"
    )
    assert receipt["selection_feasibility"]["requested_pi_hi"] == 0.5
    assert receipt["selection_feasibility"]["feasible_at_requested_pi_hi"] is True
    # The budget search stopped on the draw's own feasibility at the requested
    # threshold and recorded every probe (microcosm#355, S2 refusal).
    search = receipt["selection_budget_search"]
    assert search["feasible_draw_pi_hi"] == 0.5
    assert search["selected_feasible"] is True
    assert search["stopped_on"] == "acceptable_within_tolerance"
    assert search["probes"][-1]["verdict"] == "feasible"
    assert receipt["selection_feasibility"]["budget_search"] is search
    # Certainties at 0.5 can only grow relative to the exact-one set.
    assert (
        receipt["selection_receipt"]["certainty_count"] >= receipt["protected_carriers"]
    )
    assert sized.result.frame.n("household") == 2
    for bad in (0.0, 1.5, -0.1, True):
        with pytest.raises(ValueError, match="pi_hi"):
            refit_uk_dataset_size(
                frame,
                dense,
                households=2,
                epochs=2,
                learning_rate=0.02,
                seed=7,
                pi_hi=bad,
            )


def test_size_refit_baseline_pi_floor_trims_the_baseline_and_is_recorded():
    from microcosm.build.uk_runtime.dataset_size import refit_uk_dataset_size
    from microcosm.calibrate import Target, TargetSet, calibrate

    frame = _clone_frame()
    dense = calibrate(
        frame,
        TargetSet(
            [Target("count", "household", lambda f: np.ones(f.n("household")), 3)]
        ),
        epochs=2,
    )
    common = dict(households=2, epochs=2, learning_rate=0.02, seed=7, pi_hi=0.5)
    plain = refit_uk_dataset_size(frame, dense, **common)
    floored = refit_uk_dataset_size(frame, dense, baseline_pi_floor=1.0, **common)
    assert plain.receipt["baseline_pi_floor"] == 0.0
    assert plain.receipt["refit_baseline"] == "normalized_horvitz_thompson_w_over_q"
    assert plain.receipt["stretch_reference"] == plain.receipt["refit_baseline"]
    assert plain.receipt["baseline_floored_rows"] == 0
    assert floored.receipt["baseline_pi_floor"] == 1.0
    assert (
        floored.receipt["refit_baseline"]
        == "normalized_horvitz_thompson_w_over_q_floored"
    )
    assert floored.receipt["stretch_reference"] == floored.receipt["refit_baseline"]
    # The same draw either way: the floor is a refit setting, not a selection one.
    assert floored.receipt["pool_row_indices"] == plain.receipt["pool_row_indices"]
    assert (
        floored.receipt["inclusion_probabilities"]
        == (plain.receipt["inclusion_probabilities"])
    )
    q = np.asarray(plain.receipt["inclusion_probabilities"])
    assert floored.receipt["baseline_floored_rows"] == int((q < 1.0).sum())
    for receipt in (plain.receipt, floored.receipt):
        share = receipt["baseline_mass_share_certainties"]
        assert share is None or 0.0 <= share <= 1.0
    # With every probability floored at one, the baseline is the dense weights
    # renormalised to the pool mass, so no row is inflated by its draw.
    baseline = np.asarray(floored.result.initial_weights)
    dense_selected = np.asarray(dense.weights)[np.asarray(floored.support)]
    assert np.allclose(baseline / baseline.sum(), dense_selected / dense_selected.sum())
    for bad in (-0.1, 1.5, True):
        with pytest.raises(ValueError, match="baseline_pi_floor"):
            refit_uk_dataset_size(frame, dense, baseline_pi_floor=bad, **common)


def test_size_refit_refuses_unsupported_nonzero_targets_by_name():
    from microcosm.build.uk_runtime.dataset_size import (
        refit_uk_dataset_size,
        unsupported_nonzero_targets,
    )
    from microcosm.calibrate import Target, TargetSet, calibrate

    frame = _clone_frame()
    dense = calibrate(
        frame,
        TargetSet(
            [
                Target("count", "household", lambda f: np.ones(f.n("household")), 3),
                Target("nobody", "household", lambda f: np.zeros(f.n("household")), 5),
            ]
        ),
        epochs=2,
    )
    assert [
        name.split("@")[0] for name in unsupported_nonzero_targets(dense.problem)
    ] == ["nobody"]
    with pytest.raises(ValueError, match="no supporting household.*nobody"):
        refit_uk_dataset_size(
            frame, dense, households=2, epochs=2, learning_rate=0.02, seed=7
        )


def test_size_refusal_at_the_draw_carries_the_feasibility_numbers(monkeypatch):
    import microcosm.build.uk_runtime.dataset_size as sizing
    from microcosm.calibrate import Target, TargetSet, calibrate

    frame = _clone_frame()
    dense = calibrate(
        frame,
        TargetSet(
            [Target("count", "household", lambda f: np.ones(f.n("household")), 3)]
        ),
        epochs=2,
    )

    def refuse(*_args, **_kwargs):
        raise ValueError("degenerate boundary mass: synthetic refusal.")

    monkeypatch.setattr(sizing, "select_exact_k", refuse)
    with pytest.raises(ValueError) as caught:
        sizing.refit_uk_dataset_size(
            frame, dense, households=2, epochs=2, learning_rate=0.02, seed=7
        )
    message = str(caught.value)
    assert message.startswith("degenerate boundary mass: synthetic refusal.")
    assert "Selection feasibility (requested pi_hi=1): " in message
    payload = json.loads(
        message.split("Selection feasibility (requested pi_hi=1): ", 1)[1]
    )
    assert payload["requested_pi_hi"] == 1.0
    assert payload["requested_households"] == 2
    assert payload["pool_households"] == 3
    assert "pi_hi_scan" in payload


def test_size_holdout_uses_compact_support_and_reselects_each_fold(monkeypatch):
    import microcosm.build.uk_runtime.local_rowwise as runtime

    metrics = pd.DataFrame({"households": [1.0, 2.0, 3.0]}, index=[101, 102, 103])
    calls = []

    def solve(frame, training, **kwargs):
        calls.append((len(training.targets), kwargs["dataset_households"]))
        return type(
            "Solve",
            (),
            {"weights": np.array([1.0, 1.0]), "selected_support": np.array([0, 2])},
        )()

    monkeypatch.setattr(runtime, "solve_uk_rowwise_weights_under_doctrine", solve)
    # The holdout helper needs at least one target in each of five folds.
    richer = build_uk_rowwise_local_matrix(
        pd.DataFrame(
            {
                name: [1.0, 2.0, 3.0]
                for name in ("households", "tenure/social_rent", "tenure/private_rent")
            },
            index=metrics.index,
        ),
        _assigned(),
        pd.DataFrame(
            {
                "code": ["E001", "S001"],
                **{
                    name: [3.0, 3.0]
                    for name in (
                        "households",
                        "tenure/social_rent",
                        "tenure/private_rent",
                    )
                },
            }
        ),
    )
    monkeypatch.setattr(
        runtime, "_derive_uk_local_bound_families_from_target_frame", lambda *a, **k: ()
    )
    receipt = rotated_uk_local_holdout(
        _clone_frame(), richer, dataset_households=2, epochs=1
    )
    assert len(calls) == 5
    assert all(n < len(richer.targets) and k == 2 for n, k in calls)
    assert np.isfinite(receipt["mean_holdout_loss"])


def test_size_refit_freezes_population_dependent_measures():
    from microcosm.build.uk_runtime.dataset_size import refit_uk_dataset_size
    from microcosm.calibrate import Target, TargetSet, calibrate

    frame = _clone_frame()
    # A population-normalized measure changes if evaluated on two instead of
    # three households. The refit must retain the full-pool value of 1/3.
    targets = TargetSet(
        [
            Target(
                "normalized",
                "household",
                lambda f: np.full(f.n("household"), 1 / f.n("household")),
                1,
            )
        ]
    )
    dense = calibrate(frame, targets, epochs=2)
    small = refit_uk_dataset_size(
        frame, dense, households=2, epochs=2, learning_rate=0.02, seed=7
    )
    np.testing.assert_allclose(small.result.problem.matrix.toarray(), [[1 / 3, 1 / 3]])


def test_full_size_returns_dense_reference_without_search():
    from microcosm.build.uk_runtime.dataset_size import refit_uk_dataset_size
    from microcosm.calibrate import Target, TargetSet, calibrate

    frame = _clone_frame()
    dense = calibrate(
        frame,
        TargetSet(
            [Target("count", "household", lambda f: np.ones(f.n("household")), 3)]
        ),
        epochs=1,
    )
    full = refit_uk_dataset_size(
        frame, dense, households=3, epochs=1, learning_rate=0.02, seed=7
    )
    assert full.result is dense
    assert full.support.tolist() == [0, 1, 2]


def test_size_refuses_budget_smaller_than_protected_carriers():
    from microcosm.build.uk_runtime.dataset_size import refit_uk_dataset_size
    from microcosm.calibrate import Target, TargetSet, calibrate

    frame = _clone_frame()
    targets = TargetSet(
        [
            Target(
                str(i),
                "household",
                lambda f, i=i: (
                    f.table("household").household_id.to_numpy() == i
                ).astype(float),
                1,
            )
            for i in [101, 102, 103]
        ]
    )
    dense = calibrate(frame, targets, epochs=1)
    with pytest.raises(ValueError, match="3 protected target carriers"):
        refit_uk_dataset_size(
            frame, dense, households=2, epochs=1, learning_rate=0.02, seed=7
        )


def _size_problem():
    metrics = pd.DataFrame({"households": [1.0, 1.0, 1.0]}, index=[101, 102, 103])
    return build_uk_rowwise_local_matrix(
        metrics,
        _assigned(),
        pd.DataFrame({"code": ["E001", "S001"], "households": [2.0, 1.0]}),
    )


def test_size_selection_can_be_searched_once_and_drawn_from_again():
    from microcosm.build.uk_runtime.dataset_size import (
        refit_uk_dataset_size,
        select_uk_dataset_size,
    )
    from microcosm.calibrate import Target, TargetSet, calibrate

    frame = _clone_frame()
    dense = calibrate(
        frame,
        TargetSet(
            [Target("count", "household", lambda f: np.ones(f.n("household")), 3)]
        ),
        epochs=2,
    )
    one_shot = refit_uk_dataset_size(
        frame, dense, households=2, epochs=2, learning_rate=0.02, seed=7, pi_hi=0.5
    )
    selection = select_uk_dataset_size(
        frame, dense, households=2, epochs=2, learning_rate=0.02, seed=7, pi_hi=0.5
    )
    assert selection.search_pi_hi == 0.5
    assert selection.selection.l0_lambda == one_shot.receipt["selection_l0_lambda"]
    reused = refit_uk_dataset_size(
        frame,
        dense,
        households=2,
        epochs=2,
        learning_rate=0.02,
        seed=7,
        pi_hi=0.5,
        selection=selection,
    )
    np.testing.assert_array_equal(reused.support, one_shot.support)
    np.testing.assert_array_equal(reused.result.weights, one_shot.result.weights)
    assert one_shot.receipt["selection_reused"] is False
    assert reused.receipt["selection_reused"] is True
    assert reused.receipt["selection_search_pi_hi"] == 0.5
    # A different draw threshold on the same search is allowed and recorded.
    redrawn = refit_uk_dataset_size(
        frame,
        dense,
        households=2,
        epochs=2,
        learning_rate=0.02,
        seed=7,
        pi_hi=1.0,
        selection=selection,
    )
    assert redrawn.receipt["selection_pi_hi"] == 1.0
    assert redrawn.receipt["selection_search_pi_hi"] == 0.5
    assert redrawn.receipt["selection_feasibility"]["search_pi_hi"] == 0.5
    # Different search inputs refuse by name.
    with pytest.raises(ValueError, match="epochs: selection 2 != 3"):
        refit_uk_dataset_size(
            frame,
            dense,
            households=2,
            epochs=3,
            learning_rate=0.02,
            seed=7,
            selection=selection,
        )
    with pytest.raises(ValueError, match="full-pool"):
        select_uk_dataset_size(
            frame, dense, households=3, epochs=2, learning_rate=0.02, seed=7
        )


def test_size_checkpoint_resumes_the_draw_on_the_rederived_pool(tmp_path):
    from microcosm.build.uk_runtime.size_checkpoint import (
        SIZE_CHECKPOINT_ARRAYS_FILENAME,
        SIZE_CHECKPOINT_MANIFEST_FILENAME,
        load_uk_size_checkpoint,
    )

    frame = _clone_frame()
    problem = _size_problem()
    identity = {"input_sha256": "abc", "seed": 7, "epochs": 2, "households": 2}
    common = dict(
        bound_families=["census_households/constituency"],
        dataset_households=2,
        epochs=2,
        seed=7,
    )
    checkpoint_dir = tmp_path / "run-a"
    first = solve_uk_rowwise_weights_under_doctrine(
        frame,
        problem,
        selection_pi_hi=0.5,
        size_checkpoint_dir=checkpoint_dir,
        checkpoint_identity=identity,
        **common,
    )
    assert (checkpoint_dir / SIZE_CHECKPOINT_ARRAYS_FILENAME).is_file()
    manifest = json.loads(
        (checkpoint_dir / SIZE_CHECKPOINT_MANIFEST_FILENAME).read_text()
    )
    assert manifest["identity"] == identity
    assert manifest["selection"]["search_pi_hi"] == 0.5
    assert manifest["pool"]["households"] == 3
    written = first.size_receipt["checkpoint"]["written"]
    assert written["stage"] == "before_exact_count_draw"
    assert written["arrays_sha256"] == manifest["arrays_sha256"]
    # Nothing in the receipt differs between identical runs (Vahid, #877):
    # no timestamp, no absolute path.
    assert "written_at" not in written and "directory" not in written
    assert first.size_receipt["certainty_share"] == pytest.approx(
        first.size_receipt["selection_receipt"]["certainty_count"] / 2
    )
    assert (
        first.size_receipt["boundary_draws"]
        == 2 - (first.size_receipt["selection_receipt"]["certainty_count"])
    )
    assert first.size_receipt["zero_target_rows"] == 0

    # Resume: no dense solve, no search; the draw and refit reproduce the run.
    resumed = solve_uk_rowwise_weights_under_doctrine(
        frame,
        problem,
        selection_pi_hi=0.5,
        resume_size_checkpoint=checkpoint_dir,
        checkpoint_identity=identity,
        **common,
    )
    np.testing.assert_array_equal(resumed.selected_support, first.selected_support)
    np.testing.assert_array_equal(resumed.weights, first.weights)
    assert resumed.final_loss == first.final_loss
    assert resumed.dense_reference is not None
    assert resumed.dense_reference.final_loss == first.dense_reference.final_loss
    np.testing.assert_array_equal(
        resumed.dense_reference.weights, first.dense_reference.weights
    )
    assert resumed.size_receipt["selection_reused"] is True
    assert resumed.size_receipt["checkpoint"]["resumed_from"]["identity"] == identity
    assert (
        resumed.size_receipt["selection_l0_lambda"]
        == (first.size_receipt["selection_l0_lambda"])
    )
    # Another draw threshold on the same checkpoint is a candidate knob, recorded.
    redrawn = solve_uk_rowwise_weights_under_doctrine(
        frame,
        problem,
        selection_pi_hi=1.0,
        resume_size_checkpoint=checkpoint_dir,
        checkpoint_identity=identity,
        **common,
    )
    assert redrawn.size_receipt["selection_pi_hi"] == 1.0
    assert redrawn.size_receipt["selection_search_pi_hi"] == 0.5

    # A resume under a changed doctrine refuses even when the identity was
    # not asked to carry it (the second lock, Vahid's should-fix 1).
    import dataclasses

    from microcosm.build.uk_runtime import local_rowwise as lr_module

    drifted_doctrine = dataclasses.replace(
        lr_module.UK_LOCAL_SOLVE_DOCTRINE,
        max_weight_ratio=lr_module.UK_LOCAL_SOLVE_DOCTRINE.max_weight_ratio * 2,
    )
    original_doctrine = lr_module.UK_LOCAL_SOLVE_DOCTRINE
    lr_module.UK_LOCAL_SOLVE_DOCTRINE = drifted_doctrine
    try:
        with pytest.raises(ValueError, match="different doctrine.*max_weight_ratio"):
            solve_uk_rowwise_weights_under_doctrine(
                frame,
                problem,
                resume_size_checkpoint=checkpoint_dir,
                checkpoint_identity=identity,
                **common,
            )
    finally:
        lr_module.UK_LOCAL_SOLVE_DOCTRINE = original_doctrine
    provenance_run = solve_uk_rowwise_weights_under_doctrine(
        frame,
        problem,
        resume_size_checkpoint=checkpoint_dir,
        checkpoint_identity=identity,
        **common,
    )
    assert provenance_run.size_receipt["checkpoint"]["resumed_from"]["provenance"] == {}

    # Identity, pool and surface drift refuse by name.
    with pytest.raises(ValueError, match="epochs: checkpoint 2 != run 3"):
        solve_uk_rowwise_weights_under_doctrine(
            frame,
            problem,
            resume_size_checkpoint=checkpoint_dir,
            checkpoint_identity={**identity, "epochs": 3},
            **common,
        )
    with pytest.raises(ValueError, match="absent in checkpoint"):
        solve_uk_rowwise_weights_under_doctrine(
            frame,
            problem,
            resume_size_checkpoint=checkpoint_dir,
            checkpoint_identity={**identity, "ladder_sha256": "x"},
            **common,
        )
    from microcosm.build.uk_runtime.local_rowwise import _rowwise_target_set

    other_pool = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "person_household_id": [101, 102, 104],
                "person_benunit_id": [11, 12, 13],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [11, 12, 13]}),
        household=pd.DataFrame(
            {"household_id": [101, 102, 104], "household_weight": [1.0, 1.0, 1.0]}
        ),
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
    )
    with pytest.raises(ValueError, match="pool differs"):
        load_uk_size_checkpoint(
            checkpoint_dir,
            frame=other_pool,
            target_set=_rowwise_target_set(problem),
            identity=identity,
        )
    drifted = build_uk_rowwise_local_matrix(
        pd.DataFrame({"households": [1.0, 1.0, 1.0]}, index=[101, 102, 103]),
        _assigned(),
        pd.DataFrame({"code": ["E001", "S001"], "households": [2.5, 1.0]}),
    )
    with pytest.raises(ValueError, match="target surface differs"):
        solve_uk_rowwise_weights_under_doctrine(
            frame,
            drifted,
            resume_size_checkpoint=checkpoint_dir,
            checkpoint_identity=identity,
            **common,
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        solve_uk_rowwise_weights_under_doctrine(
            frame,
            problem,
            size_checkpoint_dir=checkpoint_dir,
            checkpoint_identity=identity,
            **common,
        )
    with pytest.raises(ValueError, match="dataset_households"):
        solve_uk_rowwise_weights_under_doctrine(
            frame,
            problem,
            bound_families=["census_households/constituency"],
            epochs=2,
            seed=7,
            size_checkpoint_dir=tmp_path / "no-size",
        )


def test_progress_lines_cover_the_dense_solve_the_probes_and_the_refit():
    from microcosm.build.uk_runtime.solve_progress import uk_solve_progress_callback

    frame = _clone_frame()
    problem = _size_problem()
    lines: list[str] = []
    solve_uk_rowwise_weights_under_doctrine(
        frame,
        problem,
        bound_families=["census_households/constituency"],
        dataset_households=2,
        epochs=2,
        seed=7,
        selection_pi_hi=0.5,
        progress=lines.append,
    )
    text = "\n".join(lines)
    # The dense solve, each probe's epochs, the probe verdicts, the stop line
    # and the refit all appear, timestamped, in that order.
    dense = next(i for i, line in enumerate(lines) if "dense solve: epoch 2/2" in line)
    probe_epoch = next(
        i for i, line in enumerate(lines) if "probe 1/10 (lambda" in line
    )
    probe_done = next(i for i, line in enumerate(lines) if "probe 1/10 done:" in line)
    stopped = next(i for i, line in enumerate(lines) if "search stopped:" in line)
    refit = next(i for i, line in enumerate(lines) if "refit: epoch 2/2" in line)
    assert dense < probe_epoch < probe_done < stopped < refit
    assert "open_probability_mass" in lines[probe_done]
    assert "verdict feasible" in text
    assert "drawable True" in lines[stopped]
    assert all(line[8] == "Z" for line in lines), lines[:2]

    # The formatter itself: a loss line only every ``every`` epochs and at the
    # last epoch; probe and stop events in one line each.
    sink: list[str] = []
    callback = uk_solve_progress_callback(sink.append, every=3)
    for epoch in range(1, 8):
        callback(
            {"kind": "calibration_epoch", "epoch": epoch, "epochs": 7, "loss": 0.5}
        )
    assert [line.split("epoch ")[1].split(" ")[0] for line in sink] == [
        "3/7",
        "6/7",
        "7/7",
    ]
    callback(
        {
            "kind": "budget_probe",
            "budget_iteration": 2,
            "budget_iters": 10,
            "l0_lambda": 1e-6,
            "measure": 54834,
            "target_records": 55000,
            "budget_basis": "open_probability_mass",
            "verdict": "boundary_mass_short",
            "certainty_count": 54563,
            "boundary_draw": 437,
            "boundary_mass": 290.6,
            "boundary_max": 0.947,
        }
    )
    assert sink[-1].endswith(
        "probe 2/10 done: lambda 1e-06, open_probability_mass 54834 for 55000 "
        "requested, verdict boundary_mass_short, certainties 54563, boundary draw "
        "437 from mass 290.6 (max 0.947)"
    )
    callback({"kind": "something_else"})
    assert len(sink) == 4
    with pytest.raises(ValueError, match="every"):
        uk_solve_progress_callback(sink.append, every=0)
