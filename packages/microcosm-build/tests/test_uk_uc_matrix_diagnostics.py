"""Target comparisons preserve source ancestry and their declared controls."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


@pytest.fixture(scope="module")
def diagnostic_tool():
    root = Path(__file__).resolve().parents[3]
    tools = root / "tools"
    sys.path.insert(0, str(tools))
    try:
        spec = importlib.util.spec_from_file_location(
            "uc_matrix_diagnostics", tools / "diagnose_uk_uc_matrix.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(tools))


def test_source_support_groups_clones_before_concentration(diagnostic_tool):
    # Two clones are the same source. Their combined contribution is six,
    # equal to the independently observed third record's contribution.
    row = sparse.csr_matrix([[1.0, 1.0, 2.0, 0.0]])
    result = diagnostic_tool.source_support_metrics(
        row, np.array([2.0, 4.0, 3.0, 100.0]), np.array([8, 8, 9, 10])
    )
    assert result["household_support"] == 3
    assert result["original_source_household_support"] == 2
    assert result["source_contribution_ess"] == 2
    assert result["largest_source_contribution_share"] == 0.5
    assert result["largest_source_absolute_contribution"] == 6


def test_empty_source_support_is_explicit(diagnostic_tool):
    result = diagnostic_tool.source_support_metrics(
        sparse.csr_matrix((1, 3)), np.ones(3), np.arange(3)
    )
    assert result == {
        "household_support": 0,
        "original_source_household_support": 0,
        "source_contribution_ess": 0.0,
        "largest_source_contribution_share": 0.0,
        "largest_source_absolute_contribution": 0.0,
    }


def test_support_accepts_production_sparse_array(diagnostic_tool):
    matrix = sparse.csr_array([[1.0, 0.0], [0.0, 0.0]])
    rows = diagnostic_tool.matrix_support_metrics(matrix, np.ones(2), np.arange(2))
    assert rows[0]["household_support"] == 1
    assert rows[1]["household_support"] == 0


def test_joint_report_counts_benefit_units_and_sources_in_gb(
    diagnostic_tool, monkeypatch, tmp_path
):
    tables = {
        "person": pd.DataFrame(
            {"person_benunit_id": [1, 1, 2, 3], "person_household_id": [8, 8, 8, 9]}
        ),
        "benunit": pd.DataFrame(
            {"benunit_id": [1, 2, 3], "benunit_source_id": [11, 12, 13]}
        ),
        "household": pd.DataFrame(
            {"household_id": [8, 9], "household_source_id": [18, 19]}
        ),
    }
    monkeypatch.setattr(
        diagnostic_tool,
        "compute_uc_paid_diagnostic_masks",
        lambda *_: {"paid.total": np.ones(3, dtype=bool)},
    )
    resolver = SimpleNamespace(
        year=2025,
        simulation=SimpleNamespace(
            tax_benefit_system=SimpleNamespace(
                variables=dict.fromkeys(diagnostic_tool.BU_VARIABLES)
            )
        ),
        compute=lambda entity, variable: (
            np.array(["WALES", "NORTHERN_IRELAND"])
            if variable == "region"
            else np.ones(len(tables[entity])),
            "test",
        ),
    )
    diagnostic_tool.write_uc_support_diagnostics(
        SimpleNamespace(table=tables.__getitem__),
        resolver,
        {"prior": np.array([2.0, 5.0]), "calibrated": np.array([3.0, 10.0])},
        tmp_path,
    )
    row = json.loads((tmp_path / "uc_joint_diagnostics.json").read_text())["rows"][0]
    assert row["prior"] == 4
    assert row["calibrated"] == 6
    assert row["benefit_unit_support"] == 2
    assert row["household_support"] == 1
    assert row["original_source_benefit_unit_support"] == 2
    assert row["original_source_household_support"] == 1


@pytest.mark.requires_uk
def test_complete_joint_export_on_released_uk_engine(diagnostic_tool, tmp_path):
    from microcosm.build.uk_runtime.national_frame import uk_national_frame
    from microcosm.frame import WeightKind

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_benunit_id": [10, 10, 20, 20],
            "person_household_id": [100, 100, 200, 200],
            "age": [35, 8, 40, 38],
            "is_benunit_head": [True, False, True, False],
            "is_parent": [True, False, False, False],
            "is_uc_claimant": [True, False, True, True],
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": [10, 20],
            "benunit_source_id": [10, 20],
            "dependent_children": [1, 0],
            "is_married": [False, False],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [100, 200],
            "household_source_id": [100, 200],
            "region": ["WALES", "NORTHERN_IRELAND"],
            "council_tax": [1200.0, 0.0],
            "rent": [6000.0, 6000.0],
            "tenure_type": ["RENT_PRIVATELY", "RENT_PRIVATELY"],
        }
    )
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=2024,
        household_weights=np.ones(2),
        weight_kind=WeightKind.IMPORTANCE,
    )
    resolver = diagnostic_tool.UKMeasureResolver(
        simulation_source=None, frame=frame, year=2025, scratch_dir=tmp_path / "engine"
    )
    diagnostic_tool.write_uc_support_diagnostics(
        frame,
        resolver,
        {"prior": np.ones(2), "calibrated": np.array([2.0, 3.0])},
        tmp_path,
    )
    report = json.loads((tmp_path / "uc_joint_diagnostics.json").read_text())
    assert len(report["rows"]) == 39
    assert report["unavailable_optional_components"] == [
        "uc_net_earned_income",
        "uc_capital_income",
    ]
    assert "uc_tariff_income" in report["available_component_variables"]
    private = pd.read_pickle(tmp_path / "benefit_units_private.pkl")
    assert private["great_britain"].tolist() == [True, False]
    assert len(json.loads((tmp_path / "protected_outcomes.json").read_text())) == 6


def test_comparison_refuses_a_changed_population_or_prior(diagnostic_tool):
    baseline = {
        "household_ids": np.array([1, 2]),
        "household_source_ids": np.array([8, 9]),
        "target_loss_weights": np.array([1.0]),
        "initial_weights": np.array([3.0, 4.0]),
        "names": np.array(["a"]),
        "target_values": np.array([20.0]),
    }
    with pytest.raises(ValueError, match="household identities"):
        diagnostic_tool.validate_comparison_controls(
            baseline,
            household_source_ids=baseline["household_source_ids"],
            target_loss_weights=baseline["target_loss_weights"],
            household_ids=np.array([2, 1]),
            initial_weights=baseline["initial_weights"],
            names=["a"],
            target_values=np.array([21.0]),
        )
    with pytest.raises(ValueError, match="prior weights"):
        diagnostic_tool.validate_comparison_controls(
            baseline,
            household_source_ids=baseline["household_source_ids"],
            target_loss_weights=baseline["target_loss_weights"],
            household_ids=baseline["household_ids"],
            initial_weights=np.array([3.0, 5.0]),
            names=["a"],
            target_values=np.array([21.0]),
        )


def test_target_changes_are_enumerated_not_called_unchanged(diagnostic_tool):
    baseline = {
        "household_ids": np.array([1, 2]),
        "household_source_ids": np.array([8, 9]),
        "target_loss_weights": np.array([0.5, 0.5]),
        "initial_weights": np.array([3.0, 4.0]),
        "names": np.array(["a", "b"]),
        "target_values": np.array([20.0, 30.0]),
    }
    result = diagnostic_tool.validate_comparison_controls(
        baseline,
        household_source_ids=baseline["household_source_ids"],
        target_loss_weights=baseline["target_loss_weights"],
        household_ids=baseline["household_ids"],
        initial_weights=baseline["initial_weights"],
        names=["a", "b"],
        target_values=np.array([21.0, 30.0]),
    )
    assert result == {
        "same_household_order_and_priors": True,
        "target_values_and_roster_unchanged": False,
        "added_rows": [],
        "removed_rows": [],
        "changed_target_values": [{"name": "a", "old": 20.0, "new": 21.0}],
    }


@pytest.mark.parametrize("names", [["a", "c"], ["b", "a"], ["a"]])
def test_target_only_comparison_refuses_roster_or_order_drift(diagnostic_tool, names):
    baseline = {
        "household_ids": np.array([1, 2]),
        "household_source_ids": np.array([8, 9]),
        "target_loss_weights": np.array([0.5, 0.5]),
        "initial_weights": np.array([3.0, 4.0]),
        "names": np.array(["a", "b"]),
        "target_values": np.array([20.0, 30.0]),
    }
    with pytest.raises(ValueError, match="target names/order"):
        diagnostic_tool.validate_comparison_controls(
            baseline,
            household_source_ids=baseline["household_source_ids"],
            target_loss_weights=baseline["target_loss_weights"],
            household_ids=baseline["household_ids"],
            initial_weights=baseline["initial_weights"],
            names=names,
            target_values=np.ones(len(names)),
        )


def test_released_baseline_allows_only_reviewed_historical_removals(diagnostic_tool):
    old = {
        "names": np.array(["a@2025", "b@2025"]),
        "target_values": np.array([20.0, 30.0]),
    }
    assert diagnostic_tool.validate_inherited_contract(
        old, ["a@2025"], np.array([20.0]), allowed_removals={"b"}
    ) == {"removed_reviewed_rows": ["b@2025"], "surviving_values_unchanged": True}
    with pytest.raises(ValueError, match="unreviewed"):
        diagnostic_tool.validate_inherited_contract(
            old, ["a@2025"], np.array([20.0]), allowed_removals=set()
        )
    with pytest.raises(ValueError, match="values"):
        diagnostic_tool.validate_inherited_contract(
            old, ["a@2025"], np.array([21.0]), allowed_removals={"b"}
        )


@pytest.mark.parametrize(
    "field, changed, match",
    [
        ("household_source_ids", np.array([8, 10]), "source household"),
        ("target_loss_weights", np.array([0.2, 0.8]), "loss coefficients"),
    ],
)
def test_comparison_refuses_changed_ancestry_and_loss_weights(
    diagnostic_tool, field, changed, match
):
    baseline = {
        "household_ids": np.array([1, 2]),
        "initial_weights": np.array([3.0, 4.0]),
        "names": np.array(["a", "b"]),
        "target_values": np.array([20.0, 30.0]),
        "household_source_ids": np.array([8, 9]),
        "target_loss_weights": np.array([0.5, 0.5]),
    }
    arguments = {key: value for key, value in baseline.items()}
    arguments[field] = changed
    with pytest.raises(ValueError, match=match):
        diagnostic_tool.validate_comparison_controls(baseline, **arguments)


@pytest.mark.parametrize(
    "field",
    [
        "replay_receipt_sha256",
        "model_year",
        "source_year",
        "versions",
        "code_sha256",
        "solver_epochs",
    ],
)
def test_comparison_refuses_changed_or_unrecorded_runtime_controls(
    diagnostic_tool, field
):
    context = {key: "stable" for key in diagnostic_tool.COMPARISON_CONTEXT_FIELDS}
    diagnostic_tool.validate_comparison_context(context, dict(context))
    changed = {**context, field: "changed"}
    with pytest.raises(ValueError, match=field):
        diagnostic_tool.validate_comparison_context(context, changed)
    missing = dict(context)
    del missing[field]
    with pytest.raises(ValueError, match=field):
        diagnostic_tool.validate_comparison_context(missing, context)


def test_input_hash_receipt_refuses_midrun_edits(diagnostic_tool, tmp_path):
    declaration = tmp_path / "contract.json"
    declaration.write_text('{"value":1}')
    paths = {"contract": declaration}
    captured = diagnostic_tool.capture_file_hashes(paths)
    diagnostic_tool.verify_file_hashes(paths, captured)
    declaration.write_text('{"value":2}')
    with pytest.raises(ValueError, match="bytes changed"):
        diagnostic_tool.verify_file_hashes(paths, captured)


def test_pinned_source_feed_change_is_enumerated_without_relaxing_data_controls(
    diagnostic_tool,
):
    baseline = {field: "fixed" for field in diagnostic_tool.COMPARISON_CONTEXT_FIELDS}
    baseline.update(facts_sha256="old-facts", manifest_sha256="old-manifest")
    current = {
        **baseline,
        "facts_sha256": "new-facts",
        "manifest_sha256": "new-manifest",
    }
    assert diagnostic_tool.validate_comparison_context(baseline, current) == {
        "facts_sha256": {"baseline": "old-facts", "current": "new-facts"},
        "manifest_sha256": {"baseline": "old-manifest", "current": "new-manifest"},
    }


def test_comparison_requires_receipted_baseline_weighting_inputs(diagnostic_tool):
    diagnostic_tool.validate_baseline_inputs(
        {"inputs_private_sha256": "captured"}, "captured"
    )
    for receipt in ({}, {"inputs_private_sha256": "other"}):
        with pytest.raises(ValueError, match="completed receipt"):
            diagnostic_tool.validate_baseline_inputs(receipt, "captured")


def test_code_manifest_includes_the_numerical_and_relationship_seams(diagnostic_tool):
    paths = diagnostic_tool.diagnostic_code_paths()
    assert {
        "tool",
        "microcosm.calibrate.solve",
        "microcosm.calibrate.matrix",
        "microcosm.build.uk_runtime.uc_target_measurements",
        "microcosm.frame.adapters.policyengine_uk",
    } <= paths.keys()
    assert all(path.is_file() for path in paths.values())


def test_missing_source_ancestry_cannot_disappear_from_ess(diagnostic_tool):
    with pytest.raises(ValueError, match="complete source"):
        diagnostic_tool.source_support_metrics(
            sparse.csr_matrix([[1.0, 1.0]]), np.ones(2), np.array([8.0, np.nan])
        )


def test_saved_matrix_vectors_support_common_evaluation_and_source_removal(
    diagnostic_tool, tmp_path
):
    matrix = sparse.csr_array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
    weights = np.array([2.0, 4.0, 1.0])
    source_ids = np.array([8, 8, 9])
    sparse.save_npz(tmp_path / "matrix.npz", matrix)
    np.savez_compressed(
        tmp_path / "inputs.npz",
        household_ids=np.array([1, 2, 3]),
        household_source_ids=source_ids,
        initial_weights=weights,
        target_loss_weights=np.array([0.5, 0.5]),
        names=np.array(["a", "b"]),
        target_values=np.array([13.0, 4.0]),
    )
    saved = np.load(tmp_path / "inputs.npz", allow_pickle=False)
    loaded = sparse.load_npz(tmp_path / "matrix.npz")
    np.testing.assert_array_equal(
        loaded @ saved["initial_weights"], np.array([13.0, 4.0])
    )
    keep = saved["household_source_ids"] != 8
    np.testing.assert_array_equal(
        loaded[:, keep] @ saved["initial_weights"][keep], np.array([3.0, 0.0])
    )
