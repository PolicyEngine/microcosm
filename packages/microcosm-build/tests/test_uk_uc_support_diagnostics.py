import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

_TOOL_PATH = Path(__file__).resolve().parents[3] / "tools" / "diagnose_uk_uc_support.py"
_SPEC = importlib.util.spec_from_file_location("diagnose_uk_uc_support", _TOOL_PATH)
support_tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(support_tool)
lp_probe = support_tool.lp_probe
row_support = support_tool.row_support
weighted_loss = support_tool.weighted_loss


def test_support_groups_clones_and_distinguishes_prior_capacity():
    row = sparse.csr_matrix([[1.0, 1.0, 2.0, 0.0]])
    r = row_support(
        row,
        np.array([2.0, 4.0, 3.0, 100.0]),
        np.array([1.0, 1.0, 2.0, 100.0]),
        np.array([8, 8, 9, 10]),
        30.0,
    )
    assert r["household_support"] == 3 and r["original_source_household_support"] == 2
    assert (
        r["source_contribution_ess"] == 2
        and r["largest_source_contribution_share"] == 0.5
    )
    assert r["cap_only_maximum"] == 60
    assert r["fitted_estimate"] == 12


def test_zero_support_is_not_turned_into_feasibility():
    r = row_support(
        sparse.csr_matrix((1, 2)), np.ones(2), np.ones(2), np.arange(2), 100.0
    )
    assert (
        r["zero_support_positive_target"]
        and not r["row_capacity_can_intersect_tolerance"]
    )
    probe = lp_probe(
        sparse.csr_matrix((1, 2)), np.array([100.0]), np.ones(2), ["a@2025"], ["a"]
    )
    assert (
        probe["status"] == "infeasible_by_zero_support" and not probe["linprog_called"]
    )


def test_lp_finds_named_subset_but_refuses_joint_inconsistency():
    a = sparse.csr_matrix([[1.0, 0.0], [1.0, 0.0]])
    b = np.array([5.0, 9.0])
    prior = np.ones(2)
    names = ["a@2025", "b@2025"]
    assert (
        lp_probe(a, b, prior, names, ["a"])["status"] == "feasible_for_named_rows_only"
    )
    assert (
        lp_probe(a, b, prior, names, ["a", "b"])["status"]
        == "infeasible_for_named_rows_under_cap"
    )


def test_lp_respects_weight_cap_and_missing_rows():
    a = sparse.csr_matrix([[1.0]])
    assert (
        lp_probe(a, np.array([20.0]), np.array([1.0]), ["a@2025"], ["a"])["status"]
        == "infeasible_for_named_rows_under_cap"
    )
    with pytest.raises(ValueError, match="missing rows"):
        lp_probe(a, np.array([20.0]), np.array([1.0]), ["a@2025"], ["missing"])


def test_lp_does_not_silently_select_another_period():
    with pytest.raises(ValueError, match="Ambiguous period"):
        lp_probe(
            sparse.csr_matrix([[1.0], [1.0]]),
            np.array([2.0, 3.0]),
            np.ones(1),
            ["a@2024", "a@2025"],
            ["a"],
        )


def test_supported_probe_does_not_silently_drop_an_unexpected_zero_row():
    supported_uc_probe = support_tool.supported_uc_probe
    rows = [{"name": "new_zero@2025", "zero_support_positive_target": True}]
    r = supported_uc_probe(
        sparse.csr_matrix([[0.0]]),
        np.array([10.0]),
        np.ones(1),
        ["new_zero@2025"],
        rows,
    )
    assert r["status"] == "not_run_unexpected_zero_support"
    assert r["unexpected_zero_rows"] == ["new_zero@2025"]


def test_supported_probe_explicitly_records_known_zero_exclusions():
    supported_uc_probe = support_tool.supported_uc_probe
    known_zero_rows = support_tool.KNOWN_ZERO_ROWS
    zero = sorted(known_zero_rows)[0] + "@2025"
    rows = [
        {"name": zero, "zero_support_positive_target": True},
        {"name": "supported@2025", "zero_support_positive_target": False},
    ]
    r = supported_uc_probe(
        sparse.csr_matrix([[0.0], [1.0]]),
        np.array([10.0, 2.0]),
        np.ones(1),
        [zero, "supported@2025"],
        rows,
    )
    assert r["status"] == "feasible_for_named_rows_only"
    assert r["excluded_zero_rows"] == [zero] and not r["full_uc_probe"]


def test_no_refit_loss_preserves_original_coefficients_and_row_scales():
    targets = np.array([100.0, 10.0, 0.0])
    estimates = np.array([80.0, 8.0, 30.0])
    coefficients = np.array([0.2, 0.3, 0.5])
    assert weighted_loss(estimates, targets, coefficients) == pytest.approx(5.1)
    # One original source, with multiple clone columns, is removed as a group.
    a = sparse.csr_matrix([[1.0, 2.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    weights = np.array([2.0, 4.0, 3.0])
    source = np.array([8, 8, 9])
    remove = source == 8
    delta = np.asarray(a[:, remove] @ weights[remove])
    np.testing.assert_array_equal(delta, np.array([10.0, 0.0, 0.0]))
    np.testing.assert_array_equal(a @ weights - delta, np.array([0.0, 3.0, 0.0]))


def test_signed_contrast_removal_uses_both_cap_bounds_and_component_tolerances():
    signed_contrast = support_tool.signed_contrast
    names = [
        "dwp.uc.households_children_5_or_more@2025",
        "dwp.uc.two_child_limit.households_5_children@2025",
        "dwp.uc.two_child_limit.households_6_plus_children@2025",
    ]
    a = sparse.csr_matrix(
        [[1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    report, private = signed_contrast(
        a,
        np.array([100.0, 30.0, 20.0]),
        names,
        np.array([2.0, 4.0, 3.0, 5.0]),
        np.array([1.0, 1.0, 2.0, 3.0]),
        np.array([8, 8, 9, 10]),
    )
    assert report["cap_only_minimum"] == -20 and report["cap_only_maximum"] == 40
    assert report["optimizer_implied_contrast_interval"] == [42.5, 57.5]
    assert [c["name"] for c in report["components"]] == names
    by_source = {r["source_household_id"]: r for r in private}
    assert by_source[8]["remaining_cap_only_minimum"] == -20
    assert by_source[8]["remaining_cap_only_maximum"] == 30
    assert by_source[9]["remaining_cap_only_minimum"] == 0
    assert by_source[9]["remaining_cap_only_maximum"] == 40
    assert by_source[9]["no_refit_contrast_contribution_loss"] == -3


def test_computational_contrast_can_have_one_source_despite_many_component_sources():
    signed_contrast = support_tool.signed_contrast
    names = [
        "dwp.uc.households_children_5_or_more@2025",
        "dwp.uc.two_child_limit.households_5_children@2025",
        "dwp.uc.two_child_limit.households_6_plus_children@2025",
    ]
    a = sparse.csr_matrix([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    r, private = signed_contrast(
        a,
        np.array([10.0, 3.0, 2.0]),
        names,
        np.array([2.0, 5.0, 3.0]),
        np.ones(3),
        np.array([8, 9, 10]),
    )
    assert r["household_support"] == 1 and r["original_source_household_support"] == 1
    assert (
        r["source_contribution_ess"] == 1
        and r["largest_source_contribution_share"] == 1
    )
    assert r["one_source_supplies_all_nonzero_contrast_columns"]
    assert r["zero_contrast_excluded_by_component_implied_interval"]
    assert (
        private[0]["remaining_cap_only_minimum"]
        == private[0]["remaining_cap_only_maximum"]
        == 0
    )
    assert not private[0]["remaining_capacity_can_intersect_component_implied_interval"]


def write_saved_run(tmp_path, **overrides):
    """Tiny synthetic matrix with each named diagnostic row and authenticated bytes."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bases = list(
        dict.fromkeys(
            [
                *support_tool.BROAD,
                *sorted(support_tool.FIVE_PLUS),
                *sorted(support_tool.KNOWN_ZERO_ROWS),
                "obr.income_tax",
                "obr.ni",
                "obr.state_pension",
                "obr.child_benefit",
                "hmrc/employment_income_count_income_band_0_to_5000",
            ]
        )
    )
    matrix = sparse.csr_matrix(np.ones((len(bases), 3)))
    matrix = matrix.tolil()
    for index, base in enumerate(bases):
        if base in support_tool.KNOWN_ZERO_ROWS:
            matrix[index] = 0
    sparse.save_npz(run_dir / "matrix.npz", matrix.tocsr())
    arrays = {
        "names": np.array([base + "@2025" for base in bases]),
        "target_values": np.full(len(bases), 3.0),
        "target_loss_weights": np.ones(len(bases)),
        "target_families": np.array(
            [
                "dwp_universal_credit" if base.startswith("dwp") else "other"
                for base in bases
            ]
        ),
        "household_ids": np.array([10, 11, 12]),
        "household_source_ids": np.array([8, 8, 9]),
        "initial_weights": np.ones(3),
    }
    fitted = overrides.pop("fitted_weights", np.ones(3))
    arrays.update(overrides)
    np.savez(run_dir / "inputs_private.npz", **arrays)
    np.savez(run_dir / "weights_private.npz", weights=fitted)
    receipt = {
        field: support_tool.digest(run_dir / name)
        for name, field in (
            ("matrix.npz", "matrix_sha256"),
            ("inputs_private.npz", "inputs_private_sha256"),
            ("weights_private.npz", "weights_private_sha256"),
        )
    }
    (run_dir / "receipt.json").write_text(json.dumps(receipt))
    return run_dir


@pytest.mark.parametrize(
    "source_ids",
    [
        np.array([8.0, np.nan, 9.0]),
        np.array([8.0, np.inf, 9.0]),
        np.array([8.0, 8.5, 9.0]),
        np.array(["8", "", "9"]),
        np.array([8, -1, 9]),
        np.array([True, False, True]),
    ],
)
def test_saved_run_refuses_missing_or_invalid_source_identity(tmp_path, source_ids):
    run_dir = write_saved_run(tmp_path, household_source_ids=source_ids)
    with pytest.raises(ValueError, match="household_source_ids"):
        support_tool.analyze(run_dir, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "override",
    [
        {"initial_weights": np.ones((3, 1))},
        {"household_source_ids": np.array([[8, 8, 9]])},
        {"household_ids": np.array([10, 11])},
        {"target_values": np.ones(2)},
        {"target_loss_weights": np.ones((1, 21))},
        {"target_families": np.array(["dwp_universal_credit"])},
        {"fitted_weights": np.ones((3, 1))},
    ],
)
def test_saved_run_refuses_unaligned_or_nonvector_arrays(tmp_path, override):
    run_dir = write_saved_run(tmp_path, **override)
    with pytest.raises(ValueError, match="one-dimensional|aligned"):
        support_tool.analyze(run_dir, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("failure", ["missing", "mismatch", "not_mapping"])
def test_saved_run_requires_matching_completed_receipt(tmp_path, failure):
    run_dir = write_saved_run(tmp_path)
    receipt = run_dir / "receipt.json"
    if failure == "missing":
        receipt.unlink()
    elif failure == "mismatch":
        receipt.write_text(json.dumps({"matrix_sha256": "0" * 64}))
    else:
        receipt.write_text("null")
    with pytest.raises(ValueError, match="receipt"):
        support_tool.analyze(run_dir, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("tolerance", [0.05, 0.1])
def test_saved_run_reports_named_scope_and_custom_tolerance(tmp_path, tolerance):
    run_dir = write_saved_run(tmp_path)
    output = tmp_path / "output"
    result = support_tool.analyze(run_dir, output, tolerance=tolerance)
    report = json.loads((output / "aggregate_support_report.json").read_text())
    assert result["zero_rows"] == 2
    assert report["controls"]["weight_cap_ratio"] == 10
    assert report["controls"]["original_source_household_count"] == 2
    assert (
        report["full_uc_necessary_conditions"]["can_satisfy_all_uc_rows_at_tolerance"]
        is False
    )
    probe = report["lp_probes"][
        f"supported_uc_rows_excluding_named_zero_cells_{100 * tolerance:g}pct"
    ]
    assert probe["status"] == "feasible_for_named_rows_only"
    assert not probe["full_uc_probe"] and not probe["fixed_mass_constraint"]
    assert len(probe["excluded_zero_rows"]) == 2
    assert all(p["tolerance"] == tolerance for p in report["lp_probes"].values())
    text = (output / "aggregate_support_report.md").read_text()
    assert f"{100 * tolerance:g}% tolerance" in text
    if tolerance != 0.05:
        assert "5% tolerance" not in text
    assert "source_household_id" not in json.dumps(report)


@pytest.mark.parametrize("tolerance", [0.0, 1.0, -0.1, np.nan, np.inf])
def test_analyze_refuses_invalid_tolerance_before_writing(tmp_path, tolerance):
    run_dir = write_saved_run(tmp_path)
    with pytest.raises(ValueError, match="Tolerance"):
        support_tool.analyze(run_dir, tmp_path / "output", tolerance=tolerance)
    assert not (tmp_path / "output").exists()


def test_saved_run_requires_source_identity_array(tmp_path):
    run_dir = write_saved_run(tmp_path)
    path = run_dir / "inputs_private.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: archive[name]
            for name in archive.files
            if name != "household_source_ids"
        }
    np.savez(path, **arrays)
    receipt_path = run_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["inputs_private_sha256"] = support_tool.digest(path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(
        ValueError, match="missing required arrays.*household_source_ids"
    ):
        support_tool.analyze(run_dir, tmp_path / "output")


def test_cli_has_no_unreceipted_bypass(tmp_path):
    with pytest.raises(SystemExit) as error:
        support_tool.main(
            [
                "--run-dir",
                str(tmp_path / "run"),
                "--output-dir",
                str(tmp_path / "out"),
                "--allow-unreceipted-development",
            ]
        )
    assert error.value.code == 2
