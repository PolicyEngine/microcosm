import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

_PATH = Path(__file__).resolve().parents[3] / "tools" / "diagnose_uk_uc_lp_evidence.py"
_SPEC = importlib.util.spec_from_file_location("uc_lp_evidence_tool", _PATH)
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def run_arrays(matrix, targets, *, prior=None, current=None, names=None, source=None):
    matrix = sparse.csr_matrix(matrix, dtype=float)
    rows, columns = matrix.shape
    return {
        "matrix": matrix,
        "targets": np.array(targets, dtype=float),
        "names": np.array(
            names
            if names is not None
            else [tool.LONE_PARENT, *[f"other{i}@2025" for i in range(rows - 1)]]
        ),
        "prior": np.array(
            prior if prior is not None else np.ones(columns), dtype=float
        ),
        "current": np.array(
            current if current is not None else np.ones(columns), dtype=float
        ),
        "source_ids": np.array(
            source if source is not None else np.arange(columns), dtype=int
        ),
        "household_ids": np.arange(columns),
        "families": np.full(rows, "other"),
        "coefficients": np.ones(rows),
    }


def profile_run():
    names = [tool.LONE_PARENT, tool.CGT, *tool.RETAINED_CGT, *tool.ZERO_ROWS]
    run = run_arrays(
        [[1, 0], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0]],
        [6, 5, 5, 5, 1, 1],
        prior=[10, 10],
        current=[5, 5],
        names=names,
    )
    run["families"][[0, 4, 5]] = "dwp_universal_credit"
    run["hashes"] = {
        name: "a" * 64
        for name in (
            "receipt.json",
            "matrix.npz",
            "inputs_private.npz",
            "weights_private.npz",
        )
    }
    return run


def test_mixed_tolerances_change_feasibility_without_changing_rows():
    run = run_arrays([[1, 0], [1, 0]], [5, 6])
    names = run["names"].tolist()
    strict = tool.mixed_probe(run, names, [0.05, 0.05])
    relaxed = tool.mixed_probe(run, names, [0.05, 0.25])
    assert strict["scipy_status"] == 2
    assert relaxed["scipy_status"] == 0
    assert abs(relaxed["lone_parent_relative_error"]) <= 0.05 + 1e-12


def test_fixed_mass_includes_columns_unused_by_target_rows():
    run = run_arrays([[1, 0]], [6], prior=[10, 10], current=[5, 5])
    result = tool.mixed_probe(
        run, run["names"].tolist(), [0.05], multipliers=(0.75, 1.25)
    )
    assert result["scipy_status"] == 0
    assert result["active_weight_columns"] == 2
    assert result["witness_concentration"]["households"][
        "weight_mass"
    ] == pytest.approx(10)
    assert result["relative_total_mass_error"] < 1e-10


def test_wider_redistribution_can_restore_feasibility():
    run = run_arrays([[1, 0]], [7.5], prior=[10, 10], current=[5, 5])
    names = run["names"].tolist()
    assert (
        tool.mixed_probe(run, names, [0.05], multipliers=(0.75, 1.25))["scipy_status"]
        == 2
    )
    assert (
        tool.mixed_probe(run, names, [0.05], multipliers=(0.5, 1.5))["scipy_status"]
        == 0
    )


def test_redistribution_still_intersects_prior_cap():
    run = run_arrays([[1, 0]], [7.5], prior=[0.5, 10], current=[5, 5])
    result = tool.mixed_probe(
        run, run["names"].tolist(), [0.05], multipliers=(0.5, 1.5)
    )
    assert result["scipy_status"] == 2


def test_zero_prior_stays_zero_and_does_not_break_mass():
    run = run_arrays([[1, 1]], [5], prior=[0, 1], current=[0, 5])
    tool.validate_arrays(run)
    result = tool.mixed_probe(
        run, run["names"].tolist(), [0.05], multipliers=(0.75, 1.25)
    )
    assert result["scipy_status"] == 0
    assert result["active_weight_columns"] == 1
    assert result["witness_concentration"]["households"]["positive_weight_count"] == 1


@pytest.mark.parametrize("values", [[0.05, 0.25], [float("nan")], [0], [1]])
def test_tolerances_must_match_rows_and_be_finite(values):
    run = run_arrays([[1]], [1])
    with pytest.raises(ValueError, match="tolerance"):
        tool.mixed_probe(run, run["names"].tolist(), values)


def test_postverification_rejects_false_success(monkeypatch):
    run = run_arrays([[1]], [5])
    monkeypatch.setattr(
        tool,
        "linprog",
        lambda *a, **k: SimpleNamespace(status=0, message="fake", x=np.array([100.0])),
    )
    with pytest.raises(ValueError, match="independent"):
        tool.mixed_probe(run, run["names"].tolist(), [0.05])


def test_postverification_rejects_mass_error(monkeypatch):
    run = run_arrays([[1, 0]], [6], prior=[10, 10], current=[5, 5])
    monkeypatch.setattr(
        tool,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            status=0, message="fake", x=np.array([0.6, 0.5])
        ),
    )
    with pytest.raises(ValueError, match="fixed-mass"):
        tool.mixed_probe(run, run["names"].tolist(), [0.05], multipliers=(0.75, 1.25))


def test_timeout_is_inconclusive(monkeypatch):
    run = run_arrays([[1]], [5])
    monkeypatch.setattr(
        tool,
        "linprog",
        lambda *a, **k: SimpleNamespace(status=1, message="Time limit reached"),
    )
    result = tool.mixed_probe(run, run["names"].tolist(), [0.05])
    assert result["status"] == "time_limit_inconclusive"


def test_bounded_profile_never_widens_after_timeout(monkeypatch):
    calls = []

    def probe(*a, **kw):
        calls.append(kw["multipliers"])
        return {"scipy_status": 1, "status": "time_limit_inconclusive"}

    monkeypatch.setattr(tool, "mixed_probe", probe)
    result = tool.bounded_redistribution(profile_run())
    assert calls == [(0.75, 1.25)]
    assert result["wider_redistribution_probe"]["status"] == "not_run"


def test_bounded_profile_widens_only_after_infeasibility(monkeypatch):
    calls = []

    def probe(*a, **kw):
        calls.append(kw["multipliers"])
        return {
            "scipy_status": 2,
            "status": "infeasible_for_named_bounds_and_tolerances",
        }

    monkeypatch.setattr(tool, "mixed_probe", probe)
    result = tool.bounded_redistribution(profile_run())
    assert calls == [(0.75, 1.25), (0.5, 1.5)]
    assert result["wider_redistribution_probe"]["status"] == "run"


def test_unexpected_zero_rows_are_not_silently_removed():
    run = profile_run()
    run["matrix"] = run["matrix"].tolil()
    run["matrix"][0] = 0
    run["matrix"] = run["matrix"].tocsr()
    run["matrix"].eliminate_zeros()
    with pytest.raises(ValueError, match="zero rows"):
        tool.full_supported(run)


def test_full_profile_keeps_hmrc_cgt_and_records_only_approved_exclusions(monkeypatch):
    monkeypatch.setattr(tool, "mixed_probe", lambda *a, **k: {"scipy_status": 2})
    result = tool.full_supported(profile_run())
    assert result["probes"][0]["excluded_adjudicated_row"] is None
    for probe in result["probes"]:
        assert set(tool.RETAINED_CGT) <= set(probe["row_names"])
        assert probe["excluded_zero_rows"] == tool.ZERO_ROWS
    assert result["probes"][2]["row_relative_tolerances"] == {
        tool.LONE_PARENT: 0.05,
        **{name: 0.25 for name in tool.RETAINED_CGT},
    }


@pytest.mark.parametrize(
    "field,values,error",
    [
        ("names", [tool.LONE_PARENT, tool.LONE_PARENT], "unique"),
        (
            "names",
            [tool.LONE_PARENT, "dwp.uc.households_single_with_children@2024"],
            "Ambiguous",
        ),
        ("source_ids", [0.0, float("nan")], "integers"),
        ("household_ids", [0, 0], "unique"),
        ("targets", [1.0, float("inf")], "finite"),
        ("current", [11.0, 1.0], "prior cap"),
        ("prior", [0.0, 1.0], "prior cap"),
        ("coefficients", [0.0, 0.0], "positive"),
    ],
)
def test_invalid_archived_arrays_refused(field, values, error):
    run = run_arrays([[1, 0], [0, 1]], [1, 1])
    run[field] = np.array(values)
    with pytest.raises(ValueError, match=error):
        tool.validate_arrays(run)


def test_exact_row_lookup_refuses_missing_or_duplicate_requests():
    run = run_arrays([[1]], [1])
    with pytest.raises(ValueError, match="missing"):
        tool.exact_indices(run, ["wrong@2025"])
    with pytest.raises(ValueError, match="unique"):
        tool.exact_indices(run, [tool.LONE_PARENT, tool.LONE_PARENT])


def test_concentration_groups_descendants_and_defines_top_one_percent():
    result = tool.concentration(np.array([2.0, 4.0, 4.0]), np.array([0, 0, 1]))
    assert result["households"]["effective_sample_size"] == pytest.approx(100 / 36)
    assert result["original_source_households"][
        "effective_sample_size"
    ] == pytest.approx(100 / 52)
    assert result["households"]["top_one_percent_entity_count"] == 1
    assert result["households"]["top_one_percent_weight_share"] == 0.4
    assert result["original_source_households"]["top_one_percent_weight_share"] == 0.6


def test_protected_changes_are_from_saved_rows_and_cgt_is_evaluated_not_constrained():
    run = profile_run()
    result = tool.mixed_probe(
        run,
        [tool.LONE_PARENT, *tool.RETAINED_CGT],
        [0.05, 0.25, 0.25],
        multipliers=(0.75, 1.25),
    )
    assert result["scipy_status"] == 0
    cgt = result["protected_matrix_outcome_changes"][0]
    assert cgt["name"] == tool.CGT and not cgt["constrained"]
    assert cgt["change_percent"] == pytest.approx(
        100 * (cgt["witness_estimate"] / cgt["current_estimate"] - 1)
    )
    serialized = json.dumps(result)
    assert "source_ids" not in serialized and '"weights"' not in serialized


def write_run(tmp_path):
    run = profile_run()
    directory = tmp_path / "run"
    directory.mkdir()
    sparse.save_npz(directory / "matrix.npz", run["matrix"])
    np.savez(
        directory / "inputs_private.npz",
        names=run["names"],
        target_values=run["targets"],
        initial_weights=run["prior"],
        household_source_ids=run["source_ids"],
        household_ids=run["household_ids"],
        target_families=run["families"],
        target_loss_weights=run["coefficients"],
    )
    np.savez(directory / "weights_private.npz", weights=run["current"])
    receipt = {
        "model_year": 2025,
        **{
            key: tool.support.digest(directory / name)
            for name, key in [
                ("matrix.npz", "matrix_sha256"),
                ("inputs_private.npz", "inputs_private_sha256"),
                ("weights_private.npz", "weights_private_sha256"),
            ]
        },
    }
    (directory / "receipt.json").write_text(json.dumps(receipt))
    return directory


def test_authenticated_input_boundary_and_hash_refusal(tmp_path):
    directory = write_run(tmp_path)
    run = tool.load_run(directory)
    assert run["matrix"].shape == (6, 2)
    with (directory / "weights_private.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="receipt"):
        tool.load_run(directory)


def test_producer_refuses_existing_output_directory_before_solving(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        tool,
        "load_run",
        lambda *a: pytest.fail("must refuse before loading or solving"),
    )
    with pytest.raises(ValueError, match="new"):
        tool.produce(tmp_path, tmp_path)


def test_exact_historical_check_ignores_only_runtime_and_historical_description():
    expected = {
        tool.PROFILE_KEYS["full-supported"]: {
            "purpose": "historical",
            "aggregate_receipt_sha256": "old",
            "probes": [{"elapsed_seconds": 10, "value": 0.1, "row_names": ["x@2025"]}],
        }
    }
    actual = {
        "full-supported": {
            "probes": [{"elapsed_seconds": 11, "value": 0.1, "row_names": ["x@2025"]}]
        }
    }
    assert (
        tool.verify_historical(actual, expected)["status"]
        == "exact_deterministic_parity"
    )
    actual["full-supported"]["probes"][0]["value"] = np.nextafter(0.1, 1.0)
    with pytest.raises(ValueError, match="Replay mismatch"):
        tool.verify_historical(actual, expected)


def test_producer_exports_only_aggregate_results_and_attests_both_sources(
    tmp_path, monkeypatch
):
    directory = write_run(tmp_path)
    monkeypatch.setattr(
        tool, "full_supported", lambda run: {"count": len(run["names"])}
    )
    output = tmp_path / "output"
    result = tool.produce(directory, output, "full-supported")
    report = json.loads((output / "lp-evidence-reproduction.json").read_text())
    assert result["producer_sha256"] == tool.support.digest(_PATH)
    assert result["support_helper_sha256"] == tool.support.digest(tool._HELPER_PATH)
    assert report["profiles"] == {"full-supported": {"count": 6}}
    assert str(directory) not in json.dumps(report)
    assert set(p.name for p in output.iterdir()) == {
        "lp-evidence-reproduction.json",
        "verification.json",
    }


def test_named_profile_preserves_explicit_cumulative_row_order():
    names = [
        *[name + "@2025" for name in tool.support.BROAD],
        *[name + "@2025" for name in tool.ADDITIONS],
        *tool.ZERO_ROWS,
    ]
    matrix = np.ones((len(names), 1))
    matrix[-2:] = 0
    run = run_arrays(matrix, np.ones(len(names)), names=names)
    run["families"] = np.array(
        [
            "dwp_universal_credit"
            if name
            in [*[name + "@2025" for name in tool.support.BROAD], *tool.ZERO_ROWS]
            else "other"
            for name in names
        ]
    )
    run["hashes"] = profile_run()["hashes"]
    result = tool.named_opposition(run)
    assert len(result["probes"]) == 6
    assert (
        result["full_active_uc_evaluation"]["zero_support_positive_rows"]
        == tool.ZERO_ROWS
    )
    for i, probe in enumerate(result["probes"][:5]):
        assert probe["row_names"] == names[:10] + [
            name + "@2025" for name in tool.ADDITIONS[: i + 1]
        ]
        assert probe["relative_tolerance"] == 0.05 and probe["scipy_status"] == 0
    assert result["probes"][-1]["row_names"] == names[:-2]


def test_input_mutation_during_probe_prevents_receipt(tmp_path, monkeypatch):
    directory = write_run(tmp_path)

    def changed(run):
        with (directory / "weights_private.npz").open("ab") as stream:
            stream.write(b"mutation")
        return {"count": 6}

    monkeypatch.setattr(tool, "full_supported", changed)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="changed during"):
        tool.produce(directory, output, "full-supported")
    assert not output.exists()


def test_protected_numeric_comparison_remains_exact_when_narrative_is_historical():
    expected = {
        "material_tradeoffs": {
            "interpretation": "historical prose",
            "selected_changes": [{"change_percent": -14.0}],
        }
    }
    tool.assert_equal(
        {"material_tradeoffs": {"selected_changes": [{"change_percent": -14.0}]}},
        expected,
    )
    with pytest.raises(ValueError, match="Replay mismatch"):
        tool.assert_equal(
            {"material_tradeoffs": {"selected_changes": [{"change_percent": -13.0}]}},
            expected,
        )
