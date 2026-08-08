import importlib.util
import sys
from pathlib import Path

import pytest


def _load_compare_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "compare_us_fiscal_diagnostics.py"
    spec = importlib.util.spec_from_file_location("compare_us_fiscal_diagnostics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(
    name: str,
    *,
    target: float,
    final_estimate: float,
    family: str = "irs_soi",
    measure_mode: str = "sum",
    source_measure_id: str = "payment_amount",
) -> dict[str, object]:
    return {
        "name": f"{name}@2024",
        "target_name": name,
        "period": 2024,
        "entity": "household",
        "measure": {"kind": "column", "name": name},
        "filter": None,
        "source": "fixture",
        "metadata": {
            "measure_mode": measure_mode,
            "source_measure_id": source_measure_id,
        },
        "target": target,
        "compiled_target": target,
        "initial_estimate": final_estimate,
        "final_estimate": final_estimate,
        "relative_error": (final_estimate - target) / max(abs(target), 1.0),
        "within_tolerance": None,
        "registry": {"family": family, "se": None, "signed": False, "notes": ""},
    }


def test_compare_diagnostics_uses_shared_surface_and_weighted_loss() -> None:
    compare = _load_compare_module()
    candidate = {
        "options": {"target_loss_scales": {"cap": 1.0}},
        "targets": [
            _row("amount_small", target=100.0, final_estimate=110.0),
            _row("amount_large", target=400.0, final_estimate=500.0),
            _row(
                "count",
                target=25.0,
                final_estimate=20.0,
                measure_mode="indicator_sum",
                source_measure_id="return_count",
            ),
            _row("candidate_only", target=1.0, final_estimate=1.0),
        ],
    }
    incumbent = {
        "targets": [
            _row("amount_small", target=100.0, final_estimate=100.0),
            _row("amount_large", target=400.0, final_estimate=400.0),
            _row(
                "count",
                target=25.0,
                final_estimate=25.0,
                measure_mode="indicator_sum",
                source_measure_id="return_count",
            ),
            _row("incumbent_only", target=1.0, final_estimate=1.0),
        ],
    }

    payload = compare.compare_diagnostics(
        candidate_payload=candidate,
        incumbent_payload=incumbent,
    )

    assert payload["summary"]["shared_targets"] == 3
    assert payload["summary"]["candidate_only_targets"] == 1
    assert payload["summary"]["incumbent_only_targets"] == 1
    assert payload["summary"]["incumbent_weighted_loss"] == 0.0
    assert payload["summary"]["candidate_weighted_loss"] == pytest.approx(0.2)
    bases = {row["name"]: row for row in payload["bases"]}
    assert bases["amount"]["weight_share"] == pytest.approx(0.5)
    assert bases["count"]["weight_share"] == pytest.approx(0.5)
    assert payload["top_candidate_regressions"][0]["target"] == "count"


def test_compare_diagnostics_rejects_changed_shared_target_values() -> None:
    compare = _load_compare_module()
    candidate = {"targets": [_row("same_name", target=100.0, final_estimate=100.0)]}
    incumbent = {"targets": [_row("same_name", target=200.0, final_estimate=200.0)]}

    with pytest.raises(ValueError, match="changed shared target values"):
        compare.compare_diagnostics(
            candidate_payload=candidate,
            incumbent_payload=incumbent,
        )


def test_compare_diagnostics_rejects_duplicate_target_keys() -> None:
    compare = _load_compare_module()
    candidate = {
        "targets": [
            _row("duplicate", target=100.0, final_estimate=100.0),
            _row("duplicate", target=100.0, final_estimate=100.0),
        ],
    }
    incumbent = {"targets": [_row("duplicate", target=100.0, final_estimate=100.0)]}

    with pytest.raises(ValueError, match="Duplicate diagnostic target key"):
        compare.compare_diagnostics(
            candidate_payload=candidate,
            incumbent_payload=incumbent,
        )


def test_compare_diagnostics_filters_top_regressions_and_improvements_by_sign() -> None:
    compare = _load_compare_module()
    candidate = {
        "targets": [
            _row("improves", target=100.0, final_estimate=100.0),
            _row("regresses", target=100.0, final_estimate=120.0),
        ],
    }
    incumbent = {
        "targets": [
            _row("improves", target=100.0, final_estimate=130.0),
            _row("regresses", target=100.0, final_estimate=100.0),
        ],
    }

    payload = compare.compare_diagnostics(
        candidate_payload=candidate,
        incumbent_payload=incumbent,
    )

    assert [row["target"] for row in payload["top_candidate_regressions"]] == [
        "regresses"
    ]
    assert [row["target"] for row in payload["top_candidate_improvements"]] == [
        "improves"
    ]


def test_write_comparison_emits_json_csv_and_markdown(tmp_path) -> None:
    compare = _load_compare_module()
    payload = {
        "summary": {
            "shared_targets": 1,
            "candidate_only_targets": 0,
            "incumbent_only_targets": 0,
            "target_loss_cap": 1.0,
            "target_loss_weighting": "fixture",
            "candidate_weighted_loss": 0.2,
            "incumbent_weighted_loss": 0.1,
            "candidate_within_10pct": 0.0,
            "incumbent_within_10pct": 1.0,
            "weighted_loss_delta": 0.1,
        },
        "families": [
            {
                "name": "irs_soi",
                "target_count": 1,
                "weight_share": 1.0,
                "candidate_loss_contribution": 0.2,
                "incumbent_loss_contribution": 0.1,
                "loss_contribution_delta": 0.1,
                "candidate_mean_loss": 0.2,
                "incumbent_mean_loss": 0.1,
            }
        ],
        "bases": [],
        "top_candidate_regressions": [
            {
                "target": "row",
                "family": "irs_soi",
                "basis": "amount",
                "weight": 1.0,
                "weight_share": 1.0,
                "target_value": 100.0,
                "candidate_estimate": 120.0,
                "incumbent_estimate": 110.0,
                "candidate_relative_error": 0.2,
                "incumbent_relative_error": 0.1,
                "candidate_loss": 0.2,
                "incumbent_loss": 0.1,
                "loss_contribution_delta": 0.1,
            }
        ],
        "top_candidate_improvements": [],
        "candidate_only_target_examples": [],
        "incumbent_only_target_examples": [],
    }

    compare.write_comparison(payload, tmp_path)

    assert (tmp_path / "target_fit_comparison.json").exists()
    assert (tmp_path / "family_comparison.csv").exists()
    assert (tmp_path / "top_regressions.csv").exists()
    assert (
        "# Microcosm US Target Fit Comparison"
        in (tmp_path / "target_fit_comparison.md").read_text()
    )
