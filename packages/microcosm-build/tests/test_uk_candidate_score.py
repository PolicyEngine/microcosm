"""The UK candidate score on the common resolvable surface (microcosm#967).

The pruning loop is exercised against a stand-in for the scoring frame that
fails the incumbent's resolution the way the real resolver does — a
``MeasureResolutionError`` naming the absent measure, with the resolution
receipt's skip rows — while both frames score on raw columns, so the suite
needs no engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.target_materialization import MeasureResolutionError
from microcosm.build.uk_runtime import candidate_score
from microcosm.build.uk_runtime.candidate_score import (
    _sha256_file,
    evaluate_uk_candidate_against_incumbent,
    evaluation_block,
    prune_incumbent_unresolvable,
    pruned_warning,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    write_uk_national_frame,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import WeightKind

MEASURE_B = "household.measure_b"


def _registry() -> TargetRegistry:
    return TargetRegistry(
        [
            TargetSpec(
                name="target_a",
                entity="household",
                measure="measure_a",
                value=10.0,
                source="synthetic",
                family="family_a",
                metadata={"contract_target_id": "test.target_a"},
            ),
            TargetSpec(
                name="target_b",
                entity="household",
                measure="measure_b",
                value=20.0,
                source="synthetic",
                family="family_b",
                metadata={"contract_target_id": "test.target_b"},
            ),
        ],
        country="uk",
    )


def _write(path: Path, *, measure_a, measure_b=None) -> None:
    n = 2
    person = pd.DataFrame(
        {
            "person_id": np.arange(n),
            "person_household_id": np.arange(n),
            "person_benunit_id": np.arange(n),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(n)})
    household = {"household_id": np.arange(n), "measure_a": measure_a}
    if measure_b is not None:
        household["measure_b"] = measure_b
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=pd.DataFrame(household),
        time_period="2025",
        weight_kind=WeightKind.CALIBRATED,
        household_weights=np.ones(n),
    )
    write_uk_national_frame(frame, path)


def _twins(tmp_path: Path) -> tuple[Path, Path]:
    candidate = tmp_path / "candidate.h5"
    incumbent = tmp_path / "incumbent.h5"
    # The candidate hits target_a exactly and carries measure_b; the
    # incumbent misses target_a and never carried measure_b.
    _write(candidate, measure_a=[5.0, 5.0], measure_b=[10.0, 10.0])
    _write(incumbent, measure_a=[4.0, 4.0])
    return candidate, incumbent


def _resolver_factory(path, frame):
    # A resolver is present (so the incumbent is probed) but scoring stays
    # on raw columns: the real resolver is stood in for by ``_intercept``.
    return None


def _intercept(monkeypatch, *, failing_h5: Path, message: str, receipt: dict):
    """Fail the scoring frame for ``failing_h5`` while ``target_b`` is on the
    surface, the way the resolver fails on a measure the artifact lacks."""

    original = candidate_score._scored_frame
    calls: list[dict] = []

    def fake(h5_path, registry, calibration_year, factory, *, band_edge_registry=None):
        calls.append(
            {
                "h5": Path(h5_path),
                "targets": [spec.name for spec in registry.specs],
                "band_edge_registry": band_edge_registry,
            }
        )
        if Path(h5_path) == failing_h5 and any(
            spec.name == "target_b" for spec in registry.specs
        ):
            raise MeasureResolutionError(message, receipt=receipt)
        return original(h5_path, registry, calibration_year, None)

    monkeypatch.setattr(candidate_score, "_scored_frame", fake)
    return calls


_SKIP_RECEIPT = {
    "skips": [
        {"name": "target_b", "measure": "measure_b", "reason": "'household.measure_b'"}
    ]
}


def test_prunes_the_target_the_incumbent_cannot_materialize_and_scores_the_rest(
    monkeypatch, tmp_path
):
    pytest.importorskip("tables")
    candidate, incumbent = _twins(tmp_path)
    full = _registry()
    calls = _intercept(
        monkeypatch,
        failing_h5=incumbent,
        message=f"provider does not know {MEASURE_B}",
        receipt=_SKIP_RECEIPT,
    )

    score = evaluate_uk_candidate_against_incumbent(
        candidate_h5=candidate,
        incumbent_h5=incumbent,
        candidate_sha256=_sha256_file(candidate),
        incumbent_sha256=_sha256_file(incumbent),
        target_registry=full,
        calibration_year=2025,
        measure_resolver_factory=_resolver_factory,
    )

    pruned = score["incumbent_unresolvable_pruned"]
    assert pruned["n_pruned"] == 1
    assert pruned["n_scored"] == 1
    assert pruned["n_surface"] == 2
    assert pruned["measures"] == [MEASURE_B]
    assert pruned["families"] == {"family_b": 1}
    assert pruned["pruned_targets"]["target_b"] == {
        "name": "target_b",
        "family": "family_b",
        "unresolvable_measure": MEASURE_B,
        "reason": f"provider does not know {MEASURE_B}",
    }
    # Both arms were scored on the common surface only.
    assert score["register"]["n_specs"] == 1
    assert [row["target"] for row in score["target_drift"]] == ["target_a@0"]
    # The candidate hit its target and the incumbent missed: rule 1 passes.
    assert score["evaluation"]["verdict"] == "passed"
    assert score["evaluation"]["rule_1"]["passed"] is True
    assert score["evaluation"]["scored_surface"] == {
        "n_scored": 1,
        "n_pruned": 1,
        "n_surface": 2,
    }
    # A pruned surface never redraws its own band edges (#803): the scoring
    # calls carry the full register as the band-edge register.
    scoring_calls = [call for call in calls if call["targets"] == ["target_a"]]
    assert scoring_calls and all(
        call["band_edge_registry"] is full for call in scoring_calls
    )
    warning = pruned_warning(score)
    assert warning is not None
    assert MEASURE_B in warning and "1 target(s)" in warning and "target_b" in warning
    assert json.dumps(score)  # the receipt is serialisable as written


def test_binding_fallback_names_the_target_when_the_receipt_does_not(
    monkeypatch, tmp_path
):
    pytest.importorskip("tables")
    candidate, incumbent = _twins(tmp_path)
    _intercept(
        monkeypatch,
        failing_h5=incumbent,
        message=f"provider failed computing {MEASURE_B}: no such column",
        receipt={"skips": []},
    )
    contract = {
        "test.target_b": {"bindings": {"policyengine": {"value_variable": "measure_b"}}}
    }

    surface, pruned = prune_incumbent_unresolvable(
        incumbent, _registry(), 2025, _resolver_factory, contract_targets=contract
    )

    assert [spec.name for spec in surface.specs] == ["target_a"]
    assert pruned["target_b"].unresolvable_measure == MEASURE_B


def test_refuses_to_prune_blindly(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    candidate, incumbent = _twins(tmp_path)
    _intercept(
        monkeypatch,
        failing_h5=incumbent,
        message=f"provider does not know {MEASURE_B}",
        receipt={"skips": []},
    )

    with pytest.raises(MeasureResolutionError, match="refusing to prune blindly"):
        prune_incumbent_unresolvable(
            incumbent, _registry(), 2025, _resolver_factory, contract_targets={}
        )


def test_a_failure_naming_no_measure_is_re_raised(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    candidate, incumbent = _twins(tmp_path)
    _intercept(
        monkeypatch,
        failing_h5=incumbent,
        message="measure materialization made no progress on round 0",
        receipt={"skips": []},
    )

    with pytest.raises(MeasureResolutionError, match="made no progress"):
        prune_incumbent_unresolvable(
            incumbent, _registry(), 2025, _resolver_factory, contract_targets={}
        )


def test_pruning_off_restores_the_refusal(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    candidate, incumbent = _twins(tmp_path)
    _intercept(
        monkeypatch,
        failing_h5=incumbent,
        message=f"provider does not know {MEASURE_B}",
        receipt=_SKIP_RECEIPT,
    )

    with pytest.raises(MeasureResolutionError, match="does not know"):
        evaluate_uk_candidate_against_incumbent(
            candidate_h5=candidate,
            incumbent_h5=incumbent,
            candidate_sha256=_sha256_file(candidate),
            incumbent_sha256=_sha256_file(incumbent),
            target_registry=_registry(),
            calibration_year=2025,
            measure_resolver_factory=_resolver_factory,
            prune_incumbent_unresolvable_measures=False,
        )


def test_a_candidate_side_gap_still_refuses(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    candidate = tmp_path / "candidate.h5"
    incumbent = tmp_path / "incumbent.h5"
    _write(candidate, measure_a=[5.0, 5.0])
    _write(incumbent, measure_a=[4.0, 4.0], measure_b=[10.0, 10.0])
    _intercept(
        monkeypatch,
        failing_h5=candidate,
        message=f"provider does not know {MEASURE_B}",
        receipt=_SKIP_RECEIPT,
    )

    # Only the incumbent is probed for pruning; a measure the candidate
    # cannot materialize is a defect and refuses as before.
    with pytest.raises(MeasureResolutionError, match="does not know"):
        evaluate_uk_candidate_against_incumbent(
            candidate_h5=candidate,
            incumbent_h5=incumbent,
            candidate_sha256=_sha256_file(candidate),
            incumbent_sha256=_sha256_file(incumbent),
            target_registry=_registry(),
            calibration_year=2025,
            measure_resolver_factory=_resolver_factory,
        )


def test_verdict_fails_when_the_incumbent_fits_better(tmp_path):
    pytest.importorskip("tables")
    candidate = tmp_path / "candidate.h5"
    incumbent = tmp_path / "incumbent.h5"
    _write(candidate, measure_a=[5.0, 5.0], measure_b=[5.0, 5.0])
    _write(incumbent, measure_a=[4.0, 4.0], measure_b=[10.0, 10.0])

    score = evaluate_uk_candidate_against_incumbent(
        candidate_h5=candidate,
        incumbent_h5=incumbent,
        candidate_sha256=_sha256_file(candidate),
        incumbent_sha256=_sha256_file(incumbent),
        target_registry=_registry(),
        calibration_year=2025,
    )

    assert score["incumbent_unresolvable_pruned"]["n_pruned"] == 0
    assert pruned_warning(score) is None
    assert score["evaluation"]["scored_surface"] == {
        "n_scored": 2,
        "n_pruned": 0,
        "n_surface": 2,
    }
    assert score["evaluation"]["rule_1"]["candidate_full_loss"] == pytest.approx(0.25)
    assert score["evaluation"]["rule_1"]["incumbent_full_loss"] == pytest.approx(0.1)
    assert score["evaluation"]["verdict"] == "failed"
    assert evaluation_block(score) == score["evaluation"]
