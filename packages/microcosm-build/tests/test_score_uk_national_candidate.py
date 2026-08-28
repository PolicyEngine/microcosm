from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    write_uk_national_frame,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import WeightKind
from tools.score_uk_national_candidate import (
    _load_registry,
    _sha256_file,
    main,
    score_uk_national_candidate,
)


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
            ),
            TargetSpec(
                name="target_b",
                entity="household",
                measure="measure_b",
                value=20.0,
                source="synthetic",
                family="family_b",
            ),
        ],
        country="uk",
    )


def _write(path, *, measure_a, measure_b) -> None:
    n = 2
    person = pd.DataFrame(
        {
            "person_id": np.arange(n),
            "person_household_id": np.arange(n),
            "person_benunit_id": np.arange(n),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(n)})
    household = pd.DataFrame(
        {
            "household_id": np.arange(n),
            "measure_a": measure_a,
            "measure_b": measure_b,
        }
    )
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2025",
        weight_kind=WeightKind.CALIBRATED,
        household_weights=np.ones(n),
    )
    write_uk_national_frame(frame, path)


def test_score_uk_national_candidate_scores_synthetic_twins(tmp_path) -> None:
    pytest.importorskip("tables")
    candidate = tmp_path / "candidate.h5"
    incumbent = tmp_path / "incumbent.h5"
    _write(candidate, measure_a=[5.0, 5.0], measure_b=[5.0, 5.0])
    _write(incumbent, measure_a=[4.0, 4.0], measure_b=[10.0, 10.0])

    score = score_uk_national_candidate(
        candidate_h5=candidate,
        incumbent_h5=incumbent,
        candidate_sha256=_sha256_file(candidate),
        incumbent_sha256=_sha256_file(incumbent),
        target_registry=_registry(),
        calibration_year=2025,
    )

    assert score["artifacts"]["candidate"]["label"] == candidate.stem
    assert score["artifacts"]["incumbent"]["label"] == "enhanced_frs_2024_25"
    assert score["artifacts"]["candidate"]["sha256"] == _sha256_file(candidate)
    assert score["artifacts"]["incumbent"]["sha256"] == _sha256_file(incumbent)
    assert score["target_drift"] == [
        {
            "target": "target_a@0",
            "family": "family_a",
            "candidate_relative_error": pytest.approx(0.0),
            "incumbent_relative_error": pytest.approx(-0.2),
            "winner": "candidate",
        },
        {
            "target": "target_b@0",
            "family": "family_b",
            "candidate_relative_error": pytest.approx(-0.5),
            "incumbent_relative_error": pytest.approx(0.0),
            "winner": "incumbent",
        },
    ]
    assert score["candidate_full_loss"] == pytest.approx(0.25)
    assert score["incumbent_full_loss"] == pytest.approx(0.1)
    assert score["candidate_train_loss"] == score["candidate_full_loss"]
    # With no declared split, the holdout keys report absence rather than
    # repeating the fitted loss under a name that means generalization.
    assert score["candidate_holdout_loss"] is None
    assert score["incumbent_holdout_loss"] is None
    assert score["holdout_basis"] == "none_declared"
    assert score["loss"]["train_equals_full"] is True
    assert score["candidate_target_wins"] == 1
    assert score["incumbent_target_wins"] == 1
    assert score["target_wins_by_family"] == {
        "family_a": {
            "candidate_target_wins": 1,
            "incumbent_target_wins": 0,
            "ties": 0,
        },
        "family_b": {
            "candidate_target_wins": 0,
            "incumbent_target_wins": 1,
            "ties": 0,
        },
    }


def test_score_uk_national_candidate_cli_writes_score_block(tmp_path) -> None:
    pytest.importorskip("tables")
    candidate = tmp_path / "candidate.h5"
    incumbent = tmp_path / "incumbent.h5"
    registry_json = tmp_path / "registry.json"
    output_json = tmp_path / "score.json"
    _write(candidate, measure_a=[5.0, 5.0], measure_b=[5.0, 5.0])
    _write(incumbent, measure_a=[4.0, 4.0], measure_b=[10.0, 10.0])
    _registry().to_json(registry_json)

    assert (
        main(
            [
                "--candidate-h5",
                str(candidate),
                "--candidate-sha256",
                _sha256_file(candidate),
                "--incumbent-h5",
                str(incumbent),
                "--incumbent-sha256",
                _sha256_file(incumbent),
                "--registry-json",
                str(registry_json),
                "--output-json",
                str(output_json),
                "--calibration-year",
                "2025",
                "--candidate-label",
                "explicit_candidate",
                "--incumbent-label",
                "explicit_incumbent",
                "--no-measure-resolution",
            ]
        )
        == 0
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    score = payload["score_vs_enhanced_frs"]
    assert score["artifacts"]["candidate"]["label"] == "explicit_candidate"
    assert score["artifacts"]["incumbent"]["label"] == "explicit_incumbent"
    assert score["holdout_basis"] == "none_declared"
    # An undeclared holdout reports absence, never the fitted loss wearing a
    # holdout name: June's fixture holds a genuinely different holdout value,
    # so a copied one would read as generalization that was never measured.
    assert score["candidate_holdout_loss"] is None
    assert score["incumbent_holdout_loss"] is None
    assert isinstance(score["candidate_train_loss"], float)
    assert isinstance(score["incumbent_full_loss"], float)
    assert _load_registry(registry_json).version == _registry().version


def test_scorer_refuses_artifacts_that_do_not_match_their_pins(tmp_path) -> None:
    pytest.importorskip("tables")
    candidate = tmp_path / "candidate.h5"
    incumbent = tmp_path / "incumbent.h5"
    _write(candidate, measure_a=[5.0, 5.0], measure_b=[5.0, 5.0])
    _write(incumbent, measure_a=[4.0, 4.0], measure_b=[10.0, 10.0])

    with pytest.raises(ValueError, match="candidate artifact sha mismatch"):
        score_uk_national_candidate(
            candidate_h5=candidate,
            incumbent_h5=incumbent,
            candidate_sha256="0" * 64,
            incumbent_sha256=_sha256_file(incumbent),
            target_registry=_registry(),
            calibration_year=2025,
        )


def test_scorer_refuses_a_register_that_is_not_a_validated_artifact(tmp_path) -> None:
    # The loose loader accepted any object with a specs list, so a
    # hand-edited surface could decide rule 1 unnoticed.
    loose = tmp_path / "loose.json"
    loose.write_text(
        json.dumps(
            {
                "country": "uk",
                "specs": [spec.__dict__ for spec in _registry().specs],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a microcosm target registry artifact"):
        _load_registry(loose)

    tampered = tmp_path / "tampered.json"
    _registry().to_json(tampered)
    payload = json.loads(tampered.read_text(encoding="utf-8"))
    payload["specs"][0]["value"] = 999.0
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        _load_registry(tampered)


def test_scorer_refuses_prepared_measures_it_cannot_materialize(tmp_path) -> None:
    """A production-shaped register cannot be scored off raw exported columns.

    Every packaged UK reference binds a slash-named prepared measure, and
    calibration strips those before export — so scoring without measure
    resolution has to refuse rather than report on whatever happens to bind.
    """

    pytest.importorskip("tables")
    candidate = tmp_path / "candidate.h5"
    incumbent = tmp_path / "incumbent.h5"
    _write(candidate, measure_a=[5.0, 5.0], measure_b=[5.0, 5.0])
    _write(incumbent, measure_a=[4.0, 4.0], measure_b=[10.0, 10.0])
    production_shaped = TargetRegistry(
        [
            TargetSpec(
                name="dwp.uc.households",
                entity="benunit",
                measure="dwp/uc/households",
                value=20.0,
                source="test",
                family="dwp_universal_credit",
                metadata={"contract_target_id": "dwp.uc.households"},
            )
        ],
        country="uk",
    )

    with pytest.raises(RuntimeError, match="did not materialize for scoring"):
        score_uk_national_candidate(
            candidate_h5=candidate,
            incumbent_h5=incumbent,
            candidate_sha256=_sha256_file(candidate),
            incumbent_sha256=_sha256_file(incumbent),
            target_registry=production_shaped,
            calibration_year=2025,
        )
