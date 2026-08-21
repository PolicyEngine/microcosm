from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.national_build import write_uk_national_frame
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import WeightKind
from tools.score_uk_national_candidate import (
    _load_registry,
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
        target_registry=_registry(),
    )

    assert score["candidate_full_loss"] == pytest.approx(0.25)
    assert score["incumbent_full_loss"] == pytest.approx(0.1)
    assert score["candidate_train_loss"] == score["candidate_full_loss"]
    assert score["candidate_holdout_loss"] == score["candidate_full_loss"]
    assert score["holdout_basis"] == "none_declared"
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
    registry_json.write_text(
        json.dumps(
            {
                "country": "uk",
                "specs": [
                    asdict for asdict in (_spec.__dict__ for _spec in _registry().specs)
                ],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--candidate-h5",
                str(candidate),
                "--incumbent-h5",
                str(incumbent),
                "--target-registry-json",
                str(registry_json),
                "--output-json",
                str(output_json),
            ]
        )
        == 0
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["score_vs_enhanced_frs"]["holdout_basis"] == "none_declared"
    assert _load_registry(registry_json).version == _registry().version
