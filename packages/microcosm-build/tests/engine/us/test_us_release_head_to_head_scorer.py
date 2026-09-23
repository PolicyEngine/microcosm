"""US-engine cases for the replacement head-to-head scorer."""

# ruff: noqa: F403, F405

from pathlib import Path

import pytest

from microcosm.build.us_runtime.h5_io import write_nullable_us_h5
from microcosm.frame import US_SCHEMA, Frame
from test_support.microcosm_build.us_release_head_to_head_scorer import *


def test_incumbent_and_candidate_h5_loaders_preserve_scored_contract(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    module = _load_head_to_head_module()
    registry = _tiny_registry()
    incumbent_path = tmp_path / "incumbent.h5"
    candidate_path = tmp_path / "candidate.h5"
    broken_path = tmp_path / "candidate_missing_measure.h5"
    incumbent_frame = _tiny_frame(measure_values=(1.0, 2.0))
    candidate_frame = _tiny_frame(measure_values=(3.0, 4.0))
    broken_tables = {
        entity: candidate_frame.table(entity).copy()
        for entity in candidate_frame.entities
    }
    broken_tables["household"] = broken_tables["household"].drop(columns=["n_flagged"])
    broken_frame = Frame(
        broken_tables,
        US_SCHEMA,
        {"household": candidate_frame.weights_for("household")},
    )
    for path, frame in (
        (incumbent_path, incumbent_frame),
        (candidate_path, candidate_frame),
        (broken_path, broken_frame),
    ):
        write_nullable_us_h5(
            frame,
            path,
            period=2024,
            artifact_kind="replacement_scorecard_fixture",
        )

    incumbent = module.load_artifact(incumbent_path)
    candidate = module.load_artifact(candidate_path)
    broken = module.load_artifact(broken_path)
    incumbent_contract = module.scored_column_contract(
        incumbent.frame,
        registry.specs,
        artifact_name="incumbent",
    )
    candidate_contract = module.scored_column_contract(
        candidate.frame,
        registry.specs,
        artifact_name="candidate",
    )

    module._assert_identical_scored_contracts(
        {"incumbent": incumbent_contract, "candidate": candidate_contract}
    )
    assert incumbent.loader["kind"] == candidate.loader["kind"]
    with pytest.raises(ValueError, match="lacks scored column"):
        module.scored_column_contract(
            broken.frame,
            registry.specs,
            artifact_name="candidate",
        )


def test_historical_formula_owned_h5_scores_with_drop_receipt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    module = _load_head_to_head_module()
    artifact_path = tmp_path / "formula_owned_with_leaf.h5"
    write_nullable_us_h5(
        _tiny_frame_with_marketplace_columns(
            formula_owned=True,
            interview_leaf=True,
        ),
        artifact_path,
        period=2024,
        artifact_kind="replacement_scorecard_fixture",
    )

    loaded = module.load_artifact(artifact_path)
    person_columns = set(loaded.frame.table("person").columns)
    assert "has_marketplace_health_coverage" not in person_columns
    assert "has_marketplace_health_coverage_at_interview" in person_columns

    payload = _score_loaded_as_incumbent(module, monkeypatch, loaded)
    receipt = payload["artifacts"]["incumbent"]["normalization_receipts"][
        "historical_formula_owned_columns"
    ]
    assert receipt == {
        "count": 1,
        "columns_by_entity": {
            "person": ["has_marketplace_health_coverage"],
        },
    }
    markdown = module.render_markdown(payload)
    assert "Dropped column count: **1**" in markdown
    assert "`person`: `has_marketplace_health_coverage`" in markdown


def test_historical_formula_owned_h5_refuses_missing_leaf(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    module = _load_head_to_head_module()
    artifact_path = tmp_path / "formula_owned_without_leaf.h5"
    write_nullable_us_h5(
        _tiny_frame_with_marketplace_columns(
            formula_owned=True,
            interview_leaf=False,
        ),
        artifact_path,
        period=2024,
        artifact_kind="replacement_scorecard_fixture",
    )

    with pytest.raises(ValueError) as exc_info:
        module.load_artifact(artifact_path)

    message = str(exc_info.value)
    assert "has_marketplace_health_coverage" in message
    assert "has_marketplace_health_coverage_at_interview" in message
    assert "required input leaves are absent" in message


def test_clean_historical_h5_scores_with_empty_drop_receipt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    module = _load_head_to_head_module()
    artifact_path = tmp_path / "clean.h5"
    write_nullable_us_h5(
        _tiny_frame_with_marketplace_columns(
            formula_owned=False,
            interview_leaf=True,
        ),
        artifact_path,
        period=2024,
        artifact_kind="replacement_scorecard_fixture",
    )

    loaded = module.load_artifact(artifact_path)
    payload = _score_loaded_as_incumbent(module, monkeypatch, loaded)
    receipt = payload["artifacts"]["incumbent"]["normalization_receipts"][
        "historical_formula_owned_columns"
    ]
    assert receipt == {"count": 0, "columns_by_entity": {}}
    markdown = module.render_markdown(payload)
    assert "Dropped column count: **0**" in markdown
    assert "No formula-owned columns were present." in markdown
