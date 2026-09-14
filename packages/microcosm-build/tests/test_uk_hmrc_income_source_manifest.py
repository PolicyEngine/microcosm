"""Canonical raw-spine HMRC contracts bind current producers and official sources."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from microcosm.build.uk_runtime.hmrc_source_contract import (
    assert_uk_hmrc_income_source_contract_current,
    uk_hmrc_weighted_qrf_output_columns,
)

MANIFEST = (
    Path(__file__).resolve().parents[1] / "src/microcosm/build/uk/source_stages.json"
)
INCOME = "hmrc_spi_income_spine"


def _payload():
    return json.loads(MANIFEST.read_text())


def _stage(payload, name=INCOME):
    return next(stage for stage in payload["stages"] if stage["stage"] == name)


def _operation(payload, name, kind):
    return next(
        operation
        for operation in _stage(payload, name)["operations"]
        if operation["kind"] == kind
    )


def test_canonical_hmrc_family_matches_runtime_and_has_no_candidate_dependency():
    assert_uk_hmrc_income_source_contract_current()
    payload = _payload()
    names = {stage["stage"] for stage in payload["stages"]}
    assert {"frs_hmrc_spine_leaves", "spi_support_channel", INCOME} <= names
    assert not ({"frs_hmrc_retained_leaves", "hmrc_spi_income"} & names)
    for name in ("frs_hmrc_spine_leaves", "spi_support_channel", INCOME):
        stage = _stage(payload, name)
        assert "base_candidate" not in stage
        assert "verify_certified_candidate" not in {
            o["kind"] for o in stage["operations"]
        }
    assert not MANIFEST.with_name("hmrc_income_source_stages.json").exists()


def test_official_hmrc_sources_and_sampling_weights_remain_bound():
    stage = _stage(_payload())
    artifacts = {artifact["role"]: artifact for artifact in stage["artifacts"]}
    assert (
        artifacts["qrf_donor"]["sha256"]
        == "5ef829461060c91a2a47be59ad541d9b519fc3976d66ca80d4920f711bb96f66"
    )
    assert (
        artifacts["published_fact_surface"]["sha256"]
        == "ad063b06b2bdeef8600dbbb09d48153337a4966f8c7eea50df7a2e0304ebd73e"
    )
    assert artifacts["published_fact_surface"]["mapped_build_period"] == 2024
    assert artifacts["published_fact_surface"]["vintage"] == "2023-24"
    payload = _payload()
    first = _operation(payload, INCOME, "fit_weighted_qrf_stage1")
    second = _operation(payload, INCOME, "fit_weighted_qrf_stage2")
    assert first["source_sampling_weight"] == "FACT"
    assert first["post_sample_fit_weight"] == "uniform"
    assert first["double_apply_source_weight"] is False
    assert first["seed"] == 42 and second["seed"] == 43
    assert second["weight_mapping"] == "household_to_person"


@pytest.mark.parametrize(
    "stage_name,kind,field,replacement,match",
    [
        (
            "frs_hmrc_spine_leaves",
            "retain_adjudicated_frs_hmrc_leaves",
            "population",
            "candidate_h5",
            "frs_leaves.population",
        ),
        (
            "frs_hmrc_spine_leaves",
            "retain_adjudicated_frs_hmrc_leaves",
            "source_vintage",
            "2023-24",
            "source_vintage",
        ),
        (
            "spi_support_channel",
            "allocate_zero_weight_prior_mass",
            "share",
            0.1,
            "prior.mass_share",
        ),
        (
            "spi_support_channel",
            "allocate_zero_weight_prior_mass",
            "strata",
            [],
            "prior.strata",
        ),
        (
            "spi_support_channel",
            "stack_zero_weight_donors",
            "count",
            0,
            "count drifted",
        ),
        (INCOME, "strict_read_private_table", "weight", "uniform", "strict.weight"),
        (
            INCOME,
            "fit_weighted_qrf_stage1",
            "post_sample_fit_weight",
            "FACT",
            "post_sample_fit_weight",
        ),
        (INCOME, "fit_weighted_qrf_stage1", "source_columns", {}, "source_columns"),
        (INCOME, "fit_weighted_qrf_stage1", "seed", 43, "seed drifted"),
        (INCOME, "fit_weighted_qrf_stage2", "predictors", ["age"], "stage2.predictors"),
        (INCOME, "redraw_columns_from_fitted_qrf", "rows", "all", "base redraw"),
        (
            INCOME,
            "materialize_hmrc_income_bands_fail_closed",
            "component_columns",
            {},
            "component_columns",
        ),
        (
            INCOME,
            "classify_hmrc_income_facts_with_reviewed_fences",
            "required_fact_count",
            207,
            "required_fact_count",
        ),
        (
            INCOME,
            "gate_distributional_effective_mass",
            "required_support_channel",
            "frs",
            "required_support_channel",
        ),
        (
            INCOME,
            "gate_distributional_effective_mass",
            "minimum_nondefault_mass_share",
            0.01,
            "minimum_nondefault_mass_share",
        ),
    ],
)
def test_source_contract_rejects_drift(
    tmp_path, stage_name, kind, field, replacement, match
):
    payload = _payload()
    _operation(payload, stage_name, kind)[field] = replacement
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        assert_uk_hmrc_income_source_contract_current(path)


@pytest.mark.parametrize(
    "collection,key", [("artifacts", "role"), ("operations", "kind")]
)
def test_duplicate_source_declarations_are_refused(tmp_path, collection, key):
    payload = _payload()
    values = _stage(payload)[collection]
    values.append(dict(values[0]))
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=f"duplicate {key}"):
        assert_uk_hmrc_income_source_contract_current(path)


def test_missing_canonical_stage_is_refused(tmp_path):
    payload = _payload()
    payload["stages"] = [
        s for s in payload["stages"] if s["stage"] != "spi_support_channel"
    ]
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exactly one spi_support_channel"):
        assert_uk_hmrc_income_source_contract_current(path)


def test_tail_concentration_surface_contains_both_canonical_model_stages():
    payload = _payload()
    expected = tuple(
        dict.fromkeys(
            output
            for kind in ("fit_weighted_qrf_stage1", "fit_weighted_qrf_stage2")
            for output in _operation(payload, INCOME, kind)["outputs"]
        )
    )
    assert uk_hmrc_weighted_qrf_output_columns() == expected
    assert {"gift_aid", "self_employment_income"} <= set(expected)
