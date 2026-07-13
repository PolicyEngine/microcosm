"""Contract checks for the aggregate real-donor HMRC replay artifacts."""

from __future__ import annotations

import json
from collections import Counter
from importlib.resources import files
from typing import Any

from populace.build.uk_runtime.hmrc_income import (
    HMRC_SPI_COLLATED_ODS_SHA256,
    HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
    HMRC_SPI_INCOME_COMPONENTS,
    HMRC_SPI_TARGET_RECORD_COUNT,
)
from populace.build.uk_runtime.hmrc_replay import FULL_FRS_TI_BAND_FENCE_ID
from populace.build.uk_runtime.release_input_coverage import (
    DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE,
)
from populace.build.uk_runtime.spi_income import (
    SPI_DONOR_SHA256,
    SPI_DONOR_SIZE_BYTES,
)

_PACKAGE = "populace.build.uk"
_REPLAY_RESOURCE = "hmrc_income_replay_report.json"
_GATE_RESOURCE = "hmrc_income_release_gate_report.json"
_DISTRIBUTIONAL_COLUMNS = frozenset(("gift_aid", "charitable_investment_gifts"))
_FORBIDDEN_ROW_LEVEL_KEYS = frozenset(
    {
        "person_id",
        "person_ids",
        "person_household_id",
        "person_benunit_id",
        "household_id",
        "household_ids",
        "benunit_id",
        "benunit_ids",
        "row_data",
        "sample_rows",
        "records",
        "draws",
        "predictions",
    }
)


def _resource(name: str) -> dict[str, Any]:
    payload = json.loads(files(_PACKAGE).joinpath(name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _walk_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(map(str, value)) | {
            key for child in value.values() for key in _walk_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _walk_keys(child)}
    return set()


def test_real_donor_replay_adjudicates_the_complete_208_fact_surface() -> None:
    replay = _resource(_REPLAY_RESOURCE)
    summary = replay["summary"]
    facts = replay["facts"]

    assert replay["report_kind"] == "uk_hmrc_income_208_fact_replay"
    assert summary == {
        "all_exclusions_fenced": True,
        "all_facts_adjudicated": True,
        "comparison_coverage_count": 0,
        "comparison_coverage_share": 0.0,
        "directional_fail": 0,
        "directional_pass": 0,
        "exact_fail": 0,
        "exact_pass": 0,
        "excluded_with_fence": HMRC_SPI_TARGET_RECORD_COUNT,
        "release_blocking_comparison_failures": 0,
        "status": "reviewed_exclusions_only",
        "total_facts": HMRC_SPI_TARGET_RECORD_COUNT,
    }
    assert len(facts) == HMRC_SPI_TARGET_RECORD_COUNT
    assert Counter(fact["component"] for fact in facts) == {
        component: len(HMRC_SPI_INCOME_BAND_LOWER_BOUNDS) * 2
        for component in HMRC_SPI_INCOME_COMPONENTS
    }
    assert Counter(fact["measure"] for fact in facts) == {
        "amount": HMRC_SPI_TARGET_RECORD_COUNT // 2,
        "count": HMRC_SPI_TARGET_RECORD_COUNT // 2,
    }
    assert Counter(fact["total_income_lower_bound"] for fact in facts) == {
        lower_bound: len(HMRC_SPI_INCOME_COMPONENTS) * 2
        for lower_bound in HMRC_SPI_INCOME_BAND_LOWER_BOUNDS
    }
    for fact in facts:
        assert fact["classification"] == "excluded"
        assert fact["outcome"] == "excluded_with_fence"
        assert fact["fence_ids"] == [FULL_FRS_TI_BAND_FENCE_ID]
        assert fact["blocked_dependencies"] == ["hmrc_spi_assessable_income"]
        assert fact["estimate"] is None
        assert fact["delta"] is None
        assert fact["relative_delta"] is None


def test_real_replay_binds_sources_identity_and_positive_mass() -> None:
    replay = _resource(_REPLAY_RESOURCE)
    sources = replay["source_evidence"]
    qrf = replay["qrf_evidence"]
    coverage = replay["effective_mass_evidence"]

    assert sources["spi_donor"] == {
        "release": "2022-23",
        "rows_used": 100_000,
        "sha256": SPI_DONOR_SHA256,
        "size_bytes": SPI_DONOR_SIZE_BYTES,
    }
    assert sources["hmrc_surface"]["sha256"] == HMRC_SPI_COLLATED_ODS_SHA256
    assert sources["hmrc_surface"]["mapped_build_period"] == "2023"
    assert qrf["fits"] == {
        "uk_frs_only_spi_fill": {"weight_kind": "importance"},
        "uk_spi_2022_23_income": {"weight_kind": "design"},
    }
    assert qrf["post_draw_identity"] == {
        "exact": True,
        "formula": "TEI + TII",
        "rows_checked": qrf["spi_prediction_rows"],
    }
    assert qrf["stage2_training_rows"] > 0
    assert qrf["spi_prediction_rows"] > 0
    assert coverage["minimum_nondefault_mass_share"] == (
        DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE
    )
    assert coverage["required_support_channel"] == "spi"
    assert set(coverage["columns"]) == _DISTRIBUTIONAL_COLUMNS
    assert all(
        share >= DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE
        for share in coverage["columns"].values()
    )


def test_frozen_143_plus_2_gate_refuses_stale_exclusions_only() -> None:
    gate = _resource(_GATE_RESOURCE)
    coverage = gate["input_coverage"]
    details = coverage["details"]

    assert gate["enforced"] is True
    assert coverage["passed"] is False
    assert coverage["failures"] == [
        "Stale reviewed exclusions — the column carries signal now, promote "
        "it to a hard requirement: ['charitable_investment_gifts', 'gift_aid']."
    ]
    assert details["required_columns"] == 143
    assert set(details["reviewed_exclusions"]) == _DISTRIBUTIONAL_COLUMNS
    assert set(details["stale_exclusions"]) == _DISTRIBUTIONAL_COLUMNS
    assert details["missing"] == []
    assert details["degenerate_required"] == []
    assert details["insufficient_effective_mass"] == []
    for column in _DISTRIBUTIONAL_COLUMNS:
        diagnostic = details["effective_mass_by_column"][column]
        assert diagnostic["positive_mass_signal_rows"] > 0
        assert diagnostic["effective_signal_mass_share"] >= (
            DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE
        )


def test_committed_replay_artifacts_contain_no_row_level_payloads_or_local_paths() -> (
    None
):
    for resource in (_REPLAY_RESOURCE, _GATE_RESOURCE):
        payload = _resource(resource)
        assert not (_walk_keys(payload) & _FORBIDDEN_ROW_LEVEL_KEYS)
        serialized = json.dumps(payload, allow_nan=False, sort_keys=True)
        assert "/Users/" not in serialized
        assert "put2223uk.tab" not in serialized
