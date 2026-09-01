"""Tests for the UK local-target census (microcosm#495 increment 1).

The census is the executable inventory of the UK local calibration surface:
which local-area metric families the runtime can compute today, which official
UK local statistics could supply target values for them, and which binding
fences constrain that work. The committed JSON is an anti-drift artifact — a
metric added to ``local_targets.py`` without a census classification must fail
closed, and a stale committed census must fail its currency assertion.
"""

from __future__ import annotations

import json

import pytest

from microcosm.build.uk_runtime import metric_names
from microcosm.build.uk_runtime.hmrc_income import HMRC_SPI_TARGET_RECORD_COUNT
from microcosm.build.uk_runtime.hmrc_replay import FULL_FRS_TI_BAND_FENCE_ID
from microcosm.build.uk_runtime.local_target_census import (
    CENSUS_KIND,
    CENSUS_SCHEMA_VERSION,
    METRIC_STATUS_BOUND_IN_CODE,
    SOURCE_STATUS_DOCUMENTED_UNPINNED,
    SOURCE_STATUS_PINNED_IN_LADDER,
    SOURCE_STATUS_PINNED_IN_LEDGER_FACTS,
    SOURCE_STATUS_SIGNED_DEFERRED,
    assert_uk_local_target_census_current,
    build_uk_local_target_census,
    committed_uk_local_target_census_path,
    load_uk_local_target_census,
    write_uk_local_target_census,
)


def test_census_top_level_shape() -> None:
    census = build_uk_local_target_census()
    assert census["schema_version"] == CENSUS_SCHEMA_VERSION
    assert census["census_kind"] == CENSUS_KIND
    assert census["area_types"] == ["constituency", "la"]
    for key in ("metrics", "families", "sources", "binding_fences"):
        assert isinstance(census[key], list)
        assert census[key], f"census {key!r} must not be empty"


def test_census_covers_every_bound_metric_exactly_once() -> None:
    census = build_uk_local_target_census()
    rows = {row["name"]: row for row in census["metrics"]}
    assert len(rows) == len(census["metrics"]), "metric rows must be unique by name"

    for area_type in ("constituency", "la"):
        for name in metric_names(area_type):
            assert name in rows, f"in-code metric {name!r} missing from census"
            assert area_type in rows[name]["area_types"]

    in_code = set(metric_names("constituency")) | set(metric_names("la"))
    assert set(rows) == in_code, "census must not invent metrics absent from code"
    for row in census["metrics"]:
        assert row["status"] == METRIC_STATUS_BOUND_IN_CODE


def test_census_families_partition_metrics_and_reference_known_ids() -> None:
    census = build_uk_local_target_census()
    families = {row["family"]: row for row in census["families"]}
    source_ids = {row["source_id"] for row in census["sources"]}
    fence_ids = {row["fence_id"] for row in census["binding_fences"]}

    for metric in census["metrics"]:
        assert metric["family"] in families, (
            f"metric {metric['name']!r} references unknown family {metric['family']!r}"
        )
    for family in families.values():
        members = [
            metric
            for metric in census["metrics"]
            if metric["family"] == family["family"]
        ]
        assert members, f"family {family['family']!r} has no metrics"
        assert family["sources"], (
            f"family {family['family']!r} must document at least one official "
            "source (or be removed)"
        )
        for source_id in family["sources"]:
            assert source_id in source_ids
        for fence_id in family.get("adjudications", []):
            assert fence_id in fence_ids


def test_census_source_rows_are_reviewed_pointers() -> None:
    census = build_uk_local_target_census()
    statuses = {source["status"] for source in census["sources"]}
    assert SOURCE_STATUS_DOCUMENTED_UNPINNED not in statuses
    for source in census["sources"]:
        assert source["url"].startswith("https://"), source["source_id"]
        assert source["publisher"], source["source_id"]
        assert source["product"], source["source_id"]
        assert source["verified_on"], source["source_id"]
        assert source["status"] in {
            SOURCE_STATUS_DOCUMENTED_UNPINNED,
            SOURCE_STATUS_PINNED_IN_LEDGER_FACTS,
            SOURCE_STATUS_PINNED_IN_LADDER,
            SOURCE_STATUS_SIGNED_DEFERRED,
        }
        if source["status"] == SOURCE_STATUS_PINNED_IN_LEDGER_FACTS:
            pin = source["ledger_fact_pin"]
            assert pin["facts_sha256"] == (
                "4395a4e76a75332cc77a7dc1ea5d3c49b36e0d268c8449474bc129aa24e38c48"
            )
            assert pin["source_commit"] == "33ca98a"
        if source["status"] == SOURCE_STATUS_SIGNED_DEFERRED:
            assert source["signed_reason_id"], source["source_id"]
            assert source["signed_rationale"], source["source_id"]
        assert source["geographies"], source["source_id"]
        assert set(source["geographies"]) <= {"constituency", "la", "msoa"}


def test_banded_fence_carries_the_national_authority() -> None:
    census = build_uk_local_target_census()
    fences = {row["fence_id"]: row for row in census["binding_fences"]}
    assert FULL_FRS_TI_BAND_FENCE_ID in fences
    banded = fences[FULL_FRS_TI_BAND_FENCE_ID]
    assert banded["fenced_fact_count"] == HMRC_SPI_TARGET_RECORD_COUNT
    assert "band" in banded["rule"].lower()

    # The SPI-frame proxy fence must gate the HMRC family explicitly.
    hmrc_families = [
        row
        for row in census["families"]
        if any(
            m["family"] == row["family"] and m["name"].startswith("hmrc/")
            for m in census["metrics"]
        )
    ]
    assert hmrc_families
    for family in hmrc_families:
        assert "hmrc_spi_frame_model_proxy" in family["adjudications"]
        assert FULL_FRS_TI_BAND_FENCE_ID in family["adjudications"]


def test_fences_declare_enforcement_and_gate_reviewed_families() -> None:
    census = build_uk_local_target_census()
    for fence in census["binding_fences"]:
        assert fence["enforcement"] == "review_required_before_binding"
    assert "review_required_before_binding" in census["status_definitions"], (
        "fence enforcement status must be defined, not implied"
    )

    adjudications = {
        row["family"]: set(row.get("adjudications", [])) for row in census["families"]
    }
    assert "uc_unit_vs_household_grain" in adjudications["uc_households"]
    assert "ons_bhc_ahc_noncomparable" in adjudications["equivalised_income"]
    assert "population_universe_private_households" in adjudications["age_structure"]
    assert census["scope"], "census must declare its metric-surface scope"


def test_census_disclosure_fence_names_country_as_winning_grain() -> None:
    census = build_uk_local_target_census()
    fences = {row["fence_id"]: row for row in census["binding_fences"]}
    rule = fences["census_disclosure_control_noise"]["rule"]
    assert "standing cross-grain rule" in rule
    assert "uk_runtime.ledger_targets" in rule
    assert "country wins" in rule
    assert "national same-concept control" in rule


def test_unclassified_metric_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from microcosm.build.uk_runtime import local_target_census as module

    real_metric_names = module.metric_names

    def with_mystery(area_type: str, **kwargs):
        names = real_metric_names(area_type, **kwargs)
        return (*names, "mystery/unclassified_metric")

    monkeypatch.setattr(module, "metric_names", with_mystery)
    with pytest.raises(ValueError, match="mystery/unclassified_metric"):
        build_uk_local_target_census()


def test_near_miss_metric_names_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from microcosm.build.uk_runtime import local_target_census as module

    real_metric_names = module.metric_names
    for near_miss in ("uc_householdsX", "rent/private_rent_typo", "agex/0_10"):

        def with_near_miss(area_type: str, *, _name=near_miss, **kwargs):
            return (*real_metric_names(area_type, **kwargs), _name)

        monkeypatch.setattr(module, "metric_names", with_near_miss)
        with pytest.raises(ValueError, match="census family classification"):
            build_uk_local_target_census()


def test_area_metric_order_pins_order_and_duplicates_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    census = build_uk_local_target_census()
    for area_type in ("constituency", "la"):
        assert census["area_metric_order"][area_type] == list(metric_names(area_type))

    from microcosm.build.uk_runtime import local_target_census as module

    real_metric_names = module.metric_names

    def with_duplicate(area_type: str, **kwargs):
        names = real_metric_names(area_type, **kwargs)
        return (*names, names[0])

    monkeypatch.setattr(module, "metric_names", with_duplicate)
    with pytest.raises(ValueError, match="duplicate metric name"):
        build_uk_local_target_census()


def test_currency_assertion_catches_json_type_drift(tmp_path) -> None:
    path = tmp_path / "census.json"
    write_uk_local_target_census(path)
    tampered = load_uk_local_target_census(path)
    # Python equality would accept True == 1; canonical JSON must not.
    assert tampered["schema_version"] == 1
    tampered["schema_version"] = True
    path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="stale"):
        assert_uk_local_target_census_current(path)


def test_returned_census_is_isolated_from_module_registers() -> None:
    first = build_uk_local_target_census()
    first["families"][0]["sources"].append("poisoned_source")
    first["sources"][0]["geographies"].append("poisoned_geography")
    first["binding_fences"][0]["rule"] = "poisoned"
    second = build_uk_local_target_census()
    assert "poisoned_source" not in second["families"][0]["sources"]
    assert "poisoned_geography" not in second["sources"][0]["geographies"]
    assert second["binding_fences"][0]["rule"] != "poisoned"


def test_duplicate_registry_ids_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from microcosm.build.uk_runtime import local_target_census as module

    duplicated = (*module._SOURCES, dict(module._SOURCES[0]))
    monkeypatch.setattr(module, "_SOURCES", duplicated)
    with pytest.raises(ValueError, match="duplicate source id"):
        module.build_uk_local_target_census()


def test_census_round_trips_and_currency_assertion(tmp_path) -> None:
    path = tmp_path / "census.json"
    written = write_uk_local_target_census(path)
    assert written == path
    assert load_uk_local_target_census(path) == build_uk_local_target_census()
    assert_uk_local_target_census_current(path)

    tampered = load_uk_local_target_census(path)
    tampered["metrics"][0]["status"] = "improvised"
    path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="stale"):
        assert_uk_local_target_census_current(path)


def test_committed_census_is_current() -> None:
    committed = committed_uk_local_target_census_path()
    assert committed.is_file(), (
        "committed census artifact is missing; run "
        "`uv run python tools/census_uk_local_targets.py` to generate it"
    )
    assert_uk_local_target_census_current()


def test_census_driver_check_and_write(tmp_path) -> None:
    import importlib.util
    from pathlib import Path

    driver_path = (
        Path(__file__).resolve().parents[3] / "tools" / "census_uk_local_targets.py"
    )
    spec = importlib.util.spec_from_file_location("census_driver", driver_path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    out = tmp_path / "census.json"
    assert driver.main(["--out", str(out)]) == 0
    assert load_uk_local_target_census(out) == build_uk_local_target_census()
    assert driver.main(["--out", str(out), "--check"]) == 0

    tampered = load_uk_local_target_census(out)
    tampered["schema_version"] = 999
    out.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    assert driver.main(["--out", str(out), "--check"]) == 1

    # The no-out default resolves to the committed artifact; the check-only
    # route must pass against it without writing anything.
    assert driver.main(["--check"]) == 0
